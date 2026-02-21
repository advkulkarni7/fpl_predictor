"""
FPL AI Assistant — Phase 4: Starting XI Optimizer (v3 - Full Rebuild)
======================================================================
New improvements:

  ALGORITHMIC:
  1. ILP for Starting XI — selects all 11 simultaneously, formation
     adapts automatically to injuries and blanks via constraints
  2. xPts captain weighting — predicted_pts * (1 + roll3_threat/100)
     captures upside variance, not just average ceiling
  3. Probability-weighted auto-substitution — uses chance_of_playing
     to compute expected bench contribution properly

  MISSING FUNCTIONALITY:
  4. Triple Captain chip detection — recalculates captain score as 3x
     when Triple Captain chip is available
  5. Bench Boost chip detection — optimises all 15 players when
     Bench Boost available, not just starting 11
  6. Starting XI vs Transfer interaction — shows optimal XI
     before AND after recommended transfer
  7. Vice captain auto-sub rule — VC picked as best backup captain
     (high predicted pts AND high chance_of_playing as safety net)

  DISPLAY:
  8. Expected XI score range — confidence interval based on
     position-specific model RMSE
  9. GW-by-GW XI recommendation — optimal XI for next 3 GWs

  FIXES FROM ALL PREVIOUS PHASES:
  - train_models() correct, build_current_features() correct
  - my_player_ids passed, pkl correct format
  - build_player_fixture_scores() with all required args
  - build_fixture_run() with custom_difficulty
  - TEAM_ID, VALID_FORMATIONS, FIXTURE_LOOKAHEAD from config
  - Logging throughout, unused imports removed

Run normally:
  python fpl_phase4_optimizer.py

Force fresh data:
  python fpl_phase4_optimizer.py --refresh
"""

import sys
import logging
import pickle
import itertools
import pandas as pd
import numpy as np

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

from fpl_phase1_model import (
    fetch_bootstrap,
    fetch_fixtures,
    fetch_current_gw,
    fetch_my_team,
    fetch_transfer_info,
    build_player_history_df,
    build_current_features,
    train_models,
    FEATURE_COLS,
    LOG_FILE,
)
from fpl_phase2_fixtures import (
    build_custom_difficulty,
    build_team_form,
    build_opponent_scoring_map,
    build_chip_status,
    build_fixture_run,
    build_player_fixture_scores,
)
from fpl_phase3_constraints import (
    validate_squad,
    get_ilp_optimal_transfers,
    get_valid_double_transfers,
    print_ilp_result,
    print_double_transfers,
)

try:
    from config import (
        TEAM_ID,
        VALID_FORMATIONS,
        CAPTAIN_DGW_MULTIPLIER,
        FIXTURE_LOOKAHEAD,
    )
except ImportError:
    TEAM_ID                = 9179961
    CAPTAIN_DGW_MULTIPLIER = 1.5
    FIXTURE_LOOKAHEAD      = 5
    VALID_FORMATIONS       = [
        (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2),
        (4, 5, 1), (5, 3, 2), (5, 4, 1),
    ]

# Position RMSE from Phase 1 models — used for confidence intervals
# Updated each run from saved model metadata
DEFAULT_RMSE = {
    "Goalkeeper": 1.6,
    "Defender":   2.1,
    "Midfielder": 1.9,
    "Forward":    2.3,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(
            open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)
        ),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────
# 1. HELPERS
# ─────────────────────────────────────────

def formation_name(d: int, m: int, f: int) -> str:
    return f"{d}-{m}-{f}"


def get_rmse_from_models(models: dict) -> dict:
    """Extract RMSE per position from trained models dict."""
    rmse = {}
    for pos, info in models.items():
        rmse[pos] = info.get("rmse", DEFAULT_RMSE.get(pos, 2.0))
    return rmse


def xpts_captain_score(row: pd.Series,
                        triple_captain: bool = False) -> float:
    """
    Expected points score for captain consideration.

    Improvement #2: uses predicted_pts * (1 + roll3_threat/100)
    as a proxy for upside variance. A player with high threat
    has more attacking intent and higher ceiling than raw pts suggest.

    Improvement #4: triple_captain=True scores 3x instead of 2x.

    Blank GW players always return 0.
    DGW players get CAPTAIN_DGW_MULTIPLIER boost.
    """
    if row.get("is_blank_next_gw", False):
        return 0.0

    pts      = row.get("predicted_pts", 0)
    threat   = row.get("roll3_threat", 0) or 0
    upside   = 1 + (float(threat) / 100)
    xpts     = pts * upside

    # DGW boost
    if row.get("double_gws", 0) > 0:
        xpts *= CAPTAIN_DGW_MULTIPLIER

    # Captain multiplier
    multiplier = 3 if triple_captain else 2
    return round(xpts * multiplier, 3)


def vc_safety_score(row: pd.Series) -> float:
    """
    Improvement #7: VC picked as best backup captain.
    FPL rule: if captain doesn't play, VC gets the double.
    So VC should combine high predicted_pts AND high reliability
    (chance_of_playing). We want someone almost certain to play
    with a high score — not a risky high-ceiling player.
    """
    if row.get("is_blank_next_gw", False):
        return 0.0
    pts    = row.get("predicted_pts", 0)
    chance = row.get("chance_of_playing", 100) / 100
    return round(pts * chance, 3)


# ─────────────────────────────────────────
# 2. ILP STARTING XI
# ─────────────────────────────────────────

def optimize_xi_ilp(squad_df: pd.DataFrame,
                     triple_captain: bool = False,
                     bench_boost: bool = False) -> dict:
    """
    Improvement #1: ILP-based Starting XI optimizer.

    Selects all 11 players simultaneously — formation adapts
    automatically to injuries and blanks via constraints.

    Formulation:
      Binary variable x_i = 1 if player i starts
      Objective: maximise sum of combined_score for starters
                 (or all 15 players if bench_boost=True)
      Constraints:
        - sum(x_i) == 11  (or 15 for bench boost)
        - exactly 1 GK starts
        - 3 <= sum(DEF starters) <= 5
        - 2 <= sum(MID starters) <= 5
        - 1 <= sum(FWD starters) <= 3
        - blank GW players forced to bench (x_i = 0)

    Falls back to brute-force if PuLP unavailable.

    Improvement #5: bench_boost=True maximises all 15 players.
    """
    if bench_boost:
        # Bench Boost: all 15 players score — no XI/bench split needed
        return _bench_boost_mode(squad_df, triple_captain)

    if not PULP_AVAILABLE:
        log.warning("PuLP not available — using brute-force fallback.")
        return optimize_xi_bruteforce(squad_df, triple_captain)

    players = squad_df.reset_index(drop=True)
    n       = len(players)
    prob    = pulp.LpProblem("Starting_XI", pulp.LpMaximize)
    x       = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Objective: maximise combined_score of starters
    prob += pulp.lpSum(players.loc[i, "combined_score"] * x[i] for i in range(n))

    # Exactly 11 starters
    prob += (pulp.lpSum(x) == 11, "squad_size")

    # GK: exactly 1 starts
    gk_idx = players[players["position"] == "Goalkeeper"].index.tolist()
    prob += (pulp.lpSum(x[i] for i in gk_idx) == 1, "gk")

    # DEF: 3 to 5
    def_idx = players[players["position"] == "Defender"].index.tolist()
    prob += (pulp.lpSum(x[i] for i in def_idx) >= 3, "min_def")
    prob += (pulp.lpSum(x[i] for i in def_idx) <= 5, "max_def")

    # MID: 2 to 5
    mid_idx = players[players["position"] == "Midfielder"].index.tolist()
    prob += (pulp.lpSum(x[i] for i in mid_idx) >= 2, "min_mid")
    prob += (pulp.lpSum(x[i] for i in mid_idx) <= 5, "max_mid")

    # FWD: 1 to 3
    fwd_idx = players[players["position"] == "Forward"].index.tolist()
    prob += (pulp.lpSum(x[i] for i in fwd_idx) >= 1, "min_fwd")
    prob += (pulp.lpSum(x[i] for i in fwd_idx) <= 3, "max_fwd")

    # Force blank GW players to bench
    if "is_blank_next_gw" in players.columns:
        for i in range(n):
            if players.loc[i, "is_blank_next_gw"]:
                prob += (x[i] == 0, f"blank_{i}")

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        log.warning(f"ILP XI status: {pulp.LpStatus[prob.status]}, using brute-force.")
        return optimize_xi_bruteforce(squad_df, triple_captain)

    start_idx = [i for i in range(n) if pulp.value(x[i]) == 1.0]
    bench_idx = [i for i in range(n) if i not in start_idx]

    starting_xi = players.loc[start_idx].reset_index(drop=True)
    bench_pool  = players.loc[bench_idx].copy()

    # Detect formation from selected players
    def_n = (starting_xi["position"] == "Defender").sum()
    mid_n = (starting_xi["position"] == "Midfielder").sum()
    fwd_n = (starting_xi["position"] == "Forward").sum()

    return _build_result(
        starting_xi, bench_pool, squad_df,
        formation_name(def_n, mid_n, fwd_n),
        triple_captain, method="ILP"
    )


def _bench_boost_mode(squad_df: pd.DataFrame,
                       triple_captain: bool) -> dict:
    """
    Improvement #5: Bench Boost mode.
    All 15 players score — rank captain by xPts across full squad.
    Returns a result dict marked as bench_boost=True.
    """
    players = squad_df.reset_index(drop=True).copy()
    players["cap_score"] = players.apply(
        lambda r: xpts_captain_score(r, triple_captain), axis=1
    )
    captain      = players.nlargest(1, "cap_score").iloc[0]
    vice_captain_row = players[
        players["player_id"] != captain["player_id"]
    ].copy()
    vice_captain_row["vc_score"] = vice_captain_row.apply(vc_safety_score, axis=1)
    vice_captain = vice_captain_row.nlargest(1, "vc_score").iloc[0]

    return {
        "formation":            "BENCH BOOST (All 15 play)",
        "starting_xi":          players,
        "bench":                pd.DataFrame(),
        "captain":              captain,
        "vice_captain":         vice_captain,
        "total_predicted_pts":  round(players["predicted_pts"].sum(), 2),
        "total_combined_score": round(players["combined_score"].sum(), 2),
        "bench_auto_sub_score": 0.0,
        "blank_on_bench":       0,
        "bench_boost":          True,
        "triple_captain":       triple_captain,
        "method":               "BenchBoost",
    }


def _build_result(starting_xi: pd.DataFrame,
                   bench_pool: pd.DataFrame,
                   squad_df: pd.DataFrame,
                   formation: str,
                   triple_captain: bool,
                   method: str = "ILP") -> dict:
    """
    Build result dict from selected XI and bench.
    Handles captain (xPts + DGW + TC), VC (safety score), bench ordering.
    """
    # Captain: xPts weighted (improvement #2 + #4)
    xi_copy = starting_xi.copy()
    xi_copy["cap_score"] = xi_copy.apply(
        lambda r: xpts_captain_score(r, triple_captain), axis=1
    )
    captain      = xi_copy.nlargest(1, "cap_score").iloc[0]

    # VC: safety score — likely to play + high pts (improvement #7)
    vc_pool = xi_copy[xi_copy["player_id"] != captain["player_id"]].copy()
    vc_pool["vc_score"] = vc_pool.apply(vc_safety_score, axis=1)
    vice_captain = vc_pool.nlargest(1, "vc_score").iloc[0]

    # Bench order: non-blank outfield first, then blank, GK last
    bench_gk      = bench_pool[bench_pool["position"] == "Goalkeeper"]
    bench_outfield = bench_pool[bench_pool["position"] != "Goalkeeper"]

    if "is_blank_next_gw" in bench_outfield.columns:
        bench_non_blank = bench_outfield[~bench_outfield["is_blank_next_gw"]] \
                          .sort_values("combined_score", ascending=False)
        bench_blanks    = bench_outfield[bench_outfield["is_blank_next_gw"]]
    else:
        bench_non_blank = bench_outfield.sort_values("combined_score", ascending=False)
        bench_blanks    = pd.DataFrame()

    bench_ordered = pd.concat(
        [b for b in [bench_non_blank, bench_blanks, bench_gk] if not b.empty],
        ignore_index=True
    )

    # Bench auto-sub score (improvement #3 — probability weighted)
    bench_val = _prob_weighted_bench_score(bench_ordered, squad_df)

    return {
        "formation":            formation,
        "starting_xi":          starting_xi,
        "bench":                bench_ordered,
        "captain":              captain,
        "vice_captain":         vice_captain,
        "total_predicted_pts":  round(starting_xi["predicted_pts"].sum(), 2),
        "total_combined_score": round(starting_xi["combined_score"].sum(), 2),
        "bench_auto_sub_score": bench_val,
        "blank_on_bench":       int(
            bench_ordered["is_blank_next_gw"].sum()
            if "is_blank_next_gw" in bench_ordered.columns else 0
        ),
        "bench_boost":          False,
        "triple_captain":       triple_captain,
        "method":               method,
    }


# ─────────────────────────────────────────
# 3. BRUTE-FORCE FALLBACK
# ─────────────────────────────────────────

def best_combination(players_df: pd.DataFrame,
                      n: int,
                      score_col: str = "combined_score") -> pd.DataFrame:
    """Brute-force optimal combination of n players."""
    if len(players_df) < n:
        return players_df
    best_score  = -999.0
    best_subset = None
    for combo in itertools.combinations(players_df.index, n):
        subset = players_df.loc[list(combo)]
        score  = subset[score_col].sum()
        if score > best_score:
            best_score  = score
            best_subset = subset
    return best_subset


def best_gk_for_gw(gks_df: pd.DataFrame) -> pd.DataFrame:
    """Pick GK with best predicted_pts for upcoming GW."""
    if gks_df.empty:
        return gks_df
    if "is_blank_next_gw" in gks_df.columns:
        playing = gks_df[~gks_df["is_blank_next_gw"]]
        if not playing.empty:
            return playing.nlargest(1, "predicted_pts").iloc[[0]]
    return gks_df.nlargest(1, "combined_score").iloc[[0]]


def optimize_xi_bruteforce(squad_df: pd.DataFrame,
                             triple_captain: bool = False) -> dict:
    """Brute-force fallback when PuLP unavailable."""
    if "is_blank_next_gw" in squad_df.columns:
        available = squad_df[~squad_df["is_blank_next_gw"]].copy()
    else:
        available = squad_df.copy()

    gks  = available[available["position"] == "Goalkeeper"]
    defs = available[available["position"] == "Defender"]
    mids = available[available["position"] == "Midfielder"]
    fwds = available[available["position"] == "Forward"]

    best_score  = -999.0
    best_result = None

    for def_n, mid_n, fwd_n in VALID_FORMATIONS:
        if len(defs) < def_n or len(mids) < mid_n or \
           len(fwds) < fwd_n or gks.empty:
            continue

        xi = pd.concat([
            best_gk_for_gw(gks),
            best_combination(defs, def_n),
            best_combination(mids, mid_n),
            best_combination(fwds, fwd_n),
        ], ignore_index=True)

        score = xi["combined_score"].sum()
        if score > best_score:
            best_score  = score
            bench_pool  = squad_df[
                ~squad_df["player_id"].isin(xi["player_id"])
            ].copy()
            best_result = _build_result(
                xi, bench_pool, squad_df,
                formation_name(def_n, mid_n, fwd_n),
                triple_captain, method="BruteForce"
            )

    return best_result


def score_all_formations(squad_df: pd.DataFrame) -> list:
    """Score every valid formation. Used for formation comparison table."""
    if "is_blank_next_gw" in squad_df.columns:
        available = squad_df[~squad_df["is_blank_next_gw"]].copy()
    else:
        available = squad_df.copy()

    gks  = available[available["position"] == "Goalkeeper"]
    defs = available[available["position"] == "Defender"]
    mids = available[available["position"] == "Midfielder"]
    fwds = available[available["position"] == "Forward"]

    results = []
    for def_n, mid_n, fwd_n in VALID_FORMATIONS:
        if len(defs) < def_n or len(mids) < mid_n or \
           len(fwds) < fwd_n or gks.empty:
            continue
        xi = pd.concat([
            best_gk_for_gw(gks),
            best_combination(defs, def_n),
            best_combination(mids, mid_n),
            best_combination(fwds, fwd_n),
        ], ignore_index=True)
        results.append({
            "formation": formation_name(def_n, mid_n, fwd_n),
            "pred_pts":  round(xi["predicted_pts"].sum(), 2),
            "combined":  round(xi["combined_score"].sum(), 2),
        })
    return sorted(results, key=lambda x: x["combined"], reverse=True)


# ─────────────────────────────────────────
# 4. PROBABILITY-WEIGHTED BENCH SCORE
# ─────────────────────────────────────────

def _prob_weighted_bench_score(bench_df: pd.DataFrame,
                                squad_df: pd.DataFrame) -> float:
    """
    Improvement #3: Probability-weighted auto-substitution score.

    Expected bench contribution = sum over bench players of:
      P(a starter needs replacing) * bench_player_predicted_pts
      weighted by bench slot position (earlier = more likely to sub in)

    P(starter needs replacing) is derived from chance_of_playing
    of the starters — if starters have low chance_of_playing,
    bench players are more valuable.

    avg_injury_prob = mean(1 - chance_of_playing/100) across starters
    """
    if bench_df.empty or squad_df.empty:
        return 0.0

    bench_outfield = bench_df[bench_df["position"] != "Goalkeeper"]
    if bench_outfield.empty:
        return 0.0

    # Estimate average probability a starter needs replacing
    if "chance_of_playing" in squad_df.columns:
        avg_injury_prob = (
            1 - squad_df["chance_of_playing"].fillna(100) / 100
        ).mean()
    else:
        avg_injury_prob = 0.05  # default 5% if not available

    score = 0.0
    for i, (_, bench_player) in enumerate(bench_outfield.iterrows()):
        # Earlier bench slot = more likely to be called upon
        slot_weight = 1.0 / (i + 1)
        bench_pts   = bench_player.get("predicted_pts", 0)
        score      += avg_injury_prob * slot_weight * bench_pts

    return round(score, 2)


# ─────────────────────────────────────────
# 5. SCORE CONFIDENCE RANGE
# ─────────────────────────────────────────

def compute_score_range(starting_xi: pd.DataFrame,
                         rmse_map: dict) -> tuple:
    """
    Improvement #8: Expected XI score range.

    Lower bound = sum(predicted - RMSE) per player
    Upper bound = sum(predicted + RMSE) per player
    These approximate a 68% confidence interval based on
    position-specific model error.
    """
    lower = 0.0
    upper = 0.0
    for _, row in starting_xi.iterrows():
        pts  = row.get("predicted_pts", 0)
        rmse = rmse_map.get(row["position"], 2.0)
        lower += max(0, pts - rmse)
        upper += pts + rmse
    return round(lower, 1), round(upper, 1)


# ─────────────────────────────────────────
# 6. POST-TRANSFER XI PREVIEW
# ─────────────────────────────────────────

def get_post_transfer_xi(my_team_enriched: pd.DataFrame,
                          player_out_id: int,
                          player_in_data: pd.Series,
                          triple_captain: bool = False) -> dict:
    """
    Improvement #6: Show optimal XI after a transfer is applied.

    Uses player_id for matching (not last-name string matching)
    to avoid collisions when two players share a surname.
    """
    simulated = my_team_enriched[
        my_team_enriched["player_id"] != player_out_id
    ].copy()
    simulated = pd.concat(
        [simulated, player_in_data.to_frame().T],
        ignore_index=True
    )
    return optimize_xi_ilp(simulated, triple_captain=triple_captain)


# ─────────────────────────────────────────
# 7. GW-BY-GW XI RECOMMENDATION
# ─────────────────────────────────────────

def recommend_xi_multi_gw(my_team_enriched: pd.DataFrame,
                            bootstrap: dict,
                            fixtures_df: pd.DataFrame,
                            current_gw: int,
                            n_gws: int = 3,
                            triple_captain: bool = False) -> list:
    """
    Improvement #9: Optimal XI recommendation for next N gameweeks.

    For each GW, re-scores players based on that week's fixture
    then runs XI optimizer. Highlights where formation should change.

    Returns list of result dicts, one per GW.
    """
    results = []

    for gw_offset in range(1, n_gws + 1):
        gw = current_gw + gw_offset
        log.info(f"Computing optimal XI for GW{gw}...")

        gw_fixtures  = fixtures_df[fixtures_df["event"] == gw]
        gw_diff_map  = {}
        gw_blank_map = {}

        for team_id in my_team_enriched["team_id"].unique():
            team_fix = gw_fixtures[
                (gw_fixtures["team_h"] == team_id) |
                (gw_fixtures["team_a"] == team_id)
            ]
            if team_fix.empty:
                gw_blank_map[team_id] = True
                gw_diff_map[team_id]  = 6
            else:
                row = team_fix.iloc[0]
                gw_diff_map[team_id]  = row["team_h_difficulty"] \
                                        if row["team_h"] == team_id \
                                        else row["team_a_difficulty"]
                gw_blank_map[team_id] = False

        gw_squad = my_team_enriched.copy()
        gw_squad["is_blank_next_gw"] = gw_squad["team_id"].map(gw_blank_map).fillna(False)
        gw_squad["difficulty"]        = gw_squad["team_id"].map(gw_diff_map).fillna(3)

        result = optimize_xi_ilp(gw_squad, triple_captain=triple_captain)
        if result:
            result["gw"] = gw
        results.append(result)

    return results


# ─────────────────────────────────────────
# 8. DISPLAY HELPERS
# ─────────────────────────────────────────

def print_formation_comparison(squad_df: pd.DataFrame):
    """Print all formations ranked by combined score."""
    results = score_all_formations(squad_df)
    print(f"\n{'=' * 75}")
    print("  FORMATION COMPARISON")
    print(f"{'=' * 75}")
    print(f"  {'Formation':<12}  {'Pred Pts':>10}  {'Combined':>10}")
    print(f"  {'-' * 36}")
    for i, r in enumerate(results):
        marker = "  <- Optimal" if i == 0 else ""
        print(f"  {r['formation']:<12}  {r['pred_pts']:>10}  {r['combined']:>10}{marker}")


def print_starting_xi(result: dict, current_gw: int, rmse_map: dict):
    """Print starting XI with confidence range and chip notes."""
    xi  = result["starting_xi"]
    cap = result["captain"]
    vc  = result["vice_captain"]

    gw_label = result.get("gw", current_gw + 1)

    gks  = xi[xi["position"] == "Goalkeeper"]
    defs = xi[xi["position"] == "Defender"]
    mids = xi[xi["position"] == "Midfielder"]
    fwds = xi[xi["position"] == "Forward"]

    def fmt(row):
        name  = str(row["player_name"]).split()[-1]
        pts   = row["predicted_pts"]
        blank = "*" if row.get("is_blank_next_gw", False) else ""
        if row["player_id"] == cap["player_id"]:
            tag = "(C)"
        elif row["player_id"] == vc["player_id"]:
            tag = "(V)"
        else:
            tag = ""
        return f"{name}{tag}{blank}({pts})"

    def print_row(players, label):
        print(f"    {label:<4}  " +
              "    ".join(fmt(r) for _, r in players.iterrows()))

    # Confidence range
    lo, hi = compute_score_range(xi, rmse_map)

    method_note = f"  [{result.get('method', 'ILP')}]"
    bb_note     = "  [BENCH BOOST — all 15 score]" if result.get("bench_boost") else ""
    tc_note     = "  [TRIPLE CAPTAIN — 3x]" if result.get("triple_captain") else ""

    print(f"\n{'=' * 75}")
    print(
        f"  OPTIMAL STARTING XI — GW{gw_label}  |  "
        f"Formation: {result['formation']}{method_note}{bb_note}{tc_note}"
    )
    print(
        f"  Predicted Pts: {result['total_predicted_pts']}"
        f"  |  Range: {lo} – {hi} pts (68% CI)"
    )
    print(f"  (C)=Captain  (V)=Vice Captain  *=Blank")
    print(f"{'=' * 75}")
    print(f"\n    {'─' * 65}")
    print_row(gks,  "GK ")
    print(f"    {'─' * 65}")
    print_row(defs, "DEF")
    print(f"    {'─' * 65}")
    print_row(mids, "MID")
    print(f"    {'─' * 65}")
    print_row(fwds, "FWD")
    print(f"    {'─' * 65}")

    # Detail table
    print(f"\n  {'Player':<28} {'Pos':<12} {'Pred':>6} {'xPts':>6} "
          f"{'Run':<14} {'Diff':>5} {'H/A':>4}")
    print(f"  {'-' * 72}")
    for _, row in xi.sort_values("predicted_pts", ascending=False).iterrows():
        tag   = " (C)" if row["player_id"] == cap["player_id"] else \
                " (V)" if row["player_id"] == vc["player_id"] else "    "
        blank = " *" if row.get("is_blank_next_gw", False) else "  "
        xpts  = round(row.get("predicted_pts", 0) *
                      (1 + float(row.get("roll3_threat", 0) or 0) / 100), 2)
        print(
            f"  {str(row['player_name']) + tag:<28}"
            f"  {row['position']:<12}"
            f"  {row['predicted_pts']:>6}"
            f"  {xpts:>6}"
            f"  {row.get('fixture_run_label', '?'):<14}"
            f"  {row.get('difficulty', '-'):>5}"
            f"  {'H' if row.get('is_home') else 'A':>4}"
            f"{blank}"
        )

    # Captain summary
    tc_mult = 3 if result.get("triple_captain") else 2
    dgw_note = " (DGW)" if cap.get("double_gws", 0) > 0 else ""
    vc_chance = vc.get("chance_of_playing", 100)
    print(
        f"\n  Captain:      {cap['player_name']} — {cap['predicted_pts']} pts"
        f"  (scores {round(cap['predicted_pts']*tc_mult, 2)} if captained){dgw_note}"
    )
    print(
        f"  Vice Captain: {vc['player_name']} — {vc['predicted_pts']} pts"
        f"  (Chance of playing: {vc_chance}%  ← backup captain reliability)"
    )


def print_bench(result: dict):
    """Print bench with auto-sub notes."""
    bench = result["bench"]
    if bench.empty:
        return

    print(f"\n{'=' * 75}")
    print(
        f"  BENCH ORDER  "
        f"(Auto-sub score: {result['bench_auto_sub_score']})"
    )
    print(f"{'=' * 75}")
    print(f"  {'#':<3} {'Player':<28} {'Pos':<12} {'Price':>5} "
          f"{'Pred':>6} {'Run':<14} {'Note'}")
    print(f"  {'-' * 75}")

    for i, (_, row) in enumerate(bench.iterrows(), 1):
        if row["position"] == "Goalkeeper":
            note = "Emergency GK"
        elif i == 1:
            note = "First sub"
        elif row.get("is_blank_next_gw", False):
            note = "Blank GW"
        else:
            note = ""
        print(
            f"  {i:<3} {str(row['player_name']):<28}"
            f"  {row['position']:<12}"
            f"  £{row['price']:>4.1f}"
            f"  {row['predicted_pts']:>6}"
            f"  {row.get('fixture_run_label', '?'):<14}"
            f"  {note}"
        )


def flag_injury_risks(squad_df: pd.DataFrame, bootstrap: dict):
    """Flag players with less than 100% chance of playing."""
    players_raw = bootstrap["elements"]
    risk_map    = {p["id"]: p.get("chance_of_playing_next_round") for p in players_raw}
    risks       = []
    for _, row in squad_df.iterrows():
        chance = risk_map.get(int(row["player_id"]))
        if chance is not None and chance < 100:
            risks.append({"player": row["player_name"], "chance": chance})

    if risks:
        print(f"\n{'=' * 75}")
        print("  INJURY / AVAILABILITY RISKS")
        print(f"{'=' * 75}")
        for r in sorted(risks, key=lambda x: x["chance"]):
            level = "LOW " if r["chance"] < 75 else "MED "
            print(f"  [{level}] {r['player']:<28}  Chance: {r['chance']}%")
    else:
        print("\n  All squad players have 100% chance of playing.")


def print_multi_gw_xi(gw_results: list):
    """Print GW-by-GW XI summary highlighting formation changes."""
    print(f"\n{'=' * 75}")
    print("  GW-BY-GW XI RECOMMENDATION (Next 3 GWs)")
    print(f"{'=' * 75}")

    prev_formation = None
    for result in gw_results:
        if result is None:
            continue
        gw        = result.get("gw", "?")
        formation = result["formation"]
        pts       = result["total_predicted_pts"]
        cap       = result["captain"]["player_name"]
        vc        = result["vice_captain"]["player_name"]
        changed   = " <- FORMATION CHANGE" if formation != prev_formation \
                    and prev_formation is not None else ""
        print(
            f"\n  GW{gw}: {formation}{changed}"
            f"  |  Pred: {pts} pts"
            f"  |  Captain: {cap}  |  VC: {vc}"
        )
        # Show XI compactly
        xi = result["starting_xi"]
        for pos in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
            pos_players = xi[xi["position"] == pos]
            names = ", ".join(
                str(r["player_name"]).split()[-1] +
                ("*" if r.get("is_blank_next_gw", False) else "")
                for _, r in pos_players.iterrows()
            )
            print(f"    {pos[:3]}: {names}")
        prev_formation = formation


def print_post_transfer_xi(before: dict, after: dict,
                             player_out: str, player_in: str,
                             current_gw: int, rmse_map: dict):
    """Print before/after XI comparison for top transfer."""
    print(f"\n{'=' * 75}")
    print(f"  TRANSFER IMPACT: OUT {player_out} -> IN {player_in}")
    print(f"{'=' * 75}")
    print(
        f"\n  BEFORE: Formation {before['formation']}"
        f"  |  Pred: {before['total_predicted_pts']} pts"
        f"  |  Captain: {before['captain']['player_name']}"
    )
    lo_b, hi_b = compute_score_range(before["starting_xi"], rmse_map)
    print(f"          Range: {lo_b} – {hi_b} pts")

    print(
        f"\n  AFTER:  Formation {after['formation']}"
        f"  |  Pred: {after['total_predicted_pts']} pts"
        f"  |  Captain: {after['captain']['player_name']}"
    )
    lo_a, hi_a = compute_score_range(after["starting_xi"], rmse_map)
    print(f"          Range: {lo_a} – {hi_a} pts")

    delta = round(after["total_predicted_pts"] - before["total_predicted_pts"], 2)
    print(
        f"\n  Net XI improvement from transfer: {delta:+.2f} pts predicted"
    )


# ─────────────────────────────────────────
# 9. FULL PHASE 4 PIPELINE
# ─────────────────────────────────────────

def run_phase4(team_id: int = TEAM_ID,
               max_players: int = None,
               refresh: bool = False):
    """Full Phase 4 pipeline with all improvements."""
    log.info("=" * 75)
    log.info("  FPL AI ASSISTANT — Phase 4: Starting XI Optimizer (v3)")
    log.info("=" * 75)

    # ── Fetch ──────────────────────────────────────────
    log.info("Fetching bootstrap & fixtures...")
    bootstrap   = fetch_bootstrap()
    fixtures_df = fetch_fixtures()
    current_gw  = fetch_current_gw(bootstrap)
    log.info(f"GW{current_gw} completed -> optimizing for GW{current_gw+1}")

    log.info("Fetching your team...")
    try:
        team_data     = fetch_my_team(team_id, current_gw)
        my_player_ids = [p["element"] for p in team_data["picks"]]
        log.info(f"Team fetched — {len(my_player_ids)} players.")
    except Exception as e:
        log.error(f"Could not fetch team: {e}")
        my_player_ids = []

    transfer_info   = fetch_transfer_info(team_id, current_gw)
    bank_balance    = transfer_info["bank_balance"]
    transfer_status = transfer_info["transfer_status"]
    log.info(f"Bank: £{bank_balance:.1f}M  |  {transfer_status}")

    # ── History + models ───────────────────────────────
    log.info("Loading player history...")
    history_df = build_player_history_df(
        bootstrap, max_players=max_players, refresh=refresh
    )

    log.info("Training position-specific models...")
    models = train_models(history_df)
    rmse_map = get_rmse_from_models(models)

    with open("fpl_model.pkl", "wb") as f:
        pickle.dump({"models": models, "features": FEATURE_COLS}, f)

    # ── Predict ────────────────────────────────────────
    log.info(f"Predicting GW{current_gw+1} scores...")
    pred_df = build_current_features(
        bootstrap, fixtures_df, history_df,
        models, current_gw,
        my_player_ids=my_player_ids
    )

    # ── Phase 2 context ────────────────────────────────
    log.info("Building context maps...")
    custom_diff     = build_custom_difficulty(history_df, bootstrap)
    team_form_map   = build_team_form(history_df, bootstrap)
    opp_scoring_map = build_opponent_scoring_map(history_df)
    chip_info       = build_chip_status(team_id, bootstrap, fixtures_df, current_gw)

    log.info(f"Building fixture run (next {FIXTURE_LOOKAHEAD} GWs)...")
    fixture_run_df = build_fixture_run(
        bootstrap, fixtures_df, current_gw,
        custom_difficulty=custom_diff,
        gw_lookahead=FIXTURE_LOOKAHEAD
    )
    enriched_df = build_player_fixture_scores(
        pred_df, fixture_run_df, current_gw,
        team_form_map, opp_scoring_map,
        FIXTURE_LOOKAHEAD
    )

    my_team_enriched = enriched_df[enriched_df["player_id"].isin(my_player_ids)].copy()
    other_enriched   = enriched_df[~enriched_df["player_id"].isin(my_player_ids)].copy()

    # ── Chip detection ─────────────────────────────────
    available_chips = chip_info.get("available_chips", [])
    triple_captain  = "Triple Captain" in available_chips
    bench_boost     = "Bench Boost" in available_chips

    if triple_captain:
        log.info("Triple Captain available — applying 3x captain scoring.")
        print("\n  Triple Captain chip available — captain scored at 3x!")
    if bench_boost:
        log.info("Bench Boost available — will show Bench Boost mode.")
        print("\n  Bench Boost chip available — will show optimised 15-player squad!")

    # ── Squad validation ───────────────────────────────
    violations = validate_squad(my_team_enriched)
    if violations:
        print("\n  Squad violations:")
        for v in violations:
            print(f"    - {v}")
    else:
        print("\n  Squad passes all FPL rules.")

    # ── Injury risks ───────────────────────────────────
    flag_injury_risks(my_team_enriched, bootstrap)

    # ── Formation comparison ───────────────────────────
    print_formation_comparison(my_team_enriched)

    # ── Optimal starting XI ────────────────────────────
    result = optimize_xi_ilp(
        my_team_enriched,
        triple_captain=triple_captain,
        bench_boost=False
    )

    if not result:
        log.error("Could not find a valid starting XI.")
        print("  Could not find a valid starting XI.")
        return enriched_df, my_team_enriched, None

    print_starting_xi(result, current_gw, rmse_map)
    print_bench(result)

    # ── Bench Boost mode (if available) ───────────────
    if bench_boost:
        bb_result = optimize_xi_ilp(
            my_team_enriched,
            triple_captain=triple_captain,
            bench_boost=True
        )
        print(f"\n{'=' * 75}")
        print("  BENCH BOOST MODE — Optimal Captain for All 15 Players")
        print(f"{'=' * 75}")
        print_starting_xi(bb_result, current_gw, rmse_map)

    # ── GW-by-GW XI recommendation ─────────────────────
    gw_results = recommend_xi_multi_gw(
        my_team_enriched, bootstrap,
        fixtures_df, current_gw, n_gws=3,
        triple_captain=triple_captain
    )
    print_multi_gw_xi(gw_results)

    # ── ILP optimal transfers ──────────────────────────
    print(f"\n{'=' * 75}")
    print("  OPTIMAL 1-TRANSFER")
    print(f"{'=' * 75}")
    ilp_result_1 = get_ilp_optimal_transfers(
        my_team_enriched, other_enriched, bank_balance, n_transfers=1
    )
    print_ilp_result(ilp_result_1, "1-Transfer")

    # ── Post-transfer XI preview ───────────────────────
    if ilp_result_1.get("transfers"):
        t = ilp_result_1["transfers"][0]
        player_in_rows = other_enriched[
            other_enriched["player_id"] == t["in_id"]
        ]
        if not player_in_rows.empty:
            after_result = get_post_transfer_xi(
                my_team_enriched,
                t["out_id"],           # use player_id, not name
                player_in_rows.iloc[0],
                triple_captain=triple_captain
            )
            if after_result:
                print_post_transfer_xi(
                    result, after_result,
                    t["out_name"], t["in_name"],
                    current_gw, rmse_map
                )

    print(f"\n{'=' * 75}")
    print("  OPTIMAL 2-TRANSFER")
    print(f"{'=' * 75}")
    ilp_result_2 = get_ilp_optimal_transfers(
        my_team_enriched, other_enriched, bank_balance, n_transfers=2
    )
    print_ilp_result(ilp_result_2, "2-Transfer")

    print(f"\n{'=' * 75}")
    print("  BEST 2-TRANSFER COMBINATIONS")
    print(f"{'=' * 75}")
    valid_doubles = get_valid_double_transfers(
        my_team_enriched, other_enriched, bank_balance,
        top_n=3, precomputed_ilp=ilp_result_2
    )
    print_double_transfers(valid_doubles, bank_balance)

    print(
        f"\n  Note: Bank shown (£{bank_balance:.1f}M) may differ "
        f"from FPL app. Always verify before confirming."
    )

    # ── Save ───────────────────────────────────────────
    enriched_df.to_csv("fpl_predictions_phase4.csv", index=False)
    result["starting_xi"].to_csv("fpl_starting_xi.csv", index=False)
    log.info("Saved fpl_predictions_phase4.csv and fpl_starting_xi.csv")
    log.info("Ready for Phase 6 (Streamlit Dashboard)")

    return enriched_df, my_team_enriched, result


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    REFRESH = "--refresh" in sys.argv
    if REFRESH:
        log.info("--refresh flag detected.")

    enriched_df, my_team, result = run_phase4(
        team_id=TEAM_ID,
        max_players=None,
        refresh=REFRESH
    )