"""
FPL AI Assistant — Phase 4: Starting XI Optimizer (v5)
======================================================
New in v5 (5 algorithmic additions):

  🔴 ALGORITHMIC:
  1. ILP soft injury penalty — objective now weights each player by
     p_plays_full so rotation risks don't start at full value.
     objective_i = combined_score_i * p_plays_full_i
     A player predicted 9 pts but only 40% chance of playing contributes
     3.6 to the objective, not 9. Formation adapts to real availability.

  2. Monte Carlo captain — imports run_monte_carlo_captain from Phase 3
     and surfaces win_prob + expected_captain_gain for top-3 options.
     Shows exactly how likely each captain choice is to be optimal across
     1,000 simulations from the pts_low/pts_high quantile distribution.

  3. Bench ordering from Phase 3 — imports get_bench_order_recommendation
     from Phase 3 v5, which uses bench_ev = expected_pts × P(auto-sub
     needed) × gk_penalty. Replaces the inferior combined_score sort.
     GK always last. First sub is the highest bench_ev outfield player.

  🟡 FEATURE:
  4. fixture_trend decay in multi-GW captain score — for GW+2 and GW+3
     captain recommendations, captain_ev is penalised by fixture_trend:
     gw_cap_ev = captain_ev - max(0, fixture_trend * gw_offset * 0.3)
     A player whose fixtures get harder over the window is correctly
     downgraded as a future captain option.

  5. Post-transfer bench score delta — print_post_transfer_xi now shows
     the change in bench auto-sub score after the transfer:
     bench_delta = after_bench_score - before_bench_score
     Sometimes a transfer improves the bench more than the XI.

Fix in v5.1:

  🟡 FIX — Bench ordering XI mismatch resolved (v5.1):
     The original v5 _build_result called get_bench_order_recommendation(
     squad_df) which independently re-selects its own "best 11" from all
     15 players using a plain combined_score greedy sort. The ILP selects
     its XI using a p_plays_full-weighted objective, so the two XIs
     legitimately differ whenever any squad player has p_plays_full < 1.0.
     When they diverged, the name-rank lookup assigned rank 999 to ILP
     bench players that weren't on Phase 3's bench, causing all of them to
     silently fall back to combined_score ordering — defeating bench_ev
     entirely for precisely the squads with injury/rotation uncertainty
     (where correct bench ordering matters most).

     Fixed by replacing the cross-function bench ordering call with a new
     self-contained helper _order_bench_by_ev(bench_pool, starting_xi)
     that computes bench_ev directly from the ILP's actual starters and
     bench players, with no re-selection step. get_bench_order_recommendation
     is no longer imported or called in _build_result.

Changes preserved from v4 (10 fixes):
  - Phase 1 v5 pipeline (component blend, expected_pts, prices)
  - pkl freshness check
  - xpts_captain_score uses expected_pts not predicted_pts
  - vc_safety_score uses p_plays_full not chance_of_playing
  - _prob_weighted_bench_score uses p_plays_full for starters
  - compute_score_range uses pts_low/pts_high quantile CI
  - recommend_xi_multi_gw per-GW correct scoring
  - Captain uses captain_ev from Phase 2 v5
  - Bench Boost uses captain_ev
  - cs_probability_map passed to build_player_fixture_scores

Run normally:  python fpl_phase4_optimizer.py
Force refresh: python fpl_phase4_optimizer.py --refresh
"""

import os
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
    train_component_models,
    predict_component_pts,
    add_price_predictions,
    train_price_model,
    compute_expected_pts,
    FEATURE_COLS,
    LOG_FILE,
    COMPONENT_BLEND_WEIGHT,
)
from fpl_phase2_fixtures import (
    build_custom_difficulty,
    build_team_form,
    build_opponent_scoring_map,
    build_chip_status,
    build_fixture_run,
    build_player_fixture_scores,
    build_cs_probability_map,
)
from fpl_phase3_constraints import (
    validate_squad,
    get_ilp_optimal_transfers,
    get_valid_double_transfers,
    print_ilp_result,
    print_double_transfers,
    run_monte_carlo_captain,
    # get_bench_order_recommendation intentionally NOT imported here.
    # _build_result now uses the self-contained _order_bench_by_ev helper
    # which operates directly on the ILP's actual bench_pool and starting_xi,
    # avoiding the XI mismatch bug described in the v5.1 docstring.
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

# Fallback RMSE per position (used when pts_low/pts_high absent)
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
    """Return formation string e.g. '4-4-2'."""
    return f"{d}-{m}-{f}"


def get_rmse_from_models(models: dict) -> dict:
    """Extract RMSE per position from trained models dict."""
    rmse = {}
    for pos, info in models.items():
        rmse[pos] = info.get("rmse", DEFAULT_RMSE.get(pos, 2.0))
    return rmse


def _xpts_base(row: pd.Series) -> float:
    """
    Return the rotation-risk-adjusted expected pts for a player.
    Prefers expected_pts (Phase 1 v5) over predicted_pts (v3 fallback).
    """
    xpts = row.get("expected_pts")
    if xpts is not None and not pd.isna(xpts) and float(xpts) > 0:
        return float(xpts)
    return float(row.get("predicted_pts", 0))


def xpts_captain_score(row: pd.Series,
                        triple_captain: bool = False) -> float:
    """
    Expected points score for captain consideration.

    v4 fix: uses expected_pts (rotation-risk adjusted) as base instead
    of predicted_pts. roll3_threat upside proxy preserved.

    Multiplier: 3x if Triple Captain available, else 2x.
    DGW boost via CAPTAIN_DGW_MULTIPLIER.
    Blank GW players always return 0.
    """
    if row.get("is_blank_next_gw", False):
        return 0.0

    # v4: use expected_pts (rotation-aware) as base
    pts    = _xpts_base(row)
    threat = float(row.get("roll3_threat", 0) or 0)
    upside = 1.0 + (threat / 100.0)
    xpts   = pts * upside

    if row.get("double_gws", 0) > 0:
        xpts *= CAPTAIN_DGW_MULTIPLIER

    multiplier = 3 if triple_captain else 2
    return round(xpts * multiplier, 3)


def vc_safety_score(row: pd.Series) -> float:
    """
    VC safety score — combines high expected pts with high reliability.

    v4 fix: uses expected_pts * p_plays_full (Phase 1 v5 continuous
    rotation probability) instead of predicted_pts * chance_of_playing.
    p_plays_full is derived from rolling minutes + injury %, more sensitive
    than the raw FPL categorical field. Falls back gracefully.
    """
    if row.get("is_blank_next_gw", False):
        return 0.0

    pts = _xpts_base(row)

    # v4: prefer p_plays_full (Phase 1 v5) over chance_of_playing (v3)
    p_full = row.get("p_plays_full")
    if p_full is not None and not pd.isna(p_full):
        reliability = float(p_full)
    else:
        reliability = float(row.get("chance_of_playing", 100)) / 100.0

    return round(pts * reliability, 3)


# ─────────────────────────────────────────
# 2. ILP STARTING XI
# ─────────────────────────────────────────

def optimize_xi_ilp(squad_df: pd.DataFrame,
                     triple_captain: bool = False,
                     bench_boost: bool = False) -> dict:
    """
    ILP-based Starting XI optimizer.

    Selects all 11 players simultaneously — formation adapts automatically
    to injuries and blanks via constraints. Falls back to brute-force if
    PuLP unavailable.

    bench_boost=True: all 15 players score, selects best captain.
    """
    if bench_boost:
        return _bench_boost_mode(squad_df, triple_captain)

    if not PULP_AVAILABLE:
        log.warning("PuLP not available — using brute-force fallback.")
        return optimize_xi_bruteforce(squad_df, triple_captain)

    players = squad_df.reset_index(drop=True)
    n       = len(players)
    prob    = pulp.LpProblem("Starting_XI", pulp.LpMaximize)
    x       = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Item 1 (v5): soft injury penalty — weight objective by p_plays_full.
    # objective_i = combined_score_i * p_plays_full_i
    # A player with p_plays_full=0.4 contributes 40% of their combined_score.
    # Falls back to 1.0 (no penalty) when p_plays_full column absent.
    has_p_full = "p_plays_full" in players.columns
    def _obj_weight(i: int) -> float:
        if not has_p_full:
            return 1.0
        val = players.loc[i, "p_plays_full"]
        return float(val) if (val is not None and not pd.isna(val)) else 1.0

    # Objective: maximise p_plays_full-weighted combined_score of starters
    prob += pulp.lpSum(
        players.loc[i, "combined_score"] * _obj_weight(i) * x[i]
        for i in range(n)
    )

    # Exactly 11 starters
    prob += (pulp.lpSum(x) == 11, "squad_size")

    # GK: exactly 1 starts
    gk_idx  = players[players["position"] == "Goalkeeper"].index.tolist()
    prob += (pulp.lpSum(x[i] for i in gk_idx) == 1, "gk")

    # DEF: 3–5
    def_idx = players[players["position"] == "Defender"].index.tolist()
    prob += (pulp.lpSum(x[i] for i in def_idx) >= 3, "min_def")
    prob += (pulp.lpSum(x[i] for i in def_idx) <= 5, "max_def")

    # MID: 2–5
    mid_idx = players[players["position"] == "Midfielder"].index.tolist()
    prob += (pulp.lpSum(x[i] for i in mid_idx) >= 2, "min_mid")
    prob += (pulp.lpSum(x[i] for i in mid_idx) <= 5, "max_mid")

    # FWD: 1–3
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

    def_n = int((starting_xi["position"] == "Defender").sum())
    mid_n = int((starting_xi["position"] == "Midfielder").sum())
    fwd_n = int((starting_xi["position"] == "Forward").sum())

    return _build_result(
        starting_xi, bench_pool, squad_df,
        formation_name(def_n, mid_n, fwd_n),
        triple_captain, method="ILP"
    )


def _bench_boost_mode(squad_df: pd.DataFrame,
                       triple_captain: bool) -> dict:
    """
    Bench Boost mode — all 15 players score.

    v4: uses captain_ev (Phase 2 v5) for captain selection if present,
    falls back to xpts_captain_score.
    """
    players = squad_df.reset_index(drop=True).copy()

    # v4 fix #9: use captain_ev if available
    if "captain_ev" in players.columns:
        cap_col = "captain_ev"
        # Apply TC multiplier on top (captain_ev is already 2x, multiply by 1.5x for 3x)
        if triple_captain:
            players["_cap_sort"] = players["captain_ev"] * 1.5
        else:
            players["_cap_sort"] = players["captain_ev"]
        captain = players.nlargest(1, "_cap_sort").iloc[0]
    else:
        players["_cap_score"] = players.apply(
            lambda r: xpts_captain_score(r, triple_captain), axis=1
        )
        captain = players.nlargest(1, "_cap_score").iloc[0]

    vc_pool = players[players["player_id"] != captain["player_id"]].copy()
    vc_pool["_vc_score"] = vc_pool.apply(vc_safety_score, axis=1)
    vice_captain = vc_pool.nlargest(1, "_vc_score").iloc[0]

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


def _order_bench_by_ev(bench_pool: pd.DataFrame,
                        starting_xi: pd.DataFrame) -> pd.DataFrame:
    """
    Order the ILP bench players by expected auto-sub contribution (bench_ev).

    bench_ev per player = xpts * p_at_least_one_miss * gk_penalty
      where p_at_least_one_miss = max(0.05, 1 - prod(p_plays_full for starters))
      and   gk_penalty = 0.3 for GK (rare auto-sub), 1.0 for outfield.

    GK is always last regardless of bench_ev.
    Outfield players are sorted by bench_ev descending.

    WHY THIS EXISTS (v5.1 fix):
    The previous approach called get_bench_order_recommendation(squad_df)
    from Phase 3, which re-runs its own greedy XI selector on all 15 players
    using a plain combined_score sort. The ILP selects its XI using a
    p_plays_full-weighted objective — so the two XIs differ whenever any
    player has p_plays_full < 1.0. When they diverged, ILP bench players
    were not found in Phase 3's bench_order list, got rank 999 in the
    name-lookup, and silently fell back to combined_score sort — exactly
    the wrong outcome for squads with injury/rotation risk, where correct
    bench ordering matters most.

    This helper uses the ILP's actual starting_xi to compute
    p_at_least_one_miss and ranks the ILP's actual bench_pool directly.
    No re-selection. No name-lookup. No silent fallback.
    """
    if bench_pool.empty:
        return bench_pool.reset_index(drop=True)

    # P(at least one starter misses) — floored at 0.05 matching Phase 3 v5.1
    if "p_plays_full" in starting_xi.columns:
        p_full_vals = starting_xi["p_plays_full"].fillna(1.0).astype(float).values
    else:
        p_full_vals = np.ones(len(starting_xi))
    p_miss = max(0.05, 1.0 - float(np.prod(p_full_vals)))

    xpts_col = "expected_pts" if "expected_pts" in bench_pool.columns else "predicted_pts"
    pool = bench_pool.copy()

    pool["_bench_ev"] = pool.apply(
        lambda r: round(
            float(r.get(xpts_col, 0) or 0) * p_miss *
            (0.3 if r["position"] == "Goalkeeper" else 1.0),
            3,
        ),
        axis=1,
    )
    pool["_is_gk"] = (pool["position"] == "Goalkeeper")

    # Sort: outfield first (ascending _is_gk = False before True),
    # within outfield by bench_ev descending.
    bench_ordered = (
        pool.sort_values(["_is_gk", "_bench_ev"], ascending=[True, False])
        .drop(columns=["_bench_ev", "_is_gk"])
        .reset_index(drop=True)
    )
    return bench_ordered


def _build_result(starting_xi: pd.DataFrame,
                   bench_pool: pd.DataFrame,
                   squad_df: pd.DataFrame,
                   formation: str,
                   triple_captain: bool,
                   method: str = "ILP") -> dict:
    """
    Build result dict from selected XI and bench.

    v4 fix #8: captain selection prefers captain_ev (Phase 2 v5) over
    local xpts_captain_score when the column is available.
    """
    xi_copy = starting_xi.copy()

    # v4 fix #8: use captain_ev if present
    if "captain_ev" in xi_copy.columns:
        if triple_captain:
            xi_copy["_cap_sort"] = xi_copy["captain_ev"] * 1.5
        else:
            xi_copy["_cap_sort"] = xi_copy["captain_ev"]
        captain = xi_copy.nlargest(1, "_cap_sort").iloc[0]
    else:
        xi_copy["_cap_score"] = xi_copy.apply(
            lambda r: xpts_captain_score(r, triple_captain), axis=1
        )
        captain = xi_copy.nlargest(1, "_cap_score").iloc[0]

    # VC: safety score using expected_pts * p_plays_full (v4 fix #4)
    vc_pool = xi_copy[xi_copy["player_id"] != captain["player_id"]].copy()
    vc_pool["_vc_score"] = vc_pool.apply(vc_safety_score, axis=1)
    vice_captain = vc_pool.nlargest(1, "_vc_score").iloc[0]

    # v5.1: bench ordering via _order_bench_by_ev — computes bench_ev
    # directly from the ILP's actual starting_xi and bench_pool.
    # Replaces the previous get_bench_order_recommendation(squad_df) call
    # which re-selected its own XI and caused ordering mismatches for any
    # squad with injury/rotation risk. See _order_bench_by_ev docstring.
    bench_ordered = _order_bench_by_ev(bench_pool, starting_xi)

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
    """Pick GK with best expected_pts (fallback predicted_pts) for upcoming GW."""
    if gks_df.empty:
        return gks_df
    if "is_blank_next_gw" in gks_df.columns:
        playing = gks_df[~gks_df["is_blank_next_gw"]]
        if not playing.empty:
            # v4: prefer expected_pts for GK selection
            sort_col = "expected_pts" if "expected_pts" in playing.columns else "predicted_pts"
            return playing.nlargest(1, sort_col).iloc[[0]]
    sort_col = "expected_pts" if "expected_pts" in gks_df.columns else "combined_score"
    return gks_df.nlargest(1, sort_col).iloc[[0]]


def optimize_xi_bruteforce(squad_df: pd.DataFrame,
                             triple_captain: bool = False) -> dict:
    """Brute-force Starting XI fallback when PuLP unavailable."""
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
            best_score = score
            bench_pool = squad_df[
                ~squad_df["player_id"].isin(xi["player_id"])
            ].copy()
            best_result = _build_result(
                xi, bench_pool, squad_df,
                formation_name(def_n, mid_n, fwd_n),
                triple_captain, method="BruteForce"
            )
    return best_result


def score_all_formations(squad_df: pd.DataFrame) -> list:
    """Score every valid formation for the formation comparison table."""
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
        # v4: show expected_pts alongside predicted_pts
        xpts_total = round(
            xi["expected_pts"].sum() if "expected_pts" in xi.columns
            else xi["predicted_pts"].sum(), 2
        )
        results.append({
            "formation": formation_name(def_n, mid_n, fwd_n),
            "pred_pts":  round(xi["predicted_pts"].sum(), 2),
            "xpts":      xpts_total,
            "combined":  round(xi["combined_score"].sum(), 2),
        })
    return sorted(results, key=lambda x: x["combined"], reverse=True)


# ─────────────────────────────────────────
# 4. PROBABILITY-WEIGHTED BENCH SCORE
# ─────────────────────────────────────────

def _prob_weighted_bench_score(bench_df: pd.DataFrame,
                                squad_df: pd.DataFrame) -> float:
    """
    Probability-weighted auto-substitution score.

    v4 fix: uses p_plays_full (Phase 1 v5 continuous rotation probability)
    for starters instead of chance_of_playing (raw FPL categorical).
    Falls back to chance_of_playing if p_plays_full absent.

    E[bench contribution] = sum over bench slots of:
      P(auto-sub needed) * (1/(slot+1)) * bench_xpts
    """
    if bench_df.empty or squad_df.empty:
        return 0.0

    bench_outfield = bench_df[bench_df["position"] != "Goalkeeper"]
    if bench_outfield.empty:
        return 0.0

    # v4: prefer p_plays_full for starters
    if "p_plays_full" in squad_df.columns:
        avg_injury_prob = float(
            (1.0 - squad_df["p_plays_full"].fillna(1.0)).mean()
        )
    elif "chance_of_playing" in squad_df.columns:
        avg_injury_prob = float(
            (1.0 - squad_df["chance_of_playing"].fillna(100) / 100.0).mean()
        )
    else:
        avg_injury_prob = 0.05   # 5% default

    score = 0.0
    for i, (_, bp) in enumerate(bench_outfield.iterrows()):
        slot_weight = 1.0 / (i + 1)
        # Use expected_pts for bench player value (v4)
        bench_pts = float(bp.get("expected_pts", bp.get("predicted_pts", 0)))
        score    += avg_injury_prob * slot_weight * bench_pts

    return round(score, 2)


# ─────────────────────────────────────────
# 5. SCORE CONFIDENCE RANGE
# ─────────────────────────────────────────

def compute_score_range(starting_xi: pd.DataFrame,
                         rmse_map: dict) -> tuple:
    """
    Expected XI score confidence interval.

    v4 fix: uses pts_low / pts_high (Q10/Q90 from Phase 1 v5 quantile
    regression) when available — these are player-specific and asymmetric.
    Falls back to symmetric RMSE ± if quantile columns absent.

    Returns (lower_bound, upper_bound) as a 68%/80% CI approximation.
    """
    has_quantiles = (
        "pts_low"  in starting_xi.columns and
        "pts_high" in starting_xi.columns
    )

    lower = 0.0
    upper = 0.0

    for _, row in starting_xi.iterrows():
        pts = float(row.get("predicted_pts", 0))
        if has_quantiles:
            lo = float(row.get("pts_low",  max(0, pts - rmse_map.get(row["position"], 2.0))))
            hi = float(row.get("pts_high", pts + rmse_map.get(row["position"], 2.0)))
        else:
            rmse = rmse_map.get(row["position"], 2.0)
            lo   = max(0.0, pts - rmse)
            hi   = pts + rmse
        lower += lo
        upper += hi

    ci_label = "Q10–Q90" if has_quantiles else "68% CI (RMSE)"
    return round(lower, 1), round(upper, 1), ci_label


# ─────────────────────────────────────────
# 6. POST-TRANSFER XI PREVIEW
# ─────────────────────────────────────────

def get_post_transfer_xi(my_team_enriched: pd.DataFrame,
                          player_out_id: int,
                          player_in_data: pd.Series,
                          triple_captain: bool = False) -> dict:
    """
    Show optimal XI after applying a transfer.
    Uses player_id matching to avoid surname collision bugs.
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

def _gw_specific_score(row: pd.Series, gw: int, current_gw: int) -> float:
    """
    Compute a GW-specific player score using per-GW fixture difficulty.

    v4 fix: replaces the wrong approach of using combined_score (which
    reflects GW+1 fixture run) for all future GWs. For GW+N we use:
      gw_xscore = expected_pts * (6 - gw_difficulty) / 3.0
    where gw_difficulty comes from the gw{N}_difficulty column stored
    by Phase 2 build_player_fixture_scores.

    The /3.0 normalises to a comparable scale with combined_score.
    Falls back to combined_score when the column is absent (e.g. blanks).
    """
    gw_col    = f"gw{gw}_difficulty"
    gw_blank  = row.get(f"gw{gw}_opponent", "") == "BLANK"
    if gw_blank:
        return 0.0

    diff = float(row.get(gw_col, row.get("avg_difficulty", 3.0)))
    xpts = _xpts_base(row)
    return round(xpts * (6.0 - diff) / 3.0, 3)


def _gw_captain_score(row: pd.Series,
                       gw: int,
                       current_gw: int,
                       triple_captain: bool = False) -> float:
    """
    Item 4 (v5): GW-specific captain score with fixture_trend decay.

    For future GWs, a player's captain EV is penalised if their fixtures
    are trending harder (positive fixture_trend = getting tougher).

    Formula:
      gw_cap_ev = base_captain_ev - max(0, fixture_trend * gw_offset * 0.3)

    where gw_offset = gw - current_gw (1 for next GW, 2 for GW+2, etc.)
    and the 0.3 factor is a conservative penalty weight.

    Blank GW players return 0.0.
    """
    if str(row.get(f"gw{gw}_opponent", "")) == "BLANK":
        return 0.0

    gw_offset = gw - current_gw

    # Base: use captain_ev if available, else compute from xpts
    if "captain_ev" in row.index and not pd.isna(row.get("captain_ev")):
        base_ev = float(row["captain_ev"])
        if triple_captain:
            base_ev *= 1.5   # captain_ev is already 2x; 1.5x gives 3x
    else:
        base_ev = xpts_captain_score(row, triple_captain)

    # fixture_trend decay: positive trend = harder fixtures = lower captain value
    trend   = float(row.get("fixture_trend", 0) or 0)
    penalty = max(0.0, trend * gw_offset * 0.3)

    return round(max(0.0, base_ev - penalty), 3)


def recommend_xi_multi_gw(my_team_enriched: pd.DataFrame,
                            bootstrap: dict,
                            fixtures_df: pd.DataFrame,
                            current_gw: int,
                            n_gws: int = 3,
                            triple_captain: bool = False) -> list:
    """
    Optimal XI recommendation for next N gameweeks.

    v4 fix: for each GW computes a GW-specific score using
    expected_pts * (6 - gw_difficulty) rather than reusing
    the GW+1 combined_score for all future GWs. Blank/double
    GW status per player is derived from gw{N}_opponent column.
    """
    results = []

    for gw_offset in range(1, n_gws + 1):
        gw = current_gw + gw_offset
        log.info(f"Computing optimal XI for GW{gw}...")

        gw_squad = my_team_enriched.copy()

        # Update blank status from Phase 2 stored per-GW columns
        gw_opp_col   = f"gw{gw}_opponent"
        gw_diff_col  = f"gw{gw}_difficulty"
        gw_home_col  = f"gw{gw}_home"

        if gw_opp_col in gw_squad.columns:
            gw_squad["is_blank_next_gw"] = (
                gw_squad[gw_opp_col].astype(str) == "BLANK"
            )
            # v4: compute GW-specific score as ILP objective
            gw_squad["_gw_score"] = gw_squad.apply(
                lambda r: _gw_specific_score(r, gw, current_gw), axis=1
            )
            # Temporarily override combined_score for this GW's ILP
            gw_squad["combined_score"] = gw_squad["_gw_score"]
            # v5 Item 4: inject GW-adjusted captain_ev with fixture_trend decay
            gw_squad["captain_ev"] = gw_squad.apply(
                lambda r: _gw_captain_score(r, gw, current_gw, triple_captain),
                axis=1
            )
        else:
            # Fallback: use fixture data from live fixtures_df
            gw_fixtures  = fixtures_df[fixtures_df["event"] == gw]
            gw_blank_map = {}
            gw_diff_map  = {}
            for team_id in gw_squad["team_id"].unique():
                team_fix = gw_fixtures[
                    (gw_fixtures["team_h"] == team_id) |
                    (gw_fixtures["team_a"] == team_id)
                ]
                if team_fix.empty:
                    gw_blank_map[team_id] = True
                    gw_diff_map[team_id]  = 6
                else:
                    r = team_fix.iloc[0]
                    gw_diff_map[team_id]  = float(
                        r["team_h_difficulty"] if r["team_h"] == team_id
                        else r["team_a_difficulty"]
                    )
                    gw_blank_map[team_id] = False
            gw_squad["is_blank_next_gw"] = (
                gw_squad["team_id"].map(gw_blank_map).fillna(False)
            )
            gw_squad["combined_score"] = gw_squad.apply(
                lambda r: 0.0 if r["is_blank_next_gw"] else
                _xpts_base(r) * (6.0 - gw_diff_map.get(r["team_id"], 3.0)) / 3.0,
                axis=1
            )
            # Also inject trend-adjusted captain_ev for the fallback path
            gw_squad["captain_ev"] = gw_squad.apply(
                lambda r: _gw_captain_score(r, gw, current_gw, triple_captain),
                axis=1
            )

        result = optimize_xi_ilp(gw_squad, triple_captain=triple_captain)
        if result:
            result["gw"] = gw
        results.append(result)

    return results


# ─────────────────────────────────────────
# 8. PIPELINE HELPER
# ─────────────────────────────────────────

def _load_or_train_models(history_df: pd.DataFrame,
                           refresh: bool) -> dict:
    """
    Load models from fpl_model.pkl if <12h old, else retrain.
    Avoids 30–60s retraining every run when Phase 1/2/3 already ran.
    """
    pkl_path = "fpl_model.pkl"
    if not refresh and os.path.exists(pkl_path):
        age_h = (
            pd.Timestamp.now() -
            pd.Timestamp.fromtimestamp(os.path.getmtime(pkl_path))
        ).total_seconds() / 3600
        if age_h < 12:
            try:
                with open(pkl_path, "rb") as f:
                    saved = pickle.load(f)
                models = saved.get("models", {})
                if models:
                    log.info(f"✅ Loaded models from {pkl_path} (age {age_h:.1f}h).")
                    return models
            except Exception as e:
                log.warning(f"Could not load {pkl_path}: {e} — retraining.")
    log.info("🤖 Training position-specific models...")
    models = train_models(history_df)
    with open(pkl_path, "wb") as f:
        pickle.dump({"models": models, "features": FEATURE_COLS}, f)
    log.info(f"💾 Models saved → {pkl_path}")
    return models


# ─────────────────────────────────────────
# 9. DISPLAY HELPERS
# ─────────────────────────────────────────

def print_formation_comparison(squad_df: pd.DataFrame) -> None:
    """Print all formations ranked by combined score. v4: shows xPts column."""
    results = score_all_formations(squad_df)
    has_xpts = any("xpts" in r for r in results)
    print(f"\n{'=' * 75}")
    print("  FORMATION COMPARISON")
    print(f"{'=' * 75}")
    if has_xpts:
        print(f"  {'Formation':<12}  {'xPts':>8}  {'Pred Pts':>10}  {'Combined':>10}")
    else:
        print(f"  {'Formation':<12}  {'Pred Pts':>10}  {'Combined':>10}")
    print(f"  {'-' * 44}")
    for i, r in enumerate(results):
        marker = "  <- Optimal" if i == 0 else ""
        if has_xpts:
            print(f"  {r['formation']:<12}  {r.get('xpts', r['pred_pts']):>8}  "
                  f"{r['pred_pts']:>10}  {r['combined']:>10}{marker}")
        else:
            print(f"  {r['formation']:<12}  {r['pred_pts']:>10}  {r['combined']:>10}{marker}")


def print_captain_mc(mc_results: list, triple_captain: bool = False) -> None:
    """
    Item 2 (v5): Display Monte Carlo captain analysis — win_prob + top-3.

    Shows the probability each player is the optimal captain choice across
    1,000 simulations sampled from pts_low/pts_high quantile distributions.
    """
    if not mc_results:
        return

    tc_note = "  [TRIPLE CAPTAIN — 3x]" if triple_captain else ""
    print(f"\n{'=' * 75}")
    print(f"  CAPTAIN ANALYSIS — Monte Carlo (1,000 simulations){tc_note}")
    print(f"  Simulates from pts_low–pts_high distribution to find optimal captain")
    print(f"{'=' * 75}")
    print(f"  {'Player':<28} {'Win%':>6} {'Cap EV':>8} {'vs others':>10} "
          f"{'Run':<14} {'Home'}")
    print(f"  {'-' * 75}")

    for i, r in enumerate(mc_results[:5]):
        rank_label = ["★ Captain", "  Vice Cap", "  3rd opt ", "  4th opt ", "  5th opt "][i]
        home_str   = "H" if r.get("is_home") else "A"
        dgw_str    = " (DGW)" if r.get("double_gws", 0) > 0 else ""
        gain_str   = f"{r['expected_captain_gain']:+.2f}"
        print(
            f"  {rank_label}  {str(r['player_name']):<28}"
            f"  {r['win_prob']*100:>5.1f}%"
            f"  {r['captain_ev']:>7.1f}"
            f"  {gain_str:>10}"
            f"  {r['fixture_run']:<14}"
            f"  {home_str}{dgw_str}"
        )

    top = mc_results[0]
    print(
        f"\n  Best captain by simulation: {top['player_name']}"
        f"  (wins captaincy decision {top['win_prob']*100:.1f}% of the time)"
    )
    print(f"  expected_captain_gain = average pts gained vs captaining someone else: "
          f"{top['expected_captain_gain']:+.2f}")


def print_starting_xi(result: dict, current_gw: int, rmse_map: dict) -> None:
    """Print starting XI with CI range and chip notes. v4: uses pts_low/pts_high CI."""
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
        # Show expected_pts if available, else predicted_pts
        pts   = float(row.get("expected_pts", row["predicted_pts"]))
        blank = "*" if row.get("is_blank_next_gw", False) else ""
        if row["player_id"] == cap["player_id"]:   tag = "(C)"
        elif row["player_id"] == vc["player_id"]:  tag = "(V)"
        else:                                        tag = ""
        return f"{name}{tag}{blank}({pts:.1f})"

    def print_row(players, label):
        print(f"    {label:<4}  " +
              "    ".join(fmt(r) for _, r in players.iterrows()))

    # v4: use pts_low/pts_high when available
    lo, hi, ci_label = compute_score_range(xi, rmse_map)

    method_note = f"  [{result.get('method', 'ILP')}]"
    bb_note     = "  [BENCH BOOST — all 15 score]" if result.get("bench_boost") else ""
    tc_note     = "  [TRIPLE CAPTAIN — 3x]" if result.get("triple_captain") else ""

    # xPts total
    xpts_total = round(
        xi["expected_pts"].sum() if "expected_pts" in xi.columns
        else xi["predicted_pts"].sum(), 2
    )

    print(f"\n{'=' * 75}")
    print(
        f"  OPTIMAL STARTING XI — GW{gw_label}  |  "
        f"Formation: {result['formation']}{method_note}{bb_note}{tc_note}"
    )
    print(
        f"  xPts: {xpts_total}"
        f"  |  Pred: {result['total_predicted_pts']}"
        f"  |  Range: {lo}–{hi} ({ci_label})"
    )
    print(f"  (C)=Captain  (V)=Vice Captain  *=Blank  xPts=rotation-adjusted")
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

    # Detail table — v4: show expected_pts, captain_ev
    has_cap_ev = "captain_ev" in xi.columns
    print(f"\n  {'Player':<28} {'Pos':<12} {'xPts':>6} {'Pred':>6} {'Cap EV':>7} "
          f"{'Run':<14} {'Diff':>5}")
    print(f"  {'-' * 78}")
    for _, row in xi.sort_values("combined_score", ascending=False).iterrows():
        tag   = " (C)" if row["player_id"] == cap["player_id"] else \
                " (V)" if row["player_id"] == vc["player_id"] else "    "
        blank = " *" if row.get("is_blank_next_gw", False) else "  "
        xpts  = round(_xpts_base(row) * (1 + float(row.get("roll3_threat", 0) or 0) / 100), 2)
        cap_ev_val = f"{float(row.get('captain_ev', xpts*2)):.1f}" if has_cap_ev else "—"
        print(
            f"  {str(row['player_name']) + tag:<28}"
            f"  {row['position']:<12}"
            f"  {xpts:>6.2f}"
            f"  {row['predicted_pts']:>6}"
            f"  {cap_ev_val:>7}"
            f"  {row.get('fixture_run_label', '?'):<14}"
            f"  {row.get('difficulty', '-'):>5}"
            f"{blank}"
        )

    # Captain summary
    tc_mult  = 3 if result.get("triple_captain") else 2
    dgw_note = " (DGW)" if cap.get("double_gws", 0) > 0 else ""
    cap_xpts = round(_xpts_base(cap), 2)
    cap_ev   = float(cap.get("captain_ev", cap_xpts * tc_mult))

    # VC reliability
    vc_pfull  = vc.get("p_plays_full")   # v5 fix: was cap.get (wrong variable)
    vc_rel    = float(vc_pfull) if vc_pfull is not None and not pd.isna(vc_pfull) \
                else float(vc.get("chance_of_playing", 100)) / 100.0
    vc_rel_pct = round(vc_rel * 100)

    print(
        f"\n  Captain:      {cap['player_name']}"
        f"  xPts:{cap_xpts}  Cap EV:{cap_ev:.1f}"
        f"  (×{tc_mult} = {round(cap_xpts * tc_mult, 2)}){dgw_note}"
    )
    print(
        f"  Vice Captain: {vc['player_name']}"
        f"  xPts:{round(_xpts_base(vc), 2)}"
        f"  Reliability:{vc_rel_pct}%  ← backup captain safety"
    )


def print_bench(result: dict) -> None:
    """Print bench with auto-sub notes. v4: shows expected_pts."""
    bench = result["bench"]
    if bench.empty:
        return

    print(f"\n{'=' * 75}")
    print(f"  BENCH ORDER  (Expected auto-sub contribution: {result['bench_auto_sub_score']} pts)")
    print(f"{'=' * 75}")
    print(f"  {'#':<3} {'Player':<28} {'Pos':<12} {'Price':>5} "
          f"{'xPts':>6} {'Pred':>6} {'Run':<14} {'Note'}")
    print(f"  {'-' * 80}")

    for i, (_, row) in enumerate(bench.iterrows(), 1):
        if row["position"] == "Goalkeeper":
            note = "Emergency GK"
        elif i == 1:
            note = "First sub ★"
        elif row.get("is_blank_next_gw", False):
            note = "Blank GW"
        else:
            note = ""
        xpts_val = round(_xpts_base(row), 2)
        print(
            f"  {i:<3} {str(row['player_name']):<28}"
            f"  {row['position']:<12}"
            f"  £{row['price']:>4.1f}"
            f"  {xpts_val:>6.2f}"
            f"  {row['predicted_pts']:>6}"
            f"  {row.get('fixture_run_label', '?'):<14}"
            f"  {note}"
        )


def flag_injury_risks(squad_df: pd.DataFrame, bootstrap: dict) -> None:
    """Flag players with less than 100% chance of playing."""
    players_raw = bootstrap["elements"]
    risk_map    = {
        p["id"]: p.get("chance_of_playing_next_round")
        for p in players_raw
    }
    risks = []
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


def print_multi_gw_xi(gw_results: list) -> None:
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
        changed   = " ← FORMATION CHANGE" if formation != prev_formation \
                    and prev_formation is not None else ""
        print(
            f"\n  GW{gw}: {formation}{changed}"
            f"  |  Pred: {pts} pts"
            f"  |  Captain: {cap}  VC: {vc}"
        )
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
                             current_gw: int, rmse_map: dict) -> None:
    """Print before/after XI comparison for top transfer. v5: shows bench delta."""
    print(f"\n{'=' * 75}")
    print(f"  TRANSFER IMPACT: OUT {player_out} -> IN {player_in}")
    print(f"{'=' * 75}")

    lo_b, hi_b, ci_b = compute_score_range(before["starting_xi"], rmse_map)
    lo_a, hi_a, ci_a = compute_score_range(after["starting_xi"],  rmse_map)

    bench_before = before.get("bench_auto_sub_score", 0.0)
    bench_after  = after.get("bench_auto_sub_score",  0.0)
    bench_delta  = round(bench_after - bench_before, 2)

    print(
        f"\n  BEFORE: Formation {before['formation']}"
        f"  |  Pred: {before['total_predicted_pts']} pts"
        f"  |  Range: {lo_b}–{hi_b}"
        f"  |  Captain: {before['captain']['player_name']}"
        f"  |  Bench sub EV: {bench_before}"
    )
    print(
        f"  AFTER:  Formation {after['formation']}"
        f"  |  Pred: {after['total_predicted_pts']} pts"
        f"  |  Range: {lo_a}–{hi_a}"
        f"  |  Captain: {after['captain']['player_name']}"
        f"  |  Bench sub EV: {bench_after}"
    )
    xi_delta    = round(after["total_predicted_pts"] - before["total_predicted_pts"], 2)
    bench_note  = f"  Bench sub EV: {bench_delta:+.2f}"
    print(f"\n  Net XI improvement:  {xi_delta:+.2f} predicted pts")
    print(f"  Bench EV change:     {bench_delta:+.2f} expected bench contribution")
    total_delta = round(xi_delta + bench_delta, 2)
    print(f"  Total expected gain: {total_delta:+.2f} pts  (XI + bench combined)")


# ─────────────────────────────────────────
# 10. FULL PHASE 4 PIPELINE
# ─────────────────────────────────────────

def run_phase4(team_id: int = TEAM_ID,
               max_players: int = None,
               refresh: bool = False):
    """
    Full Phase 4 v4 pipeline.

    v4: applies full Phase 1 v5 predictions (component blend,
    expected_pts, price predictions, cs_probability_map).
    All XI selection, captain, VC, and bench decisions now use
    rotation-risk-adjusted expected_pts rather than raw predicted_pts.
    """
    log.info("=" * 75)
    log.info("  FPL AI ASSISTANT — Phase 4: Starting XI Optimizer (v5)")
    log.info("=" * 75)

    # ── Fetch ──────────────────────────────────────────────────────
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

    # ── History + models (fix 2: load pkl if fresh) ────────────────
    log.info("Loading player history...")
    history_df = build_player_history_df(
        bootstrap, max_players=max_players, refresh=refresh
    )
    models   = _load_or_train_models(history_df, refresh)
    rmse_map = get_rmse_from_models(models)

    # ── Phase 1 v5 full prediction pipeline (fix 1) ───────────────
    log.info(f"Predicting GW{current_gw+1} scores...")
    pred_df = build_current_features(
        bootstrap, fixtures_df, history_df,
        models, current_gw, my_player_ids=my_player_ids
    )
    log.info("🧩 Component models...")
    component_models = train_component_models(history_df)
    pred_df          = predict_component_pts(component_models, pred_df)
    direct_w         = 1.0 - COMPONENT_BLEND_WEIGHT
    pred_df["predicted_pts"] = (
        direct_w * pred_df["predicted_pts"] +
        COMPONENT_BLEND_WEIGHT * pred_df["pts_from_components"]
    ).round(2)
    pred_df     = compute_expected_pts(pred_df)
    price_model = train_price_model(history_df)
    pred_df     = add_price_predictions(price_model, pred_df)

    # ── Phase 2 context (fix 10: pass cs_prob_map) ────────────────
    log.info("Building context maps...")
    custom_diff     = build_custom_difficulty(history_df, bootstrap)
    team_form_map   = build_team_form(history_df, bootstrap)
    opp_scoring_map = build_opponent_scoring_map(history_df)
    cs_prob_map     = build_cs_probability_map(history_df)
    chip_info       = build_chip_status(team_id, bootstrap, fixtures_df, current_gw)

    log.info(f"Building fixture run ({FIXTURE_LOOKAHEAD} GWs)...")
    fixture_run_df = build_fixture_run(
        bootstrap, fixtures_df, current_gw,
        custom_difficulty=custom_diff,
        gw_lookahead=FIXTURE_LOOKAHEAD,
    )
    enriched_df = build_player_fixture_scores(
        pred_df, fixture_run_df, current_gw,
        team_form_map, opp_scoring_map,
        FIXTURE_LOOKAHEAD,
        cs_probability_map=cs_prob_map,   # fix 10
    )

    my_team_enriched = enriched_df[enriched_df["player_id"].isin(my_player_ids)].copy()
    other_enriched   = enriched_df[~enriched_df["player_id"].isin(my_player_ids)].copy()

    # ── Chip detection ─────────────────────────────────────────────
    available_chips = chip_info.get("available_chips", [])
    triple_captain  = "Triple Captain" in available_chips
    bench_boost     = "Bench Boost" in available_chips

    if triple_captain:
        log.info("Triple Captain available — applying 3x captain scoring.")
        print("\n  ⭐ Triple Captain chip available — captain scored at 3x!")
    if bench_boost:
        log.info("Bench Boost available — will show Bench Boost mode.")
        print("\n  📈 Bench Boost chip available — all 15 players score!")

    # ── Squad validation ───────────────────────────────────────────
    violations = validate_squad(my_team_enriched)
    if violations:
        print("\n  Squad violations:")
        for v in violations: print(f"    - {v}")
    else:
        print("\n  Squad passes all FPL rules.")

    # ── Injury risks ───────────────────────────────────────────────
    flag_injury_risks(my_team_enriched, bootstrap)

    # ── Formation comparison ───────────────────────────────────────
    print_formation_comparison(my_team_enriched)

    # ── Optimal starting XI ────────────────────────────────────────
    result = optimize_xi_ilp(
        my_team_enriched,
        triple_captain=triple_captain,
        bench_boost=False,
    )

    if not result:
        log.error("Could not find a valid starting XI.")
        print("  Could not find a valid starting XI.")
        return enriched_df, my_team_enriched, None

    print_starting_xi(result, current_gw, rmse_map)

    # Item 2 (v5): Monte Carlo captain analysis — shows win_prob + top-3
    log.info("🎲 Running Monte Carlo captain analysis...")
    mc_results = run_monte_carlo_captain(my_team_enriched)
    print_captain_mc(mc_results, triple_captain=triple_captain)

    print_bench(result)

    # ── Bench Boost mode (if available) ───────────────────────────
    if bench_boost:
        bb_result = optimize_xi_ilp(
            my_team_enriched,
            triple_captain=triple_captain,
            bench_boost=True,
        )
        print(f"\n{'=' * 75}")
        print("  BENCH BOOST MODE — Optimal Captain for All 15 Players")
        print(f"{'=' * 75}")
        print_starting_xi(bb_result, current_gw, rmse_map)

    # ── GW-by-GW XI recommendation ─────────────────────────────────
    gw_results = recommend_xi_multi_gw(
        my_team_enriched, bootstrap,
        fixtures_df, current_gw, n_gws=3,
        triple_captain=triple_captain,
    )
    print_multi_gw_xi(gw_results)

    # ── ILP optimal transfers ──────────────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"  OPTIMAL 1-TRANSFER (ILP{'  — PuLP' if PULP_AVAILABLE else '  — Greedy'})")
    print(f"{'=' * 75}")
    ilp_result_1 = get_ilp_optimal_transfers(
        my_team_enriched, other_enriched, bank_balance, n_transfers=1
    )
    print_ilp_result(ilp_result_1, "1-Transfer")

    # ── Post-transfer XI preview ───────────────────────────────────
    if ilp_result_1.get("transfers"):
        t = ilp_result_1["transfers"][0]
        player_in_rows = other_enriched[
            other_enriched["player_id"] == t["in_id"]
        ]
        if not player_in_rows.empty:
            after_result = get_post_transfer_xi(
                my_team_enriched,
                t["out_id"],
                player_in_rows.iloc[0],
                triple_captain=triple_captain,
            )
            if after_result:
                print_post_transfer_xi(
                    result, after_result,
                    t["out_name"], t["in_name"],
                    current_gw, rmse_map,
                )

    print(f"\n{'=' * 75}")
    print(f"  OPTIMAL 2-TRANSFER (ILP{'  — PuLP' if PULP_AVAILABLE else '  — Greedy'})")
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
        top_n=3, precomputed_ilp=ilp_result_2,
    )
    print_double_transfers(valid_doubles, bank_balance)

    print(
        f"\n  Note: Bank shown (£{bank_balance:.1f}M) may differ "
        f"from FPL app. Always verify before confirming."
    )

    # ── Save ───────────────────────────────────────────────────────
    enriched_df.to_csv("fpl_predictions_phase4.csv", index=False)
    result["starting_xi"].to_csv("fpl_starting_xi.csv", index=False)
    log.info("Saved fpl_predictions_phase4.csv and fpl_starting_xi.csv")
    log.info("✅ Phase 4 v4 complete — ready for Phase 7 (LLM Analyst)")

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
        refresh=REFRESH,
    )
