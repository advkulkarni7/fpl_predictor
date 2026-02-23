"""
FPL AI Assistant — Phase 2: Fixture Run Analysis (v5)
=====================================================================
Changes over v4 (8 new improvements):

  🔴 HIGH IMPACT — ALGORITHMIC:
  1. EWMA decay in build_custom_difficulty — recent defensive performances
     weighted exponentially higher (span=5). A team keeping 5 clean sheets
     recently is rated much easier than a flat 10-GW average would show.

  2. Home/away split in build_custom_difficulty — each team now has separate
     home and away defensive difficulty ratings. Returns
     {team_id: {"home": float, "away": float}}. build_fixture_run picks the
     correct venue value: if team plays at home, opponent defends away —
     use opponent's away defensive difficulty (typically easier).

  3. Position-aware DGW bonus multiplier in compute_combined_score —
     flat 0.5× replaced by: GK=0.30, DEF=0.50, MID=0.70, FWD=0.70.

  🟡 MEDIUM IMPACT — FEATURE ENRICHMENT:
  4. Fixture difficulty trend — fixture_trend column: slope of difficulty
     across the lookahead window. Positive = getting harder, negative =
     getting easier. Useful for transfer timing decisions.

  5. Head-to-head clean sheet probability map — build_cs_probability_map()
     computes per (player_id, opponent_team_id) CS rate from history.
     Gives GK/DEF a cs_history_bonus in combined_score when they have a
     strong record keeping clean sheets vs the next opponent.

  6. Captain EV — captain_ev = 2*expected_pts + variance_bonus where
     variance_bonus = (pts_high - pts_low) * 0.15. High-ceiling players
     are better captains at same xPts. Captains ranked by captain_ev.

  🟢 ROBUSTNESS:
  7. BGW proportional fixture quality scaling — fixture_quality in
     compute_combined_score is scaled by (non_blank_gws / total_gws).
     A player with 2/5 non-blank GWs gets 40% fixture quality, not 100%.

  8. compare_players cross-squad search — accepts optional squad_df so
     players in your own team can still be found. Name similarity
     suggestions on no match.

Changes over v3 (preserved from v4):
  - Label mismatch fixed (emoji strings 🟢/🟡/🔴)
  - build_team_form uses match results (W/D/L), not FPL pts
  - run_phase2 applies full Phase 1 v5 pipeline
  - Wildcard half-season chip detection fixed
  - expected_pts as base for combined_score
  - Captain confidence intervals (pts_low/pts_high)
  - Transfer suggestions: expected_pts, price change, crowd signals
  - Transfer timing: rotation risk, price change signals
  - OPP_SCORING_BLEND configurable
  - Model pkl loaded if fresh, else retrains
  - Dead fixture_run_label removed

Run normally:  python fpl_phase2_fixtures.py
Force refresh: python fpl_phase2_fixtures.py --refresh
"""

import os
import sys
import logging
import pickle
import requests
import pandas as pd
import numpy as np

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

try:
    from config import (
        TEAM_ID, FIXTURE_LOOKAHEAD,
        FIXTURE_EASY_THRESHOLD, FIXTURE_MODERATE_THRESHOLD,
        CUSTOM_DIFFICULTY_BLEND, CUSTOM_DIFF_WINDOW,
        CUSTOM_DIFF_GOALS_WEIGHT, CUSTOM_DIFF_XGC_WEIGHT, CUSTOM_DIFF_CS_WEIGHT,
        COMBINED_NEXT_GW_WEIGHT, COMBINED_FIXTURE_WEIGHT, COMBINED_DGW_BONUS_WEIGHT,
    )
except ImportError:
    TEAM_ID                    = 9179961
    FIXTURE_LOOKAHEAD          = 5
    FIXTURE_EASY_THRESHOLD     = 2.8
    FIXTURE_MODERATE_THRESHOLD = 3.5
    CUSTOM_DIFFICULTY_BLEND    = 0.6
    CUSTOM_DIFF_WINDOW         = 10
    CUSTOM_DIFF_GOALS_WEIGHT   = 0.55
    CUSTOM_DIFF_XGC_WEIGHT     = 0.20
    CUSTOM_DIFF_CS_WEIGHT      = 0.25
    COMBINED_NEXT_GW_WEIGHT    = 0.5
    COMBINED_FIXTURE_WEIGHT    = 0.3
    COMBINED_DGW_BONUS_WEIGHT  = 0.2

try:
    from config import OPP_SCORING_BLEND
except ImportError:
    OPP_SCORING_BLEND = 0.30

_WILDCARD_SPLIT_GW = 20

# ── Item 3: position-aware DGW bonus multipliers ──────────────────
DGW_POSITION_MULTIPLIER = {
    "Goalkeeper": 0.30,
    "Defender":   0.50,
    "Midfielder": 0.70,
    "Forward":    0.70,
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
# 1. CUSTOM DIFFICULTY  (v5: EWMA + home/away)
# ─────────────────────────────────────────

def _ewma_weighted_mean(series: pd.Series,
                         rounds: pd.Series,
                         span: int = 5) -> float:
    """
    Exponentially weighted mean where higher round numbers carry more weight.
    weight_i = exp(-decay * (max_round - round_i)), decay = 2/(span+1).
    NaN values are dropped before computation.
    """
    df = pd.DataFrame({"val": series.values, "rnd": rounds.values})
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df["rnd"] = pd.to_numeric(df["rnd"], errors="coerce")
    df = df.dropna(subset=["val", "rnd"])
    if df.empty:
        return float("nan")
    max_r = float(df["rnd"].max())
    decay = 2.0 / (span + 1)
    w = np.exp(-decay * (max_r - df["rnd"]))
    return float((df["val"] * w).sum() / w.sum())



def build_custom_difficulty(history_df: pd.DataFrame,
                             bootstrap: dict,
                             window: int = CUSTOM_DIFF_WINDOW) -> dict:
    """
    Build venue-split defensive difficulty per team with EWMA recency weighting.

    Returns {team_id: {"home": float, "away": float}} (1.0–5.0 each).
    build_fixture_run picks the correct value per fixture venue:
      - team at home → opponent defends away → use opponent["away"] difficulty
      - team away    → opponent defends home → use opponent["home"] difficulty
    """
    teams_df        = pd.DataFrame(bootstrap["teams"])
    team_id_by_name = teams_df.set_index("name")["id"].to_dict()

    required = {"team_name", "round", "fixture", "goals_conceded", "was_home"}
    missing  = required - set(history_df.columns)
    if missing:
        log.warning(f"Cannot build custom difficulty — missing: {sorted(missing)}")
        return {}

    recent = history_df[
        history_df["round"] >= history_df["round"].max() - window
    ].copy()
    if recent.empty:
        return {}

    if "minutes" in recent.columns:
        recent = recent[recent["minutes"] > 0].copy()
        if recent.empty:
            return {}

    group_cols = ["team_name", "round", "fixture", "was_home"]
    agg_spec   = {"goals_conceded": "max"}
    if "expected_goals_conceded" in recent.columns:
        agg_spec["expected_goals_conceded"] = "max"

    team_match = (
        recent[group_cols + list(agg_spec.keys())]
        .groupby(group_cols, as_index=False)
        .agg(agg_spec)
    )

    def _minmax_s(s: pd.Series, invert: bool = False) -> pd.Series:
        lo, hi = float(s.min()), float(s.max())
        if hi - lo < 1e-9:
            return pd.Series(np.full(len(s), 0.5), index=s.index)
        out = (s - lo) / (hi - lo)
        return 1 - out if invert else out

    def _raw_stats(sub: pd.DataFrame) -> tuple:
        gc  = _ewma_weighted_mean(sub["goals_conceded"], sub["round"])
        cs  = _ewma_weighted_mean((sub["goals_conceded"] == 0).astype(float), sub["round"])
        xgc = (
            _ewma_weighted_mean(sub["expected_goals_conceded"], sub["round"])
            if "expected_goals_conceded" in sub.columns else gc
        )
        return gc, xgc, cs

    def _to_difficulty(stats_df: pd.DataFrame) -> pd.Series:
        gc_easy  = _minmax_s(stats_df["gc"].astype(float),      invert=False)
        xgc_easy = _minmax_s(stats_df["xgc"].astype(float),     invert=False)
        cs_hard  = _minmax_s(stats_df["cs_rate"].astype(float), invert=False)
        raw = (
            CUSTOM_DIFF_GOALS_WEIGHT * (1 - gc_easy) +
            CUSTOM_DIFF_XGC_WEIGHT   * (1 - xgc_easy) +
            CUSTOM_DIFF_CS_WEIGHT    * cs_hard
        )
        r_lo, r_hi = float(raw.min()), float(raw.max())
        if r_hi - r_lo < 0.05:
            ranked = pd.Series(raw.values, index=raw.index).rank(pct=True)
            return (1 + 4 * ranked).clip(1.0, 5.0).round(2)
        return (1 + 4 * ((raw - r_lo) / (r_hi - r_lo))).clip(1.0, 5.0).round(2)

    team_names = list(team_id_by_name.keys())
    home_rows: list = []
    away_rows: list = []

    for tn in team_names:
        for label, flag in [("home", 1), ("away", 0)]:
            sub = team_match[(team_match["team_name"] == tn) & (team_match["was_home"] == flag)]
            if len(sub) < 2:
                sub = team_match[team_match["team_name"] == tn]
            if sub.empty:
                gc, xgc, cs = np.nan, np.nan, np.nan
            else:
                gc, xgc, cs = _raw_stats(sub)
            rec = {"team_name": tn, "gc": gc, "xgc": xgc, "cs_rate": cs}
            (home_rows if label == "home" else away_rows).append(rec)

    home_df = pd.DataFrame(home_rows).set_index("team_name")
    away_df = pd.DataFrame(away_rows).set_index("team_name")
    for df in (home_df, away_df):
        for col in ["gc", "xgc", "cs_rate"]:
            med = float(df[col].median())
            df[col] = df[col].fillna(med if not np.isnan(med) else 1.0)

    home_diff = _to_difficulty(home_df)
    home_diff.index = home_df.index
    away_diff = _to_difficulty(away_df)
    away_diff.index = away_df.index

    custom_diff: dict = {}
    for tn in team_names:
        tid = team_id_by_name.get(tn)
        if tid is None:
            continue
        custom_diff[tid] = {
            "home": float(home_diff.get(tn, 3.0)),
            "away": float(away_diff.get(tn, 3.0)),
        }

    log.info(
        "Custom difficulty v5 (EWMA+venue-split) for %d teams — "
        "home avg=%.2f  away avg=%.2f",
        len(custom_diff),
        float(np.mean([v["home"] for v in custom_diff.values()])),
        float(np.mean([v["away"] for v in custom_diff.values()])),
    )
    return custom_diff


# ─────────────────────────────────────────
# 2. TEAM FORM  (match-result based, v4)
# ─────────────────────────────────────────

def build_team_form(history_df: pd.DataFrame,
                    bootstrap: dict,
                    window: int = 5) -> dict:
    """
    Recent form per team over last N GWs using MATCH RESULTS (W=3,D=1,L=0).
    Normalised by dividing by 3.0. {team_id -> form_score (0.0–1.0)}
    """
    teams_df    = pd.DataFrame(bootstrap["teams"])
    team_ids    = teams_df["id"].tolist()
    team_id_map = teams_df.set_index("name")["id"].to_dict()

    required = {"goals_scored", "goals_conceded", "team_name", "round", "fixture"}
    if not required.issubset(set(history_df.columns)):
        log.warning("build_team_form: missing columns, using neutral 0.5 fallback.")
        return {t: 0.5 for t in team_ids}

    recent = history_df[
        history_df["round"] >= history_df["round"].max() - window
    ].copy()
    if recent.empty:
        return {t: 0.5 for t in team_ids}

    match_df = (
        recent[["team_name", "round", "fixture", "goals_scored", "goals_conceded"]]
        .groupby(["team_name", "round", "fixture"], as_index=False)
        .agg({"goals_scored": "max", "goals_conceded": "max"})
    )

    def _result_pts(row) -> int:
        if row["goals_scored"] > row["goals_conceded"]: return 3
        if row["goals_scored"] == row["goals_conceded"]: return 1
        return 0

    match_df["result_pts"] = match_df.apply(_result_pts, axis=1)
    team_form_pts = match_df.groupby("team_name")["result_pts"].mean()

    form_map: dict = {}
    for team_name, avg_pts in team_form_pts.items():
        tid = team_id_map.get(team_name)
        if tid is not None:
            form_map[tid] = round(float(avg_pts) / 3.0, 3)

    for tid in team_ids:
        if tid not in form_map:
            form_map[tid] = 0.5

    log.info(
        "Team form (match-result): %d teams, avg=%.3f, min=%.3f, max=%.3f",
        len(form_map),
        float(np.mean(list(form_map.values()))),
        float(np.min(list(form_map.values()))),
        float(np.max(list(form_map.values()))),
    )
    return form_map


# ─────────────────────────────────────────
# 3. OPPONENT SCORING MAP
# ─────────────────────────────────────────

def build_opponent_scoring_map(history_df: pd.DataFrame) -> dict:
    """
    {(player_id, opponent_team_id) -> avg_points} from history.
    """
    if not {"opponent_team", "total_points", "player_id"}.issubset(set(history_df.columns)):
        return {}
    scoring_map = (
        history_df
        .groupby(["player_id", "opponent_team"])["total_points"]
        .mean().round(2).to_dict()
    )
    log.info(f"Opponent scoring map: {len(scoring_map)} pairs.")
    return scoring_map


# ─────────────────────────────────────────
# 4. H2H CLEAN SHEET PROBABILITY MAP  (new in v5)
# ─────────────────────────────────────────

def build_cs_probability_map(history_df: pd.DataFrame) -> dict:
    """
    Item 5: Per (player_id, opponent_team_id) clean sheet rate from history.
    Meaningful primarily for GK/DEF. Used in build_player_fixture_scores
    to apply a cs_history_bonus to combined_score.

    Returns {(player_id, opponent_team_id) -> cs_rate (0.0–1.0)}
    """
    required = {"player_id", "opponent_team", "clean_sheets", "minutes"}
    if not required.issubset(set(history_df.columns)):
        log.warning("build_cs_probability_map: missing columns, returning empty.")
        return {}

    df = history_df[history_df["minutes"] > 0].copy()
    cs_map = (
        df.groupby(["player_id", "opponent_team"])["clean_sheets"]
        .mean().round(3).to_dict()
    )
    log.info(f"CS probability map: {len(cs_map)} player/opponent pairs.")
    return cs_map


# ─────────────────────────────────────────
# 5. CHIP STATUS  (half-season wildcard fix, v4)
# ─────────────────────────────────────────

def build_chip_status(team_id: int,
                       bootstrap: dict,
                       fixtures_df: pd.DataFrame,
                       current_gw: int) -> dict:
    """
    Detect available chips and upcoming DGW opportunities.
    Wildcard detection is half-season aware (GW1–19 / GW20–38).
    """
    BASE_URL        = "https://fantasy.premierleague.com/api"
    available_chips = []

    try:
        r = requests.get(f"{BASE_URL}/entry/{team_id}/")
        r.raise_for_status()
        entry       = r.json()
        chips_played = entry.get("chips", []) or []

        chips_used: dict = {}
        for c in chips_played:
            if not isinstance(c, dict):
                continue
            name = c.get("name", "")
            gw   = c.get("event", 0) or 0
            chips_used.setdefault(name, []).append(gw)

        # Wildcard: half-season aware
        in_first_half     = current_gw < _WILDCARD_SPLIT_GW
        wc_used_this_half = any(
            (g < _WILDCARD_SPLIT_GW) == in_first_half
            for g in chips_used.get("wildcard", [])
        )
        if not wc_used_this_half:
            half_label = "first half" if in_first_half else "second half"
            available_chips.append(f"Wildcard ({half_label})")

        for key, label in {"freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}.items():
            if key not in chips_used:
                available_chips.append(label)

    except Exception as e:
        log.warning(f"Could not fetch chip status: {e}")

    gws     = range(current_gw + 1, current_gw + 1 + FIXTURE_LOOKAHEAD)
    dgw_gws = []
    for gw in gws:
        gw_fix = fixtures_df[fixtures_df["event"] == gw]
        counts = pd.concat([gw_fix["team_h"], gw_fix["team_a"]]).value_counts()
        n_dgw  = int((counts > 1).sum())
        if n_dgw > 0:
            dgw_gws.append({"gw": gw, "dgw_teams": n_dgw})

    fh_gw = max(dgw_gws, key=lambda x: x["dgw_teams"])["gw"] if dgw_gws else None

    return {
        "available_chips":         available_chips,
        "dgw_gws":                 dgw_gws,
        "free_hit_recommendation": fh_gw,
    }


# ─────────────────────────────────────────
# 6. FIXTURE MOMENTUM
# ─────────────────────────────────────────

def compute_fixture_momentum(gw_difficulties: list) -> float:
    """
    Exponentially weighted difficulty (GW+1 weighted most).
    Weights: 0.5^i for i=0,1,2,...  Lower = easier run.
    """
    if not gw_difficulties:
        return 3.0
    weights = [0.5 ** i for i in range(len(gw_difficulties))]
    total_w = sum(weights)
    return round(sum(d * w for d, w in zip(gw_difficulties, weights)) / total_w, 2)


# ─────────────────────────────────────────
# 7. FIXTURE TREND  (new in v5)
# ─────────────────────────────────────────

def compute_fixture_trend(gw_difficulties: list,
                           gw_numbers: list = None) -> float:
    """
    Item 4: Calendar-aware slope of difficulty across the lookahead window.
    Positive = fixtures get harder over time. Negative = getting easier.

    Uses actual GW numbers as the x-axis (not consecutive 0,1,2,...) so that
    blank GW gaps are correctly reflected in the slope magnitude.

    Example: difficulties [2.0, 2.5, 4.0, 4.5] at GWs [26, 27, 29, 30]
      Without fix: x = [0, 1, 2, 3]  — blank at GW28 invisible, slope underestimated
      With fix:    x = [26, 27, 29, 30] — gap of 2 after blank, slope correct

    Falls back to consecutive indices if gw_numbers not provided
    (backward compat with any external callers).
    Returns 0.0 if fewer than 2 non-blank GWs.
    """
    if len(gw_difficulties) < 2:
        return 0.0

    y = np.array(gw_difficulties, dtype=float)
    x = (
        np.array(gw_numbers, dtype=float)
        if gw_numbers is not None and len(gw_numbers) == len(gw_difficulties)
        else np.arange(len(gw_difficulties), dtype=float)
    )

    x_mean = x.mean()
    y_mean = y.mean()
    num    = float(((x - x_mean) * (y - y_mean)).sum())
    den    = float(((x - x_mean) ** 2).sum())
    if den < 1e-9:
        return 0.0
    return round(num / den, 3)


# ─────────────────────────────────────────
# 8. FIXTURE RUN BUILDER  (v5: venue-aware difficulty lookup)
# ─────────────────────────────────────────

def fixture_run_labels_dynamic(avg_difficulty_series: pd.Series) -> pd.Series:
    """
    Percentile-binned fixture run labels.
    Bottom 33% = 🟢 Easy, mid 33% = 🟡 Moderate, top 33% = 🔴 Tough.
    """
    s = avg_difficulty_series.astype(float)
    if s.empty:
        return pd.Series(dtype=object)

    q1 = float(s.quantile(0.333))
    q2 = float(s.quantile(0.666))

    def _lab(v: float) -> str:
        if v <= q1: return "🟢 Easy"
        if v <= q2: return "🟡 Moderate"
        return "🔴 Tough"

    labels = s.apply(_lab)
    counts = labels.value_counts().to_dict()
    log.info(
        "Run labels: q33=%.2f q66=%.2f | Easy=%d Moderate=%d Tough=%d",
        q1, q2,
        counts.get("🟢 Easy", 0), counts.get("🟡 Moderate", 0), counts.get("🔴 Tough", 0),
    )
    return labels


def build_fixture_run(bootstrap: dict,
                       fixtures_df: pd.DataFrame,
                       current_gw: int,
                       custom_difficulty: dict = None,
                       gw_lookahead: int = FIXTURE_LOOKAHEAD) -> pd.DataFrame:
    """
    Build fixture table (next N GWs) for every team.

    v5: custom_difficulty now returns {team_id: {"home": float, "away": float}}.
    Venue-appropriate difficulty is selected per fixture:
      - team at home (home=True) → opponent defends away → use opp["away"]
      - team away   (home=False) → opponent defends home → use opp["home"]
    Falls back to FPL's official difficulty if no custom value exists.
    """
    teams_df = pd.DataFrame(bootstrap["teams"])
    team_map = teams_df.set_index("id")["name"].to_dict()
    gws      = range(current_gw + 1, current_gw + 1 + gw_lookahead)
    rows: list = []

    for gw in gws:
        gw_fixtures      = fixtures_df[fixtures_df["event"] == gw]
        team_fixture_map: dict = {}

        for _, fix in gw_fixtures.iterrows():
            for team, opp, fpl_diff, home in [
                (fix["team_h"], fix["team_a"], fix["team_h_difficulty"], True),
                (fix["team_a"], fix["team_h"], fix["team_a_difficulty"], False),
            ]:
                if custom_difficulty:
                    opp_diff = custom_difficulty.get(opp)
                    if isinstance(opp_diff, dict):
                        # team at home → opp defends away; team away → opp defends home
                        venue_key  = "away" if home else "home"
                        custom_val = float(opp_diff.get(venue_key, fpl_diff))
                    elif opp_diff is not None:
                        custom_val = float(opp_diff)   # backward compat: flat value
                    else:
                        custom_val = float(fpl_diff)
                    diff = CUSTOM_DIFFICULTY_BLEND * custom_val + (1.0 - CUSTOM_DIFFICULTY_BLEND) * float(fpl_diff)
                else:
                    diff = float(fpl_diff)

                team_fixture_map.setdefault(team, []).append({
                    "team_id":    team,
                    "team_name":  team_map.get(team, "Unknown"),
                    "gw":         gw,
                    "opponent":   team_map.get(opp, "Unknown"),
                    "opponent_id": opp,
                    "difficulty": round(diff, 2),
                    "is_home":    int(home),
                    "is_blank":   False,
                    "is_double":  False,
                })

        for team_id, team_name in team_map.items():
            fixtures_this_gw = team_fixture_map.get(team_id, [])
            if not fixtures_this_gw:
                rows.append({
                    "team_id": team_id, "team_name": team_name, "gw": gw,
                    "opponent": "BLANK", "opponent_id": -1,
                    "difficulty": 6, "is_home": 0, "is_blank": True, "is_double": False,
                })
            elif len(fixtures_this_gw) >= 2:
                avg_diff = round(np.mean([f["difficulty"] for f in fixtures_this_gw]), 1)
                opps     = " & ".join(f["opponent"] for f in fixtures_this_gw)
                rows.append({
                    "team_id": team_id, "team_name": team_name, "gw": gw,
                    "opponent": opps, "opponent_id": fixtures_this_gw[0]["opponent_id"],
                    "difficulty": avg_diff, "is_home": fixtures_this_gw[0]["is_home"],
                    "is_blank": False, "is_double": True,
                })
            else:
                rows.append(fixtures_this_gw[0])

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 9. COMBINED SCORE  (v5: position DGW + BGW scaling)
# ─────────────────────────────────────────

def compute_combined_score(expected_pts: float,
                            avg_difficulty: float,
                            is_blank_next: bool,
                            double_gws: int,
                            is_home_next: bool,
                            position: str,
                            team_form: float,
                            momentum_score: float,
                            non_blank_gws: int = None,
                            total_gws: int = None) -> float:
    """
    Combined score incorporating all context factors.

    v5 changes:
    - Item 3: DGW bonus uses position-specific multiplier (DGW_POSITION_MULTIPLIER)
    - Item 7: fixture_quality scaled by non_blank_gws/total_gws when provided.
      A player with 2 games out of 5 should only get 40% fixture quality, not 100%.
    """
    home_multiplier = 1.0
    if is_home_next and position in ("Midfielder", "Forward"):
        home_multiplier = 1.10
    elif is_home_next and position == "Defender":
        home_multiplier = 1.05

    next_gw_score = 0.0 if is_blank_next else expected_pts * home_multiplier

    if position == "Defender":
        fixture_quality = (6 - momentum_score) * expected_pts * 1.2
    elif position == "Goalkeeper":
        fixture_quality = (6 - momentum_score) * expected_pts * 1.1
    else:
        fixture_quality = (6 - momentum_score) * expected_pts

    # Item 7: scale fixture_quality by fraction of non-blank GWs
    if non_blank_gws is not None and total_gws is not None and total_gws > 0:
        fixture_quality *= float(non_blank_gws) / float(total_gws)

    # Item 3: position-specific DGW multiplier
    dgw_mult  = DGW_POSITION_MULTIPLIER.get(position, 0.50)
    dgw_bonus = double_gws * expected_pts * dgw_mult

    form_modifier = 0.95 + (team_form * 0.10)

    raw_score = (
        next_gw_score   * COMBINED_NEXT_GW_WEIGHT +
        fixture_quality * COMBINED_FIXTURE_WEIGHT +
        dgw_bonus       * COMBINED_DGW_BONUS_WEIGHT
    ) * form_modifier

    return round(raw_score, 2)


# ─────────────────────────────────────────
# 10. ENRICH PREDICTIONS  (v5: trend + CS bonus + captain EV)
# ─────────────────────────────────────────

def build_player_fixture_scores(pred_df: pd.DataFrame,
                                 fixture_run_df: pd.DataFrame,
                                 current_gw: int,
                                 team_form_map: dict,
                                 opponent_scoring_map: dict,
                                 gw_lookahead: int = FIXTURE_LOOKAHEAD,
                                 cs_probability_map: dict = None) -> pd.DataFrame:
    """
    Attach fixture run data to every player row.

    v5 additions:
    - fixture_trend column (slope of difficulty across lookahead)
    - non_blank_gws passed to compute_combined_score for proportional scaling
    - cs_history_bonus for GK/DEF based on h2h CS probability map
    - captain_ev = 2*expected_pts + variance_bonus (pts_high-pts_low)*0.15
    """
    if cs_probability_map is None:
        cs_probability_map = {}

    gws     = list(range(current_gw + 1, current_gw + 1 + gw_lookahead))
    next_gw = current_gw + 1

    pivot_rows: list = []
    for team_id, group in fixture_run_df.groupby("team_id"):
        row              = {"team_id": team_id}
        blanks = doubles = 0
        diffs            = []
        gw_nums          = []   # actual GW numbers for non-blank fixtures (fixes trend calc)
        is_blank_next_gw = is_home_next_gw = False

        for gw in gws:
            gw_row = group[group["gw"] == gw]
            if gw_row.empty:
                row[f"gw{gw}_difficulty"] = 6
                row[f"gw{gw}_opponent"]   = "BLANK"
                row[f"gw{gw}_home"]       = 0
                blanks += 1
                if gw == next_gw:
                    is_blank_next_gw = True
            else:
                r = gw_row.iloc[0]
                row[f"gw{gw}_difficulty"]  = r["difficulty"]
                row[f"gw{gw}_opponent"]    = r["opponent"]
                row[f"gw{gw}_opponent_id"] = r.get("opponent_id", -1)
                row[f"gw{gw}_home"]        = r["is_home"]
                if r["is_blank"]:
                    blanks += 1
                    if gw == next_gw: is_blank_next_gw = True
                if r["is_double"]:
                    doubles += 1
                if not r["is_blank"]:
                    diffs.append(r["difficulty"])
                    gw_nums.append(gw)   # record actual GW position for trend calc
                if gw == next_gw:
                    is_home_next_gw = bool(r["is_home"])

        non_blank = len(diffs)
        row["avg_difficulty"]   = round(np.mean(diffs), 2) if diffs else 6.0
        row["blank_gws"]        = blanks
        row["double_gws"]       = doubles
        row["non_blank_gws"]    = non_blank
        row["is_blank_next_gw"] = is_blank_next_gw
        row["is_home_next_gw"]  = is_home_next_gw
        row["momentum_score"]   = compute_fixture_momentum(diffs)
        row["fixture_trend"]    = compute_fixture_trend(diffs, gw_nums)  # Item 4 (calendar-aware)
        pivot_rows.append(row)

    pivot_df = pd.DataFrame(pivot_rows)
    enriched = pred_df.merge(pivot_df, on="team_id", how="left")

    for col in ["avg_difficulty", "blank_gws", "double_gws", "non_blank_gws",
                "is_blank_next_gw", "is_home_next_gw", "momentum_score", "fixture_trend"]:
        enriched[col] = enriched[col].fillna(
            3.0   if col in ("avg_difficulty", "momentum_score") else
            0.0   if col == "fixture_trend" else
            False if "is_" in col else 0
        )

    enriched["blank_gws"]     = enriched["blank_gws"].astype(int)
    enriched["double_gws"]    = enriched["double_gws"].astype(int)
    enriched["non_blank_gws"] = enriched["non_blank_gws"].astype(int)

    enriched["fixture_run_label"] = fixture_run_labels_dynamic(enriched["avg_difficulty"])

    # Sanity check
    ad     = enriched["avg_difficulty"].astype(float)
    lcount = enriched["fixture_run_label"].value_counts(dropna=False)
    log.info("avg_difficulty: min=%.2f median=%.2f max=%.2f", float(ad.min()), float(ad.median()), float(ad.max()))
    log.info("labels: Easy=%d Moderate=%d Tough=%d (n=%d)",
             int(lcount.get("🟢 Easy",0)), int(lcount.get("🟡 Moderate",0)),
             int(lcount.get("🔴 Tough",0)), len(enriched))
    if not lcount.empty:
        dom = float(lcount.max()) / len(enriched)
        if dom > 0.90:
            log.warning("Labels imbalanced: %s is %.1f%%", str(lcount.idxmax()), dom * 100)

    # Opponent scoring bonus
    def get_opp_bonus(row) -> float:
        if row.get("is_blank_next_gw", False): return 0.0
        opp_id   = row.get(f"gw{next_gw}_opponent_id", row.get("opponent_team"))
        hist_avg = opponent_scoring_map.get((row["player_id"], opp_id))
        if hist_avg is None: return 0.0
        base = row.get("expected_pts", row["predicted_pts"])
        return round(hist_avg - float(base), 2)

    enriched["opp_scoring_bonus"] = enriched.apply(get_opp_bonus, axis=1)
    enriched["adj_predicted_pts"] = (
        enriched["predicted_pts"] + enriched["opp_scoring_bonus"] * OPP_SCORING_BLEND
    ).round(2)

    # Item 5: h2h CS bonus for GK/DEF
    def get_cs_bonus(row) -> float:
        if row.get("is_blank_next_gw", False): return 0.0
        if row["position"] not in ("Goalkeeper", "Defender"): return 0.0
        opp_id  = row.get(f"gw{next_gw}_opponent_id", row.get("opponent_team"))
        cs_rate = cs_probability_map.get((row["player_id"], opp_id))
        if cs_rate is None: return 0.0
        # Bonus proportional to historical CS rate above league average (≈0.30)
        return round(max(0.0, float(cs_rate) - 0.30) * 2.0, 2)

    enriched["cs_history_bonus"] = enriched.apply(get_cs_bonus, axis=1)

    # Base score for combined calculation
    def _base(row) -> float:
        xpts = row.get("expected_pts")
        if xpts is not None and not pd.isna(xpts) and xpts > 0:
            return float(xpts) + float(row["opp_scoring_bonus"]) * OPP_SCORING_BLEND \
                   + float(row["cs_history_bonus"])
        return float(row["adj_predicted_pts"])

    total_gws = gw_lookahead
    enriched["combined_score"] = enriched.apply(
        lambda r: compute_combined_score(
            expected_pts   = _base(r),
            avg_difficulty = r["avg_difficulty"],
            is_blank_next  = bool(r.get("is_blank_next_gw", False)),
            double_gws     = int(r.get("double_gws", 0)),
            is_home_next   = bool(r.get("is_home_next_gw", False)),
            position       = r["position"],
            team_form      = team_form_map.get(int(r.get("team_id", 0)), 0.5),
            momentum_score = float(r.get("momentum_score", 3.0)),
            non_blank_gws  = int(r.get("non_blank_gws", total_gws)),
            total_gws      = total_gws,
        ),
        axis=1,
    )

    # Value score
    enriched["value_score"] = (
        enriched["combined_score"] / enriched["price"].replace(0, np.nan)
    ).round(3)

    # Item 6: Captain EV = 2*xPts + variance_bonus
    has_intervals = "pts_low" in enriched.columns and "pts_high" in enriched.columns
    if has_intervals:
        xpts_col = enriched.get("expected_pts", enriched["predicted_pts"])
        variance_bonus = (
            enriched["pts_high"].astype(float) - enriched["pts_low"].astype(float)
        ) * 0.15
        enriched["captain_ev"] = (
            2.0 * xpts_col.astype(float) + variance_bonus
        ).round(2)
    else:
        xpts_col = enriched.get("expected_pts", enriched["predicted_pts"])
        enriched["captain_ev"] = (2.0 * xpts_col.astype(float)).round(2)

    return enriched


# ─────────────────────────────────────────
# 11. COMPARE PLAYERS  (v5: cross-squad + suggestions on no match)
# ─────────────────────────────────────────

def compare_players(player_a_name: str,
                     player_b_name: str,
                     enriched_df: pd.DataFrame,
                     current_gw: int,
                     gw_lookahead: int = FIXTURE_LOOKAHEAD,
                     squad_df: pd.DataFrame = None) -> None:
    """
    Side-by-side GW-by-GW comparison.

    v5: accepts optional squad_df so players in your own team can be found
    even if they were filtered from the transfer pool. If squad_df is
    provided it is searched first, then enriched_df. On no match, prints
    name similarity suggestions from the combined pool.
    """
    gws = list(range(current_gw + 1, current_gw + 1 + gw_lookahead))

    # Combine squad_df if provided for cross-squad search
    search_df = (
        pd.concat([squad_df, enriched_df], ignore_index=True)
        .drop_duplicates(subset=["player_id"]) if squad_df is not None
        else enriched_df
    )

    def find_player(name: str):
        mask = search_df["player_name"].str.contains(name, case=False, na=False)
        if mask.sum() > 0:
            return search_df[mask].iloc[0]
        # Suggest similar names
        all_names = search_df["player_name"].dropna().tolist()
        close = [n for n in all_names if name.lower() in n.lower()][:5]
        if close:
            print(f"  '{name}' not found. Did you mean: {', '.join(close)}?")
        else:
            print(f"  '{name}' not found. Check spelling.")
        return None

    pa = find_player(player_a_name)
    pb = find_player(player_b_name)
    if pa is None or pb is None:
        return

    print(f"\n{'=' * 80}")
    print(f"  PLAYER COMPARISON")
    print(f"{'=' * 80}")
    print(f"  {'Metric':<22}  {str(pa['player_name']):<28}  {str(pb['player_name']):<28}")
    print(f"  {'-' * 76}")

    for label, va, vb in [
        ("Position",       pa["position"],                          pb["position"]),
        ("Team",           pa["team_name"],                         pb["team_name"]),
        ("Price",          f"£{pa['price']:.1f}M",                 f"£{pb['price']:.1f}M"),
        ("Predicted Pts",  pa["predicted_pts"],                     pb["predicted_pts"]),
        ("Expected Pts",   pa.get("expected_pts", "—"),             pb.get("expected_pts", "—")),
        ("Floor (Q10)",    pa.get("pts_low",  "—"),                 pb.get("pts_low",  "—")),
        ("Ceiling (Q90)",  pa.get("pts_high", "—"),                 pb.get("pts_high", "—")),
        ("Captain EV",     pa.get("captain_ev", "—"),               pb.get("captain_ev", "—")),
        ("Rotation Risk",  f"{(1-pa.get('p_plays_full',1))*100:.0f}%",
                           f"{(1-pb.get('p_plays_full',1))*100:.0f}%"),
        ("Avg Difficulty", pa.get("avg_difficulty", 3),             pb.get("avg_difficulty", 3)),
        ("Fixture Trend",  pa.get("fixture_trend", 0),              pb.get("fixture_trend", 0)),
        ("Fixture Run",    pa.get("fixture_run_label", "?"),        pb.get("fixture_run_label", "?")),
        ("Momentum",       pa.get("momentum_score", 3),             pb.get("momentum_score", 3)),
        ("Blank GWs",      pa.get("blank_gws", 0),                  pb.get("blank_gws", 0)),
        ("Double GWs",     pa.get("double_gws", 0),                 pb.get("double_gws", 0)),
        ("CS History Bon", pa.get("cs_history_bonus", 0),           pb.get("cs_history_bonus", 0)),
        ("Price Change",   pa.get("predicted_price_change", 0),     pb.get("predicted_price_change", 0)),
        ("Combined Score", pa.get("combined_score", 0),             pb.get("combined_score", 0)),
        ("Value Score",    pa.get("value_score", 0),                pb.get("value_score", 0)),
    ]:
        print(f"  {label:<22}  {str(va):<28}  {str(vb):<28}")

    print(f"\n  GW-by-GW Fixture Breakdown:")
    print(f"  {'GW':<6}  {str(pa['player_name'])[:24]:<24}  {str(pb['player_name'])[:24]:<24}")
    print(f"  {'-' * 60}")
    for gw in gws:
        a_str = f"Diff:{pa.get(f'gw{gw}_difficulty','-')} {'H' if pa.get(f'gw{gw}_home',0) else 'A'} vs {str(pa.get(f'gw{gw}_opponent',''))[:14]}"
        b_str = f"Diff:{pb.get(f'gw{gw}_difficulty','-')} {'H' if pb.get(f'gw{gw}_home',0) else 'A'} vs {str(pb.get(f'gw{gw}_opponent',''))[:14]}"
        print(f"  GW{gw:<4}  {a_str:<28}  {b_str:<28}")

    winner = pa["player_name"] if pa.get("combined_score",0) >= pb.get("combined_score",0) else pb["player_name"]
    margin = round(abs(pa.get("combined_score",0) - pb.get("combined_score",0)), 2)
    print(f"\n  Verdict: {winner} is the better pick by combined score margin of {margin}")
    print(f"  (xPts + fixture run + form + home/away + rotation risk + CS history)")


# ─────────────────────────────────────────
# 12. TRANSFER TIMING
# ─────────────────────────────────────────

def transfer_timing_recommendation(player_in_name: str,
                                    player_out_name: str,
                                    enriched_df: pd.DataFrame,
                                    current_gw: int) -> None:
    """
    ACT NOW vs WAIT decision with 8 signals including rotation risk,
    price change, crowd momentum, fixture trend.
    """
    def find_player(name):
        mask = enriched_df["player_name"].str.contains(name, case=False, na=False)
        return enriched_df[mask].iloc[0] if mask.sum() > 0 else None

    p_in  = find_player(player_in_name)
    p_out = find_player(player_out_name)
    if p_in is None or p_out is None:
        print("  Could not find one or both players.")
        return

    print(f"\n{'=' * 80}")
    print(f"  TRANSFER TIMING: OUT {p_out['player_name']} → IN {p_in['player_name']}")
    print(f"{'=' * 80}")

    next_gw  = current_gw + 1
    next2_gw = current_gw + 2

    in_xpts  = float(p_in.get("expected_pts",  p_in.get("predicted_pts", 0)))
    out_xpts = float(p_out.get("expected_pts", p_out.get("predicted_pts", 0)))
    immediate_gain  = round(in_xpts - out_xpts, 2)
    in_blank_now    = bool(p_in.get("is_blank_next_gw", False))
    in_dgw_next     = "&" in str(p_in.get(f"gw{next2_gw}_opponent", ""))
    in_diff_now     = p_in.get(f"gw{next_gw}_difficulty", 3)
    in_diff_next2   = p_in.get(f"gw{next2_gw}_difficulty", 3)
    out_diff_now    = p_out.get(f"gw{next_gw}_difficulty", 3)
    in_price_chg    = float(p_in.get("predicted_price_change",  0))
    out_price_chg   = float(p_out.get("predicted_price_change", 0))
    out_rot_risk    = float(1 - p_out.get("p_plays_full", 1.0))
    in_net_xfer     = int(p_in.get("transfers_in_event",  0)) - int(p_in.get("transfers_out_event",  0))
    out_net_xfer    = int(p_out.get("transfers_in_event", 0)) - int(p_out.get("transfers_out_event", 0))
    in_trend        = float(p_in.get("fixture_trend", 0))

    reasons: list = []
    recommendation  = "ACT NOW"

    if in_blank_now:
        recommendation = "WAIT"
        reasons.append(f"{p_in['player_name']} has a BLANK GW{next_gw} — wait one week.")
    if float(out_diff_now) <= 2 and not in_blank_now:
        recommendation = "WAIT"
        reasons.append(f"{p_out['player_name']} has easy fixture (diff {out_diff_now}) GW{next_gw}.")
    if out_rot_risk > 0.35 and not in_blank_now:
        recommendation = "ACT NOW"
        reasons.append(f"{p_out['player_name']} has {out_rot_risk*100:.0f}% rotation risk — act sooner.")
    if in_dgw_next and not in_blank_now:
        recommendation = "ACT NOW"
        reasons.append(f"{p_in['player_name']} has DGW in GW{next2_gw} — capture both fixtures.")
    if immediate_gain < 0 and not in_dgw_next:
        recommendation = "WAIT"
        reasons.append(f"Immediate xPts gain {immediate_gain:+.2f} — {p_out['player_name']} expected to outscore.")
    if out_price_chg < -0.05:
        recommendation = "ACT NOW"
        reasons.append(f"{p_out['player_name']} price predicted to fall {out_price_chg:+.1f}M — sell before drop.")
    if in_price_chg > 0.05:
        reasons.append(f"{p_in['player_name']} price rising {in_price_chg:+.1f}M — buy sooner.")
    if in_net_xfer > 50000:
        reasons.append(f"{in_net_xfer:,} net managers buying {p_in['player_name']} this GW.")
    if out_net_xfer < -50000:
        reasons.append(f"{abs(out_net_xfer):,} net managers selling {p_out['player_name']}.")
    if in_trend < -0.2:
        reasons.append(f"{p_in['player_name']} fixtures get easier over run (trend={in_trend:+.2f}).")
    elif in_trend > 0.2:
        reasons.append(f"Warning: {p_in['player_name']} fixtures get harder over run (trend={in_trend:+.2f}).")

    mark = "✅" if recommendation == "ACT NOW" else "⏳"
    print(f"\n  Recommendation: {mark} {recommendation}")
    print(f"\n  Reasons:")
    if reasons:
        for r in reasons: print(f"    - {r}")
    else:
        print(f"    - Immediate gain {immediate_gain:+.2f} xPts is worthwhile now.")

    print(f"\n  Summary:")
    print(f"    Immediate xPts gain GW{next_gw}:  {immediate_gain:+.2f}")
    print(f"    In-player GW{next_gw}: diff {in_diff_now}  GW{next2_gw}: diff {in_diff_next2}{'  (DGW)' if in_dgw_next else ''}")
    print(f"    Fixture trend (IN):  {in_trend:+.2f}  (negative = gets easier)")
    print(f"    5GW combined gain:   {p_in.get('combined_score',0) - p_out.get('combined_score',0):+.2f}")
    print(f"    Price:  OUT {out_price_chg:+.1f}M  IN {in_price_chg:+.1f}M")


# ─────────────────────────────────────────
# 13. DISPLAY HELPERS
# ─────────────────────────────────────────

def print_fixture_run_table(enriched_df: pd.DataFrame,
                             current_gw: int,
                             gw_lookahead: int = FIXTURE_LOOKAHEAD,
                             title: str = "FIXTURE RUN") -> None:
    """Print fixture run table with xPts, trend, and value columns."""
    gws      = list(range(current_gw + 1, current_gw + 1 + gw_lookahead))
    has_xpts = "expected_pts" in enriched_df.columns

    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    gw_headers = "   ".join(f"GW{gw}" for gw in gws)
    if has_xpts:
        print(f"{'Player':<28} {'Pos':<4} {'£':<5} {'Pred':>5} {'xPts':>5} {'Trnd':>5} {'Val':>5} {'Run':>14} {'B':>2} {'D':>2}   {gw_headers}")
    else:
        print(f"{'Player':<28} {'Pos':<4} {'£':<5} {'Pred':>5} {'Trnd':>5} {'Val':>5} {'Run':>14} {'B':>2} {'D':>2}   {gw_headers}")
    print("-" * 100)

    for _, row in enriched_df.sort_values("combined_score", ascending=False).iterrows():
        gw_labels = []
        for gw in gws:
            diff = int(row.get(f"gw{gw}_difficulty", 6))
            opp  = str(row.get(f"gw{gw}_opponent", ""))
            home = row.get(f"gw{gw}_home", 0)
            if opp == "BLANK":
                gw_labels.append(" B ")
            elif "&" in opp:
                gw_labels.append(f"{diff}D")
            else:
                gw_labels.append(f"{diff}{'h' if home else 'a'}")

        gw_str       = "   ".join(gw_labels)
        blank_marker = "*" if row.get("is_blank_next_gw") else " "
        val          = row.get("value_score", 0)
        trend        = float(row.get("fixture_trend", 0))
        trend_str    = f"{trend:+.1f}"

        if has_xpts:
            xpts = float(row.get("expected_pts", row["predicted_pts"]))
            print(
                f"{str(row['player_name']):<28}{str(row['position']):<4} "
                f"£{row['price']:<4.1f} {row['predicted_pts']:>5}{blank_marker}"
                f"{xpts:>5.2f} {trend_str:>5} {val:>5.2f} "
                f"{str(row['fixture_run_label']):>14} "
                f"{int(row.get('blank_gws',0)):>2} {int(row.get('double_gws',0)):>2}   {gw_str}"
            )
        else:
            print(
                f"{str(row['player_name']):<28}{str(row['position']):<4} "
                f"£{row['price']:<4.1f} {row['predicted_pts']:>5}{blank_marker}"
                f"{trend_str:>5} {val:>5.2f} "
                f"{str(row['fixture_run_label']):>14} "
                f"{int(row.get('blank_gws',0)):>2} {int(row.get('double_gws',0)):>2}   {gw_str}"
            )

    print("\n  Legend: B=Blank  D=Double  h=Home  a=Away  *=Blank next GW  "
          "Trnd=Fixture trend (+hard/-easy)  xPts=Rotation-adj pts  Val=Score/£M")


# ─────────────────────────────────────────
# 14. CHIP STATUS DISPLAY
# ─────────────────────────────────────────

def print_chip_status(chip_info: dict) -> None:
    """Display chip availability and DGW opportunities."""
    print(f"\n{'=' * 80}")
    print("  CHIP STATUS & DGW OPPORTUNITIES")
    print(f"{'=' * 80}")
    chips = chip_info.get("available_chips", [])
    print(f"\n  Available chips: {', '.join(chips)}" if chips else "\n  No chips available.")
    dgw_gws = chip_info.get("dgw_gws", [])
    if dgw_gws:
        print(f"\n  Upcoming Double Gameweeks:")
        for d in dgw_gws:
            print(f"    GW{d['gw']}: {d['dgw_teams']} teams with double fixtures")
        fh_gw = chip_info.get("free_hit_recommendation")
        if fh_gw and "Free Hit" in chips:
            print(f"\n  💡 Free Hit Recommendation: Use in GW{fh_gw} (most DGW teams)")
        if any("Wildcard" in c for c in chips) and dgw_gws:
            print(f"  💡 Wildcard Tip: Activate before a DGW to build DGW-heavy squad.")
    else:
        print(f"\n  No double gameweeks detected in next {FIXTURE_LOOKAHEAD} GWs.")


# ─────────────────────────────────────────
# 15. TRANSFER SUGGESTIONS
# ─────────────────────────────────────────

def show_transfer_suggestions_phase2(my_team_enriched: pd.DataFrame,
                                      other_enriched: pd.DataFrame,
                                      bank_balance: float,
                                      enriched_df: pd.DataFrame,
                                      current_gw: int) -> None:
    """
    Transfer suggestions ranked by 5GW combined_score gain.
    Shows expected_pts gain, price change, crowd signal, fixture trend.
    """
    has_xpts      = "expected_pts"           in other_enriched.columns
    has_price_chg = "predicted_price_change" in other_enriched.columns
    has_transfers = "transfers_in_event"     in other_enriched.columns

    next_col = "expected_pts" if has_xpts else "predicted_pts"

    def compute_suggestions(budget: float) -> pd.DataFrame:
        suggestions = []
        for _, my_row in my_team_enriched.iterrows():
            same_pos = other_enriched[other_enriched["position"] == my_row["position"]].copy()
            same_pos["gain"]      = (same_pos["combined_score"] - my_row["combined_score"]).round(2)
            same_pos["next_gain"] = (
                same_pos[next_col].astype(float) - float(my_row.get(next_col, my_row["predicted_pts"]))
            ).round(2)
            same_pos["cost_diff"] = (same_pos["price"] - my_row["price"]).round(1)
            same_pos["replace"]   = my_row["player_name"]
            same_pos["budget_ok"] = same_pos["cost_diff"] <= budget
            suggestions.append(same_pos)
        return pd.concat(suggestions).sort_values("gain", ascending=False)

    def _fmt(r) -> str:
        blank_tag = " ⚠️ BLANK" if r.get("is_blank_next_gw") else ""
        dgw_tag   = " 🔄 DGW"  if r.get("double_gws", 0) > 0 else ""
        pchg      = float(r.get("predicted_price_change", 0))
        price_tag = " 📈" if pchg > 0.05 else " 📉" if pchg < -0.05 else ""
        trend     = float(r.get("fixture_trend", 0))
        trend_tag = f"  trend:{trend:+.1f}" if abs(trend) > 0.1 else ""
        xpts_str  = f"  xPts:{r['next_gain']:+.2f}" if has_xpts else f"  Next:{r['next_gain']:+.2f}"
        crowd_str = ""
        if has_transfers:
            net = int(r.get("transfers_in_event", 0)) - int(r.get("transfers_out_event", 0))
            if abs(net) > 20000:
                crowd_str = f"  👥{net:+,}"
        return (
            f"  OUT:{str(r['replace']):24s} → IN:{str(r['player_name']):24s}"
            f"  [{r['position']:3s}]  5GW:+{r['gain']:.2f}{xpts_str}"
            f"  Run:{r['fixture_run_label']}  Val:{r.get('value_score',0):.2f}"
            f"  Cost:{r['cost_diff']:+.1f}M{trend_tag}{dgw_tag}{blank_tag}{price_tag}{crowd_str}"
        )

    def print_suggestions(sug_df: pd.DataFrame, budget: float) -> pd.DataFrame:
        print(f"\n  💰 Best Within-Budget Transfers (Bank: £{budget:.1f}M):")
        budget_top = (
            sug_df[sug_df["budget_ok"] & (sug_df["gain"] > 0)]
            .drop_duplicates("player_name").head(5)
        )
        if budget_top.empty:
            print("  No affordable upgrades found.")
        else:
            for _, r in budget_top.iterrows(): print(_fmt(r))
        print(f"\n  💸 Best Transfers Regardless of Budget:")
        for _, r in (sug_df[sug_df["gain"] > 0].drop_duplicates("player_name").head(5).iterrows()):
            print(_fmt(r))
        return budget_top

    sug_df     = compute_suggestions(bank_balance)
    budget_top = print_suggestions(sug_df, bank_balance)

    if not budget_top.empty:
        top = budget_top.iloc[0]
        print(f"\n  Auto-checking transfer timing for top suggestion...")
        transfer_timing_recommendation(top["player_name"], top["replace"], enriched_df, current_gw)

    if len(budget_top) < 3:
        print(f"\n  ⚠️  Note: Bank (£{bank_balance:.1f}M) may differ from app.")
        user_input = input("   Enter actual bank from FPL app (or Enter to skip): £").strip()
        if user_input:
            try:
                nb = float(user_input)
                if nb > bank_balance:
                    log.info(f"Budget overridden to £{nb:.1f}M")
                    print_suggestions(compute_suggestions(nb), nb)
                else:
                    print("  Budget not higher, skipping.")
            except ValueError:
                print("  Invalid input.")
    else:
        print(f"\n  ⚠️  Bank (£{bank_balance:.1f}M) may differ from app. Double-check before confirming.")


# ─────────────────────────────────────────
# 16. PIPELINE HELPERS
# ─────────────────────────────────────────

def _load_or_train_models(history_df: pd.DataFrame,
                           current_gw: int,
                           refresh: bool) -> dict:
    """Load from fpl_model.pkl if <12h old, else retrain."""
    pkl_path = "fpl_model.pkl"
    if not refresh and os.path.exists(pkl_path):
        age_h = (
            pd.Timestamp.now() - pd.Timestamp.fromtimestamp(os.path.getmtime(pkl_path))
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
# 17. FULL PHASE 2 PIPELINE
# ─────────────────────────────────────────

def run_phase2(team_id: int = TEAM_ID,
               max_players: int = None,
               refresh: bool = False):
    """Full Phase 2 v5 pipeline."""
    log.info("=" * 80)
    log.info("  FPL AI ASSISTANT — Phase 2: Fixture Run Analysis (v5)")
    log.info("=" * 80)

    log.info("Fetching bootstrap & fixtures...")
    bootstrap   = fetch_bootstrap()
    fixtures_df = fetch_fixtures()
    current_gw  = fetch_current_gw(bootstrap)
    log.info(f"GW{current_gw} complete → analysing GW{current_gw+1}–GW{current_gw+FIXTURE_LOOKAHEAD}")

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

    log.info("Loading player history...")
    history_df = build_player_history_df(bootstrap, max_players=max_players, refresh=refresh)

    models = _load_or_train_models(history_df, current_gw, refresh)

    # Full Phase 1 v5 prediction pipeline
    log.info(f"Predicting GW{current_gw+1}...")
    pred_df = build_current_features(bootstrap, fixtures_df, history_df, models, current_gw, my_player_ids=my_player_ids)

    log.info("🧩 Component models...")
    component_models = train_component_models(history_df)
    pred_df = predict_component_pts(component_models, pred_df)
    direct_w = 1.0 - COMPONENT_BLEND_WEIGHT
    pred_df["predicted_pts"] = (direct_w * pred_df["predicted_pts"] + COMPONENT_BLEND_WEIGHT * pred_df["pts_from_components"]).round(2)
    pred_df = compute_expected_pts(pred_df)

    log.info("💰 Price model...")
    price_model = train_price_model(history_df)
    pred_df     = add_price_predictions(price_model, pred_df)

    log.info("Building context maps...")
    custom_diff     = build_custom_difficulty(history_df, bootstrap)
    team_form_map   = build_team_form(history_df, bootstrap)
    opp_scoring_map = build_opponent_scoring_map(history_df)
    cs_prob_map     = build_cs_probability_map(history_df)
    chip_info       = build_chip_status(team_id, bootstrap, fixtures_df, current_gw)

    log.info(f"Building fixture run ({FIXTURE_LOOKAHEAD} GWs)...")
    fixture_run_df = build_fixture_run(bootstrap, fixtures_df, current_gw, custom_difficulty=custom_diff, gw_lookahead=FIXTURE_LOOKAHEAD)
    enriched_df    = build_player_fixture_scores(
        pred_df, fixture_run_df, current_gw,
        team_form_map, opp_scoring_map, FIXTURE_LOOKAHEAD,
        cs_probability_map=cs_prob_map,
    )

    my_team_enriched = enriched_df[enriched_df["player_id"].isin(my_player_ids)].copy()
    other_enriched   = enriched_df[~enriched_df["player_id"].isin(my_player_ids)].copy()

    print_chip_status(chip_info)

    print_fixture_run_table(
        my_team_enriched, current_gw,
        title=f"YOUR SQUAD — GW{current_gw+1}–GW{current_gw+FIXTURE_LOOKAHEAD} Fixture Run",
    )

    # Captain recommendation — v5: ranked by captain_ev
    print(f"\n{'=' * 80}")
    print("  CAPTAIN RECOMMENDATION (Captain EV — xPts + Ceiling Bonus)")
    print(f"{'=' * 80}")

    captainable   = my_team_enriched[~my_team_enriched["is_blank_next_gw"]]
    if captainable.empty: captainable = my_team_enriched

    cap_sort      = "captain_ev" if "captain_ev" in captainable.columns else "combined_score"
    has_intervals = "pts_low" in captainable.columns and "pts_high" in captainable.columns

    for i, (_, row) in enumerate(captainable.nlargest(3, cap_sort).iterrows()):
        if i >= 3: break
        label     = ["Captain     ", "Vice Captain", "3rd Option  "][i]
        xpts      = float(row.get("expected_pts", row["predicted_pts"]))
        cap_ev    = float(row.get("captain_ev", xpts * 2))
        dgw_note  = " (DGW)" if row.get("double_gws", 0) > 0 else ""
        home_note = " (H)" if row.get("is_home_next_gw") else " (A)"
        score_str = (
            f"  xPts:{xpts:.2f} | Floor:{float(row.get('pts_low',0)):.1f} | "
            f"Ceiling:{float(row.get('pts_high',0)):.1f} | Cap EV:{cap_ev:.1f}"
            if has_intervals
            else f"  xPts:{xpts:.2f} | Cap EV:{cap_ev:.1f}"
        )
        print(
            f"  {label}: {str(row['player_name']):28s}{score_str}"
            f"  Run:{row['fixture_run_label']}  Score:{row['combined_score']}{home_note}{dgw_note}"
        )

    print(f"\n{'=' * 80}")
    print("  TRANSFER SUGGESTIONS (5-GW Outlook)")
    print(f"{'=' * 80}")
    show_transfer_suggestions_phase2(my_team_enriched, other_enriched, bank_balance, enriched_df, current_gw)

    print(f"\n{'=' * 80}")
    print("  TOP PLAYERS TO TARGET BY POSITION")
    print(f"{'=' * 80}")

    base_cols = ["player_name","team_name","price","predicted_pts","fixture_run_label",
                 "blank_gws","double_gws","combined_score","value_score","is_blank_next_gw"]
    extra     = [c for c in ["expected_pts","pts_low","pts_high","captain_ev","fixture_trend"]
                 if c in other_enriched.columns]

    for pos in ["Goalkeeper","Defender","Midfielder","Forward"]:
        pos_df = other_enriched[other_enriched["position"] == pos]
        print(f"\n  Top 5 {pos}s — by Combined Score:")
        sc = [c for c in base_cols + extra if c in pos_df.columns]
        top = pos_df.nlargest(5, "combined_score")[sc].copy()
        top["next_gw"] = top["is_blank_next_gw"].apply(lambda x: "BLANK" if x else "plays")
        print(top.drop(columns=["is_blank_next_gw"]).to_string(index=False))

        print(f"\n  Top 5 {pos}s — by Value (Score/£M):")
        vc = [c for c in ["player_name","team_name","price","predicted_pts","expected_pts",
                           "fixture_run_label","fixture_trend","combined_score","value_score"]
              if c in pos_df.columns]
        print(pos_df.nlargest(5, "value_score")[vc].to_string(index=False))

    enriched_df.to_csv("fpl_predictions_phase2.csv", index=False)
    log.info("Enriched predictions saved → fpl_predictions_phase2.csv")
    log.info("✅ Phase 2 v5 complete — ready for Phase 3")

    return enriched_df, my_team_enriched


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    REFRESH = "--refresh" in sys.argv
    if REFRESH:
        log.info("--refresh flag detected.")

    enriched_df, my_team = run_phase2(team_id=TEAM_ID, max_players=None, refresh=REFRESH)

    bootstrap_final  = fetch_bootstrap()
    current_gw_final = fetch_current_gw(bootstrap_final)

    print(f"\n{'=' * 80}")
    print("  PLAYER COMPARISON (Optional) — searches both your squad and transfer pool")
    print(f"{'=' * 80}")
    print("  Compare any two players side by side. Press Enter to skip.\n")

    a = input("  Enter first player name (or Enter to skip): ").strip()
    if a:
        b = input("  Enter second player name: ").strip()
        if b:
            compare_players(a, b, enriched_df, current_gw_final, squad_df=my_team)
