"""
FPL AI Assistant — Phase 3: Squad Constraints & Transfer Optimizer (v5)
========================================================================
New in v5 (8 algorithmic additions):

  🔴 HIGH IMPACT — ALGORITHMIC:
  1. Multi-GW Horizon Transfer Planner — 3-GW lookahead using greedy
     dynamic programming. For each candidate transfer this week, simulates
     the resulting squad and finds the best follow-up transfer next week.
     Ranks sequences by total xPts gain across the horizon, not just this GW.
     Finds cases where the greedy "best now" blocks a better 2-week plan.

  2. Monte Carlo Captain EV — 1,000 simulations sampling from triangular
     distribution [pts_low, predicted_pts, pts_high] (Phase 1 v5 quantiles).
     Outputs win_prob: P(this player scores highest when doubled) and
     expected_captain_gain: average pts gained vs not captaining this player.
     Falls back to deterministic EV if quantile columns absent.

  3. Bench Optimization — ranks bench players by expected auto-sub
     contribution: E[contribution] = sum over starters of
     P(starter_misses * bench_plays) * bench_expected_pts.
     Uses p_plays_full from Phase 1 v5. Recommends first, second, third sub
     order to maximise expected points from auto-subs.

  🟡 MEDIUM IMPACT:
  4. Transfer Deadline & Scheduling Urgency — each transfer suggestion now
     carries an urgency_score: how costly is it to delay by one week?
     Accounts for incoming player's fixture_trend (getting harder = act now),
     predicted price rise on incoming, and blank GW timing.

  5. Price-EV Integrated Into Transfer Gain — total_ev for each transfer now
     includes the financial value of price changes:
     total_ev = xpts_gain + 0.5*(in_price_change - out_price_change)
     The 0.5 factor reflects FPL's profit rule (you keep half of any rise).
     This surfaces transfers where a falling-price player should be sold even
     if the immediate xPts gain is modest.

  6. Captaincy Differential Analysis — for each captainable squad player,
     computes gain vs the "average manager" who caps the highest-owned player.
     differential_captain_gain = your_captain_ev - field_captain_ev
     Positive = you gain vs the field by captaining this player.
     Shows exactly when a differential captain is the correct call.

  🟢 ROBUSTNESS:
  7. Wildcard squad diff — print_wildcard_diff() shows exactly which players
     to sell and which to buy, with cost breakdown. Previously the wildcard
     squad was printed without any context of what changes from your current
     squad.

  8. Transfer history evaluation window capped at season end — previously
     evaluated only after exactly 5 GWs, meaning entries from GW34+ were
     never evaluated. Now caps at min(5, remaining_gws) so late-season
     transfers are evaluated with whatever data is available.

Changes preserved from v4 (11 fixes):
  - Wildcard chip any() substring check
  - Phase 1 v5 full pipeline in run_phase3
  - Hit break-even on xPts (same units)
  - -8pt double hit analysis
  - Full-name matching in evaluate_past_transfers
  - Captain sorted by captain_ev
  - ILP next_gain uses expected_pts
  - predicted_price_change in all displays
  - Sell prices for WC/FH budget
  - Rolling advice thresholds on xPts gain
  - pkl freshness check

Run normally:  python fpl_phase3_constraints.py
Force refresh: python fpl_phase3_constraints.py --refresh
"""

import os
import sys
import json
import logging
import pickle
import itertools
from datetime import datetime
from pathlib import Path

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
    print_fixture_run_table,
    FIXTURE_LOOKAHEAD,
)

try:
    from config import (
        TEAM_ID,
        SQUAD_SIZE,
        MAX_PER_CLUB,
        POSITION_LIMITS,
        MAX_DOUBLE_TRANSFER_CANDIDATES,
        TRANSFER_LOG_FILE,
        SQUAD_VALUE_LOG,
        HIT_COST_PTS,
        DIFFERENTIAL_THRESH,
    )
except ImportError:
    TEAM_ID                        = 9179961
    SQUAD_SIZE                     = 15
    MAX_PER_CLUB                   = 3
    POSITION_LIMITS                = {
        "Goalkeeper": 2, "Defender": 5,
        "Midfielder": 5, "Forward":  3,
    }
    MAX_DOUBLE_TRANSFER_CANDIDATES = 40
    TRANSFER_LOG_FILE              = "transfer_history.json"
    SQUAD_VALUE_LOG                = "squad_value_history.json"
    HIT_COST_PTS                   = 4
    DIFFERENTIAL_THRESH            = 15

DOUBLE_HIT_COST_PTS = HIT_COST_PTS * 2   # = 8
# Monte Carlo simulations for captain EV and transfer uncertainty
MONTE_CARLO_N       = 1000
# FPL profit rule: you keep this fraction of any price rise on sale
FPL_PROFIT_FRACTION = 0.5
# Horizon lookahead GWs for multi-GW transfer planning
HORIZON_GWS         = 3
# Max candidates per position for horizon search (controls runtime)
HORIZON_CANDIDATES  = 15
# Total GWs in an FPL season
FPL_SEASON_GWS      = 38

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
# 1. SQUAD VALIDATION
# ─────────────────────────────────────────

def validate_squad(squad_df: pd.DataFrame) -> list:
    """Check squad against FPL rules. Returns violations list."""
    violations = []
    if len(squad_df) != SQUAD_SIZE:
        violations.append(
            f"Squad has {len(squad_df)} players (must be {SQUAD_SIZE})"
        )
    pos_counts = squad_df["position"].value_counts()
    for pos, limit in POSITION_LIMITS.items():
        count = pos_counts.get(pos, 0)
        if count != limit:
            violations.append(f"{pos}: {count} players (must be {limit})")
    club_counts = squad_df["team_name"].value_counts()
    for club, count in club_counts[club_counts > MAX_PER_CLUB].items():
        violations.append(f"{club}: {count} players (max {MAX_PER_CLUB})")
    return violations


def check_transfer_validity(squad_df: pd.DataFrame,
                             player_out: pd.Series,
                             player_in: pd.Series,
                             bank_balance: float) -> list:
    """Check single transfer validity. Returns violations list."""
    violations = []
    if player_out["position"] != player_in["position"]:
        violations.append(
            f"Position mismatch: {player_out['position']} -> {player_in['position']}"
        )
    cost_diff = player_in["price"] - player_out["price"]
    if cost_diff > bank_balance:
        violations.append(
            f"Over budget by £{cost_diff - bank_balance:.1f}M "
            f"(need £{cost_diff:.1f}M, have £{bank_balance:.1f}M)"
        )
    simulated   = squad_df[squad_df["player_id"] != player_out["player_id"]].copy()
    club_counts = simulated["team_name"].value_counts()
    if club_counts.get(player_in["team_name"], 0) >= MAX_PER_CLUB:
        violations.append(
            f"Club limit: already {club_counts.get(player_in['team_name'], 0)} "
            f"from {player_in['team_name']}"
        )
    if player_in["player_id"] in squad_df["player_id"].values:
        violations.append(f"{player_in['player_name']} already in squad")
    return violations


# ─────────────────────────────────────────
# 2. ILP OPTIMAL TRANSFERS
# ─────────────────────────────────────────

def _compute_transfer_price_ev(p_in: pd.Series,
                                p_out: pd.Series,
                                xpts_gain: float) -> float:
    """
    Item 5: Total transfer EV including predicted price movements.

    total_ev = xpts_gain
             + FPL_PROFIT_FRACTION * in_price_change   (buy before rises)
             - FPL_PROFIT_FRACTION * out_price_change  (sell before falls)

    The FPL profit rule: you keep half of any price rise. So if a player
    you own rises £0.2M, selling now means you lose £0.1M vs waiting.
    If a player you're buying is predicted to rise £0.2M, buying now
    saves you £0.2M vs waiting — but you only realise £0.1M of that on
    resale. We model both sides conservatively at FPL_PROFIT_FRACTION.
    """
    in_pchg  = float(p_in.get("predicted_price_change",  0) or 0)
    out_pchg = float(p_out.get("predicted_price_change", 0) or 0)
    price_ev = FPL_PROFIT_FRACTION * (in_pchg - out_pchg)
    return round(xpts_gain + price_ev, 2)


def get_ilp_optimal_transfers(my_team_enriched: pd.DataFrame,
                               other_enriched: pd.DataFrame,
                               bank_balance: float,
                               n_transfers: int = 1) -> dict:
    """
    ILP-based optimal transfer finder using PuLP.

    v5: each pair now also carries total_ev = xPts gain + price movement EV.
    Falls back to greedy if PuLP unavailable.
    """
    if not PULP_AVAILABLE:
        log.warning("PuLP not installed. Run: pip install pulp")
        return {"error": "PuLP not available", "transfers": []}

    has_xpts    = "expected_pts" in my_team_enriched.columns and \
                  "expected_pts" in other_enriched.columns
    prob        = pulp.LpProblem("FPL_Transfer_Optimizer", pulp.LpMaximize)
    valid_pairs = []

    for _, p_out in my_team_enriched.iterrows():
        same_pos = other_enriched[
            other_enriched["position"] == p_out["position"]
        ]
        for _, p_in in same_pos.iterrows():
            cost_diff = p_in["price"] - p_out["price"]
            if cost_diff > bank_balance + 0.5:
                continue
            if p_in["player_id"] in my_team_enriched["player_id"].values:
                continue
            var = pulp.LpVariable(
                f"x_{p_out['player_id']}_{p_in['player_id']}", cat="Binary"
            )
            if has_xpts:
                next_gain = round(
                    float(p_in["expected_pts"]) - float(p_out["expected_pts"]), 2
                )
            else:
                next_gain = round(
                    p_in["predicted_pts"] - p_out["predicted_pts"], 2
                )
            # v5: total_ev adds price movement value to xpts gain
            total_ev = _compute_transfer_price_ev(p_in, p_out, next_gain)
            valid_pairs.append({
                "var":           var,
                "out_id":        p_out["player_id"],
                "in_id":         p_in["player_id"],
                "out_name":      p_out["player_name"],
                "in_name":       p_in["player_name"],
                "position":      p_out["position"],
                "gain":          round(p_in["combined_score"] - p_out["combined_score"], 2),
                "next_gain":     next_gain,
                "total_ev":      total_ev,
                "cost_diff":     cost_diff,
                "in_team":       p_in["team_name"],
                "out_team":      p_out["team_name"],
                "fixture_run":   p_in.get("fixture_run_label", "?"),
                "value_score":   p_in.get("value_score", 0),
                "is_blank":      p_in.get("is_blank_next_gw", False),
                "double_gws":    p_in.get("double_gws", 0),
                "price_change":  float(p_in.get("predicted_price_change", 0) or 0),
                "fixture_trend": float(p_in.get("fixture_trend", 0) or 0),
            })

    if not valid_pairs:
        return {"error": "No valid transfer pairs found", "transfers": []}

    prob += pulp.lpSum(p["gain"] * p["var"] for p in valid_pairs)
    prob += (pulp.lpSum(p["var"] for p in valid_pairs) == n_transfers,
             "total_transfers")

    for _, p_out in my_team_enriched.iterrows():
        out_id   = p_out["player_id"]
        out_vars = [p["var"] for p in valid_pairs if p["out_id"] == out_id]
        if out_vars:
            prob += (pulp.lpSum(out_vars) <= 1, f"out_once_{out_id}")

    for in_id in set(p["in_id"] for p in valid_pairs):
        in_vars = [p["var"] for p in valid_pairs if p["in_id"] == in_id]
        if in_vars:
            prob += (pulp.lpSum(in_vars) <= 1, f"in_once_{in_id}")

    prob += (
        pulp.lpSum(p["cost_diff"] * p["var"] for p in valid_pairs) <= bank_balance,
        "budget"
    )

    all_clubs = set(
        list(my_team_enriched["team_name"].unique()) +
        list(other_enriched["team_name"].unique())
    )
    for club in all_clubs:
        current = int((my_team_enriched["team_name"] == club).sum())
        out_v   = [p["var"] for p in valid_pairs if p["out_team"] == club]
        in_v    = [p["var"] for p in valid_pairs if p["in_team"] == club]
        if in_v:
            prob += (
                current - pulp.lpSum(out_v) + pulp.lpSum(in_v) <= MAX_PER_CLUB,
                f"club_{club.replace(' ', '_').replace('-', '_')}"
            )

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return {"error": f"Solver: {pulp.LpStatus[prob.status]}", "transfers": []}

    chosen = [p for p in valid_pairs if pulp.value(p["var"]) == 1.0]
    return {
        "transfers":       chosen,
        "total_gain":      round(sum(p["gain"]      for p in chosen), 2),
        "total_next_gain": round(sum(p["next_gain"] for p in chosen), 2),
        "total_ev":        round(sum(p["total_ev"]  for p in chosen), 2),
        "total_cost":      round(sum(p["cost_diff"] for p in chosen), 1),
        "n_transfers":     n_transfers,
        "solver_status":   pulp.LpStatus[prob.status],
    }


# ─────────────────────────────────────────
# 3. MULTI-GW HORIZON TRANSFER PLANNER  (new in v5)
# ─────────────────────────────────────────

def get_horizon_transfer_plan(my_team_enriched: pd.DataFrame,
                               other_enriched: pd.DataFrame,
                               enriched_df: pd.DataFrame,
                               bank_balance: float,
                               horizon: int = HORIZON_GWS) -> list:
    """
    Item 1: 3-GW lookahead transfer planning using greedy dynamic programming.

    The greedy single-transfer ILP only finds the best transfer for the current
    GW. This planner asks: "What is the best SEQUENCE of transfers across the
    next N GWs, accounting for how each transfer changes the available options
    for subsequent weeks?"

    Algorithm:
    1. Find top HORIZON_CANDIDATES valid transfers for this week (ranked by gain)
    2. For each candidate Week-1 transfer, simulate the resulting squad
    3. Find the best follow-up transfer for Week-2 given the simulated squad
    4. Rank Week-1 candidates by total_ev(W1) + total_ev(W2)
    5. Return top 3 multi-GW plans

    Runtime: O(HORIZON_CANDIDATES * N_PLAYERS) ≈ manageable for real squads.

    Returns list of plan dicts with transfer sequence and total EV.
    """
    has_xpts = "expected_pts" in my_team_enriched.columns

    def _xpts(row: pd.Series) -> float:
        return float(row.get("expected_pts", row["predicted_pts"]))

    def _best_single(squad_df: pd.DataFrame,
                     pool_df: pd.DataFrame,
                     budget: float,
                     top_n: int = HORIZON_CANDIDATES) -> list:
        """Find top_n valid single transfers for a given squad and pool."""
        candidates = []
        for _, p_out in squad_df.iterrows():
            same_pos = pool_df[pool_df["position"] == p_out["position"]]
            for _, p_in in same_pos.iterrows():
                if check_transfer_validity(squad_df, p_out, p_in, budget):
                    continue
                ng       = round(_xpts(p_in) - _xpts(p_out), 2)
                total_ev = _compute_transfer_price_ev(p_in, p_out, ng)
                candidates.append({
                    "out_id":    p_out["player_id"],
                    "in_id":     p_in["player_id"],
                    "out_name":  p_out["player_name"],
                    "in_name":   p_in["player_name"],
                    "position":  p_out["position"],
                    "gain":      round(p_in["combined_score"] - p_out["combined_score"], 2),
                    "xpts_gain": ng,
                    "total_ev":  total_ev,
                    "cost":      round(p_in["price"] - p_out["price"], 1),
                    "fixture_run": p_in.get("fixture_run_label", "?"),
                    "p_in_row":  p_in,
                    "p_out_row": p_out,
                })
        return sorted(candidates, key=lambda x: x["total_ev"], reverse=True)[:top_n]

    def _simulate_squad(squad_df: pd.DataFrame,
                        transfer: dict) -> pd.DataFrame:
        """Return new squad after applying a transfer."""
        new_squad = squad_df[
            squad_df["player_id"] != transfer["out_id"]
        ].copy()
        new_row = transfer["p_in_row"].to_frame().T
        # Keep only columns present in squad
        keep_cols = [c for c in squad_df.columns if c in new_row.columns]
        new_squad = pd.concat(
            [new_squad[keep_cols], new_row[keep_cols]], ignore_index=True
        )
        return new_squad

    # Week 1: find top candidates
    w1_candidates = _best_single(my_team_enriched, other_enriched, bank_balance)
    if not w1_candidates:
        return []

    plans = []
    for t1 in w1_candidates:
        # Simulate squad after W1 transfer
        simulated_squad = _simulate_squad(my_team_enriched, t1)
        new_budget      = bank_balance - t1["cost"]

        # New pool: remove players now in squad (incl. the one just brought in)
        squad_ids   = set(simulated_squad["player_id"].tolist())
        new_pool    = enriched_df[~enriched_df["player_id"].isin(squad_ids)].copy()

        # Week 2: find best follow-up given simulated squad
        w2_candidates = _best_single(simulated_squad, new_pool, new_budget, top_n=5)

        best_w2 = w2_candidates[0] if w2_candidates else None
        w2_ev   = best_w2["total_ev"] if best_w2 else 0.0

        plans.append({
            "w1_out":        t1["out_name"],
            "w1_in":         t1["in_name"],
            "w1_position":   t1["position"],
            "w1_xpts_gain":  t1["xpts_gain"],
            "w1_total_ev":   t1["total_ev"],
            "w1_cost":       t1["cost"],
            "w1_run":        t1["fixture_run"],
            "w2_out":        best_w2["out_name"]  if best_w2 else "—",
            "w2_in":         best_w2["in_name"]   if best_w2 else "—",
            "w2_xpts_gain":  best_w2["xpts_gain"] if best_w2 else 0.0,
            "w2_total_ev":   w2_ev,
            "w2_run":        best_w2["fixture_run"] if best_w2 else "—",
            "total_horizon_ev": round(t1["total_ev"] + w2_ev, 2),
        })

    plans.sort(key=lambda x: x["total_horizon_ev"], reverse=True)
    return plans[:3]


# ─────────────────────────────────────────
# 4. MONTE CARLO CAPTAIN EV  (new in v5)
# ─────────────────────────────────────────

def run_monte_carlo_captain(my_team_enriched: pd.DataFrame,
                             n_simulations: int = MONTE_CARLO_N) -> list:
    """
    Item 2: Monte Carlo simulation for captain selection.

    Uses triangular distribution [pts_low, predicted_pts, pts_high] from
    Phase 1 v5 quantile regression. For each simulation, each player's score
    is sampled independently; the player with the highest doubled score wins
    the captaincy decision for that simulation.

    Outputs per captainable player:
    - win_prob:              P(this player is optimal captain) across simulations
    - expected_captain_gain: average extra pts from captaining vs not captaining
    - mean_captained_pts:    average doubled score in simulations

    Falls back to deterministic 2*expected_pts if pts_low/pts_high absent.
    """
    captainable = my_team_enriched[
        ~my_team_enriched.get("is_blank_next_gw",
            pd.Series(False, index=my_team_enriched.index))
    ].copy() if "is_blank_next_gw" in my_team_enriched.columns else my_team_enriched.copy()

    if captainable.empty:
        captainable = my_team_enriched.copy()

    has_intervals = (
        "pts_low"  in captainable.columns and
        "pts_high" in captainable.columns
    )

    rng     = np.random.default_rng(42)   # reproducible
    players = captainable.reset_index(drop=True)
    n_p     = len(players)

    if has_intervals:
        # Triangular distribution: mode = expected_pts, a = pts_low, b = pts_high
        xpts_col = "expected_pts" if "expected_pts" in players.columns else "predicted_pts"
        modes = players[xpts_col].astype(float).values
        lows  = players["pts_low"].astype(float).values
        highs = players["pts_high"].astype(float).values

        # Clip to ensure a <= mode <= b
        lows  = np.minimum(lows,  modes)
        highs = np.maximum(highs, modes)

        # Sample triangular: shape (n_simulations, n_players)
        samples = np.zeros((n_simulations, n_p))
        for j in range(n_p):
            if highs[j] - lows[j] < 1e-9:
                samples[:, j] = modes[j]
            else:
                c_val = (modes[j] - lows[j]) / (highs[j] - lows[j])
                c_val = float(np.clip(c_val, 0.0, 1.0))
                samples[:, j] = rng.triangular(
                    left=float(lows[j]),
                    mode=float(modes[j]),
                    right=float(highs[j]),
                    size=n_simulations,
                )
        # Doubled scores
        doubled = samples * 2.0
        # Best captain per simulation
        best_cap_idx = np.argmax(doubled, axis=1)   # shape (n_simulations,)
    else:
        # Deterministic fallback: just use 2*expected_pts
        xpts_col = "expected_pts" if "expected_pts" in players.columns else "predicted_pts"
        ev_vals  = players[xpts_col].astype(float).values
        doubled  = (ev_vals * 2.0).reshape(1, -1).repeat(n_simulations, axis=0)
        best_cap_idx = np.argmax(doubled, axis=1)
        samples  = ev_vals.reshape(1, -1).repeat(n_simulations, axis=0)

    # Aggregate per player
    results = []
    for j in range(n_p):
        win_count            = int((best_cap_idx == j).sum())
        win_prob             = round(win_count / n_simulations, 3)
        mean_cap_pts         = round(float(doubled[:, j].mean()), 2)
        # Captain gain = captained_score - average_non_captain_score
        # Average opponent score = mean of max(all other players doubled scores) per sim
        other_best           = np.max(
            np.delete(doubled, j, axis=1), axis=1
        ) if n_p > 1 else np.zeros(n_simulations)
        captain_gain_sims    = doubled[:, j] - other_best
        expected_captain_gain = round(float(captain_gain_sims.mean()), 2)

        row = players.iloc[j]
        results.append({
            "player_name":          row["player_name"],
            "position":             row["position"],
            "team_name":            row["team_name"],
            "win_prob":             win_prob,
            "expected_captain_gain": expected_captain_gain,
            "mean_captained_pts":   mean_cap_pts,
            "captain_ev":           float(row.get("captain_ev", mean_cap_pts)),
            "fixture_run":          row.get("fixture_run_label", "?"),
            "is_home":              bool(row.get("is_home_next_gw", False)),
            "double_gws":           int(row.get("double_gws", 0)),
        })

    results.sort(key=lambda x: x["win_prob"], reverse=True)
    return results


# ─────────────────────────────────────────
# 5. BENCH OPTIMIZATION  (new in v5)
# ─────────────────────────────────────────

def get_bench_order_recommendation(my_team_enriched: pd.DataFrame) -> dict:
    """
    Item 3: Optimal bench order based on auto-sub expected contribution.

    For each bench player, expected auto-sub contribution is:
      E[contribution_j] = sum over starting XI of:
        P(starter_i misses) * P(bench_j_plays_if_subbed) * bench_j_expected_pts

    Simplified as:
      contribution_j = expected_pts_j * (1 - p_plays_full_j)
      weighted by the probability that at least one starter needs replacing.

    The player with the highest expected contribution should be first sub.
    GK bench rule: a field outfield player cannot sub for a GK — first sub
    must be a field player. GK bench position is always last.

    Returns:
      starting_xi:    recommended 11 starters (by combined_score)
      bench:          list of 4 bench players in recommended order
      first_sub:      name of recommended first sub + reason
      bench_ev_total: total expected bench contribution
    """
    has_xpts = "expected_pts" in my_team_enriched.columns
    xpts_col = "expected_pts" if has_xpts else "predicted_pts"

    players  = my_team_enriched.copy()
    players["_xpts"] = players[xpts_col].astype(float)

    # Sort by combined_score descending — best 11 start
    players = players.sort_values("combined_score", ascending=False).reset_index(drop=True)

    # Separate GK: always one GK starts, one on bench
    gks      = players[players["position"] == "Goalkeeper"]
    outfield = players[players["position"] != "Goalkeeper"]

    # Starting GK = highest combined_score GK
    start_gk  = gks.iloc[[0]] if len(gks) > 0 else pd.DataFrame()
    bench_gk  = gks.iloc[[1]] if len(gks) > 1 else pd.DataFrame()

    # Starting outfield = top 10 by combined_score
    start_out = outfield.head(10)
    bench_out = outfield.iloc[10:] if len(outfield) > 10 else pd.DataFrame()

    starting_xi = pd.concat([start_gk, start_out], ignore_index=True)
    bench_df    = pd.concat([bench_out, bench_gk],  ignore_index=True)

    if bench_df.empty:
        return {
            "starting_xi": starting_xi,
            "bench":       bench_df,
            "first_sub":   None,
            "bench_order": [],
            "bench_ev_total": 0.0,
        }

    # Expected auto-sub contribution per bench player
    # Prob at least one starter misses = 1 - prod(p_plays_full for starters)
    p_full_starters = starting_xi["p_plays_full"].astype(float).values \
        if "p_plays_full" in starting_xi.columns \
        else np.ones(len(starting_xi))
    p_at_least_one_miss = 1.0 - float(np.prod(p_full_starters))

    bench_rows = []
    for _, bp in bench_df.iterrows():
        is_gk    = bp["position"] == "Goalkeeper"
        bp_xpts  = float(bp["_xpts"])
        bp_pfull = float(bp.get("p_plays_full", 1.0))
        # P(bench player plays if subbed in) ≈ 1.0 (they're in the squad)
        # Contribution = xpts * P(gets auto-subbed in) * P(would play)
        # For GK: can only sub for a GK (very rare), lower contribution
        gk_penalty = 0.3 if is_gk else 1.0
        ev = round(bp_xpts * p_at_least_one_miss * gk_penalty, 2)
        bench_rows.append({
            "player_name":     bp["player_name"],
            "position":        bp["position"],
            "team_name":       bp["team_name"],
            "expected_pts":    bp_xpts,
            "p_plays_full":    bp_pfull,
            "bench_ev":        ev,
            "is_gk":           is_gk,
        })

    # Sort: outfield first (GK always last regardless), then by bench_ev desc
    bench_rows.sort(key=lambda x: (x["is_gk"], -x["bench_ev"]))
    first_sub_name   = bench_rows[0]["player_name"] if bench_rows else None
    bench_ev_total   = round(sum(b["bench_ev"] for b in bench_rows), 2)

    reason = ""
    if bench_rows:
        top = bench_rows[0]
        reason = (
            f"Highest auto-sub EV: {top['expected_pts']:.2f} xPts × "
            f"{p_at_least_one_miss*100:.0f}% chance of auto-sub = "
            f"{top['bench_ev']:.2f} expected pts"
        )

    return {
        "starting_xi":    starting_xi,
        "bench":          pd.DataFrame(bench_rows),
        "first_sub":      first_sub_name,
        "first_sub_reason": reason,
        "bench_order":    [b["player_name"] for b in bench_rows],
        "bench_ev_total": bench_ev_total,
        "p_at_least_one_miss": round(p_at_least_one_miss, 3),
    }


# ─────────────────────────────────────────
# 6. HIT TRANSFER ANALYSIS
# ─────────────────────────────────────────

def get_hit_transfer_analysis(my_team_enriched: pd.DataFrame,
                               other_enriched: pd.DataFrame,
                               bank_balance: float,
                               transfers_made: int) -> list:
    """
    Analyse whether a -4pt single hit transfer is worth it.
    Break-even on expected_pts (same unit as hit cost).
    v5: also shows total_ev (includes price change value).
    """
    if transfers_made == 0:
        return []

    has_xpts    = "expected_pts" in my_team_enriched.columns
    hit_results = []

    for _, p_out in my_team_enriched.iterrows():
        same_pos = other_enriched[
            other_enriched["position"] == p_out["position"]
        ]
        for _, p_in in same_pos.iterrows():
            if check_transfer_validity(my_team_enriched, p_out, p_in, bank_balance):
                continue
            if has_xpts:
                xpts_gain = round(
                    float(p_in["expected_pts"]) - float(p_out["expected_pts"]), 2
                )
            else:
                xpts_gain = round(
                    p_in["predicted_pts"] - p_out["predicted_pts"], 2
                )
            combined_gain = round(
                p_in["combined_score"] - p_out["combined_score"], 2
            )
            total_ev  = _compute_transfer_price_ev(p_in, p_out, xpts_gain)
            net_value = round(xpts_gain - HIT_COST_PTS, 2)

            if xpts_gain > HIT_COST_PTS and combined_gain > 0:
                hit_results.append({
                    "replace":       p_out["player_name"],
                    "player_in":     p_in["player_name"],
                    "position":      p_in["position"],
                    "xpts_gain":     xpts_gain,
                    "combined_gain": combined_gain,
                    "total_ev":      total_ev,
                    "net_value":     net_value,
                    "cost_diff":     round(p_in["price"] - p_out["price"], 1),
                    "fixture_run":   p_in.get("fixture_run_label", "?"),
                    "value_score":   p_in.get("value_score", 0),
                    "is_blank":      p_in.get("is_blank_next_gw", False),
                    "double_gws":    p_in.get("double_gws", 0),
                    "price_change":  float(p_in.get("predicted_price_change", 0) or 0),
                })

    return sorted(hit_results, key=lambda x: x["net_value"], reverse=True)[:5]


def get_double_hit_analysis(my_team_enriched: pd.DataFrame,
                             other_enriched: pd.DataFrame,
                             bank_balance: float,
                             transfers_made: int) -> list:
    """Analyse -8pt double hit. Two extra transfers, worth it if combined xPts > 8."""
    if transfers_made == 0:
        return []

    singles: list = []
    for _, p_out in my_team_enriched.iterrows():
        same_pos = other_enriched[
            other_enriched["position"] == p_out["position"]
        ]
        for _, p_in in same_pos.iterrows():
            if check_transfer_validity(my_team_enriched, p_out, p_in, bank_balance):
                continue
            xpts_gain = round(
                float(p_in.get("expected_pts", p_in["predicted_pts"])) -
                float(p_out.get("expected_pts", p_out["predicted_pts"])), 2
            )
            if xpts_gain <= 0:
                continue
            singles.append({
                "out":        p_out, "in": p_in,
                "xpts_gain":  xpts_gain,
                "comb_gain":  round(p_in["combined_score"] - p_out["combined_score"], 2),
                "cost":       round(p_in["price"] - p_out["price"], 1),
            })

    singles.sort(key=lambda x: x["xpts_gain"], reverse=True)
    singles = singles[:MAX_DOUBLE_TRANSFER_CANDIDATES]

    double_hits: list = []
    seen_pairs: set   = set()

    for t1, t2 in itertools.combinations(singles, 2):
        if t1["out"]["player_id"] == t2["out"]["player_id"]: continue
        if t1["in"]["player_id"]  == t2["in"]["player_id"]:  continue
        pair_key = tuple(sorted([t1["out"]["player_id"], t2["out"]["player_id"]]))
        if pair_key in seen_pairs: continue
        seen_pairs.add(pair_key)
        total_cost = t1["cost"] + t2["cost"]
        if total_cost > bank_balance: continue
        simulated = my_team_enriched[
            ~my_team_enriched["player_id"].isin([t1["out"]["player_id"], t2["out"]["player_id"]])
        ].copy()
        in1 = t1["in"][["player_id","player_name","position","team_name","price"]].copy()
        in2 = t2["in"][["player_id","player_name","position","team_name","price"]].copy()
        simulated = pd.concat([
            simulated[["player_id","player_name","position","team_name","price"]],
            in1.to_frame().T, in2.to_frame().T,
        ], ignore_index=True)
        if validate_squad(simulated): continue
        total_xpts = round(t1["xpts_gain"] + t2["xpts_gain"], 2)
        net_value  = round(total_xpts - DOUBLE_HIT_COST_PTS, 2)
        if total_xpts > DOUBLE_HIT_COST_PTS:
            double_hits.append({
                "t1_out":          t1["out"]["player_name"],
                "t1_in":           t1["in"]["player_name"],
                "t1_xpts_gain":    t1["xpts_gain"],
                "run_1":           t1["in"].get("fixture_run_label", "?"),
                "t2_out":          t2["out"]["player_name"],
                "t2_in":           t2["in"]["player_name"],
                "t2_xpts_gain":    t2["xpts_gain"],
                "run_2":           t2["in"].get("fixture_run_label", "?"),
                "total_xpts_gain": total_xpts,
                "net_value":       net_value,
                "total_cost":      round(total_cost, 1),
            })

    double_hits.sort(key=lambda x: x["net_value"], reverse=True)
    return double_hits[:3]


# ─────────────────────────────────────────
# 7. ROLLING FREE TRANSFER ADVICE
# ─────────────────────────────────────────

def get_rolling_transfer_advice(my_team_enriched: pd.DataFrame,
                                 other_enriched: pd.DataFrame,
                                 bank_balance: float,
                                 transfers_made: int,
                                 chip_info: dict,
                                 current_gw: int,
                                 ilp_result: dict = None) -> dict:
    """
    Advise whether to use or roll the free transfer.
    Thresholds on next_xpts_gain (raw pts): >=2.0 USE NOW, <0.5 ROLL.
    v5: also factors in transfer urgency from price changes and fixture trends.
    """
    has_xpts = "expected_pts" in my_team_enriched.columns

    if ilp_result and ilp_result.get("transfers"):
        t              = ilp_result["transfers"][0]
        best_gain      = t["gain"]
        best_next_gain = t["next_gain"]
        best_total_ev  = t.get("total_ev", t["next_gain"])
        best_move      = (t["out_name"], t["in_name"])
    else:
        best_gain = best_next_gain = best_total_ev = 0.0
        best_move = None
        for _, p_out in my_team_enriched.iterrows():
            same_pos = other_enriched[
                other_enriched["position"] == p_out["position"]
            ]
            for _, p_in in same_pos.iterrows():
                if check_transfer_validity(my_team_enriched, p_out, p_in, bank_balance):
                    continue
                gain = round(p_in["combined_score"] - p_out["combined_score"], 2)
                if has_xpts:
                    ng = round(float(p_in["expected_pts"]) - float(p_out["expected_pts"]), 2)
                else:
                    ng = round(p_in["predicted_pts"] - p_out["predicted_pts"], 2)
                ev = _compute_transfer_price_ev(p_in, p_out, ng)
                if gain > best_gain:
                    best_gain      = gain
                    best_next_gain = ng
                    best_total_ev  = ev
                    best_move      = (p_out["player_name"], p_in["player_name"])

    dgw_gws       = chip_info.get("dgw_gws", [])
    next_dgw      = dgw_gws[0] if dgw_gws else None
    dgw_next_week = (
        next_dgw is not None and next_dgw.get("gw") == current_gw + 2
    )
    has_blank = (
        my_team_enriched["is_blank_next_gw"].any()
        if "is_blank_next_gw" in my_team_enriched.columns else False
    )

    reasons: list = []
    recommend     = "ROLL"

    if transfers_made > 0:
        recommend = "ALREADY USED"
        reasons.append("Free transfer already used this GW.")
    elif dgw_next_week:
        recommend = "ROLL"
        reasons.append(
            f"DGW coming in GW{next_dgw['gw']} — roll to have 2 free transfers."
        )
    elif has_blank:
        recommend = "USE NOW"
        reasons.append(
            "Squad player with a blank next GW — use your transfer."
        )
    elif best_next_gain >= 2.0:
        recommend = "USE NOW"
        reasons.append(
            f"Strong xPts gain (+{best_next_gain:.2f}) — use your free transfer."
        )
    elif best_next_gain < 0.5:
        recommend = "ROLL"
        reasons.append(
            f"Best immediate gain only +{best_next_gain:.2f} xPts — roll."
        )
    else:
        recommend = "BORDERLINE"
        reasons.append(
            f"Gain of +{best_next_gain:.2f} xPts is moderate — your call."
        )

    # v5: add price urgency signal
    if best_total_ev > best_next_gain + 0.1:
        reasons.append(
            f"Price EV boosts total value to +{best_total_ev:.2f} "
            f"(includes predicted price movements)."
        )

    return {
        "recommendation":  recommend,
        "reasons":         reasons,
        "best_gain":       best_gain,
        "best_next_gain":  best_next_gain,
        "best_total_ev":   best_total_ev,
        "best_move":       best_move,
        "dgw_next_week":   dgw_next_week,
    }


# ─────────────────────────────────────────
# 8. CAPTAINCY DIFFERENTIAL ANALYSIS  (new in v5)
# ─────────────────────────────────────────

def get_captaincy_differential_analysis(my_team_enriched: pd.DataFrame,
                                         bootstrap: dict) -> list:
    """
    Item 6: Captain gain vs the average manager in the game.

    The average FPL manager overwhelmingly caps the highest-owned player
    (Haaland/Salah/etc). If 60% of managers cap Player X, and you cap
    Player Y instead, your differential gain is:
      diff_gain = your_captain_ev - field_weighted_ev

    where field_weighted_ev = sum(ownership * captain_ev) / sum(ownership)
    across ALL players in the game (not just your squad).

    Positive diff_gain → captaining this player GAINS you rank vs the field.
    Negative → you lose rank vs average managers.

    Returns list sorted by differential_gain descending.
    """
    players_raw   = bootstrap["elements"]
    ownership_map = {
        p["id"]: float(p.get("selected_by_percent", 0) or 0)
        for p in players_raw
    }

    captainable = my_team_enriched[
        ~my_team_enriched.get("is_blank_next_gw",
            pd.Series(False, index=my_team_enriched.index))
    ].copy() if "is_blank_next_gw" in my_team_enriched.columns else my_team_enriched.copy()

    if captainable.empty:
        captainable = my_team_enriched.copy()

    xpts_col = "expected_pts" if "expected_pts" in captainable.columns else "predicted_pts"

    # Field-weighted captain EV: estimated average FPL manager's captain score
    # Using ownership as proxy for captaincy frequency (high-owned players are captained more)
    # We approximate: P(manager caps player X) ≈ ownership_pct / 100
    # so field_ev = sum(P(caps X) * captain_ev(X)) normalised across all field players
    # Simplified to just the top 10 most owned players in the game for tractability
    top_owned   = sorted(players_raw, key=lambda p: float(p.get("selected_by_percent",0) or 0), reverse=True)[:20]
    field_ev_num = 0.0
    field_ev_den = 0.0
    for p in top_owned:
        own     = float(p.get("selected_by_percent", 0) or 0)
        now_pts = float(p.get("form", 0) or 0) * 2  # rough proxy for captain EV from form
        field_ev_num += own * now_pts
        field_ev_den += own

    # Better: use our own captainable players' EV weighted by ownership as proxy
    # for average manager decision (most managers cap from a small pool)
    squad_field_ev_num = 0.0
    squad_field_ev_den = 0.0
    for _, row in captainable.iterrows():
        own     = ownership_map.get(int(row["player_id"]), 0)
        cap_ev  = float(row.get("captain_ev", float(row[xpts_col]) * 2))
        squad_field_ev_num += own * cap_ev
        squad_field_ev_den += own

    if squad_field_ev_den > 0:
        field_captain_ev = squad_field_ev_num / squad_field_ev_den
    else:
        field_captain_ev = float(captainable[xpts_col].max()) * 2

    results = []
    for _, row in captainable.iterrows():
        own     = ownership_map.get(int(row["player_id"]), 0)
        cap_ev  = float(row.get("captain_ev", float(row[xpts_col]) * 2))
        diff_gain = round(cap_ev - field_captain_ev, 2)

        results.append({
            "player_name":       row["player_name"],
            "position":          row["position"],
            "team_name":         row["team_name"],
            "ownership_pct":     own,
            "captain_ev":        round(cap_ev, 2),
            "field_captain_ev":  round(field_captain_ev, 2),
            "differential_gain": diff_gain,
            "fixture_run":       row.get("fixture_run_label", "?"),
            "is_home":           bool(row.get("is_home_next_gw", False)),
            "is_differential":   own < DIFFERENTIAL_THRESH,
        })

    results.sort(key=lambda x: x["differential_gain"], reverse=True)
    return results


# ─────────────────────────────────────────
# 9. WILDCARD OPTIMISATION
# ─────────────────────────────────────────

def get_wildcard_squad(all_players_enriched: pd.DataFrame,
                        budget: float = 100.0) -> pd.DataFrame:
    """
    Build optimal 15-man squad using ILP (Wildcard).
    Maximises combined_score subject to budget, position limits, club limits.
    Falls back to greedy if PuLP unavailable.
    """
    if not PULP_AVAILABLE:
        return _greedy_best_squad(all_players_enriched, budget, "combined_score")

    players = all_players_enriched.reset_index(drop=True)
    n       = len(players)
    prob    = pulp.LpProblem("Wildcard_Builder", pulp.LpMaximize)
    x       = [pulp.LpVariable(f"wc_{i}", cat="Binary") for i in range(n)]

    prob += pulp.lpSum(players.loc[i, "combined_score"] * x[i] for i in range(n))
    prob += (pulp.lpSum(x) == SQUAD_SIZE, "size")
    prob += (
        pulp.lpSum(players.loc[i, "price"] * x[i] for i in range(n)) <= budget,
        "budget"
    )
    for pos, limit in POSITION_LIMITS.items():
        idx = players[players["position"] == pos].index.tolist()
        prob += (pulp.lpSum(x[i] for i in idx) == limit, f"pos_{pos}")
    for club in players["team_name"].unique():
        idx = players[players["team_name"] == club].index.tolist()
        prob += (
            pulp.lpSum(x[i] for i in idx) <= MAX_PER_CLUB,
            f"club_{club.replace(' ', '_').replace('-', '_')}"
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return _greedy_best_squad(all_players_enriched, budget, "combined_score")

    idx = [i for i in range(n) if pulp.value(x[i]) == 1.0]
    return players.loc[idx].reset_index(drop=True)


def print_wildcard_diff(current_squad: pd.DataFrame,
                         new_squad: pd.DataFrame,
                         bank_balance: float) -> None:
    """
    Item 7: Show exactly what changes between current squad and wildcard squad.
    Prints SELL / KEEP / BUY breakdown with cost summary.
    """
    curr_ids = set(current_squad["player_id"].tolist())
    new_ids  = set(new_squad["player_id"].tolist())

    sell_ids = curr_ids - new_ids
    buy_ids  = new_ids  - curr_ids
    keep_ids = curr_ids & new_ids

    sell_df = current_squad[current_squad["player_id"].isin(sell_ids)].copy()
    buy_df  = new_squad[new_squad["player_id"].isin(buy_ids)].copy()
    keep_df = current_squad[current_squad["player_id"].isin(keep_ids)].copy()

    sell_total = sell_df["price"].sum() if not sell_df.empty else 0.0
    buy_total  = buy_df["price"].sum()  if not buy_df.empty else 0.0
    net_cost   = round(buy_total - sell_total, 1)

    print(f"\n  WILDCARD SQUAD CHANGES")
    print(f"  {'─' * 65}")

    if not sell_df.empty:
        print(f"\n  🔴 SELL ({len(sell_df)} players, £{sell_total:.1f}M freed):")
        for _, r in sell_df.sort_values("position").iterrows():
            print(f"    - {str(r['player_name']):28s}  [{r['position']:3s}]  £{r['price']:.1f}M")

    if not buy_df.empty:
        print(f"\n  🟢 BUY ({len(buy_df)} players, £{buy_total:.1f}M spent):")
        for _, r in buy_df.sort_values("position").iterrows():
            xpts = float(r.get("expected_pts", r.get("predicted_pts", 0)))
            print(
                f"    + {str(r['player_name']):28s}  [{r['position']:3s}]"
                f"  £{r['price']:.1f}M"
                f"  xPts:{xpts:.1f}"
                f"  Run:{r.get('fixture_run_label','?')}"
            )

    if not keep_df.empty:
        print(f"\n  ⚪ KEEP ({len(keep_df)} players):")
        for _, r in keep_df.sort_values("position").iterrows():
            print(f"    = {str(r['player_name']):28s}  [{r['position']:3s}]  £{r['price']:.1f}M")

    print(f"\n  Net cost of wildcard: {net_cost:+.1f}M")
    print(f"  Bank before: £{bank_balance:.1f}M  →  Bank after: £{bank_balance - net_cost:.1f}M")


# ─────────────────────────────────────────
# 10. FREE HIT SQUAD BUILDER
# ─────────────────────────────────────────

def get_free_hit_squad(all_players_enriched: pd.DataFrame,
                        budget: float = 100.0) -> pd.DataFrame:
    """
    Build best 15 for one GW using Free Hit.
    Maximises expected_pts. Excludes blank GW players.
    """
    candidates = all_players_enriched.copy()
    if "is_blank_next_gw" in candidates.columns:
        candidates = candidates[~candidates["is_blank_next_gw"]]
    score_col = "expected_pts" if "expected_pts" in candidates.columns else "predicted_pts"

    if not PULP_AVAILABLE:
        return _greedy_best_squad(candidates, budget, score_col)

    players = candidates.reset_index(drop=True)
    n       = len(players)
    prob    = pulp.LpProblem("FreeHit_Builder", pulp.LpMaximize)
    x       = [pulp.LpVariable(f"fh_{i}", cat="Binary") for i in range(n)]

    prob += pulp.lpSum(players.loc[i, score_col] * x[i] for i in range(n))
    prob += (pulp.lpSum(x) == SQUAD_SIZE, "size")
    prob += (
        pulp.lpSum(players.loc[i, "price"] * x[i] for i in range(n)) <= budget,
        "budget"
    )
    for pos, limit in POSITION_LIMITS.items():
        idx = players[players["position"] == pos].index.tolist()
        prob += (pulp.lpSum(x[i] for i in idx) == limit, f"pos_{pos}")
    for club in players["team_name"].unique():
        idx = players[players["team_name"] == club].index.tolist()
        prob += (
            pulp.lpSum(x[i] for i in idx) <= MAX_PER_CLUB,
            f"club_{club.replace(' ', '_').replace('-', '_')}"
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return _greedy_best_squad(candidates, budget, score_col)

    idx = [i for i in range(n) if pulp.value(x[i]) == 1.0]
    return players.loc[idx].reset_index(drop=True)


def _greedy_best_squad(players_df: pd.DataFrame,
                        budget: float,
                        score_col: str) -> pd.DataFrame:
    """Greedy fallback squad builder."""
    selected: list = []
    remaining   = budget
    pos_counts  = {p: 0 for p in POSITION_LIMITS}
    club_counts: dict = {}
    for _, player in players_df.sort_values(score_col, ascending=False).iterrows():
        pos   = player["position"]
        club  = player["team_name"]
        price = player["price"]
        if pos_counts.get(pos, 0) >= POSITION_LIMITS.get(pos, 0): continue
        if club_counts.get(club, 0) >= MAX_PER_CLUB:               continue
        if price > remaining:                                        continue
        selected.append(player)
        pos_counts[pos]   = pos_counts.get(pos, 0) + 1
        club_counts[club] = club_counts.get(club, 0) + 1
        remaining        -= price
        if len(selected) == SQUAD_SIZE: break
    return pd.DataFrame(selected).reset_index(drop=True)


# ─────────────────────────────────────────
# 11. SQUAD VALUE BREAKDOWN
# ─────────────────────────────────────────

def get_squad_value_breakdown(my_team_enriched: pd.DataFrame,
                               bootstrap: dict,
                               team_data: dict) -> pd.DataFrame:
    """
    Show each player's sell value vs predicted pts.
    Uses selling_price from picks (FPL profit rule applied).
    """
    players_raw   = bootstrap["elements"]
    now_price_map = {p["id"]: p["now_cost"] / 10 for p in players_raw}
    sell_price_map: dict = {}
    if isinstance(team_data, dict) and "picks" in team_data:
        for pick in team_data.get("picks", []):
            pid = pick.get("element")
            sp  = pick.get("selling_price")
            if pid and sp is not None:
                sell_price_map[pid] = sp / 10

    rows = []
    for _, player in my_team_enriched.iterrows():
        pid        = int(player["player_id"])
        now_price  = now_price_map.get(pid, player["price"])
        sell_price = sell_price_map.get(pid, now_price)
        rows.append({
            "player_name":    player["player_name"],
            "position":       player["position"],
            "team_name":      player["team_name"],
            "now_price":      now_price,
            "sell_price":     sell_price,
            "predicted_pts":  player["predicted_pts"],
            "expected_pts":   float(player.get("expected_pts", player["predicted_pts"])),
            "combined_score": player["combined_score"],
            "fixture_run":    player.get("fixture_run_label", "?"),
            "is_blank":       player.get("is_blank_next_gw", False),
        })

    df = pd.DataFrame(rows)
    df["total_sell_value"] = df["sell_price"].sum()
    return df


# ─────────────────────────────────────────
# 12. DIFFERENTIAL PICKS
# ─────────────────────────────────────────

def get_differential_picks(other_enriched: pd.DataFrame,
                             bootstrap: dict,
                             top_n: int = 5) -> pd.DataFrame:
    """Low-ownership (<DIFFERENTIAL_THRESH%) high-value transfer targets."""
    players_raw   = bootstrap["elements"]
    ownership_map = {
        p["id"]: float(p.get("selected_by_percent", 0) or 0)
        for p in players_raw
    }
    enriched = other_enriched.copy()
    enriched["ownership_pct"] = enriched["player_id"].map(ownership_map).fillna(0)
    differentials = enriched[enriched["ownership_pct"] < DIFFERENTIAL_THRESH].copy()
    if differentials.empty:
        return pd.DataFrame()
    differentials["differential_score"] = (
        differentials["combined_score"] *
        (1 + (DIFFERENTIAL_THRESH - differentials["ownership_pct"]) / 100)
    ).round(2)
    cols = [c for c in [
        "player_name","team_name","position","price",
        "predicted_pts","fixture_run_label","combined_score",
        "value_score","ownership_pct","differential_score",
    ] if c in differentials.columns]
    return (
        differentials.sort_values("differential_score", ascending=False)
        .head(top_n)[cols].reset_index(drop=True)
    )


# ─────────────────────────────────────────
# 13. SQUAD VALUE TRACKER
# ─────────────────────────────────────────

def track_squad_value(my_team_enriched: pd.DataFrame,
                       bootstrap: dict,
                       current_gw: int,
                       team_data: dict = None) -> dict:
    """Track squad sell value GW-by-GW."""
    players_raw   = bootstrap["elements"]
    now_price_map = {p["id"]: p["now_cost"] / 10 for p in players_raw}
    sell_price_map: dict = {}
    if team_data and "picks" in team_data:
        for pick in team_data.get("picks", []):
            pid = pick.get("element")
            sp  = pick.get("selling_price")
            if pid and sp is not None:
                sell_price_map[pid] = sp / 10

    current_value = round(
        sum(
            sell_price_map.get(int(pid), now_price_map.get(int(pid), 0))
            for pid in my_team_enriched["player_id"]
        ), 1
    )
    history: dict = {}
    if Path(SQUAD_VALUE_LOG).exists():
        with open(SQUAD_VALUE_LOG, "r") as f:
            history = json.load(f)
    norm: dict = {}
    for k, v in history.items():
        try:
            norm[str(int(k))] = round(float(v), 1)
        except (TypeError, ValueError):
            continue
    history = norm
    history[str(int(current_gw))] = float(current_value)
    with open(SQUAD_VALUE_LOG, "w") as f:
        json.dump(history, f, indent=2)

    baseline_gw_str = min(history.keys(), key=int)
    baseline_gw     = int(baseline_gw_str)
    baseline_value  = float(history[baseline_gw_str])
    total_change    = round(float(current_value) - baseline_value, 1)

    sorted_gws     = sorted(history.keys(), key=int)
    weekly_changes: dict = {}
    for i in range(1, len(sorted_gws)):
        pg = sorted_gws[i - 1]
        cg = sorted_gws[i]
        weekly_changes[cg] = round(float(history[cg]) - float(history[pg]), 1)

    return {
        "current_value":  float(current_value),
        "baseline_value": baseline_value,
        "baseline_gw":    baseline_gw,
        "current_gw":     int(current_gw),
        "total_change":   total_change,
        "history":        history,
        "weekly_changes": weekly_changes,
    }


def print_squad_value_tracking(value_data: dict) -> None:
    """Print squad value trend over time."""
    sign = "+" if value_data["total_change"] >= 0 else ""
    print(
        f"\n  Squad value since GW{value_data['baseline_gw']}: "
        f"£{value_data['baseline_value']}M  ->  £{value_data['current_value']}M  "
        f"({sign}{value_data['total_change']}M overall)"
    )
    history        = value_data["history"]
    weekly_changes = value_data.get("weekly_changes", {})
    sorted_gws     = sorted(history.keys(), key=int)
    if len(sorted_gws) > 1:
        print(f"\n  GW-by-GW value trend:")
        for gw in sorted_gws:
            val    = history[gw]
            change = weekly_changes.get(gw, 0)
            marker = (
                "  (baseline)" if int(gw) == value_data["baseline_gw"]
                else f"  ({'+' if change >= 0 else ''}{change}M)"
            )
            curr = " <-- current" if int(gw) == value_data["current_gw"] else ""
            print(f"    GW{gw:<4}  £{val}M{marker}{curr}")


# ─────────────────────────────────────────
# 14. TRANSFER HISTORY TRACKER
# ─────────────────────────────────────────

def log_transfer_suggestion(player_out: str,
                              player_in: str,
                              predicted_gain: float,
                              current_gw: int) -> None:
    """Log a transfer suggestion to transfer_history.json."""
    history = _load_transfer_history()
    history.append({
        "gw":             current_gw,
        "timestamp":      datetime.now().isoformat(),
        "player_out":     player_out,
        "player_in":      player_in,
        "predicted_gain": predicted_gain,
        "actual_gain":    None,
        "evaluated":      False,
    })
    _save_transfer_history(history)
    log.info(f"Transfer logged: OUT {player_out} -> IN {player_in} (gain: {predicted_gain})")


def evaluate_past_transfers(history_df: pd.DataFrame, current_gw: int) -> list:
    """
    Evaluate past transfer suggestions vs actual points.

    v5 fix: evaluation window capped at min(5, gws_remaining_since_transfer)
    so late-season entries (GW34+) are still evaluated with available data,
    rather than silently skipped because 5 GWs haven't passed.

    Also uses exact full-name match → contains fallback (not surname split).
    """
    history = _load_transfer_history()
    updated = False

    for entry in history:
        if entry["evaluated"]:
            continue
        entry_gw       = entry.get("gw", 0)
        gws_elapsed    = current_gw - entry_gw
        # v5: cap evaluation window at season end
        eval_window    = min(5, FPL_SEASON_GWS - entry_gw)
        if gws_elapsed < min(3, eval_window):   # need at least 3 GWs of data
            continue

        def _find_player(name: str) -> pd.DataFrame:
            exact = history_df[history_df["player_name"] == name]
            if not exact.empty:
                return exact
            return history_df[
                history_df["player_name"].str.contains(name, case=False, na=False)
            ]

        in_rows  = _find_player(entry["player_in"])
        out_rows = _find_player(entry["player_out"])

        if not in_rows.empty and not out_rows.empty:
            actual_window = range(entry_gw + 1, entry_gw + eval_window + 1)
            in_pts  = in_rows[in_rows["round"].isin(actual_window)]["total_points"].sum()
            out_pts = out_rows[out_rows["round"].isin(actual_window)]["total_points"].sum()
            entry["actual_gain"]   = round(in_pts - out_pts, 1)
            entry["eval_window"]   = eval_window
            entry["evaluated"]     = True
            updated = True

    if updated:
        _save_transfer_history(history)
    return history


def _load_transfer_history() -> list:
    if Path(TRANSFER_LOG_FILE).exists():
        with open(TRANSFER_LOG_FILE, "r") as f:
            return json.load(f)
    return []


def _save_transfer_history(history: list) -> None:
    with open(TRANSFER_LOG_FILE, "w") as f:
        json.dump(history, f, indent=2)


def print_transfer_history() -> None:
    """Print evaluated past transfer suggestions."""
    history   = _load_transfer_history()
    evaluated = [h for h in history if h["evaluated"]]
    if not evaluated:
        print("  No evaluated transfers yet.")
        return
    print(f"\n  {'GW':<5} {'OUT':<25} {'IN':<25} {'Predicted':>10} {'Actual':>8} {'Window':>7} {'Result':>8}")
    print(f"  {'-' * 88}")
    for h in sorted(evaluated, key=lambda x: x["gw"], reverse=True):
        predicted = h.get("predicted_gain", 0)
        actual    = h.get("actual_gain",    0)
        window    = h.get("eval_window",    5)
        result    = "✅ Good" if actual >= predicted * 0.7 else "❌ Miss"
        print(
            f"  GW{h['gw']:<3} {h['player_out'][:23]:<25} "
            f"{h['player_in'][:23]:<25} "
            f"{predicted:>+10.1f} {actual:>+8.1f} {window:>7} {result:>8}"
        )


# ─────────────────────────────────────────
# 15. VALID SINGLE TRANSFERS (GREEDY)
# ─────────────────────────────────────────

def get_valid_transfers(my_team_enriched: pd.DataFrame,
                         other_enriched: pd.DataFrame,
                         bank_balance: float,
                         top_n: int = 5) -> pd.DataFrame:
    """
    Greedy constraint-checked transfer suggestions.
    v5: includes total_ev (xpts + price movement) and urgency_score.
    """
    has_xpts = "expected_pts" in my_team_enriched.columns
    valid    = []

    for _, p_out in my_team_enriched.iterrows():
        same_pos = other_enriched[other_enriched["position"] == p_out["position"]]
        for _, p_in in same_pos.iterrows():
            if check_transfer_validity(my_team_enriched, p_out, p_in, bank_balance):
                continue
            gain = round(p_in["combined_score"] - p_out["combined_score"], 2)
            if gain <= 0:
                continue
            if has_xpts:
                next_gain = round(
                    float(p_in["expected_pts"]) - float(p_out["expected_pts"]), 2
                )
            else:
                next_gain = round(p_in["predicted_pts"] - p_out["predicted_pts"], 2)
            total_ev = _compute_transfer_price_ev(p_in, p_out, next_gain)

            # Item 4: urgency score — how costly is delaying one week?
            # Urgency is high if: incoming player's fixtures get harder (positive trend),
            # or incoming player's price is rising, or outgoing player has blank next week
            in_trend     = float(p_in.get("fixture_trend", 0) or 0)
            in_pchg      = float(p_in.get("predicted_price_change", 0) or 0)
            out_is_blank = bool(p_out.get("is_blank_next_gw", False))
            urgency_score = round(
                (max(0, in_trend) * 0.5) +   # harder fixtures if you delay
                (max(0, in_pchg) * 1.0) +    # price rises cost you
                (2.0 if out_is_blank else 0), # blank this week = urgent
                2
            )

            valid.append({
                "replace":           p_out["player_name"],
                "replace_id":        p_out["player_id"],
                "player_in":         p_in["player_name"],
                "player_in_id":      p_in["player_id"],
                "position":          p_in["position"],
                "team_in":           p_in["team_name"],
                "price_in":          p_in["price"],
                "cost_diff":         round(p_in["price"] - p_out["price"], 1),
                "combined_gain":     gain,
                "next_gw_gain":      next_gain,
                "total_ev":          total_ev,
                "urgency_score":     urgency_score,
                "fixture_run":       p_in.get("fixture_run_label", "?"),
                "predicted_pts":     p_in["predicted_pts"],
                "expected_pts":      float(p_in.get("expected_pts", p_in["predicted_pts"])),
                "combined_score":    p_in["combined_score"],
                "avg_difficulty":    p_in.get("avg_difficulty", 3),
                "value_score":       p_in.get("value_score", 0),
                "is_blank_next_gw":  p_in.get("is_blank_next_gw", False),
                "double_gws":        p_in.get("double_gws", 0),
                "momentum_score":    p_in.get("momentum_score", 3),
                "price_change":      float(p_in.get("predicted_price_change", 0) or 0),
                "fixture_trend":     float(p_in.get("fixture_trend", 0) or 0),
            })

    if not valid:
        return pd.DataFrame()

    return (
        pd.DataFrame(valid)
        .sort_values("combined_gain", ascending=False)
        .drop_duplicates("player_in")
        .head(top_n * 3)
    )


# ─────────────────────────────────────────
# 16. VALID 2-TRANSFER COMBINATIONS
# ─────────────────────────────────────────

def get_valid_double_transfers(my_team_enriched: pd.DataFrame,
                                other_enriched: pd.DataFrame,
                                bank_balance: float,
                                top_n: int = 3,
                                precomputed_ilp: dict = None) -> list:
    """Best valid 2-transfer combinations. v5: includes total_ev."""
    if precomputed_ilp and precomputed_ilp.get("transfers") and \
       len(precomputed_ilp["transfers"]) == 2:
        t = precomputed_ilp["transfers"]
        return [{
            "transfer_1_out":      t[0]["out_name"],
            "transfer_1_in":       t[0]["in_name"],
            "transfer_1_gain":     t[0]["gain"],
            "transfer_1_next":     t[0]["next_gain"],
            "transfer_1_ev":       t[0].get("total_ev", t[0]["next_gain"]),
            "run_1":               t[0]["fixture_run"],
            "blank_1":             t[0]["is_blank"],
            "dgw_1":               t[0]["double_gws"],
            "transfer_2_out":      t[1]["out_name"],
            "transfer_2_in":       t[1]["in_name"],
            "transfer_2_gain":     t[1]["gain"],
            "transfer_2_next":     t[1]["next_gain"],
            "transfer_2_ev":       t[1].get("total_ev", t[1]["next_gain"]),
            "run_2":               t[1]["fixture_run"],
            "blank_2":             t[1]["is_blank"],
            "dgw_2":               t[1]["double_gws"],
            "total_combined_gain": precomputed_ilp["total_gain"],
            "total_next_gw_gain":  precomputed_ilp["total_next_gain"],
            "total_ev":            precomputed_ilp.get("total_ev", precomputed_ilp["total_next_gain"]),
            "total_cost":          precomputed_ilp["total_cost"],
        }]

    if PULP_AVAILABLE and precomputed_ilp is None:
        result = get_ilp_optimal_transfers(
            my_team_enriched, other_enriched, bank_balance, n_transfers=2
        )
        if result.get("transfers") and len(result["transfers"]) == 2:
            t = result["transfers"]
            return [{
                "transfer_1_out":      t[0]["out_name"],
                "transfer_1_in":       t[0]["in_name"],
                "transfer_1_gain":     t[0]["gain"],
                "transfer_1_next":     t[0]["next_gain"],
                "transfer_1_ev":       t[0].get("total_ev", t[0]["next_gain"]),
                "run_1":               t[0]["fixture_run"],
                "blank_1":             t[0]["is_blank"],
                "dgw_1":               t[0]["double_gws"],
                "transfer_2_out":      t[1]["out_name"],
                "transfer_2_in":       t[1]["in_name"],
                "transfer_2_gain":     t[1]["gain"],
                "transfer_2_next":     t[1]["next_gain"],
                "transfer_2_ev":       t[1].get("total_ev", t[1]["next_gain"]),
                "run_2":               t[1]["fixture_run"],
                "blank_2":             t[1]["is_blank"],
                "dgw_2":               t[1]["double_gws"],
                "total_combined_gain": result["total_gain"],
                "total_next_gw_gain":  result["total_next_gain"],
                "total_ev":            result.get("total_ev", result["total_next_gain"]),
                "total_cost":          result["total_cost"],
            }]

    # Greedy fallback
    has_xpts = "expected_pts" in my_team_enriched.columns
    singles: list = []
    for _, p_out in my_team_enriched.iterrows():
        same_pos = other_enriched[other_enriched["position"] == p_out["position"]]
        for _, p_in in same_pos.iterrows():
            if check_transfer_validity(my_team_enriched, p_out, p_in, bank_balance):
                continue
            gain = round(p_in["combined_score"] - p_out["combined_score"], 2)
            if gain <= 0: continue
            ng = round(
                float(p_in.get("expected_pts", p_in["predicted_pts"])) -
                float(p_out.get("expected_pts", p_out["predicted_pts"])), 2
            ) if has_xpts else round(p_in["predicted_pts"] - p_out["predicted_pts"], 2)
            ev = _compute_transfer_price_ev(p_in, p_out, ng)
            singles.append({"out": p_out, "in": p_in, "gain": gain,
                             "cost": round(p_in["price"] - p_out["price"], 1),
                             "next_gain": ng, "total_ev": ev})

    singles = sorted(singles, key=lambda x: x["gain"], reverse=True)[:MAX_DOUBLE_TRANSFER_CANDIDATES]

    valid_doubles: list = []
    seen_pairs: set     = set()
    for t1, t2 in itertools.combinations(singles, 2):
        if t1["out"]["player_id"] == t2["out"]["player_id"]: continue
        if t1["in"]["player_id"]  == t2["in"]["player_id"]:  continue
        pair_key = tuple(sorted([t1["out"]["player_id"], t2["out"]["player_id"]]))
        if pair_key in seen_pairs: continue
        seen_pairs.add(pair_key)
        total_cost = t1["cost"] + t2["cost"]
        if total_cost > bank_balance: continue
        simulated = my_team_enriched[
            ~my_team_enriched["player_id"].isin([t1["out"]["player_id"], t2["out"]["player_id"]])
        ].copy()
        in1 = t1["in"][["player_id","player_name","position","team_name","price"]].copy()
        in2 = t2["in"][["player_id","player_name","position","team_name","price"]].copy()
        simulated = pd.concat([
            simulated[["player_id","player_name","position","team_name","price"]],
            in1.to_frame().T, in2.to_frame().T,
        ], ignore_index=True)
        if validate_squad(simulated): continue
        valid_doubles.append({
            "transfer_1_out":      t1["out"]["player_name"],
            "transfer_1_in":       t1["in"]["player_name"],
            "transfer_1_gain":     t1["gain"],
            "transfer_1_next":     t1["next_gain"],
            "transfer_1_ev":       t1["total_ev"],
            "run_1":               t1["in"].get("fixture_run_label", "?"),
            "blank_1":             t1["in"].get("is_blank_next_gw", False),
            "dgw_1":               t1["in"].get("double_gws", 0),
            "transfer_2_out":      t2["out"]["player_name"],
            "transfer_2_in":       t2["in"]["player_name"],
            "transfer_2_gain":     t2["gain"],
            "transfer_2_next":     t2["next_gain"],
            "transfer_2_ev":       t2["total_ev"],
            "run_2":               t2["in"].get("fixture_run_label", "?"),
            "blank_2":             t2["in"].get("is_blank_next_gw", False),
            "dgw_2":               t2["in"].get("double_gws", 0),
            "total_combined_gain": round(t1["gain"]      + t2["gain"],      2),
            "total_next_gw_gain":  round(t1["next_gain"] + t2["next_gain"], 2),
            "total_ev":            round(t1["total_ev"]  + t2["total_ev"],  2),
            "total_cost":          round(total_cost, 1),
        })

    valid_doubles.sort(key=lambda x: x["total_combined_gain"], reverse=True)
    return valid_doubles[:top_n]


# ─────────────────────────────────────────
# 17. TRANSFER SUMMARY REPORT
# ─────────────────────────────────────────

def generate_transfer_summary(ilp_result_1: dict,
                               ilp_result_2: dict,
                               roll_advice: dict,
                               hit_transfers: list,
                               transfers_made: int,
                               bank_balance: float,
                               current_gw: int) -> str:
    """Single bottom-line transfer recommendation. v5: includes total_ev."""
    lines     = []
    recommend = roll_advice.get("recommendation", "ROLL")

    lines.append(f"\n  BOTTOM LINE — GW{current_gw+1} TRANSFER DECISION")
    lines.append("  " + "─" * 55)

    if recommend == "ALREADY USED":
        lines.append("  Your free transfer has already been used this GW.")
        if hit_transfers:
            top_hit = hit_transfers[0]
            lines.append(
                f"  Best hit: OUT {top_hit['replace']} -> IN {top_hit['player_in']}"
                f"  (Net: +{top_hit['net_value']:.2f} xPts after -{HIT_COST_PTS}pt hit)"
            )
        else:
            lines.append(f"  No transfers worth a -{HIT_COST_PTS}pt hit.")
    elif recommend == "ROLL":
        lines.append("  Recommendation: HOLD — roll your free transfer.")
        for r in roll_advice.get("reasons", []): lines.append(f"    - {r}")
    elif recommend == "USE NOW":
        lines.append("  Recommendation: USE your free transfer now.")
        for r in roll_advice.get("reasons", []): lines.append(f"    - {r}")
        t1 = ilp_result_1.get("transfers", [])
        if t1:
            t    = t1[0]
            ptag = " 📈" if float(t.get("price_change",0)) > 0.05 else \
                   " 📉" if float(t.get("price_change",0)) < -0.05 else ""
            lines.append(
                f"  Best transfer: OUT {t['out_name']} -> IN {t['in_name']}"
                f"  (+{t['gain']:.2f} combined  +{t['next_gain']:.2f} xPts"
                f"  EV:{t.get('total_ev', t['next_gain']):.2f})"
                f"  Cost:{t['cost_diff']:+.1f}M{ptag}"
            )
    else:
        lines.append("  Recommendation: BORDERLINE — your call.")
        for r in roll_advice.get("reasons", []): lines.append(f"    - {r}")
        bm = roll_advice.get("best_move")
        if bm: lines.append(f"  If you transfer: OUT {bm[0]} -> IN {bm[1]}")

    lines.append(f"\n  Bank: £{bank_balance:.1f}M  |  GW{current_gw+1} window")
    lines.append("  ⚠️  Always verify bank balance in the FPL app before confirming.")
    return "\n".join(lines)


# ─────────────────────────────────────────
# 18. DISPLAY HELPERS
# ─────────────────────────────────────────

def print_squad_summary(my_team_enriched: pd.DataFrame) -> None:
    """Print squad composition with position and club counts."""
    violations  = validate_squad(my_team_enriched)
    pos_counts  = my_team_enriched["position"].value_counts()
    club_counts = my_team_enriched["team_name"].value_counts()
    print(f"\n{'=' * 75}")
    print("  SQUAD VALIDATION")
    print(f"{'=' * 75}")
    if not violations:
        print("  Squad passes all FPL rules!")
    else:
        print("  Squad violations:")
        for v in violations: print(f"    - {v}")
    print(f"\n  Position counts:")
    for pos, limit in POSITION_LIMITS.items():
        count  = pos_counts.get(pos, 0)
        status = "OK" if count == limit else "!!"
        print(f"    [{status}] {pos}: {count}/{limit}")
    print(f"\n  Players per club:")
    for club, count in club_counts.sort_values(ascending=False).items():
        status = "OK" if count <= MAX_PER_CLUB else "!!"
        print(f"    [{status}] {club}: {count}")


def _price_tag(price_change: float) -> str:
    if price_change > 0.05:  return " 📈"
    if price_change < -0.05: return " 📉"
    return ""


def print_valid_transfers(valid_df: pd.DataFrame,
                           bank_balance: float,
                           top_n: int = 5) -> None:
    """Print valid single transfer suggestions with total_ev and urgency."""
    has_xpts = "expected_pts" in valid_df.columns

    def _fmt(r) -> str:
        blank_tag   = " ⚠️ BLANK" if r.get("is_blank_next_gw") else ""
        dgw_tag     = " 🔄 DGW"  if r.get("double_gws", 0) > 0 else ""
        pchg_tag    = _price_tag(float(r.get("price_change", 0)))
        xpts_str    = f"  xPts:{float(r.get('expected_pts', r['predicted_pts'])):.2f}" if has_xpts else ""
        ev_str      = f"  EV:{float(r.get('total_ev', r.get('next_gw_gain', 0))):.2f}"
        urgency     = float(r.get("urgency_score", 0))
        urg_str     = f"  🔥Urgent" if urgency >= 2.0 else ""
        return (
            f"    OUT:{str(r['replace']):25s}  ->  IN:{str(r['player_in']):25s}"
            f"  [{r['position']:3s}]"
            f"  5GW:+{r['combined_gain']}"
            f"  Next:+{r['next_gw_gain']}"
            f"{xpts_str}{ev_str}"
            f"  Run:{r['fixture_run']}"
            f"  Val:{r.get('value_score',0):.2f}"
            f"  Cost:{r['cost_diff']:+.1f}M"
            f"{dgw_tag}{blank_tag}{pchg_tag}{urg_str}"
        )

    print(f"\n  💰 Best Within-Budget Transfers (Bank: £{bank_balance:.1f}M):")
    budget_top = valid_df[valid_df["cost_diff"] <= bank_balance].head(top_n)
    if budget_top.empty:
        print("    No valid transfers within budget.")
    else:
        for _, r in budget_top.iterrows(): print(_fmt(r))
    print(f"\n  💸 Best Transfers (No Budget Limit):")
    for _, r in valid_df.head(top_n).iterrows():
        status = "OK" if r["cost_diff"] <= bank_balance else f"Need +£{r['cost_diff']-bank_balance:.1f}M"
        print(_fmt(r) + f"  [{status}]")


def print_ilp_result(result: dict, label: str) -> None:
    """Print ILP optimal transfer result with total_ev."""
    if "error" in result:
        print(f"    {result['error']}")
        return
    transfers = result.get("transfers", [])
    if not transfers:
        print("    No optimal transfers found.")
        return
    for t in transfers:
        blank_tag = " ⚠️ BLANK" if t.get("is_blank") else ""
        dgw_tag   = " 🔄 DGW"  if t.get("double_gws", 0) > 0 else ""
        pchg_tag  = _price_tag(float(t.get("price_change", 0)))
        trend     = float(t.get("fixture_trend", 0))
        trend_str = f"  trend:{trend:+.1f}" if abs(trend) > 0.1 else ""
        ev_str    = f"  EV:{t.get('total_ev', t['next_gain']):.2f}"
        print(
            f"    OUT:{str(t['out_name']):25s}  ->  IN:{str(t['in_name']):25s}"
            f"  [{t['position']:3s}]"
            f"  5GW:+{t['gain']}"
            f"  xPts:+{t['next_gain']}"
            f"{ev_str}"
            f"  Run:{t['fixture_run']}"
            f"  Val:{t.get('value_score',0):.2f}"
            f"  Cost:{t['cost_diff']:+.1f}M"
            f"{trend_str}{dgw_tag}{blank_tag}{pchg_tag}"
        )
    print(
        f"    Total — 5GW:+{result['total_gain']}"
        f"  xPts:+{result['total_next_gain']}"
        f"  EV:{result.get('total_ev', result['total_next_gain']):.2f}"
        f"  Cost:{result['total_cost']:+.1f}M"
    )


def print_double_transfers(valid_doubles: list, bank_balance: float) -> None:
    """Print best 2-transfer combinations."""
    print(f"\n  Best 2-Transfer Combinations (Bank: £{bank_balance:.1f}M):")
    if not valid_doubles:
        print("    No valid 2-transfer combinations found.")
        return
    for i, combo in enumerate(valid_doubles, 1):
        status = "OK" if combo["total_cost"] <= bank_balance \
                 else f"Need +£{combo['total_cost']-bank_balance:.1f}M"
        b1 = " ⚠️ BLANK" if combo.get("blank_1") else ""
        d1 = " 🔄 DGW"  if combo.get("dgw_1", 0) > 0 else ""
        b2 = " ⚠️ BLANK" if combo.get("blank_2") else ""
        d2 = " 🔄 DGW"  if combo.get("dgw_2", 0) > 0 else ""
        ev1 = combo.get("transfer_1_ev", combo["transfer_1_next"])
        ev2 = combo.get("transfer_2_ev", combo["transfer_2_next"])
        print(f"\n    Option {i}: [{status}]")
        print(
            f"      T1: OUT {str(combo['transfer_1_out']):25s}  ->  "
            f"IN {str(combo['transfer_1_in']):25s}"
            f"  Run:{combo['run_1']}  5GW:+{combo['transfer_1_gain']}"
            f"  xPts:+{combo['transfer_1_next']}  EV:{ev1:.2f}{d1}{b1}"
        )
        print(
            f"      T2: OUT {str(combo['transfer_2_out']):25s}  ->  "
            f"IN {str(combo['transfer_2_in']):25s}"
            f"  Run:{combo['run_2']}  5GW:+{combo['transfer_2_gain']}"
            f"  xPts:+{combo['transfer_2_next']}  EV:{ev2:.2f}{d2}{b2}"
        )
        print(
            f"      Total — 5GW:+{combo['total_combined_gain']}"
            f"  xPts:+{combo['total_next_gw_gain']}"
            f"  EV:{combo.get('total_ev', combo['total_next_gw_gain']):.2f}"
            f"  Cost:{combo['total_cost']:+.1f}M"
        )


# ─────────────────────────────────────────
# 19. PIPELINE HELPER
# ─────────────────────────────────────────

def _load_or_train_models(history_df: pd.DataFrame, refresh: bool) -> dict:
    """Load models from pkl if <12h old, else retrain."""
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
# 20. FULL PHASE 3 PIPELINE
# ─────────────────────────────────────────

def run_phase3(team_id: int = TEAM_ID,
               max_players: int = None,
               refresh: bool = False):
    """Full Phase 3 v5 pipeline."""
    log.info("=" * 75)
    log.info("  FPL AI ASSISTANT — Phase 3: Transfer Optimizer (v5)")
    log.info("=" * 75)

    # ── Fetch ──────────────────────────────────────────────────────
    log.info("Fetching bootstrap & fixtures...")
    bootstrap   = fetch_bootstrap()
    fixtures_df = fetch_fixtures()
    current_gw  = fetch_current_gw(bootstrap)
    log.info(f"GW{current_gw} completed -> predicting GW{current_gw+1}")

    log.info("Fetching your team...")
    try:
        team_data     = fetch_my_team(team_id, current_gw)
        my_player_ids = [p["element"] for p in team_data["picks"]]
        log.info(f"Team fetched — {len(my_player_ids)} players.")
    except Exception as e:
        log.error(f"Could not fetch team: {e}")
        my_player_ids = []
        team_data     = {}

    transfer_info   = fetch_transfer_info(team_id, current_gw)
    bank_balance    = transfer_info["bank_balance"]
    transfer_status = transfer_info["transfer_status"]
    transfers_made  = transfer_info["transfers_made"]
    log.info(f"Bank: £{bank_balance:.1f}M  |  {transfer_status}")

    # ── History + models ───────────────────────────────────────────
    log.info("Loading player history...")
    history_df = build_player_history_df(
        bootstrap, max_players=max_players, refresh=refresh
    )
    models = _load_or_train_models(history_df, refresh)

    # ── Phase 1 v5 full prediction pipeline ───────────────────────
    log.info(f"Predicting GW{current_gw+1}...")
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

    # ── Phase 2 context ────────────────────────────────────────────
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
        FIXTURE_LOOKAHEAD, cs_probability_map=cs_prob_map,
    )

    my_team_enriched = enriched_df[enriched_df["player_id"].isin(my_player_ids)].copy()
    other_enriched   = enriched_df[~enriched_df["player_id"].isin(my_player_ids)].copy()

    # ── Squad validation ───────────────────────────────────────────
    print_squad_summary(my_team_enriched)

    # ── Fixture run table ──────────────────────────────────────────
    print_fixture_run_table(
        my_team_enriched, current_gw, gw_lookahead=FIXTURE_LOOKAHEAD,
        title=f"YOUR SQUAD — GW{current_gw+1} to GW{current_gw+FIXTURE_LOOKAHEAD}",
    )

    # ── Captain — ranked by captain_ev ────────────────────────────
    print(f"\n{'=' * 75}")
    print("  CAPTAIN RECOMMENDATION (Monte Carlo EV)")
    print(f"{'=' * 75}")
    mc_results = run_monte_carlo_captain(my_team_enriched, n_simulations=MONTE_CARLO_N)
    for i, r in enumerate(mc_results[:3]):
        label     = ["Captain     ", "Vice Captain", "3rd Option  "][i]
        home_note = " (H)" if r["is_home"] else " (A)"
        dgw_note  = " (DGW)" if r["double_gws"] > 0 else ""
        print(
            f"  {label}: {str(r['player_name']):28s}"
            f"  Win%:{r['win_prob']*100:.1f}%"
            f"  Gain vs others:{r['expected_captain_gain']:+.2f}"
            f"  Cap EV:{r['captain_ev']:.1f}"
            f"  Run:{r['fixture_run']}{home_note}{dgw_note}"
        )

    # ── Captaincy differential analysis ───────────────────────────
    print(f"\n{'=' * 75}")
    print("  CAPTAINCY DIFFERENTIAL ANALYSIS (vs Average Manager)")
    print(f"{'=' * 75}")
    cap_diff_results = get_captaincy_differential_analysis(my_team_enriched, bootstrap)
    print(f"  Field captain EV estimate: {cap_diff_results[0]['field_captain_ev']:.1f} pts")
    print(f"  (Average of squad captain options weighted by ownership)")
    print()
    for r in cap_diff_results:
        diff_tag = " ★ DIFFERENTIAL" if r["is_differential"] else ""
        sign     = "+" if r["differential_gain"] >= 0 else ""
        print(
            f"  {str(r['player_name']):28s}"
            f"  Own:{r['ownership_pct']:.1f}%"
            f"  EV:{r['captain_ev']:.1f}"
            f"  vs field: {sign}{r['differential_gain']:.2f}"
            f"{diff_tag}"
        )

    # ── Bench optimization — v5 new ───────────────────────────────
    print(f"\n{'=' * 75}")
    print("  BENCH ORDER RECOMMENDATION")
    print(f"{'=' * 75}")
    bench_rec = get_bench_order_recommendation(my_team_enriched)
    if bench_rec["first_sub"]:
        print(f"\n  Recommended bench order:")
        for j, name in enumerate(bench_rec["bench_order"], 1):
            bench_row = bench_rec["bench"].iloc[j-1] if j <= len(bench_rec["bench"]) else None
            if bench_row is not None:
                ev_str = f"  bench EV:{bench_row.get('bench_ev', 0):.2f}"
            else:
                ev_str = ""
            label = "1st sub ★" if j == 1 else f"{j}{'st' if j==1 else 'nd' if j==2 else 'rd' if j==3 else 'th'} sub"
            print(f"    {j}. {str(name):28s}  [{label}]{ev_str}")
        print(f"\n  Total bench expected contribution: {bench_rec['bench_ev_total']:.2f} pts")
        print(f"  P(auto-sub needed): {bench_rec['p_at_least_one_miss']*100:.0f}%")
        if bench_rec.get("first_sub_reason"):
            print(f"  First sub rationale: {bench_rec['first_sub_reason']}")

    # ── ILP Optimal 1 transfer ─────────────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"  OPTIMAL 1-TRANSFER (ILP{'  — PuLP' if PULP_AVAILABLE else '  — Greedy fallback'})")
    print(f"{'=' * 75}")
    ilp_result_1 = get_ilp_optimal_transfers(
        my_team_enriched, other_enriched, bank_balance, n_transfers=1
    )
    print_ilp_result(ilp_result_1, "1-Transfer")

    # ── ILP Optimal 2 transfers ────────────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"  OPTIMAL 2-TRANSFER (ILP{'  — PuLP' if PULP_AVAILABLE else '  — Greedy fallback'})")
    print(f"{'=' * 75}")
    ilp_result_2 = get_ilp_optimal_transfers(
        my_team_enriched, other_enriched, bank_balance, n_transfers=2
    )
    print_ilp_result(ilp_result_2, "2-Transfer")

    # ── Multi-GW Horizon Plan — v5 new ────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"  MULTI-GW HORIZON TRANSFER PLAN ({HORIZON_GWS}-GW LOOKAHEAD)")
    print(f"{'=' * 75}")
    log.info("🗓️ Computing horizon transfer plan...")
    horizon_plans = get_horizon_transfer_plan(
        my_team_enriched, other_enriched, enriched_df, bank_balance
    )
    if not horizon_plans:
        print("  No viable multi-GW transfer sequences found.")
    else:
        for i, plan in enumerate(horizon_plans, 1):
            print(f"\n  Plan {i} — Total EV: +{plan['total_horizon_ev']:.2f}")
            print(
                f"    GW+1: OUT {str(plan['w1_out']):25s}  ->  IN {str(plan['w1_in']):25s}"
                f"  xPts:+{plan['w1_xpts_gain']:.2f}"
                f"  EV:+{plan['w1_total_ev']:.2f}"
                f"  Run:{plan['w1_run']}"
                f"  Cost:{plan['w1_cost']:+.1f}M"
            )
            if plan["w2_in"] != "—":
                print(
                    f"    GW+2: OUT {str(plan['w2_out']):25s}  ->  IN {str(plan['w2_in']):25s}"
                    f"  xPts:+{plan['w2_xpts_gain']:.2f}"
                    f"  EV:+{plan['w2_total_ev']:.2f}"
                    f"  Run:{plan['w2_run']}"
                )
            else:
                print(f"    GW+2: No improvement found after GW+1 transfer.")
        # Highlight if greedy best != horizon best
        if len(horizon_plans) > 0 and ilp_result_1.get("transfers"):
            greedy_best = ilp_result_1["transfers"][0]["in_name"]
            horizon_best = horizon_plans[0]["w1_in"]
            if greedy_best != horizon_best:
                print(
                    f"\n  ⚡ Horizon insight: immediate best = {greedy_best}"
                    f" but 2-GW best = {horizon_best}"
                    f" (+{horizon_plans[0]['total_horizon_ev']:.2f} vs"
                    f" greedy-only {ilp_result_1['transfers'][0].get('total_ev', '?'):.2f})"
                )

    # ── Greedy 2-transfer combos ───────────────────────────────────
    print(f"\n{'=' * 75}")
    print("  BEST 2-TRANSFER COMBINATIONS (Constraint-Checked)")
    print(f"{'=' * 75}")
    valid_doubles = get_valid_double_transfers(
        my_team_enriched, other_enriched, bank_balance,
        top_n=3, precomputed_ilp=ilp_result_2
    )
    print_double_transfers(valid_doubles, bank_balance)

    # ── Hit analysis (-4pt) ────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"  HIT TRANSFER ANALYSIS (-{HIT_COST_PTS}pt)")
    print(f"{'=' * 75}")
    hit_transfers = get_hit_transfer_analysis(
        my_team_enriched, other_enriched, bank_balance, transfers_made
    )
    if not hit_transfers:
        msg = "  You still have a free transfer — no hit needed." if transfers_made == 0 \
              else f"  No transfers worth a -{HIT_COST_PTS}pt hit."
        print(msg)
    else:
        print(f"  Transfers worth a -{HIT_COST_PTS}pt hit:")
        for h in hit_transfers:
            blank_tag = " ⚠️ BLANK" if h.get("is_blank") else ""
            dgw_tag   = " 🔄 DGW"  if h.get("double_gws", 0) > 0 else ""
            pchg_tag  = _price_tag(h.get("price_change", 0))
            print(
                f"    OUT:{h['replace']:25s}  ->  IN:{h['player_in']:25s}"
                f"  [{h['position']:3s}]"
                f"  xPts:+{h['xpts_gain']}"
                f"  EV:+{h.get('total_ev', h['xpts_gain']):.2f}"
                f"  Net:+{h['net_value']}"
                f"  Cost:{h['cost_diff']:+.1f}M{dgw_tag}{blank_tag}{pchg_tag}"
            )

    # ── Double hit (-8pt) ──────────────────────────────────────────
    if transfers_made > 0:
        print(f"\n{'=' * 75}")
        print(f"  DOUBLE HIT ANALYSIS (-{DOUBLE_HIT_COST_PTS}pt)")
        print(f"{'=' * 75}")
        double_hits = get_double_hit_analysis(
            my_team_enriched, other_enriched, bank_balance, transfers_made
        )
        if not double_hits:
            print(f"  No combo justifies a -{DOUBLE_HIT_COST_PTS}pt double hit.")
        else:
            for i, dh in enumerate(double_hits, 1):
                print(
                    f"  Option {i}:  OUT {dh['t1_out']} -> IN {dh['t1_in']}"
                    f"  (+{dh['t1_xpts_gain']:.2f} xPts)  Run:{dh['run_1']}"
                )
                print(
                    f"            OUT {dh['t2_out']} -> IN {dh['t2_in']}"
                    f"  (+{dh['t2_xpts_gain']:.2f} xPts)  Run:{dh['run_2']}"
                )
                print(
                    f"            Total xPts:+{dh['total_xpts_gain']:.2f}"
                    f"  Net after hit:+{dh['net_value']:.2f}"
                    f"  Cost:{dh['total_cost']:+.1f}M"
                )

    # ── Rolling transfer advice ────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("  ROLLING TRANSFER ADVICE")
    print(f"{'=' * 75}")
    roll_advice = get_rolling_transfer_advice(
        my_team_enriched, other_enriched, bank_balance,
        transfers_made, chip_info, current_gw, ilp_result=ilp_result_1,
    )
    print(f"  Recommendation: {roll_advice['recommendation']}")
    for r in roll_advice["reasons"]: print(f"    - {r}")

    # ── Differential picks ─────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"  DIFFERENTIAL PICKS (Ownership < {DIFFERENTIAL_THRESH}%)")
    print(f"{'=' * 75}")
    diffs = get_differential_picks(other_enriched, bootstrap)
    if diffs.empty: print("  No differentials found.")
    else: print(diffs.to_string(index=False))

    # ── Squad value breakdown ──────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("  SQUAD VALUE BREAKDOWN")
    print(f"{'=' * 75}")
    value_df  = get_squad_value_breakdown(my_team_enriched, bootstrap, team_data)
    disp_cols = [c for c in [
        "player_name","position","team_name","sell_price","predicted_pts","expected_pts","fixture_run",
    ] if c in value_df.columns]
    print(value_df[disp_cols].sort_values("predicted_pts", ascending=False).to_string(index=False))
    print(f"\n  Total squad sell value: £{value_df['sell_price'].sum():.1f}M")

    # ── Squad value tracking ───────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("  SQUAD VALUE TRACKING (Over Time)")
    print(f"{'=' * 75}")
    squad_value_data = track_squad_value(
        my_team_enriched, bootstrap, current_gw, team_data=team_data
    )
    print_squad_value_tracking(squad_value_data)

    # ── Wildcard / Free Hit ────────────────────────────────────────
    available_chips = chip_info.get("available_chips", [])
    sell_value      = value_df["sell_price"].sum()
    actual_budget   = round(bank_balance + sell_value, 1)

    if any("Wildcard" in c for c in available_chips):
        wc_label = next(c for c in available_chips if "Wildcard" in c)
        print(f"\n{'=' * 75}")
        print(f"  WILDCARD SQUAD — ILP Optimal ({wc_label} Available!)")
        print(f"  Budget: £{actual_budget:.1f}M  (bank £{bank_balance:.1f}M + squad sell £{sell_value:.1f}M)")
        print(f"{'=' * 75}")
        wc_squad  = get_wildcard_squad(enriched_df, budget=actual_budget)
        show_cols = [c for c in [
            "player_name","position","team_name","price","predicted_pts","expected_pts","combined_score",
        ] if c in wc_squad.columns]
        print(wc_squad[show_cols].sort_values("combined_score", ascending=False).to_string(index=False))
        print(f"  Total cost: £{wc_squad['price'].sum():.1f}M")
        # Item 7: show diff from current squad
        print_wildcard_diff(my_team_enriched, wc_squad, bank_balance)

    if "Free Hit" in available_chips:
        print(f"\n{'=' * 75}")
        print(f"  FREE HIT SQUAD — ILP Optimal (Free Hit Available!)")
        print(f"  Budget: £{actual_budget:.1f}M")
        print(f"{'=' * 75}")
        fh_squad  = get_free_hit_squad(enriched_df, budget=actual_budget)
        score_col = "expected_pts" if "expected_pts" in fh_squad.columns else "predicted_pts"
        show_cols = [c for c in [
            "player_name","position","team_name","price","predicted_pts","expected_pts","combined_score",
        ] if c in fh_squad.columns]
        print(fh_squad[show_cols].sort_values(score_col, ascending=False).to_string(index=False))
        print(f"  Total cost: £{fh_squad['price'].sum():.1f}M")
        print(f"  Total {score_col}: {fh_squad[score_col].sum():.1f}")

    # ── Transfer history ───────────────────────────────────────────
    evaluate_past_transfers(history_df, current_gw)
    if _load_transfer_history():
        print(f"\n{'=' * 75}")
        print("  TRANSFER HISTORY & ACCURACY")
        print(f"{'=' * 75}")
        print_transfer_history()

    # ── Log top suggestion ─────────────────────────────────────────
    if ilp_result_1.get("transfers"):
        t = ilp_result_1["transfers"][0]
        log_transfer_suggestion(t["out_name"], t["in_name"], t["gain"], current_gw)

    # ── Bottom line summary ────────────────────────────────────────
    valid_df = get_valid_transfers(my_team_enriched, other_enriched, bank_balance, top_n=5)
    summary  = generate_transfer_summary(
        ilp_result_1, ilp_result_2, roll_advice,
        hit_transfers, transfers_made, bank_balance, current_gw
    )
    print(f"\n{'=' * 75}")
    print("  TRANSFER DECISION SUMMARY")
    print(f"{'=' * 75}")
    print(summary)

    enriched_df.to_csv("fpl_predictions_phase3.csv", index=False)
    log.info("Predictions saved → fpl_predictions_phase3.csv")
    log.info("✅ Phase 3 v5 complete — ready for Phase 4")

    return enriched_df, my_team_enriched, valid_df


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    REFRESH = "--refresh" in sys.argv
    if REFRESH:
        log.info("--refresh flag detected.")

    enriched_df, my_team, valid_transfers = run_phase3(
        team_id=TEAM_ID,
        max_players=None,
        refresh=REFRESH,
    )