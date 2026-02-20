"""
FPL AI Assistant — Phase 3: Team Composition Constraints
=========================================================
Builds on Phase 2 to enforce proper FPL squad rules:
  - Max 3 players per club in your 15-man squad
  - Position slot limits: 2 GK, 5 DEF, 5 MID, 3 FWD
  - Budget hard cap (bank balance)
  - No suggesting players already in your squad
  - Validates your CURRENT squad against FPL rules
  - Shows which constraints would be violated per suggestion
  - Filters all transfer suggestions through constraint checker
  - Best 1-transfer and best 2-transfer combinations

Run normally (uses cache):
  python fpl_phase3_constraints.py

Force fresh data:
  python fpl_phase3_constraints.py --refresh
"""

import os
import sys
import pickle
import itertools
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
    train_model,
    FEATURE_COLS,
)
from fpl_phase2_fixtures import (
    build_fixture_run,
    build_player_fixture_scores,
    print_fixture_run_table,
)

# ─────────────────────────────────────────
# 1. FPL SQUAD RULES
# ─────────────────────────────────────────

POSITION_LIMITS = {
    "Goalkeeper": 2,
    "Defender":   5,
    "Midfielder": 5,
    "Forward":    3,
}
MAX_PER_CLUB  = 3
SQUAD_SIZE    = 15


def validate_squad(squad_df: pd.DataFrame) -> list:
    """
    Check a 15-man squad against FPL rules.
    Returns a list of violation messages (empty = valid squad).
    """
    violations = []

    # Squad size
    if len(squad_df) != SQUAD_SIZE:
        violations.append(f"Squad has {len(squad_df)} players (must be {SQUAD_SIZE})")

    # Position limits
    pos_counts = squad_df["position"].value_counts()
    for pos, limit in POSITION_LIMITS.items():
        count = pos_counts.get(pos, 0)
        if count != limit:
            violations.append(f"{pos}: {count} players (must be {limit})")

    # Club limits
    club_counts = squad_df["team_name"].value_counts()
    over_limit  = club_counts[club_counts > MAX_PER_CLUB]
    for club, count in over_limit.items():
        violations.append(f"{club}: {count} players (max {MAX_PER_CLUB})")

    return violations


def check_transfer_validity(squad_df: pd.DataFrame,
                             player_out: pd.Series,
                             player_in: pd.Series,
                             bank_balance: float) -> list:
    """
    Check if a single transfer is valid under FPL rules.
    Returns list of violation messages (empty = valid transfer).
    """
    violations = []

    # Same position
    if player_out["position"] != player_in["position"]:
        violations.append(
            f"Position mismatch: {player_out['position']} -> {player_in['position']}"
        )

    # Budget
    cost_diff = player_in["price"] - player_out["price"]
    if cost_diff > bank_balance:
        violations.append(
            f"Over budget by £{cost_diff - bank_balance:.1f}M "
            f"(need £{cost_diff:.1f}M, have £{bank_balance:.1f}M)"
        )

    # Club limit — simulate squad after transfer
    simulated = squad_df[squad_df["player_id"] != player_out["player_id"]].copy()
    club_counts = simulated["team_name"].value_counts()
    incoming_club_count = club_counts.get(player_in["team_name"], 0)
    if incoming_club_count >= MAX_PER_CLUB:
        violations.append(
            f"Club limit: already have {incoming_club_count} players from {player_in['team_name']}"
        )

    # Not already in squad
    if player_in["player_id"] in squad_df["player_id"].values:
        violations.append(f"{player_in['player_name']} is already in your squad")

    return violations


# ─────────────────────────────────────────
# 2. VALID SINGLE TRANSFER SUGGESTIONS
# ─────────────────────────────────────────

def get_valid_transfers(my_team_enriched: pd.DataFrame,
                         other_enriched: pd.DataFrame,
                         bank_balance: float,
                         top_n: int = 5) -> pd.DataFrame:
    """
    Generate transfer suggestions that pass ALL FPL constraint checks.
    Ranked by 5GW combined score gain.
    """
    valid_suggestions = []

    for _, player_out in my_team_enriched.iterrows():
        same_pos = other_enriched[
            other_enriched["position"] == player_out["position"]
        ].copy()

        for _, player_in in same_pos.iterrows():
            violations = check_transfer_validity(
                my_team_enriched, player_out, player_in, bank_balance
            )

            if not violations:
                gain      = round(player_in["combined_score"] - player_out["combined_score"], 2)
                next_gain = round(player_in["predicted_pts"]  - player_out["predicted_pts"],  2)
                cost_diff = round(player_in["price"] - player_out["price"], 1)

                if gain > 0:
                    valid_suggestions.append({
                        "replace":            player_out["player_name"],
                        "replace_id":         player_out["player_id"],
                        "player_in":          player_in["player_name"],
                        "player_in_id":       player_in["player_id"],
                        "position":           player_in["position"],
                        "team_in":            player_in["team_name"],
                        "price_in":           player_in["price"],
                        "cost_diff":          cost_diff,
                        "combined_gain":      gain,
                        "next_gw_gain":       next_gain,
                        "fixture_run":        player_in["fixture_run_label"],
                        "predicted_pts":      player_in["predicted_pts"],
                        "combined_score":     player_in["combined_score"],
                        "avg_difficulty":     player_in["avg_difficulty"],
                    })

    if not valid_suggestions:
        return pd.DataFrame()

    sug_df = pd.DataFrame(valid_suggestions).sort_values(
        "combined_gain", ascending=False
    ).drop_duplicates("player_in").head(top_n * 3)

    return sug_df


# ─────────────────────────────────────────
# 3. VALID 2-TRANSFER COMBINATIONS
# ─────────────────────────────────────────

def get_valid_double_transfers(my_team_enriched: pd.DataFrame,
                                other_enriched: pd.DataFrame,
                                bank_balance: float,
                                top_n: int = 3) -> list:
    """
    Find the best valid 2-transfer combinations.
    Checks both transfers are valid simultaneously (club limits, budget, etc).
    """
    # First get all valid single transfers
    single_transfers = []
    for _, player_out in my_team_enriched.iterrows():
        same_pos = other_enriched[other_enriched["position"] == player_out["position"]]
        for _, player_in in same_pos.iterrows():
            violations = check_transfer_validity(
                my_team_enriched, player_out, player_in, bank_balance
            )
            if not violations:
                gain = round(player_in["combined_score"] - player_out["combined_score"], 2)
                if gain > 0:
                    single_transfers.append({
                        "out":      player_out,
                        "in":       player_in,
                        "gain":     gain,
                        "cost":     round(player_in["price"] - player_out["price"], 1),
                        "next_gain": round(player_in["predicted_pts"] - player_out["predicted_pts"], 2),
                    })

    # Sort by gain, take top candidates to limit combinations
    single_transfers = sorted(single_transfers, key=lambda x: x["gain"], reverse=True)[:20]

    valid_doubles = []
    seen_pairs    = set()

    for t1, t2 in itertools.combinations(single_transfers, 2):
        # Can't transfer same player out twice
        if t1["out"]["player_id"] == t2["out"]["player_id"]:
            continue
        # Can't bring in same player twice
        if t1["in"]["player_id"] == t2["in"]["player_id"]:
            continue

        # Deduplicate pairs
        pair_key = tuple(sorted([t1["out"]["player_id"], t2["out"]["player_id"]]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        # Combined cost check
        total_cost = t1["cost"] + t2["cost"]
        if total_cost > bank_balance:
            continue

        # Simulate squad after both transfers
        simulated = my_team_enriched[
            ~my_team_enriched["player_id"].isin([
                t1["out"]["player_id"],
                t2["out"]["player_id"]
            ])
        ].copy()

        # Add both incoming players temporarily
        in1 = t1["in"][["player_id", "player_name", "position", "team_name", "price"]].copy()
        in2 = t2["in"][["player_id", "player_name", "position", "team_name", "price"]].copy()
        simulated = pd.concat([
            simulated[["player_id", "player_name", "position", "team_name", "price"]],
            in1.to_frame().T,
            in2.to_frame().T
        ], ignore_index=True)

        # Validate simulated squad
        violations = validate_squad(simulated)
        if violations:
            continue

        total_gain      = round(t1["gain"] + t2["gain"], 2)
        total_next_gain = round(t1["next_gain"] + t2["next_gain"], 2)

        valid_doubles.append({
            "transfer_1_out":  t1["out"]["player_name"],
            "transfer_1_in":   t1["in"]["player_name"],
            "transfer_1_gain": t1["gain"],
            "transfer_2_out":  t2["out"]["player_name"],
            "transfer_2_in":   t2["in"]["player_name"],
            "transfer_2_gain": t2["gain"],
            "total_combined_gain": total_gain,
            "total_next_gw_gain":  total_next_gain,
            "total_cost":          round(total_cost, 1),
            "run_1":               t1["in"]["fixture_run_label"],
            "run_2":               t2["in"]["fixture_run_label"],
        })

    valid_doubles.sort(key=lambda x: x["total_combined_gain"], reverse=True)
    return valid_doubles[:top_n]


# ─────────────────────────────────────────
# 4. DISPLAY HELPERS
# ─────────────────────────────────────────

def print_valid_transfers(valid_df: pd.DataFrame,
                           bank_balance: float,
                           top_n: int = 5):
    """Print valid single transfer suggestions."""
    print(f"\n💰 Best Valid Transfers (Bank: £{bank_balance:.1f}M):")

    budget_transfers = valid_df[valid_df["cost_diff"] <= bank_balance].head(top_n)
    if budget_transfers.empty:
        print("  No valid transfers found within budget.")
    else:
        for _, r in budget_transfers.iterrows():
            print(
                f"  OUT: {str(r['replace']):25s}  ->  IN: {str(r['player_in']):25s}"
                f"  [{r['position']:3s}]"
                f"  5GW: +{r['combined_gain']}"
                f"  Next GW: +{r['next_gw_gain']}"
                f"  Run: {r['fixture_run']}"
                f"  Cost: {r['cost_diff']:+.1f}M"
                f"  ✅ Valid"
            )

    print(f"\n💸 Best Valid Transfers (No Budget Limit):")
    all_transfers = valid_df.head(top_n)
    for _, r in all_transfers.iterrows():
        affordable = "✅ Affordable" if r["cost_diff"] <= bank_balance else f"⚠️  Need £{r['cost_diff'] - bank_balance:.1f}M more"
        print(
            f"  OUT: {str(r['replace']):25s}  ->  IN: {str(r['player_in']):25s}"
            f"  [{r['position']:3s}]"
            f"  5GW: +{r['combined_gain']}"
            f"  Next GW: +{r['next_gw_gain']}"
            f"  Run: {r['fixture_run']}"
            f"  Cost: {r['cost_diff']:+.1f}M"
            f"  {affordable}"
        )


def print_double_transfers(valid_doubles: list, bank_balance: float):
    """Print best 2-transfer combinations."""
    print(f"\n🔄 Best 2-Transfer Combinations (Bank: £{bank_balance:.1f}M):")

    if not valid_doubles:
        print("  No valid 2-transfer combinations found within budget.")
        return

    for i, combo in enumerate(valid_doubles, 1):
        affordable = combo["total_cost"] <= bank_balance
        status     = "✅ Affordable" if affordable else f"⚠️  Need £{combo['total_cost'] - bank_balance:.1f}M more"
        print(f"\n  Option {i}: {status}")
        print(f"    Transfer 1: OUT {combo['transfer_1_out']:25s}  ->  IN {combo['transfer_1_in']:25s}  Run: {combo['run_1']}  5GW: +{combo['transfer_1_gain']}")
        print(f"    Transfer 2: OUT {combo['transfer_2_out']:25s}  ->  IN {combo['transfer_2_in']:25s}  Run: {combo['run_2']}  5GW: +{combo['transfer_2_gain']}")
        print(f"    Total 5GW Gain: +{combo['total_combined_gain']}  |  Next GW Gain: +{combo['total_next_gw_gain']}  |  Total Cost: {combo['total_cost']:+.1f}M")


# ─────────────────────────────────────────
# 5. FULL PHASE 3 PIPELINE
# ─────────────────────────────────────────

def run_phase3(team_id: int, max_players: int = None, refresh: bool = False):
    """
    Full Phase 3 pipeline:
      1. Run Phase 1 + 2 data pipeline
      2. Validate your current squad against FPL rules
      3. Generate valid single transfer suggestions
      4. Generate valid 2-transfer combinations
      5. Display everything with constraint status
    """
    print("=" * 75)
    print("  FPL AI ASSISTANT — Phase 3: Squad Constraints & Transfer Optimizer")
    print("=" * 75)

    # ── Fetch data ─────────────────────────────────────
    print("\n⬇️  Fetching data...")
    bootstrap   = fetch_bootstrap()
    fixtures_df = fetch_fixtures()

    current_gw = fetch_current_gw(bootstrap)
    print(f"📅 Last completed GW: {current_gw}  ->  Predicting GW{current_gw+1} to GW{current_gw+5}")

    print("⬇️  Fetching your team...")
    try:
        team_data     = fetch_my_team(team_id, current_gw)
        my_player_ids = [p["element"] for p in team_data["picks"]]
        print("✅ Team fetched.")
    except Exception as e:
        print(f"⚠️  Could not fetch team: {e}")
        my_player_ids = []

    transfer_info   = fetch_transfer_info(team_id, current_gw)
    bank_balance    = transfer_info["bank_balance"]
    transfer_status = transfer_info["transfer_status"]
    print(f"💰 Bank: £{bank_balance:.1f}M  |  Transfers: {transfer_status}")

    # ── Player history + model ─────────────────────────
    print(f"\n📚 Loading player history...")
    history_df = build_player_history_df(bootstrap, max_players=max_players, refresh=refresh)

    print("\n🤖 Training model...")
    model, pos_enc, opp_enc, _ = train_model(history_df)

    with open("fpl_model.pkl", "wb") as f:
        pickle.dump({
            "model": model, "pos_enc": pos_enc,
            "opp_enc": opp_enc, "features": FEATURE_COLS
        }, f)

    # ── Predict + fixture run ──────────────────────────
    print(f"\n🔮 Predicting GW{current_gw+1} scores...")
    pred_df = build_current_features(
        bootstrap, fixtures_df, history_df, pos_enc, opp_enc, current_gw
    )
    pred_df["predicted_pts"] = model.predict(pred_df[FEATURE_COLS]).round(2)
    pred_df["predicted_pts"] = pred_df["predicted_pts"].clip(lower=0)

    print("\n📆 Building fixture run...")
    fixture_run_df = build_fixture_run(bootstrap, fixtures_df, current_gw, gw_lookahead=5)
    enriched_df    = build_player_fixture_scores(pred_df, fixture_run_df, current_gw, gw_lookahead=5)

    my_team_enriched = enriched_df[enriched_df["player_id"].isin(my_player_ids)].copy()
    other_enriched   = enriched_df[~enriched_df["player_id"].isin(my_player_ids)].copy()

    # ── Validate current squad ─────────────────────────
    print(f"\n{'=' * 75}")
    print("  SQUAD VALIDATION")
    print(f"{'=' * 75}")

    violations = validate_squad(my_team_enriched)
    if not violations:
        print("✅ Your squad passes all FPL rules!")
    else:
        print("⚠️  Squad violations found:")
        for v in violations:
            print(f"  ❌ {v}")

    # Show squad composition summary
    print(f"\n📋 Squad Composition:")
    pos_counts  = my_team_enriched["position"].value_counts()
    club_counts = my_team_enriched["team_name"].value_counts()
    for pos, limit in POSITION_LIMITS.items():
        count = pos_counts.get(pos, 0)
        status = "✅" if count == limit else "❌"
        print(f"  {status} {pos}: {count}/{limit}")
    print(f"\n🏟️  Players per club:")
    for club, count in club_counts.sort_values(ascending=False).items():
        status = "✅" if count <= MAX_PER_CLUB else "❌"
        print(f"  {status} {club}: {count}")

    # ── Fixture run table ──────────────────────────────
    print_fixture_run_table(
        my_team_enriched, current_gw,
        title=f"YOUR SQUAD — GW{current_gw+1} to GW{current_gw+5}"
    )
    print("\n  Legend: B = Blank GW   D = Double GW   Number = Difficulty (1=Easy, 5=Hard)")

    # ── Captain pick ───────────────────────────────────
    print(f"\n{'=' * 75}")
    print("  CAPTAIN RECOMMENDATION")
    print(f"{'=' * 75}")
    labels = ["🏆 Captain     ", "🥈 Vice Captain", "🥉 3rd Option  "]
    for i, (_, row) in enumerate(my_team_enriched.nlargest(3, "combined_score").iterrows()):
        print(
            f"{labels[i]}: {str(row['player_name']):25s}"
            f"  Next GW: {row['predicted_pts']} pts"
            f"  5GW Run: {row['fixture_run_label']}"
            f"  Score: {row['combined_score']}"
        )

    # ── Valid single transfers ─────────────────────────
    print(f"\n{'=' * 75}")
    print("  BEST 1-TRANSFER (Constraint-Checked)")
    print(f"{'=' * 75}")
    print("  ✅ All suggestions below are valid under FPL rules\n")

    valid_df = get_valid_transfers(my_team_enriched, other_enriched, bank_balance, top_n=5)
    if valid_df.empty:
        print("  No valid transfers found.")
    else:
        print_valid_transfers(valid_df, bank_balance, top_n=5)

    # ── Valid 2-transfer combinations ─────────────────
    print(f"\n{'=' * 75}")
    print("  BEST 2-TRANSFER COMBINATIONS (Constraint-Checked)")
    print(f"{'=' * 75}")
    print("  ✅ All combinations below are valid under FPL rules\n")

    valid_doubles = get_valid_double_transfers(
        my_team_enriched, other_enriched, bank_balance, top_n=3
    )
    print_double_transfers(valid_doubles, bank_balance)

    # ── API bank warning ───────────────────────────────
    print(f"\n⚠️  Note: Bank shown (£{bank_balance:.1f}M) may differ slightly from the FPL app.")
    print(f"   Always double-check before confirming a transfer.")

    # ── Save ───────────────────────────────────────────
    enriched_df.to_csv("fpl_predictions_phase3.csv", index=False)
    print("\n✅ Predictions saved to fpl_predictions_phase3.csv")
    print("✅ Ready for Phase 4 (Starting XI + Formation Optimizer)")

    return enriched_df, my_team_enriched, valid_df


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    MY_TEAM_ID = 9179961   # <- Your FPL team ID

    REFRESH = "--refresh" in sys.argv
    if REFRESH:
        print("🔄 --refresh flag detected. Fetching fresh data.\n")

    enriched_df, my_team, valid_transfers = run_phase3(
        team_id=MY_TEAM_ID,
        max_players=None,
        refresh=REFRESH
    )