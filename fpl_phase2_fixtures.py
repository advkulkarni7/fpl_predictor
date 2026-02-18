"""
FPL AI Assistant — Phase 2: Fixture Run Analysis
=================================================
Builds on Phase 1 predictions to add:
  - Next 5 GW fixture difficulty scores per player
  - Fixture run rating (avg difficulty, colour coded)
  - Blank GW detection (no fixture that week)
  - Double GW detection (two fixtures that week)
  - Combined score: next GW prediction + 5GW fixture outlook
  - Updated transfer suggestions using 5-GW outlook, not just next GW
  - Top players by position with fixture context
  - Saves enriched predictions to fpl_predictions_phase2.csv

Run normally (uses player history cache):
  python fpl_phase2_fixtures.py

Force fresh data from API:
  python fpl_phase2_fixtures.py --refresh
"""

import os
import sys
import pickle
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

# ─────────────────────────────────────────
# 1. FIXTURE RUN BUILDER
# ─────────────────────────────────────────

def build_fixture_run(bootstrap: dict, fixtures_df: pd.DataFrame,
                      current_gw: int, gw_lookahead: int = 5) -> pd.DataFrame:
    """
    For every team, build a fixture table covering the next N gameweeks.
    Each row = one team, one GW, with:
      - difficulty (1-5, or 6 for blank)
      - opponent name
      - is_home flag
      - is_blank flag
      - is_double flag (two fixtures that GW)
    """
    teams_df = pd.DataFrame(bootstrap["teams"])
    team_map = teams_df.set_index("id")["name"].to_dict()

    gws  = range(current_gw + 1, current_gw + 1 + gw_lookahead)
    rows = []

    for gw in gws:
        gw_fixtures = fixtures_df[fixtures_df["event"] == gw]

        # Collect all fixtures per team this GW
        team_fixture_map = {}
        for _, fix in gw_fixtures.iterrows():
            for team, opp, diff, home in [
                (fix["team_h"], fix["team_a"], fix["team_h_difficulty"], True),
                (fix["team_a"], fix["team_h"], fix["team_a_difficulty"], False),
            ]:
                if team not in team_fixture_map:
                    team_fixture_map[team] = []
                team_fixture_map[team].append({
                    "team_id":    team,
                    "team_name":  team_map.get(team, "Unknown"),
                    "gw":         gw,
                    "opponent":   team_map.get(opp, "Unknown"),
                    "difficulty": diff,
                    "is_home":    int(home),
                    "is_blank":   False,
                    "is_double":  False,
                })

        # Every team gets a row — blank if no fixture
        for team_id, team_name in team_map.items():
            fixtures_this_gw = team_fixture_map.get(team_id, [])

            if not fixtures_this_gw:
                # Blank GW — penalise heavily
                rows.append({
                    "team_id":    team_id,
                    "team_name":  team_name,
                    "gw":         gw,
                    "opponent":   "BLANK",
                    "difficulty": 6,
                    "is_home":    0,
                    "is_blank":   True,
                    "is_double":  False,
                })
            elif len(fixtures_this_gw) >= 2:
                # Double GW — avg difficulty, list both opponents
                avg_diff = round(np.mean([f["difficulty"] for f in fixtures_this_gw]), 1)
                opps     = " & ".join(f["opponent"] for f in fixtures_this_gw)
                rows.append({
                    "team_id":    team_id,
                    "team_name":  team_name,
                    "gw":         gw,
                    "opponent":   opps,
                    "difficulty": avg_diff,
                    "is_home":    fixtures_this_gw[0]["is_home"],
                    "is_blank":   False,
                    "is_double":  True,
                })
            else:
                rows.append(fixtures_this_gw[0])

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 2. ENRICH PREDICTIONS WITH FIXTURE RUN
# ─────────────────────────────────────────

def build_player_fixture_scores(pred_df: pd.DataFrame,
                                 fixture_run_df: pd.DataFrame,
                                 current_gw: int,
                                 gw_lookahead: int = 5) -> pd.DataFrame:
    """
    Attach fixture run data to every player's prediction row.
    Adds per-GW difficulty columns, avg_difficulty, blank/double counts,
    fixture run label, and combined_score blending next GW pts + run outlook.
    """
    gws = list(range(current_gw + 1, current_gw + 1 + gw_lookahead))

    # Pivot fixture run: one row per team with GW columns
    pivot_rows = []
    for team_id, group in fixture_run_df.groupby("team_id"):
        row     = {"team_id": team_id}
        blanks  = 0
        doubles = 0
        diffs   = []

        for gw in gws:
            gw_row = group[group["gw"] == gw]
            if gw_row.empty:
                row[f"gw{gw}_difficulty"] = 6
                row[f"gw{gw}_opponent"]   = "BLANK"
                blanks += 1
            else:
                r = gw_row.iloc[0]
                row[f"gw{gw}_difficulty"] = r["difficulty"]
                row[f"gw{gw}_opponent"]   = r["opponent"]
                if r["is_blank"]:
                    blanks += 1
                if r["is_double"]:
                    doubles += 1
                if not r["is_blank"]:
                    diffs.append(r["difficulty"])

        row["avg_difficulty"] = round(np.mean(diffs), 2) if diffs else 6
        row["blank_gws"]      = blanks
        row["double_gws"]     = doubles
        pivot_rows.append(row)

    pivot_df = pd.DataFrame(pivot_rows)
    enriched = pred_df.merge(pivot_df, on="team_id", how="left")

    # Fixture run label
    def label(avg):
        if avg <= 2.4:   return "🟢 Easy"
        elif avg <= 3.2: return "🟡 Moderate"
        else:            return "🔴 Tough"

    enriched["fixture_run_label"] = enriched["avg_difficulty"].apply(label)

    # Combined score: 60% next GW pts + 40% fixture run bonus
    # (6 - avg_difficulty) gives a 0-5 scale where lower difficulty = higher bonus
    enriched["combined_score"] = (
        enriched["predicted_pts"] * 0.6 +
        (6 - enriched["avg_difficulty"]) * enriched["predicted_pts"] * 0.4
    ).round(2)

    return enriched


# ─────────────────────────────────────────
# 3. DISPLAY HELPERS
# ─────────────────────────────────────────

def print_fixture_run_table(enriched_df: pd.DataFrame,
                             current_gw: int,
                             gw_lookahead: int = 5,
                             title: str = "FIXTURE RUN"):
    """Print a clean fixture run table sorted by combined score."""
    gws = list(range(current_gw + 1, current_gw + 1 + gw_lookahead))

    print(f"\n{'=' * 75}")
    print(f"  {title}")
    print(f"{'=' * 75}")

    gw_headers = "  ".join(f"GW{gw}" for gw in gws)
    print(f"{'Player':<28} {'Pos':<4} {'£':<5} {'Pred':>5} {'Run':>14} {'B':>2} {'D':>2}  {gw_headers}")
    print("-" * 75)

    for _, row in enriched_df.sort_values("combined_score", ascending=False).iterrows():
        gw_diffs = "   ".join(
            str(int(row.get(f"gw{gw}_difficulty", "-"))) for gw in gws
        )
        # Add DGW marker
        gw_labels = []
        for gw in gws:
            diff = int(row.get(f"gw{gw}_difficulty", 6))
            opp  = str(row.get(f"gw{gw}_opponent", ""))
            if opp == "BLANK":
                gw_labels.append(" B ")
            elif "&" in opp:
                gw_labels.append(f"{diff}D")
            else:
                gw_labels.append(f" {diff} ")

        gw_str = "  ".join(gw_labels)
        print(
            f"{str(row['player_name']):<28} "
            f"{str(row['position']):<4} "
            f"£{row['price']:<4.1f} "
            f"{row['predicted_pts']:>5} "
            f"{str(row['fixture_run_label']):>14} "
            f"{int(row.get('blank_gws', 0)):>2} "
            f"{int(row.get('double_gws', 0)):>2}  "
            f"{gw_str}"
        )


# ─────────────────────────────────────────
# 4. TRANSFER SUGGESTIONS (5-GW OUTLOOK)
# ─────────────────────────────────────────

def show_transfer_suggestions_phase2(my_team_enriched: pd.DataFrame,
                                      other_enriched: pd.DataFrame,
                                      bank_balance: float):
    """
    Transfer suggestions ranked by combined_score gain (5-GW outlook).
    Also shows next GW gain and fixture run label for context.
    Prompts for manual budget override if limited options found.
    """
    def compute_suggestions(budget: float) -> pd.DataFrame:
        suggestions = []
        for _, my_row in my_team_enriched.iterrows():
            same_pos = other_enriched[
                other_enriched["position"] == my_row["position"]
            ].copy()
            same_pos["gain"]      = (same_pos["combined_score"] - my_row["combined_score"]).round(2)
            same_pos["next_gain"] = (same_pos["predicted_pts"]  - my_row["predicted_pts"]).round(2)
            same_pos["cost_diff"] = (same_pos["price"] - my_row["price"]).round(1)
            same_pos["replace"]   = my_row["player_name"]
            same_pos["budget_ok"] = same_pos["cost_diff"] <= budget
            suggestions.append(same_pos)
        return pd.concat(suggestions).sort_values("gain", ascending=False)

    def print_suggestions(sug_df: pd.DataFrame, budget: float):
        print(f"\n💰 Best Within-Budget Transfers (Bank: £{budget:.1f}M):")
        budget_top = (sug_df[sug_df["budget_ok"] & (sug_df["gain"] > 0)]
                      .drop_duplicates("player_name")
                      .head(5))
        if budget_top.empty:
            print("  No affordable upgrades found.")
        else:
            for _, r in budget_top.iterrows():
                print(
                    f"  OUT: {str(r['replace']):25s}  ->  IN: {str(r['player_name']):25s}"
                    f"  [{r['position']:3s}]"
                    f"  5GW Gain: +{r['gain']}"
                    f"  Next GW: +{r['next_gain']}"
                    f"  Run: {r['fixture_run_label']}"
                    f"  Cost: {r['cost_diff']:+.1f}M"
                )

        print(f"\n💸 Best Transfers Regardless of Budget:")
        all_top = (sug_df[sug_df["gain"] > 0]
                   .drop_duplicates("player_name")
                   .head(5))
        for _, r in all_top.iterrows():
            print(
                f"  OUT: {str(r['replace']):25s}  ->  IN: {str(r['player_name']):25s}"
                f"  [{r['position']:3s}]"
                f"  5GW Gain: +{r['gain']}"
                f"  Next GW: +{r['next_gain']}"
                f"  Run: {r['fixture_run_label']}"
                f"  Cost: {r['cost_diff']:+.1f}M"
            )
        return budget_top

    sug_df     = compute_suggestions(bank_balance)
    budget_top = print_suggestions(sug_df, bank_balance)

    if len(budget_top) < 3:
        print(f"\n⚠️  Note: Due to FPL API limitations, your bank balance (£{bank_balance:.1f}M)")
        print(f"   may differ slightly from the FPL app. Double-check before confirming.")
        print(f"\n❓ Only {len(budget_top)} affordable option(s) found within £{bank_balance:.1f}M.")
        user_input = input(
            "   Enter your actual bank balance from the FPL app (or press Enter to skip): £"
        ).strip()
        if user_input:
            try:
                new_budget = float(user_input)
                if new_budget > bank_balance:
                    print(f"\n🔄 Re-running with updated budget: £{new_budget:.1f}M...")
                    sug_df_new = compute_suggestions(new_budget)
                    print_suggestions(sug_df_new, new_budget)
                else:
                    print("  Budget not higher, skipping.")
            except ValueError:
                print("  Invalid input, skipping.")
    else:
        print(f"\n⚠️  Note: Bank shown (£{bank_balance:.1f}M) may differ slightly from the FPL app.")
        print(f"   Always double-check before confirming a transfer.")


# ─────────────────────────────────────────
# 5. FULL PHASE 2 PIPELINE
# ─────────────────────────────────────────

def run_phase2(team_id: int, max_players: int = None, refresh: bool = False):
    """
    Full Phase 2 pipeline:
      1. Run Phase 1 data pipeline (with cache support)
      2. Build fixture run for next 5 GWs
      3. Enrich player predictions with fixture run scores
      4. Display your squad fixture run table
      5. Captain recommendation with fixture context
      6. Transfer suggestions using 5-GW combined score
      7. Top players by position for scouting
    """
    print("=" * 75)
    print("  FPL AI ASSISTANT — Phase 2: Fixture Run Analysis")
    print("=" * 75)

    # ── Fetch data ─────────────────────────────────────
    print("\n⬇️  Fetching bootstrap & fixtures...")
    bootstrap   = fetch_bootstrap()
    fixtures_df = fetch_fixtures()

    current_gw = fetch_current_gw(bootstrap)
    print(f"📅 Last completed GW: {current_gw}  ->  Analysing GW{current_gw+1} to GW{current_gw+5}")

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

    # ── Player history (cached) ────────────────────────
    print(f"\n📚 Loading player history...")
    history_df = build_player_history_df(bootstrap, max_players=max_players, refresh=refresh)

    # ── Train model ────────────────────────────────────
    print("\n🤖 Training model...")
    model, pos_enc, opp_enc, _ = train_model(history_df)

    with open("fpl_model.pkl", "wb") as f:
        pickle.dump({
            "model":    model,
            "pos_enc":  pos_enc,
            "opp_enc":  opp_enc,
            "features": FEATURE_COLS
        }, f)

    # ── Predict next GW ────────────────────────────────
    print(f"\n🔮 Predicting GW{current_gw+1} scores...")
    pred_df = build_current_features(
        bootstrap, fixtures_df, history_df, pos_enc, opp_enc, current_gw
    )
    pred_df["predicted_pts"] = model.predict(pred_df[FEATURE_COLS]).round(2)
    pred_df["predicted_pts"] = pred_df["predicted_pts"].clip(lower=0)

    # ── Build fixture run ──────────────────────────────
    print("\n📆 Building fixture run analysis (next 5 GWs)...")
    fixture_run_df = build_fixture_run(bootstrap, fixtures_df, current_gw, gw_lookahead=5)
    enriched_df    = build_player_fixture_scores(pred_df, fixture_run_df, current_gw, gw_lookahead=5)

    # Split into my team vs rest
    my_team_enriched = enriched_df[enriched_df["player_id"].isin(my_player_ids)].copy()
    other_enriched   = enriched_df[~enriched_df["player_id"].isin(my_player_ids)].copy()

    # ── Your squad fixture run table ───────────────────
    print_fixture_run_table(
        my_team_enriched,
        current_gw,
        title=f"YOUR SQUAD — GW{current_gw+1} to GW{current_gw+5} Fixture Run"
    )
    print("\n  Legend: B = Blank GW   D = Double GW   Number = Difficulty (1=Easy, 5=Hard)")

    # ── Captain recommendation ─────────────────────────
    print(f"\n{'=' * 75}")
    print("  CAPTAIN RECOMMENDATION")
    print(f"{'=' * 75}")
    captain_candidates = my_team_enriched.nlargest(3, "combined_score")
    labels = ["🏆 Captain     ", "🥈 Vice Captain", "🥉 3rd Option  "]
    for i, (_, row) in enumerate(captain_candidates.iterrows()):
        print(
            f"{labels[i]}: {str(row['player_name']):25s}"
            f"  Next GW: {row['predicted_pts']} pts"
            f"  5GW Run: {row['fixture_run_label']}"
            f"  Combined Score: {row['combined_score']}"
        )

    # ── Transfer suggestions (5-GW) ────────────────────
    print(f"\n{'=' * 75}")
    print("  TRANSFER SUGGESTIONS (5-GW Outlook)")
    print(f"{'=' * 75}")
    show_transfer_suggestions_phase2(my_team_enriched, other_enriched, bank_balance)

    # ── Top players by position ────────────────────────
    print(f"\n{'=' * 75}")
    print("  TOP PLAYERS TO TARGET BY POSITION")
    print(f"{'=' * 75}")
    for pos in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
        print(f"\n🏅 Top 5 {pos}s (by 5GW Combined Score):")
        top = (
            other_enriched[other_enriched["position"] == pos]
            .nlargest(5, "combined_score")
            [["player_name", "team_name", "price",
              "predicted_pts", "avg_difficulty",
              "fixture_run_label", "blank_gws",
              "double_gws", "combined_score"]]
        )
        print(top.to_string(index=False))

    # ── Save enriched predictions ──────────────────────
    enriched_df.to_csv("fpl_predictions_phase2.csv", index=False)
    print("\n✅ Enriched predictions saved to fpl_predictions_phase2.csv")
    print("✅ Ready for Phase 3 (Team Composition Constraints + ILP Optimizer)")

    return enriched_df, my_team_enriched


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    MY_TEAM_ID = 9179961   # <- Your FPL team ID

    # Detect --refresh flag
    REFRESH = "--refresh" in sys.argv
    if REFRESH:
        print("🔄 --refresh flag detected. Will fetch fresh data from API.\n")

    # Use max_players=100 for a quick ~20s test
    # Use max_players=None for full dataset (uses cache after first run)
    enriched_df, my_team = run_phase2(
        team_id=MY_TEAM_ID,
        max_players=None,
        refresh=REFRESH
    )