"""
FPL AI Assistant — Phase 1: Deep ML Model
==========================================
Improvements over v1:
  - Fetches full per-GW player history via /api/element-summary/{id}/
  - Proper prediction target: actual points scored in a GW
  - Rich feature set:
      * Rolling avg points (last 3 & 5 GWs)
      * Rolling avg minutes (last 3 GWs)
      * Home/Away flag
      * Opponent team difficulty
      * Goals, assists, clean sheets (rolling)
      * Bonus points (rolling)
      * Threat, creativity, influence (ICT rolling)
      * Position encoded
      * Player price (now_cost)
  - Train/test split with proper evaluation (RMSE + R²)
  - Saves model + feature columns for reuse in later phases
  - Outputs a clean predicted_scores DataFrame ready for Phase 2+
  - Bank balance from history endpoint (deadline snapshot)
  - Warning about API bank limitation + interactive budget override
  - Correct GW detection using is_current / is_next from events list
  - Player history caching — fast reloads, refresh with --refresh flag
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────
# 1. API HELPERS
# ─────────────────────────────────────────

BASE_URL   = "https://fantasy.premierleague.com/api"
CACHE_FILE = "player_history_cache.csv"

def fetch_bootstrap() -> dict:
    """Master FPL data — all players, teams, positions, events."""
    r = requests.get(f"{BASE_URL}/bootstrap-static/")
    r.raise_for_status()
    return r.json()

def fetch_player_history(player_id: int) -> list:
    """Per-gameweek history for a single player."""
    try:
        r = requests.get(f"{BASE_URL}/element-summary/{player_id}/", timeout=10)
        r.raise_for_status()
        return r.json().get("history", [])
    except Exception:
        return []

def fetch_fixtures() -> pd.DataFrame:
    """All fixtures with difficulty ratings."""
    r = requests.get(f"{BASE_URL}/fixtures/")
    r.raise_for_status()
    return pd.DataFrame(r.json())

def fetch_current_gw(bootstrap: dict) -> int:
    """
    Returns the true current/live gameweek using the events list.
    Priority:
      1. is_current == True  -> that GW is live right now
      2. is_next == True     -> deadline not hit yet, current = next - 1
      3. Fallback            -> last finished GW
    """
    events = bootstrap["events"]

    for event in events:
        if event.get("is_current"):
            return event["id"]

    for event in events:
        if event.get("is_next"):
            return event["id"] - 1

    finished = [e for e in events if e.get("finished")]
    if finished:
        return finished[-1]["id"]

    return 1

def fetch_my_team(team_id: int, current_gw: int) -> dict:
    """Picks for your FPL team in the given gameweek."""
    picks_url = f"{BASE_URL}/entry/{team_id}/event/{current_gw}/picks/"
    r = requests.get(picks_url)
    r.raise_for_status()
    return r.json()

def fetch_transfer_info(team_id: int, current_gw: int) -> dict:
    """
    Fetches bank balance and transfer status from the history endpoint.
    Note: bank value is a deadline snapshot — may differ slightly from the app.
    """
    try:
        history_url = f"{BASE_URL}/entry/{team_id}/history/"
        r = requests.get(history_url)
        r.raise_for_status()
        history = r.json()

        last           = history["current"][-1]
        bank_balance   = last["bank"] / 10
        transfers_made = last["event_transfers"]
        transfer_cost  = last["event_transfers_cost"]

        if transfers_made == 0:
            transfer_status = "1 free transfer available"
        elif transfer_cost == 0:
            transfer_status = "Free transfer already used this GW"
        else:
            transfer_status = f"{transfers_made} transfers made — {transfer_cost} pt hit taken"

        return {
            "bank_balance":    bank_balance,
            "transfers_made":  transfers_made,
            "transfer_status": transfer_status,
        }

    except Exception as e:
        print(f"⚠️  Could not fetch transfer info: {e}")
        return {
            "bank_balance":    0.0,
            "transfers_made":  0,
            "transfer_status": "Unknown (defaulting to 1 free transfer)",
        }

# ─────────────────────────────────────────
# 2. BUILD HISTORICAL DATASET (WITH CACHE)
# ─────────────────────────────────────────

def rolling_avg(series: pd.Series, window: int) -> pd.Series:
    """Shift-then-roll so we never leak future data into features."""
    return series.shift(1).rolling(window, min_periods=1).mean()

def _fetch_fresh_history(bootstrap: dict, max_players: int = None) -> pd.DataFrame:
    """
    Fetches GW-by-GW history for every active player from the API.
    This is the slow path — takes ~100s for all players.
    Called only when cache is missing or --refresh is used.
    """
    players_raw = bootstrap["elements"]
    teams_df    = pd.DataFrame(bootstrap["teams"])
    pos_df      = pd.DataFrame(bootstrap["element_types"])

    team_map = teams_df.set_index("id")["name"].to_dict()
    pos_map  = pos_df.set_index("id")["singular_name"].to_dict()

    active = [p for p in players_raw if p["status"] == "a"]
    if max_players:
        active = active[:max_players]

    all_rows = []
    print(f"  Fetching history for {len(active)} players — ~{len(active)//5}s...")

    for i, player in enumerate(active):
        if i % 50 == 0:
            print(f"  {i}/{len(active)}")

        pid       = player["id"]
        pos_name  = pos_map.get(player["element_type"], "Unknown")
        team_name = team_map.get(player["team"], "Unknown")
        price     = player["now_cost"] / 10

        history = fetch_player_history(pid)
        if not history:
            continue

        df_h = pd.DataFrame(history)
        df_h = df_h.sort_values("round").reset_index(drop=True)

        # Rolling features — shift(1) prevents leaking current GW data
        df_h["roll3_pts"]        = rolling_avg(df_h["total_points"],             3)
        df_h["roll5_pts"]        = rolling_avg(df_h["total_points"],             5)
        df_h["roll3_mins"]       = rolling_avg(df_h["minutes"],                  3)
        df_h["roll3_goals"]      = rolling_avg(df_h["goals_scored"],             3)
        df_h["roll3_assists"]    = rolling_avg(df_h["assists"],                  3)
        df_h["roll3_clean"]      = rolling_avg(df_h["clean_sheets"],             3)
        df_h["roll3_bonus"]      = rolling_avg(df_h["bonus"],                    3)
        df_h["roll3_threat"]     = rolling_avg(df_h["threat"].astype(float),     3)
        df_h["roll3_creativity"] = rolling_avg(df_h["creativity"].astype(float), 3)
        df_h["roll3_influence"]  = rolling_avg(df_h["influence"].astype(float),  3)

        df_h["player_id"]   = pid
        df_h["player_name"] = f"{player['first_name']} {player['second_name']}"
        df_h["position"]    = pos_name
        df_h["team_name"]   = team_name
        df_h["price"]       = price
        df_h["is_home"]     = df_h["was_home"].astype(int)
        df_h["target"]      = df_h["total_points"]

        all_rows.append(df_h)
        time.sleep(0.05)

    print(f"  Done. {len(all_rows)} players processed.")
    return pd.concat(all_rows, ignore_index=True)


def build_player_history_df(bootstrap: dict,
                             max_players: int = None,
                             refresh: bool = False) -> pd.DataFrame:
    """
    Returns player history DataFrame.
    - If cache exists and refresh=False: loads from disk instantly (~2s)
    - If cache missing or refresh=True:  fetches from API (~100s) and saves cache

    Usage:
      python fpl_phase1_model.py           # uses cache if available
      python fpl_phase1_model.py --refresh # forces fresh API fetch
    """
    if not refresh and os.path.exists(CACHE_FILE):
        print(f"📦 Loading player history from cache ({CACHE_FILE})...")
        df = pd.read_csv(CACHE_FILE)
        print(f"✅ Cache loaded — {len(df)} rows, {df['player_id'].nunique()} players.")
        return df

    if refresh:
        print("🔄 Refresh flag detected — fetching fresh player history from API...")
    else:
        print("📡 No cache found — fetching player history from API...")

    df = _fetch_fresh_history(bootstrap, max_players)
    df.to_csv(CACHE_FILE, index=False)
    print(f"💾 Player history cached to {CACHE_FILE}")
    return df

# ─────────────────────────────────────────
# 3. PREPARE FEATURES
# ─────────────────────────────────────────

FEATURE_COLS = [
    "roll3_pts", "roll5_pts",
    "roll3_mins",
    "roll3_goals", "roll3_assists", "roll3_clean", "roll3_bonus",
    "roll3_threat", "roll3_creativity", "roll3_influence",
    "is_home",
    "opponent_team",
    "difficulty",
    "price",
    "pos_encoded",
]

def prepare_features(df: pd.DataFrame):
    """Encode categoricals and drop rows with missing features."""

    pos_encoder = LabelEncoder()
    df["pos_encoded"] = pos_encoder.fit_transform(df["position"])

    if "difficulty" not in df.columns:
        df["difficulty"] = 3
    df["difficulty"] = pd.to_numeric(df["difficulty"], errors="coerce").fillna(3)

    opp_encoder = LabelEncoder()
    df["opponent_team"] = opp_encoder.fit_transform(df["opponent_team"].astype(str))

    df_clean = df.dropna(subset=FEATURE_COLS + ["target"]).copy()
    df_clean = df_clean[df_clean["roll3_pts"].notna()]

    return df_clean, pos_encoder, opp_encoder

# ─────────────────────────────────────────
# 4. TRAIN MODEL
# ─────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """Train a Gradient Boosting model and evaluate it."""

    df_feat, pos_enc, opp_enc = prepare_features(df)

    X = df_feat[FEATURE_COLS]
    y = df_feat["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    print(f"\n📊 Model Evaluation:")
    print(f"   RMSE : {rmse:.3f} pts")
    print(f"   R2   : {r2:.3f}")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    print("\n🔍 Top Feature Importances:")
    print(importances.sort_values(ascending=False).head(8).to_string())

    return model, pos_enc, opp_enc, df_feat

# ─────────────────────────────────────────
# 5. BUILD NEXT-GW FEATURE VECTORS
# ─────────────────────────────────────────

def build_current_features(bootstrap: dict, fixtures_df: pd.DataFrame,
                            history_df: pd.DataFrame,
                            pos_enc: LabelEncoder, opp_enc: LabelEncoder,
                            current_gw: int) -> pd.DataFrame:
    """
    For each active player, build their feature vector for the NEXT gameweek
    using their most recent rolling stats + upcoming fixture difficulty.
    """
    players_raw = bootstrap["elements"]
    teams_df    = pd.DataFrame(bootstrap["teams"])
    pos_df      = pd.DataFrame(bootstrap["element_types"])
    team_map    = teams_df.set_index("id")["name"].to_dict()
    pos_map     = pos_df.set_index("id")["singular_name"].to_dict()

    next_gw  = current_gw + 1
    upcoming = fixtures_df[fixtures_df["event"] == next_gw]

    next_fixture_map = {}
    for _, row in upcoming.iterrows():
        next_fixture_map[row["team_h"]] = {
            "difficulty":    row["team_h_difficulty"],
            "is_home":       1,
            "opponent_team": row["team_a"]
        }
        next_fixture_map[row["team_a"]] = {
            "difficulty":    row["team_a_difficulty"],
            "is_home":       0,
            "opponent_team": row["team_h"]
        }

    active = [
        p for p in players_raw
        if p["status"] == "a" and
        (p.get("chance_of_playing_next_round") is None or
         (p.get("chance_of_playing_next_round") or 100) >= 75)
    ]

    rows = []
    for player in active:
        pid = player["id"]

        p_hist = history_df[history_df["player_id"] == pid].sort_values("round")
        if p_hist.empty:
            continue
        last = p_hist.iloc[-1]

        fixture = next_fixture_map.get(player["team"], {})
        if not fixture:
            continue

        pos_name = pos_map.get(player["element_type"], "Unknown")
        try:
            pos_encoded = pos_enc.transform([pos_name])[0]
        except Exception:
            pos_encoded = 0

        try:
            opp_encoded = opp_enc.transform([str(fixture["opponent_team"])])[0]
        except Exception:
            opp_encoded = 0

        rows.append({
            "player_id":        pid,
            "player_name":      f"{player['first_name']} {player['second_name']}",
            "position":         pos_name,
            "team_name":        team_map.get(player["team"], "Unknown"),
            "price":            player["now_cost"] / 10,
            "team_id":          player["team"],
            "roll3_pts":        last.get("roll3_pts", 0),
            "roll5_pts":        last.get("roll5_pts", 0),
            "roll3_mins":       last.get("roll3_mins", 0),
            "roll3_goals":      last.get("roll3_goals", 0),
            "roll3_assists":    last.get("roll3_assists", 0),
            "roll3_clean":      last.get("roll3_clean", 0),
            "roll3_bonus":      last.get("roll3_bonus", 0),
            "roll3_threat":     last.get("roll3_threat", 0),
            "roll3_creativity": last.get("roll3_creativity", 0),
            "roll3_influence":  last.get("roll3_influence", 0),
            "is_home":          fixture["is_home"],
            "opponent_team":    opp_encoded,
            "difficulty":       fixture["difficulty"],
            "pos_encoded":      pos_encoded,
        })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────
# 6. TRANSFER SUGGESTIONS
# ─────────────────────────────────────────

def show_transfer_suggestions(my_team_df: pd.DataFrame,
                               other_players: pd.DataFrame,
                               bank_balance: float):
    """
    Show transfer suggestions within budget.
    If limited options found, prompt user to enter their actual
    bank balance from the FPL app for a wider search.
    """
    def compute_suggestions(budget: float):
        suggestions = []
        for _, my_row in my_team_df.iterrows():
            same_pos = other_players[other_players["position"] == my_row["position"]].copy()
            same_pos["gain"]      = (same_pos["predicted_pts"] - my_row["predicted_pts"]).round(2)
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
            print("  No affordable upgrades found within this budget.")
        else:
            for _, r in budget_top.iterrows():
                print(f"  OUT: {r['replace']:25s}  ->  IN: {r['player_name']:25s}"
                      f"  [{r['position']:3s}]  Gain: +{r['gain']}  Cost: {r['cost_diff']:+.1f}M")

        print("\n💸 Best Transfers Regardless of Budget:")
        all_top = (sug_df[sug_df["gain"] > 0]
                   .drop_duplicates("player_name")
                   .head(5))
        for _, r in all_top.iterrows():
            print(f"  OUT: {r['replace']:25s}  ->  IN: {r['player_name']:25s}"
                  f"  [{r['position']:3s}]  Gain: +{r['gain']}  Cost: {r['cost_diff']:+.1f}M")

        return budget_top

    sug_df     = compute_suggestions(bank_balance)
    budget_top = print_suggestions(sug_df, bank_balance)

    if len(budget_top) < 3:
        print(f"\n⚠️  Note: Due to FPL API limitations, your bank balance shown (£{bank_balance:.1f}M)")
        print(f"   may differ slightly from what the FPL app shows.")
        print(f"   Please double-check your bank in the app before confirming any transfer.")
        print(f"\n❓ Only {len(budget_top)} affordable option(s) found within £{bank_balance:.1f}M.")

        user_input = input(
            f"   Enter your actual bank balance from the FPL app "
            f"(or press Enter to skip): £"
        ).strip()

        if user_input:
            try:
                new_budget = float(user_input)
                if new_budget > bank_balance:
                    print(f"\n🔄 Re-running with updated budget: £{new_budget:.1f}M...")
                    sug_df_new = compute_suggestions(new_budget)
                    print_suggestions(sug_df_new, new_budget)
                else:
                    print("  Budget not higher than current value, skipping.")
            except ValueError:
                print("  Invalid input, skipping.")
    else:
        print(f"\n⚠️  Note: Due to FPL API limitations, your bank balance (£{bank_balance:.1f}M)")
        print(f"   may differ slightly from the FPL app. Always double-check before confirming a transfer.")

# ─────────────────────────────────────────
# 7. FULL PIPELINE
# ─────────────────────────────────────────

def run_pipeline(team_id: int, max_players: int = None, refresh: bool = False):
    """
    End-to-end Phase 1 pipeline.
    Pass refresh=True to force a fresh API fetch instead of using cache.
    """
    print("=" * 55)
    print("  FPL AI ASSISTANT — Phase 1: Deep ML Model")
    print("=" * 55)

    print("\n⬇️  Fetching bootstrap data...")
    bootstrap   = fetch_bootstrap()
    fixtures_df = fetch_fixtures()

    current_gw = fetch_current_gw(bootstrap)
    print(f"📅 Last completed GW: {current_gw}  ->  Predicting for GW{current_gw + 1}")

    print("⬇️  Fetching your team...")
    try:
        team_data     = fetch_my_team(team_id, current_gw)
        my_player_ids = [p["element"] for p in team_data["picks"]]
        print(f"✅ Team fetched successfully.")
    except Exception as e:
        print(f"⚠️  Could not fetch team: {e}")
        my_player_ids = []

    transfer_info   = fetch_transfer_info(team_id, current_gw)
    bank_balance    = transfer_info["bank_balance"]
    transfer_status = transfer_info["transfer_status"]
    print(f"💰 Bank: £{bank_balance:.1f}M  |  Transfers: {transfer_status}")

    print(f"\n📚 Building player history dataset...")
    history_df = build_player_history_df(bootstrap, max_players=max_players, refresh=refresh)

    print("\n🤖 Training Gradient Boosting model...")
    model, pos_enc, opp_enc, df_feat = train_model(history_df)

    with open("fpl_model.pkl", "wb") as f:
        pickle.dump({
            "model":    model,
            "pos_enc":  pos_enc,
            "opp_enc":  opp_enc,
            "features": FEATURE_COLS
        }, f)
    print("\n💾 Model saved to fpl_model.pkl")

    print(f"\n🔮 Predicting GW{current_gw + 1} scores for all active players...")
    pred_df = build_current_features(
        bootstrap, fixtures_df, history_df, pos_enc, opp_enc, current_gw
    )
    pred_df["predicted_pts"] = model.predict(pred_df[FEATURE_COLS]).round(2)
    pred_df["predicted_pts"] = pred_df["predicted_pts"].clip(lower=0)

    my_team_df    = pred_df[pred_df["player_id"].isin(my_player_ids)].copy()
    other_players = pred_df[~pred_df["player_id"].isin(my_player_ids)].copy()

    print("\n" + "=" * 55)
    print(f"  YOUR SQUAD — GW{current_gw + 1} Predictions")
    print("=" * 55)
    display_cols = ["player_name", "position", "price", "predicted_pts", "difficulty", "is_home"]
    print(my_team_df[display_cols].sort_values(
        "predicted_pts", ascending=False
    ).to_string(index=False))

    if not my_team_df.empty:
        captain      = my_team_df.nlargest(1, "predicted_pts").iloc[0]
        vice_captain = my_team_df.nlargest(2, "predicted_pts").iloc[1]
        print(f"\n🏆 Captain:      {captain['player_name']} — {captain['predicted_pts']} predicted pts")
        print(f"🥈 Vice Captain: {vice_captain['player_name']} — {vice_captain['predicted_pts']} predicted pts")

    print("\n" + "=" * 55)
    print("  TRANSFER SUGGESTIONS")
    print("=" * 55)
    show_transfer_suggestions(my_team_df, other_players, bank_balance)

    pred_df.to_csv("fpl_predictions.csv", index=False)
    print("\n✅ Full predictions saved to fpl_predictions.csv")
    print("✅ Ready for Phase 2 (Fixture Run Analysis + Squad Optimizer)")

    return model, pred_df, my_team_df

# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    MY_TEAM_ID = 9179961   # <- Your FPL team ID

    # Detect --refresh flag
    REFRESH = "--refresh" in sys.argv
    if REFRESH:
        print("🔄 --refresh flag detected. Will fetch fresh data from API.\n")

    # Use max_players=100 for a quick ~20s test run
    # Use max_players=None for all active players (~5-10 mins, first run only)
    model, pred_df, my_team_df = run_pipeline(
        team_id=MY_TEAM_ID,
        max_players=None,
        refresh=REFRESH
    )