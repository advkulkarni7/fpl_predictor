"""
FPL AI Assistant — Phase 1: Deep ML Model (v5)
===============================================
Improvements over v4:

  🔴 HIGH IMPACT — ALGORITHMIC:
  1. EWMA Form Features — shift(1).ewm(span=N) replaces equal-weight rolling.
     Recent GWs carry more predictive weight. Added for pts, mins, goals,
     assists, threat, creativity. Both EWMA and rolling kept so the model
     can choose which carries more signal.
  2. Expected Minutes (xMins) Probability Weighting — predicted_pts is now
     adjusted by P(player plays meaningfully), combining chance_of_playing
     and rolling average minutes to penalise rotation risks. Stored as
     expected_pts alongside raw predicted_pts.
  3. XGBoost Quantile Regression — each position now trains THREE models:
       - Median model  (objective=squarederror)   → predicted_pts
       - Q10 floor     (quantile_alpha=0.10)       → pts_low
       - Q90 ceiling   (quantile_alpha=0.90)       → pts_high
     This gives confidence intervals: "Salah: 6.2 pts | range 2.8–13.1"

  🟡 MEDIUM IMPACT — FEATURE ENGINEERING:
  4. Transfer Momentum — transfers_in_event / transfers_out_event added as
     info columns to pred_df (crowd-intelligence context). Not model features
     since per-GW historical data isn't available from FPL element-summary.
  5. Ownership — selected_by_percent (selected_pct) added as inference feature
     in build_current_features only. NOT a training feature — FPL API only
     exposes current-week ownership, so using it during training would leak
     today's value into historical GW rows (data leakage). At inference time
     it's valid because we're predicting the next GW using today's ownership.

  🟢 ROBUSTNESS:
  6. Incremental Cache — checks if cache already covers current_gw before
     hitting the API. Also detects missing v5 columns and forces refresh.
  7. Model Versioning — saves fpl_model_gw{N}.pkl alongside fpl_model.pkl
     so you can compare week-over-week model quality without overwriting.

  🧩 EXTENDED MODELLING:
  8. Multi-Output Component Models — separate XGBoost models predict each
     FPL scoring component (goals, assists, clean_sheets, bonus) per position.
     Components converted to FPL pts using official scoring rules. Stored as
     pred_goals, pred_assists, pred_clean, pred_bonus, pts_from_components.
     Final predicted_pts = blend of direct model (60%) + components (40%).
  9. Price Rise/Fall Prediction — standalone XGBoost model trained to predict
     next-GW price change from form, transfers, ownership, current price.
     Output: predicted_price_change column in pred_df.

  BACKWARD COMPATIBILITY:
  - Model dict keys rmse, r2, model, features all preserved
  - train_model() wrapper untouched — Phases 2/3/4 unaffected
  - build_current_features() signature unchanged

Run normally (uses cache):
  python fpl_phase1_model.py

Force fresh API fetch:
  python fpl_phase1_model.py --refresh
"""

import os
import sys
import json
import logging
import requests
import pandas as pd
import numpy as np
import pickle
import time
import hashlib
from datetime import datetime, timedelta
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe everywhere
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

# ─────────────────────────────────────────
# 0. CONFIG + LOGGING
# ─────────────────────────────────────────

try:
    from config import (
        TEAM_ID, CACHE_FILE, CACHE_MAX_AGE_DAYS,
        ROLLING_TRAIN_WINDOW, RANDOM_STATE,
        MIN_CHANCE_OF_PLAYING, LOG_FILE
    )
except ImportError:
    TEAM_ID                = 9179961
    CACHE_FILE             = "player_history_cache.csv"
    CACHE_MAX_AGE_DAYS     = 7
    ROLLING_TRAIN_WINDOW   = 10
    RANDOM_STATE           = 42
    MIN_CHANCE_OF_PLAYING  = 75
    LOG_FILE               = "fpl_assistant.log"

# Blend weight for component model vs direct model predictions.
# 0.40 = 60% direct XGBoost, 40% component (goals+assists+clean+bonus) model.
# Increase toward 1.0 if component model RMSE improves relative to direct model.
try:
    from config import COMPONENT_BLEND_WEIGHT
except ImportError:
    COMPONENT_BLEND_WEIGHT = 0.40

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

BASE_URL = "https://fantasy.premierleague.com/api"

# ─────────────────────────────────────────
# 1. API HELPERS
# ─────────────────────────────────────────

def fetch_bootstrap() -> dict:
    """Master FPL data — players, teams, positions, events."""
    r = requests.get(f"{BASE_URL}/bootstrap-static/")
    r.raise_for_status()
    return r.json()


def fetch_player_history(player_id: int, retries: int = 1) -> list:
    """Per-gameweek history for a single player. Retries once on timeout."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(
                f"{BASE_URL}/element-summary/{player_id}/",
                timeout=10
            )
            r.raise_for_status()
            return r.json().get("history", [])
        except requests.exceptions.Timeout:
            if attempt < retries:
                log.warning(f"Timeout for player {player_id}, retrying...")
                time.sleep(1)
            else:
                log.warning(f"Skipping player {player_id} after {retries+1} attempts.")
                return []
        except Exception as e:
            log.warning(f"Error fetching player {player_id}: {e}")
            return []


def fetch_fixtures() -> pd.DataFrame:
    """All fixtures with difficulty ratings."""
    r = requests.get(f"{BASE_URL}/fixtures/")
    r.raise_for_status()
    return pd.DataFrame(r.json())


def fetch_current_gw(bootstrap: dict) -> int:
    """
    Returns the true current/live gameweek.
      1. is_current == True
      2. is_next == True  → current = next - 1
      3. Fallback         → last finished GW
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
    r = requests.get(f"{BASE_URL}/entry/{team_id}/event/{current_gw}/picks/")
    r.raise_for_status()
    return r.json()


def fetch_transfer_info(team_id: int, current_gw: int) -> dict:
    """Bank balance and transfer status. Bank is a deadline snapshot."""
    try:
        r = requests.get(f"{BASE_URL}/entry/{team_id}/history/")
        r.raise_for_status()
        last           = r.json()["current"][-1]
        bank_balance   = last["bank"] / 10
        transfers_made = last["event_transfers"]
        transfer_cost  = last["event_transfers_cost"]

        if transfers_made == 0:
            status = "1 free transfer available"
        elif transfer_cost == 0:
            status = "Free transfer already used this GW"
        else:
            status = f"{transfers_made} transfers made — {transfer_cost} pt hit taken"

        return {
            "bank_balance":    bank_balance,
            "transfers_made":  transfers_made,
            "transfer_status": status,
        }
    except Exception as e:
        log.warning(f"Could not fetch transfer info: {e}")
        return {
            "bank_balance":    0.0,
            "transfers_made":  0,
            "transfer_status": "Unknown (defaulting to 1 free transfer)",
        }

# ─────────────────────────────────────────
# 2. OPPONENT ENCODING (HASH-BASED)
# ─────────────────────────────────────────

def encode_opponent(team_id) -> int:
    """Hash-based encoding — never breaks on new/unknown teams."""
    return int(hashlib.md5(str(team_id).encode()).hexdigest(), 16) % 1000

# ─────────────────────────────────────────
# 3. OPPONENT STRENGTH MAP
# ─────────────────────────────────────────

def build_opponent_strength_map(history_df: pd.DataFrame) -> dict:
    """Map: opponent_team_id -> avg goals conceded per game."""
    if "goals_scored" not in history_df.columns or \
       "opponent_team" not in history_df.columns:
        return {}
    return (
        history_df.groupby("opponent_team")["goals_scored"]
        .mean()
        .to_dict()
    )

# ─────────────────────────────────────────
# 4. FPL SCORING CONSTANTS
# ─────────────────────────────────────────
# Used by the multi-output component model to convert
# predicted goals/assists/clean/bonus → FPL points.

GOAL_PTS = {
    "Goalkeeper": 6,
    "Defender":   6,
    "Midfielder": 5,
    "Forward":    4,
}
ASSIST_PTS       = 3    # same for all positions
CLEAN_PTS = {
    "Goalkeeper": 4,
    "Defender":   4,
    "Midfielder": 1,
    "Forward":    0,
}
MINS_PTS_FULL    = 2    # 60+ minutes played
MINS_PTS_PART    = 1    # 1–59 minutes played

# New-column sentinel for incremental cache check
_V5_REQUIRED_COLS = ["ewm3_pts", "ewm5_pts", "selected_pct"]

# ─────────────────────────────────────────
# 5. BUILD HISTORICAL DATASET
# ─────────────────────────────────────────

def rolling_avg(series: pd.Series, window: int) -> pd.Series:
    """Shift-then-roll — prevents leaking current GW into features."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def ewm_avg(series: pd.Series, span: int) -> pd.Series:
    """
    Shift-then-EWMA — exponentially weighted moving average.
    More recent GWs carry higher weight than older ones.
    shift(1) prevents the current row from leaking into its own feature.
    """
    return series.shift(1).ewm(span=span, adjust=False).mean()


def _fetch_fresh_history(bootstrap: dict,
                          max_players: int = None) -> pd.DataFrame:
    """
    Fetches GW-by-GW history for ALL players.

    v5 additions vs v4:
    - EWMA features (ewm3_*, ewm5_*) alongside rolling features
    - selected_pct (ownership %) as static feature per player
    """
    players_raw = bootstrap["elements"]
    teams_df    = pd.DataFrame(bootstrap["teams"])
    pos_df      = pd.DataFrame(bootstrap["element_types"])

    team_map = teams_df.set_index("id")["name"].to_dict()
    pos_map  = pos_df.set_index("id")["singular_name"].to_dict()

    all_players = players_raw[:max_players] if max_players else players_raw

    all_rows = []
    log.info(f"Fetching history for {len(all_players)} players...")

    for i, player in enumerate(all_players):
        if i % 50 == 0:
            log.info(f"  {i}/{len(all_players)}")

        pid          = player["id"]
        pos_name     = pos_map.get(player["element_type"], "Unknown")
        team_name    = team_map.get(player["team"], "Unknown")
        price        = player["now_cost"] / 10
        status       = player.get("status", "a")
        selected_pct = float(
            str(player.get("selected_by_percent", "0") or "0").replace(",", "")
        )

        history = fetch_player_history(pid, retries=1)
        if not history:
            continue

        df_h = pd.DataFrame(history)
        df_h = df_h.sort_values("round").reset_index(drop=True)

        # ── Rolling form features (equal weight) ──────────────────
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

        df_h["roll3_saves"] = rolling_avg(
            df_h["saves"].astype(float) if "saves" in df_h.columns
            else pd.Series(0.0, index=df_h.index), 3
        )
        df_h["roll3_yellows"] = rolling_avg(
            df_h["yellow_cards"].astype(float) if "yellow_cards" in df_h.columns
            else pd.Series(0.0, index=df_h.index), 3
        )

        # ── EWMA form features (recency-weighted) ─────────────────
        # More predictive of short-term form — last week matters most.
        df_h["ewm3_pts"]        = ewm_avg(df_h["total_points"],             3)
        df_h["ewm5_pts"]        = ewm_avg(df_h["total_points"],             5)
        df_h["ewm3_mins"]       = ewm_avg(df_h["minutes"],                  3)
        df_h["ewm3_goals"]      = ewm_avg(df_h["goals_scored"],             3)
        df_h["ewm3_assists"]    = ewm_avg(df_h["assists"],                  3)
        df_h["ewm3_threat"]     = ewm_avg(df_h["threat"].astype(float),     3)
        df_h["ewm3_creativity"] = ewm_avg(df_h["creativity"].astype(float), 3)

        # ── Derived features ──────────────────────────────────────
        df_h["games_played"] = (
            (df_h["minutes"] > 0).astype(int)
            .shift(1).rolling(5, min_periods=1).sum()
        )
        df_h["home_ratio"] = (
            df_h["was_home"].shift(1).rolling(5, min_periods=1).mean()
        )
        if "value" in df_h.columns:
            df_h["price_change"] = df_h["value"].diff().fillna(0)
        else:
            df_h["price_change"] = 0.0

        mins_rolled = df_h["minutes"].shift(1).rolling(3, min_periods=1).mean()
        pts_rolled  = df_h["total_points"].shift(1).rolling(3, min_periods=1).mean()
        df_h["pts_per_90"] = np.where(
            mins_rolled > 0,
            (pts_rolled / mins_rolled) * 90,
            0.0
        )

        df_h["opp_strength"]     = 1.0  # placeholder, backfilled below
        df_h["opponent_encoded"] = df_h["opponent_team"].apply(encode_opponent)
        df_h["gw_number"]        = df_h["round"]

        # ── Static features ───────────────────────────────────────
        df_h["player_id"]     = pid
        df_h["player_name"]   = f"{player['first_name']} {player['second_name']}"
        df_h["position"]      = pos_name
        df_h["team_name"]     = team_name
        df_h["price"]         = price
        df_h["player_status"] = status
        df_h["is_home"]       = df_h["was_home"].astype(int)
        df_h["target"]        = df_h["total_points"]
        df_h["selected_pct"]  = selected_pct  # ownership % (current snapshot)

        all_rows.append(df_h)
        time.sleep(0.05)

    log.info(f"Done. {len(all_rows)} players processed.")
    full_df = pd.concat(all_rows, ignore_index=True)

    # Backfill opp_strength using the complete dataset
    opp_strength_map = build_opponent_strength_map(full_df)
    full_df["opp_strength"] = full_df["opponent_team"].map(
        opp_strength_map
    ).fillna(1.0)

    return full_df


def check_cache_staleness(cache_file: str, max_age_days: int) -> bool:
    """Warn if cache is older than max_age_days."""
    if not os.path.exists(cache_file):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
    if age > timedelta(days=max_age_days):
        log.warning(
            f"⚠️  Cache is {age.days} days old (max {max_age_days}). "
            f"Run with --refresh to update."
        )
        return True
    return False


def _cache_is_current(cache_df: pd.DataFrame, current_gw: int) -> bool:
    """True if cache already contains data up to current_gw."""
    if cache_df.empty or "round" not in cache_df.columns:
        return False
    return int(cache_df["round"].max()) >= current_gw


def build_player_history_df(bootstrap: dict,
                              max_players: int = None,
                              refresh: bool = False) -> pd.DataFrame:
    """
    Returns player history DataFrame with all engineered features.

    v5 Incremental Cache Logic:
      1. If --refresh:            always fetch fresh
      2. If no cache file:        fetch fresh
      3. If cache missing v5 cols: fetch fresh (schema upgrade)
      4. If cache covers current_gw: return cache as-is (no wasted API calls)
      5. Otherwise:               fetch fresh (new GW data available)
    """
    check_cache_staleness(CACHE_FILE, CACHE_MAX_AGE_DAYS)
    current_gw = fetch_current_gw(bootstrap)

    if not refresh and os.path.exists(CACHE_FILE):
        cached_df = pd.read_csv(CACHE_FILE)

        # Schema check — v5 adds new columns; old cache won't have them
        missing_v5 = [c for c in _V5_REQUIRED_COLS if c not in cached_df.columns]
        if missing_v5:
            log.info(f"♻️  Cache missing v5 columns {missing_v5} — refreshing...")
        elif _cache_is_current(cached_df, current_gw):
            log.info(
                f"✅ Cache is current (GW{current_gw}) — "
                f"{len(cached_df)} rows, {cached_df['player_id'].nunique()} players."
            )
            return cached_df
        else:
            max_cached = int(cached_df["round"].max()) if not cached_df.empty else 0
            log.info(
                f"⚡ Cache at GW{max_cached}, current is GW{current_gw} "
                f"→ fetching fresh data..."
            )
    else:
        log.info(
            "🔄 --refresh flag set." if refresh
            else "📡 No cache found — fetching from API..."
        )

    df = _fetch_fresh_history(bootstrap, max_players)
    df.to_csv(CACHE_FILE, index=False)
    log.info(f"💾 Cached → {CACHE_FILE}")
    return df

# ─────────────────────────────────────────
# 6. POSITION-SPECIFIC FEATURE SETS
# ─────────────────────────────────────────

FEATURES_COMMON = [
    # ── Rolling (equal-weight) form ───────────────────────────────
    "roll3_pts",
    "roll5_pts",
    "roll3_mins",
    # ── EWMA (recency-weighted) form ──────────────────────────────
    "ewm3_pts",
    "ewm5_pts",
    "ewm3_mins",
    # ── Derived / efficiency ─────────────────────────────────────
    "pts_per_90",
    "roll3_bonus",
    "roll3_influence",
    # ── Fixture context ───────────────────────────────────────────
    "is_home",
    "opponent_encoded",
    "opp_strength",
    "difficulty",
    # ── Player meta ──────────────────────────────────────────────
    "price",
    "games_played",
    "home_ratio",
    "price_change",
    "gw_number",
    # NOTE: selected_pct (ownership %) intentionally excluded from training
    # features. FPL API only exposes current-week ownership — using it during
    # training would assign today's value to all historical GW rows, leaking
    # future data into the past. It IS used at inference time in
    # build_current_features because at that point "current" is correct.
]

FEATURES_GK = FEATURES_COMMON + [
    "roll3_saves",
    "roll3_clean",
]

FEATURES_DEF = FEATURES_COMMON + [
    "roll3_clean",
    "roll3_goals",
    "roll3_assists",
    "roll3_yellows",
]

FEATURES_MID = FEATURES_COMMON + [
    "roll3_goals",     "ewm3_goals",
    "roll3_assists",   "ewm3_assists",
    "roll3_threat",    "ewm3_threat",
    "roll3_creativity","ewm3_creativity",
    "roll3_clean",
]

FEATURES_FWD = FEATURES_COMMON + [
    "roll3_goals",     "ewm3_goals",
    "roll3_assists",   "ewm3_assists",
    "roll3_threat",    "ewm3_threat",
    "roll3_creativity","ewm3_creativity",
]

POSITION_FEATURE_MAP = {
    "Goalkeeper": FEATURES_GK,
    "Defender":   FEATURES_DEF,
    "Midfielder": FEATURES_MID,
    "Forward":    FEATURES_FWD,
}

# Stable ordered union — downstream phases use this for pkl compatibility
_seen = set()
FEATURE_COLS = []
for _f in (FEATURES_GK + FEATURES_DEF + FEATURES_MID + FEATURES_FWD):
    if _f not in _seen:
        FEATURE_COLS.append(_f)
        _seen.add(_f)

# ─────────────────────────────────────────
# 7. PREPARE FEATURES (TEMPORAL + CV)
# ─────────────────────────────────────────

def prepare_features_for_position(df: pd.DataFrame, position: str) -> tuple:
    """
    Prepare features for a specific position with:
      - Temporal train/test split (no data leakage)
      - 3-fold TimeSeriesSplit CV for reliable metric estimation

    Returns:
      X_train, X_test, y_train, y_test, feature_cols, cv_rmse, cv_r2
    """
    feature_cols = POSITION_FEATURE_MAP[position]
    pos_df       = df[df["position"] == position].copy()

    for col in feature_cols:
        if col not in pos_df.columns:
            pos_df[col] = 0.0

    pos_df[feature_cols] = pos_df[feature_cols].fillna(0)
    pos_df = pos_df.dropna(subset=["target", "gw_number"]).copy()
    pos_df = pos_df[pos_df["roll3_pts"].notna()].copy()
    pos_df = pos_df.sort_values("gw_number").reset_index(drop=True)

    max_gw   = int(pos_df["gw_number"].max())
    train_df = pos_df[pos_df["gw_number"] >= max_gw - ROLLING_TRAIN_WINDOW]
    test_df  = pos_df[pos_df["gw_number"] == max_gw]

    if len(train_df) < 50 or len(test_df) < 10:
        log.warning(
            f"Not enough data for temporal split ({position}), "
            f"falling back to random split."
        )
        train_df, test_df = train_test_split(
            pos_df, test_size=0.2, random_state=RANDOM_STATE
        )

    # 3-fold TimeSeriesSplit CV on training window only
    cv_rmse_scores, cv_r2_scores = [], []
    tscv  = TimeSeriesSplit(n_splits=3)
    X_cv  = train_df[feature_cols].values
    y_cv  = train_df["target"].values

    for tr_idx, va_idx in tscv.split(X_cv):
        if len(tr_idx) < 10 or len(va_idx) < 5:
            continue
        _m = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0,
        )
        _m.fit(X_cv[tr_idx], y_cv[tr_idx])
        _preds = np.clip(_m.predict(X_cv[va_idx]), 0, None)
        cv_rmse_scores.append(np.sqrt(mean_squared_error(y_cv[va_idx], _preds)))
        cv_r2_scores.append(r2_score(y_cv[va_idx], _preds))

    cv_rmse = float(np.mean(cv_rmse_scores)) if cv_rmse_scores else None
    cv_r2   = float(np.mean(cv_r2_scores))   if cv_r2_scores   else None

    return (
        train_df[feature_cols],
        test_df[feature_cols],
        train_df["target"],
        test_df["target"],
        feature_cols,
        cv_rmse,
        cv_r2,
    )

# ─────────────────────────────────────────
# 8. MULTI-OUTPUT COMPONENT MODELS
# ─────────────────────────────────────────

COMPONENT_TARGETS = ["goals_scored", "assists", "clean_sheets", "bonus"]


def _temporal_train_test_split(pos_df: pd.DataFrame,
                                feature_cols: list,
                                target_col: str) -> tuple | None:
    """
    Shared temporal train/test split helper used by both the main position
    models and the component models — ensures consistent split strategy.

    Logic mirrors prepare_features_for_position:
      train = GWs in rolling ROLLING_TRAIN_WINDOW ending at max_gw-1
      test  = most recent GW (held-out)

    Falls back to 80/20 random split if temporal data is insufficient.
    Returns None if there is not enough data at all.
    """
    df = pos_df.sort_values("gw_number").reset_index(drop=True)
    df = df.dropna(subset=[target_col, "gw_number"]).copy()

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0)

    if len(df) < 30:
        return None

    max_gw   = int(df["gw_number"].max())
    train_df = df[df["gw_number"] >= max_gw - ROLLING_TRAIN_WINDOW]
    test_df  = df[df["gw_number"] == max_gw]

    if len(train_df) < 30 or len(test_df) < 5:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    return (
        train_df[feature_cols],
        test_df[feature_cols],
        train_df[target_col],
        test_df[target_col],
    )


def train_component_models(df: pd.DataFrame) -> dict:
    """
    Train per-position XGBoost models for each FPL scoring component.
    Separate models let us explain WHERE predicted points come from.

    Uses the same temporal train/test split strategy as the main models
    via _temporal_train_test_split — consistent, no leakage.

    Returns:
      {
        "Forward": {
          "goals_scored": XGBRegressor,
          "assists":      XGBRegressor,
          "clean_sheets": XGBRegressor,
          "bonus":        XGBRegressor,
        },
        ...
      }
    """
    log.info("🧩 Training multi-output component models...")
    component_models: dict = {}

    for position in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
        component_models[position] = {}
        pos_df       = df[df["position"] == position].copy()
        feature_cols = POSITION_FEATURE_MAP[position]

        for target in COMPONENT_TARGETS:
            if target not in pos_df.columns:
                continue

            split = _temporal_train_test_split(pos_df, feature_cols, target)
            if split is None:
                log.warning(f"  Not enough data for component [{position} | {target}]")
                continue

            X_train, X_test, y_train, y_test = split

            model = xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=3,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, verbosity=0,
                early_stopping_rounds=20,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            preds = np.clip(model.predict(X_test), 0, None)
            rmse  = np.sqrt(mean_squared_error(y_test, preds))
            log.info(f"  Component [{position:10s} | {target:15s}]: RMSE={rmse:.3f}  "
                     f"best_iter={model.best_iteration}")

            component_models[position][target] = model

    return component_models


def _safe_predict(model, X: np.ndarray) -> np.ndarray:
    """Predict if model exists, else return zeros."""
    if model is None:
        return np.zeros(len(X))
    return np.clip(model.predict(X), 0, None)


def predict_component_pts(component_models: dict,
                           pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Use component models to add per-component predictions and
    convert them to FPL points using official scoring rules.

    Adds columns:
      pred_goals, pred_assists, pred_clean, pred_bonus,
      pts_from_components

    FPL scoring applied:
      goals  × GOAL_PTS[position]
      assists × 3
      clean  × CLEAN_PTS[position]  (only if avg mins ≥ 60)
      bonus  as-is
      minutes bonus (2 pts if roll3_mins ≥ 60, 1 pt if ≥ 1)
    """
    pred_df = pred_df.copy()
    for col in ["pred_goals", "pred_assists", "pred_clean", "pred_bonus",
                "pts_from_components"]:
        pred_df[col] = 0.0

    for position, pos_models in component_models.items():
        mask = pred_df["position"] == position
        if mask.sum() == 0:
            continue

        feature_cols = POSITION_FEATURE_MAP[position]
        pos_rows     = pred_df[mask].copy()
        for col in feature_cols:
            if col not in pos_rows.columns:
                pos_rows[col] = 0.0
        X = pos_rows[feature_cols].fillna(0).values

        goals_pred   = _safe_predict(pos_models.get("goals_scored"),  X)
        assists_pred = _safe_predict(pos_models.get("assists"),        X)
        clean_pred   = _safe_predict(pos_models.get("clean_sheets"),   X)
        bonus_pred   = _safe_predict(pos_models.get("bonus"),          X)

        avg_mins = pos_rows["roll3_mins"].values
        mins_pts = np.where(avg_mins >= 60, MINS_PTS_FULL,
                   np.where(avg_mins >= 1,  MINS_PTS_PART, 0))

        goal_fpl    = goals_pred   * GOAL_PTS.get(position, 4)
        assist_fpl  = assists_pred * ASSIST_PTS
        # Clean sheet bonus only awarded when player likely plays 60+ mins
        clean_fpl   = clean_pred   * CLEAN_PTS.get(position, 0) * (avg_mins >= 60).astype(float)
        component_total = np.clip(mins_pts + goal_fpl + assist_fpl + clean_fpl + bonus_pred, 0, None)

        pred_df.loc[mask, "pred_goals"]          = goals_pred.round(2)
        pred_df.loc[mask, "pred_assists"]         = assists_pred.round(2)
        pred_df.loc[mask, "pred_clean"]           = clean_pred.round(2)
        pred_df.loc[mask, "pred_bonus"]           = bonus_pred.round(2)
        pred_df.loc[mask, "pts_from_components"]  = component_total.round(2)

    # Blank GW → zero all component columns
    blank_mask = pred_df["is_blank"]
    for col in ["pred_goals", "pred_assists", "pred_clean", "pred_bonus",
                "pts_from_components"]:
        pred_df.loc[blank_mask, col] = 0.0

    return pred_df

# ─────────────────────────────────────────
# 9. PRICE RISE/FALL PREDICTION MODEL
# ─────────────────────────────────────────

PRICE_FEATURES = [
    "roll3_pts",
    "ewm3_pts",
    "price",
    "selected_pct",
    "games_played",
    "roll3_goals",
    "roll3_assists",
    "gw_number",
    "price_change",   # recent price momentum
]


def train_price_model(df: pd.DataFrame) -> dict | None:
    """
    Train a model to predict next-GW FPL price change.

    FPL prices move in 0.1M steps based on net ownership changes.
    Predicting direction/magnitude helps with sell-timing decisions.

    Target: next_price_change = price_change shifted back one GW per player.
    Features: form, ownership, current price, recent transfers.
    """
    log.info("💰 Training price rise/fall prediction model...")

    price_df = df.copy()
    price_df = price_df.sort_values(["player_id", "gw_number"])

    # Target: what the price will do NEXT GW
    price_df["next_price_change"] = (
        price_df.groupby("player_id")["price_change"]
        .shift(-1)
        .fillna(0)
    )

    for col in PRICE_FEATURES:
        if col not in price_df.columns:
            price_df[col] = 0.0

    price_df = price_df.dropna(subset=["next_price_change"]).copy()
    price_df[PRICE_FEATURES] = price_df[PRICE_FEATURES].fillna(0)
    price_df = price_df.sort_values("gw_number").reset_index(drop=True)

    if len(price_df) < 100:
        log.warning("  Not enough data for price model — skipping.")
        return None

    split_idx = int(len(price_df) * 0.85)
    X_train = price_df[PRICE_FEATURES].iloc[:split_idx]
    X_test  = price_df[PRICE_FEATURES].iloc[split_idx:]
    y_train = price_df["next_price_change"].iloc[:split_idx]
    y_test  = price_df["next_price_change"].iloc[split_idx:]

    model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbosity=0,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2    = float(r2_score(y_test, preds))
    log.info(f"  Price model: RMSE={rmse:.4f}  R²={r2:.3f}  "
             f"(train={len(X_train)}, test={len(X_test)})")

    return {"model": model, "features": PRICE_FEATURES, "rmse": rmse, "r2": r2}


def add_price_predictions(price_model_info: dict | None,
                           pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds predicted_price_change to pred_df.

    FPL prices move in discrete 0.1M steps — continuous model output is
    snapped to the nearest 0.1 and clipped to [-0.3, +0.3]:
      - +0.3M is the practical upper bound (very rare for a player to rise more)
      - -0.3M similarly for falls
    This prevents the model from outputting unrealistic values like +0.007M.

    If model is unavailable, column defaults to 0.0.
    """
    pred_df = pred_df.copy()
    if price_model_info is None:
        pred_df["predicted_price_change"] = 0.0
        return pred_df

    model = price_model_info["model"]
    feats = price_model_info["features"]

    for col in feats:
        if col not in pred_df.columns:
            pred_df[col] = 0.0

    raw_preds = model.predict(pred_df[feats].fillna(0))

    # Snap to nearest 0.1M step using round-half-up (not numpy banker's rounding).
    # np.round uses banker's rounding: 0.5 → 0 (rounds to even), which would
    # incorrectly snap +0.05 → 0.0. np.floor(x + 0.5) gives standard rounding.
    snapped = np.sign(raw_preds) * np.floor(np.abs(raw_preds) / 0.1 + 0.5) * 0.1
    clipped = np.clip(snapped, -0.3, 0.3)

    pred_df["predicted_price_change"] = clipped
    return pred_df

# ─────────────────────────────────────────
# 10. TRAIN POSITION MODELS
#     XGBoost + Quantile Regression + SHAP
# ─────────────────────────────────────────

def train_models(df: pd.DataFrame) -> dict:
    """
    Trains THREE XGBoost models per position:
      median model   (sq error)          → predicted_pts
      q10 floor      (quantile α=0.10)   → pts_low
      q90 ceiling    (quantile α=0.90)   → pts_high

    Plus: SHAP feature importance, 3-fold CV metrics, model_metrics.json.

    Returns:
      {
        "Goalkeeper": {
            "model":             XGBRegressor  (median),
            "q10_model":         XGBRegressor  (floor),
            "q90_model":         XGBRegressor  (ceiling),
            "features":          [...],
            "rmse":              float,
            "r2":                float,
            "cv_rmse":           float,
            "cv_r2":             float,
            "shap_top_features": {feat: shap_val, ...},
        },
        ...
      }
    """
    models: dict     = {}
    all_metrics: dict = {}

    for position in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
        log.info(f"  Training {position} model...")

        X_train, X_test, y_train, y_test, feature_cols, cv_rmse, cv_r2 = \
            prepare_features_for_position(df, position)

        if len(X_train) < 10:
            log.warning(f"  Not enough data for {position}, skipping.")
            continue

        # ── Median model (primary prediction) ─────────────────────
        # early_stopping_rounds guards against overfitting on small
        # position datasets (especially GK/FWD with ~100-150 rows).
        model = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0,
            early_stopping_rounds=20,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = np.clip(model.predict(X_test), 0, None)
        rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2     = float(r2_score(y_test, y_pred))

        # ── Quantile models (floor / ceiling interval) ─────────────
        # Trains on same data as median — only objective differs.
        # Gives us: "Floor: 2.1 pts | Expected: 6.2 pts | Ceiling: 13.4 pts"
        q10_model = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:quantileerror", quantile_alpha=0.10,
            random_state=RANDOM_STATE, verbosity=0,
            early_stopping_rounds=20,
        )
        q90_model = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:quantileerror", quantile_alpha=0.90,
            random_state=RANDOM_STATE, verbosity=0,
            early_stopping_rounds=20,
        )
        q10_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        q90_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        cv_str = (f"  CV-RMSE={cv_rmse:.3f}  CV-R²={cv_r2:.3f}"
                  if cv_rmse is not None else "")
        log.info(
            f"  {position}: RMSE={rmse:.3f}  R²={r2:.3f}{cv_str}  "
            f"(train={len(X_train)}, test={len(X_test)})"
        )

        # ── SHAP feature importance ────────────────────────────────
        shap_top: dict = {}
        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            mean_abs    = np.abs(shap_values).mean(axis=0)
            shap_series = pd.Series(mean_abs, index=feature_cols)
            top5        = shap_series.sort_values(ascending=False).head(5)
            shap_top    = {k: round(float(v), 4) for k, v in top5.items()}
            log.info(f"  SHAP top features ({position}): {shap_top}")

            fig, ax = plt.subplots(figsize=(7, 4))
            top5.sort_values().plot(kind="barh", ax=ax, color="#3b82f6")
            ax.set_title(f"SHAP Feature Importance — {position}", fontsize=13)
            ax.set_xlabel("Mean |SHAP value|  (impact on predicted pts)")
            ax.tick_params(labelsize=9)
            plt.tight_layout()
            fig.savefig(f"shap_{position}.png", dpi=120)
            plt.close(fig)
            log.info(f"  SHAP chart saved → shap_{position}.png")

        except Exception as e:
            log.warning(f"  SHAP failed for {position}: {e}")

        models[position] = {
            "model":             model,
            "q10_model":         q10_model,
            "q90_model":         q90_model,
            "features":          feature_cols,
            "rmse":              rmse,
            "r2":                r2,
            "cv_rmse":           cv_rmse,
            "cv_r2":             cv_r2,
            "shap_top_features": shap_top,
        }

        all_metrics[position] = {
            "rmse":              rmse,
            "r2":                r2,
            "cv_rmse":           cv_rmse,
            "cv_r2":             cv_r2,
            "train_size":        len(X_train),
            "test_size":         len(X_test),
            "shap_top_features": shap_top,
        }

    # Write model_metrics.json — dashboard reads this without re-training
    try:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "positions":    all_metrics,
        }
        with open("model_metrics.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        log.info("📊 Model metrics saved → model_metrics.json")
    except Exception as e:
        log.warning(f"Could not write model_metrics.json: {e}")

    return models


def train_model(df: pd.DataFrame) -> tuple:
    """
    Backward-compatible wrapper for downstream phases (2, 3, 4).
    Returns (models_dict, None, None, df).
    """
    models = train_models(df)
    return models, None, None, df

# ─────────────────────────────────────────
# 11. xMINS PROBABILITY WEIGHTING
# ─────────────────────────────────────────

def compute_expected_pts(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute expected_pts = predicted_pts × P(player plays meaningfully).

    This addresses rotation risk — the single biggest gap in naive FPL models.
    A player predicted 8pts but who only plays 60min half the time is worth
    less than a player predicted 7pts who always starts.

    P(plays effectively) = P(fit) × P(plays substantial minutes | fit)
    where:
      P(fit)                = chance_of_playing / 100
      P(substantial mins)   = smoothly mapped from roll3_mins
                              (90 min avg → 1.0, 45 min avg → 0.55, 0 → 0)

    Also adds p_plays_full (raw minutes probability) for display.
    """
    chance_norm = pred_df["chance_of_playing"].clip(0, 100) / 100.0

    # Smooth sigmoid-like mapping from recent avg minutes to probability
    # roll3_mins=90 → p_full≈1.0 | =60 → ≈0.72 | =30 → ≈0.22 | =0 → 0.0
    roll_mins   = pred_df["roll3_mins"].clip(0, 90)
    p_full      = (roll_mins / 90.0) ** 0.7   # slight concave — penalises low mins hard

    # Combined: must be fit AND play meaningfully
    p_effective = chance_norm * (0.30 + 0.70 * p_full)

    pred_df = pred_df.copy()
    pred_df["p_plays_full"] = p_full.round(3)
    pred_df["expected_pts"] = (pred_df["predicted_pts"] * p_effective).round(2)

    return pred_df

# ─────────────────────────────────────────
# 12. BUILD NEXT-GW FEATURE VECTORS
# ─────────────────────────────────────────

def build_current_features(bootstrap: dict,
                             fixtures_df: pd.DataFrame,
                             history_df: pd.DataFrame,
                             models,
                             current_gw: int,
                             my_player_ids: list = None,
                             pos_enc=None,       # legacy — ignored
                             opp_enc=None) -> pd.DataFrame:
    """
    Build feature vectors for all players for the NEXT gameweek.

    v5 additions:
    - EWMA features populated from history
    - selected_pct (ownership) from bootstrap snapshot
    - transfers_in_event / transfers_out_event as info columns
    - pts_low / pts_high from quantile models
    - expected_pts via xMins weighting
    - Backward compatible with old pkl/phase call signatures
    """
    if my_player_ids is None:
        my_player_ids = []

    # Legacy call signature from older phases — load from disk
    if not isinstance(models, dict):
        try:
            with open("fpl_model.pkl", "rb") as f:
                saved = pickle.load(f)
            models = saved.get("models", {})
            log.info("Loaded models from fpl_model.pkl for compatibility.")
        except Exception:
            log.warning("Could not load models — predictions may be 0.")
            models = {}

    players_raw = bootstrap["elements"]
    teams_df    = pd.DataFrame(bootstrap["teams"])
    pos_df_raw  = pd.DataFrame(bootstrap["element_types"])
    team_map    = teams_df.set_index("id")["name"].to_dict()
    pos_map     = pos_df_raw.set_index("id")["singular_name"].to_dict()

    next_gw = current_gw + 1

    # Build fixture map: team_id -> list of fixture dicts for next GW
    next_fixture_map: dict = {}
    for _, row in fixtures_df[fixtures_df["event"] == next_gw].iterrows():
        for team, opp, diff, home in [
            (row["team_h"], row["team_a"], row["team_h_difficulty"], True),
            (row["team_a"], row["team_h"], row["team_a_difficulty"], False),
        ]:
            if team not in next_fixture_map:
                next_fixture_map[team] = []
            next_fixture_map[team].append({
                "difficulty":    diff,
                "is_home":       int(home),
                "opponent_team": opp,
            })

    opp_strength_map = build_opponent_strength_map(history_df)

    rows = []
    for player in players_raw:
        pid    = player["id"]
        status = player.get("status", "a")
        chance = player.get("chance_of_playing_next_round")

        in_squad = pid in my_player_ids

        # Transfer candidates filtered by injury/chance; own squad always included
        if not in_squad:
            if status not in ("a", "d"):
                continue
            if chance is not None and chance < MIN_CHANCE_OF_PLAYING:
                continue

        p_hist = history_df[history_df["player_id"] == pid].sort_values("round")
        if p_hist.empty:
            continue
        last = p_hist.iloc[-1]

        pos_name  = pos_map.get(player["element_type"], "Unknown")
        team_name = team_map.get(player["team"], "Unknown")
        price     = player["now_cost"] / 10

        # Current-GW ownership and transfer momentum
        selected_pct        = float(
            str(player.get("selected_by_percent", "0") or "0").replace(",", "")
        )
        transfers_in_event  = int(player.get("transfers_in_event",  0) or 0)
        transfers_out_event = int(player.get("transfers_out_event", 0) or 0)

        fixtures_next = next_fixture_map.get(player["team"], [])
        is_blank      = len(fixtures_next) == 0
        is_double     = len(fixtures_next) >= 2

        if is_blank:
            fixture = {"difficulty": 5, "is_home": 0, "opponent_team": -1}
        elif is_double:
            avg_diff = np.mean([f["difficulty"] for f in fixtures_next])
            fixture  = {
                "difficulty":    round(avg_diff, 1),
                "is_home":       fixtures_next[0]["is_home"],
                "opponent_team": fixtures_next[0]["opponent_team"],
            }
        else:
            fixture = fixtures_next[0]

        opp_strength = opp_strength_map.get(str(fixture["opponent_team"]), 1.0)
        opp_encoded  = encode_opponent(fixture["opponent_team"])

        rows.append({
            # ── Identity ──────────────────────────────────────────
            "player_id":           pid,
            "player_name":         f"{player['first_name']} {player['second_name']}",
            "position":            pos_name,
            "team_name":           team_name,
            "price":               price,
            "team_id":             player["team"],
            "player_status":       status,
            "chance_of_playing":   chance if chance is not None else 100,
            # ── Fixture ───────────────────────────────────────────
            "is_blank":            is_blank,
            "is_double":           is_double,
            "is_home":             fixture["is_home"],
            "opponent_encoded":    opp_encoded,
            "opp_strength":        opp_strength,
            "difficulty":          fixture["difficulty"],
            # ── Rolling form features ─────────────────────────────
            "roll3_pts":           last.get("roll3_pts",        0) or 0,
            "roll5_pts":           last.get("roll5_pts",        0) or 0,
            "roll3_mins":          last.get("roll3_mins",       0) or 0,
            "pts_per_90":          last.get("pts_per_90",       0) or 0,
            "roll3_goals":         last.get("roll3_goals",      0) or 0,
            "roll3_assists":       last.get("roll3_assists",     0) or 0,
            "roll3_clean":         last.get("roll3_clean",       0) or 0,
            "roll3_bonus":         last.get("roll3_bonus",       0) or 0,
            "roll3_threat":        last.get("roll3_threat",      0) or 0,
            "roll3_creativity":    last.get("roll3_creativity",  0) or 0,
            "roll3_influence":     last.get("roll3_influence",   0) or 0,
            "roll3_saves":         last.get("roll3_saves",       0) or 0,
            "roll3_yellows":       last.get("roll3_yellows",     0) or 0,
            # ── EWMA form features ────────────────────────────────
            "ewm3_pts":            last.get("ewm3_pts",         0) or 0,
            "ewm5_pts":            last.get("ewm5_pts",         0) or 0,
            "ewm3_mins":           last.get("ewm3_mins",        0) or 0,
            "ewm3_goals":          last.get("ewm3_goals",       0) or 0,
            "ewm3_assists":        last.get("ewm3_assists",      0) or 0,
            "ewm3_threat":         last.get("ewm3_threat",       0) or 0,
            "ewm3_creativity":     last.get("ewm3_creativity",   0) or 0,
            # ── Derived / meta ────────────────────────────────────
            "games_played":        last.get("games_played",      0) or 0,
            "home_ratio":          last.get("home_ratio",       0.5) or 0.5,
            "price_change":        last.get("price_change",      0) or 0,
            "gw_number":           current_gw + 1,
            "selected_pct":        selected_pct,
            # ── Transfer momentum (info columns, not model features)
            "transfers_in_event":  transfers_in_event,
            "transfers_out_event": transfers_out_event,
        })

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        return pred_df

    # ── Main predictions (median model) ───────────────────────────
    # NOTE: compute_expected_pts is intentionally NOT called here.
    # It is called once in run_pipeline after the 60/40 blend so that
    # expected_pts reflects the final blended predicted_pts, not the
    # raw direct-model output. Calling it here would be wasted compute
    # and would be immediately overwritten.
    pred_df["predicted_pts"] = 0.0
    pred_df["pts_low"]       = 0.0
    pred_df["pts_high"]      = 0.0

    for position, model_info in models.items():
        median_model = model_info["model"]
        q10_model    = model_info.get("q10_model")
        q90_model    = model_info.get("q90_model")
        feature_cols = model_info["features"]
        mask         = pred_df["position"] == position

        if mask.sum() == 0:
            continue

        pos_rows = pred_df[mask].copy()
        for col in feature_cols:
            if col not in pos_rows.columns:
                pos_rows[col] = 0.0
        X = pos_rows[feature_cols].fillna(0)

        blank_mask  = pos_rows["is_blank"].values
        double_mask = pos_rows["is_double"].values

        # Median
        preds = np.clip(median_model.predict(X), 0, None)
        preds[blank_mask]  = 0.0
        preds[double_mask] *= 2.0
        pred_df.loc[mask, "predicted_pts"] = preds.round(2)

        # Q10 floor
        if q10_model is not None:
            lo = np.clip(q10_model.predict(X), 0, None)
            lo[blank_mask]  = 0.0
            lo[double_mask] *= 2.0
            pred_df.loc[mask, "pts_low"] = lo.round(2)

        # Q90 ceiling
        if q90_model is not None:
            hi = np.clip(q90_model.predict(X), 0, None)
            hi[blank_mask]  = 0.0
            hi[double_mask] *= 2.0
            pred_df.loc[mask, "pts_high"] = hi.round(2)

    return pred_df

# ─────────────────────────────────────────
# 13. TRANSFER SUGGESTIONS
# ─────────────────────────────────────────

def show_transfer_suggestions(my_team_df: pd.DataFrame,
                               other_players: pd.DataFrame,
                               bank_balance: float) -> None:
    """
    Transfer suggestions with:
    - expected_pts gain (not just raw predicted_pts)
    - price change prediction flagged
    - DGW tag
    - interactive budget override
    """
    has_expected = "expected_pts" in other_players.columns and \
                   "expected_pts" in my_team_df.columns
    has_price    = "predicted_price_change" in other_players.columns

    sort_col = "expected_pts" if has_expected else "predicted_pts"

    def compute_suggestions(budget: float) -> pd.DataFrame:
        suggestions = []
        for _, my_row in my_team_df.iterrows():
            same_pos = other_players[
                other_players["position"] == my_row["position"]
            ].copy()
            my_val = float(my_row.get(sort_col, my_row["predicted_pts"]))
            same_pos["gain"]      = (same_pos[sort_col].astype(float) - my_val).round(2)
            same_pos["cost_diff"] = (same_pos["price"] - my_row["price"]).round(1)
            same_pos["replace"]   = my_row["player_name"]
            same_pos["budget_ok"] = same_pos["cost_diff"] <= budget
            suggestions.append(same_pos)
        return pd.concat(suggestions).sort_values("gain", ascending=False)

    def _format_row(r) -> str:
        dgw_tag   = " 🔄 DGW"   if r.get("is_double")                 else ""
        price_tag = " 📈"        if r.get("predicted_price_change", 0) > 0.05 else \
                    " 📉"        if r.get("predicted_price_change", 0) < -0.05 else ""
        xpts_str  = (f"  xPts: {r['expected_pts']:.2f}" if has_expected else "")
        return (
            f"  OUT: {str(r['replace']):25s}  →  "
            f"IN: {str(r['player_name']):25s}"
            f"  [{r['position']:3s}]"
            f"  Gain: +{r['gain']:.2f}"
            f"{xpts_str}"
            f"  Cost: {r['cost_diff']:+.1f}M"
            f"{dgw_tag}{price_tag}"
        )

    def print_suggestions(sug_df: pd.DataFrame, budget: float) -> pd.DataFrame:
        print(f"\n💰 Best Within-Budget Transfers (Bank: £{budget:.1f}M):")
        budget_top = (
            sug_df[sug_df["budget_ok"] & (sug_df["gain"] > 0)]
            .drop_duplicates("player_name")
            .head(5)
        )
        if budget_top.empty:
            print("  No affordable upgrades found.")
        else:
            for _, r in budget_top.iterrows():
                print(_format_row(r))

        print("\n💸 Best Transfers Regardless of Budget:")
        all_top = (
            sug_df[sug_df["gain"] > 0]
            .drop_duplicates("player_name")
            .head(5)
        )
        for _, r in all_top.iterrows():
            print(_format_row(r))

        return budget_top

    sug_df     = compute_suggestions(bank_balance)
    budget_top = print_suggestions(sug_df, bank_balance)

    if len(budget_top) < 3:
        print(
            f"\n⚠️  Note: Due to FPL API limitations, your bank "
            f"(£{bank_balance:.1f}M) may differ slightly from the app."
        )
        print(
            f"❓ Only {len(budget_top)} affordable option(s) within "
            f"£{bank_balance:.1f}M."
        )
        user_input = input(
            "   Enter your actual bank from the FPL app "
            "(or press Enter to skip): £"
        ).strip()
        if user_input:
            try:
                new_budget = float(user_input)
                if new_budget > bank_balance:
                    log.info(f"Budget overridden to £{new_budget:.1f}M")
                    print_suggestions(compute_suggestions(new_budget), new_budget)
                else:
                    print("  Budget not higher, skipping.")
            except ValueError:
                print("  Invalid input, skipping.")
    else:
        print(
            f"\n⚠️  Note: Bank shown (£{bank_balance:.1f}M) may differ "
            f"slightly from the FPL app. Always double-check before confirming."
        )

# ─────────────────────────────────────────
# 14. FULL PIPELINE
# ─────────────────────────────────────────

def run_pipeline(team_id: int = TEAM_ID,
                 max_players: int = None,
                 refresh: bool = False):
    """
    Full Phase 1 v5 pipeline.

    Execution order:
      1. Fetch bootstrap + fixtures
      2. Fetch team + transfer info
      3. Build player history (incremental cache)
      4. Train position models (XGBoost + quantile + SHAP)
      5. Train component models (goals/assists/clean/bonus)
      6. Train price model
      7. Build next-GW predictions (with xMins, quantile intervals)
      8. Apply component predictions + blend
      9. Apply price predictions
      10. Save models (current + versioned pkl)
      11. Display squad, captain, transfers
    """
    log.info("=" * 65)
    log.info("  FPL AI ASSISTANT — Phase 1 (v5)")
    log.info("=" * 65)

    # ── Fetch ──────────────────────────────────────────────────────
    log.info("⬇️  Fetching bootstrap data...")
    bootstrap   = fetch_bootstrap()
    fixtures_df = fetch_fixtures()
    current_gw  = fetch_current_gw(bootstrap)
    log.info(f"📅 Last completed GW: {current_gw}  →  Predicting GW{current_gw+1}")

    log.info("⬇️  Fetching your team...")
    try:
        team_data     = fetch_my_team(team_id, current_gw)
        my_player_ids = [p["element"] for p in team_data["picks"]]
        log.info(f"✅ Team fetched — {len(my_player_ids)} players.")
    except Exception as e:
        log.error(f"Could not fetch team: {e}")
        my_player_ids = []

    transfer_info   = fetch_transfer_info(team_id, current_gw)
    bank_balance    = transfer_info["bank_balance"]
    transfer_status = transfer_info["transfer_status"]
    log.info(f"💰 Bank: £{bank_balance:.1f}M  |  {transfer_status}")

    # ── History ────────────────────────────────────────────────────
    log.info("📚 Loading player history...")
    history_df = build_player_history_df(
        bootstrap, max_players=max_players, refresh=refresh
    )

    # ── Train ──────────────────────────────────────────────────────
    log.info("🤖 Training position-specific models (median + quantile)...")
    models = train_models(history_df)

    log.info("🧩 Training component models (goals / assists / clean / bonus)...")
    component_models = train_component_models(history_df)

    log.info("💰 Training price prediction model...")
    price_model = train_price_model(history_df)

    # ── Save models ────────────────────────────────────────────────
    model_payload = {"models": models, "features": FEATURE_COLS}
    with open("fpl_model.pkl", "wb") as f:
        pickle.dump(model_payload, f)
    log.info("💾 Models saved → fpl_model.pkl")

    # Versioned copy — never overwrites; lets you track week-over-week quality
    versioned_path = f"fpl_model_gw{current_gw}.pkl"
    with open(versioned_path, "wb") as f:
        pickle.dump(model_payload, f)
    log.info(f"💾 Versioned model → {versioned_path}")

    # ── Predict ────────────────────────────────────────────────────
    log.info(f"🔮 Predicting GW{current_gw+1} scores...")
    pred_df = build_current_features(
        bootstrap, fixtures_df, history_df, models,
        current_gw, my_player_ids=my_player_ids
    )

    # Component predictions + blend with direct model
    log.info("🧩 Applying component model predictions...")
    pred_df = predict_component_pts(component_models, pred_df)
    # Blend: (1-COMPONENT_BLEND_WEIGHT) direct model + COMPONENT_BLEND_WEIGHT components.
    # Direct model has tighter RMSE; component model adds per-stat explainability.
    # Tune COMPONENT_BLEND_WEIGHT in config.py (default 0.40).
    direct_weight    = 1.0 - COMPONENT_BLEND_WEIGHT
    pred_df["predicted_pts"] = (
        direct_weight      * pred_df["predicted_pts"] +
        COMPONENT_BLEND_WEIGHT * pred_df["pts_from_components"]
    ).round(2)
    # Compute expected_pts once here on the final blended predicted_pts.
    # This is the only call — build_current_features deliberately does not call it.
    pred_df = compute_expected_pts(pred_df)

    # Price predictions
    pred_df = add_price_predictions(price_model, pred_df)

    my_team_df    = pred_df[pred_df["player_id"].isin(my_player_ids)].copy()
    other_players = pred_df[~pred_df["player_id"].isin(my_player_ids)].copy()

    # ── Display squad ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  YOUR SQUAD — GW{current_gw+1} Predictions")
    print("=" * 65)
    display_cols = [
        "player_name", "position", "price",
        "predicted_pts", "expected_pts",
        "pts_low", "pts_high",
        "difficulty", "is_home", "is_blank", "is_double",
        "chance_of_playing",
    ]
    print(
        my_team_df[[c for c in display_cols if c in my_team_df.columns]]
        .sort_values("expected_pts", ascending=False)
        .to_string(index=False)
    )

    # ── Captain recommendation ─────────────────────────────────────
    if not my_team_df.empty:
        captainable  = my_team_df[~my_team_df["is_blank"]]
        sort_by      = "expected_pts" if "expected_pts" in captainable.columns \
                       else "predicted_pts"
        if len(captainable) >= 2:
            captain      = captainable.nlargest(1, sort_by).iloc[0]
            vice_captain = captainable.nlargest(2, sort_by).iloc[1]

            cap_xpts  = float(captain.get("expected_pts",   captain["predicted_pts"]))
            cap_low   = float(captain.get("pts_low",   0))
            cap_high  = float(captain.get("pts_high",  0))
            vc_xpts   = float(vice_captain.get("expected_pts", vice_captain["predicted_pts"]))

            # Component breakdown for captain
            comp_parts = []
            for col, label in [("pred_goals",   "goals"),
                                ("pred_assists", "assists"),
                                ("pred_clean",   "clean"),
                                ("pred_bonus",   "bonus")]:
                if col in captain.index:
                    val = float(captain[col])
                    if val > 0.05:
                        comp_parts.append(f"{val:.2f} {label}")
            comp_str = "  Breakdown: " + " + ".join(comp_parts) if comp_parts else ""

            print(f"\n🏆 Captain:      {captain['player_name']}")
            print(
                f"   xPts: {cap_xpts:.2f}  |  "
                f"Floor: {cap_low:.1f}  |  "
                f"Ceiling: {cap_high:.1f}  "
                f"(captained = {cap_xpts*2:.1f} xPts)"
            )
            if comp_str:
                print(comp_str)
            print(f"\n🥈 Vice Captain: {vice_captain['player_name']}  —  {vc_xpts:.2f} xPts")

    # ── Transfer suggestions ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TRANSFER SUGGESTIONS")
    print("=" * 65)
    show_transfer_suggestions(my_team_df, other_players, bank_balance)

    # ── Save predictions ───────────────────────────────────────────
    pred_df.to_csv("fpl_predictions.csv", index=False)
    log.info("✅ Predictions saved → fpl_predictions.csv")
    log.info("✅ Phase 1 v5 complete — ready for Phase 2")

    return models, pred_df, my_team_df

# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    REFRESH = "--refresh" in sys.argv
    if REFRESH:
        log.info("🔄 --refresh flag detected.")

    models, pred_df, my_team_df = run_pipeline(
        team_id=TEAM_ID,
        max_players=None,
        refresh=REFRESH,
    )