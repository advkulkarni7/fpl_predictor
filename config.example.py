# ─────────────────────────────────────────
# FPL AI Assistant — Configuration Template
# ─────────────────────────────────────────
# INSTRUCTIONS:
#   1. Copy this file and rename it to config.py
#   2. Fill in your actual values below
#   3. Never commit config.py to git (it's in .gitignore)
#
# config.py is your local file with real keys.
# config.example.py is this template — safe to commit to GitHub.

# ── Your FPL team ID ──────────────────────────────────────────────
# Find it in the URL when logged into the FPL website:
# https://fantasy.premierleague.com/entry/YOUR_TEAM_ID/history
TEAM_ID = 0000000

# ── Cache settings ────────────────────────────────────────────────
CACHE_FILE         = "player_history_cache.csv"
CACHE_MAX_AGE_DAYS = 7   # warn if cache is older than this

# ── Model settings ────────────────────────────────────────────────
ROLLING_TRAIN_WINDOW = 10   # train on last N gameweeks only
RANDOM_STATE         = 42

# ── Transfer settings ─────────────────────────────────────────────
MIN_CHANCE_OF_PLAYING = 75  # % threshold for transfer candidates

# ── FPL Squad rules ───────────────────────────────────────────────
SQUAD_SIZE    = 15
MAX_PER_CLUB  = 3
POSITION_LIMITS = {
    "Goalkeeper": 2,
    "Defender":   5,
    "Midfielder": 5,
    "Forward":    3,
}
MAX_DOUBLE_TRANSFER_CANDIDATES = 40

# ── Phase 3 constants ─────────────────────────────────────────────
TRANSFER_LOG_FILE   = "transfer_history.json"
SQUAD_VALUE_LOG     = "squad_value_history.json"
HIT_COST_PTS        = 4
DIFFERENTIAL_THRESH = 15

# ── Fixture run settings ──────────────────────────────────────────
FIXTURE_LOOKAHEAD          = 5
FIXTURE_EASY_THRESHOLD     = 3
FIXTURE_MODERATE_THRESHOLD = 4
CUSTOM_DIFFICULTY_BLEND    = 0.6
CUSTOM_DIFF_WINDOW         = 10
CUSTOM_DIFF_GOALS_WEIGHT   = 0.55
CUSTOM_DIFF_XGC_WEIGHT     = 0.20
CUSTOM_DIFF_CS_WEIGHT      = 0.25
FIXTURE_BLANK_PENALTY      = 6.0

# ── Combined score weights (must sum to 1.0) ──────────────────────
COMBINED_NEXT_GW_WEIGHT   = 0.5
COMBINED_FIXTURE_WEIGHT   = 0.3
COMBINED_DGW_BONUS_WEIGHT = 0.2

# ── Phase 4 — Starting XI settings ───────────────────────────────
VALID_FORMATIONS = [
    (3, 4, 3),
    (3, 5, 2),
    (4, 3, 3),
    (4, 4, 2),
    (4, 5, 1),
    (5, 3, 2),
    (5, 4, 1),
]
CAPTAIN_DGW_MULTIPLIER = 1.5

# ── Logging ───────────────────────────────────────────────────────
LOG_FILE = "fpl_assistant.log"

# ── Phase 7 — AI Analyst API keys ────────────────────────────────
# Get your keys from:
#   Groq:         https://console.groq.com          (free)
#   NewsAPI:      https://newsapi.org               (free — 100 req/day)
#   API-Football: https://rapidapi.com              (free — 100 req/day)
#   Odds API:     https://theoddsapi.com            (free — 500 req/month)
GROQ_API_KEY     = "your_groq_api_key_here"
NEWSAPI_KEY      = "your_newsapi_key_here"
API_FOOTBALL_KEY = "your_rapidapi_key_here"
ODDS_API_KEY     = "your_odds_api_key_here"