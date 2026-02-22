"""
FPL AI Assistant — Phase 7: RAG-Enhanced LLM Analyst (v4)
==========================================================
New sources added in v4:
  - The Odds API   — bookmaker odds with quota manager + 6hr cache
                     odds shift faster than any journalist — detects
                     team news leaks before articles are published
  - Understat      — xG and xA per player (no key needed)
                     enriches context with expected performance data
  - Fantasy Football Scout RSS — FPL-specific injury/team news,
                     more targeted than BBC/Sky for FPL decisions

Previous improvements (v3):
  ALGORITHMIC:
  1. FPL-specific NLP keyword weighting
  2. Contradiction detection
  3. Player mention graph
  4. Source staleness detection

  MISSING FUNCTIONALITY:
  5. Deadline awareness
  6. Memory of past advice
  7. Proactive alerts
  8. Multi-player comparison
  9. FPL community sentiment
  10. Fixture context in news

Install:
  pip install groq feedparser newsapi-python requests understat

Keys in config.py:
  GROQ_API_KEY     = "gsk_..."
  NEWSAPI_KEY      = "..."
  API_FOOTBALL_KEY = "..."
  ODDS_API_KEY     = "..."   (from theoddsapi.com — free tier)
"""

import re
import logging
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np

try:
    import feedparser
    FEEDPARSER_OK = True
except ImportError:
    FEEDPARSER_OK = False

try:
    from newsapi import NewsApiClient
    NEWSAPI_OK = True
except ImportError:
    NEWSAPI_OK = False

try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

try:
    import understat
    import asyncio
    try:
        import nest_asyncio
        nest_asyncio.apply()
        NEST_ASYNCIO_OK = True
    except ImportError:
        NEST_ASYNCIO_OK = False
    UNDERSTAT_OK = True
except ImportError:
    UNDERSTAT_OK = False
    NEST_ASYNCIO_OK = False

try:
    from config import (
        GROQ_API_KEY, NEWSAPI_KEY, API_FOOTBALL_KEY,
        TRANSFER_LOG_FILE, ODDS_API_KEY,
    )
except ImportError:
    GROQ_API_KEY      = ""
    NEWSAPI_KEY       = ""
    API_FOOTBALL_KEY  = ""
    ODDS_API_KEY      = ""
    TRANSFER_LOG_FILE = "transfer_history.json"

log = logging.getLogger(__name__)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

GROQ_MODEL             = "llama-3.3-70b-versatile"
MAX_TOKENS             = 1024
FETCH_TIMEOUT          = 6
MAX_ARTICLES           = 3
MAX_CHAT_HISTORY       = 10
MAX_PLAYERS_IN_CONTEXT = 4
CONTEXT_MAX_CHARS      = 6500
CONCERN_CHANCE_THRESHOLD = 100
EPL_LEAGUE_ID          = "39"
STALE_NEWS_HOURS       = 48     # articles older than this flagged as stale
DEADLINE_URGENT_HOURS  = 3      # hours before deadline = urgent mode
DEADLINE_CAUTION_HOURS = 24     # hours before deadline = caution mode
CACHE_TTL_SECONDS      = 300

RAPIDAPI_HOST         = "api-football-v1.p.rapidapi.com"
RAPIDAPI_LINEUPS_URL  = f"https://{RAPIDAPI_HOST}/v3/lineups"
RAPIDAPI_FIXTURES_URL = f"https://{RAPIDAPI_HOST}/v3/fixtures"
RAPIDAPI_INJURIES_URL = f"https://{RAPIDAPI_HOST}/v3/injuries"

# The Odds API
ODDS_API_BASE_URL     = "https://api.the-odds-api.com/v4/sports"
ODDS_API_SPORT        = "soccer_epl"
ODDS_CACHE_FILE       = "odds_cache.json"
ODDS_USAGE_FILE       = "odds_usage.json"
ODDS_MONTHLY_LIMIT    = 500
ODDS_SAFETY_BUFFER    = 10      # stop at limit - buffer
ODDS_CACHE_HOURS      = 6       # re-fetch after this many hours (normal days)
ODDS_CACHE_MATCHDAY_HOURS = 2   # re-fetch sooner on deadline day

# Understat
UNDERSTAT_CACHE_FILE  = "understat_cache.json"
UNDERSTAT_CACHE_HOURS = 24      # xG data changes once per matchday

GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search"
    "?q={query}&hl=en-GB&gl=GB&ceid=GB:en"
)

RSS_SOURCES = {
    "BBC Sport":             "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "Sky Sports":            "https://www.skysports.com/rss/12040",
    "Fantasy Football Scout":"https://www.fantasyfootballscout.co.uk/feed/",
    "Guardian Football":     "https://www.theguardian.com/football/rss",
    "Reddit FPL":            "https://www.reddit.com/r/FantasyPL.rss",
}

# ── Improvement #1: Weighted FPL keywords ─────────────────────────
# Higher weight = stronger signal about player availability
CONCERN_KEYWORD_WEIGHTS = {
    "ruled out":       1.0,
    "confirmed out":   1.0,
    "not in squad":    1.0,
    "miss":            0.8,
    "suspended":       0.9,
    "red card":        0.9,
    "injury":          0.7,
    "injured":         0.7,
    "withdrawn":       0.8,
    "doubt":           0.5,
    "knock":           0.4,
    "late test":       0.35,
    "late fitness":    0.35,
    "75%":             0.55,
    "50%":             0.7,
    "25%":             0.85,
    "on bench":        0.6,
    "bench":           0.4,
}

CLEAR_KEYWORD_WEIGHTS = {
    "confirmed starter": 1.0,
    "fit and available": 1.0,
    "100%":              0.9,
    "fully fit":         0.9,
    "no concerns":       0.8,
    "fit":               0.6,
    "starting":          0.7,
    "available":         0.5,
    "no injury":         0.8,
    "returns":           0.5,
    "back in training":  0.7,
}

# Common English words that are never player names
_COMMON_WORDS = {
    "should", "would", "could", "captain", "transfer", "bench", "squad",
    "team", "player", "players", "season", "gameweek", "league", "fixture",
    "football", "premier", "injury", "injured", "playing", "starting",
    "about", "think", "recommend", "best", "this", "that", "with",
    "from", "your", "their", "which", "what", "when", "week", "next",
    "differential", "ownership", "wildcard", "freehit", "boost", "triple",
    "double", "blank", "formation", "worth", "taking", "explain", "tell",
    "have", "does", "look", "like", "good", "great", "start", "bring",
    "into", "onto", "mine", "over", "under", "free", "sell", "move",
    "late", "last", "also", "only", "just", "most", "more", "than",
    "will", "been", "were", "they", "them", "then", "some", "with",
}

SYSTEM_PROMPT = """You are an expert Fantasy Premier League (FPL) analyst assistant.
You have access to live player data, injury news, fixture analysis, transfer recommendations,
bookmaker odds movements, xG/xA statistics, and community sentiment.

Rules:
- Give concise, direct, actionable verdicts. Maximum 4 sentences unless asked for more detail.
- NEVER invent or hallucinate statistics. Only reference numbers explicitly in the context.
- Always ground answers in the data given. Say so honestly if data is missing.
- Use FPL-specific language (GW, xPts, DGW, FT, hit, clean sheet, etc.)
- When recommending a captain, state expected points and risk factors.
- When recommending a transfer, state 5GW gain and next GW gain.
- When sources conflict, acknowledge the uncertainty explicitly.
- When odds have moved significantly, treat this as a strong team news signal.
- When deadline is close (<3hrs), prioritise confirmed lineup data over predictions.
- When external sources are unavailable, rely on FPL API data only.
- When xG data is available, use it to identify over/underperforming players.
- Always end with one clear bottom-line recommendation in **bold**.

Tone: Confident, direct, like a trusted FPL pundit — not generic AI advice."""

_response_cache: dict[str, tuple[str, float]] = {}


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def _current_season(bootstrap: dict = None) -> str:
    if bootstrap:
        try:
            events = bootstrap.get("events", [])
            if events:
                first = events[0].get("deadline_time", "")
                if first:
                    year  = int(first[:4])
                    month = int(first[5:7])
                    return str(year if month >= 7 else year - 1)
        except Exception:
            pass
    now = datetime.now()
    return str(now.year if now.month >= 7 else now.year - 1)


def _safe_content(text: str, max_chars: int = 200) -> str:
    return str(text or "")[:max_chars]


def _context_hash(question: str, squad_ids: tuple) -> str:
    key = f"{question.lower().strip()}|{'_'.join(str(i) for i in sorted(squad_ids))}"
    return hashlib.md5(key.encode()).hexdigest()


def _article_age_hours(published_str: str) -> float | None:
    """Parse article timestamp and return age in hours. None if unparseable."""
    if not published_str:
        return None
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    for fmt in formats:
        try:
            pub_dt = datetime.strptime(published_str.strip(), fmt)
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - pub_dt).total_seconds() / 3600
        except Exception:
            continue
    return None


# ─────────────────────────────────────────
# IMPROVEMENT #1: WEIGHTED KEYWORD SCORING
# ─────────────────────────────────────────

def _weighted_signal(content: str) -> tuple[float, float]:
    """
    Compute weighted concern and clear signal scores.
    Returns (concern_score, clear_score) where higher = stronger signal.
    """
    text           = content.lower()
    concern_score  = 0.0
    clear_score    = 0.0

    for kw, weight in CONCERN_KEYWORD_WEIGHTS.items():
        if kw in text:
            concern_score = max(concern_score, weight)

    for kw, weight in CLEAR_KEYWORD_WEIGHTS.items():
        if kw in text:
            clear_score = max(clear_score, weight)

    return concern_score, clear_score


# ─────────────────────────────────────────
# IMPROVEMENT #2: CONTRADICTION DETECTION
# ─────────────────────────────────────────

def detect_contradictions(sources: list[dict]) -> list[str]:
    """
    Compare HIGH trust sources against MEDIUM/LOW trust sources.
    Returns list of contradiction messages to inject into context.
    """
    contradictions = []
    available = [s for s in sources if s.get("available", False)]

    high_sources   = [s for s in available if s.get("trust") == "HIGH"]
    other_sources  = [s for s in available if s.get("trust") != "HIGH"]

    for high in high_sources:
        h_concern, h_clear = _weighted_signal(high.get("content", ""))

        for other in other_sources:
            o_concern, o_clear = _weighted_signal(other.get("content", ""))

            # High trust says fine but lower trust says concern
            if h_clear >= 0.6 and o_concern >= 0.7:
                contradictions.append(
                    f"⚠️ SOURCE CONFLICT: {high['source']} suggests available "
                    f"but {other['source']} reports concern "
                    f"(concern signal: {o_concern:.1f}). Monitor before deadline."
                )
            # High trust says concern but lower trust says fine
            elif h_concern >= 0.7 and o_clear >= 0.6:
                contradictions.append(
                    f"⚠️ SOURCE CONFLICT: {high['source']} flags concern "
                    f"but {other['source']} reports available. "
                    f"FPL API data is more reliable — trust the FPL status."
                )

    return contradictions


# ─────────────────────────────────────────
# IMPROVEMENT #3: PLAYER MENTION GRAPH
# ─────────────────────────────────────────

def build_mention_graph(sources: list[dict],
                         enriched_df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Scan article content for co-mentions of players.
    Returns {player_name: [co_mentioned_players]}.
    Used to detect contextually linked players
    (e.g. "Salah could replace Semenyo") for better captain comparisons.
    """
    all_player_names = enriched_df["player_name"].tolist()
    last_names       = {p.split()[-1].lower(): p for p in all_player_names}
    mention_graph    = {}

    for source in sources:
        content = source.get("content", "").lower()
        mentioned = [full for last, full in last_names.items() if last in content]

        for player in mentioned:
            co = [p for p in mentioned if p != player]
            if co:
                if player not in mention_graph:
                    mention_graph[player] = []
                mention_graph[player].extend(co)

    # Deduplicate
    return {k: list(set(v)) for k, v in mention_graph.items()}


def get_co_mentioned_players(player_name: str,
                              mention_graph: dict) -> list[str]:
    """Get players co-mentioned with the given player."""
    return mention_graph.get(player_name, [])


# ─────────────────────────────────────────
# IMPROVEMENT #4: SOURCE STALENESS
# ─────────────────────────────────────────

def check_source_staleness(sources: list[dict]) -> list[str]:
    """
    Flag sources with articles older than STALE_NEWS_HOURS.
    Returns list of staleness warning strings.
    """
    warnings = []
    for s in sources:
        content = s.get("content", "")
        # Extract timestamps from content like "(Mon, 10 Feb 2025 14:00:00 +0000)"
        ts_matches = re.findall(
            r'\((\w{3},\s+\d{2}\s+\w{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}[^)]*)\)',
            content
        )
        for ts in ts_matches[:1]:
            age = _article_age_hours(ts)
            if age is not None and age > STALE_NEWS_HOURS:
                warnings.append(
                    f"⏰ STALE NEWS: {s['source']} article is "
                    f"{int(age)}hrs old — injury status may have changed."
                )
    return warnings


# ─────────────────────────────────────────
# IMPROVEMENT #5: DEADLINE AWARENESS
# ─────────────────────────────────────────

def get_deadline_status(bootstrap: dict, current_gw: int) -> dict:
    """
    Compute how close we are to the GW deadline.
    Returns dict with hours_remaining, urgency, and advice_mode.
    """
    try:
        events = bootstrap.get("events", [])
        next_event = next(
            (e for e in events if e.get("id") == current_gw + 1), None
        )
        if not next_event:
            return {"hours_remaining": None, "urgency": "UNKNOWN",
                    "advice_mode": "standard", "deadline_str": "Unknown"}

        deadline_str = next_event.get("deadline_time", "")
        deadline_dt  = datetime.fromisoformat(
            deadline_str.replace("Z", "+00:00")
        )
        now          = datetime.now(timezone.utc)
        hours_left   = (deadline_dt - now).total_seconds() / 3600

        if hours_left < 0:
            urgency     = "PASSED"
            advice_mode = "post_deadline"
        elif hours_left <= DEADLINE_URGENT_HOURS:
            urgency     = "URGENT"
            advice_mode = "urgent"
        elif hours_left <= DEADLINE_CAUTION_HOURS:
            urgency     = "CAUTION"
            advice_mode = "caution"
        else:
            urgency     = "COMFORTABLE"
            advice_mode = "standard"

        return {
            "hours_remaining": round(hours_left, 1),
            "urgency":         urgency,
            "advice_mode":     advice_mode,
            "deadline_str":    deadline_str[:16],
        }
    except Exception as e:
        log.warning(f"Deadline status error: {e}")
        return {"hours_remaining": None, "urgency": "UNKNOWN",
                "advice_mode": "standard", "deadline_str": "Unknown"}


def deadline_context_text(deadline_status: dict) -> str:
    """Build deadline context string to inject into LLM prompt."""
    urgency = deadline_status.get("urgency", "UNKNOWN")
    hours   = deadline_status.get("hours_remaining")
    mode    = deadline_status.get("advice_mode", "standard")

    if urgency == "PASSED":
        return "DEADLINE PASSED — transfers locked. Focus captain/lineup advice only."
    elif urgency == "URGENT":
        return (
            f"DEADLINE IN {hours:.1f} HOURS — URGENT. "
            "Prioritise confirmed lineup data. "
            "If player doubt exists, recommend safe alternative immediately."
        )
    elif urgency == "CAUTION":
        return (
            f"Deadline in {hours:.1f} hours. "
            "Monitor injury news closely before confirming transfers."
        )
    else:
        hrs_str = f"{hours:.0f} hours" if hours else "plenty of time"
        return f"Deadline: {deadline_status.get('deadline_str','?')} ({hrs_str} remaining). Standard advice mode."


# ─────────────────────────────────────────
# IMPROVEMENT #6: MEMORY OF PAST ADVICE
# ─────────────────────────────────────────

def load_transfer_memory(current_gw: int) -> str:
    """
    Load past transfer suggestions from transfer_history.json.
    Returns a summary string to inject into LLM context.
    """
    path = Path(TRANSFER_LOG_FILE)
    if not path.exists():
        return ""

    try:
        with open(path, encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        return ""

    if not history:
        return ""

    lines = ["=== PAST TRANSFER ADVICE (Memory) ==="]
    recent = [h for h in history if current_gw - int(h.get("gw", 0)) <= 5]

    for h in sorted(recent, key=lambda x: x.get("gw", 0), reverse=True)[:5]:
        gw        = h.get("gw", "?")
        p_out     = h.get("player_out", "?")
        p_in      = h.get("player_in", "?")
        predicted = h.get("predicted_gain", 0)
        actual    = h.get("actual_gain")
        evaluated = h.get("evaluated", False)

        if evaluated and actual is not None:
            result = "✅ Good" if actual >= predicted * 0.7 else "❌ Miss"
            lines.append(
                f"  GW{gw}: OUT {p_out} → IN {p_in} | "
                f"Predicted +{predicted:.1f} | Actual +{actual:.1f} | {result}"
            )
        else:
            lines.append(
                f"  GW{gw}: OUT {p_out} → IN {p_in} | "
                f"Predicted +{predicted:.1f} | (Pending evaluation)"
            )

    if len(lines) == 1:
        return ""

    return "\n".join(lines)


# ─────────────────────────────────────────
# IMPROVEMENT #7: PROACTIVE ALERTS
# ─────────────────────────────────────────

def generate_proactive_alerts(my_team: pd.DataFrame,
                               xi_result: dict,
                               news_map: dict,
                               chance_map: dict,
                               chip_info: dict,
                               deadline_status: dict,
                               current_gw: int) -> list[dict]:
    """
    Scan squad and upcoming context to generate proactive alerts.
    Each alert has: level (critical/warning/info), title, message.
    Shown at top of AI Analyst page before user asks anything.
    """
    alerts = []

    # 1. Injury alerts — players with low chance of playing
    for _, row in my_team.iterrows():
        chance = chance_map.get(int(row["player_id"]), 100) or 100
        news   = news_map.get(int(row["player_id"]), "") or ""
        if chance is not None and chance < 75:
            alerts.append({
                "level":   "critical",
                "title":   f"🔴 {row['player_name']} at risk",
                "message": f"{chance}% chance of playing. {news[:100] or 'Check FPL app.'}",
            })
        elif chance is not None and chance < 100:
            alerts.append({
                "level":   "warning",
                "title":   f"🟡 {row['player_name']} — monitor",
                "message": f"{chance}% chance. {news[:80] or 'Doubt — watch for late news.'}",
            })

    # 2. Captain has blank
    if xi_result:
        cap = xi_result.get("captain", {})
        cap_name = cap.get("player_name", "")
        cap_row  = my_team[my_team["player_name"] == cap_name]
        if not cap_row.empty:
            if cap_row.iloc[0].get("is_blank_next_gw", False):
                alerts.append({
                    "level":   "critical",
                    "title":   f"🔴 Captain {cap_name} has a BLANK",
                    "message": "Your recommended captain has no fixture. Change captain before deadline.",
                })

    # 3. Deadline urgency
    urgency = deadline_status.get("urgency", "COMFORTABLE")
    hours   = deadline_status.get("hours_remaining")
    if urgency == "URGENT" and hours is not None:
        alerts.append({
            "level":   "critical",
            "title":   f"⏰ Deadline in {hours:.1f} hours",
            "message": "Act now — confirm transfers and captain before the deadline passes.",
        })
    elif urgency == "CAUTION" and hours is not None:
        alerts.append({
            "level":   "warning",
            "title":   f"⏰ Deadline in {hours:.0f} hours",
            "message": "Monitor late injury news before confirming your team.",
        })

    # 4. DGW opportunity with chip available
    dgw_gws       = chip_info.get("dgw_gws", []) if chip_info else []
    available_chips = chip_info.get("available_chips", []) if chip_info else []
    if dgw_gws:
        next_dgw = dgw_gws[0]
        gw_num   = next_dgw.get("gw")
        n_teams  = next_dgw.get("dgw_teams", 0)
        if gw_num == current_gw + 1 or gw_num == current_gw + 2:
            if "Bench Boost" in available_chips:
                alerts.append({
                    "level":   "info",
                    "title":   f"⭐ DGW{gw_num} — Bench Boost available",
                    "message": f"{n_teams} teams with double fixtures. Consider activating Bench Boost.",
                })
            if "Free Hit" in available_chips:
                alerts.append({
                    "level":   "info",
                    "title":   f"⭐ DGW{gw_num} — Free Hit available",
                    "message": f"{n_teams} teams with double fixtures. Free Hit could maximise DGW points.",
                })

    # 5. Low DGW coverage
    dgw_players = int(my_team.get("double_gws", pd.Series(0)).gt(0).sum()) \
                  if "double_gws" in my_team.columns else 0
    if dgw_gws and dgw_players < 3:
        alerts.append({
            "level":   "warning",
            "title":   "📊 Low DGW coverage",
            "message": f"You have {dgw_players} players with DGW fixtures. "
                       "Consider transferring in DGW assets.",
        })

    return alerts


# ─────────────────────────────────────────
# IMPROVEMENT #9: COMMUNITY SENTIMENT
# ─────────────────────────────────────────

def fetch_reddit_sentiment(player_name: str) -> dict:
    """
    Fetch Reddit r/FantasyPL posts mentioning the player.
    Compute a simple sentiment score based on post titles.
    Positive words (captain, buy, essential) vs negative (sell, avoid, bench, doubt).
    """
    if not FEEDPARSER_OK:
        return {"source": "Reddit FPL", "trust": "LOW",
                "content": "feedparser not installed.", "available": False}
    try:
        feed      = feedparser.parse(RSS_SOURCES["Reddit FPL"])
        last_name = player_name.split()[-1].lower()
        pos_words = {"captain", "buy", "essential", "start", "brilliant",
                     "amazing", "haul", "good", "great", "pick", "target"}
        neg_words = {"sell", "avoid", "bench", "doubt", "injury", "miss",
                     "bad", "blank", "rotate", "rotation", "drop", "terrible"}

        pos_score = 0
        neg_score = 0
        mentions  = []

        for entry in feed.entries:
            title   = entry.get("title", "").lower()
            summary = (entry.get("summary", "") or "").lower()
            if last_name in title or last_name in summary:
                words  = set(re.findall(r'\b\w+\b', title))
                pos_score += len(words & pos_words)
                neg_score += len(words & neg_words)
                mentions.append(entry.get("title", "")[:100])

        if not mentions:
            return {"source": "Reddit FPL", "trust": "LOW",
                    "content": f"No Reddit FPL discussions found for {player_name}.",
                    "available": True}

        total = pos_score + neg_score
        if total == 0:
            sentiment_str = "Neutral community sentiment"
        elif pos_score > neg_score * 1.5:
            pct = round(pos_score / total * 100)
            sentiment_str = f"Positive community sentiment ({pct}% positive signals)"
        elif neg_score > pos_score * 1.5:
            pct = round(neg_score / total * 100)
            sentiment_str = f"Negative community sentiment ({pct}% negative signals)"
        else:
            sentiment_str = "Mixed community sentiment"

        content = (
            f"{sentiment_str} across {len(mentions)} posts. "
            f"Sample: {mentions[0]}"
        )
        return {"source": "Reddit FPL", "trust": "LOW",
                "content": content, "available": True, "timestamp": "Live"}
    except Exception as e:
        return {"source": "Reddit FPL", "trust": "LOW",
                "content": f"Reddit fetch failed: {str(e)[:80]}", "available": False}


# ─────────────────────────────────────────
# SOURCE FETCHERS (from v2, kept intact)
# ─────────────────────────────────────────

def fetch_fpl_news(player_name: str,
                   player_id: int,
                   news_map: dict,
                   chance_map: dict) -> dict:
    news   = news_map.get(player_id, "") or ""
    chance = chance_map.get(player_id)
    label  = (
        f"{chance}% chance of playing. {news}" if chance is not None and chance < 100
        else news if news
        else "No injury news. Player available."
    )
    return {"source": "FPL API", "trust": "HIGH", "content": label,
            "chance": chance, "available": True, "timestamp": "Live"}


def fetch_api_football_lineup(player_name: str, team_name: str,
                               current_gw: int, season: str) -> dict:
    if not API_FOOTBALL_KEY:
        return {"source": "API-Football", "trust": "HIGH",
                "content": "API key not configured.", "available": False}
    try:
        headers = {"X-RapidAPI-Key": API_FOOTBALL_KEY,
                   "X-RapidAPI-Host": RAPIDAPI_HOST}
        params  = {"league": EPL_LEAGUE_ID, "season": season,
                   "round": f"Regular Season - {current_gw + 1}"}
        resp = requests.get(RAPIDAPI_FIXTURES_URL, headers=headers,
                            params=params, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        fixtures   = resp.json().get("response", [])
        fixture_id = None
        for fix in fixtures:
            home = fix.get("teams", {}).get("home", {}).get("name", "")
            away = fix.get("teams", {}).get("away", {}).get("name", "")
            if team_name.lower() in home.lower() or team_name.lower() in away.lower():
                fixture_id = fix.get("fixture", {}).get("id")
                break
        if not fixture_id:
            return {"source": "API-Football", "trust": "HIGH",
                    "content": "Fixture not found — lineup not confirmed yet.",
                    "available": True, "timestamp": "Live"}
        lr = requests.get(RAPIDAPI_LINEUPS_URL, headers=headers,
                          params={"fixture": fixture_id}, timeout=FETCH_TIMEOUT)
        lr.raise_for_status()
        for tl in lr.json().get("response", []):
            if team_name.lower() in tl.get("team", {}).get("name", "").lower():
                starters = [p.get("player", {}).get("name", "")
                            for p in tl.get("startXI", [])]
                subs     = [p.get("player", {}).get("name", "")
                            for p in tl.get("substitutes", [])]
                last = player_name.split()[-1].lower()
                in_xi  = any(last in s.lower() for s in starters)
                on_sub = any(last in s.lower() for s in subs)
                content = (
                    f"CONFIRMED STARTER: {player_name} in starting XI." if in_xi
                    else f"ON BENCH: {player_name} named substitute."
                    if on_sub else f"NOT IN SQUAD: {player_name} absent."
                )
                return {"source": "API-Football", "trust": "HIGH",
                        "content": content, "available": True, "timestamp": "Live"}
        return {"source": "API-Football", "trust": "HIGH",
                "content": "Lineup not yet confirmed.", "available": True, "timestamp": "Live"}
    except Exception as e:
        return {"source": "API-Football", "trust": "HIGH",
                "content": f"Lineup fetch failed: {str(e)[:80]}", "available": False}


def fetch_api_football_injuries(team_name: str, current_gw: int,
                                 season: str) -> dict:
    if not API_FOOTBALL_KEY:
        return {"source": "API-Football Injuries", "trust": "HIGH",
                "content": "API key not configured.", "available": False}
    try:
        headers = {"X-RapidAPI-Key": API_FOOTBALL_KEY,
                   "X-RapidAPI-Host": RAPIDAPI_HOST}
        params  = {"league": EPL_LEAGUE_ID, "season": season,
                   "round": f"Regular Season - {current_gw + 1}"}
        resp = requests.get(RAPIDAPI_FIXTURES_URL, headers=headers,
                            params=params, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        fixture_id = None
        for fix in resp.json().get("response", []):
            home = fix.get("teams", {}).get("home", {}).get("name", "")
            away = fix.get("teams", {}).get("away", {}).get("name", "")
            if team_name.lower() in home.lower() or team_name.lower() in away.lower():
                fixture_id = fix.get("fixture", {}).get("id")
                break
        if not fixture_id:
            return {"source": "API-Football Injuries", "trust": "HIGH",
                    "content": "Fixture not found.", "available": True}
        ir = requests.get(RAPIDAPI_INJURIES_URL, headers=headers,
                          params={"fixture": fixture_id}, timeout=FETCH_TIMEOUT)
        ir.raise_for_status()
        injuries = ir.json().get("response", [])
        if not injuries:
            return {"source": "API-Football Injuries", "trust": "HIGH",
                    "content": f"No injuries reported for {team_name}.",
                    "available": True, "timestamp": "Live"}
        snippets = [f"{i.get('player',{}).get('name','?')}: "
                    f"{i.get('player',{}).get('reason','?')}"
                    for i in injuries[:5]]
        return {"source": "API-Football Injuries", "trust": "HIGH",
                "content": f"{team_name} injuries: " + " | ".join(snippets),
                "available": True, "timestamp": "Live"}
    except Exception as e:
        return {"source": "API-Football Injuries", "trust": "HIGH",
                "content": f"Injuries fetch failed: {str(e)[:80]}", "available": False}


def fetch_newsapi_articles(player_name: str,
                            max_articles: int = MAX_ARTICLES) -> dict:
    if not NEWSAPI_OK or not NEWSAPI_KEY:
        return {"source": "NewsAPI", "trust": "MEDIUM",
                "content": "NewsAPI not configured.", "available": False}
    try:
        client   = NewsApiClient(api_key=NEWSAPI_KEY)
        response = client.get_everything(
            q=f"{player_name} football",
            language="en", sort_by="publishedAt", page_size=max_articles,
        )
        articles = response.get("articles", [])
        if not articles:
            return {"source": "NewsAPI", "trust": "MEDIUM",
                    "content": f"No recent articles for {player_name}.", "available": True}
        snippets = []
        for a in articles[:max_articles]:
            title  = a.get("title", "")
            desc   = _safe_content(a.get("description", ""), 120)
            src    = a.get("source", {}).get("name", "")
            pub_at = a.get("publishedAt", "")
            try:
                pub_dt  = datetime.fromisoformat(pub_at.replace("Z", "+00:00"))
                hours   = int((datetime.now(timezone.utc) - pub_dt).total_seconds() / 3600)
                age_str = f"{hours}h ago"
            except Exception:
                age_str = pub_at[:10]
            snippets.append(f'[{src} · {age_str}] "{title}" — {desc}')
        return {"source": "NewsAPI", "trust": "MEDIUM",
                "content": "\n".join(snippets), "available": True, "timestamp": "Live"}
    except Exception as e:
        return {"source": "NewsAPI", "trust": "MEDIUM",
                "content": f"NewsAPI error: {str(e)[:80]}", "available": False}


def fetch_rss_news(player_name: str, source_name: str,
                   feed_url: str, max_items: int = MAX_ARTICLES) -> dict:
    if not FEEDPARSER_OK:
        return {"source": source_name, "trust": "MEDIUM",
                "content": "feedparser not installed.", "available": False}
    try:
        feed      = feedparser.parse(feed_url)
        last_name = player_name.split()[-1].lower()
        matches   = []
        for entry in feed.entries:
            title   = entry.get("title", "")
            summary = entry.get("summary", "") or ""
            if last_name in title.lower() or last_name in summary.lower():
                published = entry.get("published", "")
                matches.append(f'"{title[:120]}" ({published[:16]})')
            if len(matches) >= max_items:
                break
        if not matches:
            return {"source": source_name, "trust": "LOW",
                    "content": f"No mentions of {player_name}.", "available": True}
        return {"source": source_name, "trust": "MEDIUM",
                "content": "\n".join(matches), "available": True, "timestamp": "Live"}
    except Exception as e:
        return {"source": source_name, "trust": "LOW",
                "content": f"RSS failed: {str(e)[:80]}", "available": False}


def fetch_google_news(player_name: str,
                       max_items: int = MAX_ARTICLES) -> dict:
    if not FEEDPARSER_OK:
        return {"source": "Google News", "trust": "LOW",
                "content": "feedparser not installed.", "available": False}
    try:
        query = requests.utils.quote(f"{player_name} FPL Premier League")
        url   = GOOGLE_NEWS_RSS.format(query=query)
        feed  = feedparser.parse(url)
        items = []
        for entry in feed.entries[:max_items]:
            title     = entry.get("title", "")
            published = entry.get("published", "")
            items.append(f'"{title[:120]}" ({published[:16]})')
        if not items:
            return {"source": "Google News", "trust": "LOW",
                    "content": f"No results for {player_name}.", "available": True}
        return {"source": "Google News", "trust": "LOW",
                "content": "\n".join(items), "available": True, "timestamp": "Live"}
    except Exception as e:
        return {"source": "Google News", "trust": "LOW",
                "content": f"Google News error: {str(e)[:80]}", "available": False}


# ─────────────────────────────────────────
# THE ODDS API — QUOTA MANAGER + CACHE
# ─────────────────────────────────────────

def _odds_quota_ok() -> bool:
    """
    Check monthly quota before making an Odds API call.
    Resets counter automatically on the 1st of each month.
    Returns True if quota is available.
    """
    if not ODDS_API_KEY:
        return False
    path      = Path(ODDS_USAGE_FILE)
    now_month = datetime.now().strftime("%Y-%m")
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                usage = json.load(f)
            if usage.get("month") != now_month:
                usage = {"month": now_month, "requests_used": 0,
                         "limit": ODDS_MONTHLY_LIMIT}
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(usage, f)
        else:
            usage = {"month": now_month, "requests_used": 0,
                     "limit": ODDS_MONTHLY_LIMIT}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(usage, f)
        return usage.get("requests_used", 0) < (
            usage.get("limit", ODDS_MONTHLY_LIMIT) - ODDS_SAFETY_BUFFER
        )
    except Exception:
        return False


def _increment_odds_usage():
    """Increment the monthly Odds API request counter."""
    path      = Path(ODDS_USAGE_FILE)
    now_month = datetime.now().strftime("%Y-%m")
    try:
        usage = {"month": now_month, "requests_used": 0,
                 "limit": ODDS_MONTHLY_LIMIT}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                usage = json.load(f)
        usage["requests_used"] = usage.get("requests_used", 0) + 1
        with open(path, "w", encoding="utf-8") as f:
            json.dump(usage, f)
    except Exception:
        pass


def _load_odds_cache() -> dict:
    """Load cached odds data. Returns {} if cache missing or stale."""
    path = Path(ODDS_CACHE_FILE)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
        fetched_at = cache.get("fetched_at", "")
        if fetched_at:
            age_hours = (
                datetime.now(timezone.utc) -
                datetime.fromisoformat(fetched_at)
            ).total_seconds() / 3600
            cache_ttl = (
                ODDS_CACHE_MATCHDAY_HOURS
                if cache.get("is_matchday", False)
                else ODDS_CACHE_HOURS
            )
            if age_hours < cache_ttl:
                return cache.get("data", {})
    except Exception:
        pass
    return {}


def _save_odds_cache(data: dict, is_matchday: bool = False):
    """Save odds data to local cache."""
    try:
        with open(ODDS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "is_matchday": is_matchday,
                "data": data,
            }, f, indent=2)
    except Exception:
        pass


def fetch_odds_batch(squad_player_names: list[str],
                      deadline_status: dict = None) -> dict:
    """
    Batch fetch bookmaker odds for all squad players in ONE API call.
    Caches for 6 hours (2 hours on matchday) to preserve quota.
    Returns dict: {player_last_name: {market: odds_info}}

    Strategy: fetch all EPL match odds, then match player names
    to bookmaker scorer/assist markets. One call covers all 20 teams.
    """
    if not ODDS_API_KEY:
        return {}

    # Check cache first — most calls will hit cache
    cached = _load_odds_cache()
    if cached:
        log.info("Odds data served from cache.")
        return cached

    # Check quota before making real API call
    if not _odds_quota_ok():
        log.warning("Odds API monthly quota exhausted.")
        return {}

    try:
        is_matchday = (
            deadline_status.get("urgency") in ("URGENT", "CAUTION")
            if deadline_status else False
        )

        # Fetch player props (anytime scorer) for EPL
        url    = f"{ODDS_API_BASE_URL}/{ODDS_API_SPORT}/odds"
        params = {
            "apiKey":  ODDS_API_KEY,
            "regions": "uk",
            "markets": "h2h,totals",
            "oddsFormat": "decimal",
        }
        resp = requests.get(url, params=params, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        _increment_odds_usage()

        games   = resp.json()
        odds_by_team: dict[str, dict] = {}

        for game in games:
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            bookmakers = game.get("bookmakers", [])

            for bm in bookmakers[:1]:  # first bookmaker only
                for market in bm.get("markets", []):
                    market_key = market.get("key", "")
                    for outcome in market.get("outcomes", []):
                        name  = outcome.get("name", "")
                        price = outcome.get("price", 0)
                        for team in [home_team, away_team]:
                            last = team.split()[-1].lower()
                            if last not in odds_by_team:
                                odds_by_team[last] = {}
                            odds_by_team[last][market_key] = {
                                "price":       price,
                                "home_team":   home_team,
                                "away_team":   away_team,
                                "bookmaker":   bm.get("title", "?"),
                            }

        # Match squad players to team odds
        result: dict[str, dict] = {}
        for pname in squad_player_names:
            last = pname.split()[-1].lower()
            if last in odds_by_team:
                result[pname] = odds_by_team[last]

        _save_odds_cache(result, is_matchday=is_matchday)
        log.info(f"Odds fetched for {len(result)} players. Cached for "
                 f"{'2hrs (matchday)' if is_matchday else '6hrs'}.")
        return result

    except Exception as e:
        log.warning(f"Odds API batch fetch error: {e}")
        return {}


def get_player_odds_context(player_name: str,
                             odds_data: dict) -> dict:
    """
    Extract odds context for a single player from the batch odds dict.
    Returns source result dict compatible with the rest of the pipeline.
    """
    player_odds = odds_data.get(player_name, {})
    if not player_odds:
        return {"source": "The Odds API", "trust": "MEDIUM",
                "content": f"No odds data for {player_name}.",
                "available": True, "timestamp": "Cached"}

    snippets = []
    for market, info in list(player_odds.items())[:3]:
        price     = info.get("price", "?")
        bookmaker = info.get("bookmaker", "?")
        snippets.append(f"{market}: {price} ({bookmaker})")

    content = f"Current odds for {player_name}: " + " | ".join(snippets)
    return {"source": "The Odds API", "trust": "MEDIUM",
            "content": content, "available": True, "timestamp": "Cached"}


def get_odds_usage_summary() -> str:
    """Return a human-readable summary of Odds API quota usage."""
    path = Path(ODDS_USAGE_FILE)
    if not path.exists():
        return "Odds API: 0/500 requests used this month."
    try:
        with open(path, encoding="utf-8") as f:
            usage = json.load(f)
        used  = usage.get("requests_used", 0)
        limit = usage.get("limit", ODDS_MONTHLY_LIMIT)
        month = usage.get("month", "?")
        return f"Odds API: {used}/{limit} requests used ({month})"
    except Exception:
        return "Odds API: usage data unavailable."


# ─────────────────────────────────────────
# UNDERSTAT — xG / xA DATA
# ─────────────────────────────────────────

def _load_understat_cache() -> dict:
    """Load cached Understat xG data."""
    path = Path(UNDERSTAT_CACHE_FILE)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
        fetched_at = cache.get("fetched_at", "")
        if fetched_at:
            age = (datetime.now(timezone.utc) -
                   datetime.fromisoformat(fetched_at)).total_seconds() / 3600
            if age < UNDERSTAT_CACHE_HOURS:
                return cache.get("data", {})
    except Exception:
        pass
    return {}


def _save_understat_cache(data: dict):
    try:
        with open(UNDERSTAT_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "data": data,
            }, f, indent=2)
    except Exception:
        pass


def fetch_understat_xg(player_name: str,
                        bootstrap: dict = None) -> dict:
    """
    Fetch xG and xA data for a player from Understat.
    No API key required. Cached for 24 hours.

    Fixes:
    - Uses _current_season() — no longer hardcoded to 2024
    - Uses get_players() search instead of dumping full league
    - Uses nest_asyncio for Streamlit event loop compatibility
    """
    if not UNDERSTAT_OK:
        return {"source": "Understat", "trust": "MEDIUM",
                "content": "understat not installed. Run: pip install understat nest_asyncio",
                "available": False}

    # Check cache first
    cache = _load_understat_cache()
    if player_name in cache:
        return {"source": "Understat", "trust": "MEDIUM",
                "content": cache[player_name], "available": True,
                "timestamp": "Cached"}

    season = int(_current_season(bootstrap))

    try:
        async def _fetch():
            async with understat.Understat() as u:
                last = player_name.split()[-1]
                # Search by player name — targeted, not full league dump
                results = await u.get_players_stats(
                    league="EPL",
                    season=season,
                    options={"player_name": last},
                )
                if not results:
                    # Fallback: search in league players with name filter
                    all_players = await u.get_league_players(
                        "EPL", season,
                        options={"player_name": last}
                    )
                    return all_players[0] if all_players else None
                return results[0] if results else None

        # Streamlit-safe event loop execution
        if NEST_ASYNCIO_OK:
            loop   = asyncio.get_event_loop()
            result = loop.run_until_complete(_fetch())
        else:
            loop   = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(_fetch())
            finally:
                loop.close()

        if not result:
            content = f"No Understat data found for {player_name}."
            return {"source": "Understat", "trust": "MEDIUM",
                    "content": content, "available": True}

        xg      = float(result.get("xG", 0))
        xa      = float(result.get("xA", 0))
        goals   = float(result.get("goals", 0))
        assists = float(result.get("assists", 0))
        mins    = float(result.get("time", 0))

        xg_diff   = goals - xg
        xa_diff   = assists - xa
        xg_signal = (
            f"overperforming xG by +{xg_diff:.2f}" if xg_diff > 0.5
            else f"underperforming xG by {xg_diff:.2f}" if xg_diff < -0.5
            else "inline with xG"
        )
        xa_signal = (
            f"overperforming xA by +{xa_diff:.2f}" if xa_diff > 0.5
            else f"underperforming xA by {xa_diff:.2f}" if xa_diff < -0.5
            else "inline with xA"
        )

        content = (
            f"xG: {xg:.2f} (goals: {goals:.0f}, {xg_signal}) | "
            f"xA: {xa:.2f} (assists: {assists:.0f}, {xa_signal}) | "
            f"Minutes: {mins:.0f}"
        )

        cache[player_name] = content
        _save_understat_cache(cache)

        return {"source": "Understat", "trust": "MEDIUM",
                "content": content, "available": True, "timestamp": "Cached"}

    except Exception as e:
        log.warning(f"Understat error for {player_name}: {e}")
        return {"source": "Understat", "trust": "MEDIUM",
                "content": f"Understat fetch failed: {str(e)[:80]}",
                "available": False}


# ─────────────────────────────────────────
# PARALLEL FETCHER
# ─────────────────────────────────────────

def fetch_all_sources(player_name: str, player_id: int, team_name: str,
                      current_gw: int, news_map: dict, chance_map: dict,
                      season: str, force_full: bool = False,
                      odds_data: dict = None,
                      deadline_status: dict = None,
                      bootstrap: dict = None) -> list[dict]:
    """
    Fetch all sources in parallel using ThreadPoolExecutor.
    Now includes Odds API (from batch cache) and Understat xG.
    odds_data is pre-fetched batch dict — passed in to avoid
    re-fetching per player (preserves Odds API quota).
    bootstrap passed through to Understat for dynamic season.
    """
    results = [fetch_fpl_news(player_name, player_id, news_map, chance_map)]

    chance      = chance_map.get(player_id, 100) or 100
    news        = news_map.get(player_id, "") or ""
    has_concern = force_full or chance < CONCERN_CHANCE_THRESHOLD or len(news) > 0

    if not has_concern:
        if odds_data:
            results.append(get_player_odds_context(player_name, odds_data))
        results.append(fetch_understat_xg(player_name, bootstrap))
        return results

    tasks = {
        "api_lineup":    lambda: fetch_api_football_lineup(
            player_name, team_name, current_gw, season),
        "api_injuries":  lambda: fetch_api_football_injuries(
            team_name, current_gw, season),
        "newsapi":       lambda: fetch_newsapi_articles(player_name),
        "google":        lambda: fetch_google_news(player_name),
        "reddit":        lambda: fetch_reddit_sentiment(player_name),
        "understat":     lambda: fetch_understat_xg(player_name, bootstrap),
    }
    for sn, su in list(RSS_SOURCES.items())[:3]:  # BBC, Sky, FFS
        tasks[f"rss_{sn}"] = lambda s=sn, u=su: fetch_rss_news(player_name, s, u)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(futures, timeout=FETCH_TIMEOUT + 2):
            try:
                results.append(future.result(timeout=FETCH_TIMEOUT))
            except Exception as e:
                results.append({"source": futures[future], "trust": "LOW",
                                 "content": "Fetch timed out.", "available": False})

    if odds_data:
        results.append(get_player_odds_context(player_name, odds_data))

    external_ok = [r for r in results[1:] if r.get("available")]
    if not external_ok:
        results.append({"source": "External", "trust": "LOW",
                        "content": "All external sources unavailable. Using FPL API only.",
                        "available": False})
    return results


# ─────────────────────────────────────────
# CONFIDENCE SCORING (weighted v2)
# ─────────────────────────────────────────

def build_source_confidence(sources: list[dict]) -> tuple[str, float]:
    """
    Compute weighted confidence score using keyword weights from improvement #1.
    """
    available = [s for s in sources if s.get("available", False)]
    high      = [s for s in available if s.get("trust") == "HIGH"]
    medium    = [s for s in available if s.get("trust") == "MEDIUM"]

    total_concern = 0.0
    total_clear   = 0.0

    for s in available:
        c_score, cl_score = _weighted_signal(s.get("content", ""))
        total_concern += c_score
        total_clear   += cl_score

    total = len(available)
    if total == 0:
        return "UNKNOWN", 40.0

    dominant_signal = max(total_concern, total_clear)
    base_conf       = min(1.0, dominant_signal / max(total, 1)) * 100
    trust_bonus     = len(high) * 10 + len(medium) * 4
    score           = min(95.0, base_conf + trust_bonus)

    label = "HIGH" if score >= 75 else "MEDIUM" if score >= 50 else "LOW"
    return label, round(score, 1)


def format_sources_display(sources: list[dict],
                            contradictions: list[str] = None,
                            staleness_warnings: list[str] = None) -> str:
    """Format sources with contradiction and staleness warnings."""
    lines = ["**📡 Sources consulted:**"]
    for s in sources:
        icon    = "✅" if s.get("available") else "⚪"
        trust   = s.get("trust", "?")
        content = _safe_content(s.get("content", ""), 100)
        ts      = s.get("timestamp", "")
        lines.append(f"{icon} **{s['source']}** [{trust}] — {content}"
                     f"{f'  _{ts}_' if ts else ''}")

    if contradictions:
        lines.append("\n**⚠️ Source conflicts detected:**")
        for c in contradictions:
            lines.append(f"  {c}")

    if staleness_warnings:
        lines.append("\n**⏰ Staleness warnings:**")
        for w in staleness_warnings:
            lines.append(f"  {w}")

    return "\n".join(lines)


# ─────────────────────────────────────────
# CONTEXT BUILDERS
# ─────────────────────────────────────────

def build_squad_context(my_team: pd.DataFrame, xi_result: dict,
                         bank_balance: float, transfers_made: int,
                         available_chips: list, current_gw: int) -> str:
    def _xpts_local(row) -> float:
        try:
            v = row.get("expected_pts")
            if v is not None and not pd.isna(v):
                return float(v)
        except Exception:
            pass
        try:
            return float(row.get("predicted_pts", 0.0))
        except Exception:
            return 0.0

    lines = [f"=== MY SQUAD — GW{current_gw + 1} ==="]
    lines.append(
        f"Bank: £{bank_balance:.1f}M | "
        f"{'1 Free Transfer' if transfers_made == 0 else 'FT Used'} | "
        f"Chips: {', '.join(available_chips) if available_chips else 'None'}"
    )
    if xi_result:
        cap = xi_result.get("captain", {})
        vc  = xi_result.get("vice_captain", {})
        cap_pts = _xpts_local(cap) if hasattr(cap, "get") else float(cap.get("predicted_pts", 0)) if isinstance(cap, dict) else 0.0
        vc_pts = _xpts_local(vc) if hasattr(vc, "get") else float(vc.get("predicted_pts", 0)) if isinstance(vc, dict) else 0.0
        cap_ev_txt = ""
        if isinstance(cap, dict) and "captain_ev" in cap:
            try:
                cap_ev_txt = f" | Cap EV: {float(cap.get('captain_ev', 0.0)):.1f}"
            except Exception:
                cap_ev_txt = ""
        lines.append(
            f"XI: {xi_result.get('formation','?')} | "
            f"Pred: {xi_result.get('total_predicted_pts',0)} pts | "
            f"Cap: {cap.get('player_name','?')} ({cap_pts:.2f} xPts){cap_ev_txt} | "
            f"VC: {vc.get('player_name','?')} ({vc_pts:.2f} xPts)"
        )
        xi = xi_result.get("starting_xi", pd.DataFrame())
        if not xi.empty:
            lines.append("\nStarting XI:")
            for _, row in xi.sort_values("predicted_pts", ascending=False).iterrows():
                blank = " [BLANK]" if row.get("is_blank_next_gw") else ""
                dgw   = " [DGW]"   if row.get("double_gws", 0) > 0 else ""
                rel_txt = ""
                if "p_plays_full" in row.index:
                    try:
                        rel_txt = f"  Full:{float(row.get('p_plays_full', 1.0)):.0%}"
                    except Exception:
                        rel_txt = ""
                lines.append(
                    f"  {row['player_name']:24s} {row['position']:3s} "
                    f"£{row['price']:.1f}  xPts:{_xpts_local(row):.2f}  "
                    f"Run:{row.get('fixture_run_label','?')}{blank}{dgw}{rel_txt}"
                )
        bench = xi_result.get("bench", pd.DataFrame())
        if bench is not None and not bench.empty:
            lines.append("Bench: " + " | ".join(
                f"{r['player_name']} ({r['position']}, {_xpts_local(r):.1f}xPts)"
                for _, r in bench.iterrows()
            ))
    return "\n".join(lines)


def build_transfer_context(ilp_1, ilp_2, roll_advice, hit_transfers) -> str:
    lines = ["=== TRANSFER ANALYSIS ==="]
    if roll_advice:
        lines.append(f"Advice: {roll_advice.get('recommendation','?')}")
        for r in roll_advice.get("reasons", []):
            lines.append(f"  — {r}")
    if ilp_1 and ilp_1.get("transfers"):
        t = ilp_1["transfers"][0]
        lines.append(
            f"1-Transfer: OUT {t['out_name']} → IN {t['in_name']} | "
            f"5GW +{ilp_1.get('total_gain',0)} | "
            f"Next GW +{ilp_1.get('total_next_gain',0)} | "
            f"Cost {ilp_1.get('total_cost',0):+.1f}M"
        )
    if ilp_2 and ilp_2.get("transfers") and len(ilp_2["transfers"]) == 2:
        t1, t2 = ilp_2["transfers"][0], ilp_2["transfers"][1]
        lines.append(
            f"2-Transfer: {t1['out_name']}→{t1['in_name']} + "
            f"{t2['out_name']}→{t2['in_name']} | "
            f"Total 5GW +{ilp_2.get('total_gain',0)}"
        )
    if hit_transfers:
        h = hit_transfers[0]
        lines.append(
            f"Hit transfer: OUT {h['replace']} → IN {h['player_in']} "
            f"(net after -4: +{h.get('net_value',0):.1f})"
        )
    return "\n".join(lines)


def build_player_context(player_name: str, enriched_df: pd.DataFrame,
                          sources: list[dict], current_gw: int,
                          mention_graph: dict = None) -> str:
    """
    Build player context with:
    - Improvement #10: fixture explicitly linked to injury status
    - Improvement #3: co-mentioned players flagged
    """
    last = player_name.split()[-1]
    rows = enriched_df[
        enriched_df["player_name"].str.contains(last, case=False, na=False)
    ]
    if rows.empty:
        return f"No model data for {player_name}."

    row   = rows.iloc[0]
    lines = [f"=== PLAYER: {row['player_name']} ==="]
    lines.append(
        f"Team: {row.get('team_name','?')} | {row['position']} | "
        f"£{row['price']:.1f}M | Owned: {row.get('ownership_pct','?')}%"
    )
    lines.append(
        f"GW{current_gw+1} pred: {row['predicted_pts']} pts | "
        f"xPts: {round(row['predicted_pts']*(1+float(row.get('roll3_threat',0) or 0)/100),2)} | "
        f"5GW score: {row.get('combined_score',0)}"
    )

    # Improvement #10: Link injury news directly to next fixture
    fpl_source = next((s for s in sources if s.get("source") == "FPL API"), None)
    next_opp   = row.get(f"gw{current_gw+1}_opponent", "?")
    next_diff  = row.get(f"gw{current_gw+1}_difficulty", "?")
    next_home  = "H" if row.get(f"gw{current_gw+1}_home", 0) else "A"

    if fpl_source:
        chance_val = fpl_source.get("chance")
        status_str = (
            f"STATUS: {chance_val}% chance of playing"
            if chance_val is not None and chance_val < 100
            else "STATUS: Available"
        )
        lines.append(
            f"{status_str} | "
            f"Next fixture: {next_opp} ({next_home}) D:{next_diff} — "
            f"{'RISK: If absent, miss this fixture entirely' if chance_val and chance_val < 75 else 'Fixture context noted'}"
        )

    # Fixtures
    gw_parts = []
    for offset in range(1, 6):
        gw   = current_gw + offset
        opp  = row.get(f"gw{gw}_opponent", "?")
        diff = row.get(f"gw{gw}_difficulty", "?")
        home = "H" if row.get(f"gw{gw}_home", 0) else "A"
        gw_parts.append(f"GW{gw}:{opp}({home})D:{diff}")
    lines.append("Fixtures: " + " | ".join(gw_parts))
    lines.append(
        f"Run: {row.get('fixture_run_label','?')} | "
        f"Blanks: {row.get('blank_gws',0)} | "
        f"DGWs: {row.get('double_gws',0)} | "
        f"Momentum: {row.get('momentum_score',3):.2f}"
    )

    # News
    lines.append("News:")
    for s in sources:
        if s.get("available") and s.get("content"):
            lines.append(
                f"  [{s['source']} · {s['trust']}] "
                f"{_safe_content(s['content'], 200)}"
            )

    # Co-mentions (improvement #3)
    if mention_graph:
        co = get_co_mentioned_players(row["player_name"], mention_graph)
        if co:
            lines.append(f"Co-mentioned in articles: {', '.join(co[:3])}")

    return "\n".join(lines)


# ─────────────────────────────────────────
# IMPROVEMENT #8: MULTI-PLAYER COMPARISON
# ─────────────────────────────────────────

def build_comparison_context(player_names: list[str],
                               enriched_df: pd.DataFrame,
                               current_gw: int) -> str:
    """
    Build a structured head-to-head comparison matrix for
    multiple players. Used when question asks about 2+ players.
    """
    if len(player_names) < 2:
        return ""

    lines = ["=== PLAYER COMPARISON MATRIX ==="]
    headers    = ["Metric"] + player_names[:4]
    rows_data  = {}
    player_rows = {}

    for name in player_names[:4]:
        last = name.split()[-1]
        matches = enriched_df[
            enriched_df["player_name"].str.contains(last, case=False, na=False)
        ]
        if not matches.empty:
            player_rows[name] = matches.iloc[0]

    metrics = [
        ("Pred Pts",      lambda r: f"{r['predicted_pts']:.2f}"),
        ("5GW Score",     lambda r: f"{r.get('combined_score',0):.2f}"),
        ("Price",         lambda r: f"£{r['price']:.1f}M"),
        ("Avg Diff",      lambda r: f"{r.get('avg_difficulty',3):.2f}"),
        ("Run",           lambda r: str(r.get('fixture_run_label','?'))),
        ("Blank GWs",     lambda r: str(r.get('blank_gws',0))),
        ("DGWs",          lambda r: str(r.get('double_gws',0))),
        ("Momentum",      lambda r: f"{r.get('momentum_score',3):.2f}"),
        ("Value/£M",      lambda r: f"{r.get('value_score',0):.3f}"),
    ]

    for metric_name, getter in metrics:
        row_vals = [metric_name]
        for name in player_names[:4]:
            if name in player_rows:
                try:
                    row_vals.append(getter(player_rows[name]))
                except Exception:
                    row_vals.append("?")
            else:
                row_vals.append("N/A")
        lines.append("  " + " | ".join(f"{v:12s}" for v in row_vals))

    # Verdict
    if len(player_rows) >= 2:
        best = max(player_rows.items(),
                   key=lambda x: float(x[1].get("combined_score", 0)))
        lines.append(f"\nModel verdict: {best[0]} leads on 5GW combined score "
                     f"({float(best[1].get('combined_score',0)):.2f})")

    return "\n".join(lines)


def detect_question_intent(question: str) -> dict:
    q_lower = question.lower()
    intent  = {
        "about_captain":      any(kw in q_lower for kw in [
            "captain", "armband", "cap", "vc", "vice captain"]),
        "about_transfer":     any(kw in q_lower for kw in [
            "transfer", "buy", "sell", "bring in", "move", "swap",
            "replace", "upgrade", "downgrade"]),
        "about_injury":       any(kw in q_lower for kw in [
            "injury", "injured", "fit", "doubt", "miss", "available",
            "concern", "knock", "late", "news", "status"]),
        "about_fixture":      any(kw in q_lower for kw in [
            "fixture", "difficulty", "opponent", "run", "schedule",
            "blank", "double", "dgw", "fdr"]),
        "about_squad":        any(kw in q_lower for kw in [
            "squad", "team", "bench", "formation", "xi", "lineup", "starting"]),
        "about_chip":         any(kw in q_lower for kw in [
            "chip", "wildcard", "free hit", "bench boost", "triple captain",
            "tc", "bb", "wc", "boost"]),
        "about_differential": any(kw in q_lower for kw in [
            "differential", "ownership", "low owned", "unpopular", "own"]),
        "about_price":        any(kw in q_lower for kw in [
            "price", "rise", "fall", "value", "sell now", "buy now"]),
        "about_hit":          any(kw in q_lower for kw in [
            "hit", "-4", "points hit", "minus four"]),
        "about_comparison":   any(kw in q_lower for kw in [
            "or", "vs", "versus", "compare", "between", "better",
            "prefer", "choose"]),
        "player_names":       [],
    }

    words = question.split()
    for word in words:
        clean = re.sub(r"[^a-zA-Z]", "", word)
        if (len(clean) >= 4 and clean[0].isupper()
                and clean.lower() not in _COMMON_WORDS):
            intent["player_names"].append(clean)

    intent["player_names"] = list(dict.fromkeys(intent["player_names"]))
    return intent


def _prioritised_truncate(context_parts: list[str], max_chars: int) -> str:
    if not context_parts:
        return ""
    result = context_parts[0]
    for part in context_parts[1:]:
        candidate = result + "\n\n" + part
        if len(candidate) <= max_chars:
            result = candidate
        else:
            remaining = max_chars - len(result) - 4
            if remaining > 200:
                result += "\n\n" + part[:remaining] + "\n...[truncated]"
            break
    return result


# ─────────────────────────────────────────
# CONTEXT ASSEMBLER
# ─────────────────────────────────────────

def assemble_context(question: str, my_team, others, enriched_df,
                      xi_result, bank_balance, transfers_made,
                      available_chips, current_gw, news_map, chance_map,
                      season, deadline_status, bootstrap=None,
                      ilp_1=None, ilp_2=None,
                      roll_advice=None, hit_transfers=None) -> tuple[str, list[dict], list[str], list[str]]:
    """
    Returns (context_str, all_sources, contradictions, staleness_warnings).
    """
    intent        = detect_question_intent(question)
    all_sources   = []
    context_parts = []

    # Deadline context
    dl_text = deadline_context_text(deadline_status)

    # Squad context
    context_parts.append(build_squad_context(
        my_team, xi_result, bank_balance, transfers_made,
        available_chips, current_gw
    ))
    context_parts[0] = f"[DEADLINE: {dl_text}]\n\n" + context_parts[0]

    # Transfer memory
    memory = load_transfer_memory(current_gw)
    if memory:
        context_parts.append(memory)

    # Transfer context
    if intent["about_transfer"] or intent["about_captain"] or intent["about_hit"]:
        context_parts.append(build_transfer_context(
            ilp_1, ilp_2, roll_advice, hit_transfers or []
        ))

    # Comparison context
    if intent["about_comparison"] and len(intent["player_names"]) >= 2:
        context_parts.append(build_comparison_context(
            intent["player_names"], enriched_df, current_gw
        ))

    # Player context
    players_to_fetch: set[tuple] = set()

    for name in intent["player_names"]:
        matches = enriched_df[
            enriched_df["player_name"].str.contains(name, case=False, na=False)
        ]
        if not matches.empty:
            r = matches.iloc[0]
            players_to_fetch.add((r["player_name"], int(r["player_id"]),
                                   r.get("team_name", "")))

    if intent["about_captain"] and xi_result:
        xi = xi_result.get("starting_xi", pd.DataFrame())
        if not xi.empty:
            for _, row in xi.nlargest(3, "predicted_pts").iterrows():
                players_to_fetch.add((row["player_name"], int(row["player_id"]),
                                       row.get("team_name", "")))

    if intent["about_injury"]:
        for _, row in my_team.iterrows():
            chance = chance_map.get(int(row["player_id"]), 100) or 100
            news   = news_map.get(int(row["player_id"]), "") or ""
            if chance < 100 or news:
                players_to_fetch.add((row["player_name"], int(row["player_id"]),
                                       row.get("team_name", "")))

    if intent["about_differential"] and not others.empty:
        for _, row in others.nlargest(3, "combined_score").iterrows():
            players_to_fetch.add((row["player_name"], int(row["player_id"]),
                                   row.get("team_name", "")))

    # Pre-fetch odds batch ONCE for all squad players (quota-efficient)
    squad_names = my_team["player_name"].tolist() if hasattr(my_team, "player_name") else []
    odds_data   = fetch_odds_batch(squad_names, deadline_status)

    # Fetch sources & build player contexts
    for pname, pid, tname in list(players_to_fetch)[:MAX_PLAYERS_IN_CONTEXT]:
        sources = fetch_all_sources(pname, pid, tname, current_gw,
                                    news_map, chance_map, season,
                                    force_full=intent["about_injury"],
                                    odds_data=odds_data,
                                    deadline_status=deadline_status,
                                    bootstrap=bootstrap)
        all_sources.extend(sources)

        # Build mention graph for co-mention detection
        mention_graph = build_mention_graph(sources, enriched_df)

        context_parts.append(build_player_context(
            pname, enriched_df, sources, current_gw, mention_graph
        ))

    # Run contradiction and staleness detection
    contradictions      = detect_contradictions(all_sources)
    staleness_warnings  = check_source_staleness(all_sources)

    # Inject contradictions into context
    if contradictions:
        context_parts.append(
            "=== SOURCE CONFLICTS ===\n" + "\n".join(contradictions)
        )

    full_context = _prioritised_truncate(context_parts, CONTEXT_MAX_CHARS)
    return full_context, all_sources, contradictions, staleness_warnings


# ─────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────

def call_groq(question: str, context: str,
              chat_history: list[dict] = None,
              stream: bool = False):
    if not GROQ_OK:
        return "Groq not installed. Run: pip install groq"
    if not GROQ_API_KEY:
        return "GROQ_API_KEY not set in config.py"
    try:
        client   = Groq(api_key=GROQ_API_KEY)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if chat_history:
            for msg in chat_history[-(MAX_CHAT_HISTORY * 2):]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({
            "role":    "user",
            "content": f"LIVE DATA CONTEXT:\n{context}\n\n---\n\nQUESTION: {question}",
        })
        if stream:
            return client.chat.completions.create(
                model=GROQ_MODEL, messages=messages,
                max_tokens=MAX_TOKENS, temperature=0.3, stream=True,
            )
        else:
            resp = client.chat.completions.create(
                model=GROQ_MODEL, messages=messages,
                max_tokens=MAX_TOKENS, temperature=0.3, stream=False,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"Groq error: {e}")
        return f"LLM error: {str(e)[:200]}"


# ─────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────

def _get_cached(key: str) -> str | None:
    if key in _response_cache:
        resp, ts = _response_cache[key]
        if (datetime.now(timezone.utc).timestamp() - ts) < CACHE_TTL_SECONDS:
            return resp
        del _response_cache[key]
    return None


def _set_cached(key: str, resp: str):
    _response_cache[key] = (resp, datetime.now(timezone.utc).timestamp())


# ─────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────

def run_analyst(question: str, my_team, others, enriched_df,
                xi_result, bank_balance, transfers_made,
                available_chips, current_gw, news_map, chance_map,
                bootstrap=None, chat_history=None, ilp_1=None,
                ilp_2=None, roll_advice=None, hit_transfers=None,
                use_stream=False) -> dict:
    """
    Main entry point. Returns full result dict.
    """
    season          = _current_season(bootstrap)
    deadline_status = get_deadline_status(bootstrap or {}, current_gw)

    squad_ids  = tuple(my_team["player_id"].tolist()) if hasattr(my_team, "player_id") else ()
    cache_key  = _context_hash(question, squad_ids)
    cached_ans = _get_cached(cache_key)

    if cached_ans and not use_stream:
        return {
            "answer":           cached_ans,
            "sources":          [],
            "confidence":       ("HIGH", 85.0),
            "source_display":   "*(Served from cache — same question within 5 mins)*",
            "context_used":     "",
            "contradictions":   [],
            "staleness":        [],
            "deadline_status":  deadline_status,
            "cached":           True,
        }

    context, sources, contradictions, staleness = assemble_context(
        question=question, my_team=my_team, others=others,
        enriched_df=enriched_df, xi_result=xi_result,
        bank_balance=bank_balance, transfers_made=transfers_made,
        available_chips=available_chips, current_gw=current_gw,
        news_map=news_map, chance_map=chance_map, season=season,
        deadline_status=deadline_status, bootstrap=bootstrap,
        ilp_1=ilp_1, ilp_2=ilp_2,
        roll_advice=roll_advice, hit_transfers=hit_transfers,
    )

    answer = call_groq(question, context, chat_history, stream=use_stream)

    if not use_stream and isinstance(answer, str):
        _set_cached(cache_key, answer)

    conf_label, conf_score = build_source_confidence(sources)
    source_display = format_sources_display(sources, contradictions, staleness)

    return {
        "answer":          answer,
        "sources":         sources,
        "confidence":      (conf_label, conf_score),
        "source_display":  source_display,
        "context_used":    context,
        "contradictions":  contradictions,
        "staleness":       staleness,
        "deadline_status": deadline_status,
        "odds_usage":      get_odds_usage_summary(),
        "cached":          False,
    }


# ─────────────────────────────────────────
# STATUS FLAGS (for dashboard status bar)
# ─────────────────────────────────────────

ANALYST_STATUS = {
    "groq":       GROQ_OK,
    "newsapi":    NEWSAPI_OK,
    "feedparser": FEEDPARSER_OK,
    "understat":  UNDERSTAT_OK,
    "odds_api":   bool(ODDS_API_KEY),
}


# ─────────────────────────────────────────
# QUICK QUESTIONS
# ─────────────────────────────────────────

QUICK_QUESTIONS = [
    "Who should I captain this week?",
    "Any injury concerns in my squad?",
    "Should I use my free transfer or roll it?",
    "Who are the best differentials under £6M?",
    "Is it worth taking a -4 hit this week?",
    "Which players should I target for the DGW?",
    "Explain the top transfer recommendation",
    "How does my fixture run look for the next 5 GWs?",
    "Should I activate my bench boost this week?",
    "Who is rising in price that I should buy now?",
]
