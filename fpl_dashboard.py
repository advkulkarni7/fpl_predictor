"""
FPL AI Assistant ” Phase 6: Streamlit Dashboard
================================================
Interactive web dashboard bringing all phases together.

6 Pages:
  1. My Squad      ” pitch layout, KPI cards, injury flags
  2. Fixture Planner ” interactive colour-coded heatmap
  3. Transfer Planner ” ILP suggestions, before/after XI preview
  4. Scout             ” scatter plots, filters, side-by-side comparison
  5. Captain Picker   ” xPts ranking, chip detection, DGW awareness
  6. Season Tracker   ” squad value trend, transfer accuracy

Run:
  streamlit run fpl_dashboard.py

Install dependencies first:
  pip install streamlit plotly
"""

import os
import sys
import json
import logging
import importlib.util
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LAST_TEAM_ID_PATH = Path(".streamlit") / "last_team_id.txt"
logger = logging.getLogger(__name__)


def _load_last_team_id() -> int | None:
    """
    Read team ID from the URL query parameter ?team_id=XXXXX.

    st.query_params is scoped to the individual user's browser tab — each user
    has their own URL, so there is no sharing between users. The ID persists as
    long as the user keeps the URL (or bookmarks it), giving them the convenience
    of not re-entering it every visit without any cross-user leakage.

    Returns None if the parameter is absent or invalid.
    """
    try:
        raw = st.query_params.get("team_id", "")
        digits_only = "".join(ch for ch in str(raw).strip() if ch.isdigit())
        if digits_only and int(digits_only) > 0:
            return int(digits_only)
    except Exception:
        pass
    return None


def _save_last_team_id(team_id: int | str):
    """
    Persist team ID to the URL query parameter ?team_id=XXXXX.

    This is per-user (scoped to their browser tab/URL) and survives page
    refreshes. The user can bookmark the URL to skip the entry gate on return.
    No server-side file is written — no cross-user leakage.
    """
    try:
        digits_only = "".join(ch for ch in str(team_id).strip() if ch.isdigit())
        if digits_only and int(digits_only) > 0:
            st.query_params["team_id"] = digits_only
    except Exception:
        pass

try:
    from fpl_phase1_model import (
        fetch_bootstrap, fetch_fixtures, fetch_current_gw,
        fetch_my_team, fetch_transfer_info,
        build_player_history_df, build_current_features,
        train_models, FEATURE_COLS,
    )
    from fpl_phase2_fixtures import (
        build_custom_difficulty, build_team_form,
        build_opponent_scoring_map, build_chip_status,
        build_fixture_run, build_player_fixture_scores,
        FIXTURE_LOOKAHEAD,
    )
    from fpl_phase3_constraints import (
        validate_squad, get_ilp_optimal_transfers,
        get_valid_double_transfers, get_hit_transfer_analysis,
        get_rolling_transfer_advice, get_differential_picks,
        get_squad_value_breakdown, track_squad_value,
    )
    from fpl_phase4_optimizer import (
        optimize_xi_ilp, score_all_formations,
        compute_score_range, get_rmse_from_models,
        xpts_captain_score,
    )
    try:
        from config import TEAM_ID, VALID_FORMATIONS, POSITION_LIMITS
    except ImportError:
        # Streamlit Cloud-safe fallback: load repo-tracked config.example.py if config.py is absent.
        _cfg_example = Path(__file__).with_name("config.example.py")
        if _cfg_example.exists():
            _spec = importlib.util.spec_from_file_location("_fpl_config_example", str(_cfg_example))
            if _spec and _spec.loader:
                _cfg_mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_cfg_mod)
                TEAM_ID = int(getattr(_cfg_mod, "TEAM_ID", 9179961))
                VALID_FORMATIONS = getattr(_cfg_mod, "VALID_FORMATIONS", [
                    (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2),
                    (4, 5, 1), (5, 3, 2), (5, 4, 1),
                ])
                POSITION_LIMITS = getattr(_cfg_mod, "POSITION_LIMITS", {
                    "Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3,
                })
            else:
                raise ImportError("Could not load config.example.py")
        else:
            TEAM_ID = 9179961
            VALID_FORMATIONS = [
                (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2),
                (4, 5, 1), (5, 3, 2), (5, 4, 1),
            ]
            POSITION_LIMITS = {
                "Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3,
            }
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Optional advanced analytics imports (capability-gated to preserve compatibility)
try:
    from fpl_phase1_model import (
        train_component_models, predict_component_pts,
        add_price_predictions, train_price_model, compute_expected_pts,
        COMPONENT_BLEND_WEIGHT,
    )
    HAS_ADV_COMPONENT_PIPELINE = True
    HAS_PRICE_MODEL = True
except ImportError:
    train_component_models = None
    predict_component_pts = None
    add_price_predictions = None
    train_price_model = None
    compute_expected_pts = None
    COMPONENT_BLEND_WEIGHT = 0.40
    HAS_ADV_COMPONENT_PIPELINE = False
    HAS_PRICE_MODEL = False

try:
    from fpl_phase2_fixtures import build_cs_probability_map
    HAS_CS_PROB_MAP = True
except ImportError:
    build_cs_probability_map = None
    HAS_CS_PROB_MAP = False

try:
    from fpl_phase3_constraints import (
        run_monte_carlo_captain,
        get_captaincy_differential_analysis,
        get_horizon_transfer_plan,
        get_double_hit_analysis,
    )
    HAS_CAPTAIN_MC = True
    HAS_CAPTAIN_DIFF = True
    HAS_HORIZON_PLAN = True
    HAS_DOUBLE_HIT = True
except ImportError:
    run_monte_carlo_captain = None
    get_captaincy_differential_analysis = None
    get_horizon_transfer_plan = None
    get_double_hit_analysis = None
    HAS_CAPTAIN_MC = False
    HAS_CAPTAIN_DIFF = False
    HAS_HORIZON_PLAN = False
    HAS_DOUBLE_HIT = False

try:
    from fpl_phase7_analyst import (
        run_analyst, QUICK_QUESTIONS,
        ANALYST_STATUS, generate_proactive_alerts,
        get_deadline_status,
        get_odds_usage_summary,
    )
    ANALYST_AVAILABLE = True
except ImportError as e:
    ANALYST_AVAILABLE = False
    ANALYST_ERROR = str(e)
    ANALYST_STATUS = {}
    get_deadline_status = None

try:
    from db.snapshot_reader import get_latest_ready_snapshot, load_latest_ready_snapshot_bundle
    HAS_SNAPSHOT_META_DB = True
except ImportError as e:
    logger.info("Snapshot metadata DB integration disabled: %s", e)
    get_latest_ready_snapshot = None
    load_latest_ready_snapshot_bundle = None
    HAS_SNAPSHOT_META_DB = False

st.set_page_config(
    page_title="FPL AI Assistant",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg: #090e1a;
    --surface: #111a2e;
    --surface-soft: #0f1730;
    --primary: #27e8a7;
    --accent: #37b6ff;
    --warning: #ffb547;
    --danger: #ff5d73;
    --text: #eaf2ff;
    --muted: #90a2be;
    --line: #1f3459;
    --radius-md: 12px;
    --radius-sm: 10px;
    --shadow: 0 10px 28px rgba(4, 9, 20, 0.55);
}

html, body {
    font-family: 'Syne', sans-serif;
    background: radial-gradient(1200px 600px at 5% -10%, #142243 0%, var(--bg) 45%),
                radial-gradient(900px 500px at 95% 0%, #0f2342 0%, var(--bg) 40%),
                var(--bg);
    color: var(--text);
}

.main .block-container {
    font-family: 'Syne', sans-serif;
}


[data-testid="stAppViewContainer"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1224 0%, #090e1a 100%);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * {
    color: #cfdcf6 !important;
}


.main .block-container {
    padding: 1.4rem 2rem 1.6rem;
    max-width: 1440px;
}

.fpl-card {
    background: linear-gradient(160deg, var(--surface) 0%, var(--surface-soft) 100%);
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    padding: 1rem 1.15rem;
    margin-bottom: 0.9rem;
    box-shadow: var(--shadow);
    transform: translateY(0);
    transition: transform 220ms ease, border-color 220ms ease, box-shadow 220ms ease;
    animation: cardIn 360ms ease both;
}
.fpl-card:hover {
    transform: translateY(-2px);
    border-color: #2b4f87;
    box-shadow: 0 12px 30px rgba(10, 18, 36, 0.65);
}

.kpi-block {
    background: linear-gradient(155deg, #122140 0%, #0f1a33 100%);
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    padding: 0.9rem 1rem;
    text-align: center;
    animation: cardIn 320ms ease both;
}
.kpi-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.64rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--accent);
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--primary);
    line-height: 1;
}
.kpi-delta {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 0.2rem;
}

.section-header {
    font-size: 0.66rem;
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--accent);
    margin-bottom: 0.95rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--line);
}

.transfer-card {
    background: linear-gradient(160deg, #122140 0%, #0d1a30 100%);
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
    transition: border-color 220ms ease, transform 220ms ease;
    animation: cardIn 360ms ease both;
}
.transfer-card:hover {
    transform: translateY(-2px);
    border-color: var(--primary);
}
.transfer-out { color: var(--danger); font-weight: 700; }
.transfer-in  { color: var(--primary); font-weight: 700; }
.transfer-gain {
    font-family: 'Space Mono', monospace;
    font-size: 1.02rem;
    color: var(--primary);
}

.badge-low  { color: var(--danger); font-weight: 700; font-size: 0.7rem; }
.badge-med  { color: var(--warning); font-weight: 700; font-size: 0.7rem; }
.badge-ok   { color: var(--primary); font-weight: 700; font-size: 0.7rem; }

.rec-box {
    background: linear-gradient(160deg, #102a21 0%, #091810 100%);
    border: 1px solid var(--primary);
    border-radius: var(--radius-md);
    padding: 1.05rem 1.2rem;
    margin: 1rem 0;
    box-shadow: var(--shadow);
    animation: cardIn 360ms ease both;
}
.rec-box.warning {
    background: linear-gradient(160deg, #2f2310 0%, #1a1308 100%);
    border-color: var(--warning);
}
.rec-box.danger {
    background: linear-gradient(160deg, #32141d 0%, #1a0d13 100%);
    border-color: var(--danger);
}

.entity-line {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-top: 0.25rem;
}
.team-badge {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    object-fit: contain;
    background: rgba(255, 255, 255, 0.06);
}
.lineup-board {
    position: relative;
    border: 1px solid #1f4d2f;
    border-radius: 14px;
    padding: 1rem 0.9rem 0.9rem;
    background:
        linear-gradient(180deg, rgba(5, 39, 26, 0.94), rgba(2, 26, 17, 0.96)),
        repeating-linear-gradient(
            0deg,
            rgba(255,255,255,0.0) 0,
            rgba(255,255,255,0.0) 59px,
            rgba(95, 169, 118, 0.08) 60px
        );
    box-shadow: 0 10px 28px rgba(2, 14, 8, 0.6);
}
.lineup-row {
    display: flex;
    justify-content: center;
    gap: 0.65rem;
    margin: 0.55rem 0;
    flex-wrap: wrap;
}
.lineup-label {
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    color: #7fd7ff;
    margin-top: 0.35rem;
}
.xi-tile {
    width: clamp(122px, 24vw, 150px);
    min-width: 118px;
    max-width: 100%;
    border: 1px solid #24543c;
    background: linear-gradient(160deg, rgba(11, 34, 23, 0.9), rgba(10, 26, 20, 0.92));
    border-radius: 11px;
    padding: 0.45rem 0.5rem;
    transition: transform 160ms ease, border-color 160ms ease;
}
.xi-tile:hover {
    transform: translateY(-2px);
    border-color: #2cae7f;
}
.xi-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.26rem;
}
.xi-pts {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    border-radius: 999px;
    padding: 0.1rem 0.42rem;
    border: 1px solid transparent;
}
.xi-pts.elite { color: #27e8a7; border-color: rgba(39,232,167,0.45); background: rgba(39,232,167,0.08); }
.xi-pts.good  { color: #37b6ff; border-color: rgba(55,182,255,0.45); background: rgba(55,182,255,0.08); }
.xi-pts.mid   { color: #ffb547; border-color: rgba(255,181,71,0.45); background: rgba(255,181,71,0.08); }
.xi-pts.low   { color: #ff5d73; border-color: rgba(255,93,115,0.45); background: rgba(255,93,115,0.08); }
.xi-name {
    font-size: 0.79rem;
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.xi-meta {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    color: #9dc2e8;
    font-size: 0.68rem;
    margin-top: 0.2rem;
}
.xi-role {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    border-radius: 999px;
    padding: 0.09rem 0.36rem;
    border: 1px solid #2d5a8f;
    color: #a9d9ff;
}
.xi-role.cap { border-color: #ffb547; color: #ffcf88; }
.xi-role.vc { border-color: #37b6ff; color: #9adfff; }
.xi-role.blank { border-color: #ff5d73; color: #ff9eac; }
.xi-role.dgw { border-color: #27e8a7; color: #8dffd9; }
.player-face {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    object-fit: cover;
    border: 1px solid #33588d;
    box-shadow: 0 0 0 1px rgba(13, 22, 40, 0.8), 0 8px 18px rgba(4, 10, 23, 0.6);
    background: #0b1224;
}
.player-face-sm {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    object-fit: cover;
    border: 1px solid #2b4e84;
    background: #0b1224;
}

[data-testid="stTabs"] button {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.84rem;
    color: #8ea4c2;
    transition: color 180ms ease, transform 180ms ease;
}
[data-testid="stTabs"] button:hover {
    color: var(--accent);
    transform: translateY(-1px);
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
}

.dataframe {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
}

.js-plotly-plot {
    border-radius: var(--radius-sm);
    overflow: hidden;
    border: 1px solid var(--line);
}
.js-plotly-plot .modebar {
    display: none !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(160deg, #122140 0%, #0f1a32 100%);
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    padding: 0.72rem 0.95rem;
    box-shadow: var(--shadow);
    transition: transform 180ms ease, border-color 180ms ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: #2e538d;
}
[data-testid="stMetricLabel"] {
    color: var(--accent) !important;
    font-size: 0.7rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--primary) !important;
    font-size: 1.5rem !important;
}
[data-testid="stMetricDelta"] {
    color: var(--muted) !important;
}

[data-testid="stDivider"] {
    border-color: #1a2f54;
}

.skeleton-card {
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    background: linear-gradient(90deg, #0f1730 25%, #152443 50%, #0f1730 75%);
    background-size: 200% 100%;
    animation: shimmer 1.25s ease-in-out infinite;
}
.skeleton-sm { height: 42px; margin: 0.25rem 0; }
.skeleton-md { height: 88px; margin: 0.45rem 0; }
.skeleton-lg { height: 220px; margin: 0.55rem 0; }

@keyframes cardIn {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

@media (max-width: 980px) {
    .main .block-container {
        padding: 1rem 0.8rem 1.3rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
    }
    .xi-tile {
        width: clamp(114px, 40vw, 138px);
        padding: 0.4rem 0.45rem;
    }
    .xi-name { font-size: 0.74rem; }
    .xi-meta { font-size: 0.62rem; }
}

@media (prefers-reduced-motion: reduce) {
    * {
        animation: none !important;
        transition: none !important;
    }
}
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    paper_bgcolor="#090e1a",
    plot_bgcolor="#111a2e",
    font=dict(family="Syne, sans-serif", color="#eaf2ff", size=12),
    xaxis=dict(gridcolor="#1f3459", linecolor="#1f3459", tickcolor="#37b6ff"),
    yaxis=dict(gridcolor="#1f3459", linecolor="#1f3459", tickcolor="#37b6ff"),
    colorway=["#27e8a7", "#37b6ff", "#ffb547", "#ff5d73", "#70d0ff", "#8ae8c7"],
    transition=dict(duration=380, easing="cubic-in-out"),
)

def get_theme_tokens(theme_name: str) -> dict:
    """Return UI theme tokens for light/dark modes."""
    themes = {
        "light": {
            "name": "light",
            "bg": "#F2EFE9",
            "bg_alt": "#EAE5DB",
            "surface": "#FBF9F4",
            "surface_soft": "#F5F1E8",
            "panel": "#BFB48F",
            "sidebar": "#564E58",
            "sidebar_2": "#4E4750",
            "primary": "#904E55",
            "accent": "#6A7480",
            "warning": "#C38B4F",
            "danger": "#B74D54",
            "text": "#252627",
            "muted": "#6E6966",
            "line": "#D6D0C4",
            "line_strong": "#C9C1B2",
            "shadow": "0 8px 20px rgba(37, 38, 39, 0.08)",
            "input_bg": "#F8F4EC",
            "input_text": "#252627",
            "chip_bg": "rgba(144,78,85,0.08)",
            "topbar_bg": "rgba(251, 249, 244, 0.92)",
        },
        "dark": {
            "name": "dark",
            "bg": "#090e1a",
            "bg_alt": "#0f1730",
            "surface": "#111a2e",
            "surface_soft": "#0f1730",
            "panel": "#1f3459",
            "sidebar": "#0b1224",
            "sidebar_2": "#090e1a",
            "primary": "#27e8a7",
            "accent": "#37b6ff",
            "warning": "#ffb547",
            "danger": "#ff5d73",
            "text": "#eaf2ff",
            "muted": "#90a2be",
            "line": "#1f3459",
            "line_strong": "#2b4f87",
            "shadow": "0 10px 28px rgba(4, 9, 20, 0.55)",
            "input_bg": "#111a2e",
            "input_text": "#eaf2ff",
            "chip_bg": "rgba(39,232,167,0.08)",
            "topbar_bg": "rgba(9, 14, 26, 0.92)",
        },
    }
    return themes.get(str(theme_name).lower(), themes["light"])


def build_plotly_theme(tokens: dict) -> dict:
    """Build Plotly theme from current UI tokens."""
    bg = tokens["bg"]
    surface = tokens["surface"]
    text = tokens["text"]
    line = tokens["line"]
    accent = tokens["accent"]
    return dict(
        paper_bgcolor=bg,
        plot_bgcolor=surface,
        font=dict(family="Manrope, Syne, sans-serif", color=text, size=12),
        xaxis=dict(gridcolor=line, linecolor=line, tickcolor=accent, zerolinecolor=line),
        yaxis=dict(gridcolor=line, linecolor=line, tickcolor=accent, zerolinecolor=line),
        colorway=[
            tokens["primary"],
            "#3E7E8A" if tokens["name"] == "light" else "#6FAAB5",
            "#BFB48F",
            tokens["warning"],
            tokens["danger"],
            "#8B98A6",
        ],
        transition=dict(duration=320, easing="cubic-in-out"),
    )


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba(...) for Plotly fills/markers."""
    h = str(hex_color).strip().lstrip("#")
    if len(h) != 6:
        return f"rgba(0,0,0,{float(alpha):.3f})"
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{float(alpha):.3f})"


def inject_global_styles(tokens: dict):
    """Inject theme overrides + shell styling on top of the base CSS."""
    is_light = tokens.get("name") == "light"
    sidebar_text = "#F2EFE9" if is_light else tokens["text"]
    sidebar_hover = "rgba(242,239,233,0.08)" if is_light else "rgba(255,255,255,0.05)"
    active_bg = "rgba(191,180,143,0.16)" if is_light else "rgba(191,180,143,0.12)"
    active_border = tokens["panel"] if is_light else tokens["primary"]
    card_hover = "rgba(144,78,85,0.22)" if is_light else "rgba(191,180,143,0.22)"
    rec_warning_bg = "rgba(195,139,79,0.10)" if is_light else "rgba(213,164,106,0.10)"
    rec_danger_bg = "rgba(183,77,84,0.10)" if is_light else "rgba(211,108,115,0.11)"
    table_header_bg = "rgba(86,78,88,0.06)" if is_light else "rgba(255,255,255,0.03)"
    table_row_hover = "rgba(144,78,85,0.05)" if is_light else "rgba(191,180,143,0.05)"
    dropdown_bg     = "#F4F1EB" if is_light else tokens["surface"]
    dropdown_text   = "#2F3038" if is_light else tokens["text"]
    dropdown_hover  = "rgba(144,78,85,0.14)" if is_light else "#1a2d4f"  # dark: visible navy highlight
    dropdown_hover2 = "rgba(144,78,85,0.22)" if is_light else "#1e3560"  # stronger for selected
    nav_radio_key = "Navigation"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=IBM+Plex+Mono:wght@400;500;700&display=swap');

        :root {{
            --bg: {tokens["bg"]};
            --surface: {tokens["surface"]};
            --surface-soft: {tokens["surface_soft"]};
            --primary: {tokens["primary"]};
            --accent: {tokens["accent"]};
            --warning: {tokens["warning"]};
            --danger: {tokens["danger"]};
            --text: {tokens["text"]};
            --muted: {tokens["muted"]};
            --line: {tokens["line"]};
            --shadow: {tokens["shadow"]};
        }}

        html, body, [data-testid="stAppViewContainer"] {{
            color: {tokens["text"]} !important;
            background:
                radial-gradient(900px 400px at 0% 0%, {tokens["bg_alt"]} 0%, {tokens["bg"]} 55%),
                {tokens["bg"]} !important;
            font-family: 'Manrope', 'Syne', sans-serif !important;
            scrollbar-color: {tokens["primary"]} {tokens["surface"]};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {tokens["text"]};
            font-family: 'Manrope', 'Syne', sans-serif !important;
            letter-spacing: -0.015em;
        }}
        p, li {{
            color: {tokens["text"]};
        }}
        a {{
            color: {tokens["primary"]};
        }}
        a:hover {{
            color: {tokens["accent"]};
        }}
        code, pre {{
            font-family: 'IBM Plex Mono', 'Space Mono', monospace !important;
        }}
        code {{
            background: rgba(0,0,0,0.04);
            border: 1px solid {tokens["line"]};
            border-radius: 6px;
            padding: 0.08rem 0.28rem;
        }}
        img {{
            max-width: 100%;
        }}
        *:focus-visible {{
            outline: 2px solid {tokens["primary"]} !important;
            outline-offset: 2px !important;
            border-radius: 6px;
        }}

        .main .block-container {{
            padding: 1.0rem 1.4rem 1.4rem;
            max-width: 1560px;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {tokens["sidebar"]} 0%, {tokens["sidebar_2"]} 100%) !important;
            border-right: 1px solid rgba(255,255,255,0.08) !important;
            color: {sidebar_text} !important;
        }}
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {{
            color: {sidebar_text};
        }}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h5,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h6,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] em {{
            font-family: 'Manrope', 'Syne', sans-serif !important;
        }}
        /* Preserve icon fonts used by Streamlit expanders/material icons */
        [data-testid="stSidebar"] [class*="material"],
        [data-testid="stSidebar"] [class*="icon"],
        [data-testid="stSidebar"] [data-testid="stExpander"] summary svg,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary [role="img"] {{
            font-family: unset !important;
        }}
        [data-testid="stSidebar"] div[data-testid="stRadio"]:has(input[aria-label="{nav_radio_key}"]) > div {{
            gap: 0.25rem;
        }}
        [data-testid="stSidebar"] div[data-testid="stRadio"]:has(input[aria-label="{nav_radio_key}"]) label {{
            border-radius: 10px;
            padding: 0.45rem 0.55rem;
            margin: 0.08rem 0;
            background: transparent;
            border: 1px solid transparent;
            transition: background 160ms ease, border-color 160ms ease;
        }}
        [data-testid="stSidebar"] div[data-testid="stRadio"]:has(input[aria-label="{nav_radio_key}"]) label:hover {{
            background: {sidebar_hover};
            border-color: rgba(255,255,255,0.08);
        }}
        [data-testid="stSidebar"] div[data-testid="stRadio"]:has(input[aria-label="{nav_radio_key}"]) label:has(input:checked) {{
            background: {active_bg};
            border-color: {active_border};
            box-shadow: inset 3px 0 0 {active_border};
        }}
        [data-testid="stSidebar"] div[data-testid="stRadio"]:has(input[aria-label="{nav_radio_key}"]) label p {{
            font-size: 0.86rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
        }}
        [data-testid="stSidebar"] [data-testid="stExpander"] {{
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 12px;
            background: rgba(255,255,255,0.04);
            margin-bottom: 0.35rem;
        }}
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {{
            font-weight: 700;
        }}
        [data-testid="stSidebar"] [data-testid="stExpander"] summary p {{
            margin: 0;
            font-weight: 700 !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] {{
            padding-top: 0.02rem;
            padding-left: 0.45rem;
            padding-right: 0.45rem;
            padding-bottom: 0.28rem;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stCaption {{
            margin-bottom: 0.08rem !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stNumberInput,
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stSelectbox,
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stRadio,
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stToggle,
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stButton {{
            margin-bottom: 0.2rem !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stNumberInput label,
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stSelectbox label,
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stToggle label {{
            font-size: 0.78rem !important;
            margin-bottom: 0.06rem !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stToggle label p {{
            white-space: nowrap !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stNumberInput [data-baseweb="input"] {{
            min-height: 1.95rem !important;
            border-radius: 10px !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stTextInput [data-baseweb="input"] {{
            min-height: 2.05rem !important;
            border-radius: 10px !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .st-key-team_id_row [data-testid="stHorizontalBlock"] {{
            gap: 0 !important;
            align-items: stretch !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .st-key-team_id_row [data-testid="stColumn"] {{
            padding-left: 0 !important;
            padding-right: 0 !important;
            margin: 0 !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .st-key-cfg_team_id_text [data-baseweb="input"] {{
            min-height: 2.1rem !important;
            height: 2.1rem !important;
            border-top-right-radius: 0 !important;
            border-bottom-right-radius: 0 !important;
            border-right-width: 0 !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .st-key-cfg_team_id_text [data-baseweb="input"] > div {{
            min-height: 2.1rem !important;
            height: 2.1rem !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .st-key-cfg_team_id_text [data-baseweb="input"] input {{
            height: 100% !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stNumberInput [data-baseweb="input"] input {{
            font-size: 0.86rem !important;
            padding-top: 0.2rem !important;
            padding-bottom: 0.2rem !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stTextInput [data-baseweb="input"] input {{
            font-size: 0.86rem !important;
            padding-top: 0.2rem !important;
            padding-bottom: 0.2rem !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] [data-baseweb="input"] {{
            background: rgba(0,0,0,0.16) !important;
            border-color: {tokens["line"]} !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] [data-baseweb="input"] input {{
            color: {sidebar_text} !important;
            -webkit-text-fill-color: {sidebar_text} !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] [data-baseweb="select"] *,
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] [data-baseweb="select"] div,
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] [data-baseweb="select"] span {{
            color: {sidebar_text} !important;
            -webkit-text-fill-color: {sidebar_text} !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] [data-baseweb="input"] input::placeholder {{
            color: {tokens["muted"]} !important;
            opacity: 0.95 !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stNumberInput button {{
            min-width: 1.95rem !important;
            width: 1.95rem !important;
            height: 1.95rem !important;
            min-height: 1.95rem !important;
            padding: 0 !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stNumberInput button {{
            display: none !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stNumberInput button p {{
            font-size: 0.95rem !important;
            line-height: 1 !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stNumberInput [data-baseweb="input"] input {{
            padding-right: 0.55rem !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stButton > button {{
            min-height: 2.05rem !important;
            padding-top: 0.24rem !important;
            padding-bottom: 0.24rem !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .st-key-apply_team_id_btn button {{
            min-height: 2.1rem !important;
            height: 2.1rem !important;
            min-width: 3.2rem !important;
            margin-top: 0 !important;
            border-top-left-radius: 0 !important;
            border-bottom-left-radius: 0 !important;
            border-left-width: 0 !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            line-height: 1 !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .st-key-apply_team_id_btn {{
            margin-left: 0 !important;
            display: flex !important;
            align-items: stretch !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .st-key-apply_team_id_btn button p {{
            white-space: nowrap !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.0 !important;
            letter-spacing: 0 !important;
            font-size: 1rem !important;
        }}

        .fpl-shell-topbar {{
            position: sticky;
            top: 0.3rem;
            z-index: 50;
            background: {tokens["topbar_bg"]};
            border: 1px solid {tokens["line"]};
            border-radius: 14px;
            padding: 0.7rem 0.9rem;
            margin-bottom: 0.95rem;
            box-shadow: {tokens["shadow"]};
            backdrop-filter: blur(8px);
        }}
        .fpl-shell-title {{
            font-size: 0.75rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {tokens["muted"]};
            font-family: 'IBM Plex Mono', 'Space Mono', monospace;
            margin-bottom: 0.1rem;
        }}
        .fpl-shell-page {{
            font-size: 1.25rem;
            font-weight: 800;
            color: {tokens["text"]};
            line-height: 1.1;
        }}
        .fpl-shell-chips {{
            display:flex; gap:0.45rem; flex-wrap:wrap; align-items:center; justify-content:flex-end;
        }}
        .fpl-shell-chip {{
            border: 1px solid {tokens["line_strong"]};
            background: {tokens["chip_bg"]};
            color: {tokens["text"]};
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            font-size: 0.68rem;
            font-weight: 700;
            font-family: 'IBM Plex Mono', 'Space Mono', monospace;
        }}
        .fpl-card .fpl-shell-chip {{
            border-color: {tokens["line"]};
        }}

        .fpl-card, .transfer-card, .rec-box, [data-testid="stMetric"] {{
            background: linear-gradient(180deg, {tokens["surface"]} 0%, {tokens["surface_soft"]} 100%) !important;
            border: 1px solid {tokens["line"]} !important;
            box-shadow: {tokens["shadow"]} !important;
        }}
        .home-decision-card {{
            padding: 0.72rem 0.8rem !important;
            min-height: 112px;
        }}
        .home-decision-value {{
            font-size: clamp(0.96rem, 2.15vw, 1.08rem);
            font-weight: 800;
            color: {tokens["text"]};
            line-height: 1.18;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .home-decision-sub {{
            font-size: clamp(0.72rem, 1.95vw, 0.8rem);
            color: {tokens["muted"]};
            margin-top: 0.24rem;
            line-height: 1.3;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .home-risk-line {{
            border: 1px solid {tokens["line"]};
            border-radius: 10px;
            padding: 0.42rem 0.58rem;
            margin: 0.2rem 0;
            font-size: 0.8rem;
            color: {tokens["text"]};
            background: linear-gradient(180deg, {tokens["surface"]} 0%, {tokens["surface_soft"]} 100%);
        }}
        .home-deadline-strip {{
            font-size: clamp(0.72rem, 2vw, 0.8rem);
            line-height: 1.35;
            color: {tokens["muted"]};
        }}
        .fpl-card:hover, .transfer-card:hover, [data-testid="stMetric"]:hover {{
            border-color: {card_hover} !important;
        }}
        .kpi-block {{
            background: linear-gradient(180deg, {tokens["surface"]} 0%, {tokens["surface_soft"]} 100%) !important;
            border: 1px solid {tokens["line"]} !important;
            box-shadow: none !important;
        }}
        .rec-box {{
            border-color: {tokens["line"]} !important;
            position: relative;
            overflow: hidden;
        }}
        .rec-box::before {{
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 4px;
            background: {tokens["primary"]};
        }}
        .rec-box.warning {{
            background: linear-gradient(180deg, {tokens["surface"]} 0%, {rec_warning_bg} 100%) !important;
            border-color: {tokens["warning"]} !important;
        }}
        .rec-box.warning::before {{ background: {tokens["warning"]}; }}
        .rec-box.danger {{
            background: linear-gradient(180deg, {tokens["surface"]} 0%, {rec_danger_bg} 100%) !important;
            border-color: {tokens["danger"]} !important;
        }}
        .rec-box.danger::before {{ background: {tokens["danger"]}; }}
        .kpi-label, [data-testid="stMetricLabel"] {{
            color: {tokens["muted"]} !important;
            font-family: 'IBM Plex Mono', 'Space Mono', monospace !important;
        }}
        .kpi-value, [data-testid="stMetricValue"] {{
            color: {tokens["text"]} !important;
            letter-spacing: -0.02em;
        }}
        .kpi-delta, [data-testid="stMetricDelta"] {{
            color: {tokens["muted"]} !important;
        }}
        .section-header {{
            color: {tokens["primary"]} !important;
            border-bottom-color: {tokens["line"]} !important;
            font-family: 'IBM Plex Mono', 'Space Mono', monospace !important;
        }}

        [data-testid="stMetric"] {{
            padding: 0.65rem 0.8rem !important;
            border-radius: 12px !important;
        }}
        [data-testid="stMetric"] > div {{
            gap: 0.15rem !important;
        }}

        [data-testid="stTabs"] {{
            gap: 0.2rem;
        }}
        [data-testid="stTabs"] [role="tablist"] {{
            border-bottom: 1px solid {tokens["line"]};
        }}
        [data-testid="stTabs"] button {{
            color: {tokens["muted"]} !important;
            font-weight: 700 !important;
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
        }}
        [data-testid="stTabs"] button[aria-selected="true"] {{
            color: {tokens["primary"]} !important;
            border-bottom-color: {tokens["primary"]} !important;
        }}
        [data-testid="stTabs"] button:focus-visible {{
            outline-offset: -2px !important;
        }}

        .js-plotly-plot {{
            border-color: {tokens["line"]} !important;
            background: {tokens["surface"]} !important;
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid {tokens["line"]};
            border-radius: 12px;
            overflow: auto;
            background: {tokens["surface"]};
        }}
        div[data-testid="stDataFrame"] [role="grid"] {{
            background: {tokens["surface"]} !important;
        }}
        div[data-testid="stDataFrame"] [role="columnheader"] {{
            background: {table_header_bg} !important;
            color: {tokens["text"]} !important;
            border-bottom-color: {tokens["line"]} !important;
            font-weight: 700 !important;
        }}
        div[data-testid="stDataFrame"] [role="gridcell"] {{
            color: {tokens["text"]} !important;
            border-bottom-color: {tokens["line"]} !important;
        }}
        div[data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"] {{
            background: {table_row_hover} !important;
        }}

        [data-baseweb="input"], [data-baseweb="select"], textarea {{
            background: {tokens["input_bg"]} !important;
            color: {tokens["input_text"]} !important;
            border-color: {tokens["line"]} !important;
            border-radius: 10px !important;
        }}
        .stTextInput [data-baseweb="input"] > div,
        .stNumberInput [data-baseweb="input"] > div,
        .stTextArea [data-baseweb="textarea"] > div,
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div {{
            background: {tokens["input_bg"]} !important;
            color: {tokens["input_text"]} !important;
            border-color: {tokens["line"]} !important;
        }}
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        .stSelectbox input,
        .stMultiSelect input {{
            background: {tokens["input_bg"]} !important;
            color: {tokens["input_text"]} !important;
            -webkit-text-fill-color: {tokens["input_text"]} !important;
            caret-color: {tokens["input_text"]} !important;
        }}
        .stTextInput [data-baseweb="input"] > div:hover,
        .stNumberInput [data-baseweb="input"] > div:hover,
        .stTextArea [data-baseweb="textarea"] > div:hover,
        .stSelectbox [data-baseweb="select"] > div:hover,
        .stMultiSelect [data-baseweb="select"] > div:hover,
        .stTextInput [data-baseweb="input"] > div:focus-within,
        .stNumberInput [data-baseweb="input"] > div:focus-within,
        .stTextArea [data-baseweb="textarea"] > div:focus-within,
        .stSelectbox [data-baseweb="select"] > div:focus-within,
        .stMultiSelect [data-baseweb="select"] > div:focus-within {{
            background: {tokens["input_bg"]} !important;
            color: {tokens["input_text"]} !important;
            border-color: {tokens["primary"]} !important;
        }}
        [data-baseweb="select"] * {{
            color: {tokens["input_text"]} !important;
        }}
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div {{
            background: {tokens["input_bg"]} !important;
            color: {tokens["input_text"]} !important;
        }}
        .stSelectbox [data-baseweb="select"] span,
        .stSelectbox [data-baseweb="select"] div,
        .stMultiSelect [data-baseweb="select"] span,
        .stMultiSelect [data-baseweb="select"] div {{
            color: {tokens["input_text"]} !important;
            -webkit-text-fill-color: {tokens["input_text"]} !important;
        }}
        /* ── Dropdown / listbox / menu — comprehensive BaseWeb override ── */
        [role="listbox"],
        .stSelectbox [role="listbox"],
        .stMultiSelect [role="listbox"],
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="menu"],
        [data-baseweb="menu"] ul {{
            background: {dropdown_bg} !important;
            background-color: {dropdown_bg} !important;
            border: 1px solid {tokens["line"]} !important;
            color: {dropdown_text} !important;
        }}

        /* Normal option state */
        [data-baseweb="menu"] li,
        [data-baseweb="menu"] [role="option"],
        [role="option"],
        [data-baseweb="option"] {{
            background: {dropdown_bg} !important;
            background-color: {dropdown_bg} !important;
            color: {dropdown_text} !important;
            -webkit-text-fill-color: {dropdown_text} !important;
        }}
        [data-baseweb="menu"] li *,
        [data-baseweb="menu"] [role="option"] *,
        [role="option"] *,
        [data-baseweb="option"] * {{
            color: {dropdown_text} !important;
            -webkit-text-fill-color: {dropdown_text} !important;
        }}

        /* Hover / focused / highlighted / selected — all BaseWeb states */
        [data-baseweb="menu"] li:hover,
        [data-baseweb="menu"] li:focus,
        [data-baseweb="menu"] li[data-focused="true"],
        [data-baseweb="menu"] li[data-highlighted="true"],
        [data-baseweb="menu"] li[aria-selected="true"],
        [data-baseweb="menu"] li[aria-current="true"],
        [data-baseweb="menu"] li[data-selected="true"],
        [data-baseweb="menu"] [data-focused="true"],
        [data-baseweb="menu"] [data-highlighted="true"],
        [data-baseweb="menu"] [aria-selected="true"],
        [data-baseweb="menu"] [aria-current="true"],
        [role="option"]:hover,
        [role="option"]:focus,
        [role="option"][data-focused="true"],
        [role="option"][data-highlighted="true"],
        [role="option"][aria-selected="true"],
        [data-baseweb="option"]:hover,
        [data-baseweb="option"][data-focused="true"],
        [data-baseweb="option"][aria-selected="true"],
        [data-baseweb="popover"] [role="listbox"] > *:hover,
        [data-baseweb="popover"] [role="listbox"] > *:focus,
        [data-baseweb="popover"] [role="listbox"] > *[data-focused="true"],
        [data-baseweb="popover"] [role="listbox"] > *[aria-selected="true"],
        [data-baseweb="popover"] [role="listbox"] > *[data-highlighted="true"],
        [data-baseweb="popover"] [role="listbox"] > * > *:hover,
        [data-baseweb="popover"] [role="listbox"] > * > *[data-focused="true"],
        [data-baseweb="popover"] [role="listbox"] > * > *[aria-selected="true"] {{
            background: {dropdown_hover} !important;
            background-color: {dropdown_hover} !important;
            color: {dropdown_text} !important;
            -webkit-text-fill-color: {dropdown_text} !important;
        }}
        /* Force text colour on children of hovered items too */
        [data-baseweb="menu"] li:hover *,
        [data-baseweb="menu"] li[data-focused="true"] *,
        [data-baseweb="menu"] li[aria-selected="true"] *,
        [role="option"]:hover *,
        [role="option"][data-focused="true"] *,
        [data-baseweb="option"]:hover *,
        [data-baseweb="option"][data-focused="true"] * {{
            color: {dropdown_text} !important;
            -webkit-text-fill-color: {dropdown_text} !important;
        }}
        /* Popover container */
        [data-baseweb="popover"] [role="listbox"] > *,
        [data-baseweb="popover"] [role="listbox"] > * > * {{
            background: {dropdown_bg} !important;
            background-color: {dropdown_bg} !important;
            color: {dropdown_text} !important;
            -webkit-text-fill-color: {dropdown_text} !important;
        }}
        /* Multiselect selected tags */
        [data-baseweb="tag"] {{
            background: {dropdown_hover2} !important;
            border-color: {tokens["primary"]} !important;
        }}
        [data-baseweb="tag"] span,
        [data-baseweb="tag"] [role="button"] {{
            color: {dropdown_text} !important;
            -webkit-text-fill-color: {dropdown_text} !important;
        }}
                [data-baseweb="input"] input, textarea {{
            color: {tokens["input_text"]} !important;
        }}
        [data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within {{
            border-color: {tokens["primary"]} !important;
            box-shadow: 0 0 0 1px {tokens["primary"]} inset !important;
        }}
        textarea {{
            border-radius: 12px !important;
        }}
        .stSelectbox label, .stMultiSelect label, .stNumberInput label,
        .stTextInput label, .stTextArea label, .stSlider label,
        .stToggle label {{
            color: {tokens["muted"]} !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
        }}
        .stSlider [role="slider"] {{
            box-shadow: 0 0 0 2px {tokens["surface"]}, 0 0 0 3px {tokens["primary"]} !important;
        }}
        .stToggle [data-baseweb="switch"] > div {{
            background: {tokens["line"]} !important;
        }}
        .stToggle [data-baseweb="switch"] input:checked + div {{
            background: {tokens["primary"]} !important;
        }}
        .theme-side-label {{
            font-size: 0.84rem;
            font-weight: 800;
            padding-top: 0.06rem;
            color: {tokens["muted"]};
            opacity: 0.65;
            user-select: none;
            line-height: 1.05;
            white-space: nowrap;
        }}
        .theme-side-label.left {{
            text-align: right;
            padding-right: 0.02rem;
        }}
        .theme-side-label.right {{
            text-align: left;
            padding-left: 0 !important;
            margin-left: -0.32rem;
        }}
        .theme-side-label.active {{
            color: {tokens["text"]};
            opacity: 1.0;
        }}
        .st-key-theme_capsule_wrap {{
            background: rgba(20, 32, 63, 0.13) !important;
            border: 1px solid {tokens["line"]} !important;
            border-radius: 999px !important;
            padding: 0.24rem 0.34rem 0.16rem 0.34rem !important;
            width: 196px !important;
            max-width: 196px !important;
            margin-left: auto !important;
        }}
        .st-key-theme_capsule_wrap > div {{
            gap: 0.02rem !important;
            align-items: center !important;
        }}
        .st-key-theme_capsule_wrap [data-testid="stHorizontalBlock"] {{
            flex-wrap: nowrap !important;
        }}
        .st-key-theme_capsule_wrap [data-testid="stColumn"] {{
            display: flex !important;
            align-items: center !important;
        }}
        .st-key-ui_theme_top_toggle {{
            margin-top: -0.04rem !important;
        }}
        .st-key-ui_theme_top_toggle [data-baseweb="switch"] {{
            display: flex !important;
            justify-content: center !important;
        }}
        .st-key-ui_theme_top_toggle [data-baseweb="switch"] > div {{
            min-width: 68px !important;
            width: 68px !important;
            height: 34px !important;
            border-radius: 999px !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
            background: #8ea6e3 !important;
        }}
        .st-key-ui_theme_top_toggle [data-baseweb="switch"] input:checked + div {{
            background: #111a34 !important;
        }}
        .st-key-ui_theme_top_toggle [data-baseweb="switch"] > div > div {{
            width: 28px !important;
            height: 28px !important;
            margin-top: 2px !important;
            background: #f2f6ff !important;
            box-shadow: 0 3px 10px rgba(8, 16, 44, 0.28) !important;
        }}
        .stButton > button {{
            border-radius: 10px !important;
            border: 1px solid {tokens["line"]} !important;
            font-weight: 700 !important;
            background: {tokens["surface"]} !important;
            color: {tokens["text"]} !important;
        }}
        .stButton > button[kind="primary"] {{
            background: {tokens["primary"]} !important;
            color: {"#F2EFE9" if is_light else "#252627"} !important;
            border-color: {tokens["primary"]} !important;
        }}
        .stButton > button:hover {{
            border-color: {tokens["primary"]} !important;
        }}
        .stButton > button:focus-visible {{
            box-shadow: 0 0 0 3px rgba(144,78,85,0.15) !important;
        }}

        [data-testid="stDivider"] {{
            border-color: {tokens["line"]} !important;
        }}
        [data-testid="stAlert"] {{
            border-radius: 12px !important;
            border: 1px solid {tokens["line"]} !important;
        }}
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span,
        [data-testid="stAlert"] div,
        [data-testid="stAlert"] li,
        [data-testid="stAlert"] strong {{
            color: {tokens["text"]} !important;
        }}
        [data-baseweb="popover"] {{
            background: linear-gradient(180deg, {dropdown_bg} 0%, {dropdown_bg} 100%) !important;
            border: 1px solid {tokens["line"]} !important;
            border-radius: 12px !important;
            color: {dropdown_text} !important;
            box-shadow: {tokens["shadow"]} !important;
        }}
        [data-baseweb="popover"] * {{
            color: {tokens["text"]};
        }}
        [data-baseweb="popover"] p,
        [data-baseweb="popover"] li,
        [data-baseweb="popover"] strong,
        [data-baseweb="popover"] h1,
        [data-baseweb="popover"] h2,
        [data-baseweb="popover"] h3,
        [data-baseweb="popover"] h4 {{
            color: {tokens["text"]} !important;
        }}
        [data-baseweb="popover"] code {{
            color: {tokens["primary"]} !important;
            background: rgba(0,0,0,0.03) !important;
            border-color: {tokens["line"]} !important;
        }}
        [data-baseweb="popover"] a {{
            color: {tokens["primary"]} !important;
        }}
        [data-baseweb="popover"] hr {{
            border-color: {tokens["line"]} !important;
        }}
        [data-testid="stDialog"] [role="dialog"] {{
            width: min(92vw, 860px) !important;
            max-width: 860px !important;
            border-radius: 16px !important;
            border: 1px solid {tokens["line"]} !important;
            background: linear-gradient(180deg, {tokens["surface"]} 0%, {tokens["surface_soft"]} 100%) !important;
            color: {tokens["text"]} !important;
            box-shadow: {tokens["shadow"]} !important;
        }}
        [data-testid="stDialog"] [role="dialog"] * {{
            color: {tokens["text"]};
        }}
        [data-testid="stDialog"] [role="dialog"] p,
        [data-testid="stDialog"] [role="dialog"] li,
        [data-testid="stDialog"] [role="dialog"] strong,
        [data-testid="stDialog"] [role="dialog"] h1,
        [data-testid="stDialog"] [role="dialog"] h2,
        [data-testid="stDialog"] [role="dialog"] h3,
        [data-testid="stDialog"] [role="dialog"] h4 {{
            color: {tokens["text"]} !important;
        }}
        [data-testid="stDialog"] [role="dialog"] code {{
            color: {tokens["primary"]} !important;
            background: rgba(0,0,0,0.03) !important;
            border: 1px solid {tokens["line"]} !important;
            border-radius: 6px !important;
        }}
        [data-testid="stExpander"] {{
            border: 1px solid {tokens["line"]};
            border-radius: 12px;
            background: linear-gradient(180deg, {tokens["surface"]} 0%, {tokens["surface_soft"]} 100%);
        }}
        [data-testid="stExpander"] summary {{
            padding-top: 0.08rem;
            padding-bottom: 0.08rem;
        }}
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * {{
            color: {tokens["text"]} !important;
        }}
        [data-testid="stExpanderDetails"] p,
        [data-testid="stExpanderDetails"] span,
        [data-testid="stExpanderDetails"] li,
        [data-testid="stExpanderDetails"] strong,
        [data-testid="stExpanderDetails"] div {{
            color: {tokens["text"]};
        }}
        .main [data-testid="stMarkdownContainer"] p,
        .main [data-testid="stMarkdownContainer"] li,
        .main [data-testid="stMarkdownContainer"] span,
        .main [data-testid="stMarkdownContainer"] strong {{
            color: {tokens["text"]} !important;
        }}
        [data-testid="stChatMessage"] {{
            border: 1px solid {tokens["line"]};
            border-radius: 12px;
            background: linear-gradient(180deg, {tokens["surface"]} 0%, {tokens["surface_soft"]} 100%);
            padding: 0.25rem 0.35rem;
            margin-bottom: 0.35rem;
        }}
        [data-testid="stChatInput"] {{
            border: 1px solid {tokens["line"]};
            border-radius: 12px;
            padding: 0.35rem 0.5rem;
            margin-top: 0.45rem;
            background: linear-gradient(180deg, {tokens["surface"]} 0%, {tokens["surface_soft"]} 100%);
            box-shadow: {tokens["shadow"]};
        }}
        [data-testid="stChatInput"] textarea {{
            background: {tokens["input_bg"]} !important;
            border-color: {tokens["line"]} !important;
            color: {tokens["input_text"]} !important;
        }}
        [data-testid="stChatInput"] textarea::placeholder {{
            color: {tokens["muted"]} !important;
            opacity: 0.95 !important;
        }}
        [data-testid="stSpinner"] {{
            color: {tokens["primary"]} !important;
        }}

        @media (max-width: 980px) {{
            .main .block-container {{
                padding: 0.75rem 0.75rem 1.1rem;
            }}
            .fpl-shell-page {{ font-size: 1.05rem; }}
            .fpl-shell-topbar {{ padding: 0.6rem 0.7rem; }}
            .fpl-shell-chips {{ justify-content:flex-start; }}
            [data-testid="stSidebar"] [data-testid="stExpander"] {{
                margin-bottom: 0.45rem;
            }}
        }}
        @media (max-width: 640px) {{
            .fpl-shell-topbar {{
                position: static;
                margin-bottom: 0.65rem;
            }}
            .fpl-shell-page {{ font-size: 0.98rem; }}
            .fpl-shell-title {{
                font-size: 0.68rem;
                letter-spacing: 0.10em;
            }}
            .fpl-shell-chip {{
                font-size: 0.62rem;
                padding: 0.16rem 0.42rem;
            }}
            [data-testid="stMetricValue"] {{
                font-size: 1.15rem !important;
            }}
            [data-testid="stTabs"] button {{
                font-size: 0.78rem !important;
            }}
            .home-decision-card {{
                min-height: 104px;
                padding: 0.65rem 0.7rem !important;
            }}
            .home-decision-value {{
                white-space: nowrap;
            }}
            .home-decision-sub {{
                -webkit-line-clamp: 2;
            }}
            .home-risk-line {{
                font-size: 0.78rem;
                padding: 0.45rem 0.55rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_current_page_title(page: str) -> str:
    titles = {
        "Home": "Decision Center",
        "Fixture Planner": "Fixture Planner",
        "Transfer Planner": "Transfer Planner",
        "Scout": "Scout",
        "My Squad": "My Squad (Captain & Chips)",
        "Season Tracker": "Season Tracker",
        "AI Analyst": "AI Analyst",
    }
    return titles.get(page, page or "Dashboard")


def render_sidebar_settings(
    *,
    dev_mode: bool,
    last_refresh_dt: datetime,
    freshness_label: str,
):
    """Condensed left-sidebar settings panel."""
    with st.expander("Settings", expanded=bool(st.session_state.get("ui_settings_expanded", False))):
        team_row = st.container(key="team_id_row")
        team_cols = team_row.columns([3.9, 1.45], gap="small")
        team_id_error_msg = ""
        with team_cols[0]:
            st.text_input(
                "Team ID",
                key="cfg_team_id_text",
                placeholder="Enter your FPL Team ID",
                help="Enter digits and click GO to apply.",
                label_visibility="collapsed",
            )
        with team_cols[1]:
            if st.button("GO", use_container_width=True, key="apply_team_id_btn"):
                raw_team_id = str(st.session_state.get("cfg_team_id_text", "")).strip()
                digits_only = "".join(ch for ch in raw_team_id if ch.isdigit())
                if digits_only and int(digits_only) > 0:
                    st.session_state["cfg_team_id"] = int(digits_only)
                    _save_last_team_id(digits_only)
                    st.session_state["run"] = True
                    st.rerun()
                else:
                    team_id_error_msg = "Enter a valid numeric Team ID, then click GO."
        if team_id_error_msg:
            st.warning(team_id_error_msg)
        if st.button("Find your ID?", use_container_width=True, key="open_team_id_help"):
            st.session_state["show_team_id_help"] = True
        st.number_input(
            "Bank (£M)",
            key="cfg_bank_override",
            step=0.1,
            min_value=0.0,
        )
        dark_now = str(st.session_state.get("ui_theme", "dark")).lower() == "dark"
        dark_mode = st.toggle(
            "Theme",
            value=dark_now,
            key="ui_theme_dark_mode",
            help="Toggle dark mode on/off.",
        )
        desired_theme = "dark" if dark_mode else "light"
        if desired_theme != str(st.session_state.get("ui_theme", "dark")).lower():
            st.session_state["ui_theme"] = desired_theme
            st.rerun()
        if st.button("Refresh Data", use_container_width=True, type="primary", key="sidebar_refresh_data"):
            st.cache_data.clear()
            st.session_state["data_refreshed_at"] = datetime.now().isoformat(timespec="seconds")
            st.session_state["run"] = True
            st.rerun()
        st.caption(
            f"Last refresh: {last_refresh_dt.strftime('%Y-%m-%d %H:%M:%S')} · {freshness_label}"
        )
        if dev_mode:
            st.toggle(
                "Show QA panel",
                key="cfg_show_qa_panel",
                help="Developer diagnostics for UI/runtime health.",
            )


def render_team_id_help_dialog():
    """Centered help dialog explaining how to find the FPL Team ID."""
    help_body = """
**How to Find Your FPL Team ID**

Your FPL Team ID is a unique number linked to your Fantasy Premier League account.

**Important**

You must use a web browser — the Team ID is not visible in the official FPL mobile app.

**Steps (2025/26 Season)**

1. Go to the official Fantasy Premier League website and log in.

2. Open either the Pick Team or Points tab.

3. Scroll down and click View Gameweek History or Transfer History.

4. Check your browser’s address bar (URL).

5. Your Team ID is the number between /entry/ and the next /.

Example URL:
https://fantasy.premierleague.com/entry/12345/history

✅ Team ID: 12345

**How to Find a Friend’s Team ID**

1. Open the Leagues & Cups tab.

2. Select a league you both are in.

3. Click on your friend’s Team Name.

4. Check the URL — the number after /entry/ is their Team ID.

**Tip**

If the ID doesn’t show, make sure you’re on the website in a browser, not the FPL app.
"""
    if hasattr(st, "dialog"):
        @st.dialog("How to Find Your FPL Team ID")
        def _team_id_dialog():
            st.link_button(
                "Open Official FPL Website",
                "https://fantasy.premierleague.com/",
                use_container_width=True,
            )
            st.markdown(help_body)
        _team_id_dialog()
    else:
        # Fallback for older Streamlit versions: still show the full text inline.
        with st.expander("How to Find Your FPL Team ID", expanded=True):
            st.link_button(
                "Open Official FPL Website",
                "https://fantasy.premierleague.com/",
                use_container_width=True,
            )
            st.markdown(help_body)


def render_entry_gate():
    """Full-page entry gate shown before loading the main dashboard."""
    st.markdown(
        """
        <style>
        .st-key-entry_gate_card {
            background: linear-gradient(180deg, var(--surface) 0%, var(--surface-soft) 100%) !important;
            border: 1px solid var(--line) !important;
            border-radius: 16px !important;
            box-shadow: var(--shadow) !important;
            padding: 1.25rem 1.2rem 1rem !important;
            margin: 0 auto !important;
            max-width: 640px !important;
        }
        .entry-kicker {
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.3rem;
        }
        .entry-title {
            font-size: clamp(1.45rem, 2.6vw, 2.05rem);
            font-weight: 800;
            line-height: 1.06;
            color: var(--text);
            margin: 0 0 0.55rem 0;
        }
        .entry-sub {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.5;
            margin-bottom: 0.9rem;
        }
        @media (max-width: 900px) {
            .st-key-entry_gate_card {
                max-width: 100% !important;
                padding: 1rem 0.85rem 0.85rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:min(12vh,120px);'></div>", unsafe_allow_html=True)
    left, center, right = st.columns([1.1, 1.35, 1.1])
    with center:
        with st.container(key="entry_gate_card"):
            st.markdown("<div class='entry-kicker'>FPL AI Assistant</div>", unsafe_allow_html=True)
            st.markdown("<div class='entry-title'>Enter Team ID To Continue</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='entry-sub'>Set your FPL Team ID once, then access the full decision dashboard.</div>",
                unsafe_allow_html=True,
            )

            gate_input = st.text_input(
                "FPL ID",
                key="entry_gate_team_id",
                placeholder="e.g. 12345",
                help="Numeric Team ID from fantasy.premierleague.com",
            )
            go_col, help_col = st.columns([1.45, 1.0])
            with go_col:
                if st.button("Enter Dashboard", use_container_width=True, type="primary", key="entry_gate_go"):
                    digits_only = "".join(ch for ch in str(gate_input or "").strip() if ch.isdigit())
                    if digits_only and int(digits_only) > 0:
                        st.session_state["cfg_team_id"] = int(digits_only)
                        st.session_state["cfg_team_id_text"] = digits_only
                        _save_last_team_id(digits_only)
                        st.session_state["entry_gate_done"] = True
                        st.session_state["run"] = True
                        st.rerun()
                    else:
                        st.warning("Enter a valid numeric Team ID.")
            with help_col:
                if st.button("How do I find my FPL ID?", use_container_width=True, key="entry_gate_help"):
                    st.session_state["show_team_id_help"] = True

    if st.session_state.get("show_team_id_help", False):
        st.session_state["show_team_id_help"] = False
        render_team_id_help_dialog()


def render_top_status_bar(
    *,
    page: str,
    app_name: str,
    team_id: int,
    bank_chip: str,
    data_source_label: str,
    freshness_label: str,
    freshness_color: str,
):
    page_title = get_current_page_title(page)
    st.markdown(
        f"""
        <div class="fpl-shell-topbar">
            <div style="display:flex; justify-content:space-between; gap:0.8rem; align-items:center; flex-wrap:wrap;">
                <div>
                    <div class="fpl-shell-title">{_safe_text(app_name)}</div>
                    <div class="fpl-shell-page">{_safe_text(page_title)}</div>
                </div>
                <div class="fpl-shell-chips">
                    <span class="fpl-shell-chip">Team {int(team_id)}</span>
                    <span class="fpl-shell-chip">{_safe_text(bank_chip)}</span>
                    <span class="fpl-shell-chip">Source {_safe_text(data_source_label)}</span>
                    <span class="fpl-shell-chip" style="border-color:{_safe_text(freshness_color)}; color:{_safe_text(freshness_color)};">
                        Data {_safe_text(freshness_label)}
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_hero(
    title: str,
    subtitle: str = "",
    meta_chips: list[str] | None = None,
    kicker: str | None = None,
):
    chips_html = ""
    if meta_chips:
        chips_html = "".join(
            f"<span class='fpl-shell-chip' style='font-size:0.64rem; padding:0.18rem 0.45rem;'>{_safe_text(c)}</span>"
            for c in meta_chips
        )
    kicker_html = (
        f"<div class='kpi-label' style='margin-bottom:0.25rem;'>{_safe_text(kicker)}</div>"
        if kicker else ""
    )
    hero_html = (
        "<div class='fpl-card' style='margin-top:0.1rem; margin-bottom:0.85rem;'>"
        f"{kicker_html}"
        f"<div style='font-size:1.15rem; font-weight:800; color:var(--text);'>{_safe_text(title)}</div>"
        f"<div style='font-size:0.82rem; color:var(--muted); margin-top:0.2rem;'>{_safe_text(subtitle)}</div>"
        f"<div style='display:flex; gap:0.35rem; flex-wrap:wrap; margin-top:0.55rem;'>{chips_html}</div>"
        "</div>"
    )
    st.markdown(hero_html, unsafe_allow_html=True)


def _load_shap_json(json_path) -> dict | None:
    """Load model_metrics.json and return parsed dict, or None if unavailable."""
    try:
        from pathlib import Path as _Path
        _p = _Path(json_path)
        if _p.exists():
            return json.loads(_p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


@st.cache_data(ttl=300, show_spinner=False)
def _snapshot_metrics_to_runtime(
    model_metrics_df: pd.DataFrame,
    shap_json: dict | None = None,
) -> tuple[dict, dict]:
    """Build rmse_map and models dict from metrics DataFrame.

    shap_json: parsed contents of model_metrics.json (optional).
    When provided, shap_top_features is injected into each position's
    models entry so the Season Tracker SHAP tab has data to render.
    """
    rmse_map: dict = {}
    models: dict = {}
    if not isinstance(model_metrics_df, pd.DataFrame) or model_metrics_df.empty:
        # If we have no CSV but do have JSON, build from JSON alone
        if isinstance(shap_json, dict):
            for pos, info in shap_json.get("positions", {}).items():
                if not isinstance(info, dict):
                    continue
                try:
                    rmse_val = float(info["rmse"]) if info.get("rmse") is not None else np.nan
                except Exception:
                    rmse_val = np.nan
                if not np.isnan(rmse_val):
                    rmse_map[pos] = rmse_val
                models[pos] = {
                    "r2":                float(info.get("r2") or 0.0),
                    "naive_baseline_rmse": (
                        float(info.get("naive_baseline_rmse"))
                        if info.get("naive_baseline_rmse") is not None else None
                    ),
                    "beats_baseline":    bool(info.get("beats_baseline")) if info.get("beats_baseline") is not None else None,
                    # cv_degraded: True when CV R² < 0 (added Phase 1 v5.1).
                    # Defaults to False so old JSON files without the key are treated as non-degraded.
                    "cv_degraded":       bool(info.get("cv_degraded", False)),
                    "shap_top_features": info.get("shap_top_features") or {},
                }
        return rmse_map, models

    # Build shap lookup from JSON if available
    shap_by_pos: dict = {}
    if isinstance(shap_json, dict):
        for pos, info in shap_json.get("positions", {}).items():
            if isinstance(info, dict) and info.get("shap_top_features"):
                shap_by_pos[str(pos)] = info["shap_top_features"]

    for _, row in model_metrics_df.iterrows():
        pos = str(row.get("position", "") or "").strip()
        if not pos:
            continue
        try:
            rmse_val = float(row.get("rmse")) if row.get("rmse") is not None else np.nan
        except Exception:
            rmse_val = np.nan
        try:
            r2_val = float(row.get("r2")) if row.get("r2") is not None else 0.0
        except Exception:
            r2_val = 0.0
        if not np.isnan(rmse_val):
            rmse_map[pos] = rmse_val
        models[pos] = {
            "r2":                r2_val,
            "naive_baseline_rmse": (
                float(row.get("naive_baseline_rmse"))
                if row.get("naive_baseline_rmse") is not None and not pd.isna(row.get("naive_baseline_rmse"))
                else (
                    float((shap_json or {}).get("positions", {}).get(pos, {}).get("naive_baseline_rmse"))
                    if isinstance(shap_json, dict)
                    and (shap_json or {}).get("positions", {}).get(pos, {}).get("naive_baseline_rmse") is not None
                    else None
                )
            ),
            "beats_baseline": (
                bool(row.get("beats_baseline"))
                if row.get("beats_baseline") is not None and not pd.isna(row.get("beats_baseline"))
                else (
                    bool((shap_json or {}).get("positions", {}).get(pos, {}).get("beats_baseline"))
                    if isinstance(shap_json, dict)
                    and (shap_json or {}).get("positions", {}).get(pos, {}).get("beats_baseline") is not None
                    else None
                )
            ),
            # cv_degraded: True when CV R² < 0 (added Phase 1 v5.1).
            # Read from CSV first; fall back to JSON; default False so old
            # files without the key are treated as non-degraded.
            "cv_degraded": (
                bool(row.get("cv_degraded"))
                if row.get("cv_degraded") is not None and not pd.isna(row.get("cv_degraded"))
                else bool(
                    (shap_json or {}).get("positions", {}).get(pos, {}).get("cv_degraded", False)
                )
            ),
            "shap_top_features": shap_by_pos.get(pos, {}),
        }
    return rmse_map, models


def _attach_snapshot_fixture_wide_cols(enriched_df: pd.DataFrame, player_fixture_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild gw{n}_* wide columns expected by downstream pages from snapshot rows."""
    if (
        not isinstance(enriched_df, pd.DataFrame)
        or enriched_df.empty
        or not isinstance(player_fixture_df, pd.DataFrame)
        or player_fixture_df.empty
        or "player_id" not in enriched_df.columns
        or not {"player_id", "gw"}.issubset(player_fixture_df.columns)
    ):
        return enriched_df

    fixture_map: dict[int, dict] = {}
    for _, r in player_fixture_df.iterrows():
        try:
            pid = int(r.get("player_id"))
            gw = int(r.get("gw"))
        except Exception:
            continue
        row_map = fixture_map.setdefault(pid, {})

        opp = r.get("opponent")
        is_blank = bool(r.get("is_blank")) if r.get("is_blank") is not None else False
        if (opp is None or pd.isna(opp) or str(opp).strip() == "") and is_blank:
            opp = "BLANK"
        row_map[f"gw{gw}_opponent"] = opp
        row_map[f"gw{gw}_difficulty"] = r.get("difficulty")
        row_map[f"gw{gw}_home"] = r.get("is_home")

    if not fixture_map:
        return enriched_df

    fixture_wide = pd.DataFrame(
        [{"player_id": pid, **vals} for pid, vals in fixture_map.items()]
    )
    merged = enriched_df.merge(fixture_wide, on="player_id", how="left")
    return merged


def _build_runtime_from_snapshot_bundle(
    *,
    bootstrap: dict,
    fixtures_df: pd.DataFrame,
    current_gw: int,
    team_id: int,
    team_data: dict,
    transfer_info: dict,
    snapshot_meta: dict | None,
    snapshot_bundle: dict | None,
) -> dict | None:
    if not isinstance(snapshot_bundle, dict):
        return None

    pred_df = snapshot_bundle.get("predictions_df")
    player_fixture_df = snapshot_bundle.get("player_fixture_df")
    model_metrics_df = snapshot_bundle.get("model_metrics_df")
    if not isinstance(pred_df, pd.DataFrame) or pred_df.empty:
        return None

    enriched_df = pred_df.copy()
    if "snapshot_id" in enriched_df.columns:
        enriched_df = enriched_df.drop(columns=["snapshot_id"])
    if "raw_json" in enriched_df.columns:
        # Keep payload compact in-memory; dashboard uses normalized columns.
        enriched_df = enriched_df.drop(columns=["raw_json"])

    numeric_cols = [
        "player_id", "team_id", "price",
        "predicted_pts", "expected_pts", "pts_low", "pts_high",
        "captain_ev", "p_plays_full", "predicted_price_change",
        "combined_score", "value_score", "avg_difficulty",
        "blank_gws", "double_gws", "momentum_score",
    ]
    for col in numeric_cols:
        if col in enriched_df.columns:
            enriched_df[col] = pd.to_numeric(enriched_df[col], errors="coerce")

    for col in ["is_blank_next_gw"]:
        if col in enriched_df.columns:
            enriched_df[col] = enriched_df[col].astype("boolean")

    if "player_id" in enriched_df.columns:
        enriched_df["player_id"] = pd.to_numeric(enriched_df["player_id"], errors="coerce").astype("Int64")
    if "team_id" in enriched_df.columns:
        enriched_df["team_id"] = pd.to_numeric(enriched_df["team_id"], errors="coerce").astype("Int64")

    enriched_df = _attach_snapshot_fixture_wide_cols(enriched_df, player_fixture_df)

    my_player_ids = [int(p["element"]) for p in team_data.get("picks", []) if "element" in p]
    if not my_player_ids:
        return None

    my_team = enriched_df[enriched_df["player_id"].isin(my_player_ids)].copy()
    others = enriched_df[~enriched_df["player_id"].isin(my_player_ids)].copy()
    if my_team.empty:
        return None

    # Build SHAP data: prefer local JSON, fall back to shap_features column in DB
    _snap_shap_json = _load_shap_json(PRECOMPUTE_MODEL_METRICS_JSON_PATH)
    if _snap_shap_json is None and isinstance(model_metrics_df, pd.DataFrame) and "shap_features" in model_metrics_df.columns:
        # Reconstruct shap_json structure from DB column
        _snap_positions: dict = {}
        for _, _smr in model_metrics_df.iterrows():
            _pos = str(_smr.get("position") or "").strip()
            if not _pos:
                continue
            _shap_raw = _smr.get("shap_features")
            if _shap_raw is None:
                _shap_feats = {}
            elif isinstance(_shap_raw, str):
                try:
                    _shap_feats = json.loads(_shap_raw)
                except Exception:
                    _shap_feats = {}
            elif isinstance(_shap_raw, dict):
                _shap_feats = _shap_raw
            else:
                _shap_feats = {}
            _snap_positions[_pos] = {
                "rmse": (float(_smr.get("rmse")) if _smr.get("rmse") is not None else None),
                "r2":   (float(_smr.get("r2")) if _smr.get("r2") is not None else None),
                "shap_top_features": _shap_feats,
            }
        if _snap_positions:
            _snap_shap_json = {"positions": _snap_positions}
    rmse_map, models = _snapshot_metrics_to_runtime(model_metrics_df, shap_json=_snap_shap_json)

    try:
        chip_info = build_chip_status(team_id, bootstrap, fixtures_df, current_gw)
    except Exception:
        chip_info = {"available_chips": []}

    xi_result = optimize_xi_ilp(my_team)

    advanced_pipeline_enabled = "expected_pts" in enriched_df.columns
    advanced_pipeline_error = "served from snapshot" if advanced_pipeline_enabled else "snapshot missing expected_pts"

    return {
        "bootstrap": bootstrap,
        "fixtures_df": fixtures_df,
        "current_gw": current_gw,
        "my_player_ids": my_player_ids,
        "team_data": team_data,
        "transfer_info": transfer_info,
        "enriched_df": enriched_df,
        "my_team": my_team,
        "others": others,
        "xi_result": xi_result,
        "chip_info": chip_info,
        "rmse_map": rmse_map,
        "models": models,
        "history_df": pd.DataFrame(),
        "feature_capabilities": get_feature_capabilities(),
        "advanced_pipeline_enabled": bool(advanced_pipeline_enabled),
        "advanced_pipeline_error": advanced_pipeline_error,
        "cs_prob_map": None,
        "snapshot_meta": snapshot_meta,
        "data_source": "db_snapshot",
    }


# Precompute job runs every 12 hours (06:00 and 18:00 UTC).
# 13 hours: tight enough to detect a missed run (data goes stale before the next
# scheduled job at 24h) so the dashboard falls back to the live pipeline promptly,
# rather than the previous 18h setting which silently served 18-hour-old data.
PRECOMPUTE_MAX_AGE_HOURS = 13
PRECOMPUTE_DIR = Path(__file__).with_name("data")
PRECOMPUTE_PREDICTIONS_PATH = PRECOMPUTE_DIR / "fpl_predictions.csv"
PRECOMPUTE_FIXTURE_SCORES_PATH = PRECOMPUTE_DIR / "player_fixture_scores.csv"
PRECOMPUTE_META_PATH = PRECOMPUTE_DIR / "pipeline_meta.json"
PRECOMPUTE_MODEL_METRICS_PATH      = PRECOMPUTE_DIR / "model_metrics.csv"
PRECOMPUTE_MODEL_METRICS_JSON_PATH = PRECOMPUTE_DIR / "model_metrics.json"


def _precompute_is_fresh(meta_path: Path, data_path: Path, max_age_hours: float = PRECOMPUTE_MAX_AGE_HOURS) -> tuple[bool, dict | None]:
    if not data_path.exists():
        return False, None

    meta: dict = {}
    dt = None
    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta = loaded
            for key in ("generated_at", "created_at", "updated_at", "built_at", "timestamp"):
                if key in meta:
                    dt = _coerce_snapshot_datetime(meta.get(key))
                    if dt is not None:
                        break
        except Exception:
            meta = {}

    if dt is None:
        try:
            dt = datetime.fromtimestamp(data_path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            return False, None

    age_hours = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
    snapshot_meta = {
        "id": meta.get("snapshot_id", "file-precompute"),
        "created_at": dt.isoformat(),
        "status": str(meta.get("status", "ready")),
        "pipeline_version": meta.get("pipeline_version", "file-precompute"),
        "warnings": list(meta.get("warnings", []) or []),
    }
    return age_hours <= float(max_age_hours), snapshot_meta


def _build_runtime_from_precomputed_files(
    *,
    bootstrap: dict,
    fixtures_df: pd.DataFrame,
    current_gw: int,
    team_id: int,
    team_data: dict,
    transfer_info: dict,
    snapshot_meta: dict | None,
) -> dict | None:
    data_path = PRECOMPUTE_FIXTURE_SCORES_PATH if PRECOMPUTE_FIXTURE_SCORES_PATH.exists() else PRECOMPUTE_PREDICTIONS_PATH
    if not data_path.exists():
        return None

    try:
        enriched_df = pd.read_csv(data_path)
    except Exception:
        return None
    if enriched_df.empty or "player_id" not in enriched_df.columns:
        return None

    numeric_cols = [
        "player_id", "team_id", "price",
        "predicted_pts", "expected_pts", "pts_low", "pts_high",
        "captain_ev", "p_plays_full", "predicted_price_change",
        "combined_score", "value_score", "avg_difficulty",
        "blank_gws", "double_gws", "momentum_score",
    ]
    for col in numeric_cols:
        if col in enriched_df.columns:
            enriched_df[col] = pd.to_numeric(enriched_df[col], errors="coerce")

    if "player_id" in enriched_df.columns:
        enriched_df["player_id"] = pd.to_numeric(enriched_df["player_id"], errors="coerce").astype("Int64")
    if "team_id" in enriched_df.columns:
        enriched_df["team_id"] = pd.to_numeric(enriched_df["team_id"], errors="coerce").astype("Int64")

    my_player_ids = [int(p["element"]) for p in team_data.get("picks", []) if "element" in p]
    if not my_player_ids:
        return None

    my_team = enriched_df[enriched_df["player_id"].isin(my_player_ids)].copy()
    others = enriched_df[~enriched_df["player_id"].isin(my_player_ids)].copy()
    if my_team.empty:
        return None

    try:
        model_metrics_df = pd.read_csv(PRECOMPUTE_MODEL_METRICS_PATH) if PRECOMPUTE_MODEL_METRICS_PATH.exists() else pd.DataFrame()
    except Exception:
        model_metrics_df = pd.DataFrame()
    # Load SHAP from JSON — the CSV strips shap_top_features, JSON preserves them
    _shap_json = _load_shap_json(PRECOMPUTE_MODEL_METRICS_JSON_PATH)
    rmse_map, models = _snapshot_metrics_to_runtime(model_metrics_df, shap_json=_shap_json)

    try:
        chip_info = build_chip_status(team_id, bootstrap, fixtures_df, current_gw)
    except Exception:
        chip_info = {"available_chips": []}

    xi_result = optimize_xi_ilp(my_team)
    advanced_pipeline_enabled = "expected_pts" in enriched_df.columns
    advanced_pipeline_error = "served from precomputed file" if advanced_pipeline_enabled else "precomputed file missing expected_pts"

    return {
        "bootstrap": bootstrap,
        "fixtures_df": fixtures_df,
        "current_gw": current_gw,
        "my_player_ids": my_player_ids,
        "team_data": team_data,
        "transfer_info": transfer_info,
        "enriched_df": enriched_df,
        "my_team": my_team,
        "others": others,
        "xi_result": xi_result,
        "chip_info": chip_info,
        "rmse_map": rmse_map,
        "models": models,
        "history_df": pd.DataFrame(),
        "feature_capabilities": get_feature_capabilities(),
        "advanced_pipeline_enabled": bool(advanced_pipeline_enabled),
        "advanced_pipeline_error": advanced_pipeline_error,
        "cs_prob_map": None,
        "snapshot_meta": snapshot_meta,
        "data_source": "precomputed_file",
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data(team_id: int, refresh: bool = False):
    """Load all data from FPL API and run full pipeline. Cached for 60 mins."""
    data_warnings: list[str] = []
    bootstrap   = fetch_bootstrap()
    fixtures_df = fetch_fixtures()
    current_gw  = fetch_current_gw(bootstrap)

    team_data     = fetch_my_team(team_id, current_gw)
    my_player_ids = [p["element"] for p in team_data["picks"]]

    transfer_info = fetch_transfer_info(team_id, current_gw)

    if not refresh and HAS_SNAPSHOT_META_DB and load_latest_ready_snapshot_bundle:
        try:
            snapshot_meta, snapshot_bundle = load_latest_ready_snapshot_bundle()
            snapshot_runtime = _build_runtime_from_snapshot_bundle(
                bootstrap=bootstrap,
                fixtures_df=fixtures_df,
                current_gw=current_gw,
                team_id=team_id,
                team_data=team_data,
                transfer_info=transfer_info,
                snapshot_meta=snapshot_meta,
                snapshot_bundle=snapshot_bundle,
            )
            if snapshot_runtime is not None:
                snap_warn = []
                if isinstance(snapshot_meta, dict):
                    snap_warn = list(snapshot_meta.get("warnings", []) or [])
                snapshot_runtime["data_warnings"] = data_warnings + snap_warn
                return snapshot_runtime
        except Exception as e:
            data_warnings.append(f"DB snapshot path failed; using fallback source. Detail: {e}")

    if not refresh:
        try:
            is_fresh, file_snapshot_meta = _precompute_is_fresh(
                meta_path=PRECOMPUTE_META_PATH,
                data_path=(PRECOMPUTE_FIXTURE_SCORES_PATH if PRECOMPUTE_FIXTURE_SCORES_PATH.exists() else PRECOMPUTE_PREDICTIONS_PATH),
            )
            if is_fresh:
                precomputed_runtime = _build_runtime_from_precomputed_files(
                    bootstrap=bootstrap,
                    fixtures_df=fixtures_df,
                    current_gw=current_gw,
                    team_id=team_id,
                    team_data=team_data,
                    transfer_info=transfer_info,
                    snapshot_meta=file_snapshot_meta,
                )
                if precomputed_runtime is not None:
                    file_warn = []
                    if isinstance(file_snapshot_meta, dict):
                        file_warn = list(file_snapshot_meta.get("warnings", []) or [])
                    precomputed_runtime["data_warnings"] = data_warnings + file_warn
                    return precomputed_runtime
        except Exception as e:
            data_warnings.append(f"Precomputed file path failed; using live pipeline. Detail: {e}")

    history_df = build_player_history_df(bootstrap, refresh=refresh)
    models     = train_models(history_df)
    rmse_map   = get_rmse_from_models(models)

    pred_df = build_current_features(
        bootstrap, fixtures_df, history_df,
        models, current_gw, my_player_ids=my_player_ids
    )

    advanced_pipeline_enabled = False
    advanced_pipeline_error = ""
    if HAS_ADV_COMPONENT_PIPELINE and train_component_models and predict_component_pts and compute_expected_pts:
        try:
            component_models = train_component_models(history_df)
            pred_df = predict_component_pts(component_models, pred_df)
            if {"predicted_pts", "pts_from_components"}.issubset(pred_df.columns):
                direct_w = 1.0 - float(COMPONENT_BLEND_WEIGHT)
                pred_df["predicted_pts"] = (
                    direct_w * pd.to_numeric(pred_df["predicted_pts"], errors="coerce").fillna(0.0)
                    + float(COMPONENT_BLEND_WEIGHT) * pd.to_numeric(pred_df["pts_from_components"], errors="coerce").fillna(0.0)
                ).round(2)
            pred_df = compute_expected_pts(pred_df)
            advanced_pipeline_enabled = True
        except Exception as e:
            advanced_pipeline_error = str(e)

    if HAS_PRICE_MODEL and train_price_model and add_price_predictions:
        try:
            price_model = train_price_model(history_df)
            pred_df = add_price_predictions(price_model, pred_df)
        except Exception:
            pass

    custom_diff     = build_custom_difficulty(history_df, bootstrap)
    team_form_map   = build_team_form(history_df, bootstrap)
    opp_scoring_map = build_opponent_scoring_map(history_df)
    cs_prob_map = None
    if HAS_CS_PROB_MAP and build_cs_probability_map:
        try:
            cs_prob_map = build_cs_probability_map(history_df)
        except Exception:
            cs_prob_map = None
    chip_info       = build_chip_status(team_id, bootstrap, fixtures_df, current_gw)

    fixture_run_df = build_fixture_run(
        bootstrap, fixtures_df, current_gw,
        custom_difficulty=custom_diff,
        gw_lookahead=FIXTURE_LOOKAHEAD
    )
    try:
        if cs_prob_map is not None:
            enriched_df = build_player_fixture_scores(
                pred_df, fixture_run_df, current_gw,
                team_form_map, opp_scoring_map,
                FIXTURE_LOOKAHEAD,
                cs_probability_map=cs_prob_map,
            )
        else:
            enriched_df = build_player_fixture_scores(
                pred_df, fixture_run_df, current_gw,
                team_form_map, opp_scoring_map,
                FIXTURE_LOOKAHEAD
            )
    except TypeError:
        enriched_df = build_player_fixture_scores(
            pred_df, fixture_run_df, current_gw,
            team_form_map, opp_scoring_map,
            FIXTURE_LOOKAHEAD
        )

    my_team  = enriched_df[enriched_df["player_id"].isin(my_player_ids)].copy()
    others   = enriched_df[~enriched_df["player_id"].isin(my_player_ids)].copy()

    xi_result = optimize_xi_ilp(my_team)

    return {
        "bootstrap":     bootstrap,
        "fixtures_df":   fixtures_df,
        "current_gw":    current_gw,
        "my_player_ids": my_player_ids,
        "team_data":     team_data,
        "transfer_info": transfer_info,
        "enriched_df":   enriched_df,
        "my_team":       my_team,
        "others":        others,
        "xi_result":     xi_result,
        "chip_info":     chip_info,
        "rmse_map":      rmse_map,
        "models":        models,
        "history_df":    history_df,
        "feature_capabilities": get_feature_capabilities(),
        "advanced_pipeline_enabled": bool(advanced_pipeline_enabled),
        "advanced_pipeline_error": advanced_pipeline_error,
        "cs_prob_map":   cs_prob_map,
        "snapshot_meta": None,
        "data_source": "live_pipeline",
        "data_warnings": data_warnings,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def build_asset_maps(bootstrap: dict):
    """Build player and team image URL maps from FPL bootstrap payload."""
    player_face_map = {}
    team_badge_map = {}

    for p in bootstrap.get("elements", []):
        pid = int(p.get("id", 0))
        code = p.get("code")
        photo = str(p.get("photo", ""))
        if code:
            player_face_map[pid] = (
                f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{int(code)}.png"
            )
        elif photo:
            player_face_map[pid] = (
                "https://resources.premierleague.com/premierleague/photos/players/110x140/"
                + photo.replace(".jpg", ".png")
            )

    for t in bootstrap.get("teams", []):
        tid = int(t.get("id", 0))
        code = t.get("code")
        if code:
            team_badge_map[tid] = (
                f"https://resources.premierleague.com/premierleague/badges/70/t{int(code)}.png"
            )

    return player_face_map, team_badge_map


def _safe_text(text) -> str:
    txt = str(text) if text is not None else ""
    return (
        txt.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _xpts(row) -> float:
    """Use expected_pts when present, else predicted_pts."""
    try:
        v = row.get("expected_pts")
    except Exception:
        v = None
    if v is not None and not pd.isna(v):
        try:
            return float(v)
        except Exception:
            pass
    try:
        return float(row.get("predicted_pts", 0.0))
    except Exception:
        return 0.0


def _price_tag(change: float) -> str:
    try:
        c = float(change or 0.0)
    except Exception:
        c = 0.0
    if c > 0.05:
        return " ↑"
    if c < -0.05:
        return " ↓"
    return ""


def pick_points_col(df: pd.DataFrame) -> str:
    return "expected_pts" if isinstance(df, pd.DataFrame) and "expected_pts" in df.columns else "predicted_pts"


def pick_reliability_col(df: pd.DataFrame) -> str | None:
    if isinstance(df, pd.DataFrame) and "p_plays_full" in df.columns:
        return "p_plays_full"
    return None


def get_quantile_bounds(row) -> tuple[float | None, float | None]:
    lo = row.get("pts_low") if hasattr(row, "get") else None
    hi = row.get("pts_high") if hasattr(row, "get") else None
    try:
        lo = None if lo is None or pd.isna(lo) else float(lo)
    except Exception:
        lo = None
    try:
        hi = None if hi is None or pd.isna(hi) else float(hi)
    except Exception:
        hi = None
    return lo, hi


def get_feature_capabilities() -> dict:
    return {
        "adv_component_pipeline": bool(HAS_ADV_COMPONENT_PIPELINE),
        "price_model": bool(HAS_PRICE_MODEL),
        "cs_prob_map": bool(HAS_CS_PROB_MAP),
        "captain_mc": bool(HAS_CAPTAIN_MC),
        "captain_diff": bool(HAS_CAPTAIN_DIFF),
        "horizon_plan": bool(HAS_HORIZON_PLAN),
        "double_hit": bool(HAS_DOUBLE_HIT),
    }


ADVANCED_METRIC_COLUMNS = [
    "expected_pts",
    "pts_low",
    "pts_high",
    "captain_ev",
    "p_plays_full",
    "predicted_price_change",
]


def summarize_advanced_columns(df: pd.DataFrame) -> dict:
    if not isinstance(df, pd.DataFrame):
        return {"present": 0, "missing": ADVANCED_METRIC_COLUMNS.copy()}
    present = [c for c in ADVANCED_METRIC_COLUMNS if c in df.columns]
    missing = [c for c in ADVANCED_METRIC_COLUMNS if c not in df.columns]
    return {"present": len(present), "present_cols": present, "missing": missing}


def build_page_readiness_table(
    my_team_df: pd.DataFrame,
    others_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    caps: dict,
) -> pd.DataFrame:
    col_checks = summarize_advanced_columns(enriched_df)
    has_enriched_core = col_checks["present"]
    rows = [
        {"Page": "Home", "Needs": "expected_pts,captain_ev,p_plays_full", "Ready": "Yes" if has_enriched_core >= 1 else "Partial"},
        {"Page": "My Squad", "Needs": "expected_pts,pts_low,pts_high,priceΔ", "Ready": "Yes" if all(c in my_team_df.columns for c in ["expected_pts", "pts_low", "pts_high", "predicted_price_change"]) else "Partial"},
        {"Page": "Transfer Planner", "Needs": "horizon,double-hit,total_ev,price", "Ready": "Yes" if caps.get("horizon_plan") and caps.get("double_hit") else "Partial"},
        {"Page": "Scout", "Needs": "expected_pts,p_plays_full,priceΔ", "Ready": "Yes" if all(c in enriched_df.columns for c in ["expected_pts", "p_plays_full"]) else "Partial"},
        {"Page": "My Squad (Captain tab)", "Needs": "captain_ev,p_plays_full,MC,diff", "Ready": "Yes" if all([caps.get("captain_mc"), caps.get("captain_diff")]) and all(c in my_team_df.columns for c in ["captain_ev", "p_plays_full"]) else "Partial"},
        {"Page": "Season Tracker", "Needs": "sell-price (+ xPts optional)", "Ready": "Yes" if True else "Yes"},
        {"Page": "AI Analyst", "Needs": "xPts context text + enriched_df", "Ready": "Yes" if "expected_pts" in enriched_df.columns else "Partial"},
    ]
    return pd.DataFrame(rows)


def player_identity_html(
    player_name: str,
    team_name: str,
    face_url: str = "",
    badge_url: str = "",
    subtitle: str = "",
    face_class: str = "player-face",
) -> str:
    """Reusable player identity block with face and team badge."""
    pname = _safe_text(player_name)
    tname = _safe_text(team_name)
    subtitle_safe = _safe_text(subtitle)
    face = _safe_text(face_url) if face_url else ""
    badge = _safe_text(badge_url) if badge_url else ""

    fallback_initial = (pname[:1] or "?").upper()
    face_final = face or ""
    badge_final = badge or (
        "data:image/svg+xml;utf8,"
        "<svg xmlns='http://www.w3.org/2000/svg' width='40' height='40'>"
        "<circle cx='20' cy='20' r='19' fill='%23111a2e' stroke='%232e4f84' stroke-width='2'/>"
        "<text x='50%' y='56%' text-anchor='middle' fill='%2337b6ff' "
        "font-family='Space Mono' font-size='15'>FC</text></svg>"
    )
    subtitle_html = (
        f"<div style='font-size:0.74rem; color:var(--muted); margin-top:0.1rem;'>{subtitle_safe}</div>"
        if subtitle_safe else ""
    )

    return (
        f"<div class='entity-line'>"
        f"<img class='{face_class}' src='{face_final}' "
        "onerror=\"this.style.display='none'; this.nextElementSibling.style.display='flex';\" />"
        f"<div class='{face_class}' style='display:none; align-items:center; justify-content:center;"
        " color:var(--text); font-weight:700; background:linear-gradient(145deg,var(--surface-soft),var(--bg));'>"
        f"{fallback_initial}</div>"
        "<div style='min-width:0;'>"
        f"<div style='font-weight:700; font-size:0.96rem; line-height:1.1;'>{pname}</div>"
        "<div class='entity-line' style='margin-top:0.16rem; gap:0.35rem;'>"
        f"<img class='team-badge' src='{badge_final}' onerror=\"this.onerror=null;this.style.display='none';\" />"
        f"<span style='font-size:0.75rem; color:var(--muted);'>{tname}</span>"
        "</div>"
        f"{subtitle_html}"
        "</div>"
        "</div>"
    )


def build_lineup_board_html(xi_df: pd.DataFrame, cap_id: int, vc_id: int) -> str:
    """Render a modern visual lineup board from optimized XI dataframe."""
    if xi_df is None or xi_df.empty:
        return "<div class='fpl-card'>No XI available.</div>"

    order = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
    label_map = {
        "Forward": "ATTACK",
        "Midfielder": "MIDFIELD",
        "Defender": "DEFENCE",
        "Goalkeeper": "GOALKEEPER",
    }

    rows_html = []
    for pos in order:
        pos_df = xi_df[xi_df["position"] == pos].copy()
        if pos_df.empty:
            continue
        pos_df = pos_df.copy()
        pos_df["_xpts"] = pos_df.apply(lambda r: _xpts(r), axis=1)
        pos_df = pos_df.sort_values("_xpts", ascending=False)
        tiles = []
        for _, r in pos_df.iterrows():
            pid = int(r.get("player_id", 0))
            pts = float(_xpts(r))
            pts_text = f"{pts:.1f}"
            pts_class = "elite" if pts >= 8 else "good" if pts >= 5 else "mid" if pts >= 3 else "low"

            roles = []
            if pid == int(cap_id):
                roles.append("<span class='xi-role cap'>C</span>")
            elif pid == int(vc_id):
                roles.append("<span class='xi-role vc'>VC</span>")
            if bool(r.get("is_blank_next_gw", False)):
                roles.append("<span class='xi-role blank'>BLK</span>")
            if float(r.get("double_gws", 0) or 0) > 0:
                roles.append("<span class='xi-role dgw'>DGW</span>")
            roles_html = "".join(roles)

            face = _safe_text(r.get("player_face", ""))
            badge = _safe_text(r.get("team_badge", ""))
            pname = _safe_text(r.get("player_name", "Unknown"))
            team = _safe_text(r.get("team_name", ""))
            price = float(r.get("price", 0))
            run = _safe_text(r.get("fixture_run_label", "?"))

            tile = (
                "<div class='xi-tile'>"
                "<div class='xi-top'>"
                f"<img class='player-face-sm' src='{face}' "
                "onerror=\"this.style.display='none'; this.nextElementSibling.style.display='flex';\" />"
                "<div class='player-face-sm' style='display:none; align-items:center; justify-content:center;"
                " color:var(--text); font-weight:700; background:linear-gradient(145deg,var(--surface-soft),var(--bg));'>?</div>"
                f"<span class='xi-pts {pts_class}'>{pts_text} pts</span>"
                "</div>"
                f"<div class='xi-name'>{pname}</div>"
                "<div class='xi-meta'>"
                f"<img class='team-badge' src='{badge}' onerror=\"this.onerror=null;this.style.display='none';\" />"
                f"<span>{team} · £{price:.1f}</span>"
                "</div>"
                "<div class='xi-meta'>"
                f"<span style='font-family:Space Mono;'>Run: {run}</span>"
                f"{roles_html}"
                "</div>"
                "</div>"
            )
            tiles.append(tile)

        rows_html.append(
            f"<div class='lineup-row'>{''.join(tiles)}</div>"
            f"<div class='lineup-label'>{label_map.get(pos, pos)} · {len(pos_df)}</div>"
        )

    return f"<div class='lineup-board'>{''.join(rows_html)}</div>"


def render_decision_banner(
    title: str,
    primary_action: str,
    confidence: float,
    reasons: list[str],
    risk_level: str = "Medium",
):
    risk_cls = "danger" if risk_level == "High" else "warning" if risk_level == "Medium" else ""
    reasons_html = "".join(
        f"<li style='margin:0.15rem 0; color:var(--muted); font-size:0.82rem;'>{_safe_text(r)}</li>"
        for r in (reasons or [])
    )
    st.markdown(
        f"""
        <div class='rec-box {risk_cls}'>
            <div class='kpi-label'>{_safe_text(title)}</div>
            <div style='display:flex; justify-content:space-between; gap:0.8rem; align-items:flex-start;'>
                <div>
                    <div style='font-size:1.15rem; font-weight:800; color:var(--text);'>{_safe_text(primary_action)}</div>
                    <div style='font-size:0.78rem; color:var(--muted); margin-top:0.2rem;'>Risk: {_safe_text(risk_level)}</div>
                </div>
                <div style='text-align:right;'>
                    <div style='font-size:0.68rem; color:var(--accent); font-family:Space Mono;'>
                        CONFIDENCE
                        <span title='Confidence is signal strength from model agreement, gain margin, fixture context, and availability. It is not a guarantee.'
                              style='cursor:help; color:var(--muted); margin-left:0.2rem;'>ⓘ</span>
                    </div>
                    <div style='font-size:1.25rem; font-weight:800; color:var(--primary);'>{float(confidence):.0f}%</div>
                </div>
            </div>
            <ul style='margin:0.55rem 0 0 1.0rem; padding:0;'>{reasons_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, compact: bool = False):
    style = "margin-bottom:0.55rem; padding-bottom:0.35rem;" if compact else ""
    st.markdown(
        f"<div class='section-header' style='{style}'>{_safe_text(title)}</div>",
        unsafe_allow_html=True,
    )


def render_stat_cards(cards: list[dict], compact: bool | None = None):
    if not cards:
        return
    # Auto mode keeps rows readable on smaller screens after compact toggle removal.
    if compact is True:
        per_row = 2
    else:
        per_row = min(3, len(cards))
    for i in range(0, len(cards), per_row):
        row = cards[i:i + per_row]
        cols = st.columns(len(row))
        for idx, card in enumerate(row):
            tone = card.get("tone", "neutral")
            tone_color = {
                "positive": "var(--primary)",
                "warning": "var(--warning)",
                "danger": "var(--danger)",
                "neutral": "var(--accent)",
            }.get(tone, "var(--accent)")
            with cols[idx]:
                st.markdown(
                    f"""
                    <div class='kpi-block' style='text-align:left;'>
                        <div class='kpi-label'>{_safe_text(card.get("label", ""))}</div>
                        <div class='kpi-value' style='font-size:1.45rem; color:{tone_color};'>{_safe_text(card.get("value", ""))}</div>
                        <div class='kpi-delta'>{_safe_text(card.get("delta", ""))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_recommendation_card(
    headline: str,
    impact_now: float,
    impact_horizon: float,
    confidence: float,
    risk_notes: list[str],
    supporting_points: list[str],
):
    risks = "".join(
        f"<li style='margin:0.12rem 0; color:var(--danger); font-size:0.76rem;'>{_safe_text(r)}</li>"
        for r in (risk_notes or [])
    )
    points = "".join(
        f"<li style='margin:0.12rem 0; color:var(--muted); font-size:0.76rem;'>{_safe_text(p)}</li>"
        for p in (supporting_points or [])
    )
    st.markdown(
        f"""
        <div class='transfer-card'>
            <div class='kpi-label'>Recommendation</div>
            <div style='font-size:1rem; font-weight:800; color:var(--text); margin-bottom:0.35rem;'>{_safe_text(headline)}</div>
            <div style='display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:0.45rem;'>
                <span class='xi-role'>Next GW: {impact_now:+.2f}</span>
                <span class='xi-role'>5 GW: {impact_horizon:+.2f}</span>
                <span class='xi-role' title='Confidence is signal strength from model agreement, gain margin, fixture context, and availability. It is not a guarantee.'>
                    Confidence: {confidence:.0f}% ⓘ
                </span>
            </div>
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.8rem;'>
                <div><div class='kpi-label'>Why</div><ul style='margin:0.2rem 0 0 1rem; padding:0;'>{points}</ul></div>
                <div><div class='kpi-label'>Risks</div><ul style='margin:0.2rem 0 0 1rem; padding:0;'>{risks}</ul></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_table(
    df: pd.DataFrame,
    column_config: dict | None = None,
    default_sort: tuple[str, bool] | None = None,
    row_density: str = "normal",
):
    if df is None or df.empty:
        st.info("No data available for this view.")
        return
    table_df = df.copy()
    if default_sort and default_sort[0] in table_df.columns:
        table_df = table_df.sort_values(default_sort[0], ascending=default_sort[1])
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        height=360 if row_density == "compact" else None,
        column_config=column_config or {},
    )


def _display_name(name: str, max_len: int = 20) -> str:
    """Compact long player names for tight UI cards."""
    txt = str(name or "").strip()
    if len(txt) <= max_len:
        return txt
    parts = [p for p in txt.split(" ") if p]
    if len(parts) >= 2:
        candidate = f"{parts[0]} {parts[-1]}"
        if len(candidate) <= max_len:
            return candidate
    return txt[: max(3, max_len - 3)].rstrip() + "..."


def build_home_risk_lines(
    my_team_df: pd.DataFrame,
    chance_map: dict,
    news_map: dict,
    bench_cover: float,
    max_items: int = 4,
) -> list[str]:
    """Return deduplicated, severity-sorted risk lines for Home page."""
    risk_by_key: dict[tuple, dict] = {}
    for _, row in my_team_df.iterrows():
        pid = int(row.get("player_id", 0) or 0)
        pname = str(row.get("player_name", "Unknown"))
        chance = chance_map.get(pid)
        if chance is not None and float(chance) < 85:
            c = int(float(chance))
            level_score = 3 if c < 60 else 2
            msg = f"{_display_name(pname)} availability {c}%."
            note = str(news_map.get(pid, "") or "").strip()
            if note:
                msg = f"{msg} {note}"
            key = (pid, "availability")
            prev = risk_by_key.get(key)
            if prev is None or level_score > int(prev.get("score", 0)):
                risk_by_key[key] = {"score": level_score, "line": msg}
        has_blank = bool(row.get("is_blank_next_gw", False)) or int(row.get("blank_gws", 0) or 0) > 0
        if has_blank:
            key = (pid, "blank")
            msg = f"{_display_name(pname)} has blank-fixture risk."
            prev = risk_by_key.get(key)
            if prev is None or 3 > int(prev.get("score", 0)):
                risk_by_key[key] = {"score": 3, "line": msg}

    if float(bench_cover) < 6.0:
        level_score = 2 if float(bench_cover) >= 4.0 else 3
        risk_by_key[("team", "bench")] = {
            "score": level_score,
            "line": f"Bench depth is low ({float(bench_cover):.1f} pts).",
        }

    ranked = sorted(risk_by_key.values(), key=lambda x: int(x.get("score", 0)), reverse=True)
    return [str(r.get("line", "")) for r in ranked[: max(1, int(max_items))] if str(r.get("line", "")).strip()]


def render_home_decision_cards(
    *,
    transfer_call: str,
    transfer_move: str,
    captain_name: str,
    captain_ev: float,
    chip_label: str,
    chip_action: str,
    chip_sub: str,
    chip_color: str,
):
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.markdown(
            "<div class='fpl-card home-decision-card'>"
            "<div class='kpi-label'>Transfer</div>"
            f"<div class='home-decision-value'>{_safe_text(transfer_call)}</div>"
            f"<div class='home-decision-sub'>{_safe_text(transfer_move)}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div class='fpl-card home-decision-card'>"
            "<div class='kpi-label'>Captain</div>"
            f"<div class='home-decision-value'>{_safe_text(_display_name(captain_name))}</div>"
            f"<div class='home-decision-sub'>Cap EV {float(captain_ev):.1f}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div class='fpl-card home-decision-card'>"
            f"<div class='kpi-label'>{_safe_text(chip_label)}</div>"
            f"<div class='home-decision-value' style='color:{chip_color};'>"
            f"{_safe_text(chip_action)}</div>"
            f"<div class='home-decision-sub'>{_safe_text(chip_sub)}</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def render_home_deadline_strip(
    *,
    gw: int,
    deadline_text: str,
    hours_text: str,
    refresh_age_min: float,
    fixture_label: str = "",
):
    fix_txt = str(fixture_label or "").strip()
    fixture_html = f" · {_safe_text(fix_txt)}" if fix_txt else ""
    freshness = "Fresh" if float(refresh_age_min) <= 60 else "Stale"  # matches load_all_data TTL
    st.markdown(
        f"""
        <div class='fpl-card' style='padding:0.65rem 0.85rem; margin-top:0.35rem;'>
            <div class='home-deadline-strip'>
                GW{int(gw)}{fixture_html}
                · Deadline {_safe_text(deadline_text)}
                · {_safe_text(hours_text)} remaining
                · Data {freshness} ({float(refresh_age_min):.0f}m)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _fixture_confidence_and_swing(diffs: list[float], blanks: int = 0) -> tuple[float, float]:
    if not diffs:
        return 45.0, 0.0
    arr = np.array([float(x) for x in diffs], dtype=float)
    consistency = 1.0 / (1.0 + np.std(arr))
    gap = max(0.0, min(1.0, (3.2 - np.mean(arr)) / 2.2))
    blank_penalty = min(0.35, 0.12 * max(0, int(blanks)))
    conf = (0.55 * gap + 0.45 * consistency - blank_penalty) * 100.0
    conf = float(np.clip(conf, 25.0, 92.0))

    first = float(np.mean(arr[:2])) if len(arr) >= 2 else float(np.mean(arr))
    last = float(np.mean(arr[-2:])) if len(arr) >= 2 else float(np.mean(arr))
    swing = last - first
    return conf, swing


def compute_transfer_decision_confidence(rec: str, ilp_1: dict, hit_transfers: list[dict]) -> float:
    base = 56.0
    gain_5 = float(ilp_1.get("total_gain", 0.0) or 0.0)
    gain_1 = float(ilp_1.get("total_next_gain", 0.0) or 0.0)
    total_ev = float(ilp_1.get("total_ev", gain_5) or gain_5)
    hit_count = len(hit_transfers or [])
    top_tr = (ilp_1.get("transfers") or [{}])[0] if isinstance(ilp_1, dict) else {}
    urgency = float(top_tr.get("urgency_score", 0.0) or 0.0)
    blank_penalty = 8.0 if bool(top_tr.get("is_blank", False)) else 0.0
    spread_penalty = 0.0
    try:
        in_hi = float(top_tr.get("in_pts_high", np.nan))
        in_lo = float(top_tr.get("in_pts_low", np.nan))
        if not np.isnan(in_hi) and not np.isnan(in_lo):
            spread_penalty = min(6.0, max(0.0, in_hi - in_lo) * 0.8)
    except Exception:
        spread_penalty = 0.0
    if rec == "USE NOW":
        score = base + 8.0 + 5.0 * min(gain_5, 3.0) + 3.0 * min(gain_1, 2.0) + 3.0 * min(total_ev, 3.0)
    elif rec == "BORDERLINE":
        score = base + 1.5 + 3.0 * min(gain_5, 2.0) + 1.5 * min(gain_1, 1.5) + 2.0 * min(total_ev, 2.0)
    else:
        score = base - 7.0 + 1.0 * min(gain_5, 1.0) + 1.0 * min(total_ev, 1.0)
    score += min(4.0, urgency * 1.5)
    score -= (2.0 * hit_count + blank_penalty + spread_penalty)
    return float(np.clip(score, 30.0, 92.0))


def compute_fixture_decision_confidence(avg_diffs: list[float], next2: list[float], blank_counts: list[int]) -> float:
    if not avg_diffs or not next2:
        return 52.0
    spread = float(max(next2) - min(next2))
    consistency = 1.0 / (1.0 + float(np.std(next2)))
    blanks_penalty = min(0.25, 0.03 * float(sum(blank_counts or [0])))
    raw = (0.6 * min(1.0, spread / 2.5) + 0.4 * consistency - blanks_penalty) * 100.0
    return float(np.clip(raw, 38.0, 89.0))


def verify_runtime_schema(my_team_df: pd.DataFrame, others_df: pd.DataFrame, fixtures: pd.DataFrame) -> list[str]:
    issues = []
    core_cols = {"player_id", "player_name", "team_id", "team_name", "position", "price", "predicted_pts"}
    missing_my = sorted(core_cols - set(my_team_df.columns))
    missing_other = sorted(core_cols - set(others_df.columns))
    if missing_my:
        issues.append(f"My Squad missing columns: {', '.join(missing_my)}")
    if missing_other:
        issues.append(f"Player pool missing columns: {', '.join(missing_other)}")
    fixture_cols = {"event", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"}
    missing_fix = sorted(fixture_cols - set(fixtures.columns))
    if missing_fix:
        issues.append(f"Fixtures missing columns: {', '.join(missing_fix)}")
    return issues


def render_loading_skeleton():
    st.markdown(
        """
        <div class='skeleton-card skeleton-sm'></div>
        <div class='skeleton-card skeleton-md'></div>
        <div class='skeleton-card skeleton-md'></div>
        <div class='skeleton-card skeleton-lg'></div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=180, show_spinner=False)
def cached_ilp_transfers(my_team_df: pd.DataFrame, others_df: pd.DataFrame, bank_balance: float, n_transfers: int):
    return get_ilp_optimal_transfers(my_team_df, others_df, bank_balance, n_transfers=n_transfers)


@st.cache_data(ttl=180, show_spinner=False)
def cached_rolling_advice(my_team_df: pd.DataFrame, others_df: pd.DataFrame,
                           bank_balance: float, transfers_made: int,
                           chip_info_json: str, current_gw: int,
                           ilp_result_json: str) -> dict:
    """Cached wrapper — serialise dicts to strings for cache key hashing."""
    _chip = json.loads(chip_info_json) if chip_info_json else {}
    _ilp  = json.loads(ilp_result_json) if ilp_result_json else None
    return get_rolling_transfer_advice(
        my_team_df, others_df, bank_balance, transfers_made,
        _chip, current_gw, ilp_result=_ilp,
    )


@st.cache_data(ttl=180, show_spinner=False)
def cached_hit_analysis(my_team_df: pd.DataFrame, others_df: pd.DataFrame,
                         bank_balance: float, transfers_made: int) -> list:
    return get_hit_transfer_analysis(my_team_df, others_df, bank_balance, transfers_made)


@st.cache_data(ttl=300, show_spinner=False)
def cached_differential_picks(others_df: pd.DataFrame, bootstrap: dict, top_n: int = 15):
    return get_differential_picks(others_df, bootstrap, top_n=top_n)


@st.cache_data(ttl=300, show_spinner=False)
def cached_score_all_formations(my_team_df: pd.DataFrame):
    return score_all_formations(my_team_df)


@st.cache_data(ttl=300, show_spinner=False)
def build_ui_health_snapshot() -> dict:
    from pathlib import Path
    p = Path(__file__)
    txt = p.read_text(encoding="utf-8")
    unsafe_count = txt.count("unsafe_allow_html=True")
    plotly_calls = txt.count("st.plotly_chart(")
    plotly_cfg = txt.count('config={"displayModeBar"')
    return {
        "unsafe_html_count": unsafe_count,
        "plotly_calls": plotly_calls,
        "plotly_configured": plotly_cfg,
    }


def inject_dropdown_overrides(tokens: dict):
    """
    Re-inject dropdown CSS at the END of every page render.
    BaseWeb injects its own theme CSS after our main inject_global_styles call,
    overriding our dropdown styles. This runs last in the cascade so it wins.
    Light theme hover matches the chip/tag colour exactly for visual consistency.
    Dark theme uses a visible navy highlight.
    """
    is_light = tokens.get("name") == "light"
    # Light: solid chip colour so BaseWeb black background cannot bleed through
    # Dark: visible navy highlight
    bg       = "#F4F1EB" if is_light else tokens["surface"]
    text     = "#2F3038" if is_light else tokens["text"]
    # Chip colour from the screenshot — solid enough to fully hide BaseWeb default
    chip_bg  = "#DFC5C8" if is_light else "#1a2d4f"   # light: solid warm pink matching chips
    chip_bg2 = "#D4B0B4" if is_light else "#1e3560"   # selected: slightly deeper
    border   = tokens["line"]
    primary  = tokens["primary"]

    st.markdown(f"""<style>
    /* ── Dropdown override — late injection, wins cascade over BaseWeb ── */

    /* Container backgrounds */
    html body [data-baseweb="menu"],
    html body [data-baseweb="menu"] ul,
    html body [role="listbox"] {{
        background: {bg} !important;
        background-color: {bg} !important;
        border: 1px solid {border} !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
    }}

    /* Default option state — must be fully opaque so hover is visible */
    html body [data-baseweb="menu"] li,
    html body [data-baseweb="menu"] [role="option"],
    html body [role="option"] {{
        background: {bg} !important;
        background-color: {bg} !important;
        color: {text} !important;
        -webkit-text-fill-color: {text} !important;
    }}
    html body [data-baseweb="menu"] li *,
    html body [data-baseweb="menu"] [role="option"] *,
    html body [role="option"] * {{
        color: {text} !important;
        -webkit-text-fill-color: {text} !important;
        background: transparent !important;
    }}

    /* Hover / focused / highlighted / selected — all BaseWeb states covered */
    html body [data-baseweb="menu"] li:hover,
    html body [data-baseweb="menu"] li:focus,
    html body [data-baseweb="menu"] li[data-highlighted="true"],
    html body [data-baseweb="menu"] li[data-focused="true"],
    html body [data-baseweb="menu"] li[aria-selected="true"],
    html body [data-baseweb="menu"] li[aria-current="true"],
    html body [data-baseweb="menu"] [data-highlighted="true"],
    html body [data-baseweb="menu"] [data-focused="true"],
    html body [role="option"]:hover,
    html body [role="option"]:focus,
    html body [role="option"][data-highlighted="true"],
    html body [role="option"][data-focused="true"],
    html body [role="option"][aria-selected="true"] {{
        background: {chip_bg} !important;
        background-color: {chip_bg} !important;
        color: {text} !important;
        -webkit-text-fill-color: {text} !important;
    }}

    /* Force text visible on all children of hovered items */
    html body [data-baseweb="menu"] li:hover *,
    html body [data-baseweb="menu"] li[data-highlighted="true"] *,
    html body [data-baseweb="menu"] li[data-focused="true"] *,
    html body [role="option"]:hover *,
    html body [role="option"][data-highlighted="true"] *,
    html body [role="option"][data-focused="true"] * {{
        color: {text} !important;
        -webkit-text-fill-color: {text} !important;
        background: transparent !important;
    }}

    /* Selected multiselect tags (chips like "Midfielder x") */
    html body [data-baseweb="tag"] {{
        background: {chip_bg2} !important;
        background-color: {chip_bg2} !important;
        border-color: {primary} !important;
    }}
    html body [data-baseweb="tag"] span,
    html body [data-baseweb="tag"] [role="button"],
    html body [data-baseweb="tag"] * {{
        color: {text} !important;
        -webkit-text-fill-color: {text} !important;
    }}
    /* Currently selected item shown greyed at top of dropdown */
    html body [data-baseweb="menu"] li[aria-disabled="true"],
    html body [role="option"][aria-disabled="true"] {{
        background: {bg} !important;
        color: {text} !important;
        opacity: 0.45 !important;
    }}
    </style>""", unsafe_allow_html=True)


def _coerce_snapshot_datetime(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value))
        except Exception:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def get_snapshot_freshness_status(snapshot_meta: dict | None) -> dict | None:
    if not isinstance(snapshot_meta, dict):
        return None
    dt = _coerce_snapshot_datetime(snapshot_meta.get("created_at"))
    if dt is None:
        return None
    age_min = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 60.0)
    if age_min < 90:
        label = "Fresh"
        tone = "fresh"
    elif age_min < 240:
        label = "Aging"
        tone = "aging"
    else:
        label = "Stale"
        tone = "stale"
    return {
        "label": label,
        "tone": tone,
        "age_min": age_min,
        "created_at": dt,
        "pipeline_status": str(snapshot_meta.get("status", "ready")).lower(),
    }


def format_data_source_label(data_source: str) -> str:
    src = str(data_source or "").strip().lower()
    mapping = {
        "db_snapshot": "DB Snapshot",
        "precomputed_file": "Precomputed File",
        "live_pipeline": "Live Pipeline",
    }
    return mapping.get(src, src.replace("_", " ").title() or "Unknown")


def render_snapshot_freshness_banner(snapshot_meta: dict | None):
    status = get_snapshot_freshness_status(snapshot_meta)
    if not status:
        return
    tone_color = {
        "fresh": PLOTLY_PRIMARY,
        "aging": PLOTLY_WARNING,
        "stale": PLOTLY_DANGER,
    }.get(status["tone"], PLOTLY_PRIMARY)
    created_at = status["created_at"].strftime("%Y-%m-%d %H:%M UTC")
    age_txt = f"{status['age_min']:.0f}m ago"
    snapshot_id = snapshot_meta.get("id", "—")
    pipeline_status = str(status.get("pipeline_status", "ready")).lower()
    health_label = "Degraded" if pipeline_status == "degraded" else "Healthy"
    health_color = PLOTLY_WARNING if pipeline_status == "degraded" else PLOTLY_PRIMARY
    st.markdown(
        f"""
        <div class="fpl-card" style="padding:0.6rem 0.85rem; margin-top:-0.2rem; margin-bottom:0.75rem;">
            <div style="display:flex; justify-content:space-between; gap:0.6rem; align-items:center; flex-wrap:wrap;">
                <div style="font-size:0.78rem; color:var(--muted);">
                    Global Snapshot <span style="color:var(--text); font-weight:700;">#{_safe_text(snapshot_id)}</span>
                    · {_safe_text(created_at)} · {_safe_text(age_txt)}
                </div>
                <div style="display:flex; gap:0.4rem; align-items:center;">
                    <span class="fpl-shell-chip" style="border-color:{_safe_text(tone_color)}; color:{_safe_text(tone_color)};">
                        Snapshot {_safe_text(status["label"])}
                    </span>
                    <span class="fpl-shell-chip" style="border-color:{_safe_text(health_color)}; color:{_safe_text(health_color)};">
                        {_safe_text(health_label)}
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


dev_mode = os.getenv("FPL_DEBUG_UI", "0") == "1"
# Read team ID from ?team_id= URL param — per-user, no server-side sharing.
# Returns None on first visit (no param set), so the entry gate fires.
# Returns the user's own ID on return visits (they kept the URL/bookmark).
saved_team_id = _load_last_team_id()
if "cfg_team_id" not in st.session_state:
    # Use query-param ID if present, otherwise 0 so entry gate fires.
    # TEAM_ID from config.py is a developer default for local single-user runs only.
    st.session_state["cfg_team_id"] = int(saved_team_id) if saved_team_id else 0
if "cfg_team_id_text" not in st.session_state:
    st.session_state["cfg_team_id_text"] = str(saved_team_id) if saved_team_id else ""
if "entry_gate_team_id" not in st.session_state:
    st.session_state["entry_gate_team_id"] = str(saved_team_id) if saved_team_id else ""
if "cfg_bank_override" not in st.session_state:
    st.session_state["cfg_bank_override"] = 0.0
if "cfg_show_qa_panel" not in st.session_state:
    st.session_state["cfg_show_qa_panel"] = False
if "data_refreshed_at" not in st.session_state:
    st.session_state["data_refreshed_at"] = datetime.now().isoformat(timespec="seconds")
if "ui_theme" not in st.session_state:
    st.session_state["ui_theme"] = "dark"
if "ui_settings_expanded" not in st.session_state:
    st.session_state["ui_settings_expanded"] = False
if "show_team_id_help" not in st.session_state:
    st.session_state["show_team_id_help"] = False
if "entry_gate_done" not in st.session_state:
    # True only if this user's own URL already has ?team_id= set from a prior visit.
    # False on a brand-new visit (no param) — entry gate always fires for new users.
    st.session_state["entry_gate_done"] = bool(saved_team_id)

if not bool(st.session_state.get("entry_gate_done", False)):
    render_entry_gate()
    st.stop()

ui_tokens = get_theme_tokens(st.session_state["ui_theme"])
PLOTLY_THEME = build_plotly_theme(ui_tokens)

inject_global_styles(ui_tokens)
inject_dropdown_overrides(ui_tokens)
POSITION_COLOR_MAP = {
    "Goalkeeper": ui_tokens["primary"],
    "Defender": ui_tokens["accent"],
    "Midfielder": ui_tokens["warning"],
    "Forward": ui_tokens["danger"],
}
PLOTLY_PRIMARY = ui_tokens["primary"]
PLOTLY_ACCENT = ui_tokens["accent"]
PLOTLY_WARNING = ui_tokens["warning"]
PLOTLY_DANGER = ui_tokens["danger"]
PLOTLY_TEXT = ui_tokens["text"]
PLOTLY_SURFACE = ui_tokens["surface"]
PLOTLY_LINE = ui_tokens["line"]
PLOTLY_XPTS_SCALE = [[0, ui_tokens["line_strong"]], [0.5, ui_tokens["accent"]], [1, ui_tokens["primary"]]]
PLOTLY_RMSE_SCALE = [[0, ui_tokens["primary"]], [0.5, ui_tokens["warning"]], [1, ui_tokens["danger"]]]

PAGE_OPTIONS = [
    "Home", "My Squad", "Fixture Planner", "Transfer Planner",
    "Scout", "Season Tracker", "AI Analyst",
]
PAGE_ICONS = {
    "Home": "⌂",
    "My Squad": "◍",
    "Fixture Planner": "▦",
    "Transfer Planner": "↔",
    "Scout": "◎",
    "Season Tracker": "◔",
    "AI Analyst": "◇",
}

last_refresh_dt = datetime.fromisoformat(st.session_state["data_refreshed_at"])
age_seconds = (datetime.now() - last_refresh_dt).total_seconds()
is_stale = age_seconds > 3600  # matches load_all_data cache TTL
ui_freshness_label = "Stale" if is_stale else "Fresh"

with st.sidebar:
    st.markdown("### FPL AI Assistant")
    st.caption("Decision-first FPL workflow")
    render_sidebar_settings(
        dev_mode=dev_mode,
        last_refresh_dt=last_refresh_dt,
        freshness_label=ui_freshness_label,
    )
    st.divider()
    # Apply pending navigation before widget instantiation to avoid StreamlitAPIException
    _pending_nav = st.session_state.pop("_pending_nav", None)
    if _pending_nav and _pending_nav in PAGE_OPTIONS:
        st.session_state["nav_page_radio"] = _pending_nav
    page = st.radio(
        "Navigation",
        PAGE_OPTIONS,
        key="nav_page_radio",
        label_visibility="collapsed",
        format_func=lambda p: f"{PAGE_ICONS.get(p, '•')}  {p}",
    )
    st.caption("Planning • Analysis • Tracking")

if st.session_state.get("show_team_id_help", False):
    st.session_state["show_team_id_help"] = False
    render_team_id_help_dialog()

team_id_input = int(st.session_state["cfg_team_id"])
bank_override = float(st.session_state["cfg_bank_override"])
show_qa_panel = bool(st.session_state["cfg_show_qa_panel"]) if dev_mode else False

if team_id_input <= 0:
    st.warning("Team ID is not set. Enter your FPL Team ID in Settings and click GO.")
    fallback_team_id = int(TEAM_ID if BACKEND_AVAILABLE and int(TEAM_ID) > 0 else 9179961)
    team_id_input = fallback_team_id
    st.session_state["cfg_team_id"] = fallback_team_id
    if not str(st.session_state.get("cfg_team_id_text", "")).strip():
        st.session_state["cfg_team_id_text"] = str(fallback_team_id)



if not BACKEND_AVAILABLE:
    st.error(f"Backend import failed: `{IMPORT_ERROR}`")
    st.info("Make sure all phase files (fpl_phase1_model.py through fpl_phase4_optimizer.py) "
            "and config.py are in the same directory as fpl_dashboard.py")
    st.stop()

if "run" not in st.session_state:
    st.session_state["run"] = False

try:
    data = load_all_data(int(team_id_input), refresh=False)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.info("Check your team ID and internet connection, then click Refresh Data.")
    st.stop()

snapshot_meta = data.get("snapshot_meta") if isinstance(data, dict) else None
if snapshot_meta is None and HAS_SNAPSHOT_META_DB and get_latest_ready_snapshot:
    try:
        snapshot_meta = get_latest_ready_snapshot()
    except Exception:
        snapshot_meta = None

data_source = str(data.get("data_source", "live_pipeline"))
data_source_label = format_data_source_label(data_source)
data_warnings = list(data.get("data_warnings", []) or [])
snapshot_status = get_snapshot_freshness_status(snapshot_meta)
if snapshot_status:
    data_freshness_label = f"{snapshot_status['label']} ({snapshot_status['age_min']:.0f}m)"
    data_freshness_color = {
        "fresh": PLOTLY_PRIMARY,
        "aging": PLOTLY_WARNING,
        "stale": PLOTLY_DANGER,
    }.get(str(snapshot_status.get("tone", "fresh")), PLOTLY_PRIMARY)
else:
    data_freshness_label = ui_freshness_label
    data_freshness_color = PLOTLY_WARNING if is_stale else PLOTLY_PRIMARY

bank_balance_top = float(
    bank_override if bank_override > 0 else float(data.get("transfer_info", {}).get("bank_balance", 0.0))
)
bank_chip = f"Bank £{bank_balance_top:.1f}M"

render_top_status_bar(
    page=page,
    app_name="FPL AI Assistant",
    team_id=int(st.session_state["cfg_team_id"]),
    bank_chip=bank_chip,
    data_source_label=data_source_label,
    freshness_label=data_freshness_label,
    freshness_color=data_freshness_color,
)
if data_source != "live_pipeline" and snapshot_meta:
    render_snapshot_freshness_banner(snapshot_meta)
if data_warnings:
    with st.expander("Data Pipeline Warnings", expanded=False):
        for _warn in data_warnings[:8]:
            st.warning(str(_warn))

# Unpack
bootstrap    = data["bootstrap"]
fixtures_df  = data["fixtures_df"]
current_gw   = data["current_gw"]
team_data    = data.get("team_data", {})
transfer_info= data["transfer_info"]
enriched_df  = data["enriched_df"].copy()
my_team      = data["my_team"]
others       = data["others"]
xi_result    = data["xi_result"]
chip_info    = data["chip_info"]
rmse_map     = data["rmse_map"]
history_df   = data["history_df"]
feature_capabilities = data.get("feature_capabilities", get_feature_capabilities())
advanced_pipeline_enabled = bool(data.get("advanced_pipeline_enabled", False))
advanced_pipeline_error = str(data.get("advanced_pipeline_error", "") or "")
cs_prob_map = data.get("cs_prob_map")

schema_issues = verify_runtime_schema(my_team, others, fixtures_df)
if schema_issues:
    st.error("Runtime data schema issue detected. Please refresh data or check backend phase outputs.")
    for msg in schema_issues:
        st.caption(msg)
    st.stop()

bank_balance = bank_override if bank_override > 0 else transfer_info["bank_balance"]
transfers_made = transfer_info["transfers_made"]
available_chips = chip_info.get("available_chips", [])
triple_captain = "Triple Captain" in available_chips
bench_boost    = "Bench Boost" in available_chips
squad_violations = []
try:
    squad_violations = validate_squad(my_team)
except Exception:
    squad_violations = []

# Player news map
players_raw = bootstrap["elements"]
news_map = {p["id"]: p.get("news", "") for p in players_raw}
chance_map = {p["id"]: p.get("chance_of_playing_next_round") for p in players_raw}
ownership_map = {p["id"]: float(p.get("selected_by_percent", 0)) for p in players_raw}

teams_df = pd.DataFrame(bootstrap["teams"])
team_name_map = teams_df.set_index("id")["name"].to_dict()
team_short_map = teams_df.set_index("id")["short_name"].to_dict()
player_face_map, team_badge_map = build_asset_maps(bootstrap)

# Attach visual assets once and reuse everywhere.
for df in (my_team, others, enriched_df):
    if "player_id" in df.columns:
        df["player_face"] = df["player_id"].map(player_face_map).fillna("")
    if "team_id" in df.columns:
        df["team_badge"] = df["team_id"].map(team_badge_map).fillna("")

try:
    value_breakdown = get_squad_value_breakdown(my_team, bootstrap, team_data)
except Exception:
    value_breakdown = pd.DataFrame()
squad_sell_value = (
    float(value_breakdown["sell_price"].sum())
    if isinstance(value_breakdown, pd.DataFrame) and not value_breakdown.empty and "sell_price" in value_breakdown.columns
    else float(pd.to_numeric(my_team.get("price", 0), errors="coerce").fillna(0).sum())
)

if show_qa_panel:
    snap = build_ui_health_snapshot()
    with st.expander("QA Panel", expanded=False):
        st.caption(f"available_chips raw: {available_chips}")
        st.caption(f"triple_captain={triple_captain} | bench_boost={bench_boost}")
        render_stat_cards(
            [
                {
                    "label": "unsafe_allow_html",
                    "value": str(snap["unsafe_html_count"]),
                    "delta": "Lower is safer",
                    "tone": "positive" if snap["unsafe_html_count"] <= 25 else "warning",
                },
                {
                    "label": "Plotly Config Coverage",
                    "value": f"{snap['plotly_configured']}/{snap['plotly_calls']}",
                    "delta": "displayModeBar off + responsive",
                    "tone": "positive" if snap["plotly_calls"] == snap["plotly_configured"] else "warning",
                },
                {
                    "label": "Schema Issues",
                    "value": str(len(schema_issues)),
                    "delta": "Runtime validation",
                    "tone": "positive" if len(schema_issues) == 0 else "danger",
                },
                {
                    "label": "Adv Pipeline",
                    "value": "On" if advanced_pipeline_enabled else "Fallback",
                    "delta": "v5 enrichment active" if advanced_pipeline_enabled else (advanced_pipeline_error[:42] or "Baseline pipeline"),
                    "tone": "positive" if advanced_pipeline_enabled else "warning",
                },
            ],
            compact=False,
        )
        st.caption(
            "Capabilities: "
            + ", ".join([f"{k}={'Y' if bool(v) else 'N'}" for k, v in sorted(feature_capabilities.items())])
        )
        adv_enriched = summarize_advanced_columns(enriched_df)
        adv_my = summarize_advanced_columns(my_team)
        adv_others = summarize_advanced_columns(others)
        render_stat_cards(
            [
                {"label": "Enriched Adv Cols", "value": f"{adv_enriched['present']}/{len(ADVANCED_METRIC_COLUMNS)}", "delta": ", ".join(adv_enriched.get("present_cols", [])[:3]) or "None", "tone": "positive" if adv_enriched["present"] >= 4 else "warning"},
                {"label": "My Team Adv Cols", "value": f"{adv_my['present']}/{len(ADVANCED_METRIC_COLUMNS)}", "delta": "Expected page inputs", "tone": "positive" if adv_my["present"] >= 4 else "warning"},
                {"label": "Pool Adv Cols", "value": f"{adv_others['present']}/{len(ADVANCED_METRIC_COLUMNS)}", "delta": "Explorer/transfer pool", "tone": "positive" if adv_others["present"] >= 4 else "warning"},
            ]
        )
        with st.expander("Advanced Runtime Diagnostics", expanded=False):
            st.markdown("**Missing advanced columns (enriched_df):** " + (", ".join(adv_enriched["missing"]) if adv_enriched["missing"] else "None"))
            page_ready_df = build_page_readiness_table(my_team, others, enriched_df, feature_capabilities)
            st.dataframe(page_ready_df, use_container_width=True, hide_index=True)
            st.markdown("**Manual runtime checks (recommended order):**")
            st.markdown("1. Captain Picker: confirm `captain_ev` ranking + Monte Carlo + Differential tables render.")
            st.markdown("2. Transfer Planner: confirm Horizon Plan and Double Hit (-8) tab render with data.")
            st.markdown("3. My Squad / Scout: confirm xPts, Q10/Q90, Price Δ columns appear.")
            st.markdown("4. AI Analyst: ask a question and inspect Sources & Confidence + response quality with xPts context.")
        st.caption("Use compact layout for 768/430/390 widths during manual UI checks.")


if page == "Home":
    render_page_hero(
        "Weekly Decision Snapshot",
        "Your next-GW recommendation layer with transfer, captaincy, risk, and deadline context.",
        [
            f"GW{current_gw+1}",
            f"FT {'1' if transfers_made == 0 else 'Used'}",
            f"Chips {len(available_chips)}",
            f"Bank £{bank_balance:.1f}M",
        ],
    )

    render_section_header("This Week's 3 Decisions", compact=True)

    projected_xi = (
        float(xi_result["starting_xi"].apply(lambda r: _xpts(r), axis=1).sum())
        if xi_result and "starting_xi" in xi_result and not xi_result["starting_xi"].empty
        else float(my_team.apply(lambda r: _xpts(r), axis=1).sum())
    )
    bench_cover = (
        float(xi_result["bench"].apply(lambda r: _xpts(r), axis=1).sum())
        if xi_result and "bench" in xi_result and not xi_result["bench"].empty
        else 0.0
    )

    # Transfer decision (same backend logic as Transfer Planner, lightweight summary)
    transfer_move = "No strong move"
    try:
        home_ilp_1 = cached_ilp_transfers(my_team, others, float(bank_balance), n_transfers=1)
        home_roll = cached_rolling_advice(
            my_team, others, float(bank_balance), int(transfers_made),
            json.dumps(chip_info, default=str), int(current_gw),
            json.dumps(home_ilp_1, default=str),
        )
        home_hits = cached_hit_analysis(my_team, others, float(bank_balance), int(transfers_made))
        transfer_call = str(home_roll.get("recommendation", "HOLD"))
        transfer_conf = compute_transfer_decision_confidence(transfer_call, home_ilp_1, home_hits)
        transfer_reasons = home_roll.get("reasons", [])
        if isinstance(home_ilp_1, dict) and home_ilp_1.get("transfers"):
            tr = home_ilp_1["transfers"][0]
            transfer_move = f"{tr.get('out_name', '?')} → {tr.get('in_name', '?')}"
        elif transfer_reasons:
            transfer_move = str(transfer_reasons[0])
    except Exception:
        transfer_call = "HOLD"
        transfer_conf = 52.0
        transfer_reasons = []

    # Captain decision (same scoring logic as Captain page, summary only)
    cap_df_home = my_team.copy()
    if "p_plays_full" in cap_df_home.columns:
        cap_df_home["reliability"] = cap_df_home["p_plays_full"].fillna(1.0).astype(float)
    else:
        cap_df_home["reliability"] = (
            cap_df_home["player_id"].astype("Int64").map(chance_map).fillna(100).astype(float) / 100.0
        )
    cap_df_home["xpts_score"] = cap_df_home.apply(
        lambda r: xpts_captain_score(r, triple_captain), axis=1
    )
    blank_mask_home = (
        cap_df_home["is_blank_next_gw"].fillna(False).astype(bool)
        if "is_blank_next_gw" in cap_df_home.columns
        else pd.Series(False, index=cap_df_home.index)
    )
    cap_pool_home = cap_df_home[~blank_mask_home]
    if not cap_pool_home.empty:
        if "captain_ev" in cap_pool_home.columns:
            top_cap_home = cap_pool_home.nlargest(1, "captain_ev").iloc[0]
            captain_return = float(top_cap_home.get("captain_ev", 0.0))
        else:
            top_cap_home = cap_pool_home.nlargest(1, "xpts_score").iloc[0]
            captain_return = float(_xpts(top_cap_home)) * 2.0
        captain_pick = str(top_cap_home["player_name"])
    else:
        captain_pick = "No clear captain"
        captain_return = 0.0

    # ── Chip card logic: pick the most relevant available chip ───────────────
    # Priority: Bench Boost > Triple Captain > Free Hit > Wildcard
    # For each: decide USE vs HOLD based on context, give a one-line reason
    _chip_label  = "Chip"
    _chip_action = "No chips left"
    _chip_sub    = "All chips used this season"
    _chip_color  = "#C4BCB5"  # muted

    if bench_boost:
        # BB: worth using if bench has strong xPts and no blanks
        _bench_ev_home = (
            float(xi_result["bench"].apply(lambda r: _xpts(r), axis=1).sum())
            if xi_result and not xi_result["bench"].empty else 0.0
        )
        _bb_blanks_home = int(
            (xi_result["bench"].get("is_blank_next_gw", False).fillna(False).astype(bool).sum())
            if xi_result and not xi_result["bench"].empty
            and "is_blank_next_gw" in xi_result["bench"].columns
            else 0
        )
        if _bench_ev_home >= 10.0 and _bb_blanks_home == 0:
            _chip_action = "ACTIVATE BB"
            _chip_sub    = f"Bench xPts {_bench_ev_home:.1f} · No blanks"
            _chip_color  = "#BFB48F"
        else:
            _chip_action = "HOLD BB"
            _chip_sub    = (
                f"Bench xPts {_bench_ev_home:.1f} · {_bb_blanks_home} blank(s)"
                if _bb_blanks_home > 0
                else f"Bench xPts {_bench_ev_home:.1f} · save for better week"
            )
            _chip_color  = "#D5A46A"
        _chip_label = "Bench Boost"

    elif triple_captain:
        # TC: use when top captain has DGW or very high EV
        _tc_has_dgw = False
        _tc_cap_ev  = 0.0
        if not cap_pool_home.empty:
            _tc_row = cap_pool_home.nlargest(1, "captain_ev" if "captain_ev" in cap_pool_home.columns else "xpts_score").iloc[0]
            _tc_has_dgw = float(_tc_row.get("double_gws", 0) or 0) > 0
            _tc_cap_ev  = float(_tc_row.get("captain_ev", float(_xpts(_tc_row)) * 2.0))
        if _tc_has_dgw or _tc_cap_ev >= 14.0:
            _chip_action = "USE TC"
            _chip_sub    = f"{_display_name(captain_pick)} · {'DGW' if _tc_has_dgw else f'EV {_tc_cap_ev:.1f}'}"
            _chip_color  = "#BFB48F"
        else:
            _chip_action = "HOLD TC"
            _chip_sub    = f"EV {_tc_cap_ev:.1f} · wait for DGW"
            _chip_color  = "#D5A46A"
        _chip_label = "Triple Captain"

    elif any("Free Hit" in ac for ac in available_chips):
        # FH: flag upcoming BGW or bad fixture run
        _has_fh_dgw = bool(chip_info.get("free_hit_recommendation"))
        _fh_gw      = chip_info.get("free_hit_recommendation")
        if _has_fh_dgw:
            _chip_action = f"PLAN FH GW{_fh_gw}"
            _chip_sub    = "DGW window approaching"
            _chip_color  = "#BFB48F"
        else:
            _chip_action = "HOLD FH"
            _chip_sub    = "No DGW window yet"
            _chip_color  = "#D5A46A"
        _chip_label = "Free Hit"

    elif any("Wildcard" in ac for ac in available_chips):
        # WC: nudge if squad has several high-risk players
        _wc_risk = int((my_team["player_id"].map(chance_map).fillna(100) < 75).sum())
        _wc_blanks = int(my_team.get("blank_gws", 0).gt(0).sum()) if "blank_gws" in my_team.columns else 0
        if _wc_risk >= 3 or _wc_blanks >= 4:
            _chip_action = "CONSIDER WC"
            _chip_sub    = f"{_wc_risk} injury risk · {_wc_blanks} blanks"
            _chip_color  = "#D5A46A"
        else:
            _chip_action = "HOLD WC"
            _chip_sub    = "Squad looks stable"
            _chip_color  = "#C4BCB5"
        _chip_label = "Wildcard"

    render_home_decision_cards(
        transfer_call=transfer_call,
        transfer_move=transfer_move,
        captain_name=captain_pick,
        captain_ev=float(captain_return),
        chip_label=_chip_label,
        chip_action=_chip_action,
        chip_sub=_chip_sub,
        chip_color=_chip_color,
    )
    st.caption(
        f"Confidence {float(np.clip(transfer_conf, 35, 90)):.0f}% combines model agreement, availability, and fixture stability (not a guarantee)."
    )

    render_section_header("What Needs Attention", compact=True)
    risk_lines = build_home_risk_lines(
        my_team_df=my_team,
        chance_map=chance_map,
        news_map=news_map,
        bench_cover=float(bench_cover),
        max_items=4,
    )
    if risk_lines:
        for line in risk_lines:
            st.markdown(
                f"<div class='home-risk-line'>{_safe_text(line)}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.success("Squad looks clean for this gameweek.")

    render_section_header("Deadline", compact=True)
    next_event = next(
        (e for e in bootstrap.get("events", []) if int(e.get("id", 0)) == int(current_gw + 1)),
        None,
    )
    deadline_raw = next_event.get("deadline_time", "") if next_event else ""
    deadline_ts = pd.to_datetime(deadline_raw, utc=True, errors="coerce") if deadline_raw else pd.NaT
    now_utc = pd.Timestamp.utcnow()
    if pd.notna(deadline_ts):
        hours_left = float((deadline_ts - now_utc).total_seconds() / 3600.0)
        deadline_text = deadline_ts.strftime("%Y-%m-%d %H:%M UTC")
        hours_text = "Closed" if hours_left < 0 else f"{hours_left:.1f}h"
    else:
        deadline_text = "Unknown"
        hours_text = "N/A"

    home_refresh_dt = datetime.fromisoformat(st.session_state["data_refreshed_at"])
    refresh_age_min = max(0.0, (datetime.now() - home_refresh_dt).total_seconds() / 60.0)

    fixture_label = ""
    try:
        next_fix = fixtures_df[fixtures_df["event"] == int(current_gw + 1)].head(1)
        if not next_fix.empty:
            row = next_fix.iloc[0]
            fixture_label = f"{team_name_map.get(int(row['team_h']), '?')} vs {team_name_map.get(int(row['team_a']), '?')}"
    except Exception:
        fixture_label = ""

    render_home_deadline_strip(
        gw=int(current_gw + 1),
        deadline_text=deadline_text,
        hours_text=hours_text,
        refresh_age_min=float(refresh_age_min),
        fixture_label=fixture_label,
    )


elif page == "My Squad":
    render_page_hero(
        "My Squad",
        "Optimize your XI, review bench cover, and identify availability risks before locking in your team.",
        [
            f"GW{current_gw+1} prep",
            f"Bank £{bank_balance:.1f}M",
            "Optimized XI",
            "Bench + injury review",
        ],
    )

    # Check if current GW is actually finished before labelling it "Completed"
    _cur_event = next(
        (e for e in bootstrap.get("events", []) if int(e.get("id", 0)) == int(current_gw)),
        None,
    )
    _gw_finished = bool(_cur_event.get("finished", False)) if _cur_event else False
    _squad_header = (
        f"GW{current_gw} Completed → Optimized for GW{current_gw+1}"
        if _gw_finished
        else f"Optimized for GW{current_gw+1}"
    )
    render_section_header(_squad_header)
    if squad_violations:
        st.warning("Squad validation flags: " + " | ".join(str(v) for v in squad_violations[:4]))

    pred_total  = (xi_result["starting_xi"].apply(lambda r: _xpts(r), axis=1).sum()
                   if xi_result and "starting_xi" in xi_result else my_team.apply(lambda r: _xpts(r), axis=1).sum())
    if xi_result:
        _score_range = compute_score_range(xi_result["starting_xi"], rmse_map)
        if isinstance(_score_range, (tuple, list)) and len(_score_range) >= 2:
            lo, hi = float(_score_range[0]), float(_score_range[1])
        else:
            lo, hi = 0.0, 0.0
    else:
        lo, hi = 0.0, 0.0
    ft_label = "Free Transfer" if transfers_made == 0 else "Used"
    risk_players = (
        int(my_team["blank_gws"].gt(0).sum()) if "blank_gws" in my_team.columns else 0
    ) + int(
        (my_team["player_id"].map(chance_map).fillna(100) < 75).sum()
    )
    bench_cover = float(xi_result["bench"].apply(lambda r: _xpts(r), axis=1).sum()) if xi_result and not xi_result["bench"].empty else 0.0
    # ── Compact squad info strip ──────────────────────────────────────────────
    if xi_result:
        _formation_str  = str(xi_result.get("formation", "Best XI"))
        _risk_color     = "#D36C73" if risk_players > 4 else "#D5A46A" if risk_players > 2 else "#BFB48F"
        _range_str      = f"{lo:.0f}–{hi:.0f}" if lo > 0 or hi > 0 else "N/A"
        _ft_chip        = "1 FT" if transfers_made == 0 else "FT Used"
        st.markdown(
            "<div class='fpl-card' style='padding:0.55rem 0.9rem;margin-bottom:0.6rem;'>"
            "<div style='display:flex;flex-wrap:wrap;gap:0.5rem 1.2rem;align-items:center;'>"
            f"<span style='font-weight:800;font-size:0.95rem;'>{_safe_text(_formation_str)}</span>"
            f"<span style='font-size:0.82rem;color:#BFB48F;font-weight:700;'>{pred_total:.1f} xPts</span>"
            f"<span style='font-size:0.78rem;color:#C4BCB5;'>Range {_safe_text(_range_str)}</span>"
            f"<span style='font-size:0.78rem;color:{_risk_color};font-weight:700;'>"
            f"{risk_players} risk flag{'s' if risk_players != 1 else ''}</span>"
            f"<span style='font-size:0.78rem;color:#C4BCB5;'>Bench {bench_cover:.1f} xPts</span>"
            f"<span style='font-size:0.78rem;color:#C4BCB5;'>£{bank_balance:.1f}M bank · {_ft_chip}</span>"
            f"<span style='font-size:0.78rem;color:#C4BCB5;'>£{float(squad_sell_value):.1f}M sell value</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    if xi_result:
        xi   = xi_result["starting_xi"]
        cap  = xi_result["captain"]
        vc   = xi_result["vice_captain"]
        bench = xi_result["bench"]

        xi_cards = xi.copy()
        if "player_face" not in xi_cards.columns:
            xi_cards["player_face"] = xi_cards["player_id"].map(player_face_map)
        if "team_badge" not in xi_cards.columns:
            xi_cards["team_badge"] = xi_cards["team_id"].map(team_badge_map)

        tab_xi, tab_captain = st.tabs(["⚽  Starting XI", "★  Captain & Chips"])

        with tab_xi:
            st.markdown(
                build_lineup_board_html(
                    xi_cards,
                    int(cap["player_id"]),
                    int(vc["player_id"]),
                ),
                unsafe_allow_html=True,
            )

            l1, l2, l3, l4 = st.columns(4)
            l1.markdown("`8+ pts` elite")
            l2.markdown("`5-7 pts` strong")
            l3.markdown("`3-4 pts` playable")
            l4.markdown("`<3 pts` risky · `C` captain · `VC` vice")

            # ── Interactive player inspector ──────────────────────────────────────
            st.markdown("<div style='margin-top:1.1rem;'>", unsafe_allow_html=True)
            xi_names = xi_result["starting_xi"]["player_name"].tolist()
            bench_names = xi_result["bench"]["player_name"].tolist() if not xi_result["bench"].empty else []
            all_squad_names = xi_names + bench_names

            inspect_col, _ = st.columns([2, 3])
            with inspect_col:
                selected_player = st.selectbox(
                    "🔍 Tap a player to inspect",
                    options=["Select a player..."] + all_squad_names,
                    key="squad_player_inspect",
                    label_visibility="visible",
                )
            st.markdown("</div>", unsafe_allow_html=True)

            if selected_player and selected_player != "Select a player...":
                player_rows = enriched_df[enriched_df["player_name"] == selected_player]
                if not player_rows.empty:
                    pr = player_rows.iloc[0]
                    pid = int(pr["player_id"])
                    chance = chance_map.get(pid)
                    news = news_map.get(pid, "") or ""
                    ownership_pct = float(ownership_map.get(pid, 0))
                    pchg = float(pr.get("predicted_price_change", 0) or 0)
                    is_in_xi = selected_player in xi_names

                    # availability colour — use hex not var(--x) to avoid markdown parser issues
                    avail_color = (
                        "#D36C73" if chance is not None and chance < 75
                        else "#D5A46A" if chance is not None and chance < 100
                        else "#BFB48F"
                    )
                    avail_text = (
                        f"⚠ {chance}% chance of playing"
                        if chance is not None and chance < 100
                        else "✓ Available"
                    )

                    col_status, col_fixtures, col_replace = st.columns([1.2, 1, 1.6])

                    # ── Status card ───────────────────────────────────────────────
                    with col_status:
                        face = _safe_text(pr.get("player_face", ""))
                        badge = _safe_text(pr.get("team_badge", ""))
                        role_tag = "Starting XI" if is_in_xi else "Bench"
                        # Use inline hex/token colours to avoid var(--x) breaking markdown
                        role_color_val = "#BFB48F" if is_in_xi else "#C4BCB5"
                        cap_tag = ""
                        if pid == int(cap["player_id"]):
                            cap_tag = "<span class='xi-role cap' style='margin-left:0.4rem;'>C</span>"
                        elif pid == int(vc["player_id"]):
                            cap_tag = "<span class='xi-role vc' style='margin-left:0.4rem;'>VC</span>"
                        news_display = _safe_text(
                            (news[:160] + "...") if len(news) > 160 else news
                        ) if news else "No injury news."
                        price_str = f"\u00a3{float(pr.get('price', 0)):.1f}M"
                        xpts_str = f"{_xpts(pr):.2f} xPts"
                        own_str = f"{ownership_pct:.1f}% owned"
                        run_str = _safe_text(pr.get("fixture_run_label", "?"))
                        team_str = _safe_text(pr.get("team_name", ""))
                        pos_str = _safe_text(pr.get("position", ""))
                        player_str = _safe_text(selected_player)
                        pchg_str = _price_tag(pchg)
                        # Build as a single string — avoids markdown parser choking on CSS var(--x)
                        _status_html = (
                            "<div class='fpl-card' style='min-height:160px;'>"
                            "<div style='display:flex;align-items:center;gap:0.55rem;margin-bottom:0.5rem;'>"
                            f"<img class='player-face' src='{face}' onerror=\"this.onerror=null;this.style.display='none';\" />"
                            "<div>"
                            f"<div style='font-weight:800;font-size:0.95rem;line-height:1.1;'>{player_str}{cap_tag}</div>"
                            "<div style='display:flex;align-items:center;gap:0.3rem;margin-top:0.2rem;'>"
                            f"<img class='team-badge' src='{badge}' onerror=\"this.onerror=null;this.style.display='none';\" />"
                            f"<span style='font-size:0.75rem;color:#C4BCB5;'>{team_str} · {pos_str}</span>"
                            "</div></div></div>"
                            f"<div style='font-size:0.82rem;margin-bottom:0.3rem;'>"
                            f"<span style='color:{role_color_val};font-weight:700;'>{role_tag}</span>"
                            f" · {price_str} · <span style='color:#C9BDC3;'>{xpts_str}</span>{pchg_str}"
                            "</div>"
                            f"<div style='font-size:0.8rem;color:{avail_color};font-weight:700;margin-bottom:0.25rem;'>{avail_text}</div>"
                            f"<div style='font-size:0.75rem;color:#C4BCB5;line-height:1.4;'>{news_display}</div>"
                            f"<div style='font-size:0.72rem;color:#C4BCB5;margin-top:0.35rem;'>{own_str} · Run: {run_str}</div>"
                            "</div>"
                        )
                        st.markdown(_status_html, unsafe_allow_html=True)

                    # ── Next fixtures card ────────────────────────────────────────
                    with col_fixtures:
                        gws_ahead = list(range(current_gw + 1, current_gw + 4))
                        fix_html = "<div class='fpl-card' style='min-height:160px;'><div class='kpi-label'>Next 3 Fixtures</div>"
                        for gw in gws_ahead:
                            opp = str(pr.get(f"gw{gw}_opponent", "?") or "?")
                            diff = pr.get(f"gw{gw}_difficulty", 3)
                            home = pr.get(f"gw{gw}_home", 0)
                            try:
                                diff_f = float(diff)
                            except Exception:
                                diff_f = 3.0
                            if opp.upper() in {"BLANK", "B", ""}:
                                diff_color = "var(--muted)"
                                opp_display = "BLANK"
                                venue_display = "—"
                            else:
                                diff_color = (
                                    "#27e8a7" if diff_f <= 2
                                    else "#7fd7ff" if diff_f == 3
                                    else "#ffb547" if diff_f == 4
                                    else "#ff5d73"
                                )
                                opp_display = opp
                                venue_display = "H" if home else "A"
                            fix_html += (
                                f"<div style='display:flex; justify-content:space-between; align-items:center;"
                                f"padding:0.3rem 0; border-bottom:1px solid var(--line);'>"
                                f"<span style='font-family:Space Mono; font-size:0.7rem; color:var(--muted);'>GW{gw}</span>"
                                f"<span style='font-size:0.82rem; font-weight:700;'>{_safe_text(opp_display)}</span>"
                                f"<span style='font-family:Space Mono; font-size:0.7rem; color:{diff_color}; font-weight:700;'>"
                                f"{venue_display} D{int(diff_f)}</span>"
                                f"</div>"
                            )
                        fix_html += "</div>"
                        st.markdown(fix_html, unsafe_allow_html=True)

                    # ── Best replacements card ────────────────────────────────────
                    with col_replace:
                        player_pos = str(pr.get("position", ""))
                        player_price = float(pr.get("price", 0))
                        budget_for_replacement = player_price + bank_balance

                        same_pos = others[others["position"] == player_pos].copy()
                        same_pos["_xpts"] = same_pos.apply(lambda r: _xpts(r), axis=1)
                        same_pos["_affordable"] = same_pos["price"] <= budget_for_replacement + 0.05
                        top_reps = (
                            same_pos[same_pos["_affordable"]]
                            .nlargest(4, "_xpts")
                        )

                        rep_html = "<div class='fpl-card' style='min-height:160px;'><div class='kpi-label'>Best Replacements</div>"
                        if top_reps.empty:
                            rep_html += "<div style='font-size:0.8rem; color:var(--muted); margin-top:0.5rem;'>No affordable alternatives found.</div>"
                        else:
                            for _, rep in top_reps.iterrows():
                                gain = float(_xpts(rep)) - float(_xpts(pr))
                                cost_diff = float(rep["price"]) - player_price
                                gain_color = "var(--primary)" if gain > 0 else "var(--danger)"
                                rep_face = _safe_text(rep.get("player_face", ""))
                                rep_html += (
                                    f"<div style='display:flex; justify-content:space-between; align-items:center;"
                                    f"padding:0.28rem 0; border-bottom:1px solid var(--line);'>"
                                    f"<div style='display:flex; align-items:center; gap:0.3rem; min-width:0;'>"
                                    f"<img class='player-face-sm' src='{rep_face}' style='width:22px;height:22px;'"
                                    f"onerror=\"this.onerror=null;this.style.display='none';\" />"
                                    f"<span style='font-size:0.78rem; font-weight:700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:100px;'>"
                                    f"{_safe_text(rep['player_name'])}</span>"
                                    f"</div>"
                                    f"<div style='text-align:right; flex-shrink:0;'>"
                                    f"<span style='font-size:0.72rem; color:{gain_color}; font-weight:700;'>{gain:+.1f}pts</span>"
                                    f"<span style='font-size:0.68rem; color:var(--muted); margin-left:0.3rem;'>£{cost_diff:+.1f}M</span>"
                                    f"</div>"
                                    f"</div>"
                                )
                        rep_html += "</div>"
                        st.markdown(rep_html, unsafe_allow_html=True)

                        # Plan transfer button
                        if st.button(
                            f"Plan transfer for {selected_player.split()[-1]} →",
                            key="squad_plan_transfer_btn",
                            use_container_width=True,
                            type="primary",
                        ):
                            st.session_state["transfer_focus_player"] = selected_player
                            st.session_state["_pending_nav"] = "Transfer Planner"
                            st.rerun()

            alt_forms = cached_score_all_formations(my_team)
            if alt_forms:
                with st.expander("Formation alternatives", expanded=False):
                    alt_df = pd.DataFrame(alt_forms)[["formation", "pred_pts", "combined"]].head(5).copy()
                    best_pts = float(alt_df["pred_pts"].max()) if not alt_df.empty else 0.0
                    alt_df["delta_vs_best"] = alt_df["pred_pts"] - best_pts
                    alt_df = alt_df.rename(columns={
                        "formation": "Formation",
                        "pred_pts": "Predicted Pts",
                        "combined": "Combined Score",
                        "delta_vs_best": "Delta vs Best",
                    })
                    render_insight_table(
                        alt_df,
                        default_sort=("Predicted Pts", False),
                        row_density="compact",
                        column_config={
                            "Predicted Pts": st.column_config.NumberColumn(format="%.2f"),
                            "Combined Score": st.column_config.NumberColumn(format="%.2f"),
                            "Delta vs Best": st.column_config.NumberColumn(format="%+.2f"),
                        },
                    )


        with tab_captain:
            render_section_header("Captain & Chips")

            # ── Chip availability strip ───────────────────────────────────────
            _chips_list = ["Wildcard", "Free Hit", "Triple Captain", "Bench Boost"]
            _chip_cols = st.columns(4)
            for _ci, _chip in enumerate(_chips_list):
                _avail = any(_chip in ac for ac in available_chips)
                _avail_color = "#BFB48F" if _avail else "#D36C73"
                _chip_cols[_ci].markdown(
                    f"<div class='kpi-block'><div class='kpi-label'>{_chip}</div>"
                    f"<div style='font-size:1.1rem;font-weight:800;color:{_avail_color};'>"
                    f"{'Available' if _avail else 'Used'}</div></div>",
                    unsafe_allow_html=True,
                )

            # ── Bench Boost decision ──────────────────────────────────────────
            if bench_boost:
                _bench_eval = (
                    xi_result["bench"].copy()
                    if xi_result and "bench" in xi_result and not xi_result["bench"].empty
                    else pd.DataFrame()
                )
                if not _bench_eval.empty:
                    _bench_eval["xpts_val"] = _bench_eval.apply(lambda r: _xpts(r), axis=1).astype(float)
                    if "p_plays_full" in _bench_eval.columns:
                        _bench_eval["bench_rel"] = pd.to_numeric(
                            _bench_eval["p_plays_full"], errors="coerce"
                        ).fillna(1.0).clip(lower=0.0, upper=1.0)
                    else:
                        _bench_eval["bench_rel"] = (
                            _bench_eval["player_id"].astype("Int64").map(chance_map)
                            .fillna(100).astype(float) / 100.0
                        ).clip(lower=0.0, upper=1.0)
                    _bench_eval["blank_next"] = (
                        _bench_eval["is_blank_next_gw"].fillna(False).astype(bool)
                        if "is_blank_next_gw" in _bench_eval.columns else False
                    )
                    _bench_raw = float(_bench_eval["xpts_val"].sum())
                    _bench_ev  = float((_bench_eval["xpts_val"] * _bench_eval["bench_rel"]).sum())
                    _bench_avg = float(_bench_ev / max(len(_bench_eval), 1))
                    _avg_rel   = float(_bench_eval["bench_rel"].mean())
                    _bb_blanks = int(pd.Series(_bench_eval["blank_next"]).fillna(False).astype(bool).sum())
                    _bb_score  = _bench_ev - (1.4 * _bb_blanks) + (0.8 if _avg_rel >= 0.85 else 0.0)
                    if _bb_score >= 11.0 and _bench_avg >= 2.5 and _bb_blanks == 0:
                        _bb_call, _bb_risk = "ACTIVATE BENCH BOOST", "Low"
                    elif _bb_score >= 9.0 and _bench_avg >= 2.1:
                        _bb_call, _bb_risk = "CONSIDER BENCH BOOST", "Medium"
                    else:
                        _bb_call, _bb_risk = "HOLD BENCH BOOST", "High"
                    _bb_conf = float(np.clip(45 + 3.0 * _bb_score + 25.0 * _avg_rel - 9.0 * _bb_blanks, 35, 92))
                    render_decision_banner(
                        title="Bench Boost Decision",
                        primary_action=_bb_call,
                        confidence=_bb_conf,
                        reasons=[
                            f"Bench EV (reliability-weighted): {_bench_ev:.2f}",
                            f"Raw bench xPts: {_bench_raw:.2f} | Avg reliability: {_avg_rel:.0%}",
                            f"Bench blanks next GW: {_bb_blanks}",
                        ],
                        risk_level=_bb_risk,
                    )
                    render_stat_cards([
                        {"label": "Bench EV", "value": f"{_bench_ev:.2f}", "delta": "Reliability-weighted", "tone": "positive" if _bench_ev >= 10 else "warning"},
                        {"label": "Raw xPts", "value": f"{_bench_raw:.2f}", "delta": "Bench total", "tone": "neutral"},
                        {"label": "Avg Reliability", "value": f"{_avg_rel:.0%}", "delta": "Minutes confidence", "tone": "positive" if _avg_rel >= 0.8 else "warning"},
                        {"label": "Bench Blanks", "value": str(_bb_blanks), "delta": "Next GW", "tone": "danger" if _bb_blanks > 0 else "positive"},
                    ])

            st.divider()

            # ── Captain scoring ───────────────────────────────────────────────
            cap_df = my_team.copy()
            if "p_plays_full" in cap_df.columns:
                cap_df["reliability"] = cap_df["p_plays_full"].fillna(1.0).astype(float)
            else:
                cap_df["reliability"] = (
                    cap_df["player_id"].astype("Int64").map(chance_map).fillna(100).astype(float) / 100.0
                )
            cap_df["xpts_val"] = cap_df.apply(lambda r: _xpts(r), axis=1)
            cap_df["upside"] = (
                cap_df["xpts_val"].astype(float)
                + 0.5 * (cap_df["double_gws"].fillna(0).astype(float) if "double_gws" in cap_df.columns else 0.0)
                - 0.35 * (cap_df["blank_gws"].fillna(0).astype(float) if "blank_gws" in cap_df.columns else 0.0)
            )
            has_cap_ev = "captain_ev" in cap_df.columns
            cap_df["xpts_score"] = cap_df.apply(lambda r: xpts_captain_score(r, triple_captain), axis=1)
            cap_df["_cap_sort"] = (
                cap_df["captain_ev"].astype(float) * (1.5 if triple_captain else 1.0)
                if has_cap_ev else cap_df["xpts_score"]
            )
            _blank_mask_cap = (
                cap_df["is_blank_next_gw"].fillna(False).astype(bool)
                if "is_blank_next_gw" in cap_df.columns
                else pd.Series(False, index=cap_df.index)
            )
            cap_df["vc_score"] = (cap_df["xpts_val"].astype(float) * cap_df["reliability"]).where(~_blank_mask_cap, 0.0)
            cap_df["captain_expected_return"] = cap_df["xpts_val"].astype(float) * (3.0 if triple_captain else 2.0)
            cap_df["captain_confidence"] = np.clip(45 + 25 * cap_df["reliability"] + 4 * cap_df["xpts_val"], 40, 95)
            if {"pts_low", "pts_high"}.issubset(cap_df.columns):
                _spread_cap = (
                    pd.to_numeric(cap_df["pts_high"], errors="coerce")
                    - pd.to_numeric(cap_df["pts_low"], errors="coerce")
                ).fillna(0.0).clip(lower=0.0)
                cap_df["captain_confidence"] = np.clip(
                    cap_df["captain_confidence"] - np.minimum(8.0, _spread_cap * 1.2), 35, 95
                )

            _non_blank_cap = cap_df[~_blank_mask_cap]
            top3_cap = _non_blank_cap.nlargest(3, "_cap_sort")
            top_vc   = _non_blank_cap[~_non_blank_cap["player_id"].isin(
                top3_cap.iloc[:1]["player_id"]
            )].nlargest(1, "vc_score")

            if not top3_cap.empty:
                _cap_row = top3_cap.iloc[0]
                _vc_name = top_vc.iloc[0]["player_name"] if not top_vc.empty else "No clear VC"
                _cap_conf = float(_cap_row.get("captain_confidence", 60.0))
                _cap_risk = (
                    "Low" if float(_cap_row.get("reliability", 1.0)) >= 0.85
                    else "Medium" if float(_cap_row.get("reliability", 1.0)) >= 0.70
                    else "High"
                )
                _dgw_note = "DGW upside available" if float(_cap_row.get("double_gws", 0) or 0) > 0 else "No DGW boost"
                render_decision_banner(
                    title="Captain Decision",
                    primary_action=f"Captain {_cap_row['player_name']} | VC {_vc_name}",
                    confidence=_cap_conf,
                    reasons=[
                        f"{'Cap EV' if has_cap_ev else 'Expected return'}: {float(_cap_row.get('captain_ev', _cap_row.get('captain_expected_return', 0.0))):.1f}",
                        f"Reliability: {float(_cap_row.get('reliability', 1.0)):.0%} | Upside: {float(_cap_row.get('upside', 0.0)):.1f}",
                        _dgw_note,
                    ],
                    risk_level=_cap_risk,
                )
                render_stat_cards([
                    {"label": "Captain", "value": str(_cap_row["player_name"]), "delta": f"Conf {_cap_conf:.0f}% · {float(_cap_row.get('reliability',1.0)):.0%} reliability", "tone": "positive"},
                    {"label": "VC", "value": str(_vc_name), "delta": "Reliability-weighted backup", "tone": "neutral"},
                    {"label": "Cap EV" if has_cap_ev else "Top Return", "value": f"{float(_cap_row.get('captain_ev', _cap_row.get('captain_expected_return', 0.0))):.1f}", "delta": "Expected captained return", "tone": "positive"},
                ])

                # Captain podium
                st.markdown("**Captain Recommendations**")
                _medals_list = ["TC Captain (3x)" if triple_captain else "Captain", "Vice Captain Option", "3rd Option"]
                _cap_pod_cols = st.columns(3)
                for _pi, (_, _prow) in enumerate(top3_cap.iterrows()):
                    with _cap_pod_cols[_pi % 3]:
                        _pmult = 3 if (triple_captain and _pi == 0) else 2
                        _pdgw  = float(_prow.get("double_gws", 0) or 0) > 0
                        _pev_txt = (
                            "Cap EV: " + format(float(_prow.get("captain_ev", 0.0)), ".1f")
                            if has_cap_ev
                            else "= " + str(round(_xpts(_prow) * _pmult, 2)) + " if captained"
                        )
                        _pflags = []
                        if _pdgw: _pflags.append("DGW")
                        if triple_captain and _pi == 0: _pflags.append("TC 3x")
                        _psubline = _pev_txt + ("  " + "  ".join(_pflags) if _pflags else "")
                        _pname  = _safe_text(_prow.get("player_name", "Unknown"))
                        _pteam  = _safe_text(_prow.get("team_name", ""))
                        _prun   = _safe_text(_prow.get("fixture_run_label", "?"))
                        _pface  = _safe_text(_prow.get("player_face", ""))
                        _pbadge = _safe_text(_prow.get("team_badge", ""))
                        _pborder = "#D5A46A" if _pi == 0 else "#45474C"
                        _ppchg = _price_tag(float(_prow.get("predicted_price_change", 0) or 0))
                        _phtml = (
                            f"<div class='fpl-card' style='border-color:{_pborder};text-align:center;'>"
                            f"<div style='font-size:1.4rem;margin-bottom:0.35rem;'>{_medals_list[_pi]}</div>"
                            f"<div style='display:flex;justify-content:center;margin-bottom:0.3rem;'>"
                            f"<img class='player-face' src='{_pface}' onerror=\"this.onerror=null;this.style.display='none';\" /></div>"
                            f"<div style='display:flex;justify-content:center;align-items:center;gap:0.3rem;'>"
                            f"<img class='team-badge' src='{_pbadge}' onerror=\"this.onerror=null;this.style.display='none';\" />"
                            f"<div style='font-weight:800;font-size:0.98rem;'>{_pname}</div></div>"
                            f"<div style='color:#C4BCB5;font-size:0.76rem;margin-top:0.22rem;'>{_pteam} | {_prun}{_ppchg}</div>"
                            f"<div style='font-family:Space Mono;font-size:1.25rem;color:#BFB48F;margin-top:0.45rem;'>{_xpts(_prow):.2f} xPts</div>"
                            f"<div style='font-size:0.7rem;color:#C9BDC3;margin-top:0.15rem;'>{_safe_text(_psubline)}</div>"
                            f"<div style='display:flex;justify-content:center;gap:0.28rem;flex-wrap:wrap;margin-top:0.25rem;'>"
                            f"<span class='xi-role'>Reliability {float(_prow.get('reliability',1.0))*100:.0f}%</span>"
                            f"<span class='xi-role'>Upside {float(_prow.get('upside',0.0)):.1f}</span>"
                            f"<span class='xi-role'>Conf {float(_prow.get('captain_confidence',60)):.0f}%</span>"
                            f"</div></div>"
                        )
                        st.markdown(_phtml, unsafe_allow_html=True)

                # VC rec box
                if not top_vc.empty:
                    _vc_row2 = top_vc.iloc[0]
                    _vc_chance2 = int(round(float(_vc_row2.get("reliability", 1.0) * 100)))
                    st.markdown(
                        f"<div class='rec-box' style='margin-top:0.75rem;'>"
                        f"<div class='kpi-label'>VICE CAPTAIN</div>"
                        f"<div style='font-weight:800;font-size:0.98rem;margin-top:0.18rem;'>"
                        f"{_safe_text(_vc_row2['player_name'])}</div>"
                        f"<div style='font-size:0.8rem;color:#C4BCB5;margin-top:0.18rem;'>"
                        f"{_xpts(_vc_row2):.2f} xPts · {_vc_chance2}% reliability"
                        f" · {_safe_text(_vc_row2.get('fixture_run_label','?'))}"
                        f" · VC return if captain misses: {float(_vc_row2.get('captain_expected_return',0.0)):.1f}"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

            # Advanced analysis expanders
            if feature_capabilities.get("captain_mc") and run_monte_carlo_captain:
                with st.expander("Monte Carlo Captain Analysis (1,000 simulations)", expanded=False):
                    try:
                        _mc = run_monte_carlo_captain(my_team)
                        if _mc:
                            render_insight_table(pd.DataFrame([{
                                "Player": r.get("player_name", "?"),
                                "Win %": f"{float(r.get('win_prob', 0))*100:.1f}%",
                                "Cap EV": round(float(r.get("captain_ev", 0.0)), 2),
                                "Gain vs Others": round(float(r.get("expected_captain_gain", 0.0)), 2),
                                "Run": r.get("fixture_run", "?"),
                                "DGW": "Yes" if float(r.get("double_gws", 0) or 0) > 0 else "No",
                            } for r in _mc[:5]]), row_density="compact")
                    except Exception as _e_mc:
                        st.caption(f"Monte Carlo unavailable: {_e_mc}")

            if feature_capabilities.get("captain_diff") and get_captaincy_differential_analysis:
                with st.expander("Captaincy Differential (vs Average Manager)", expanded=False):
                    try:
                        _cdiff = get_captaincy_differential_analysis(my_team, bootstrap)
                        if _cdiff:
                            st.caption(f"Average manager captain EV: {float(_cdiff[0].get('field_captain_ev', 0.0)):.1f}")
                            render_insight_table(pd.DataFrame([{
                                "Player": r.get("player_name", "?"),
                                "Ownership %": round(float(r.get("ownership_pct", 0.0)), 1),
                                "Cap EV": round(float(r.get("captain_ev", 0.0)), 2),
                                "vs Field": round(float(r.get("differential_gain", 0.0)), 2),
                                "Verdict": "Differential" if bool(r.get("is_differential", False)) else "Template",
                                "Run": r.get("fixture_run", "?"),
                            } for r in _cdiff]), row_density="compact")
                    except Exception as _e_diff:
                        st.caption(f"Captaincy differential unavailable: {_e_diff}")

            # xPts bar chart
            st.divider()
            render_section_header("Full Squad xPts Ranking")
            _cap_sort_col = "captain_ev" if has_cap_ev else "xpts_score"
            _cap_sorted = cap_df.sort_values(_cap_sort_col, ascending=True)
            _fig_cap = go.Figure()
            _fig_cap.add_trace(go.Bar(
                x=_cap_sorted[_cap_sort_col],
                y=_cap_sorted["player_name"],
                orientation="h",
                marker=dict(color=_cap_sorted[_cap_sort_col], colorscale=PLOTLY_XPTS_SCALE),
                hovertemplate="<b>%{y}</b><br>%{x:.3f}<extra></extra>",
            ))
            _fig_cap.update_layout(
                **PLOTLY_THEME, height=420,
                xaxis_title="Captain EV" if has_cap_ev else "xPts Captain Score",
                margin=dict(l=10, r=10, t=20, b=30),
            )
            st.plotly_chart(_fig_cap, use_container_width=True, config={"displayModeBar": "hover", "responsive": True, "scrollZoom": True})

            render_section_header("Captain Matrix: Upside vs Reliability")
            _fig_matrix = px.scatter(
                cap_df.copy(),
                x="reliability",
                y="upside",
                size="xpts_val",
                color="position",
                hover_name="player_name",
                labels={
                    "reliability": "Reliability (p_plays_full)" if "p_plays_full" in cap_df.columns else "Reliability",
                    "upside": "Upside Score",
                },
                color_discrete_map=POSITION_COLOR_MAP,
            )
            _fig_matrix.update_layout(**PLOTLY_THEME, height=360, margin=dict(l=10, r=10, t=20, b=30))
            _fig_matrix.update_xaxes(tickformat=".0%")
            st.plotly_chart(_fig_matrix, use_container_width=True, config={"displayModeBar": "hover", "responsive": True, "scrollZoom": True})
    st.divider()

    if xi_result and not xi_result["bench"].empty:
        with st.expander("Bench details", expanded=True):
            bench_cols = st.columns(4)
            for i, (_, row) in enumerate(xi_result["bench"].iterrows()):
                with bench_cols[i % len(bench_cols)]:
                    chance = chance_map.get(int(row["player_id"]))
                    badge  = ("Red" if chance is not None and chance < 75 else
                              "Amber" if chance is not None and chance < 100 else "Green")
                    p_full = float(row.get("p_plays_full", (chance / 100 if chance is not None else 1.0)) or 1.0)
                    pchg = float(row.get("predicted_price_change", 0) or 0)
                    subtitle = (
                        f"{row['position']} · £{row['price']:.1f}M · {_xpts(row):.2f} xPts · "
                        f"{badge} {f'{chance}%' if chance is not None else '100%'} · Full {p_full:.0%}{_price_tag(pchg)}"
                    )
                    st.markdown(f"""
                    <div class='fpl-card'>
                        <div style='font-size:0.7rem; color:var(--accent);
                                    font-family: Space Mono; margin-bottom:0.3rem;'>
                            #{i+1} {'Emergency GK' if row['position']=='Goalkeeper'
                                   else 'First Sub' if i==0 else ''}
                        </div>
                        {player_identity_html(
                            row['player_name'],
                            row.get('team_name', ''),
                            row.get('player_face', player_face_map.get(int(row['player_id']), '')),
                            row.get('team_badge', team_badge_map.get(int(row.get('team_id', 0)), '')),
                            subtitle,
                            'player-face-sm',
                        )}
                    </div>
                    """, unsafe_allow_html=True)

    st.divider()
    injury_players = [
        (row["player_name"], chance_map.get(int(row["player_id"])),
         news_map.get(int(row["player_id"]), ""))
        for _, row in my_team.iterrows()
        if chance_map.get(int(row["player_id"])) is not None
        and chance_map.get(int(row["player_id"])) < 100
    ]


    # ── Injury & availability — visible outside expander ───────────────────
    if injury_players:
        st.divider()
        render_section_header("Injury & Availability Urgency")
        inj_df = pd.DataFrame(
            [
                {
                    "Player": n,
                    "Chance %": int(c or 0),
                    "Urgency": "High" if (c or 0) < 60 else "Medium" if (c or 0) < 85 else "Low",
                    "Notes": news or "No news available",
                }
                for n, c, news in injury_players
            ]
        ).sort_values(["Chance %", "Player"], ascending=[True, True])
        render_insight_table(inj_df, default_sort=("Chance %", True), row_density="compact")


    with st.expander("Full squad stats", expanded=False):
        render_section_header("Full Squad Stats")

        disp_cols = [
            "player_face", "team_badge", "player_name", "position", "team_name", "price",
            "predicted_pts"
        ]
        if "expected_pts" in my_team.columns:
            disp_cols.append("expected_pts")
        if "pts_low" in my_team.columns and "pts_high" in my_team.columns:
            disp_cols += ["pts_low", "pts_high"]
        disp_cols += ["combined_score", "avg_difficulty", "fixture_run_label", "blank_gws", "double_gws"]
        if "predicted_price_change" in my_team.columns:
            disp_cols.append("predicted_price_change")
        disp = my_team[disp_cols].copy()
        disp["ownership"] = disp.index.map(
            lambda i: f"{ownership_map.get(int(my_team.loc[i,'player_id']), 0):.1f}%"
            if i in my_team.index else "?"
        )
        disp = disp.rename(columns={
            "player_face": "Face",
            "team_badge": "Badge",
            "player_name": "Player", "position": "Pos",
            "team_name": "Team", "price": "£",
            "predicted_pts": "Pred", "expected_pts": "xPts", "pts_low": "Q10", "pts_high": "Q90",
            "predicted_price_change": "Price Δ", "combined_score": "5GW Score",
            "avg_difficulty": "Avg Diff", "fixture_run_label": "Run",
            "blank_gws": "Blanks", "double_gws": "DGWs",
            "ownership": "Owned%"
        }).sort_values("xPts" if "xPts" in disp.rename(columns={"expected_pts": "xPts"}).columns else "Pred", ascending=False)

        st.dataframe(disp, use_container_width=True, hide_index=True,
                     column_config={
                         "Face": st.column_config.ImageColumn("Face", width="small"),
                         "Badge": st.column_config.ImageColumn("Badge", width="small"),
                         "Pred": st.column_config.NumberColumn(format="%.2f"),
                         "xPts": st.column_config.NumberColumn(format="%.2f"),
                         "Q10": st.column_config.NumberColumn(format="%.2f"),
                         "Q90": st.column_config.NumberColumn(format="%.2f"),
                         "5GW Score": st.column_config.NumberColumn(format="%.2f"),
                         "Avg Diff": st.column_config.NumberColumn(format="%.1f"),
                         "£": st.column_config.NumberColumn(format="£%.1f"),
                     })

        if not value_breakdown.empty:
            st.divider()
            render_section_header("Sell Price Breakdown")
            vb_cols = [c for c in ["player_name", "buy_price", "current_price", "sell_price"] if c in value_breakdown.columns]
            if vb_cols:
                vb = value_breakdown[vb_cols].copy().rename(columns={
                    "player_name": "Player",
                    "buy_price": "Buy £",
                    "current_price": "Current £",
                    "sell_price": "Sell £",
                })
                render_insight_table(vb, row_density="compact")


    # ── Inline AI bar ─────────────────────────────────────────────────────────
    st.divider()
    _ai_col_q, _ai_col_btn = st.columns([4, 1])
    with _ai_col_q:
        _inline_question = st.text_input(
            "Ask AI",
            placeholder=f"Ask about your squad for GW{current_gw+1}...",
            key=f"inline_ai_{page}",
            label_visibility="collapsed",
        )
    with _ai_col_btn:
        _ask_pressed = st.button(
            "Ask AI ◇",
            use_container_width=True,
            key=f"ask_ai_{page}",
        )
    if _ask_pressed and _inline_question:
        st.session_state.setdefault("analyst_messages", []).append(
            {"role": "user", "content": _inline_question}
        )
        st.session_state["_pending_nav"] = "AI Analyst"
        st.rerun()


elif page == "Fixture Planner":
    render_page_hero(
        "Fixture Planner",
        "Scan fixture turns, blank risk, and near-term opportunities before transfer decisions.",
        [
            f"Live GW window from GW{current_gw+1}",
            "Heatmap + Compare",
            "Fixture-driven shortlist",
        ],
    )

    render_section_header("Fixture Difficulty Heatmap")

    teams_df2 = pd.DataFrame(bootstrap["teams"])
    all_team_names = teams_df2["name"].tolist()
    future_gws = sorted(
        int(gw) for gw in fixtures_df["event"].dropna().unique().tolist()
        if int(gw) >= int(current_gw + 1)
    )
    if not future_gws:
        st.warning("No upcoming gameweeks found in fixture data.")
        st.stop()
    default_start = future_gws[0]
    default_end = future_gws[min(len(future_gws) - 1, 4)]

    preset_map = {
        "My Squad Teams": sorted(my_team["team_name"].dropna().unique().tolist()),
        "Top 6": [t for t in ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man Utd", "Tottenham"] if t in all_team_names],
        "Promoted / Budget": [t for t in ["Burnley", "Leeds", "Sunderland"] if t in all_team_names],
        "Custom": [],
    }
    show_all_teams = False
    render_section_header("Filters")
    col1, col2, col3, col4 = st.columns([2, 1.4, 1, 1.2])
    with col1:
        position_filter = st.multiselect(
            "Filter by position",
            ["Goalkeeper", "Defender", "Midfielder", "Forward"],
            default=["Midfielder", "Forward"],
        )
    with col3:
        show_all_teams = st.toggle("Show all 20 teams", value=False)
        gw_start, gw_end = st.select_slider(
            "GW range",
            options=future_gws,
            value=(default_start, default_end),
        )
    with col2:
        preset = st.selectbox("Team preset", list(preset_map.keys()))
        custom_team_selection = st.multiselect(
            "Custom teams",
            all_team_names,
            default=preset_map.get(preset, []),
            disabled=show_all_teams,
        )
    with col4:
        sort_mode = st.selectbox("Sort mode", ["Easiest overall", "Easiest next 2", "Blank risk"])

    # Build fixture heatmap data
    gws = [gw for gw in future_gws if gw_start <= gw <= gw_end]
    gw_count = len(gws)
    if gw_count == 0:
        st.info("Selected GW range contains no fixtures. Pick a wider range.")
        st.stop()
    render_stat_cards(
        [
            {"label": "GW Window", "value": f"{gw_count}", "delta": f"GW{gw_start} to GW{gw_end}", "tone": "neutral"},
            {"label": "All Teams", "value": "Yes" if show_all_teams else "Filtered", "delta": "Display mode", "tone": "positive" if show_all_teams else "neutral"},
            {"label": "Sort", "value": sort_mode, "delta": "Ranking rule", "tone": "neutral"},
        ]
    )

    if show_all_teams:
        display_teams = teams_df2["name"].tolist()
        team_ids = teams_df2["id"].tolist()
    else:
        chosen_names = custom_team_selection or preset_map.get(preset, [])
        if chosen_names:
            filtered_teams = teams_df2[teams_df2["name"].isin(chosen_names)]
            team_ids = filtered_teams["id"].tolist()
            display_teams = filtered_teams["name"].tolist()
        else:
            filtered = my_team[my_team["position"].isin(position_filter)] if position_filter else my_team
            team_ids = filtered["team_id"].unique().tolist()
            display_teams = [team_name_map.get(tid, str(tid)) for tid in team_ids]

    # Pre-index fixtures by (team_id, gw) — O(1) lookup replaces 760+ DataFrame filter ops
    _fix_index: dict = {}
    for _fr in fixtures_df.to_dict("records"):
        _fgw = _fr.get("event")
        _fth = _fr.get("team_h")
        _fta = _fr.get("team_a")
        if _fgw is None or _fth is None or _fta is None:
            continue
        try:
            _fgw, _fth, _fta = int(_fgw), int(_fth), int(_fta)
        except (ValueError, TypeError):
            continue
        _opp_h = team_name_map.get(_fta, "?")
        _fix_index.setdefault((_fth, _fgw), []).append((
            team_short_map.get(_fta, _opp_h[:3].upper()), _opp_h, "H",
            float(_fr.get("team_h_difficulty", 3)), team_badge_map.get(_fta, ""),
        ))
        _opp_a = team_name_map.get(_fth, "?")
        _fix_index.setdefault((_fta, _fgw), []).append((
            team_short_map.get(_fth, _opp_a[:3].upper()), _opp_a, "A",
            float(_fr.get("team_a_difficulty", 3)), team_badge_map.get(_fth, ""),
        ))

    # Build matrix
    matrix = []
    hover_txt = []
    hover_meta = []
    cell_labels = []
    blank_counts = []
    avg_diffs = []
    gw_labels = [f"GW{g}" for g in gws]

    for tid in team_ids:
        row_diffs = []
        row_hover = []
        row_meta = []
        row_labels = []
        blanks = 0
        for gw in gws:
            fixture_entries = _fix_index.get((int(tid), gw), [])
            fixture_count = len(fixture_entries)
            if fixture_count == 0:
                row_diffs.append(0)
                row_labels.append("<b>Blank</b>")
                row_hover.append("Blank gameweek")
                row_meta.append(["Blank", "", "-", 0.0, 0])
                blanks += 1
                continue

            avg_diff = float(np.mean([x[3] for x in fixture_entries]))
            first = fixture_entries[0]
            first_short, first_opp, first_venue, _, first_badge = first

            if fixture_count == 1:
                row_diffs.append(avg_diff)
                row_labels.append(f"{first_short} ({first_venue})<br><b>{avg_diff:.1f}</b>")
                row_hover.append(f"vs {first_opp} ({'Home' if first_venue == 'H' else 'Away'}) · Difficulty {avg_diff:.1f}")
                row_meta.append([first_opp, first_badge, first_venue, avg_diff, 1])
            else:
                opp_tokens = [f"{x[0]} ({x[2]})" for x in fixture_entries[:2]]
                if fixture_count > 2:
                    opp_tokens.append(f"+{fixture_count - 2}")
                opp_label = "/".join(opp_tokens)
                venues = "/".join([x[2] for x in fixture_entries[:2]])
                row_diffs.append(avg_diff)
                row_labels.append(f"{opp_label}<br><b>{avg_diff:.1f}</b> x{fixture_count}")
                row_hover.append(f"{fixture_count} fixtures · Avg difficulty {avg_diff:.1f}")
                row_meta.append([first_opp, first_badge, venues, avg_diff, fixture_count])

        non_blank = [d for d in row_diffs if d > 0]
        avg_diff = float(np.mean(non_blank)) if non_blank else 6.0
        matrix.append(row_diffs)
        hover_txt.append(row_hover)
        hover_meta.append(row_meta)
        cell_labels.append(row_labels)
        blank_counts.append(blanks)
        avg_diffs.append(avg_diff)

    if matrix:
        next2 = []
        for row in matrix:
            non_blank2 = [d for d in row[:2] if d > 0]
            next2.append(float(np.mean(non_blank2)) if non_blank2 else 6.0)
        if sort_mode == "Easiest next 2":
            order = sorted(range(len(display_teams)), key=lambda i: (next2[i], blank_counts[i], avg_diffs[i]))
        elif sort_mode == "Blank risk":
            order = sorted(range(len(display_teams)), key=lambda i: (blank_counts[i], avg_diffs[i]))
        else:
            order = sorted(range(len(display_teams)), key=lambda i: (avg_diffs[i], blank_counts[i]))
        display_teams = [display_teams[i] for i in order]
        matrix = [matrix[i] for i in order]
        hover_txt = [hover_txt[i] for i in order]
        hover_meta = [hover_meta[i] for i in order]
        cell_labels = [cell_labels[i] for i in order]
        avg_diffs = [avg_diffs[i] for i in order]
        blank_counts = [blank_counts[i] for i in order]
        next2 = [next2[i] for i in order]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Best Run", display_teams[0], f"{avg_diffs[0]:.2f} avg diff")
        k2.metric("Worst Run", display_teams[-1], f"{avg_diffs[-1]:.2f} avg diff")
        k3.metric("Total Blanks", sum(blank_counts), f"{len(gws)} GW window")
        k4.metric("Teams Tracked", len(display_teams), "ranked easiest → hardest")

        ranked = pd.DataFrame(
            {
                "Team": display_teams,
                "Avg Diff": avg_diffs,
                "Next2 Diff": next2,
                "Blanks": blank_counts,
            }
        )
        ranked["Swing (last2-first2)"] = ranked["Avg Diff"] - ranked["Next2 Diff"]
        target_now = ranked.nsmallest(3, ["Next2 Diff", "Avg Diff"])
        avoid_now = ranked.nlargest(3, ["Next2 Diff", "Avg Diff"])
        swing_row = ranked.sort_values("Swing (last2-first2)", ascending=False).iloc[0]
        render_decision_banner(
            title="Fixture Decision",
            primary_action=f"Target now: {', '.join(target_now['Team'].head(2).tolist())}",
            confidence=compute_fixture_decision_confidence(avg_diffs, next2, blank_counts),
            reasons=[
                f"Avoid now: {', '.join(avoid_now['Team'].head(2).tolist())}",
                f"Best 2-GW swing: {swing_row['Team']} ({swing_row['Swing (last2-first2)']:+.2f})",
                f"Sort mode: {sort_mode}",
            ],
            risk_level="Medium" if int(sum(blank_counts)) > 0 else "Low",
        )

        colorscale = [
            [0.00, "#ffffff"],
            [0.01, "#ffffff"],
            [0.20, "#1f8f65"],
            [0.40, "#27e8a7"],
            [0.58, "#ffb547"],
            [0.78, "#ff7a67"],
            [1.00, "#ff5d73"],
        ]

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=gw_labels,
            y=display_teams,
            text=cell_labels,
            customdata=hover_meta,
            texttemplate="%{text}",
            textfont=dict(size=10, color=PLOTLY_TEXT, family="Space Mono"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{x}: %{customdata[0]} (%{customdata[2]})<br>"
                "Difficulty: %{customdata[3]}<br>"
                "Fixtures: %{customdata[4]}<br>"
                "<extra></extra>"
            ),
            colorscale=colorscale,
            zmin=0, zmax=5,
            xgap=2,
            ygap=2,
            showscale=True,
            colorbar=dict(
                title=dict(text="Difficulty", font=dict(color=PLOTLY_ACCENT)),
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["Easy", "2", "3", "4", "Hard"],
                tickfont=dict(color=PLOTLY_TEXT),
            ),
        ))

        fig.update_layout(
            **PLOTLY_THEME,
            height=max(380, len(display_teams) * 44 + 120),
            margin=dict(l=10, r=10, t=52, b=30),
            title=dict(
                text=f"Fixture Radar · GW{gws[0]} to GW{gws[-1]}",
                font=dict(color=PLOTLY_ACCENT, size=13, family="Space Mono"),
            ),
            dragmode=False,
        )
        fig.update_yaxes(autorange="reversed", fixedrange=True)
        fig.update_xaxes(fixedrange=True)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
                "scrollZoom": False,
                "doubleClick": False,
                "showTips": False,
            },
        )
        st.caption("Each cell shows opponent + venue (H/A) and fixture difficulty. Blanks are white. Zoom/pan is disabled for readability.")

        render_section_header("Two-Team Compare")
        compare_teams = st.multiselect(
            "Pick two teams",
            display_teams,
            default=display_teams[:2] if len(display_teams) >= 2 else display_teams,
            max_selections=2,
            key="fixture_compare_teams",
        )
        if len(compare_teams) == 2:
            a_idx = display_teams.index(compare_teams[0])
            b_idx = display_teams.index(compare_teams[1])
            cmp_rows = []
            for j, gw in enumerate(gws):
                a_d = matrix[a_idx][j] if j < len(matrix[a_idx]) else 0
                b_d = matrix[b_idx][j] if j < len(matrix[b_idx]) else 0
                cmp_rows.append(
                    {
                        "GW": f"GW{gw}",
                        f"{compare_teams[0]} Diff": a_d if a_d > 0 else "BLK",
                        f"{compare_teams[1]} Diff": b_d if b_d > 0 else "BLK",
                        "Delta (A-B)": (a_d - b_d) if (a_d > 0 and b_d > 0) else np.nan,
                    }
                )
            cmp_df = pd.DataFrame(cmp_rows)
            render_insight_table(
                cmp_df,
                row_density="compact",
                column_config={
                    "Delta (A-B)": st.column_config.NumberColumn(format="%+.1f"),
                },
            )
    else:
        st.info("No teams available for the selected filters. Adjust preset/team selection.")

    st.divider()

    render_section_header("Top Transfer Targets by Fixture Run")

    pos_tab = st.tabs(["Goalkeeper", "Defender", "Midfielder", "Forward"])
    for pos, tab in zip(["Goalkeeper", "Defender", "Midfielder", "Forward"], pos_tab):
        with tab:
            top = (
                others[others["position"] == pos]
                .nlargest(10, "combined_score")
                [[
                    "player_face", "team_badge", "player_name", "team_name", "price",
                    "predicted_pts", "avg_difficulty",
                    "fixture_run_label", "blank_gws",
                    "double_gws", "combined_score", "value_score"
                ]]
                .copy()
            )
            top["ownership"] = top.index.map(
                lambda i: f"{ownership_map.get(int(others.loc[i,'player_id']),0):.1f}%"
                if i in others.index else "?"
            )
            st.dataframe(top.rename(columns={
                "player_face":"Face",
                "team_badge":"Badge",
                "player_name":"Player","team_name":"Team","price":"£",
                "predicted_pts":"Pred","avg_difficulty":"Avg Diff",
                "fixture_run_label":"Run","blank_gws":"Blanks",
                "double_gws":"DGWs","combined_score":"5GW Score",
                "value_score":"Val/£M","ownership":"Owned%"
            }), use_container_width=True, hide_index=True,
            column_config={
                "Face": st.column_config.ImageColumn("Face", width="small"),
                "Badge": st.column_config.ImageColumn("Badge", width="small"),
            })


elif page == "Transfer Planner":
    render_page_hero(
        "Transfer Planner",
        "ILP-backed move suggestions, ranked scenarios, and before/after XI impact.",
        [
            f"Bank £{bank_balance:.1f}M",
            "1 FT" if transfers_made == 0 else "FT Used",
            f"Chips {len(available_chips)}",
        ],
    )

    # ── Focus player mode: triggered from My Squad player inspector ──────────
    _focus_player = st.session_state.get("transfer_focus_player")
    if _focus_player:
        _focus_row_match = my_team[my_team["player_name"] == _focus_player]
        if not _focus_row_match.empty:
            _fr = _focus_row_match.iloc[0]
            _fp_pos    = str(_fr.get("position", ""))
            _fp_price  = float(_fr.get("price", 0))
            _fp_budget = _fp_price + bank_balance
            _fp_xpts   = float(_xpts(_fr))
            _fp_face   = _safe_text(_fr.get("player_face", ""))
            _fp_badge  = _safe_text(_fr.get("team_badge", ""))
            _fp_team   = _safe_text(_fr.get("team_name", ""))
            _fp_run    = _safe_text(_fr.get("fixture_run_label", "?"))
            _onerr     = "this.onerror=null;this.style.display='none';"

            # ── Header banner ─────────────────────────────────────────────────
            _focus_banner = (
                "<div class='rec-box' style='margin-bottom:0.85rem;'>"
                "<div class='kpi-label'>FOCUSED TRANSFER \u2014 from My Squad</div>"
                "<div style='display:flex;align-items:center;gap:0.55rem;margin-top:0.35rem;'>"
                f"<img class='player-face-sm' src='{_fp_face}' onerror=\"{_onerr}\" />"
                f"<img class='team-badge' src='{_fp_badge}' onerror=\"{_onerr}\" />"
                "<div>"
                f"<div style='font-weight:800;font-size:1rem;'>{_safe_text(_focus_player)}</div>"
                f"<div style='font-size:0.78rem;color:#C4BCB5;'>"
                f"{_fp_team} \u00b7 \u00a3{_fp_price:.1f}M \u00b7 {_fp_xpts:.2f} xPts \u00b7 Run: {_fp_run}"
                "</div>"
                f"<div style='font-size:0.72rem;color:#BFB48F;margin-top:0.12rem;'>"
                f"Budget for replacement: \u00a3{_fp_budget:.1f}M"
                "</div>"
                "</div></div></div>"
            )
            st.markdown(_focus_banner, unsafe_allow_html=True)

            # ── Confirmed selection banner ────────────────────────────────────
            _confirmed_out = st.session_state.get("confirmed_transfer_out", "")
            _confirmed_in  = st.session_state.get("confirmed_transfer_in", "")
            if _confirmed_out == _focus_player and _confirmed_in:
                _in_match  = enriched_df[enriched_df["player_name"] == _confirmed_in]
                _in_xpts   = float(_xpts(_in_match.iloc[0])) if not _in_match.empty else 0.0
                _in_price  = float(_in_match.iloc[0].get("price", 0)) if not _in_match.empty else 0.0
                _conf_gain = _in_xpts - _fp_xpts
                _conf_cost = _in_price - _fp_price
                _conf_col  = "#BFB48F" if _conf_gain >= 0 else "#D36C73"
                st.markdown(
                    "<div class='fpl-card' style='border-color:#BFB48F;padding:0.75rem 0.95rem;margin-bottom:0.75rem;'>"
                    "<div class='kpi-label'>PLANNED TRANSFER</div>"
                    f"<div style='font-size:0.95rem;font-weight:800;margin-top:0.2rem;'>"
                    f"{_safe_text(_focus_player)}"
                    f" <span style='color:#C4BCB5;font-weight:400;'>OUT</span>"
                    f" \u00a0\u2192\u00a0 "
                    f"{_safe_text(_confirmed_in)}"
                    f" <span style='color:#C4BCB5;font-weight:400;'>IN</span>"
                    "</div>"
                    f"<div style='font-size:0.78rem;color:{_conf_col};margin-top:0.2rem;font-weight:700;'>"
                    f"{_conf_gain:+.2f} xPts \u00a0\u00b7\u00a0 {_conf_cost:+.1f}M"
                    "</div>"
                    "<div style='font-size:0.72rem;color:#C4BCB5;margin-top:0.12rem;'>"
                    "Scroll down to the ILP tabs for full analysis and before/after XI preview."
                    "</div></div>",
                    unsafe_allow_html=True,
                )

            # ── Compute replacements ──────────────────────────────────────────
            _same_pos = others[others["position"] == _fp_pos].copy()
            _same_pos["_xpts"] = _same_pos.apply(lambda r: _xpts(r), axis=1)
            _affordable = _same_pos[_same_pos["price"] <= _fp_budget + 0.05]
            _top_reps = _affordable.nlargest(5, "_xpts")

            if not _top_reps.empty:
                render_section_header(
                    f"Best {_fp_pos} replacements within \u00a3{_fp_budget:.1f}M"
                )

                # Column headers
                _hc1, _hc2, _hc3, _hc4, _hc5 = st.columns([2.8, 0.55, 2.8, 1.3, 0.9])
                _hc1.markdown(
                    "<div style='font-size:0.62rem;font-family:Space Mono;"
                    "color:#C4BCB5;letter-spacing:0.1em;padding-bottom:0.25rem;'>"
                    "OUT (CURRENT)</div>",
                    unsafe_allow_html=True,
                )
                _hc3.markdown(
                    "<div style='font-size:0.62rem;font-family:Space Mono;"
                    "color:#BFB48F;letter-spacing:0.1em;padding-bottom:0.25rem;'>"
                    "IN (REPLACEMENT)</div>",
                    unsafe_allow_html=True,
                )
                _hc4.markdown(
                    "<div style='font-size:0.62rem;font-family:Space Mono;"
                    "color:#C4BCB5;letter-spacing:0.1em;padding-bottom:0.25rem;'>"
                    "NEXT 3 GWs</div>",
                    unsafe_allow_html=True,
                )

                for _ri, (_, _rep) in enumerate(_top_reps.iterrows()):
                    _gain      = float(_rep["_xpts"]) - _fp_xpts
                    _cost_d    = float(_rep["price"]) - _fp_price
                    _gain_hex  = "#BFB48F" if _gain >= 0 else "#D36C73"
                    _arrow_sym = "\u2191" if _gain >= 0 else "\u2193"
                    _rep_face  = _safe_text(_rep.get("player_face", ""))
                    _rep_badge = _safe_text(_rep.get("team_badge", ""))
                    _rep_name  = _safe_text(_rep.get("player_name", "?"))
                    _rep_team  = _safe_text(_rep.get("team_name", ""))
                    _rep_run   = _safe_text(_rep.get("fixture_run_label", "?"))
                    _rep_price = float(_rep.get("price", 0))
                    _rep_xpts  = float(_rep["_xpts"])
                    _rep_own   = float(ownership_map.get(int(_rep.get("player_id", 0) or 0), 0))
                    _rep_pchg  = _price_tag(float(_rep.get("predicted_price_change", 0) or 0))
                    _is_sel    = (_confirmed_in == _rep.get("player_name", ""))
                    _row_bdr   = "border-color:#BFB48F;" if _is_sel else ""

                    _c1, _c2, _c3, _c4, _c5 = st.columns([2.8, 0.55, 2.8, 1.3, 0.9])

                    # OUT card
                    with _c1:
                        st.markdown(
                            "<div class='transfer-card' style='padding:0.5rem 0.65rem;margin-bottom:0.3rem;'>"
                            "<div style='display:flex;align-items:center;gap:0.38rem;'>"
                            f"<img class='player-face-sm' src='{_fp_face}' onerror=\"{_onerr}\" />"
                            f"<img class='team-badge' src='{_fp_badge}' onerror=\"{_onerr}\" />"
                            "<div style='min-width:0;'>"
                            f"<div style='font-weight:700;font-size:0.8rem;white-space:nowrap;"
                            f"overflow:hidden;text-overflow:ellipsis;'>{_safe_text(_focus_player)}</div>"
                            f"<div style='font-size:0.66rem;color:#C4BCB5;'>{_fp_team} \u00b7 \u00a3{_fp_price:.1f}M</div>"
                            f"<div style='font-size:0.66rem;color:#C4BCB5;'>{_fp_xpts:.2f} xPts</div>"
                            "</div></div></div>",
                            unsafe_allow_html=True,
                        )

                    # Arrow + delta
                    with _c2:
                        st.markdown(
                            f"<div style='text-align:center;padding-top:0.55rem;'>"
                            f"<div style='font-size:1.05rem;color:{_gain_hex};font-weight:800;'>"
                            f"{_arrow_sym}</div>"
                            f"<div style='font-size:0.62rem;color:{_gain_hex};font-weight:700;'>"
                            f"{_gain:+.1f}</div>"
                            f"<div style='font-size:0.58rem;color:#C4BCB5;'>{_cost_d:+.1f}M</div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )

                    # IN card
                    with _c3:
                        st.markdown(
                            f"<div class='transfer-card' style='padding:0.5rem 0.65rem;"
                            f"margin-bottom:0.3rem;{_row_bdr}'>"
                            "<div style='display:flex;align-items:center;gap:0.38rem;'>"
                            f"<img class='player-face-sm' src='{_rep_face}' onerror=\"{_onerr}\" />"
                            f"<img class='team-badge' src='{_rep_badge}' onerror=\"{_onerr}\" />"
                            "<div style='min-width:0;'>"
                            f"<div style='font-weight:700;font-size:0.8rem;white-space:nowrap;"
                            f"overflow:hidden;text-overflow:ellipsis;'>"
                            f"{_rep_name}{_rep_pchg}</div>"
                            f"<div style='font-size:0.66rem;color:#C4BCB5;'>"
                            f"{_rep_team} \u00b7 \u00a3{_rep_price:.1f}M</div>"
                            f"<div style='font-size:0.66rem;color:#BFB48F;font-weight:700;'>"
                            f"{_rep_xpts:.2f} xPts \u00b7 {_rep_own:.1f}% owned</div>"
                            "</div></div></div>",
                            unsafe_allow_html=True,
                        )

                    # Next 3 fixture pills
                    with _c4:
                        _gws_ahead = list(range(current_gw + 1, current_gw + 4))
                        _pills = []
                        for _gw in _gws_ahead:
                            _opp_raw  = _rep.get(f"gw{_gw}_opponent", "?")
                            _opp_str  = str(_opp_raw) if _opp_raw is not None else "?"
                            _diff_raw = _rep.get(f"gw{_gw}_difficulty", 3)
                            try:
                                _diff_f = float(_diff_raw)
                            except Exception:
                                _diff_f = 3.0
                            _home_raw = _rep.get(f"gw{_gw}_home", 0)
                            if _opp_str.upper() in {"BLANK", "B", "", "NAN"}:
                                _pill_col = "#C4BCB5"
                                _pill_txt = "BLK"
                            else:
                                _pill_col = (
                                    "#BFB48F" if _diff_f <= 2
                                    else "#D5A46A" if _diff_f <= 3
                                    else "#D36C73"
                                )
                                _venue = "H" if _home_raw else "A"
                                _pill_txt = f"{_opp_str[:3].upper()}({_venue})"
                            _pills.append(
                                f"<div style='font-size:0.6rem;color:{_pill_col};"
                                f"font-weight:700;line-height:1.6;'>{_pill_txt}</div>"
                            )
                        st.markdown(
                            f"<div style='padding-top:0.45rem;'>{''.join(_pills)}</div>",
                            unsafe_allow_html=True,
                        )

                    # Select button
                    with _c5:
                        _btn_type = "secondary" if _is_sel else "primary"
                        _btn_lbl  = "\u2713 Selected" if _is_sel else "Select"
                        if st.button(
                            _btn_lbl,
                            key=f"focus_sel_{_ri}_{_rep.get('player_name','')}",
                            use_container_width=True,
                            type=_btn_type,
                        ):
                            st.session_state["confirmed_transfer_out"] = _focus_player
                            st.session_state["confirmed_transfer_in"]  = str(_rep.get("player_name", ""))
                            st.rerun()

        # Clear button — centred, always visible when focus active
        _cleft, _cmid, _cright = st.columns([2, 1, 2])
        with _cmid:
            if st.button(
                "\u2715 Clear focus",
                key="clear_transfer_focus",
                use_container_width=True,
            ):
                for _k in ("transfer_focus_player", "confirmed_transfer_out", "confirmed_transfer_in"):
                    st.session_state.pop(_k, None)
                st.rerun()

        st.divider()

    render_section_header(
        f"Bank: £{bank_balance:.1f}M | "
        f"{'1 Free Transfer' if transfers_made == 0 else 'Free Transfer Used'}"
        f"{' | Hit analysis available' if transfers_made > 0 else ''}"
    )

    with st.spinner("Computing optimal transfers..."):
        ilp_1 = cached_ilp_transfers(my_team, others, float(bank_balance), n_transfers=1)
        ilp_2 = cached_ilp_transfers(my_team, others, float(bank_balance), n_transfers=2)
        roll   = cached_rolling_advice(
            my_team, others, float(bank_balance), int(transfers_made),
            json.dumps(chip_info, default=str), int(current_gw),
            json.dumps(ilp_1, default=str),
        )
        hit_transfers = cached_hit_analysis(my_team, others, float(bank_balance), int(transfers_made))

    rec = roll["recommendation"]
    rec_conf = compute_transfer_decision_confidence(rec, ilp_1, hit_transfers)
    rec_risk = "Low" if rec == "USE NOW" else "Medium" if rec == "BORDERLINE" else "High"
    render_decision_banner(
        title="Transfer Decision",
        primary_action=rec,
        confidence=rec_conf,
        reasons=roll.get("reasons", []),
        risk_level=rec_risk,
    )
    render_stat_cards(
        [
            {"label": "Recommendation", "value": rec, "delta": f"Confidence {rec_conf:.0f}%", "tone": "positive" if rec == "USE NOW" else "warning" if rec == "BORDERLINE" else "neutral"},
            {"label": "1FT Gain (5GW)", "value": f"{float(ilp_1.get('total_gain', 0.0)):+.2f}", "delta": "Best single transfer", "tone": "positive"},
            {"label": "2FT Gain (5GW)", "value": f"{float(ilp_2.get('total_gain', 0.0)):+.2f}", "delta": "Best double transfer", "tone": "neutral"},
            {"label": "Hit Options", "value": str(len(hit_transfers or [])), "delta": "Break-even+ candidates", "tone": "warning" if len(hit_transfers or []) else "neutral"},
        ]
    )

    transfer_face_map = (
        enriched_df[["player_name", "player_face"]]
        .drop_duplicates("player_name")
        .set_index("player_name")["player_face"]
        .to_dict()
    )
    transfer_team_name_map = (
        enriched_df[["player_name", "team_name"]]
        .drop_duplicates("player_name")
        .set_index("player_name")["team_name"]
        .to_dict()
    )
    transfer_badge_map = (
        enriched_df[["player_name", "team_badge"]]
        .drop_duplicates("player_name")
        .set_index("player_name")["team_badge"]
        .to_dict()
    )

    def _xi_chip_grid_html(names: set[str]) -> str:
        chips = []
        for nm in sorted(names):
            nm_s = _safe_text(nm)
            tm_s = _safe_text(transfer_team_name_map.get(nm, ""))
            face = _safe_text(transfer_face_map.get(nm, ""))
            badge = _safe_text(transfer_badge_map.get(nm, ""))
            chips.append(
                "<div style='display:flex; align-items:center; gap:0.38rem; "
                "padding:0.32rem 0.45rem; border:1px solid var(--line); border-radius:999px; "
                "background:var(--surface-soft); min-width:0;'>"
                f"<img class='player-face-sm' src='{face}' "
                "onerror=\"this.onerror=null;this.style.display='none';\" />"
                f"<img class='team-badge' src='{badge}' onerror=\"this.onerror=null;this.style.display='none';\" />"
                "<div style='min-width:0;'>"
                f"<div style='font-size:0.72rem; font-weight:700; color:var(--text); line-height:1.05;'>{nm_s}</div>"
                f"<div style='font-size:0.62rem; color:var(--muted); line-height:1.05;'>{tm_s}</div>"
                "</div>"
                "</div>"
            )
        return (
            "<div style='display:flex; flex-wrap:wrap; gap:0.34rem;'>"
            + "".join(chips)
            + "</div>"
        )

    # ── Recommendation summary (two-line strip above ILP tabs) ───────────────
    # Build candidates
    stack_candidates = []
    if ilp_1.get("transfers"):
        _t1 = ilp_1["transfers"][0]
        stack_candidates.append({
            "id":       "safe",
            "label":    "Safe move",
            "headline": f"{_t1['out_name']} → {_t1['in_name']}",
            "now":      float(ilp_1.get("total_next_gain", 0.0)),
            "horizon":  float(ilp_1.get("total_gain", 0.0)),
            "cost":     float(ilp_1.get("total_cost", 0.0)),
            "risk":     ["Low variance role", "Single transfer only"],
            "why":      [f"Run: {_t1.get('fixture_run','?')}", f"Position: {_t1.get('position','?')}"],
        })
    if ilp_2.get("transfers"):
        stack_candidates.append({
            "id":       "aggressive",
            "label":    "Aggressive",
            "headline": " + ".join([f"{x['out_name']} → {x['in_name']}" for x in ilp_2["transfers"][:2]]),
            "now":      float(ilp_2.get("total_next_gain", 0.0)),
            "horizon":  float(ilp_2.get("total_gain", 0.0)),
            "cost":     float(ilp_2.get("total_cost", 0.0)),
            "risk":     ["Higher variance", "Two moves lock flexibility"],
            "why":      ["Targets broader fixture turn", "Larger ceiling if both start"],
        })
    _diff_df = cached_differential_picks(others, bootstrap, top_n=1)
    _diff_candidate = None
    if not _diff_df.empty:
        _d = _diff_df.iloc[0]
        _diff_candidate = {
            "id":               "differential",
            "label":            "Differential",
            "headline":         f"Buy {_d['player_name']} ({float(_d.get('ownership_pct',0)):.1f}% owned)",
            "now":              float(_d.get("predicted_pts", 0.0)),
            "horizon":          float(_d.get("combined_score", 0.0)),
            "cost":             float(_d.get("price", 0.0)),
            "ownership_pct":    float(_d.get("ownership_pct", 0.0)),
            "differential_score": float(_d.get("differential_score", 0.0)),
            "risk":             ["Low ownership · minutes uncertainty"],
            "why":              [f"Run: {_d.get('fixture_run_label','?')}", f"Diff score: {float(_d.get('differential_score',0)):.2f}"],
        }
        stack_candidates.append(_diff_candidate)

    if stack_candidates:
        # Rank using utility score (same logic as before)
        _risk_penalty  = {"safe": 0.2, "aggressive": 0.6, "differential": 0.35}
        _rec_alignment = {
            "USE NOW":   {"safe": 0.4, "aggressive": 0.5, "differential": 0.2},
            "BORDERLINE":{"safe": 0.5, "aggressive": 0.1, "differential": 0.25},
            "HOLD":      {"safe": -0.2, "aggressive": -0.6, "differential": -0.3},
            "ROLL":      {"safe": -0.2, "aggressive": -0.6, "differential": -0.3},
        }
        _align = _rec_alignment.get(rec, {"safe": 0.0, "aggressive": 0.0, "differential": 0.0})
        _ranked = []
        for _sc in stack_candidates:
            _ng  = float(_sc.get("now", 0.0))
            _hg  = float(_sc.get("horizon", 0.0))
            _up  = max(0.0, _hg - _ng)
            _db  = 0.0
            if _sc.get("id") == "differential":
                _own_e  = float(_sc.get("ownership_pct", 15.0))
                _low_e  = max(0.0, min(1.0, (15.0 - _own_e) / 15.0))
                _ds_e   = max(0.0, float(_sc.get("differential_score", 0.0)))
                _db     = 0.35 * _low_e + 0.25 * min(_ds_e, 2.0)
            _util = (
                0.8 * _ng + 1.25 * _hg + 0.6 * _up
                - _risk_penalty.get(_sc.get("id",""), 0.3)
                + _align.get(_sc.get("id",""), 0.0)
                + _db
            )
            _ranked.append((_util, _sc))
        _ranked.sort(key=lambda x: x[0], reverse=True)
        _primary     = _ranked[0][1]
        _secondaries = [sc for _, sc in _ranked[1:]]

        # ── Line 1: Primary recommendation ────────────────────────────────
        _prim_conf  = float(np.clip(60 + 7 * max(0.0, _primary["horizon"] - _primary["now"]), 40, 88))
        _prim_gain  = float(_primary["horizon"])
        _prim_cost  = float(_primary["cost"])
        _prim_label = _safe_text(_primary["label"])
        _prim_hl    = _safe_text(_primary["headline"])
        _prim_gain_col = "#BFB48F" if _prim_gain >= 0 else "#D36C73"

        # ── Line 2: Alternatives summary ───────────────────────────────────
        _alt_parts = []
        for _s in _secondaries[:2]:
            _s_hl   = _safe_text(_s["headline"])
            _s_gain = float(_s["horizon"])
            _s_col  = "#BFB48F" if _s_gain >= 0 else "#D36C73"
            _alt_parts.append(
                f"<span style='color:#C4BCB5;'>{_safe_text(_s['label'])}:</span> "
                f"<span style='font-weight:700;'>{_s_hl}</span> "
                f"<span style='color:{_s_col};'>({_s_gain:+.1f})</span>"
            )
        _alt_html = " &nbsp;·&nbsp; ".join(_alt_parts) if _alt_parts else ""

        # ── Differential pill (separate, not in tabs) ──────────────────────
        _diff_pill_html = ""
        if _diff_candidate:
            _dp_name  = _safe_text(_diff_candidate["headline"])
            _dp_score = float(_diff_candidate.get("differential_score", 0.0))
            _dp_run   = _safe_text((_diff_df.iloc[0].get("fixture_run_label","?") if not _diff_df.empty else "?"))
            _dp_own   = float(_diff_candidate.get("ownership_pct", 0.0))
            _diff_pill_html = (
                "<div style='margin-top:0.5rem;padding-top:0.5rem;"
                "border-top:1px solid #45474C;display:flex;align-items:center;"
                "gap:0.5rem;flex-wrap:wrap;'>"
                "<span style='font-family:Space Mono;font-size:0.6rem;"
                "color:#C4BCB5;letter-spacing:0.1em;'>DIFFERENTIAL</span>"
                f"<span style='font-weight:700;font-size:0.82rem;'>{_dp_name}</span>"
                f"<span style='font-size:0.75rem;color:#C4BCB5;'>Run: {_dp_run}</span>"
                f"<span style='font-size:0.75rem;color:#BFB48F;'>Score {_dp_score:.2f}</span>"
                f"<span style='font-size:0.75rem;color:#C4BCB5;'>{_dp_own:.1f}% owned</span>"
                "</div>"
            )

        st.markdown(
            "<div class='fpl-card' style='padding:0.7rem 0.95rem;margin-bottom:0.6rem;'>"
            # Line 1 — primary
            "<div style='display:flex;align-items:baseline;gap:0.55rem;flex-wrap:wrap;'>"
            "<span style='font-family:Space Mono;font-size:0.62rem;color:#C4BCB5;"
            "letter-spacing:0.1em;'>RECOMMENDED</span>"
            f"<span style='font-weight:800;font-size:0.95rem;'>{_prim_label}: {_prim_hl}</span>"
            f"<span style='font-size:0.78rem;color:{_prim_gain_col};font-weight:700;'>"
            f"({_prim_gain:+.1f} pts)</span>"
            f"<span style='font-size:0.75rem;color:#C4BCB5;'>£{_prim_cost:+.1f}M</span>"
            f"<span style='font-size:0.72rem;color:#C4BCB5;'>Conf {_prim_conf:.0f}%</span>"
            "</div>"
            # Line 2 — alternatives
            + (
                "<div style='margin-top:0.28rem;font-size:0.78rem;color:#C4BCB5;'>"
                f"<span style='font-family:Space Mono;font-size:0.6rem;"
                "letter-spacing:0.1em;'>ALT &nbsp;</span>"
                f"{_alt_html}</div>"
                if _alt_html else ""
            )
            # Differential pill
            + _diff_pill_html
            + "</div>",
            unsafe_allow_html=True,
        )

    if feature_capabilities.get("horizon_plan") and get_horizon_transfer_plan:
        with st.expander("Multi-GW Horizon Plan (2-GW lookahead)", expanded=False):
            try:
                horizon_plans = get_horizon_transfer_plan(my_team, others, enriched_df, bank_balance)
                if not horizon_plans:
                    st.info("No viable 2-GW transfer sequence found.")
                else:
                    for i, plan in enumerate(horizon_plans[:3], 1):
                        st.markdown(f"**Plan {i}** · Total EV: `{float(plan.get('total_horizon_ev', 0.0)):+.2f}`")
                        h1, h2 = st.columns(2)
                        with h1:
                            st.markdown(
                                f"GW+1: `{plan.get('w1_out','?')}` → `{plan.get('w1_in','?')}`  "
                                f"xPts `{float(plan.get('w1_xpts_gain',0.0)):+.2f}` · EV `{float(plan.get('w1_total_ev',0.0)):+.2f}` · "
                                f"Run `{plan.get('w1_run','?')}` · Cost `£{float(plan.get('w1_cost',0.0)):+.1f}M`"
                            )
                        with h2:
                            if str(plan.get("w2_in", "—")) not in {"—", "-", ""}:
                                st.markdown(
                                    f"GW+2: `{plan.get('w2_out','?')}` → `{plan.get('w2_in','?')}`  "
                                    f"xPts `{float(plan.get('w2_xpts_gain',0.0)):+.2f}` · EV `{float(plan.get('w2_total_ev',0.0)):+.2f}` · "
                                    f"Run `{plan.get('w2_run','?')}`"
                                )
                            else:
                                st.markdown("GW+2: No strong follow-up move.")
                        if i < min(3, len(horizon_plans)):
                            st.divider()
            except Exception as e:
                st.caption(f"Horizon plan unavailable: {e}")

    with st.expander("Valid 2-Transfer Combinations (Rules + Budget)", expanded=False):
        try:
            valid_twos = get_valid_double_transfers(
                my_team,
                others,
                float(bank_balance),
                top_n=5,
                precomputed_ilp=ilp_2,
            )
            if not valid_twos:
                st.info("No valid 2-transfer combinations found.")
            else:
                vdf = pd.DataFrame(valid_twos)
                keep = [
                    "transfer_1_out", "transfer_1_in", "transfer_1_ev",
                    "transfer_2_out", "transfer_2_in", "transfer_2_ev",
                    "total_ev", "total_next_gw_gain", "total_combined_gain", "total_cost",
                ]
                keep = [c for c in keep if c in vdf.columns]
                st.dataframe(
                    vdf[keep].rename(columns={
                        "transfer_1_out": "T1 OUT",
                        "transfer_1_in": "T1 IN",
                        "transfer_1_ev": "T1 EV",
                        "transfer_2_out": "T2 OUT",
                        "transfer_2_in": "T2 IN",
                        "transfer_2_ev": "T2 EV",
                        "total_ev": "Total EV",
                        "total_next_gw_gain": "Next GW Gain",
                        "total_combined_gain": "5GW Gain",
                        "total_cost": "Cost £M",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as e:
            st.caption(f"Valid combo table unavailable: {e}")

    tab1, tab2, tab3, tab4 = st.tabs(["1 Transfer", "2 Transfers", "Take a Hit", "Double Hit (-8)"])

    def _transfer_card_meta(tr: dict, fallback_next: float = 0.0) -> tuple[float, float, float, float, str, list[str]]:
        next_gain = float(tr.get("next_gain", fallback_next) or 0.0)
        horizon_gain = float(tr.get("gain", 0.0) or 0.0)
        total_ev = float(tr.get("total_ev", horizon_gain) or horizon_gain)
        cost = float(tr.get("cost_diff", 0.0) or 0.0)
        is_blank = bool(tr.get("is_blank", False))
        has_dgw = float(tr.get("double_gws", 0) or 0) > 0
        confidence = float(np.clip(58 + 10 * total_ev + 5 * next_gain - (12 if is_blank else 0) - (4 if cost > 1.5 else 0), 35, 92))
        if is_blank:
            risk_level = "High"
        elif cost > 2.0:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        risks = []
        if is_blank:
            risks.append("Upcoming blank risk")
        if cost > 1.5:
            risks.append("Higher budget commitment")
        if not has_dgw:
            risks.append("No DGW upside")
        if not risks:
            risks.append("Role/minutes variance")
        return next_gain, horizon_gain, total_ev, confidence, risk_level, risks

    with tab1:
        if ilp_1.get("transfers"):
            t = ilp_1["transfers"][0]
            blank_note = " | BLANK" if t.get("is_blank") else ""
            dgw_note   = " | DGW"   if t.get("double_gws",0) > 0 else ""
            pchg_tag = _price_tag(float(t.get("price_change", 0) or 0))
            urg_tag = " | Urgent" if float(t.get("urgency_score", 0) or 0) >= 2.0 else ""
            out_team = transfer_team_name_map.get(t['out_name'], "")
            in_team = transfer_team_name_map.get(t['in_name'], "")
            col_a, col_b, col_c = st.columns([2, 1, 2])
            with col_a:
                st.markdown(f"""
                <div class='transfer-card'>
                    <div class='kpi-label'>TRANSFER OUT</div>
                    {player_identity_html(
                        t['out_name'],
                        out_team,
                        transfer_face_map.get(t['out_name'], ''),
                        transfer_badge_map.get(t['out_name'], ''),
                        f"{t['position']} | EV: +{float(t.get('total_ev', t['gain'])):.2f}",
                        'player-face-sm'
                    )}
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div style='text-align:center; padding-top:1.5rem;'>
                    <div class='transfer-gain'>+{float(t.get('total_ev', t['gain'])):.2f}</div>
                    <div style='font-size:0.65rem; color:var(--accent);
                                font-family:Space Mono; letter-spacing:0.1em;'>
                        TOTAL EV
                    </div>
                    <div style='font-size:0.8rem; color:var(--muted); margin-top:0.4rem;'>
                        Cost: {t['cost_diff']:+.1f}M{pchg_tag}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_c:
                st.markdown(f"""
                <div class='transfer-card' style='border-color:color-mix(in srgb, var(--primary) 28%, transparent);'>
                    <div class='kpi-label'>TRANSFER IN{urg_tag}</div>
                    {player_identity_html(
                        t['in_name'],
                        in_team,
                        transfer_face_map.get(t['in_name'], ''),
                        transfer_badge_map.get(t['in_name'], ''),
                        f"{t['position']} | {t['fixture_run']}{blank_note}{dgw_note}",
                        'player-face-sm'
                    )}
                </div>
                """, unsafe_allow_html=True)

            t_next, t_horizon, t_ev, t_conf, t_risk, t_risks = _transfer_card_meta(
                t,
                fallback_next=float(ilp_1.get("total_next_gain", 0.0)),
            )
            render_recommendation_card(
                headline=f"Transfer detail: {t['out_name']} → {t['in_name']}{pchg_tag}",
                impact_now=t_next,
                impact_horizon=t_horizon,
                confidence=t_conf,
                risk_notes=t_risks,
                supporting_points=[
                    f"Total EV: {t_ev:+.2f}",
                    f"Cost delta: £{float(t.get('cost_diff', 0.0)):+.1f}M",
                    f"Fixture run: {t.get('fixture_run', '?')}",
                    f"Risk level: {t_risk}",
                ],
            )

            render_stat_cards(
                [
                    {"label": "Total EV", "value": f"+{float(ilp_1.get('total_ev', ilp_1.get('total_gain', 0.0))):.2f}", "delta": "xPts + price effects", "tone": "positive"},
                    {"label": "5GW Combined Gain", "value": f"+{ilp_1['total_gain']:.2f}", "delta": "Horizon impact", "tone": "positive"},
                    {"label": "Next GW Gain", "value": f"+{ilp_1['total_next_gain']:.2f}", "delta": "Immediate impact", "tone": "neutral"},
                    {"label": "Net Cost", "value": f"£{ilp_1['total_cost']:+.1f}M", "delta": "Budget delta", "tone": "warning" if ilp_1["total_cost"] > 0 else "positive"},
                    {"label": "Solver", "value": str(ilp_1.get("solver_status", "ILP")), "delta": "Optimization status", "tone": "neutral"},
                ],
                compact=False,
            )

            t = ilp_1["transfers"][0]
            before_names = set(xi_result["starting_xi"]["player_name"].tolist()) if xi_result else set()
            after_names = set(before_names)
            if t["out_name"] in after_names:
                after_names.remove(t["out_name"])
                after_names.add(t["in_name"])
            in_count = len(after_names - before_names)
            out_count = len(before_names - after_names)

            render_section_header("Before vs After XI")
            p1, p2, p3 = st.columns([2, 1, 2])
            p1.markdown("**Before XI**")
            p1.markdown(_xi_chip_grid_html(before_names), unsafe_allow_html=True)
            p2.markdown(
                f"""
                <div class='kpi-block'>
                    <div class='kpi-label'>Changes</div>
                    <div class='kpi-value' style='font-size:1.2rem;'>{in_count} in / {out_count} out</div>
                    <div class='kpi-delta'>Projected XI shift</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            p3.markdown("**After XI**")
            p3.markdown(_xi_chip_grid_html(after_names), unsafe_allow_html=True)
        else:
            st.info("No beneficial 1-transfer found within budget.")

    with tab2:
        if ilp_2.get("transfers") and len(ilp_2["transfers"]) == 2:
            for i, t in enumerate(ilp_2["transfers"], 1):
                blank_note = " | BLANK" if t.get("is_blank") else ""
                dgw_note   = " | DGW"   if t.get("double_gws",0) > 0 else ""
                pchg_tag = _price_tag(float(t.get("price_change", 0) or 0))
                out_team = transfer_team_name_map.get(t['out_name'], "")
                in_team = transfer_team_name_map.get(t['in_name'], "")
                col_a, col_b, col_c = st.columns([2, 1, 2])
                with col_a:
                    st.markdown(f"""
                    <div class='transfer-card'>
                        <div class='kpi-label'>T{i} - OUT</div>
                        {player_identity_html(
                            t['out_name'],
                            out_team,
                            transfer_face_map.get(t['out_name'], ''),
                            transfer_badge_map.get(t['out_name'], ''),
                            t['position'],
                            'player-face-sm'
                        )}
                    </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""
                    <div style='text-align:center; padding-top:1rem;'>
                        <div class='transfer-gain'>+{float(t.get('total_ev', t['gain'])):.2f}</div>
                        <div style='font-size:0.65rem; color:var(--accent); font-family:Space Mono;'>
                        EV</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_c:
                    st.markdown(f"""
                    <div class='transfer-card' style='border-color:color-mix(in srgb, var(--primary) 28%, transparent);'>
                        <div class='kpi-label'>T{i} - IN{pchg_tag}</div>
                        {player_identity_html(
                            t['in_name'],
                            in_team,
                            transfer_face_map.get(t['in_name'], ''),
                            transfer_badge_map.get(t['in_name'], ''),
                            f"{t['fixture_run']}{blank_note}{dgw_note}",
                            'player-face-sm'
                        )}
                    </div>
                    """, unsafe_allow_html=True)

                t2_next, t2_horizon, t2_ev, t2_conf, t2_risk, t2_risks = _transfer_card_meta(
                    t,
                    fallback_next=float(ilp_2.get("total_next_gain", 0.0)) / max(len(ilp_2.get("transfers", [])), 1),
                )
                render_recommendation_card(
                    headline=f"T{i} detail: {t['out_name']} → {t['in_name']}{pchg_tag}",
                    impact_now=t2_next,
                    impact_horizon=t2_horizon,
                    confidence=t2_conf,
                    risk_notes=t2_risks,
                    supporting_points=[
                        f"Total EV: {t2_ev:+.2f}",
                        f"Cost delta: £{float(t.get('cost_diff', 0.0)):+.1f}M",
                        f"Fixture run: {t.get('fixture_run', '?')}",
                        f"Risk level: {t2_risk}",
                    ],
                )

            render_stat_cards(
                [
                    {"label": "Combined EV", "value": f"+{float(ilp_2.get('total_ev', ilp_2.get('total_gain', 0.0))):.2f}", "delta": "Both transfers EV", "tone": "positive"},
                    {"label": "Total 5GW Gain", "value": f"+{ilp_2['total_gain']:.2f}", "delta": "Combined horizon impact", "tone": "positive"},
                    {"label": "Total Next GW", "value": f"+{ilp_2['total_next_gain']:.2f}", "delta": "Immediate impact", "tone": "neutral"},
                    {"label": "Total Cost", "value": f"£{ilp_2['total_cost']:+.1f}M", "delta": "Budget delta", "tone": "warning" if ilp_2["total_cost"] > 0 else "positive"},
                ],
                compact=False,
            )

            before_names_2ft = set(xi_result["starting_xi"]["player_name"].tolist()) if xi_result else set()
            after_names_2ft = set(before_names_2ft)
            for tr in ilp_2.get("transfers", []):
                out_name = str(tr.get("out_name", ""))
                in_name = str(tr.get("in_name", ""))
                if out_name in after_names_2ft and in_name:
                    after_names_2ft.remove(out_name)
                    after_names_2ft.add(in_name)
            in_count_2ft = len(after_names_2ft - before_names_2ft)
            out_count_2ft = len(before_names_2ft - after_names_2ft)

            render_section_header("Before vs After XI (2 Transfers)")
            p1_2, p2_2, p3_2 = st.columns([2, 1, 2])
            p1_2.markdown("**Before XI**")
            p1_2.markdown(_xi_chip_grid_html(before_names_2ft), unsafe_allow_html=True)
            p2_2.markdown(
                f"""
                <div class='kpi-block'>
                    <div class='kpi-label'>Changes</div>
                    <div class='kpi-value' style='font-size:1.2rem;'>{in_count_2ft} in / {out_count_2ft} out</div>
                    <div class='kpi-delta'>Projected XI shift</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            p3_2.markdown("**After XI**")
            p3_2.markdown(_xi_chip_grid_html(after_names_2ft), unsafe_allow_html=True)
        else:
            st.info("No beneficial 2-transfer combination found within budget.")

    with tab3:
        if not hit_transfers:
            if transfers_made == 0:
                st.success("You still have a free transfer - no hit needed this week.")
            else:
                st.info("No transfers clear the -4 hit break-even threshold this week.")
        else:
            st.markdown("Transfers worth taking even with a **-4 point hit:**")
            for h in hit_transfers:
                h_next = float(h.get("predicted_gain_next", h.get("combined_gain", 0.0)))
                h_horizon = float(h.get("combined_gain", 0.0))
                h_conf = float(np.clip(54 + 8 * h_horizon - (10 if float(h.get("net_value", 0.0)) < 1 else 0), 30, 88))
                h_risks = ["Hit cost can erase upside", "Minutes/rotation risk"]
                if float(h.get("net_value", 0.0)) < 1:
                    h_risks.insert(0, "Low post-hit margin")
                c1, c2 = st.columns([3, 3])
                c1.markdown(f"**OUT** {h['replace']}")
                c2.markdown(f"**IN** {h['player_in']} · {h['fixture_run']}{_price_tag(float(h.get('price_change', 0) or 0))}")
                render_recommendation_card(
                    headline=f"Hit detail: {h['replace']} → {h['player_in']}{_price_tag(float(h.get('price_change', 0) or 0))}",
                    impact_now=h_next,
                    impact_horizon=h_horizon,
                    confidence=h_conf,
                    risk_notes=h_risks,
                    supporting_points=[
                        f"Post-hit net value: {float(h.get('net_value', 0.0)):+.1f}",
                        f"Fixture run: {h.get('fixture_run', '?')}",
                        "Risk level: Medium" if float(h.get("net_value", 0.0)) >= 1 else "Risk level: High",
                    ],
                )
                render_stat_cards(
                    [
                        {"label": "Gain", "value": f"+{h['combined_gain']:.1f}", "delta": "Pre-hit projection", "tone": "positive"},
                        {"label": "Net", "value": f"+{h['net_value']:.1f}", "delta": "After -4 hit", "tone": "warning" if h["net_value"] < 1 else "positive"},
                    ],
                    compact=False,
                )

    with tab4:
        if transfers_made == 0:
            st.success("You still have a free transfer. Double-hit (-8) analysis is not applicable.")
        elif not (feature_capabilities.get("double_hit") and get_double_hit_analysis):
            st.info("Double-hit analysis is unavailable in the current backend.")
        else:
            try:
                double_hits = get_double_hit_analysis(my_team, others, bank_balance, transfers_made)
                if not double_hits:
                    st.info("No double-transfer combo justifies a -8 hit this week.")
                else:
                    st.markdown("Combos worth considering despite a **-8** hit:")
                    for i, dh in enumerate(double_hits, 1):
                        conf_dh = float(np.clip(60 + 8 * float(dh.get("net_value", 0.0)), 30, 88))
                        render_recommendation_card(
                            headline=f"Option {i}: {dh.get('t1_out','?')} → {dh.get('t1_in','?')} + {dh.get('t2_out','?')} → {dh.get('t2_in','?')}",
                            impact_now=float(dh.get("total_xpts_gain", 0.0)),
                            impact_horizon=float(dh.get("total_xpts_gain", 0.0)),
                            confidence=conf_dh,
                            risk_notes=["-8 hit cost", "Requires both transfers to pay off"],
                            supporting_points=[
                                f"Combined xPts gain: {float(dh.get('total_xpts_gain', 0.0)):+.2f}",
                                f"Net after hit: {float(dh.get('net_value', 0.0)):+.2f}",
                                f"Total cost: £{float(dh.get('total_cost', 0.0)):+.1f}M",
                            ],
                        )
            except Exception as e:
                st.caption(f"Double hit analysis unavailable: {e}")


    # ── Inline AI bar ─────────────────────────────────────────────────────────
    st.divider()
    _ai_col_q, _ai_col_btn = st.columns([4, 1])
    with _ai_col_q:
        _inline_question = st.text_input(
            "Ask AI",
            placeholder=f"Ask about your squad for GW{current_gw+1}...",
            key=f"inline_ai_{page}",
            label_visibility="collapsed",
        )
    with _ai_col_btn:
        _ask_pressed = st.button(
            "Ask AI ◇",
            use_container_width=True,
            key=f"ask_ai_{page}",
        )
    if _ask_pressed and _inline_question:
        st.session_state.setdefault("analyst_messages", []).append(
            {"role": "user", "content": _inline_question}
        )
        st.session_state["_pending_nav"] = "AI Analyst"
        st.rerun()


elif page == "Scout":
    render_page_hero(
        "Scout",
        "Search the full player pool by value, ceiling, and safety with side-by-side comparisons.",
        [
            f"Pool {len(enriched_df)}",
            f"Owned {len(data['my_player_ids'])}",
            "Scatter + Differentials + Compare",
        ],
    )

    render_section_header("Scout")

    if "px_pos_filter" not in st.session_state:
        st.session_state["px_pos_filter"] = []
    if "px_price_range" not in st.session_state:
        st.session_state["px_price_range"] = (3.5, 15.0)
    if "px_min_pred" not in st.session_state:
        st.session_state["px_min_pred"] = 0.0
    if "px_search" not in st.session_state:
        st.session_state["px_search"] = ""
    if "px_mode" not in st.session_state:
        st.session_state["px_mode"] = "Value"
    if "px_pos_filter_widget" not in st.session_state:
        st.session_state["px_pos_filter_widget"] = st.session_state["px_pos_filter"]
    if "px_price_range_widget" not in st.session_state:
        st.session_state["px_price_range_widget"] = st.session_state["px_price_range"]
    if "px_min_pred_widget" not in st.session_state:
        st.session_state["px_min_pred_widget"] = st.session_state["px_min_pred"]
    if "px_search_widget" not in st.session_state:
        st.session_state["px_search_widget"] = st.session_state["px_search"]

    render_section_header("Explorer Controls")
    r_mode, r_reset = st.columns([1, 1])
    with r_mode:
        mode = st.radio("Objective mode", ["Value", "Ceiling", "Safety"], horizontal=True, key="px_mode")
    with r_reset:
        if st.button("Reset filters", use_container_width=True):
            st.session_state["px_pos_filter"] = []
            st.session_state["px_price_range"] = (3.5, 15.0)
            st.session_state["px_min_pred"] = 0.0
            st.session_state["px_search"] = ""
            st.session_state["px_pos_filter_widget"] = []
            st.session_state["px_price_range_widget"] = (3.5, 15.0)
            st.session_state["px_min_pred_widget"] = 0.0
            st.session_state["px_search_widget"] = ""
            st.rerun()

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        pos_filter = st.multiselect("Position",
            ["Goalkeeper","Defender","Midfielder","Forward"],
            key="px_pos_filter_widget")
    with f2:
        price_range = st.slider("Price Range (£M)", 3.5, 15.0, step=0.5, key="px_price_range_widget")
    with f3:
        min_pred = st.slider("Min Predicted Pts", 0.0, 15.0, step=0.5, key="px_min_pred_widget")
    with f4:
        search = st.text_input("Search player name", key="px_search_widget")

    # Persist latest filter values across mode/layout switches.
    st.session_state["px_pos_filter"] = pos_filter
    st.session_state["px_price_range"] = price_range
    st.session_state["px_min_pred"] = min_pred
    st.session_state["px_search"] = search

    # Filter data
    pool = enriched_df.copy()
    pool["ownership"] = pool["player_id"].map(ownership_map).fillna(0)
    pool["news"] = pool["player_id"].map(news_map).fillna("")

    if pos_filter:
        pool = pool[pool["position"].isin(pos_filter)]
    pool = pool[
        (pool["price"] >= price_range[0]) &
        (pool["price"] <= price_range[1]) &
        (pool["predicted_pts"] >= min_pred)
    ]
    if search:
        pool = pool[pool["player_name"].str.contains(search, case=False, na=False)]

    st.caption(f"{len(pool)} players shown")
    if not pool.empty:
        points_col_pool = pick_points_col(pool)
        pool_stats = [
            {"label": "Shown", "value": str(len(pool)), "delta": "Filtered players", "tone": "neutral"},
            {"label": "Avg xPts" if points_col_pool == "expected_pts" else "Avg Pred", "value": f"{float(pd.to_numeric(pool[points_col_pool], errors='coerce').fillna(0).mean()):.2f}", "delta": "Current filtered pool", "tone": "positive"},
            {"label": "Median Price", "value": f"£{float(pool['price'].median()):.1f}", "delta": "Current filtered pool", "tone": "neutral"},
            {"label": "Top xPts" if points_col_pool == "expected_pts" else "Top Pred", "value": f"{float(pd.to_numeric(pool[points_col_pool], errors='coerce').fillna(0).max()):.2f}", "delta": "Highest in filter", "tone": "positive"},
        ]
        render_stat_cards(pool_stats)

    tab_scatter, tab_bar, tab_compare, tab_table = st.tabs([
        "Pts vs Price", "Top Differentials", "Player Comparison", "Full Table"
    ])

    with tab_scatter:
        plot_df = pool.copy()
        if plot_df.empty:
            st.info("No players match the current filters.")
        else:
            plot_df["xpts_val"] = plot_df.apply(lambda r: _xpts(r), axis=1)
            plot_df["value_index"] = (
                plot_df["xpts_val"] / plot_df["price"].replace(0, np.nan)
            ).fillna(0.0)
            if "p_plays_full" in plot_df.columns:
                safety = plot_df["p_plays_full"].fillna(1.0).astype(float)
            elif "chance_of_playing" in plot_df.columns:
                safety = plot_df["chance_of_playing"].fillna(100).astype(float) / 100.0
            else:
                safety = pd.Series([1.0] * len(plot_df), index=plot_df.index, dtype=float)
            plot_df["objective_score"] = (
                plot_df["value_index"] if mode == "Value"
                else plot_df["xpts_val"] if mode == "Ceiling"
                else plot_df["xpts_val"] * safety
            )
            x_med = float(plot_df["price"].median())
            y_med = float(plot_df["xpts_val"].median())
            top_value = plot_df.nlargest(8, "objective_score")

            fig = px.scatter(
                plot_df,
                x="price", y="xpts_val",
                color="position",
                size="value_index",
                size_max=26,
                opacity=0.78,
                hover_name="player_name",
                hover_data={
                    "team_name": True,
                    "price": ":.1f",
                    "xpts_val": ":.2f",
                    "predicted_pts": ":.2f",
                    "combined_score": ":.2f",
                    "fixture_run_label": True,
                    "ownership": ":.1f",
                    "value_index": ":.3f",
                },
                labels={"price": "Price (£M)", "xpts_val": "xPts"},
                color_discrete_map=POSITION_COLOR_MAP,
            )

            fig.add_hline(
                y=y_med, line_dash="dot", line_color=PLOTLY_ACCENT,
                annotation_text="Median xPts", annotation_position="top left",
            )
            fig.add_vline(
                x=x_med, line_dash="dot", line_color=PLOTLY_ACCENT,
                annotation_text="Median Price", annotation_position="top right",
            )

            fig.add_trace(go.Scatter(
                x=top_value["price"],
                y=top_value["xpts_val"],
                mode="markers+text",
                text=top_value["player_name"].str.split().str[-1],
                textposition="top center",
                textfont=dict(size=10, color=PLOTLY_TEXT),
                marker=dict(
                    size=12,
                    symbol="diamond",
                    color=_hex_to_rgba(PLOTLY_PRIMARY, 0.15),
                    line=dict(color=PLOTLY_PRIMARY, width=1.8),
                ),
                name="Top Value",
                hoverinfo="skip",
            ))

            squad_pool = plot_df[plot_df["player_id"].isin(data["my_player_ids"])]
            if not squad_pool.empty:
                fig.add_trace(go.Scatter(
                    x=squad_pool["price"],
                    y=squad_pool["xpts_val"],
                    mode="markers",
                    marker=dict(size=16, color=PLOTLY_SURFACE, symbol="star",
                                line=dict(color=PLOTLY_PRIMARY, width=2)),
                    name="Your Squad",
                    hovertext=squad_pool["player_name"],
                ))

            fig.update_layout(
                **PLOTLY_THEME,
                height=470,
                margin=dict(l=10, r=10, t=22, b=32),
                legend_title_text="Role",
            )
            st.plotly_chart(
                fig, use_container_width=True,
                config={"displayModeBar": "hover", "responsive": True, "scrollZoom": True},
            )
            st.caption(f"Objective mode: {mode}. Y-axis uses {'expected_pts' if 'expected_pts' in plot_df.columns else 'predicted_pts'}. Diamonds mark top picks for the selected objective.")

            render_section_header(f"Top Picks by {mode}")
            top_strip = top_value.sort_values("objective_score", ascending=False).head(6).copy()
            strip_cols = st.columns(3)
            for i, (_, row) in enumerate(top_strip.iterrows(), 1):
                with strip_cols[(i - 1) % len(strip_cols)]:
                    objective_label = (
                        f"{float(row.get('value_index', 0.0)):.3f} value/£"
                        if mode == "Value"
                        else f"{float(row.get('xpts_val', _xpts(row))):.2f} xPts"
                        if mode == "Ceiling"
                        else f"{float(row.get('p_plays_full', row.get('reliability', row.get('chance_of_playing', 100) / 100))):.0%} reliability"
                    )
                    rank_tag = f"#{i}"
                    subtitle = f"{row['position']} | £{float(row['price']):.1f}M | {float(row.get('xpts_val', _xpts(row))):.2f} xPts{_price_tag(float(row.get('predicted_price_change', 0) or 0))}"
                    st.markdown(
                        f"""
                        <div class='fpl-card'>
                            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;'>
                                <span class='xi-role'>{rank_tag}</span>
                                <span class='xi-role'>{_safe_text(objective_label)}</span>
                            </div>
                            {player_identity_html(
                                row['player_name'],
                                row.get('team_name', ''),
                                row.get('player_face', ''),
                                row.get('team_badge', ''),
                                subtitle,
                                'player-face-sm'
                            )}
                            <div style='font-size:0.72rem; color:var(--muted); margin-top:0.35rem;'>
                                Run: {_safe_text(row.get('fixture_run_label', '?'))}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            shortlist_cols = [
                "player_face", "team_badge", "player_name", "team_name", "position",
                "price", "xpts_val", "predicted_pts", "value_index", "objective_score", "fixture_run_label"
            ]
            if "predicted_price_change" in top_value.columns:
                shortlist_cols.append("predicted_price_change")
            shortlist = (
                top_value[shortlist_cols]
                .rename(columns={
                    "player_face": "Face", "team_badge": "Badge", "player_name": "Player",
                    "team_name": "Team", "position": "Pos", "price": "£",
                    "xpts_val": "xPts", "predicted_pts": "Pred", "value_index": "Value/£",
                    "objective_score": "Objective", "fixture_run_label": "Run",
                    "predicted_price_change": "Price Δ",
                })
                .sort_values("Objective", ascending=False)
            )
            with st.expander("View shortlist table"):
                st.dataframe(
                    shortlist, use_container_width=True, hide_index=True,
                    column_config={
                        "Face": st.column_config.ImageColumn("Face", width="small"),
                        "Badge": st.column_config.ImageColumn("Badge", width="small"),
                        "£": st.column_config.NumberColumn(format="£%.1f"),
                        "xPts": st.column_config.NumberColumn(format="%.2f"),
                        "Pred": st.column_config.NumberColumn(format="%.2f"),
                        "Value/£": st.column_config.NumberColumn(format="%.3f"),
                        "Objective": st.column_config.NumberColumn(format="%.3f"),
                        "Price Δ": st.column_config.NumberColumn(format="%+.2f"),
                    },
                )

    with tab_bar:
        # Differentials - low ownership, high predicted pts
        diffs = cached_differential_picks(others, bootstrap, top_n=15)
        if not diffs.empty:
            fig = px.bar(
                diffs,
                x="differential_score", y="player_name",
                orientation="h",
                color="position",
                hover_data=["team_name", "price", "predicted_pts",
                            "ownership_pct", "fixture_run_label"],
                color_discrete_map=POSITION_COLOR_MAP,
                labels={"differential_score": "Differential Score",
                        "player_name": ""},
            )
            fig.update_layout(**PLOTLY_THEME, height=450,
                      margin=dict(l=10, r=10, t=30, b=30))
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": "hover", "responsive": True, "scrollZoom": True})
            st.caption("Differential score = combined score x low ownership bonus. "
                       f"Only players owned by <15% shown.")

    with tab_compare:
        st.markdown("Select two players to compare head-to-head:")
        all_names = sorted(enriched_df["player_name"].tolist())
        col1, col2 = st.columns(2)
        with col1:
            p1_name = st.selectbox("Player A", all_names, index=0, key="p1")
        with col2:
            p2_name = st.selectbox("Player B", all_names,
                                   index=min(1, len(all_names)-1), key="p2")

        p1 = enriched_df[enriched_df["player_name"] == p1_name]
        p2 = enriched_df[enriched_df["player_name"] == p2_name]

        if not p1.empty and not p2.empty:
            p1 = p1.iloc[0]
            p2 = p2.iloc[0]

            id_col1, id_col2 = st.columns(2)
            with id_col1:
                p1_subtitle = f"£{p1['price']:.1f}M | {_xpts(p1):.2f} xPts"
                st.markdown(
                    f"<div class='fpl-card'>{player_identity_html(p1['player_name'], p1['team_name'], p1.get('player_face',''), p1.get('team_badge',''), p1_subtitle)}</div>",
                    unsafe_allow_html=True,
                )
            with id_col2:
                p2_subtitle = f"£{p2['price']:.1f}M | {_xpts(p2):.2f} xPts"
                st.markdown(
                    f"<div class='fpl-card'>{player_identity_html(p2['player_name'], p2['team_name'], p2.get('player_face',''), p2.get('team_badge',''), p2_subtitle)}</div>",
                    unsafe_allow_html=True,
                )

            metrics = [
                ("xPts",           _xpts(p1),                 _xpts(p2)),
                ("Predicted Pts",  p1["predicted_pts"],       p2["predicted_pts"]),
                ("Combined Score", p1["combined_score"],      p2["combined_score"]),
                ("Price £M",       p1["price"],               p2["price"]),
                ("Avg Difficulty", p1.get("avg_difficulty",3),p2.get("avg_difficulty",3)),
                ("Momentum",       p1.get("momentum_score",3),p2.get("momentum_score",3)),
                ("Value Score",    p1.get("value_score",0),   p2.get("value_score",0)),
                ("p_plays_full",   p1.get("p_plays_full",1),  p2.get("p_plays_full",1)),
                ("Blank GWs",      p1.get("blank_gws",0),     p2.get("blank_gws",0)),
                ("Double GWs",     p1.get("double_gws",0),    p2.get("double_gws",0)),
            ]

            labels  = [m[0] for m in metrics]
            vals_p1 = [float(m[1] or 0) for m in metrics]
            vals_p2 = [float(m[2] or 0) for m in metrics]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name=p1_name.split()[-1], x=labels, y=vals_p1,
                marker_color=PLOTLY_PRIMARY,
            ))
            fig.add_trace(go.Bar(
                name=p2_name.split()[-1], x=labels, y=vals_p2,
                marker_color=PLOTLY_ACCENT,
            ))
            fig.update_layout(**PLOTLY_THEME, barmode="group", height=380,
                              margin=dict(l=10, r=10, t=30, b=60))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": "hover", "responsive": True, "scrollZoom": True})

            # GW breakdown table
            gws = list(range(current_gw+1, current_gw+1+FIXTURE_LOOKAHEAD))
            gw_data = []
            for gw in gws:
                gw_data.append({
                    "GW": f"GW{gw}",
                    f"{p1_name.split()[-1]} Opp":
                        p1.get(f"gw{gw}_opponent", "?"),
                    f"{p1_name.split()[-1]} Diff":
                        p1.get(f"gw{gw}_difficulty", "?"),
                    f"{p2_name.split()[-1]} Opp":
                        p2.get(f"gw{gw}_opponent", "?"),
                    f"{p2_name.split()[-1]} Diff":
                        p2.get(f"gw{gw}_difficulty", "?"),
                })
            st.dataframe(pd.DataFrame(gw_data), use_container_width=True,
                         hide_index=True)

            def _player_window_diffs(player_row):
                diffs = []
                blanks = 0
                for gw in gws:
                    d = player_row.get(f"gw{gw}_difficulty", np.nan)
                    if pd.isna(d) or str(d).upper() in {"BLANK", "B"}:
                        blanks += 1
                        diffs.append(6.0)
                    else:
                        try:
                            diffs.append(float(d))
                        except Exception:
                            diffs.append(6.0)
                            blanks += 1
                return diffs, blanks

            p1_diffs, p1_blanks = _player_window_diffs(p1)
            p2_diffs, p2_blanks = _player_window_diffs(p2)
            p1_conf, p1_swing = _fixture_confidence_and_swing(p1_diffs, p1_blanks)
            p2_conf, p2_swing = _fixture_confidence_and_swing(p2_diffs, p2_blanks)
            p1_avg = float(np.mean(p1_diffs))
            p2_avg = float(np.mean(p2_diffs))
            xpts_delta = float(_xpts(p1) - _xpts(p2))
            diff_delta = float(p2_avg - p1_avg)  # positive means p1 has easier fixtures
            # Blend projection edge and fixture edge (lower difficulty is better).
            blended_score = 0.6 * xpts_delta + 0.4 * diff_delta
            verdict = p1_name if blended_score >= 0 else p2_name
            conf = float(np.clip(52 + abs(blended_score) * 18 + abs(p1_conf - p2_conf) * 0.2, 51, 93))
            reasons = [
                f"xPts delta: {xpts_delta:+.2f}",
                f"Avg difficulty edge (blank-aware): {diff_delta:+.2f}",
                f"Blended edge score: {blended_score:+.2f}",
                f"Swing {p1_name.split()[-1]}: {p1_swing:+.2f} | {p2_name.split()[-1]}: {p2_swing:+.2f}",
                f"Blank penalty included ({p1_blanks} vs {p2_blanks})",
            ]
            render_decision_banner(
                title="Comparison Verdict",
                primary_action=f"Preferred: {verdict}",
                confidence=conf,
                reasons=reasons,
                risk_level="Medium" if abs(blended_score) < 0.35 else "Low",
            )

    with tab_table:
        disp_cols = ["player_face","team_badge","player_name","team_name","position","price"]
        if "expected_pts" in pool.columns:
            disp_cols.append("expected_pts")
        disp_cols += ["predicted_pts","combined_score","value_score",
                     "fixture_run_label","blank_gws","double_gws"]
        if "predicted_price_change" in pool.columns:
            disp_cols.append("predicted_price_change")
        disp_pool = pool[disp_cols].rename(columns={
            "player_face":"Face","team_badge":"Badge",
            "player_name":"Player","team_name":"Team","position":"Pos",
            "price":"£","expected_pts":"xPts","predicted_pts":"Pred","combined_score":"5GW",
            "value_score":"Val","fixture_run_label":"Run",
            "blank_gws":"Blanks","double_gws":"DGWs","predicted_price_change":"Price Δ",
        }).sort_values("xPts" if "expected_pts" in pool.columns else "Pred", ascending=False)
        st.dataframe(
            disp_pool, use_container_width=True, hide_index=True,
            column_config={
                "Face": st.column_config.ImageColumn("Face", width="small"),
                "Badge": st.column_config.ImageColumn("Badge", width="small"),
            }
        )



elif page == "Season Tracker":
    render_page_hero(
        "Season Tracker",
        "Track squad value, market signals, and model performance across the season.",
        [
            f"Current GW {current_gw}",
            "Value · Market · Model",
        ],
    )

    # ── Shared data computation (used across all three tabs) ──────────────────
    value_data = track_squad_value(my_team, bootstrap, current_gw, team_data=team_data)
    history    = value_data.get("history", {})
    try:
        baseline_gw_num = int(value_data.get("baseline_gw", current_gw))
    except (TypeError, ValueError):
        baseline_gw_num = int(current_gw)
    try:
        current_value_num = float(value_data.get("current_value", 0.0))
    except (TypeError, ValueError):
        current_value_num = 0.0
    try:
        baseline_value_num = float(value_data.get("baseline_value", current_value_num))
    except (TypeError, ValueError):
        baseline_value_num = current_value_num
    try:
        total_change_num = float(value_data.get("total_change", current_value_num - baseline_value_num))
    except (TypeError, ValueError):
        total_change_num = current_value_num - baseline_value_num

    change_sign = "+" if total_change_num >= 0 else ""

    # Fixture window averages for period comparison
    gw_cols = sorted(
        [_c for _c in my_team.columns if _c.startswith("gw") and _c.endswith("_difficulty")],
        key=lambda _c: int(_c[2:].split("_")[0]),
    )
    window_vals = []
    for _c in gw_cols:
        _col = pd.to_numeric(my_team[_c], errors="coerce").fillna(6.0)
        window_vals.append(float(_col.mean()))
    if window_vals:
        last5    = window_vals[:5]
        prior5   = window_vals[5:10] if len(window_vals) >= 10 else window_vals[2:7]
        last5_avg  = float(np.mean(last5))  if last5  else np.nan
        prior5_avg = float(np.mean(prior5)) if prior5 else np.nan
    else:
        last5_avg, prior5_avg = np.nan, np.nan

    # Transfer history (shared between Season scorecard and Model tab)
    _transfer_file            = "transfer_history.json"
    transfer_hit_rate         = np.nan
    pred_calibration          = np.nan
    evaluated_transfer_count  = 0
    min_evaluated_transfers   = 5
    _t_hist_loaded            = []
    try:
        from pathlib import Path as _Path
        if _Path(_transfer_file).exists():
            with open(_transfer_file, encoding="utf-8") as _f:
                _t_hist_loaded = json.load(_f)
            _t_df_score = pd.DataFrame(_t_hist_loaded)
            if not _t_df_score.empty and "evaluated" in _t_df_score.columns:
                _eval_df = _t_df_score[_t_df_score["evaluated"] == True].copy()
                evaluated_transfer_count = int(len(_eval_df))
                if (
                    len(_eval_df) >= min_evaluated_transfers
                    and {"actual_gain", "predicted_gain"}.issubset(_eval_df.columns)
                ):
                    transfer_hit_rate = float((_eval_df["actual_gain"] > 0).mean() * 100.0)
                    _pred_err         = (_eval_df["actual_gain"] - _eval_df["predicted_gain"]).abs()
                    pred_calibration  = float(max(0.0, 100.0 - _pred_err.mean() * 25.0))
    except Exception:
        pass

    value_eff              = float((current_value_num - baseline_value_num) / max(1.0, float(current_gw) - float(baseline_gw_num) + 1.0))
    has_manager_score      = (not np.isnan(transfer_hit_rate)) and (not np.isnan(pred_calibration))
    value_component        = float(np.clip(50.0 + 40.0 * value_eff, 0.0, 100.0))
    if has_manager_score:
        hit_component          = float(transfer_hit_rate)
        calibration_component  = float(pred_calibration)
        manager_score          = float(np.clip(
            0.40 * hit_component + 0.35 * calibration_component + 0.25 * value_component,
            0.0, 100.0,
        ))
        manager_tone   = "positive" if manager_score >= 70 else "warning" if manager_score >= 50 else "danger"
        manager_value  = f"{manager_score:.0f}/100"
        manager_delta  = f"Hit {hit_component:.0f} | Cal {calibration_component:.0f} | Value {value_component:.0f}"
    else:
        manager_tone   = "neutral"
        manager_value  = "N/A"
        manager_delta  = f"Need >= {min_evaluated_transfers} evaluated transfers (have {evaluated_transfer_count})"

    # ── Three tabs ─────────────────────────────────────────────────────────────
    tab_season, tab_market, tab_model = st.tabs([
        "\U0001f4c8  Season",
        "\U0001f4b0  Market",
        "\U0001f9ea  Model",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — SEASON
    # ══════════════════════════════════════════════════════════════════════════
    with tab_season:
        # Value metrics
        _sc1, _sc2, _sc3 = st.columns(3)
        _sc1.metric("Current Squad Value", f"\u00a3{current_value_num:.1f}M")
        _sc2.metric("Baseline Value",       f"\u00a3{baseline_value_num:.1f}M", f"GW{baseline_gw_num}")
        _sc3.metric("Total Change",         f"{change_sign}{total_change_num:.1f}M")
        render_stat_cards([
            {"label": "Value Delta",         "value": f"{change_sign}{total_change_num:.1f}M", "delta": "Since baseline",         "tone": "positive" if total_change_num >= 0 else "danger"},
            {"label": "Value Growth/GW",     "value": f"{value_eff:+.2f}M",                    "delta": "Trend per gameweek",      "tone": "positive" if value_eff >= 0 else "danger"},
            {"label": "Sell Value",          "value": f"\u00a3{squad_sell_value:.1f}M",         "delta": "Sell-price estimate",     "tone": "neutral"},
            {"label": "Manager Scorecard",   "value": manager_value,                             "delta": manager_delta,            "tone": manager_tone},
        ])
        if has_manager_score:
            render_stat_cards([
                {"label": "Transfer Hit Rate",       "value": f"{transfer_hit_rate:.1f}%",    "delta": "Evaluated positive gains",  "tone": "positive" if transfer_hit_rate >= 55 else "warning"},
                {"label": "Prediction Calibration",  "value": f"{pred_calibration:.1f}%",     "delta": "Predicted vs realized",     "tone": "positive" if pred_calibration >= 65 else "warning"},
            ])
        else:
            st.info(
                f"Scorecard needs \u2265{min_evaluated_transfers} evaluated transfers "
                f"(currently {evaluated_transfer_count})."
            )

        if "expected_pts" in my_team.columns:
            _xpts_tot = float(pd.to_numeric(my_team["expected_pts"], errors="coerce").fillna(0.0).sum())
            _q10_tot  = float(pd.to_numeric(my_team.get("pts_low",  0.0), errors="coerce").fillna(0.0).sum()) if "pts_low"  in my_team.columns else np.nan
            _q90_tot  = float(pd.to_numeric(my_team.get("pts_high", 0.0), errors="coerce").fillna(0.0).sum()) if "pts_high" in my_team.columns else np.nan
            _xdelta   = (f"Q10 {_q10_tot:.1f} \u00b7 Q90 {_q90_tot:.1f}"
                         if not (np.isnan(_q10_tot) or np.isnan(_q90_tot))
                         else "Rotation-adjusted expected points")
            render_stat_cards([
                {"label": "Current Squad xPts", "value": f"{_xpts_tot:.1f}", "delta": _xdelta, "tone": "neutral"},
            ])

        # Squad value trend chart
        if len(history) > 1:
            _gws_hist  = sorted(history.keys(), key=int)
            _vals_hist = [float(history[g]) for g in _gws_hist]
            _fig_val   = go.Figure()
            _fig_val.add_trace(go.Scatter(
                x=[f"GW{g}" for g in _gws_hist],
                y=_vals_hist,
                mode="lines+markers",
                line=dict(color=PLOTLY_PRIMARY, width=2.5),
                marker=dict(size=8, color=PLOTLY_PRIMARY, line=dict(color=PLOTLY_SURFACE, width=1.5)),
                fill="tozeroy",
                fillcolor=_hex_to_rgba(PLOTLY_PRIMARY, 0.08),
                hovertemplate="<b>%{x}</b><br>Value: \u00a3%{y:.1f}M<extra></extra>",
                name="Squad Value",
            ))
            _fig_val.add_hline(
                y=baseline_value_num, line_dash="dash", line_color=PLOTLY_ACCENT,
                annotation_text=f"Baseline \u00a3{baseline_value_num:.1f}M",
                annotation_font=dict(color=PLOTLY_ACCENT),
            )
            _fig_val.update_layout(
                **PLOTLY_THEME, height=300,
                title=dict(text="Squad Value Over Time", font=dict(color=PLOTLY_ACCENT, size=13, family="Space Mono")),
                yaxis_title="Value (\u00a3M)",
                margin=dict(l=10, r=10, t=50, b=30),
            )
            st.plotly_chart(_fig_val, use_container_width=True, config={"displayModeBar": "hover", "responsive": True, "scrollZoom": True})
        else:
            st.info("Run the app across multiple gameweeks to see your value trend.")

        # Period comparison
        if len(history) >= 10 and not np.isnan(last5_avg) and not np.isnan(prior5_avg):
            st.divider()
            render_section_header("Period Comparison: Last 5 vs Prior 5")
            _cmp_df = pd.DataFrame([
                {"Period": "Last 5 GWs",  "Avg Fixture Difficulty": round(last5_avg, 2)},
                {"Period": "Prior 5 GWs", "Avg Fixture Difficulty": round(prior5_avg, 2)},
                {"Period": "Delta",        "Avg Fixture Difficulty": round(last5_avg - prior5_avg, 2)},
            ])
            render_insight_table(_cmp_df, row_density="compact")

            _worked, _hurt = [], []
            if last5_avg <= prior5_avg:
                _worked.append("Fixture run improved versus prior period.")
            else:
                _hurt.append("Fixture run hardened in the latest 5 GW window.")
            if not np.isnan(transfer_hit_rate):
                (_worked if transfer_hit_rate >= 50 else _hurt).append(f"Transfer hit rate at {transfer_hit_rate:.1f}%.")
            if value_eff >= 0:
                _worked.append(f"Squad value trend positive at {value_eff:+.2f}M/GW.")
            else:
                _hurt.append(f"Value trend negative at {value_eff:+.2f}M/GW.")

            render_section_header("What worked / what hurt")
            _wc, _hc = st.columns(2)
            with _wc:
                st.markdown("**Worked**")
                for _w in _worked or ["No clear positive pattern yet."]:
                    st.markdown(f"- {_w}")
            with _hc:
                st.markdown("**Hurt**")
                for _h in _hurt or ["No major drag detected yet."]:
                    st.markdown(f"- {_h}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — MARKET
    # ══════════════════════════════════════════════════════════════════════════
    with tab_market:
        # Price risers / fallers
        if "predicted_price_change" in enriched_df.columns:
            _movers_df = enriched_df.copy()
            _movers_df["predicted_price_change"] = pd.to_numeric(
                _movers_df["predicted_price_change"], errors="coerce"
            ).fillna(0.0)
            _top_risers  = _movers_df[_movers_df["predicted_price_change"] > 0].nlargest(8, "predicted_price_change").copy()
            _top_fallers = _movers_df[_movers_df["predicted_price_change"] < 0].nsmallest(8, "predicted_price_change").copy()

            if not _top_risers.empty or not _top_fallers.empty:
                _fig_mov = make_subplots(rows=1, cols=2,
                    subplot_titles=("Top Predicted Risers", "Top Predicted Fallers"),
                    horizontal_spacing=0.12)
                if not _top_risers.empty:
                    _fig_mov.add_trace(go.Bar(
                        x=_top_risers["predicted_price_change"], y=_top_risers["player_name"],
                        orientation="h", marker_color=PLOTLY_PRIMARY,
                        hovertemplate="<b>%{y}</b><br>%{x:+.2f}M<extra></extra>", name="Risers",
                    ), row=1, col=1)
                if not _top_fallers.empty:
                    _fig_mov.add_trace(go.Bar(
                        x=_top_fallers["predicted_price_change"], y=_top_fallers["player_name"],
                        orientation="h", marker_color=PLOTLY_DANGER,
                        hovertemplate="<b>%{y}</b><br>%{x:+.2f}M<extra></extra>", name="Fallers",
                    ), row=1, col=2)
                _fig_mov.update_layout(**PLOTLY_THEME, height=360, showlegend=False, margin=dict(l=10,r=10,t=46,b=24))
                _fig_mov.update_xaxes(title_text="Price change (\u00a3M)", row=1, col=1)
                _fig_mov.update_xaxes(title_text="Price change (\u00a3M)", row=1, col=2)
                _fig_mov.update_yaxes(autorange="reversed", row=1, col=1)
                _fig_mov.update_yaxes(autorange="reversed", row=1, col=2)
                st.plotly_chart(_fig_mov, use_container_width=True, config={"displayModeBar": "hover", "responsive": True, "scrollZoom": True})
                render_stat_cards([
                    {"label": "Strongest Rise",
                     "value": f"{_top_risers.iloc[0]['player_name']} {float(_top_risers.iloc[0]['predicted_price_change']):+.2f}M" if not _top_risers.empty else "None",
                     "delta": "Highest upside in market price trend", "tone": "positive"},
                    {"label": "Strongest Fall",
                     "value": f"{_top_fallers.iloc[0]['player_name']} {float(_top_fallers.iloc[0]['predicted_price_change']):+.2f}M" if not _top_fallers.empty else "None",
                     "delta": "Highest downside in market price trend", "tone": "danger" if not _top_fallers.empty else "neutral"},
                ])
            else:
                st.caption("No meaningful price movement signals detected this GW.")
        else:
            st.caption("Price movement model output not available (`predicted_price_change` missing).")

        st.divider()

        # Ownership scatter
        if "player_id" in enriched_df.columns:
            _own_df = enriched_df.copy()
            _own_df["ownership_pct"] = _own_df["player_id"].astype("Int64").map(ownership_map).fillna(0.0).astype(float)
            _own_df["xpts_val"]      = _own_df.apply(lambda r: _xpts(r), axis=1).astype(float)
            _own_df["in_squad"]      = _own_df["player_id"].astype("Int64").isin(set(data["my_player_ids"]))

            _pool_own_avg  = float(_own_df["ownership_pct"].mean()) if not _own_df.empty else 0.0
            _squad_slice   = _own_df[_own_df["in_squad"]]
            _squad_own_avg = float(_squad_slice["ownership_pct"].mean()) if not _squad_slice.empty else 0.0
            _squad_xpts_avg = float(_squad_slice["xpts_val"].mean()) if not _squad_slice.empty else 0.0
            _diff_count    = int(
                ((_squad_slice["ownership_pct"] < 15.0) & (_squad_slice["xpts_val"] >= _squad_slice["xpts_val"].median())).sum()
            ) if not _squad_slice.empty else 0

            render_stat_cards([
                {"label": "Avg Ownership (Squad)",  "value": f"{_squad_own_avg:.1f}%",  "delta": f"Pool avg {_pool_own_avg:.1f}%",        "tone": "positive" if _squad_own_avg < _pool_own_avg else "neutral"},
                {"label": "Squad Differentials",    "value": str(_diff_count),           "delta": "Owned <15% with above-median xPts",   "tone": "positive" if _diff_count >= 3 else "warning"},
                {"label": "Squad Avg xPts",         "value": f"{_squad_xpts_avg:.2f}",   "delta": "Ownership-adjusted squad profile",    "tone": "neutral"},
            ])

            _own_plot = _own_df.nlargest(min(220, len(_own_df)), "xpts_val").copy()
            _fig_own  = px.scatter(
                _own_plot, x="ownership_pct", y="xpts_val", color="in_squad",
                hover_name="player_name",
                hover_data={"team_name": True, "position": True, "ownership_pct": ":.1f", "xpts_val": ":.2f", "predicted_pts": ":.2f"},
                labels={"ownership_pct": "Ownership %", "xpts_val": "xPts", "in_squad": "In your squad"},
                color_discrete_map={True: PLOTLY_PRIMARY, False: PLOTLY_ACCENT},
                opacity=0.78,
            )
            _fig_own.add_vline(x=_squad_own_avg, line_dash="dot", line_color=PLOTLY_PRIMARY,
                               annotation_text="Squad avg", annotation_position="top right")
            _fig_own.add_vline(x=_pool_own_avg,  line_dash="dot", line_color=PLOTLY_ACCENT,
                               annotation_text="Pool avg",  annotation_position="top left")
            _fig_own.update_layout(**PLOTLY_THEME, height=390, margin=dict(l=10,r=10,t=26,b=30))
            st.plotly_chart(_fig_own, use_container_width=True, config={"displayModeBar": "hover", "responsive": True, "scrollZoom": True})
        else:
            st.caption("Ownership positioning unavailable (`player_id` missing).")

        st.divider()

        # Market Watchlist
        render_section_header("Market Watchlist")
        if {"player_id", "player_name"}.issubset(enriched_df.columns):
            _wl = enriched_df.copy()
            _wl["ownership_pct"]          = _wl["player_id"].astype("Int64").map(ownership_map).fillna(0.0).astype(float)
            _wl["xpts_val"]               = _wl.apply(lambda r: _xpts(r), axis=1).astype(float)
            _wl["predicted_price_change"] = pd.to_numeric(_wl.get("predicted_price_change", 0.0), errors="coerce").fillna(0.0)
            _wl["in_squad"]               = _wl["player_id"].astype("Int64").isin(set(data["my_player_ids"]))
            _sq_med   = float(_wl[_wl["in_squad"]]["xpts_val"].median()) if _wl["in_squad"].any() else 0.0
            _pool_q75 = float(_wl["xpts_val"].quantile(0.75)) if not _wl.empty else 0.0

            _sell_w  = _wl[(_wl["in_squad"]) & (_wl["predicted_price_change"] <= -0.10) & (_wl["xpts_val"] <= _sq_med)].sort_values(["predicted_price_change","xpts_val"], ascending=[True,True]).head(6)
            _buy_w   = _wl[(~_wl["in_squad"]) & (_wl["predicted_price_change"] >= 0.10)  & (_wl["xpts_val"] >= _pool_q75)].sort_values(["xpts_val","predicted_price_change"], ascending=[False,False]).head(6)
            _shield  = _wl[(~_wl["in_squad"]) & (_wl["ownership_pct"] >= 35.0)           & (_wl["xpts_val"] >= _pool_q75)].sort_values(["ownership_pct","xpts_val"], ascending=[False,False]).head(6)

            render_stat_cards([
                {"label": "Sell Watch",    "value": str(len(_sell_w)), "delta": "Owned · fall risk + low xPts",           "tone": "warning" if len(_sell_w) > 0 else "neutral"},
                {"label": "Buy Watch",     "value": str(len(_buy_w)),  "delta": "Not-owned · rise momentum + strong xPts","tone": "positive" if len(_buy_w) > 0 else "neutral"},
                {"label": "Shield Picks",  "value": str(len(_shield)), "delta": "High-owned EO threats outside squad",     "tone": "warning" if len(_shield) >= 2 else "neutral"},
            ])

            _wl_rename = {"player_name":"Player","position":"Pos","team_name":"Team","xpts_val":"xPts","predicted_price_change":"Price \u0394","ownership_pct":"Owned%"}
            _wl_cfg    = {"xPts": st.column_config.NumberColumn(format="%.2f"), "Price \u0394": st.column_config.NumberColumn(format="%+.2f"), "Owned%": st.column_config.NumberColumn(format="%.1f")}
            _wl_cols   = ["player_name","position","team_name","xpts_val","predicted_price_change","ownership_pct"]

            _ww1, _ww2 = st.columns(2)
            with _ww1:
                st.markdown("**Sell Watch (Owned)**")
                if _sell_w.empty:
                    st.caption("No immediate sell-pressure flags.")
                else:
                    st.dataframe(_sell_w[_wl_cols].rename(columns=_wl_rename), use_container_width=True, hide_index=True, column_config=_wl_cfg)
            with _ww2:
                st.markdown("**Buy Watch (Not Owned)**")
                if _buy_w.empty:
                    st.caption("No high-conviction buy candidates.")
                else:
                    st.dataframe(_buy_w[_wl_cols].rename(columns=_wl_rename), use_container_width=True, hide_index=True, column_config=_wl_cfg)

            if not _shield.empty:
                with st.expander("Effective Ownership Shield (High-owned threats outside your squad)", expanded=False):
                    _sh_cols = ["player_name","position","team_name","xpts_val","ownership_pct"]
                    _sh_rename = {"player_name":"Player","position":"Pos","team_name":"Team","xpts_val":"xPts","ownership_pct":"Owned%"}
                    _sh_cfg  = {"xPts": st.column_config.NumberColumn(format="%.2f"), "Owned%": st.column_config.NumberColumn(format="%.1f")}
                    st.dataframe(_shield[_sh_cols].rename(columns=_sh_rename), use_container_width=True, hide_index=True, column_config=_sh_cfg)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — MODEL
    # ══════════════════════════════════════════════════════════════════════════
    with tab_model:
        # RMSE chart
        render_section_header("Model Performance by Position")
        _rmse_data = [
            {
                "Position": _pos,
                "RMSE (pts)": round(_rmse, 3),
                "Naive RMSE (pts)": (
                    round(float(data["models"].get(_pos, {}).get("naive_baseline_rmse")), 3)
                    if data["models"].get(_pos, {}).get("naive_baseline_rmse") is not None else np.nan
                ),
                "R2": round(data["models"].get(_pos, {}).get("r2", 0), 3),
                "Beats Baseline": (
                    "Yes" if data["models"].get(_pos, {}).get("beats_baseline") is True
                    else "No" if data["models"].get(_pos, {}).get("beats_baseline") is False
                    else "Unknown"
                ),
            }
            for _pos, _rmse in rmse_map.items()
        ]
        if _rmse_data:
            _rmse_df = pd.DataFrame(_rmse_data)
            _fig_rmse = px.bar(
                _rmse_df, x="Position", y="RMSE (pts)",
                color="RMSE (pts)", color_continuous_scale=PLOTLY_RMSE_SCALE, text="RMSE (pts)",
            )
            _fig_rmse.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            _fig_rmse.update_layout(**PLOTLY_THEME, height=300, showlegend=False,
                                    coloraxis_showscale=False, margin=dict(l=10,r=10,t=20,b=30))
            st.plotly_chart(_fig_rmse, use_container_width=True, config={"displayModeBar": "hover", "responsive": True, "scrollZoom": True})
            st.caption("RMSE = Root Mean Squared Error. Lower = more accurate. GK models typically most accurate due to consistent playing time.")
            render_insight_table(
                _rmse_df,
                row_density="compact",
                column_config={
                    "RMSE (pts)": st.column_config.NumberColumn(format="%.3f"),
                    "Naive RMSE (pts)": st.column_config.NumberColumn(format="%.3f"),
                },
            )
            _under_baseline = _rmse_df[_rmse_df["Beats Baseline"] == "No"]
            if not _under_baseline.empty:
                st.warning(
                    "Model under baseline for: "
                    + ", ".join(_under_baseline["Position"].astype(str).tolist())
                )

        # SHAP
        with st.expander("SHAP Explainability (Top Features by Position)", expanded=False):
            _shap_rows = []
            _models_obj = data.get("models", {})
            if isinstance(_models_obj, dict):
                for _pos, _minfo in _models_obj.items():
                    if not isinstance(_minfo, dict):
                        continue
                    _top_feats = _minfo.get("shap_top_features", {})
                    if not isinstance(_top_feats, dict) or not _top_feats:
                        continue
                    for _feat, _val in _top_feats.items():
                        try:
                            _shap_rows.append({"Position": str(_pos), "Feature": str(_feat), "Mean |SHAP|": float(_val)})
                        except Exception:
                            continue
            if _shap_rows:
                _shap_df = pd.DataFrame(_shap_rows).sort_values("Mean |SHAP|", ascending=False)
                _fig_shap = px.bar(
                    _shap_df, x="Mean |SHAP|", y="Feature", color="Position",
                    orientation="h", facet_col="Position", facet_col_wrap=2,
                    color_discrete_map=POSITION_COLOR_MAP,
                )
                _fig_shap.update_layout(**PLOTLY_THEME, height=520, showlegend=False, margin=dict(l=10,r=10,t=28,b=20))
                _fig_shap.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                st.plotly_chart(_fig_shap, use_container_width=True, config={"displayModeBar": "hover", "responsive": True, "scrollZoom": True})
                st.dataframe(
                    _shap_df.rename(columns={"Mean |SHAP|": "Mean SHAP"}),
                    use_container_width=True, hide_index=True,
                    column_config={"Mean SHAP": st.column_config.NumberColumn(format="%.4f")},
                )
                st.caption("Higher SHAP magnitude means greater influence on predicted points for that position model.")
            else:
                st.caption("SHAP feature importances are unavailable for this data source/run.")

        st.divider()

        # Transfer history accuracy
        render_section_header("Transfer History & Prediction Accuracy")
        if _t_hist_loaded:
            _t_df2 = pd.DataFrame(_t_hist_loaded)
            if "evaluated" not in _t_df2.columns:
                st.warning("Transfer history schema changed: missing 'evaluated' column.")
            else:
                _eval2 = _t_df2[_t_df2["evaluated"] == True]
                if not _eval2.empty:
                    _fig_acc = go.Figure()
                    _fig_acc.add_trace(go.Bar(x=_eval2["player_in"], y=_eval2["predicted_gain"], name="Predicted", marker_color=PLOTLY_ACCENT))
                    _fig_acc.add_trace(go.Bar(x=_eval2["player_in"], y=_eval2["actual_gain"],    name="Actual",    marker_color=PLOTLY_PRIMARY))
                    _fig_acc.update_layout(
                        **PLOTLY_THEME, barmode="group", height=320,
                        title=dict(text="Transfer Prediction Accuracy", font=dict(color=PLOTLY_ACCENT, size=13, family="Space Mono")),
                        xaxis_tickangle=-30, margin=dict(l=10,r=10,t=50,b=80),
                    )
                    st.plotly_chart(_fig_acc, use_container_width=True, config={"displayModeBar": "hover", "responsive": True, "scrollZoom": True})
                    _eval_disp = _eval2[["gw","player_out","player_in","predicted_gain","actual_gain"]].copy()
                    _eval_disp["result"] = _eval_disp.apply(
                        lambda r: "Good" if r["actual_gain"] >= r["predicted_gain"] * 0.7 else "Miss", axis=1
                    )
                    st.dataframe(
                        _eval_disp.rename(columns={"gw":"GW","player_out":"OUT","player_in":"IN","predicted_gain":"Predicted","actual_gain":"Actual","result":"Result"}).sort_values("GW", ascending=False),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    _pending = len(_t_df2[_t_df2["evaluated"] == False])
                    st.info(f"{_pending} transfer(s) logged but not yet evaluable (needs {FIXTURE_LOOKAHEAD}-GW horizon to complete).")

                with st.expander("All logged suggestions", expanded=False):
                    _wanted = ["gw","player_out","player_in","predicted_gain","evaluated"]
                    _safe   = [_c for _c in _wanted if _c in _t_df2.columns]
                    if len(_safe) >= 2:
                        st.dataframe(
                            _t_df2[_safe].rename(columns={"gw":"GW","player_out":"OUT","player_in":"IN","predicted_gain":"Predicted Gain","evaluated":"Evaluated"}),
                            use_container_width=True, hide_index=True,
                        )
                    else:
                        st.warning("Transfer history schema is incomplete.")
        else:
            st.info("No transfer history yet. Make some transfers via the Transfer Planner and they'll be tracked here automatically.")


elif page == "AI Analyst":
    if not ANALYST_AVAILABLE:
        st.error(f"Phase 7 backend not available: {ANALYST_ERROR}")
        st.info("Run: pip install groq feedparser newsapi-python understat nest_asyncio")
        st.stop()

    if "analyst_messages" not in st.session_state:
        st.session_state["analyst_messages"] = []

    # ── 1. Proactive alerts — first thing, no clutter ──────────────────────────
    _analyst_deadline = {}
    if get_deadline_status:
        try:
            _analyst_deadline = get_deadline_status(bootstrap, current_gw) or {}
        except Exception:
            _analyst_deadline = {}
    try:
        _proactive_alerts = generate_proactive_alerts(
            my_team=my_team,
            xi_result=xi_result,
            news_map=news_map,
            chance_map=chance_map,
            chip_info=chip_info,
            deadline_status=_analyst_deadline,
            current_gw=current_gw,
        )
    except Exception:
        _proactive_alerts = []

    if _proactive_alerts:
        for _alert in _proactive_alerts[:5]:
            _level   = str(_alert.get("level", "info")).lower()
            _title   = str(_alert.get("title", "Alert"))
            _message = str(_alert.get("message", ""))
            if _level == "critical":
                st.error(f"{_title}: {_message}")
            elif _level == "warning":
                st.warning(f"{_title}: {_message}")
            else:
                st.info(f"{_title}: {_message}")

    # ── 2. Chat thread ─────────────────────────────────────────────────────────
    for _msg in st.session_state["analyst_messages"]:
        with st.chat_message(_msg["role"]):
            st.markdown(_msg["content"])
            if _msg["role"] == "assistant" and _msg.get("sources_display"):
                with st.expander("Sources & Confidence", expanded=False):
                    st.markdown(_msg["sources_display"])
                    _conf_label = _msg.get("confidence_label", "?")
                    _conf_score = _msg.get("confidence_score", 0)
                    _conf_color = (
                        "var(--primary)" if _conf_label == "HIGH"
                        else "var(--warning)" if _conf_label == "MEDIUM"
                        else "var(--danger)"
                    )
                    st.markdown(
                        f"<div class='kpi-block' style='display:inline-block;padding:0.5rem 1rem;'>"
                        f"<div class='kpi-label'>Source Confidence</div>"
                        f"<div style='color:{_conf_color};font-weight:800;font-size:1.2rem;'>"
                        f"{_conf_label} ({_conf_score:.0f}%)</div></div>",
                        unsafe_allow_html=True,
                    )
                    _meta = _msg.get("meta", {}) if isinstance(_msg.get("meta"), dict) else {}
                    if _meta:
                        _dl      = _meta.get("deadline_status") or {}
                        _urgency = _dl.get("urgency", "UNKNOWN")
                        _hours   = _dl.get("hours_remaining")
                        _hrs_txt = "n/a" if _hours is None else f"{float(_hours):.1f}h"
                        st.caption(
                            " | ".join([
                                f"Deadline: {_urgency} ({_hrs_txt})",
                                f"Odds usage: {_meta.get('odds_usage', 'n/a')}",
                                f"From cache: {'Yes' if _meta.get('cached') else 'No'}",
                                f"Conflicts: {int(_meta.get('contradictions_count', 0))}",
                                f"Stale sources: {int(_meta.get('staleness_count', 0))}",
                            ])
                        )
                        _ctx = str(_meta.get("context_preview", "") or "")
                        if _ctx:
                            with st.expander("Context snapshot (truncated)", expanded=False):
                                st.code(_ctx, language="text")

    # ── 3. Quick question chips — all visible, no expander ────────────────────
    if QUICK_QUESTIONS:
        _qq_cols = st.columns(4)
        _quick_q = None
        for _qi, _q in enumerate(QUICK_QUESTIONS[:8]):
            with _qq_cols[_qi % 4]:
                if st.button(_q, use_container_width=True, key=f"qq_{_qi}"):
                    _quick_q = _q

    # ── 4. Clear button — only when thread has messages ────────────────────────
    if st.session_state["analyst_messages"]:
        _cleft, _cmid, _cright = st.columns([3, 1, 3])
        with _cmid:
            if st.button("Clear", key="clear_chat", use_container_width=True):
                st.session_state["analyst_messages"] = []
                st.rerun()

    # ── 5. System status (dev mode only) ──────────────────────────────────────
    if dev_mode:
        with st.expander("System Status", expanded=False):
            _status_items = [
                ("LLM (Groq)",    ANALYST_STATUS.get("groq", False),       "Ready", "Not installed"),
                ("NewsAPI",       ANALYST_STATUS.get("newsapi", False),     "Ready", "No key"),
                ("RSS Feeds",     ANALYST_STATUS.get("feedparser", False),  "Ready", "Not installed"),
                ("Understat xG",  ANALYST_STATUS.get("understat", False),  "Ready", "Not installed"),
                ("The Odds API",  ANALYST_STATUS.get("odds_api", False),   "Ready", "No key"),
            ]
            _st_cols = st.columns(len(_status_items))
            for _si, (_slabel, _sok, _ok_txt, _fail_txt) in enumerate(_status_items):
                _st_cols[_si].markdown(
                    f"<div class='kpi-block'><div class='kpi-label'>{_slabel}</div>"
                    f"<div style='font-weight:800;'>"
                    f"{_ok_txt if _sok else _fail_txt}</div></div>",
                    unsafe_allow_html=True,
                )
            if ANALYST_STATUS.get("odds_api", False):
                st.caption(get_odds_usage_summary())

    # ── 6. Padding for sticky chat input ──────────────────────────────────────
    st.markdown(
        "<style>.main .block-container{padding-bottom:8rem!important;}"
        "@media(max-width:980px){.main .block-container{padding-bottom:7rem!important;}}"
        "</style>",
        unsafe_allow_html=True,
    )

    # ── 7. Chat input (sticky bottom) + answer pipeline ───────────────────────
    _user_input = st.chat_input("Ask anything about your squad, transfers, captain, injuries...")
    _question   = _quick_q if QUICK_QUESTIONS and _quick_q else _user_input

    if _question:
        st.session_state["analyst_messages"].append({"role": "user", "content": _question})
        st.session_state["analyst_messages"] = st.session_state["analyst_messages"][-30:]

        _llm_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state["analyst_messages"][:-1]
        ]
        try:
            _cilp_1  = cached_ilp_transfers(my_team, others, float(bank_balance), n_transfers=1)
            _cilp_2  = cached_ilp_transfers(my_team, others, float(bank_balance), n_transfers=2)
            _croll   = cached_rolling_advice(
                my_team, others, float(bank_balance), int(transfers_made),
                json.dumps(chip_info, default=str), int(current_gw),
                json.dumps(_cilp_1, default=str),
            )
            _chits   = cached_hit_analysis(my_team, others, float(bank_balance), int(transfers_made))
        except Exception:
            _cilp_1 = _cilp_2 = _croll = None
            _chits  = []

        with st.spinner("Fetching live data and consulting the analyst..."):
            try:
                _result = run_analyst(
                    question=_question,
                    my_team=my_team,
                    others=others,
                    enriched_df=enriched_df,
                    xi_result=xi_result,
                    bank_balance=bank_balance,
                    transfers_made=transfers_made,
                    available_chips=available_chips,
                    current_gw=current_gw,
                    news_map=news_map,
                    chance_map=chance_map,
                    bootstrap=bootstrap,
                    chat_history=_llm_history,
                    ilp_1=_cilp_1,
                    ilp_2=_cilp_2,
                    roll_advice=_croll,
                    hit_transfers=_chits,
                )
                _clabel, _cscore = _result["confidence"]
                st.session_state["analyst_messages"].append({
                    "role":             "assistant",
                    "content":          _result["answer"],
                    "sources_display":  _result["source_display"],
                    "confidence_label": _clabel,
                    "confidence_score": _cscore,
                    "meta": {
                        "deadline_status":      _result.get("deadline_status", {}),
                        "odds_usage":           _result.get("odds_usage", "n/a"),
                        "cached":               bool(_result.get("cached", False)),
                        "contradictions_count": len(_result.get("contradictions", []) or []),
                        "staleness_count":      len(_result.get("staleness", []) or []),
                        "context_preview":      str(_result.get("context_used", "") or "")[:800],
                    },
                })
            except Exception as _ae:
                st.session_state["analyst_messages"].append({
                    "role":             "assistant",
                    "content":         f"Error running analyst: {_ae}",
                    "sources_display":  "",
                    "confidence_label": "LOW",
                    "confidence_score": 0,
                })
        st.rerun()

    st.caption(
        "Powered by Groq (Llama 3.3 70B) · "
        "Grounded in live FPL + news data · "
        "Always verify before deadline"
    )


# Re-inject at end of page too for cascade victory (early call covers st.stop() pages)
try:
    inject_dropdown_overrides(ui_tokens)
except Exception:
    pass

if page != "AI Analyst":
    st.caption("Always verify bank balance in the FPL app before confirming transfers.")
