"""
FPL AI Assistant — Phase 6: Streamlit Dashboard (v2)
=====================================================
Updated to consume Phase 1-4 v5 pipeline outputs.

Changes over v1:
  1. load_all_data now runs full Phase 1 v5 pipeline:
     component blend, compute_expected_pts, price predictions,
     build_cs_probability_map. All pages now receive expected_pts,
     pts_low, pts_high, captain_ev, p_plays_full, predicted_price_change.

  2. Fixed crash: compute_score_range returns 3-tuple (lo, hi, ci_label).
     All call sites updated to unpack 3 values.

  3. Fixed squad value: uses get_squad_value_breakdown() (sell_price)
     not raw price.sum().

  4. Captain Picker: uses captain_ev (Phase 2 v5) as primary sort,
     run_monte_carlo_captain for win_prob top-5 table,
     get_captaincy_differential_analysis for vs-field table.
     VC now uses p_plays_full reliability.

  5. Transfer Planner: shows total_ev (xPts + price movement),
     predicted_price_change 📈/📉 emoji, urgency_score flag,
     get_horizon_transfer_plan 2GW plan section,
     get_double_hit_analysis -8pt section.

  6. My Squad: shows expected_pts alongside predicted_pts,
     pts_low–pts_high CI range, p_plays_full bench reliability,
     sell_price vs now_price in squad table.

  7. Season Tracker: squad value uses sell_price from breakdown.

  8. Theme selector in sidebar now triggers st.rerun() so
     theme applies immediately.

  9. Confidence scores now derived from actual model outputs
     (total_ev, captain_ev, pts spread) not invented heuristics.

Pages: Home · My Squad · Fixture Planner · Transfer Planner ·
       Player Explorer · Captain Picker · Season Tracker · AI Analyst

Run:  streamlit run fpl_dashboard.py
"""

import os
import sys
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────
# BACKEND IMPORTS
# ─────────────────────────────────────────

try:
    from fpl_phase1_model import (
        fetch_bootstrap, fetch_fixtures, fetch_current_gw,
        fetch_my_team, fetch_transfer_info,
        build_player_history_df, build_current_features,
        train_models, train_component_models, predict_component_pts,
        add_price_predictions, train_price_model, compute_expected_pts,
        FEATURE_COLS, COMPONENT_BLEND_WEIGHT,
    )
    from fpl_phase2_fixtures import (
        build_custom_difficulty, build_team_form,
        build_opponent_scoring_map, build_chip_status,
        build_fixture_run, build_player_fixture_scores,
        build_cs_probability_map,
        FIXTURE_LOOKAHEAD,
    )
    from fpl_phase3_constraints import (
        validate_squad, get_ilp_optimal_transfers,
        get_valid_double_transfers, get_hit_transfer_analysis,
        get_rolling_transfer_advice, get_differential_picks,
        get_squad_value_breakdown, track_squad_value,
        run_monte_carlo_captain,
        get_captaincy_differential_analysis,
        get_horizon_transfer_plan,
        get_double_hit_analysis,
    )
    from fpl_phase4_optimizer import (
        optimize_xi_ilp, score_all_formations,
        compute_score_range, get_rmse_from_models,
        xpts_captain_score,
    )
    from config import TEAM_ID, VALID_FORMATIONS, POSITION_LIMITS
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    IMPORT_ERROR = str(e)

try:
    from fpl_phase7_analyst import (
        run_analyst, QUICK_QUESTIONS,
        ANALYST_STATUS, generate_proactive_alerts,
        get_odds_usage_summary,
    )
    ANALYST_AVAILABLE = True
except ImportError as e:
    ANALYST_AVAILABLE = False
    ANALYST_ERROR = str(e)
    ANALYST_STATUS = {}

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="FPL AI Assistant",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# BASE CSS (dark theme — always applied)
# ─────────────────────────────────────────

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
.main .block-container { font-family: 'Syne', sans-serif; }
[data-testid="stAppViewContainer"] { background: transparent; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1224 0%, #090e1a 100%);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * { color: #cfdcf6 !important; }
.main .block-container { padding: 1.4rem 2rem 1.6rem; max-width: 1440px; }

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
.kpi-value { font-size: 1.75rem; font-weight: 800; color: var(--primary); line-height: 1; }
.kpi-delta { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: var(--muted); margin-top: 0.2rem; }

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
.transfer-card:hover { transform: translateY(-2px); border-color: var(--primary); }
.transfer-out { color: var(--danger); font-weight: 700; }
.transfer-in  { color: var(--primary); font-weight: 700; }
.transfer-gain { font-family: 'Space Mono', monospace; font-size: 1.02rem; color: var(--primary); }
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
.rec-box.warning { background: linear-gradient(160deg, #2f2310 0%, #1a1308 100%); border-color: var(--warning); }
.rec-box.danger  { background: linear-gradient(160deg, #32141d 0%, #1a0d13 100%); border-color: var(--danger); }
.entity-line { display: flex; align-items: center; gap: 0.55rem; margin-top: 0.25rem; }
.team-badge { width: 20px; height: 20px; border-radius: 50%; object-fit: contain; background: rgba(255,255,255,0.06); }
.lineup-board {
    position: relative;
    border: 1px solid #1f4d2f;
    border-radius: 14px;
    padding: 1rem 0.9rem 0.9rem;
    background:
        linear-gradient(180deg, rgba(5,39,26,0.94), rgba(2,26,17,0.96)),
        repeating-linear-gradient(0deg, rgba(255,255,255,0.0) 0, rgba(255,255,255,0.0) 59px, rgba(95,169,118,0.08) 60px);
    box-shadow: 0 10px 28px rgba(2,14,8,0.6);
}
.lineup-row { display: flex; justify-content: center; gap: 0.65rem; margin: 0.55rem 0; flex-wrap: wrap; }
.lineup-label { text-align: center; font-family: 'Space Mono', monospace; font-size: 0.62rem; letter-spacing: 0.16em; color: #7fd7ff; margin-top: 0.35rem; }
.xi-tile {
    width: clamp(122px, 24vw, 150px); min-width: 118px; max-width: 100%;
    border: 1px solid #24543c;
    background: linear-gradient(160deg, rgba(11,34,23,0.9), rgba(10,26,20,0.92));
    border-radius: 11px; padding: 0.45rem 0.5rem;
    transition: transform 160ms ease, border-color 160ms ease;
}
.xi-tile:hover { transform: translateY(-2px); border-color: #2cae7f; }
.xi-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.26rem; }
.xi-pts {
    font-family: 'Space Mono', monospace; font-size: 0.7rem; font-weight: 700;
    border-radius: 999px; padding: 0.1rem 0.42rem; border: 1px solid transparent;
}
.xi-pts.elite { color: #27e8a7; border-color: rgba(39,232,167,0.45); background: rgba(39,232,167,0.08); }
.xi-pts.good  { color: #37b6ff; border-color: rgba(55,182,255,0.45); background: rgba(55,182,255,0.08); }
.xi-pts.mid   { color: #ffb547; border-color: rgba(255,181,71,0.45);  background: rgba(255,181,71,0.08); }
.xi-pts.low   { color: #ff5d73; border-color: rgba(255,93,115,0.45);  background: rgba(255,93,115,0.08); }
.xi-name { font-size: 0.79rem; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.xi-meta { display: flex; align-items: center; gap: 0.3rem; color: #9dc2e8; font-size: 0.68rem; margin-top: 0.2rem; }
.xi-role {
    font-family: 'Space Mono', monospace; font-size: 0.58rem;
    border-radius: 999px; padding: 0.09rem 0.36rem;
    border: 1px solid #2d5a8f; color: #a9d9ff;
}
.xi-role.cap   { border-color: #ffb547; color: #ffcf88; }
.xi-role.vc    { border-color: #37b6ff; color: #9adfff; }
.xi-role.blank { border-color: #ff5d73; color: #ff9eac; }
.xi-role.dgw   { border-color: #27e8a7; color: #8dffd9; }
.player-face {
    width: 44px; height: 44px; border-radius: 50%; object-fit: cover;
    border: 1px solid #33588d;
    box-shadow: 0 0 0 1px rgba(13,22,40,0.8), 0 8px 18px rgba(4,10,23,0.6);
    background: #0b1224;
}
.player-face-sm { width: 34px; height: 34px; border-radius: 50%; object-fit: cover; border: 1px solid #2b4e84; background: #0b1224; }
[data-testid="stTabs"] button { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.84rem; color: #8ea4c2; transition: color 180ms ease, transform 180ms ease; }
[data-testid="stTabs"] button:hover { color: var(--accent); transform: translateY(-1px); }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--primary) !important; border-bottom-color: var(--primary) !important; }
.dataframe { font-family: 'Space Mono', monospace; font-size: 0.75rem; }
.js-plotly-plot { border-radius: var(--radius-sm); overflow: hidden; border: 1px solid var(--line); }
.js-plotly-plot .modebar { display: none !important; }
[data-testid="stMetric"] { background: linear-gradient(160deg, #122140 0%, #0f1a32 100%); border: 1px solid var(--line); border-radius: var(--radius-sm); padding: 0.72rem 0.95rem; box-shadow: var(--shadow); transition: transform 180ms ease, border-color 180ms ease; }
[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: #2e538d; }
[data-testid="stMetricLabel"] { color: var(--accent) !important; font-size: 0.7rem !important; }
[data-testid="stMetricValue"] { color: var(--primary) !important; font-size: 1.5rem !important; }
[data-testid="stMetricDelta"] { color: var(--muted) !important; }
[data-testid="stDivider"] { border-color: #1a2f54; }
.skeleton-card { border: 1px solid var(--line); border-radius: var(--radius-sm); background: linear-gradient(90deg, #0f1730 25%, #152443 50%, #0f1730 75%); background-size: 200% 100%; animation: shimmer 1.25s ease-in-out infinite; }
.skeleton-sm { height: 42px; margin: 0.25rem 0; }
.skeleton-md { height: 88px; margin: 0.45rem 0; }
.skeleton-lg { height: 220px; margin: 0.55rem 0; }
.price-up   { color: #27e8a7; font-weight: 700; }
.price-down { color: #ff5d73; font-weight: 700; }
@keyframes cardIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
@media (max-width: 980px) {
    .main .block-container { padding: 1rem 0.8rem 1.3rem; }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
    .xi-tile { width: clamp(114px, 40vw, 138px); padding: 0.4rem 0.45rem; }
    .xi-name { font-size: 0.74rem; }
    .xi-meta { font-size: 0.62rem; }
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# THEME TOKENS
# ─────────────────────────────────────────

PLOTLY_THEME = dict(
    paper_bgcolor="#090e1a",
    plot_bgcolor="#111a2e",
    font=dict(family="Syne, sans-serif", color="#eaf2ff", size=12),
    xaxis=dict(gridcolor="#1f3459", linecolor="#1f3459", tickcolor="#37b6ff"),
    yaxis=dict(gridcolor="#1f3459", linecolor="#1f3459", tickcolor="#37b6ff"),
    colorway=["#27e8a7", "#37b6ff", "#ffb547", "#ff5d73", "#70d0ff", "#8ae8c7"],
    transition=dict(duration=380, easing="cubic-in-out"),
)
PLOTLY_PRIMARY = "#27e8a7"
PLOTLY_ACCENT  = "#37b6ff"
PLOTLY_WARNING = "#ffb547"
PLOTLY_DANGER  = "#ff5d73"
PLOTLY_TEXT    = "#eaf2ff"
PLOTLY_SURFACE = "#111a2e"
PLOTLY_LINE    = "#1f3459"
PLOTLY_XPTS_SCALE = [[0, "#1f3459"], [0.5, "#37b6ff"], [1, "#27e8a7"]]
PLOTLY_RMSE_SCALE = [[0, "#27e8a7"], [0.5, "#ffb547"], [1, "#ff5d73"]]
POSITION_COLOR_MAP = {
    "Goalkeeper": "#27e8a7",
    "Defender":   "#37b6ff",
    "Midfielder": "#ffb547",
    "Forward":    "#ff5d73",
}
DIFFICULTY_COLORS = {1: "#1a7a4a", 2: "#2ecc71", 3: "#ffb547", 4: "#e67e22", 5: "#ff5d73", 6: "#7f0000"}


# ─────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────

dev_mode = os.getenv("FPL_DEBUG_UI", "0") == "1"
_defaults = {
    "cfg_team_id":       int(TEAM_ID if BACKEND_AVAILABLE else 9179961),
    "cfg_bank_override": 0.0,
    "cfg_refresh":       False,
    "cfg_show_qa_panel": False,
    "data_refreshed_at": datetime.now().isoformat(timespec="seconds"),
    "ui_theme":          "dark",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚽ FPL AI")
    st.caption("ASSISTANT")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Home", "My Squad", "Fixture Planner", "Transfer Planner",
         "Player Explorer", "Captain Picker", "Season Tracker", "AI Analyst"],
        label_visibility="collapsed",
    )
    st.divider()

    # Theme selector — triggers rerun so CSS rebuilds immediately
    prev_theme = st.session_state.get("ui_theme", "dark")
    new_theme = st.selectbox("Theme", ["dark", "light"], key="ui_theme_select",
                             index=0 if prev_theme == "dark" else 1)
    if new_theme != prev_theme:
        st.session_state["ui_theme"] = new_theme
        st.rerun()

    st.caption("Decision-first FPL assistant")

# ─────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────

last_refresh_dt = datetime.fromisoformat(st.session_state["data_refreshed_at"])
age_seconds = (datetime.now() - last_refresh_dt).total_seconds()
is_stale = age_seconds > 300
freshness_label = "Stale" if is_stale else "Fresh"
freshness_color = PLOTLY_WARNING if is_stale else PLOTLY_PRIMARY
bank_chip = (
    f"Bank £{float(st.session_state['cfg_bank_override']):.1f}M"
    if float(st.session_state["cfg_bank_override"]) > 0
    else "Bank Auto"
)

topbar_left, topbar_right = st.columns([0.6, 4.4])
with topbar_left:
    with st.popover("⚙", use_container_width=True):
        st.number_input("Your FPL Team ID", key="cfg_team_id", step=1,
                        help="Find your team ID in the FPL website URL")
        st.number_input("Bank Balance Override (£M)", key="cfg_bank_override",
                        step=0.1, min_value=0.0,
                        help="Override if API bank differs from FPL app")
        st.toggle("Force Fresh Data", key="cfg_refresh",
                  help="Re-fetch all player history from API (~2 min)")
        if st.button("Refresh Data", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.session_state["data_refreshed_at"] = datetime.now().isoformat(timespec="seconds")
            st.rerun()
        st.caption(
            f"Last refresh: {last_refresh_dt.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Data is {freshness_label.lower()}"
        )
        if dev_mode:
            st.toggle("Show QA panel", key="cfg_show_qa_panel")
with topbar_right:
    st.markdown(
        f"""
        <div style='display:flex; gap:0.45rem; flex-wrap:wrap; align-items:center; margin-top:0.22rem;'>
            <span class='xi-role'>Team {int(st.session_state['cfg_team_id'])}</span>
            <span class='xi-role'>{bank_chip}</span>
            <span class='xi-role' style='border-color:{freshness_color}; color:{freshness_color};'>
                Data {freshness_label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

team_id_input  = int(st.session_state["cfg_team_id"])
bank_override  = float(st.session_state["cfg_bank_override"])
refresh        = bool(st.session_state["cfg_refresh"])
show_qa_panel  = bool(st.session_state["cfg_show_qa_panel"]) if dev_mode else False


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def _safe_text(text) -> str:
    txt = str(text) if text is not None else ""
    return (txt.replace("&", "&amp;").replace("<", "&lt;")
               .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;"))


def _xpts(row) -> float:
    """Return expected_pts if available, else predicted_pts."""
    v = row.get("expected_pts")
    if v is not None and not pd.isna(v) and float(v) > 0:
        return float(v)
    return float(row.get("predicted_pts", 0))


def _price_tag(change: float) -> str:
    if change > 0.05:  return " 📈"
    if change < -0.05: return " 📉"
    return ""


def render_section_header(title: str):
    st.markdown(f"<div class='section-header'>{_safe_text(title)}</div>", unsafe_allow_html=True)


def render_stat_cards(cards: list, compact: bool = False):
    if not cards:
        return
    per_row = 2 if compact else min(3, len(cards))
    for i in range(0, len(cards), per_row):
        row  = cards[i:i + per_row]
        cols = st.columns(len(row))
        for idx, card in enumerate(row):
            tone_color = {
                "positive": PLOTLY_PRIMARY,
                "warning":  PLOTLY_WARNING,
                "danger":   PLOTLY_DANGER,
                "neutral":  PLOTLY_ACCENT,
            }.get(card.get("tone", "neutral"), PLOTLY_ACCENT)
            with cols[idx]:
                st.markdown(
                    f"<div class='kpi-block' style='text-align:left;'>"
                    f"<div class='kpi-label'>{_safe_text(card.get('label',''))}</div>"
                    f"<div class='kpi-value' style='font-size:1.45rem; color:{tone_color};'>"
                    f"{_safe_text(card.get('value',''))}</div>"
                    f"<div class='kpi-delta'>{_safe_text(card.get('delta',''))}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


def render_decision_banner(title: str, primary_action: str, confidence: float,
                           reasons: list, risk_level: str = "Medium"):
    risk_cls = "danger" if risk_level == "High" else "warning" if risk_level == "Medium" else ""
    reasons_html = "".join(
        f"<li style='margin:0.15rem 0; color:#b9cae6; font-size:0.82rem;'>{_safe_text(r)}</li>"
        for r in (reasons or [])
    )
    st.markdown(
        f"""<div class='rec-box {risk_cls}'>
            <div class='kpi-label'>{_safe_text(title)}</div>
            <div style='display:flex; justify-content:space-between; gap:0.8rem; align-items:flex-start;'>
                <div>
                    <div style='font-size:1.15rem; font-weight:800; color:#eaf2ff;'>{_safe_text(primary_action)}</div>
                    <div style='font-size:0.78rem; color:#90a2be; margin-top:0.2rem;'>Risk: {_safe_text(risk_level)}</div>
                </div>
                <div style='text-align:right;'>
                    <div style='font-size:0.68rem; color:#37b6ff; font-family:Space Mono;'>CONFIDENCE
                        <span title='Signal strength from model agreement, gain margin, fixture context, availability. Not a guarantee.'
                              style='cursor:help; color:#90a2be; margin-left:0.2rem;'>ⓘ</span>
                    </div>
                    <div style='font-size:1.25rem; font-weight:800; color:#27e8a7;'>{float(confidence):.0f}%</div>
                </div>
            </div>
            <ul style='margin:0.55rem 0 0 1.0rem; padding:0;'>{reasons_html}</ul>
        </div>""",
        unsafe_allow_html=True,
    )


def render_recommendation_card(headline: str, impact_now: float, impact_horizon: float,
                                confidence: float, risk_notes: list, supporting_points: list):
    risks  = "".join(f"<li style='margin:0.12rem 0; color:#ffb2bf; font-size:0.76rem;'>{_safe_text(r)}</li>" for r in (risk_notes or []))
    points = "".join(f"<li style='margin:0.12rem 0; color:#bdd3f2; font-size:0.76rem;'>{_safe_text(p)}</li>" for p in (supporting_points or []))
    st.markdown(
        f"""<div class='transfer-card'>
            <div class='kpi-label'>Recommendation</div>
            <div style='font-size:1rem; font-weight:800; color:#eaf2ff; margin-bottom:0.35rem;'>{_safe_text(headline)}</div>
            <div style='display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:0.45rem;'>
                <span class='xi-role'>Next GW: {impact_now:+.2f}</span>
                <span class='xi-role'>5 GW: {impact_horizon:+.2f}</span>
                <span class='xi-role' title='Not a guarantee.'>Confidence: {confidence:.0f}% ⓘ</span>
            </div>
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.8rem;'>
                <div><div class='kpi-label'>Why</div><ul style='margin:0.2rem 0 0 1rem; padding:0;'>{points}</ul></div>
                <div><div class='kpi-label'>Risks</div><ul style='margin:0.2rem 0 0 1rem; padding:0;'>{risks}</ul></div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_insight_table(df: pd.DataFrame, column_config: dict = None,
                         default_sort: tuple = None, row_density: str = "normal"):
    if df is None or df.empty:
        st.info("No data available for this view.")
        return
    table_df = df.copy()
    if default_sort and default_sort[0] in table_df.columns:
        table_df = table_df.sort_values(default_sort[0], ascending=default_sort[1])
    st.dataframe(table_df, use_container_width=True, hide_index=True,
                 height=360 if row_density == "compact" else None,
                 column_config=column_config or {})


def render_loading_skeleton():
    st.markdown("""
        <div class='skeleton-card skeleton-sm'></div>
        <div class='skeleton-card skeleton-md'></div>
        <div class='skeleton-card skeleton-md'></div>
        <div class='skeleton-card skeleton-lg'></div>
    """, unsafe_allow_html=True)


def player_identity_html(player_name: str, team_name: str, face_url: str = "",
                         badge_url: str = "", subtitle: str = "",
                         face_class: str = "player-face") -> str:
    pname    = _safe_text(player_name)
    tname    = _safe_text(team_name)
    sub_safe = _safe_text(subtitle)
    face     = _safe_text(face_url) if face_url else ""
    badge    = _safe_text(badge_url) if badge_url else (
        "data:image/svg+xml;utf8,"
        "<svg xmlns='http://www.w3.org/2000/svg' width='40' height='40'>"
        "<circle cx='20' cy='20' r='19' fill='%23111a2e' stroke='%232e4f84' stroke-width='2'/>"
        "<text x='50%' y='56%' text-anchor='middle' fill='%2337b6ff' "
        "font-family='Space Mono' font-size='15'>FC</text></svg>"
    )
    init     = (pname[:1] or "?").upper()
    sub_html = (f"<div style='font-size:0.74rem; color:#90a2be; margin-top:0.1rem;'>{sub_safe}</div>"
                if sub_safe else "")
    return (
        f"<div class='entity-line'>"
        f"<img class='{face_class}' src='{face}' "
        "onerror=\"this.style.display='none'; this.nextElementSibling.style.display='flex';\" />"
        f"<div class='{face_class}' style='display:none; align-items:center; justify-content:center;"
        " color:#eaf2ff; font-weight:700; background:linear-gradient(145deg,#172a4a,#0d1730);'>"
        f"{init}</div>"
        "<div style='min-width:0;'>"
        f"<div style='font-weight:700; font-size:0.96rem; line-height:1.1;'>{pname}</div>"
        "<div class='entity-line' style='margin-top:0.16rem; gap:0.35rem;'>"
        f"<img class='team-badge' src='{badge}' onerror=\"this.onerror=null;this.style.display='none';\" />"
        f"<span style='font-size:0.75rem; color:#9cb0d0;'>{tname}</span>"
        "</div>"
        f"{sub_html}"
        "</div></div>"
    )


def build_lineup_board_html(xi_df: pd.DataFrame, cap_id: int, vc_id: int) -> str:
    if xi_df is None or xi_df.empty:
        return "<div class='fpl-card'>No XI available.</div>"
    order     = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
    label_map = {"Forward": "ATTACK", "Midfielder": "MIDFIELD",
                 "Defender": "DEFENCE", "Goalkeeper": "GOALKEEPER"}
    rows_html = []
    for pos in order:
        pos_df = xi_df[xi_df["position"] == pos].copy()
        if pos_df.empty:
            continue
        pos_df = pos_df.sort_values("predicted_pts", ascending=False)
        tiles  = []
        for _, r in pos_df.iterrows():
            pid   = int(r.get("player_id", 0))
            # Prefer expected_pts for display on the tile
            pts   = float(r.get("expected_pts", r.get("predicted_pts", 0.0)))
            pts_class = "elite" if pts >= 8 else "good" if pts >= 5 else "mid" if pts >= 3 else "low"
            roles = []
            if pid == int(cap_id):   roles.append("<span class='xi-role cap'>C</span>")
            elif pid == int(vc_id):  roles.append("<span class='xi-role vc'>VC</span>")
            if bool(r.get("is_blank_next_gw", False)): roles.append("<span class='xi-role blank'>BLK</span>")
            if float(r.get("double_gws", 0) or 0) > 0: roles.append("<span class='xi-role dgw'>DGW</span>")
            # price change tag
            pchg = float(r.get("predicted_price_change", 0) or 0)
            if pchg > 0.05:   roles.append("<span class='xi-role' style='border-color:#27e8a7;color:#27e8a7;'>📈</span>")
            elif pchg < -0.05: roles.append("<span class='xi-role' style='border-color:#ff5d73;color:#ff5d73;'>📉</span>")
            face  = _safe_text(r.get("player_face", ""))
            badge = _safe_text(r.get("team_badge", ""))
            pname = _safe_text(r.get("player_name", "Unknown"))
            team  = _safe_text(r.get("team_name", ""))
            price = float(r.get("price", 0))
            run   = _safe_text(r.get("fixture_run_label", "?"))
            tiles.append(
                "<div class='xi-tile'>"
                "<div class='xi-top'>"
                f"<img class='player-face-sm' src='{face}' "
                "onerror=\"this.style.display='none'; this.nextElementSibling.style.display='flex';\" />"
                "<div class='player-face-sm' style='display:none; align-items:center; justify-content:center;"
                " color:#eaf2ff; font-weight:700; background:linear-gradient(145deg,#172a4a,#0d1730);'>?</div>"
                f"<span class='xi-pts {pts_class}'>{pts:.1f}</span>"
                "</div>"
                f"<div class='xi-name'>{pname}</div>"
                "<div class='xi-meta'>"
                f"<img class='team-badge' src='{badge}' onerror=\"this.onerror=null;this.style.display='none';\" />"
                f"<span>{team} · £{price:.1f}</span>"
                "</div>"
                "<div class='xi-meta'>"
                f"<span style='font-family:Space Mono;'>Run: {run}</span>"
                f"{''.join(roles)}"
                "</div></div>"
            )
        rows_html.append(
            f"<div class='lineup-row'>{''.join(tiles)}</div>"
            f"<div class='lineup-label'>{label_map.get(pos, pos)} · {len(pos_df)}</div>"
        )
    return f"<div class='lineup-board'>{''.join(rows_html)}</div>"


def _fixture_confidence_and_swing(diffs: list, blanks: int = 0):
    if not diffs:
        return 45.0, 0.0
    arr  = np.array([float(x) for x in diffs], dtype=float)
    cons = 1.0 / (1.0 + np.std(arr))
    gap  = max(0.0, min(1.0, (3.2 - np.mean(arr)) / 2.2))
    pen  = min(0.35, 0.12 * max(0, int(blanks)))
    conf = float(np.clip((0.55 * gap + 0.45 * cons - pen) * 100.0, 25.0, 92.0))
    first = float(np.mean(arr[:2])) if len(arr) >= 2 else float(np.mean(arr))
    last  = float(np.mean(arr[-2:])) if len(arr) >= 2 else float(np.mean(arr))
    return conf, last - first


def compute_transfer_confidence(rec: str, ilp_1: dict, hit_transfers: list) -> float:
    """Derive confidence from actual model outputs: total_ev, gain margin."""
    # Use total_ev (xPts + price movement) when available, else total_gain
    ev    = float(ilp_1.get("total_ev", ilp_1.get("total_gain", 0.0)) or 0.0)
    gain1 = float(ilp_1.get("total_next_gain", 0.0) or 0.0)
    hits  = len(hit_transfers or [])
    base  = 56.0
    if rec == "USE NOW":
        score = base + 9.0 + 6.0 * min(ev, 3.0) + 4.0 * min(gain1, 2.0) - 2.0 * hits
    elif rec == "BORDERLINE":
        score = base + 2.0 + 3.5 * min(ev, 2.0) + 2.0 * min(gain1, 1.5) - 1.5 * hits
    else:
        score = base - 8.0 + 1.5 * min(ev, 1.0) - 2.0 * hits
    return float(np.clip(score, 30.0, 92.0))


def compute_fixture_decision_confidence(avg_diffs: list, next2: list, blank_counts: list) -> float:
    if not avg_diffs or not next2:
        return 52.0
    spread = float(max(next2) - min(next2))
    cons   = 1.0 / (1.0 + float(np.std(next2)))
    pen    = min(0.25, 0.03 * float(sum(blank_counts or [0])))
    return float(np.clip((0.6 * min(1.0, spread / 2.5) + 0.4 * cons - pen) * 100.0, 38.0, 89.0))


def verify_runtime_schema(my_team_df: pd.DataFrame, others_df: pd.DataFrame,
                          fixtures: pd.DataFrame) -> list:
    issues    = []
    core_cols = {"player_id", "player_name", "team_id", "team_name", "position", "price", "predicted_pts"}
    for name, df in [("My Squad", my_team_df), ("Player pool", others_df)]:
        missing = sorted(core_cols - set(df.columns))
        if missing:
            issues.append(f"{name} missing columns: {', '.join(missing)}")
    fix_cols = {"event", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"}
    missing  = sorted(fix_cols - set(fixtures.columns))
    if missing:
        issues.append(f"Fixtures missing columns: {', '.join(missing)}")
    return issues


@st.cache_data(ttl=180, show_spinner=False)
def cached_ilp_transfers(my_team_df: pd.DataFrame, others_df: pd.DataFrame,
                         bank_balance: float, n_transfers: int):
    return get_ilp_optimal_transfers(my_team_df, others_df, bank_balance, n_transfers=n_transfers)


# ─────────────────────────────────────────
# ASSET MAPS
# ─────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_asset_maps(bootstrap: dict):
    player_face_map = {}
    team_badge_map  = {}
    for p in bootstrap.get("elements", []):
        pid  = int(p.get("id", 0))
        code = p.get("code")
        if code:
            player_face_map[pid] = (
                f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{int(code)}.png"
            )
        else:
            photo = str(p.get("photo", ""))
            if photo:
                player_face_map[pid] = (
                    "https://resources.premierleague.com/premierleague/photos/players/110x140/"
                    + photo.replace(".jpg", ".png")
                )
    for t in bootstrap.get("teams", []):
        tid  = int(t.get("id", 0))
        code = t.get("code")
        if code:
            team_badge_map[tid] = (
                f"https://resources.premierleague.com/premierleague/badges/70/t{int(code)}.png"
            )
    return player_face_map, team_badge_map


# ─────────────────────────────────────────
# MAIN DATA PIPELINE  (v2: Phase 1 v5 full)
# ─────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_all_data(team_id: int, refresh: bool = False):
    """
    Full Phase 1-4 v5 pipeline. Produces expected_pts, pts_low, pts_high,
    captain_ev, p_plays_full, predicted_price_change in enriched_df.
    """
    bootstrap   = fetch_bootstrap()
    fixtures_df = fetch_fixtures()
    current_gw  = fetch_current_gw(bootstrap)

    team_data     = fetch_my_team(team_id, current_gw)
    my_player_ids = [p["element"] for p in team_data["picks"]]
    transfer_info = fetch_transfer_info(team_id, current_gw)

    history_df = build_player_history_df(bootstrap, refresh=refresh)
    models     = train_models(history_df)
    rmse_map   = get_rmse_from_models(models)

    # ── Phase 1 v5 prediction pipeline ──────────────────────────────
    pred_df = build_current_features(
        bootstrap, fixtures_df, history_df,
        models, current_gw, my_player_ids=my_player_ids
    )

    # Component blend
    component_models = train_component_models(history_df)
    pred_df = predict_component_pts(component_models, pred_df)
    direct_w = 1.0 - COMPONENT_BLEND_WEIGHT
    pred_df["predicted_pts"] = (
        direct_w * pred_df["predicted_pts"] +
        COMPONENT_BLEND_WEIGHT * pred_df["pts_from_components"]
    ).round(2)

    # Rotation-risk adjusted expected_pts + quantile bounds
    pred_df = compute_expected_pts(pred_df)

    # Price predictions
    price_model = train_price_model(history_df)
    pred_df     = add_price_predictions(price_model, pred_df)

    # ── Phase 2 context ──────────────────────────────────────────────
    custom_diff     = build_custom_difficulty(history_df, bootstrap)
    team_form_map   = build_team_form(history_df, bootstrap)
    opp_scoring_map = build_opponent_scoring_map(history_df)
    cs_prob_map     = build_cs_probability_map(history_df)
    chip_info       = build_chip_status(team_id, bootstrap, fixtures_df, current_gw)

    fixture_run_df = build_fixture_run(
        bootstrap, fixtures_df, current_gw,
        custom_difficulty=custom_diff,
        gw_lookahead=FIXTURE_LOOKAHEAD,
    )
    enriched_df = build_player_fixture_scores(
        pred_df, fixture_run_df, current_gw,
        team_form_map, opp_scoring_map,
        FIXTURE_LOOKAHEAD,
        cs_probability_map=cs_prob_map,
    )

    my_team = enriched_df[enriched_df["player_id"].isin(my_player_ids)].copy()
    others  = enriched_df[~enriched_df["player_id"].isin(my_player_ids)].copy()

    # ── Phase 4 XI optimizer ─────────────────────────────────────────
    xi_result = optimize_xi_ilp(my_team)

    return {
        "bootstrap":     bootstrap,
        "fixtures_df":   fixtures_df,
        "current_gw":    current_gw,
        "my_player_ids": my_player_ids,
        "transfer_info": transfer_info,
        "enriched_df":   enriched_df,
        "my_team":       my_team,
        "others":        others,
        "xi_result":     xi_result,
        "chip_info":     chip_info,
        "rmse_map":      rmse_map,
        "models":        models,
        "history_df":    history_df,
        "team_data":     team_data,
    }


# ─────────────────────────────────────────
# BACKEND GUARD
# ─────────────────────────────────────────

if not BACKEND_AVAILABLE:
    st.error(f"Backend import failed: `{IMPORT_ERROR}`")
    st.info("Make sure fpl_phase1_model.py through fpl_phase4_optimizer.py and config.py are present.")
    st.stop()

try:
    if refresh:
        ph = st.empty()
        with ph.container():
            render_loading_skeleton()
        with st.spinner("Refreshing data from FPL API..."):
            data = load_all_data(int(team_id_input), refresh=refresh)
        ph.empty()
    else:
        data = load_all_data(int(team_id_input), refresh=refresh)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.info("Check your team ID and internet connection, then click Refresh Data.")
    st.stop()

# ─────────────────────────────────────────
# UNPACK DATA
# ─────────────────────────────────────────

bootstrap    = data["bootstrap"]
fixtures_df  = data["fixtures_df"]
current_gw   = data["current_gw"]
transfer_info= data["transfer_info"]
enriched_df  = data["enriched_df"]
my_team      = data["my_team"]
others       = data["others"]
xi_result    = data["xi_result"]
chip_info    = data["chip_info"]
rmse_map     = data["rmse_map"]
history_df   = data["history_df"]
team_data    = data["team_data"]

schema_issues = verify_runtime_schema(my_team, others, fixtures_df)
if schema_issues:
    st.error("Runtime data schema issue detected.")
    for msg in schema_issues:
        st.caption(msg)
    st.stop()

bank_balance     = bank_override if bank_override > 0 else transfer_info["bank_balance"]
transfers_made   = transfer_info["transfers_made"]
available_chips  = chip_info.get("available_chips", [])
triple_captain   = "Triple Captain" in available_chips
bench_boost      = "Bench Boost" in available_chips

players_raw  = bootstrap["elements"]
news_map     = {p["id"]: p.get("news", "") for p in players_raw}
chance_map   = {p["id"]: p.get("chance_of_playing_next_round") for p in players_raw}
ownership_map= {p["id"]: float(p.get("selected_by_percent", 0)) for p in players_raw}

teams_df       = pd.DataFrame(bootstrap["teams"])
team_name_map  = teams_df.set_index("id")["name"].to_dict()
team_short_map = teams_df.set_index("id")["short_name"].to_dict()
player_face_map, team_badge_map = build_asset_maps(bootstrap)

# Attach visual assets
for df in (my_team, others, enriched_df):
    if "player_id" in df.columns:
        if "player_face" not in df.columns:
            df["player_face"] = df["player_id"].map(player_face_map).fillna("")
        if "team_badge" not in df.columns:
            df["team_badge"]  = df["team_id"].map(team_badge_map).fillna("")

# Squad value breakdown (uses sell_price — Phase 3 fix)
value_breakdown = get_squad_value_breakdown(my_team, bootstrap, team_data)
squad_sell_value = float(value_breakdown["sell_price"].sum()) if not value_breakdown.empty else float(my_team["price"].sum())


# ─────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────

if page == "Home":
    render_section_header("Decision Snapshot")

    # xPts total (expected_pts aware)
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

    try:
        home_ilp_1     = cached_ilp_transfers(my_team, others, float(bank_balance), n_transfers=1)
        home_roll      = get_rolling_transfer_advice(my_team, others, bank_balance, transfers_made, chip_info, current_gw, ilp_result=home_ilp_1)
        home_hits      = get_hit_transfer_analysis(my_team, others, bank_balance, transfers_made)
        transfer_call  = str(home_roll.get("recommendation", "HOLD"))
        transfer_conf  = compute_transfer_confidence(transfer_call, home_ilp_1, home_hits)
        transfer_reasons = home_roll.get("reasons", [])
        best_total_ev  = float(home_roll.get("best_total_ev", home_ilp_1.get("total_gain", 0.0)) or 0.0)
    except Exception:
        transfer_call = "HOLD"; transfer_conf = 52.0; transfer_reasons = []; best_total_ev = 0.0

    # Captain: use captain_ev if present (Phase 2 v5)
    cap_df_home = my_team.copy()
    blank_mask_home = (cap_df_home["is_blank_next_gw"].fillna(False).astype(bool)
                       if "is_blank_next_gw" in cap_df_home.columns
                       else pd.Series(False, index=cap_df_home.index))
    cap_pool_home = cap_df_home[~blank_mask_home]
    if not cap_pool_home.empty:
        if "captain_ev" in cap_pool_home.columns:
            top_cap_home = cap_pool_home.nlargest(1, "captain_ev").iloc[0]
            captain_return = float(top_cap_home.get("captain_ev", 0.0))
        else:
            cap_pool_home["_xpts_score"] = cap_pool_home.apply(lambda r: xpts_captain_score(r, triple_captain), axis=1)
            top_cap_home = cap_pool_home.nlargest(1, "_xpts_score").iloc[0]
            captain_return = float(_xpts(top_cap_home)) * 2.0
        captain_pick = str(top_cap_home["player_name"])
    else:
        captain_pick = "No clear captain"; captain_return = 0.0

    primary_map = {
        "USE NOW":   "Make the transfer now",
        "BORDERLINE":"Wait for final team news before moving",
        "HOLD":      "Roll your transfer this week",
        "ROLL":      "Roll your transfer this week",
    }
    render_decision_banner(
        title=f"GW{current_gw+1} Decision Snapshot",
        primary_action=f"{primary_map.get(transfer_call, transfer_call)} · Captain {captain_pick}",
        confidence=float(np.clip((transfer_conf + 64.0) / 2.0, 35, 90)),
        reasons=[
            f"xPts projected XI: {projected_xi:.1f} pts (rotation-adjusted)",
            f"Expected captain return: {captain_return:.1f}",
            transfer_reasons[0] if transfer_reasons else f"Bench cover: {bench_cover:.1f} pts",
        ],
        risk_level="Low" if transfer_call == "USE NOW" else "Medium" if transfer_call == "BORDERLINE" else "High",
    )
    render_stat_cards([
        {"label": "xPts Score",        "value": f"{projected_xi:.1f}",      "delta": "Optimised XI (expected_pts)",          "tone": "positive"},
        {"label": "Transfer Decision", "value": transfer_call,               "delta": f"Signal confidence {transfer_conf:.0f}%","tone": "neutral"},
        {"label": "Captain Pick",      "value": captain_pick,                "delta": f"Expected return {captain_return:.1f}", "tone": "positive"},
        {"label": "Best Total EV",     "value": f"+{best_total_ev:.2f}",     "delta": "xPts + price movement EV",             "tone": "positive" if best_total_ev > 0 else "neutral"},
    ])

    render_section_header("Top Risks")
    risk_items = []
    for _, row in my_team.iterrows():
        pid  = int(row.get("player_id", 0))
        name = str(row.get("player_name", "?"))
        chance = chance_map.get(pid)
        p_full = float(row.get("p_plays_full", 1.0) or 1.0)
        if chance is not None and float(chance) < 85:
            sev   = 100 - float(chance)
            level = "High" if float(chance) < 60 else "Medium"
            risk_items.append({"Risk": f"{name}: availability {int(chance)}%", "Level": level,
                                "Why it matters": news_map.get(pid, "") or "Low chance this GW.", "score": sev})
        elif p_full < 0.75:
            risk_items.append({"Risk": f"{name}: rotation risk p={p_full:.0%}", "Level": "Medium",
                                "Why it matters": "Phase 1 model flags rotation concern.", "score": (1 - p_full) * 60})
        if bool(row.get("is_blank_next_gw", False)):
            risk_items.append({"Risk": f"{name}: blank fixture", "Level": "High",
                                "Why it matters": "Potential zero this GW.", "score": 35})
    if bench_cover < 6.0:
        risk_items.append({"Risk": f"Bench depth low ({bench_cover:.1f} xPts)", "Level": "Medium" if bench_cover >= 4 else "High",
                            "Why it matters": "Auto-sub upside limited.", "score": 20 if bench_cover >= 4 else 30})
    if risk_items:
        risk_df = (pd.DataFrame(risk_items).sort_values(["score", "Level"], ascending=[False, True])
                   .drop(columns=["score"]).head(4))
        render_insight_table(risk_df, row_density="compact")
    else:
        st.success("No major risks detected.")

    with st.expander("Deadline & Freshness", expanded=False):
        next_event   = next((e for e in bootstrap.get("events", []) if int(e.get("id", 0)) == current_gw + 1), None)
        deadline_raw = next_event.get("deadline_time", "") if next_event else ""
        deadline_ts  = pd.to_datetime(deadline_raw, utc=True, errors="coerce") if deadline_raw else pd.NaT
        now_utc      = pd.Timestamp.utcnow()
        if pd.notna(deadline_ts):
            hours_left     = float((deadline_ts - now_utc).total_seconds() / 3600.0)
            deadline_state = "Passed" if hours_left < 0 else "Urgent" if hours_left <= 6 else "Soon" if hours_left <= 24 else "Comfortable"
            deadline_text  = deadline_ts.strftime("%Y-%m-%d %H:%M UTC")
            hours_text     = "Closed" if hours_left < 0 else f"{hours_left:.1f}h"
        else:
            deadline_state = "Unknown"; deadline_text = "Unknown"; hours_text = "N/A"
        d1, d2, d3 = st.columns(3)
        d1.metric("Next Deadline",  deadline_text, f"GW{current_gw+1} · {deadline_state}")
        d2.metric("Time Remaining", hours_text,    "Until deadline")
        refresh_age_min = max(0.0, (datetime.now() - last_refresh_dt).total_seconds() / 60.0)
        d3.metric("Last Refresh", last_refresh_dt.strftime("%H:%M:%S"), f"{'Fresh' if refresh_age_min <= 5 else 'Stale'} · {refresh_age_min:.0f}m ago")

    st.caption("Open My Squad, Transfer Planner, and Captain Picker for full detail.")


# ─────────────────────────────────────────
# MY SQUAD PAGE
# ─────────────────────────────────────────

elif page == "My Squad":
    render_section_header(f"GW{current_gw} Completed → Optimized for GW{current_gw+1}")

    # Use sell_price (Phase 3 v4 fix)
    squad_val  = squad_sell_value
    xpts_total = (
        float(xi_result["starting_xi"].apply(lambda r: _xpts(r), axis=1).sum())
        if xi_result else float(my_team.apply(lambda r: _xpts(r), axis=1).sum())
    )
    pred_total = (xi_result["starting_xi"]["predicted_pts"].sum()
                  if xi_result else my_team["predicted_pts"].sum())

    # Score range (v5: 3-tuple, uses pts_low/pts_high)
    if xi_result:
        lo, hi, ci_label = compute_score_range(xi_result["starting_xi"], rmse_map)
    else:
        lo, hi, ci_label = 0, 0, "N/A"

    ft_label     = "Free Transfer" if transfers_made == 0 else "Used"
    risk_players = int((my_team.get("blank_gws", pd.Series(0, index=my_team.index)) > 0).sum()) + int(
        (my_team["player_id"].map(chance_map).fillna(100) < 75).sum()
    )
    bench_cover = float(xi_result["bench"].apply(lambda r: _xpts(r), axis=1).sum()) if xi_result and not xi_result["bench"].empty else 0.0
    bench_sub   = float(xi_result.get("bench_auto_sub_score", 0.0)) if xi_result else 0.0

    confidence = float(np.clip(84 - risk_players * 5, 35, 90))
    render_decision_banner(
        title="My Squad Decision",
        primary_action=f"Start {xi_result.get('formation', 'Best XI') if xi_result else 'Best XI'}",
        confidence=confidence,
        reasons=[
            f"xPts projection {xpts_total:.1f} (range {lo}–{hi}, {ci_label})",
            f"Bench xPts cover {bench_cover:.1f} | Auto-sub EV {bench_sub:.2f}",
            f"Risk flags on {risk_players} players",
        ],
        risk_level="Low" if risk_players <= 2 else "Medium" if risk_players <= 4 else "High",
    )
    render_stat_cards([
        {"label": "xPts (rot-adj)",    "value": f"{xpts_total:.1f}",    "delta": "Expected pts (expected_pts)",    "tone": "positive"},
        {"label": "Pred Pts (raw)",    "value": f"{pred_total:.1f}",    "delta": "Raw predicted_pts",              "tone": "neutral"},
        {"label": "Score Range",       "value": f"{lo}–{hi}",           "delta": ci_label,                        "tone": "warning"},
        {"label": "Players at Risk",   "value": str(risk_players),      "delta": "Blanks + low availability",     "tone": "warning" if risk_players <= 3 else "danger"},
        {"label": "Bench xPts Cover",  "value": f"{bench_cover:.1f}",   "delta": f"Auto-sub EV {bench_sub:.2f}", "tone": "neutral"},
        {"label": "Squad Sell Value",  "value": f"£{squad_val:.1f}M",   "delta": "Sell price (after FPL rule)",   "tone": "positive"},
    ])

    st.divider()

    if xi_result:
        render_section_header("Optimal Starting XI")
        xi    = xi_result["starting_xi"]
        cap   = xi_result["captain"]
        vc    = xi_result["vice_captain"]
        bench = xi_result["bench"]

        xi_cards = xi.copy()
        if "player_face" not in xi_cards.columns:
            xi_cards["player_face"] = xi_cards["player_id"].map(player_face_map)
        if "team_badge" not in xi_cards.columns:
            xi_cards["team_badge"]  = xi_cards["team_id"].map(team_badge_map)

        st.markdown(
            build_lineup_board_html(xi_cards, int(cap["player_id"]), int(vc["player_id"])),
            unsafe_allow_html=True,
        )
        l1, l2, l3, l4 = st.columns(4)
        l1.markdown("`8+ xPts` elite")
        l2.markdown("`5-7 xPts` strong")
        l3.markdown("`3-4 xPts` playable")
        l4.markdown("`<3 xPts` risky · `C` captain · `VC` vice · 📈📉 price")

        alt_forms = score_all_formations(my_team)
        if alt_forms:
            with st.expander("Formation alternatives", expanded=False):
                alt_df = pd.DataFrame(alt_forms).head(5).copy()
                # xpts column added in Phase 4 v5
                has_xpts_col = "xpts" in alt_df.columns
                rename_map = {"formation": "Formation", "pred_pts": "Predicted Pts",
                               "combined": "Combined Score"}
                if has_xpts_col:
                    rename_map["xpts"] = "xPts (expected)"
                alt_df = alt_df.rename(columns=rename_map)
                best = float(alt_df["Predicted Pts"].max()) if not alt_df.empty else 0.0
                alt_df["Delta vs Best"] = alt_df["Predicted Pts"] - best
                render_insight_table(alt_df, default_sort=("Predicted Pts", False), row_density="compact",
                                     column_config={
                                         "Predicted Pts": st.column_config.NumberColumn(format="%.2f"),
                                         "Combined Score": st.column_config.NumberColumn(format="%.2f"),
                                         "Delta vs Best":  st.column_config.NumberColumn(format="%+.2f"),
                                     })

    st.divider()

    if xi_result and not xi_result["bench"].empty:
        with st.expander("Bench details", expanded=False):
            bench_cols = st.columns(4)
            for i, (_, row) in enumerate(xi_result["bench"].iterrows()):
                with bench_cols[i % 4]:
                    chance  = chance_map.get(int(row["player_id"]))
                    p_full  = float(row.get("p_plays_full", chance / 100 if chance else 1.0) or 1.0)
                    badge   = "Red" if p_full < 0.6 else "Amber" if p_full < 0.9 else "Green"
                    bench_ev_val = float(row.get("bench_ev", 0.0) or 0.0)
                    xpts_val = float(_xpts(row))
                    subtitle = (
                        f"{row['position']} · £{row['price']:.1f}M · xPts {xpts_val:.2f} · "
                        f"{badge} {p_full*100:.0f}% play prob"
                        + (f" · bench EV {bench_ev_val:.2f}" if bench_ev_val > 0 else "")
                    )
                    pchg = float(row.get("predicted_price_change", 0) or 0)
                    st.markdown(f"""
                    <div class='fpl-card'>
                        <div style='font-size:0.7rem; color:#37b6ff; font-family:Space Mono; margin-bottom:0.3rem;'>
                            #{i+1} {'Emergency GK' if row['position']=='Goalkeeper' else 'First Sub ★' if i==0 else ''}{_price_tag(pchg)}
                        </div>
                        {player_identity_html(
                            row['player_name'], row.get('team_name',''),
                            row.get('player_face', player_face_map.get(int(row['player_id']),'')),
                            row.get('team_badge', team_badge_map.get(int(row.get('team_id',0)),'')),
                            subtitle, 'player-face-sm'
                        )}
                    </div>""", unsafe_allow_html=True)

    st.divider()

    with st.expander("Detailed squad table", expanded=False):
        render_section_header("Full Squad Stats")
        has_xpts = "expected_pts" in my_team.columns
        has_plow = "pts_low" in my_team.columns
        disp_cols = ["player_face","team_badge","player_name","position","team_name","price"]
        if has_xpts:  disp_cols.append("expected_pts")
        disp_cols += ["predicted_pts"]
        if has_plow:  disp_cols += ["pts_low","pts_high"]
        disp_cols += ["combined_score","avg_difficulty","fixture_run_label","blank_gws","double_gws"]

        disp = my_team[[c for c in disp_cols if c in my_team.columns]].copy()
        disp["sell_price"] = value_breakdown["sell_price"].values if not value_breakdown.empty else disp["price"]
        disp["ownership"]  = disp.index.map(
            lambda i: f"{ownership_map.get(int(my_team.loc[i,'player_id']),0):.1f}%"
            if i in my_team.index else "?"
        )
        pchg_col = "predicted_price_change"
        if pchg_col in my_team.columns:
            disp["Price Trend"] = my_team[pchg_col].apply(lambda x: _price_tag(float(x or 0)))

        rename = {
            "player_face":"Face","team_badge":"Badge","player_name":"Player","position":"Pos",
            "team_name":"Team","price":"Now £","sell_price":"Sell £",
            "expected_pts":"xPts","predicted_pts":"Pred","pts_low":"Q10","pts_high":"Q90",
            "combined_score":"5GW","avg_difficulty":"Diff","fixture_run_label":"Run",
            "blank_gws":"Blanks","double_gws":"DGWs","ownership":"Owned%",
        }
        disp = disp.rename(columns={k:v for k,v in rename.items() if k in disp.columns})
        sort_col = "xPts" if "xPts" in disp.columns else "Pred"
        st.dataframe(disp.sort_values(sort_col, ascending=False), use_container_width=True, hide_index=True,
                     column_config={
                         "Face": st.column_config.ImageColumn("Face", width="small"),
                         "Badge": st.column_config.ImageColumn("Badge", width="small"),
                         "xPts": st.column_config.NumberColumn(format="%.2f"),
                         "Pred": st.column_config.NumberColumn(format="%.2f"),
                         "Now £": st.column_config.NumberColumn(format="£%.1f"),
                         "Sell £": st.column_config.NumberColumn(format="£%.1f"),
                     })

        injury_players = [
            (row["player_name"], chance_map.get(int(row["player_id"])), news_map.get(int(row["player_id"]), ""))
            for _, row in my_team.iterrows()
            if chance_map.get(int(row["player_id"])) is not None and chance_map.get(int(row["player_id"])) < 100
        ]
        if injury_players:
            st.divider()
            render_section_header("Injury & Availability")
            inj_df = pd.DataFrame([
                {"Player": n, "Chance %": int(c or 0),
                 "Urgency": "High" if (c or 0) < 60 else "Medium" if (c or 0) < 85 else "Low",
                 "Notes": news or "No news"}
                for n, c, news in injury_players
            ]).sort_values("Chance %")
            render_insight_table(inj_df, default_sort=("Chance %", True), row_density="compact")


# ─────────────────────────────────────────
# FIXTURE PLANNER PAGE  (unchanged logic, same as v1)
# ─────────────────────────────────────────

elif page == "Fixture Planner":
    render_section_header("Fixture Difficulty Heatmap")

    teams_df2     = pd.DataFrame(bootstrap["teams"])
    all_team_names= teams_df2["name"].tolist()
    future_gws    = sorted(int(gw) for gw in fixtures_df["event"].dropna().unique() if int(gw) >= current_gw + 1)
    if not future_gws:
        st.warning("No upcoming gameweeks found.")
        st.stop()
    default_start = future_gws[0]
    default_end   = future_gws[min(len(future_gws)-1, 4)]

    preset_map = {
        "My Squad Teams": sorted(my_team["team_name"].dropna().unique().tolist()),
        "Top 6": [t for t in ["Arsenal","Chelsea","Liverpool","Man City","Man Utd","Tottenham"] if t in all_team_names],
        "Promoted / Budget": [t for t in ["Burnley","Leeds","Sunderland"] if t in all_team_names],
        "Custom": [],
    }
    col1, col2, col3, col4 = st.columns([2, 1.4, 1, 1.2])
    with col1:
        position_filter = st.multiselect("Filter by position", ["Goalkeeper","Defender","Midfielder","Forward"], default=["Midfielder","Forward"])
    with col3:
        show_all_teams = st.toggle("Show all 20 teams", value=False)
        gw_start, gw_end = st.select_slider("GW range", options=future_gws, value=(default_start, default_end))
    with col2:
        preset = st.selectbox("Team preset", list(preset_map.keys()))
        custom_team_selection = st.multiselect("Custom teams", all_team_names, default=preset_map.get(preset, []), disabled=show_all_teams)
    with col4:
        sort_mode = st.selectbox("Sort mode", ["Easiest overall","Easiest next 2","Blank risk"])

    gws     = [gw for gw in future_gws if gw_start <= gw <= gw_end]
    gw_count= len(gws)
    if gw_count == 0:
        st.info("Selected GW range contains no fixtures.")
        st.stop()

    if show_all_teams:
        display_teams = teams_df2["name"].tolist()
        team_ids      = teams_df2["id"].tolist()
    else:
        chosen = custom_team_selection or preset_map.get(preset, [])
        if chosen:
            ft = teams_df2[teams_df2["name"].isin(chosen)]
            team_ids = ft["id"].tolist(); display_teams = ft["name"].tolist()
        else:
            filt = my_team[my_team["position"].isin(position_filter)] if position_filter else my_team
            team_ids = filt["team_id"].unique().tolist()
            display_teams = [team_name_map.get(tid, str(tid)) for tid in team_ids]

    matrix      = []
    hover_meta  = []
    cell_labels = []
    blank_counts= []
    avg_diffs   = []
    gw_labels   = [f"GW{g}" for g in gws]

    for tid in team_ids:
        row_diffs = []; row_hover = []; row_meta = []; row_labels = []; blanks = 0
        for gw in gws:
            gw_fix = fixtures_df[fixtures_df["event"] == gw]
            home = gw_fix[gw_fix["team_h"] == tid]
            away = gw_fix[gw_fix["team_a"] == tid]
            if not home.empty:
                opp_id = int(home.iloc[0]["team_a"]); opp = team_name_map.get(opp_id, "?")
                opp_s  = team_short_map.get(opp_id, opp[:3].upper()); diff = int(home.iloc[0]["team_h_difficulty"])
                row_diffs.append(diff); row_labels.append(f"{opp_s} (H)<br><b>{diff}</b>")
                row_hover.append(f"vs {opp} (H) · {diff}"); row_meta.append([opp, "", "H", diff])
            elif not away.empty:
                opp_id = int(away.iloc[0]["team_h"]); opp = team_name_map.get(opp_id, "?")
                opp_s  = team_short_map.get(opp_id, opp[:3].upper()); diff = int(away.iloc[0]["team_a_difficulty"])
                row_diffs.append(diff); row_labels.append(f"{opp_s} (A)<br><b>{diff}</b>")
                row_hover.append(f"vs {opp} (A) · {diff}"); row_meta.append([opp, "", "A", diff])
            else:
                row_diffs.append(0); row_labels.append("<b>Blank</b>")
                row_hover.append("Blank"); row_meta.append(["Blank","","-",0]); blanks += 1
        nb = [d for d in row_diffs if d > 0]
        avg_diffs.append(float(np.mean(nb)) if nb else 6.0)
        matrix.append(row_diffs); hover_meta.append(row_meta)
        cell_labels.append(row_labels); blank_counts.append(blanks)

    if matrix:
        next2 = []
        for row in matrix:
            nb2 = [d for d in row[:2] if d > 0]
            next2.append(float(np.mean(nb2)) if nb2 else 6.0)
        if sort_mode == "Easiest next 2":
            order = sorted(range(len(display_teams)), key=lambda i: (next2[i], blank_counts[i], avg_diffs[i]))
        elif sort_mode == "Blank risk":
            order = sorted(range(len(display_teams)), key=lambda i: (blank_counts[i], avg_diffs[i]))
        else:
            order = sorted(range(len(display_teams)), key=lambda i: (avg_diffs[i], blank_counts[i]))
        display_teams = [display_teams[i] for i in order]
        matrix     = [matrix[i] for i in order]
        hover_meta = [hover_meta[i] for i in order]
        cell_labels= [cell_labels[i] for i in order]
        avg_diffs  = [avg_diffs[i] for i in order]
        blank_counts=[blank_counts[i] for i in order]
        next2      = [next2[i] for i in order]

        render_stat_cards([
            {"label": "GW Window",   "value": str(gw_count),             "delta": f"GW{gw_start}–GW{gw_end}", "tone": "neutral"},
            {"label": "Best Run",    "value": display_teams[0],          "delta": f"{avg_diffs[0]:.2f} avg diff", "tone": "positive"},
            {"label": "Worst Run",   "value": display_teams[-1],         "delta": f"{avg_diffs[-1]:.2f} avg diff","tone": "danger"},
            {"label": "Total Blanks","value": str(sum(blank_counts)),     "delta": f"{gw_count} GW window",   "tone": "warning" if sum(blank_counts) > 0 else "positive"},
        ])

        ranked     = pd.DataFrame({"Team":display_teams,"Avg Diff":avg_diffs,"Next2 Diff":next2,"Blanks":blank_counts})
        ranked["Swing"] = ranked["Avg Diff"] - ranked["Next2 Diff"]
        target_now = ranked.nsmallest(3, ["Next2 Diff","Avg Diff"])
        avoid_now  = ranked.nlargest(3, ["Next2 Diff","Avg Diff"])
        swing_row  = ranked.sort_values("Swing", ascending=False).iloc[0]
        render_decision_banner(
            title="Fixture Decision",
            primary_action=f"Target now: {', '.join(target_now['Team'].head(2).tolist())}",
            confidence=compute_fixture_decision_confidence(avg_diffs, next2, blank_counts),
            reasons=[
                f"Avoid now: {', '.join(avoid_now['Team'].head(2).tolist())}",
                f"Best 2-GW swing: {swing_row['Team']} ({swing_row['Swing']:+.2f})",
                f"Sort mode: {sort_mode}",
            ],
            risk_level="Medium" if sum(blank_counts) > 0 else "Low",
        )

        colorscale = [[0,"#ffffff"],[0.01,"#ffffff"],[0.20,"#1f8f65"],[0.40,"#27e8a7"],[0.58,"#ffb547"],[0.78,"#ff7a67"],[1.00,"#ff5d73"]]
        fig = go.Figure(data=go.Heatmap(
            z=matrix, x=gw_labels, y=display_teams,
            text=cell_labels, customdata=hover_meta,
            texttemplate="%{text}",
            textfont=dict(size=10, color="#101217", family="Space Mono"),
            hovertemplate="<b>%{y}</b><br>%{x}: %{customdata[0]} (%{customdata[2]})<br>Difficulty: %{customdata[3]}<extra></extra>",
            colorscale=colorscale, zmin=0, zmax=5, xgap=2, ygap=2, showscale=True,
            colorbar=dict(title=dict(text="Difficulty",font=dict(color="#37b6ff")),
                          tickvals=[1,2,3,4,5], ticktext=["Easy","2","3","4","Hard"],
                          tickfont=dict(color="#eaf2ff")),
        ))
        fig.update_layout(**PLOTLY_THEME, height=max(380, len(display_teams)*44+120),
                          margin=dict(l=10,r=10,t=52,b=30),
                          title=dict(text=f"Fixture Radar · GW{gws[0]} to GW{gws[-1]}",font=dict(color="#37b6ff",size=13,family="Space Mono")),
                          dragmode=False)
        fig.update_yaxes(autorange="reversed", fixedrange=True)
        fig.update_xaxes(fixedrange=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False,"responsive":True,"scrollZoom":False,"doubleClick":False})
        st.caption("Each cell shows opponent + venue (H/A) and difficulty. Blanks are white.")

        render_section_header("Two-Team Compare")
        compare_teams = st.multiselect("Pick two teams", display_teams,
                                       default=display_teams[:2] if len(display_teams) >= 2 else display_teams,
                                       max_selections=2, key="fixture_compare_teams")
        if len(compare_teams) == 2:
            a_idx = display_teams.index(compare_teams[0]); b_idx = display_teams.index(compare_teams[1])
            cmp_rows = []
            for j, gw in enumerate(gws):
                a_d = matrix[a_idx][j] if j < len(matrix[a_idx]) else 0
                b_d = matrix[b_idx][j] if j < len(matrix[b_idx]) else 0
                cmp_rows.append({"GW": f"GW{gw}",
                                  f"{compare_teams[0]} Diff": a_d if a_d > 0 else "BLK",
                                  f"{compare_teams[1]} Diff": b_d if b_d > 0 else "BLK",
                                  "Delta (A-B)": (a_d-b_d) if (a_d > 0 and b_d > 0) else np.nan})
            render_insight_table(pd.DataFrame(cmp_rows), row_density="compact",
                                 column_config={"Delta (A-B)": st.column_config.NumberColumn(format="%+.1f")})
    else:
        st.info("No teams available. Adjust filters.")

    st.divider()
    render_section_header("Top Transfer Targets by Fixture Run")
    pos_tab = st.tabs(["Goalkeeper","Defender","Midfielder","Forward"])
    for pos, tab in zip(["Goalkeeper","Defender","Midfielder","Forward"], pos_tab):
        with tab:
            cols_avail = [c for c in ["player_face","team_badge","player_name","team_name","price",
                                       "predicted_pts","expected_pts","avg_difficulty",
                                       "fixture_run_label","blank_gws","double_gws",
                                       "combined_score","value_score","predicted_price_change"]
                          if c in others.columns]
            top = (others[others["position"]==pos].nlargest(10,"combined_score")[cols_avail].copy())
            top["ownership"] = top.index.map(lambda i: f"{ownership_map.get(int(others.loc[i,'player_id']),0):.1f}%" if i in others.index else "?")
            rename = {"player_face":"Face","team_badge":"Badge","player_name":"Player","team_name":"Team",
                      "price":"£","predicted_pts":"Pred","expected_pts":"xPts","avg_difficulty":"Avg Diff",
                      "fixture_run_label":"Run","blank_gws":"Blanks","double_gws":"DGWs",
                      "combined_score":"5GW","value_score":"Val/£","predicted_price_change":"Price Δ","ownership":"Owned%"}
            st.dataframe(top.rename(columns={k:v for k,v in rename.items() if k in top.columns}),
                         use_container_width=True, hide_index=True,
                         column_config={"Face": st.column_config.ImageColumn("Face",width="small"),
                                        "Badge": st.column_config.ImageColumn("Badge",width="small"),
                                        "xPts": st.column_config.NumberColumn(format="%.2f"),
                                        "Pred": st.column_config.NumberColumn(format="%.2f")})


# ─────────────────────────────────────────
# TRANSFER PLANNER PAGE  (v2: total_ev, price tags, urgency, horizon plan, double hit)
# ─────────────────────────────────────────

elif page == "Transfer Planner":
    render_section_header(
        f"Bank: £{bank_balance:.1f}M | "
        f"{'1 Free Transfer' if transfers_made == 0 else 'Free Transfer Used'}"
        f"{' | Hit analysis available' if transfers_made > 0 else ''}"
    )

    with st.spinner("Computing optimal transfers..."):
        ilp_1 = cached_ilp_transfers(my_team, others, float(bank_balance), n_transfers=1)
        ilp_2 = cached_ilp_transfers(my_team, others, float(bank_balance), n_transfers=2)
        roll  = get_rolling_transfer_advice(my_team, others, bank_balance, transfers_made,
                                            chip_info, current_gw, ilp_result=ilp_1)
        hit_transfers = get_hit_transfer_analysis(my_team, others, bank_balance, transfers_made)

    rec      = roll["recommendation"]
    rec_conf = compute_transfer_confidence(rec, ilp_1, hit_transfers)
    rec_risk = "Low" if rec == "USE NOW" else "Medium" if rec == "BORDERLINE" else "High"

    # v2: use total_ev where available
    total_ev_1 = float(ilp_1.get("total_ev", ilp_1.get("total_gain", 0.0)) or 0.0)
    total_ev_2 = float(ilp_2.get("total_ev", ilp_2.get("total_gain", 0.0)) or 0.0)

    render_decision_banner(
        title="Transfer Decision",
        primary_action=rec,
        confidence=rec_conf,
        reasons=roll.get("reasons", []),
        risk_level=rec_risk,
    )
    render_stat_cards([
        {"label": "Recommendation",    "value": rec,                           "delta": f"Confidence {rec_conf:.0f}%",       "tone": "positive" if rec=="USE NOW" else "warning" if rec=="BORDERLINE" else "neutral"},
        {"label": "1FT Total EV",      "value": f"{total_ev_1:+.2f}",          "delta": "xPts + price movement value",       "tone": "positive"},
        {"label": "2FT Total EV",      "value": f"{total_ev_2:+.2f}",          "delta": "Best double transfer EV",           "tone": "neutral"},
        {"label": "Hit Options",       "value": str(len(hit_transfers or [])), "delta": "Break-even+ hit candidates",        "tone": "warning" if len(hit_transfers or []) else "neutral"},
    ])

    transfer_face_map  = enriched_df[["player_name","player_face"]].drop_duplicates("player_name").set_index("player_name")["player_face"].to_dict()
    transfer_tname_map = enriched_df[["player_name","team_name"]].drop_duplicates("player_name").set_index("player_name")["team_name"].to_dict()
    transfer_badge_map = enriched_df[["player_name","team_badge"]].drop_duplicates("player_name").set_index("player_name")["team_badge"].to_dict()

    # Recommendation stack
    stack_candidates = []
    if ilp_1.get("transfers"):
        t = ilp_1["transfers"][0]
        pchg_tag = _price_tag(float(t.get("price_change", 0) or 0))
        urg = float(t.get("urgency_score", 0) or 0) if "urgency_score" in t else 0
        stack_candidates.append({
            "id": "safe", "label": "Safe move",
            "headline": f"{t['out_name']} → {t['in_name']}{pchg_tag}{' 🔥' if urg >= 2.0 else ''}",
            "now": float(ilp_1.get("total_next_gain", 0.0)),
            "horizon": float(ilp_1.get("total_gain", 0.0)),
            "ev": total_ev_1,
            "cost": float(ilp_1.get("total_cost", 0.0)),
            "risk": ["Low variance", "Single transfer"],
            "why": [f"Run: {t.get('fixture_run','?')}", f"Total EV: {total_ev_1:+.2f}"],
        })
    if ilp_2.get("transfers"):
        stack_candidates.append({
            "id": "aggressive", "label": "Aggressive move",
            "headline": " + ".join([f"{x['out_name']} → {x['in_name']}" for x in ilp_2["transfers"][:2]]),
            "now": float(ilp_2.get("total_next_gain", 0.0)),
            "horizon": float(ilp_2.get("total_gain", 0.0)),
            "ev": total_ev_2,
            "cost": float(ilp_2.get("total_cost", 0.0)),
            "risk": ["Higher variance", "Locks flexibility"],
            "why": ["Broader fixture turn", f"Total EV: {total_ev_2:+.2f}"],
        })
    diff_df = get_differential_picks(others, bootstrap, top_n=1)
    if not diff_df.empty:
        d = diff_df.iloc[0]
        stack_candidates.append({
            "id": "differential", "label": "Differential move",
            "headline": f"Buy {d['player_name']} ({d.get('ownership_pct',0):.1f}% owned)",
            "now": float(_xpts(d)), "horizon": float(d.get("combined_score", 0.0)),
            "ev": float(_xpts(d)), "cost": float(d.get("price", 0.0)),
            "risk": ["Low ownership volatility", "Minutes risk"],
            "why": [f"Run: {d.get('fixture_run_label','?')}", f"Diff score: {d.get('differential_score',0):.2f}"],
        })

    if stack_candidates:
        render_section_header("Recommendation Stack")
        risk_pen = {"safe": 0.0, "aggressive": 0.6, "differential": 0.4}
        rec_align = {
            "USE NOW":   {"safe": 0.4, "aggressive": 0.5, "differential": 0.2},
            "BORDERLINE":{"safe": 0.5, "aggressive": 0.1, "differential": 0.25},
            "HOLD":      {"safe":-0.2, "aggressive":-0.6, "differential":-0.3},
            "ROLL":      {"safe":-0.2, "aggressive":-0.6, "differential":-0.3},
        }
        align_map = rec_align.get(rec, {})
        ranked_stack = sorted(stack_candidates,
                              key=lambda c: 1.2*c["now"] + 0.9*c["ev"] - risk_pen.get(c["id"],0.3) + align_map.get(c["id"],0.0),
                              reverse=True)
        primary = ranked_stack[0]
        conf_p  = float(np.clip(60 + 7 * max(0.0, primary["ev"] - primary["now"]), 40, 88))
        render_recommendation_card(
            headline=f"Recommended: {primary['label']} · {primary['headline']}",
            impact_now=primary["now"], impact_horizon=primary["horizon"],
            confidence=conf_p, risk_notes=primary["risk"],
            supporting_points=[*primary["why"], f"Net cost: {primary['cost']:+.1f}M"],
        )
        if len(ranked_stack) > 1:
            with st.expander("Alternative options", expanded=False):
                for c in ranked_stack[1:]:
                    conf_c = float(np.clip(60 + 7 * max(0.0, c["ev"] - c["now"]), 40, 88))
                    render_recommendation_card(
                        headline=f"{c['label']}: {c['headline']}",
                        impact_now=c["now"], impact_horizon=c["horizon"],
                        confidence=conf_c, risk_notes=c["risk"],
                        supporting_points=[*c["why"], f"Net cost: {c['cost']:+.1f}M"],
                    )

    # Horizon plan section (Phase 3 v5)
    with st.expander("📅 Multi-GW Horizon Plan (2-GW lookahead)", expanded=False):
        try:
            horizon_plans = get_horizon_transfer_plan(my_team, others, enriched_df, bank_balance)
            if not horizon_plans:
                st.info("No viable 2-GW transfer sequence found.")
            else:
                for i, plan in enumerate(horizon_plans[:3], 1):
                    st.markdown(f"**Plan {i}** — Total EV: `+{plan['total_horizon_ev']:.2f}`")
                    h1, h2 = st.columns(2)
                    with h1:
                        pchg1 = _price_tag(float(plan.get("w1_price_change", 0) or 0))
                        st.markdown(
                            f"**GW+1:** OUT `{plan['w1_out']}` → IN `{plan['w1_in']}`{pchg1}  "
                            f"xPts `+{plan['w1_xpts_gain']:.2f}` · EV `+{plan['w1_total_ev']:.2f}` · "
                            f"Run `{plan['w1_run']}` · Cost `{plan['w1_cost']:+.1f}M`"
                        )
                    with h2:
                        if plan["w2_in"] != "—":
                            st.markdown(
                                f"**GW+2:** OUT `{plan['w2_out']}` → IN `{plan['w2_in']}`  "
                                f"xPts `+{plan['w2_xpts_gain']:.2f}` · EV `+{plan['w2_total_ev']:.2f}` · "
                                f"Run `{plan['w2_run']}`"
                            )
                        else:
                            st.markdown("**GW+2:** No improvement after GW+1 transfer.")
                    if i < len(horizon_plans[:3]):
                        st.divider()
        except Exception as e:
            st.caption(f"Horizon plan unavailable: {e}")

    # Main transfer tabs
    tab1, tab2, tab3, tab4 = st.tabs(["1 Transfer", "2 Transfers", "Take a Hit", "Double Hit (-8pt)"])

    def _transfer_meta(tr: dict, fallback_next: float = 0.0):
        nxt   = float(tr.get("next_gain", fallback_next) or 0.0)
        hor   = float(tr.get("gain", 0.0) or 0.0)
        ev    = float(tr.get("total_ev", nxt) or nxt)
        blank = bool(tr.get("is_blank", False))
        cost  = float(tr.get("cost_diff", 0.0) or 0.0)
        dgw   = float(tr.get("double_gws", 0) or 0) > 0
        urg   = float(tr.get("urgency_score", 0) or 0)
        conf  = float(np.clip(58 + 10 * ev + 5 * nxt - (12 if blank else 0) - (4 if cost > 1.5 else 0) + (3 if dgw else 0), 35, 92))
        risk  = "High" if blank else "Medium" if cost > 2.0 else "Low"
        risks = (["Upcoming blank risk"] if blank else []) + (["Higher cost"] if cost > 1.5 else []) + (["No DGW upside"] if not dgw else []) + (["Role/minutes variance"] if not blank and cost <= 1.5 else [])
        return nxt, hor, ev, conf, risk, risks or ["Minutes variance"]

    with tab1:
        if ilp_1.get("transfers"):
            t = ilp_1["transfers"][0]
            pchg_tag = _price_tag(float(t.get("price_change", 0) or 0))
            urg_tag  = " 🔥 Urgent" if float(t.get("urgency_score", 0) or 0) >= 2.0 else ""
            out_team = transfer_tname_map.get(t["out_name"], "")
            in_team  = transfer_tname_map.get(t["in_name"], "")
            col_a, col_b, col_c = st.columns([2, 1, 2])
            with col_a:
                st.markdown(f"""<div class='transfer-card'><div class='kpi-label'>TRANSFER OUT</div>
                    {player_identity_html(t['out_name'],out_team,transfer_face_map.get(t['out_name'],''),
                     transfer_badge_map.get(t['out_name'],''),f"{t['position']} | EV: {t.get('total_ev',t['gain']):.2f}",'player-face-sm')}
                </div>""", unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""<div style='text-align:center; padding-top:1.5rem;'>
                    <div class='transfer-gain'>+{t.get('total_ev',t['gain']):.2f}</div>
                    <div style='font-size:0.65rem; color:#37b6ff; font-family:Space Mono;'>TOTAL EV</div>
                    <div style='font-size:0.65rem; color:#90a2be; margin-top:0.2rem;'>xPts: +{t['next_gain']:.2f}</div>
                    <div style='font-size:0.8rem; color:#90a2be;'>Cost: {t['cost_diff']:+.1f}M{pchg_tag}</div>
                </div>""", unsafe_allow_html=True)
            with col_c:
                st.markdown(f"""<div class='transfer-card' style='border-color:rgba(39,232,167,0.28);'>
                    <div class='kpi-label'>TRANSFER IN{urg_tag}</div>
                    {player_identity_html(t['in_name'],in_team,transfer_face_map.get(t['in_name'],''),
                     transfer_badge_map.get(t['in_name'],''),f"{t.get('fixture_run','?')} {'| BLANK' if t.get('is_blank') else ''} {'| DGW' if t.get('double_gws',0) else ''}",'player-face-sm')}
                </div>""", unsafe_allow_html=True)

            t_nxt, t_hor, t_ev, t_conf, t_risk, t_risks = _transfer_meta(t, float(ilp_1.get("total_next_gain",0)))
            render_recommendation_card(
                headline=f"Transfer: {t['out_name']} → {t['in_name']}{pchg_tag}",
                impact_now=t_nxt, impact_horizon=t_hor, confidence=t_conf,
                risk_notes=t_risks,
                supporting_points=[f"Total EV: {t_ev:+.2f}", f"Run: {t.get('fixture_run','?')}", f"Risk: {t_risk}"],
            )
            render_stat_cards([
                {"label": "Total EV (xPts+price)",  "value": f"{total_ev_1:+.2f}",            "delta": "Including price movement",   "tone": "positive"},
                {"label": "5GW Combined Gain",       "value": f"+{ilp_1['total_gain']:.2f}",   "delta": "Horizon impact",             "tone": "positive"},
                {"label": "Next GW xPts Gain",       "value": f"+{ilp_1['total_next_gain']:.2f}","delta": "Immediate impact",          "tone": "neutral"},
                {"label": "Net Cost",                "value": f"£{ilp_1['total_cost']:+.1f}M", "delta": "Budget delta",               "tone": "warning" if ilp_1["total_cost"] > 0 else "positive"},
            ])
        else:
            st.info("No beneficial 1-transfer found within budget.")

    with tab2:
        if ilp_2.get("transfers") and len(ilp_2["transfers"]) == 2:
            for i, t in enumerate(ilp_2["transfers"], 1):
                pchg_tag = _price_tag(float(t.get("price_change", 0) or 0))
                out_team = transfer_tname_map.get(t["out_name"], "")
                in_team  = transfer_tname_map.get(t["in_name"], "")
                col_a, col_b, col_c = st.columns([2, 1, 2])
                with col_a:
                    st.markdown(f"""<div class='transfer-card'><div class='kpi-label'>T{i} - OUT</div>
                        {player_identity_html(t['out_name'],out_team,transfer_face_map.get(t['out_name'],''),
                         transfer_badge_map.get(t['out_name'],''),t['position'],'player-face-sm')}
                    </div>""", unsafe_allow_html=True)
                with col_b:
                    ev_t = float(t.get("total_ev", t["gain"]) or t["gain"])
                    st.markdown(f"""<div style='text-align:center; padding-top:1rem;'>
                        <div class='transfer-gain'>+{ev_t:.2f}</div>
                        <div style='font-size:0.65rem; color:#37b6ff; font-family:Space Mono;'>EV</div>
                    </div>""", unsafe_allow_html=True)
                with col_c:
                    st.markdown(f"""<div class='transfer-card' style='border-color:rgba(39,232,167,0.28);'>
                        <div class='kpi-label'>T{i} - IN{pchg_tag}</div>
                        {player_identity_html(t['in_name'],in_team,transfer_face_map.get(t['in_name'],''),
                         transfer_badge_map.get(t['in_name'],''),f"{t.get('fixture_run','?')}",'player-face-sm')}
                    </div>""", unsafe_allow_html=True)
                t2_nxt, t2_hor, t2_ev, t2_conf, t2_risk, t2_risks = _transfer_meta(t, float(ilp_2.get("total_next_gain",0))/2)
                render_recommendation_card(
                    headline=f"T{i}: {t['out_name']} → {t['in_name']}{pchg_tag}",
                    impact_now=t2_nxt, impact_horizon=t2_hor, confidence=t2_conf,
                    risk_notes=t2_risks,
                    supporting_points=[f"Total EV: {t2_ev:+.2f}", f"Run: {t.get('fixture_run','?')}", f"Risk: {t2_risk}"],
                )
            render_stat_cards([
                {"label": "Combined EV",       "value": f"{total_ev_2:+.2f}",              "delta": "Both transfers EV",       "tone": "positive"},
                {"label": "Total 5GW Gain",    "value": f"+{ilp_2['total_gain']:.2f}",     "delta": "Horizon impact",          "tone": "positive"},
                {"label": "Total Next GW",     "value": f"+{ilp_2['total_next_gain']:.2f}","delta": "Immediate impact",        "tone": "neutral"},
                {"label": "Total Cost",        "value": f"£{ilp_2['total_cost']:+.1f}M",   "delta": "Budget delta",            "tone": "warning" if ilp_2["total_cost"] > 0 else "positive"},
            ])
        else:
            st.info("No beneficial 2-transfer combination found within budget.")

    with tab3:
        if not hit_transfers:
            if transfers_made == 0:
                st.success("You still have a free transfer — no hit needed.")
            else:
                st.info(f"No transfers clear the -4pt break-even threshold.")
        else:
            st.markdown("**Transfers worth a -4pt hit:**")
            for h in hit_transfers:
                pchg_tag = _price_tag(float(h.get("price_change", 0) or 0))
                ev_h = float(h.get("total_ev", h.get("xpts_gain", 0)) or h.get("xpts_gain",0))
                conf_h = float(np.clip(54 + 8 * ev_h - (10 if float(h.get("net_value",0)) < 1 else 0), 30, 88))
                c1, c2 = st.columns([3, 3])
                c1.markdown(f"**OUT** `{h['replace']}`")
                c2.markdown(f"**IN** `{h['player_in']}` · `{h['fixture_run']}`{pchg_tag}")
                render_recommendation_card(
                    headline=f"Hit: {h['replace']} → {h['player_in']}{pchg_tag}",
                    impact_now=float(h.get("xpts_gain", 0)),
                    impact_horizon=float(h.get("combined_gain", 0)),
                    confidence=conf_h,
                    risk_notes=["Hit cost -4pts", "Minutes/rotation risk"] + (["Low post-hit margin"] if float(h.get("net_value",0)) < 1 else []),
                    supporting_points=[
                        f"Total EV: {ev_h:+.2f} (incl. price)",
                        f"Net after hit: {float(h.get('net_value',0)):+.1f} pts",
                        f"Run: {h.get('fixture_run','?')}",
                    ],
                )

    with tab4:
        if transfers_made == 0:
            st.success("You have a free transfer — double hit not applicable.")
        else:
            try:
                double_hits = get_double_hit_analysis(my_team, others, bank_balance, transfers_made)
                if not double_hits:
                    st.info("No double-transfer combo justifies a -8pt hit this week.")
                else:
                    st.markdown("**Combos worth a -8pt double hit:**")
                    for i, dh in enumerate(double_hits, 1):
                        conf_dh = float(np.clip(60 + 8 * dh["net_value"], 30, 88))
                        render_recommendation_card(
                            headline=f"Option {i}: {dh['t1_out']} → {dh['t1_in']} + {dh['t2_out']} → {dh['t2_in']}",
                            impact_now=dh["total_xpts_gain"],
                            impact_horizon=dh["total_xpts_gain"],
                            confidence=conf_dh,
                            risk_notes=["-8pt hit cost", "Needs both to pay off"],
                            supporting_points=[
                                f"Combined xPts gain: +{dh['total_xpts_gain']:.2f}",
                                f"Net after -8pt hit: +{dh['net_value']:.2f}",
                                f"Total cost: {dh['total_cost']:+.1f}M",
                            ],
                        )
            except Exception as e:
                st.caption(f"Double hit analysis unavailable: {e}")


# ─────────────────────────────────────────
# PLAYER EXPLORER PAGE
# ─────────────────────────────────────────

elif page == "Player Explorer":
    render_section_header("Player Explorer")

    for k, v in [("px_pos_filter",["Midfielder","Forward"]),("px_price_range",(4.0,13.0)),
                 ("px_min_pred",2.0),("px_search",""),("px_mode","Value")]:
        if k not in st.session_state:
            st.session_state[k] = v

    r_mode, r_reset = st.columns([1, 1])
    with r_mode:
        mode = st.radio("Objective mode", ["Value","Ceiling","Safety"], horizontal=True, key="px_mode")
    with r_reset:
        if st.button("Reset filters", use_container_width=True):
            for k, v in [("px_pos_filter",["Midfielder","Forward"]),("px_price_range",(4.0,13.0)),("px_min_pred",2.0),("px_search","")]:
                st.session_state[k] = v
            st.rerun()

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        pos_filter  = st.multiselect("Position", ["Goalkeeper","Defender","Midfielder","Forward"], default=st.session_state["px_pos_filter"])
    with f2:
        price_range = st.slider("Price Range (£M)", 3.5, 15.0, st.session_state["px_price_range"], 0.5)
    with f3:
        min_pred    = st.slider("Min Predicted Pts", 0.0, 15.0, st.session_state["px_min_pred"], 0.5)
    with f4:
        search      = st.text_input("Search player name", st.session_state["px_search"])
    st.session_state.update({"px_pos_filter": pos_filter, "px_price_range": price_range, "px_min_pred": min_pred, "px_search": search})

    pool = enriched_df.copy()
    pool["ownership"] = pool["player_id"].map(ownership_map).fillna(0)
    pool["news"]      = pool["player_id"].map(news_map).fillna("")
    if pos_filter:    pool = pool[pool["position"].isin(pos_filter)]
    pool = pool[(pool["price"] >= price_range[0]) & (pool["price"] <= price_range[1]) & (pool["predicted_pts"] >= min_pred)]
    if search:        pool = pool[pool["player_name"].str.contains(search, case=False, na=False)]

    st.caption(f"{len(pool)} players shown")

    if not pool.empty:
        render_stat_cards([
            {"label": "Shown",       "value": str(len(pool)),                                         "delta": "Filtered players",         "tone": "neutral"},
            {"label": "Avg xPts",    "value": f"{float(pool.apply(lambda r: _xpts(r), axis=1).mean()):.2f}", "delta": "Expected pts avg",  "tone": "positive"},
            {"label": "Median Price","value": f"£{float(pool['price'].median()):.1f}",                "delta": "Current filter",           "tone": "neutral"},
            {"label": "Top xPts",    "value": f"{float(pool.apply(lambda r: _xpts(r), axis=1).max()):.2f}", "delta": "Highest in filter",  "tone": "positive"},
        ])

    tab_scatter, tab_bar, tab_compare, tab_table = st.tabs(["xPts vs Price","Top Differentials","Player Comparison","Full Table"])

    with tab_scatter:
        if pool.empty:
            st.info("No players match filters.")
        else:
            plot_df = pool.copy()
            plot_df["xpts_val"]   = plot_df.apply(lambda r: _xpts(r), axis=1)
            plot_df["value_index"]= (plot_df["xpts_val"] / plot_df["price"].replace(0, np.nan)).fillna(0.0)
            if "p_plays_full" in plot_df.columns:
                safety = plot_df["p_plays_full"].fillna(1.0).astype(float)
            elif "chance_of_playing" in plot_df.columns:
                safety = plot_df["chance_of_playing"].fillna(100).astype(float) / 100.0
            else:
                safety = pd.Series([1.0] * len(plot_df), index=plot_df.index)
            plot_df["objective_score"] = (
                plot_df["value_index"] if mode == "Value"
                else plot_df["xpts_val"] if mode == "Ceiling"
                else plot_df["xpts_val"] * safety
            )
            top_value = plot_df.nlargest(8, "objective_score")

            fig = px.scatter(
                plot_df, x="price", y="xpts_val",
                color="position", size="value_index", size_max=26, opacity=0.78,
                hover_name="player_name",
                hover_data={"team_name": True, "price": ":.1f", "xpts_val": ":.2f",
                             "combined_score": ":.2f", "fixture_run_label": True,
                             "ownership": ":.1f", "value_index": ":.3f"},
                labels={"price": "Price (£M)", "xpts_val": "xPts (expected_pts)"},
                color_discrete_map=POSITION_COLOR_MAP,
            )
            fig.add_hline(y=float(plot_df["xpts_val"].median()), line_dash="dot", line_color=PLOTLY_ACCENT, annotation_text="Median xPts")
            fig.add_vline(x=float(plot_df["price"].median()),    line_dash="dot", line_color=PLOTLY_ACCENT, annotation_text="Median Price")
            fig.add_trace(go.Scatter(
                x=top_value["price"], y=top_value["xpts_val"],
                mode="markers+text", text=top_value["player_name"].str.split().str[-1],
                textposition="top center", textfont=dict(size=10, color=PLOTLY_TEXT),
                marker=dict(size=12, symbol="diamond", color="rgba(39,232,167,0.15)", line=dict(color=PLOTLY_PRIMARY, width=1.8)),
                name="Top Value", hoverinfo="skip",
            ))
            squad_pool = plot_df[plot_df["player_id"].isin(data["my_player_ids"])]
            if not squad_pool.empty:
                fig.add_trace(go.Scatter(
                    x=squad_pool["price"], y=squad_pool["xpts_val"],
                    mode="markers", marker=dict(size=16, color="white", symbol="star", line=dict(color=PLOTLY_PRIMARY, width=2)),
                    name="Your Squad", hovertext=squad_pool["player_name"],
                ))
            fig.update_layout(**PLOTLY_THEME, height=470, margin=dict(l=10,r=10,t=22,b=32))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
            st.caption(f"Y-axis shows expected_pts (rotation-adjusted). Objective mode: {mode}.")

            with st.expander("View shortlist table"):
                shortlist_cols = [c for c in ["player_face","team_badge","player_name","team_name","position","price","xpts_val","value_index","objective_score","fixture_run_label","predicted_price_change"] if c in plot_df.columns]
                shortlist = top_value[shortlist_cols].rename(columns={"player_face":"Face","team_badge":"Badge","player_name":"Player","team_name":"Team","position":"Pos","price":"£","xpts_val":"xPts","value_index":"Value/£","objective_score":"Objective","fixture_run_label":"Run","predicted_price_change":"Price Δ"}).sort_values("Objective",ascending=False)
                st.dataframe(shortlist, use_container_width=True, hide_index=True,
                             column_config={"Face": st.column_config.ImageColumn("Face",width="small"),
                                            "Badge": st.column_config.ImageColumn("Badge",width="small"),
                                            "£": st.column_config.NumberColumn(format="£%.1f"),
                                            "xPts": st.column_config.NumberColumn(format="%.2f"),
                                            "Value/£": st.column_config.NumberColumn(format="%.3f")})

    with tab_bar:
        diffs = get_differential_picks(others, bootstrap, top_n=15)
        if not diffs.empty:
            fig = px.bar(
                diffs, x="differential_score", y="player_name",
                orientation="h", color="position",
                hover_data=["team_name","price","predicted_pts","ownership_pct","fixture_run_label"],
                color_discrete_map=POSITION_COLOR_MAP,
                labels={"differential_score": "Differential Score", "player_name": ""},
            )
            fig.update_layout(**PLOTLY_THEME, height=450, margin=dict(l=10,r=10,t=30,b=30))
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
            st.caption("Differential score = combined score × low ownership bonus. <15% owned only.")

    with tab_compare:
        all_names = sorted(enriched_df["player_name"].tolist())
        col1, col2 = st.columns(2)
        p1_name = col1.selectbox("Player A", all_names, index=0, key="p1")
        p2_name = col2.selectbox("Player B", all_names, index=min(1, len(all_names)-1), key="p2")
        p1 = enriched_df[enriched_df["player_name"] == p1_name]
        p2 = enriched_df[enriched_df["player_name"] == p2_name]
        if not p1.empty and not p2.empty:
            p1 = p1.iloc[0]; p2 = p2.iloc[0]
            id_col1, id_col2 = st.columns(2)
            p1_sub = f"£{p1['price']:.1f}M | xPts {_xpts(p1):.2f}"
            p2_sub = f"£{p2['price']:.1f}M | xPts {_xpts(p2):.2f}"
            id_col1.markdown(f"<div class='fpl-card'>{player_identity_html(p1['player_name'],p1['team_name'],p1.get('player_face',''),p1.get('team_badge',''),p1_sub)}</div>", unsafe_allow_html=True)
            id_col2.markdown(f"<div class='fpl-card'>{player_identity_html(p2['player_name'],p2['team_name'],p2.get('player_face',''),p2.get('team_badge',''),p2_sub)}</div>", unsafe_allow_html=True)
            metrics = [
                ("xPts (expected)",   _xpts(p1),                          _xpts(p2)),
                ("Predicted Pts",     p1["predicted_pts"],                 p2["predicted_pts"]),
                ("Combined Score",    p1["combined_score"],                p2["combined_score"]),
                ("Price £M",          p1["price"],                         p2["price"]),
                ("Avg Difficulty",    p1.get("avg_difficulty",3),          p2.get("avg_difficulty",3)),
                ("p_plays_full",      float(p1.get("p_plays_full",1.0) or 1), float(p2.get("p_plays_full",1.0) or 1)),
                ("Value Score",       p1.get("value_score",0),             p2.get("value_score",0)),
                ("Blank GWs",         p1.get("blank_gws",0),               p2.get("blank_gws",0)),
                ("Double GWs",        p1.get("double_gws",0),              p2.get("double_gws",0)),
            ]
            labels = [m[0] for m in metrics]
            vals_p1= [float(m[1] or 0) for m in metrics]
            vals_p2= [float(m[2] or 0) for m in metrics]
            fig = go.Figure()
            fig.add_trace(go.Bar(name=p1_name.split()[-1], x=labels, y=vals_p1, marker_color=PLOTLY_PRIMARY))
            fig.add_trace(go.Bar(name=p2_name.split()[-1], x=labels, y=vals_p2, marker_color=PLOTLY_ACCENT))
            fig.update_layout(**PLOTLY_THEME, barmode="group", height=380, margin=dict(l=10,r=10,t=30,b=60))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

            gws_range = list(range(current_gw+1, current_gw+1+FIXTURE_LOOKAHEAD))
            gw_data = [{"GW": f"GW{gw}",
                        f"{p1_name.split()[-1]} Opp": p1.get(f"gw{gw}_opponent","?"),
                        f"{p1_name.split()[-1]} Diff": p1.get(f"gw{gw}_difficulty","?"),
                        f"{p2_name.split()[-1]} Opp": p2.get(f"gw{gw}_opponent","?"),
                        f"{p2_name.split()[-1]} Diff": p2.get(f"gw{gw}_difficulty","?")}
                       for gw in gws_range]
            st.dataframe(pd.DataFrame(gw_data), use_container_width=True, hide_index=True)

    with tab_table:
        disp_cols = [c for c in ["player_face","team_badge","player_name","team_name","position","price",
                                  "expected_pts","predicted_pts","combined_score","value_score",
                                  "fixture_run_label","blank_gws","double_gws","predicted_price_change"]
                     if c in pool.columns]
        disp_pool = pool[disp_cols].rename(columns={
            "player_face":"Face","team_badge":"Badge","player_name":"Player","team_name":"Team",
            "position":"Pos","price":"£","expected_pts":"xPts","predicted_pts":"Pred",
            "combined_score":"5GW","value_score":"Val","fixture_run_label":"Run",
            "blank_gws":"Blanks","double_gws":"DGWs","predicted_price_change":"Price Δ",
        }).sort_values("xPts" if "xPts" in pool.rename(columns={"expected_pts":"xPts"}).columns else "Pred", ascending=False)
        st.dataframe(disp_pool, use_container_width=True, hide_index=True,
                     column_config={"Face": st.column_config.ImageColumn("Face",width="small"),
                                    "Badge": st.column_config.ImageColumn("Badge",width="small")})


# ─────────────────────────────────────────
# CAPTAIN PICKER PAGE  (v2: captain_ev, MC win_prob, captaincy differential)
# ─────────────────────────────────────────

elif page == "Captain Picker":
    render_section_header("Captain & Vice Captain Recommendation")

    chips = ["Wildcard","Free Hit","Triple Captain","Bench Boost"]
    chip_cols = st.columns(4)
    for i, chip in enumerate(chips):
        avail = chip in available_chips
        chip_cols[i].markdown(
            f"<div class='kpi-block'><div class='kpi-label'>{chip}</div>"
            f"<div style='font-size:1.2rem; font-weight:800; color:{'#27e8a7' if avail else '#ff5d73'};'>"
            f"{'Available' if avail else 'Used'}</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    cap_df = my_team.copy()
    # v2: use p_plays_full for reliability (not just chance_of_playing)
    if "p_plays_full" in cap_df.columns:
        cap_df["reliability"] = cap_df["p_plays_full"].fillna(1.0).astype(float)
    else:
        cap_df["reliability"] = cap_df["player_id"].astype("Int64").map(chance_map).fillna(100).astype(float) / 100.0

    # v2: use captain_ev when available (Phase 2 v5)
    has_cap_ev  = "captain_ev" in cap_df.columns
    cap_df["xpts_val"] = cap_df.apply(lambda r: _xpts(r), axis=1)
    cap_df["upside"]   = (cap_df["xpts_val"]
                          + 0.5 * cap_df.get("double_gws", pd.Series(0, index=cap_df.index)).fillna(0).astype(float)
                          - 0.35 * cap_df.get("blank_gws", pd.Series(0, index=cap_df.index)).fillna(0).astype(float))
    if has_cap_ev:
        # TC multiplier: captain_ev is already 2x base, multiply by 1.5 for 3x
        cap_df["_cap_sort"] = cap_df["captain_ev"] * (1.5 if triple_captain else 1.0)
    else:
        cap_df["_cap_sort"] = cap_df.apply(lambda r: xpts_captain_score(r, triple_captain), axis=1)

    blank_mask = (cap_df["is_blank_next_gw"].fillna(False).astype(bool)
                  if "is_blank_next_gw" in cap_df.columns
                  else pd.Series(False, index=cap_df.index))
    cap_df["vc_score"] = (cap_df["xpts_val"] * cap_df["reliability"]).where(~blank_mask, 0.0)
    cap_df["cap_expected_return"] = cap_df["xpts_val"] * (3 if triple_captain else 2)
    cap_df["cap_confidence"] = np.clip(
        45 + 25 * cap_df["reliability"] + 4 * cap_df["xpts_val"], 40, 95
    )

    non_blank = cap_df[~blank_mask]
    top3_cap  = non_blank.nlargest(3, "_cap_sort")
    top_vc    = non_blank[~non_blank["player_id"].isin(top3_cap.iloc[:1]["player_id"])].nlargest(1, "vc_score")

    if not top3_cap.empty:
        cap_row  = top3_cap.iloc[0]
        vc_name  = top_vc.iloc[0]["player_name"] if not top_vc.empty else "No clear VC"
        cap_conf = float(cap_row.get("cap_confidence", 60.0))
        cap_risk = "Low" if float(cap_row["reliability"]) >= 0.85 else "Medium" if float(cap_row["reliability"]) >= 0.70 else "High"
        dgw_note = "DGW upside available" if float(cap_row.get("double_gws", 0) or 0) > 0 else "No DGW boost"
        cap_ev_display = float(cap_row.get("captain_ev", cap_row["cap_expected_return"]))
        render_decision_banner(
            title="Captain Decision",
            primary_action=f"Captain {cap_row['player_name']} | VC {vc_name}",
            confidence=cap_conf,
            reasons=[
                f"xPts: {cap_row['xpts_val']:.2f} | Cap EV: {cap_ev_display:.1f}{' (from Phase 2 v5)' if has_cap_ev else ''}",
                f"Reliability: {float(cap_row['reliability']):.0%} (p_plays_full)" if "p_plays_full" in cap_df.columns else f"Availability: {float(cap_row['reliability']):.0%}",
                dgw_note,
            ],
            risk_level=cap_risk,
        )
        render_stat_cards([
            {"label": "Captain",          "value": str(cap_row["player_name"]),             "delta": f"Conf {cap_conf:.0f}% · {float(cap_row['reliability']):.0%} reliability", "tone": "positive"},
            {"label": "VC",               "value": str(vc_name),                            "delta": "p_plays_full-weighted backup",                                            "tone": "neutral"},
            {"label": "Cap Expected EV",  "value": f"{cap_ev_display:.1f}",                 "delta": "Expected captained return",                                              "tone": "positive"},
        ])

    # Captain podium
    st.markdown("**Captain Recommendations**")
    medals = ["Captain", "Vice Captain Option", "3rd Option"]
    cap_cols = st.columns(3)
    for i, (_, row) in enumerate(top3_cap.iterrows()):
        with cap_cols[i % 3]:
            mult     = 3 if (triple_captain and i == 0) else 2
            dgw      = row.get("double_gws", 0) > 0
            cap_ev_v = float(row.get("captain_ev", float(row["xpts_val"]) * mult))
            pchg     = float(row.get("predicted_price_change", 0) or 0)
            st.markdown(f"""
            <div class='fpl-card' style='border-color:{"#f6c90e" if i==0 else "#1f3459"}; text-align:center;'>
                <div style='font-size:1.5rem; margin-bottom:0.5rem;'>{medals[i]}</div>
                <div style='display:flex; justify-content:center; margin-bottom:0.4rem;'>
                    <img class='player-face' src='{_safe_text(row.get("player_face",""))}' onerror="this.onerror=null;this.style.display='none';" />
                </div>
                <div style='display:flex; justify-content:center; align-items:center; gap:0.35rem;'>
                    <img class='team-badge' src='{_safe_text(row.get("team_badge",""))}' onerror="this.onerror=null;this.style.display='none';" />
                    <div style='font-weight:800; font-size:1.05rem;'>{_safe_text(row.get("player_name",""))}</div>
                </div>
                <div style='color:#9cb0d0; font-size:0.8rem; margin-top:0.3rem;'>
                    {_safe_text(row.get("team_name",""))} | {_safe_text(row.get("fixture_run_label","?"))}{_price_tag(pchg)}
                </div>
                <div style='font-family:Space Mono; font-size:1.4rem; color:#27e8a7; margin-top:0.6rem;'>
                    xPts {float(row["xpts_val"]):.2f}
                </div>
                <div style='font-size:0.75rem; color:#37b6ff; margin-top:0.2rem;'>
                    Cap EV: {cap_ev_v:.1f} {'(Phase 2 v5)' if has_cap_ev else '(calc)'}
                    {"  DGW" if dgw else ""} {"  TC 3x" if triple_captain and i==0 else ""}
                </div>
                <div style='display:flex; justify-content:center; gap:0.35rem; flex-wrap:wrap; margin-top:0.3rem;'>
                    <span class='xi-role'>Reliability {float(row['reliability'])*100:.0f}%</span>
                    <span class='xi-role'>Upside {float(row['upside']):.1f}</span>
                    <span class='xi-role'>Conf {float(row['cap_confidence']):.0f}%</span>
                </div>
            </div>""", unsafe_allow_html=True)

    if not top_vc.empty:
        vc_row = top_vc.iloc[0]
        st.markdown(f"""<div class='rec-box' style='margin-top:1rem;'>
            <div class='kpi-label'>VICE CAPTAIN (p_plays_full-Weighted)</div>
            <div style='font-weight:800; font-size:1.1rem; margin-top:0.3rem;'>{vc_row['player_name']}</div>
            <div style='font-size:0.85rem; color:#b7c7df; margin-top:0.3rem;'>
                xPts {float(vc_row['xpts_val']):.2f} · Reliability {float(vc_row['reliability'])*100:.0f}% · {vc_row.get('fixture_run_label','?')} —
                Chosen as most reliable backup if captain misses out.
            </div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Monte Carlo captain analysis (Phase 3 v5)
    with st.expander("🎲 Monte Carlo Captain Analysis (1,000 simulations)", expanded=True):
        try:
            mc_results = run_monte_carlo_captain(my_team)
            if mc_results:
                render_section_header("Win Probability by Captain (pts_low / pts_high quantile simulation)")
                mc_rows = []
                for r in mc_results[:5]:
                    pchg = float(my_team[my_team["player_name"] == r["player_name"]]["predicted_price_change"].values[0]
                                 if "predicted_price_change" in my_team.columns and
                                 len(my_team[my_team["player_name"] == r["player_name"]]) > 0 else 0)
                    mc_rows.append({
                        "Player":             r["player_name"],
                        "Win %":              f"{r['win_prob']*100:.1f}%",
                        "Cap EV":             f"{r['captain_ev']:.1f}",
                        "Gain vs others":     f"{r['expected_captain_gain']:+.2f}",
                        "Fixture Run":        r["fixture_run"],
                        "Home/Away":          "H" if r.get("is_home") else "A",
                        "DGW":               "✅" if r.get("double_gws",0) > 0 else "—",
                        "Price Trend":        _price_tag(pchg) or "—",
                    })
                mc_df = pd.DataFrame(mc_rows)
                st.dataframe(mc_df, use_container_width=True, hide_index=True)
                top_mc = mc_results[0]
                st.caption(
                    f"Best captain by simulation: **{top_mc['player_name']}** — wins the captaincy decision "
                    f"{top_mc['win_prob']*100:.1f}% of simulations. "
                    f"Expected captain gain vs captaining someone else: {top_mc['expected_captain_gain']:+.2f} pts."
                )
        except Exception as e:
            st.caption(f"Monte Carlo analysis unavailable: {e}")

    st.divider()

    # Captaincy differential analysis (Phase 3 v5)
    with st.expander("📊 Captaincy Differential (vs Average Manager)", expanded=False):
        try:
            cap_diff_results = get_captaincy_differential_analysis(my_team, bootstrap)
            if cap_diff_results:
                field_ev = cap_diff_results[0]["field_captain_ev"]
                st.caption(f"Average manager's expected captain return (ownership-weighted): **{field_ev:.1f} pts**")
                diff_rows = []
                for r in cap_diff_results:
                    diff_rows.append({
                        "Player":           r["player_name"],
                        "Ownership %":      f"{r['ownership_pct']:.1f}%",
                        "Cap EV":           f"{r['captain_ev']:.1f}",
                        "vs Field":         f"{r['differential_gain']:+.2f}",
                        "Verdict":          "★ DIFFERENTIAL" if r["is_differential"] else "Template",
                        "Fixture Run":      r["fixture_run"],
                    })
                st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)
                st.caption("Positive 'vs Field' = captaining this player gains you rank vs average managers.")
        except Exception as e:
            st.caption(f"Captaincy differential unavailable: {e}")

    st.divider()

    render_section_header("Full Squad xPts Ranking")
    cap_sorted = cap_df.sort_values("xpts_val", ascending=True)
    fig = go.Figure()
    # Use captain_ev if available for the bar chart
    y_col = "captain_ev" if has_cap_ev else "xpts_val"
    fig.add_trace(go.Bar(
        x=cap_sorted[y_col], y=cap_sorted["player_name"],
        orientation="h",
        marker=dict(color=cap_sorted[y_col], colorscale=PLOTLY_XPTS_SCALE),
        hovertemplate="<b>%{y}</b><br>%{x:.2f}<extra></extra>",
        name="captain_ev" if has_cap_ev else "xPts",
    ))
    fig.update_layout(**PLOTLY_THEME, height=450,
                      xaxis_title="Captain EV (Phase 2 v5)" if has_cap_ev else "xPts Score",
                      margin=dict(l=10,r=10,t=20,b=30))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    render_section_header("Captain Matrix: Upside vs Reliability")
    fig2 = px.scatter(cap_df, x="reliability", y="upside", size="xpts_val",
                      color="position", hover_name="player_name",
                      labels={"reliability": "Reliability (p_plays_full)", "upside": "Upside Score"},
                      color_discrete_map=POSITION_COLOR_MAP)
    fig2.update_layout(**PLOTLY_THEME, height=380, margin=dict(l=10,r=10,t=20,b=30))
    fig2.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    st.divider()
    render_section_header("Formation Comparison")
    formations = score_all_formations(my_team)
    if formations:
        form_df = pd.DataFrame(formations).rename(columns={
            "formation":"Formation","pred_pts":"Predicted Pts","combined":"Combined Score"})
        if "xpts" in form_df.columns:
            form_df = form_df.rename(columns={"xpts": "xPts (expected)"})
        form_df["Optimal"] = ["Best" if i == 0 else "" for i in range(len(form_df))]
        st.dataframe(form_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────
# SEASON TRACKER PAGE  (v2: sell_price, expected_pts)
# ─────────────────────────────────────────

elif page == "Season Tracker":
    render_section_header("Season Performance Tracker")

    # v2: use sell_price from value_breakdown
    value_data = track_squad_value(my_team, bootstrap, current_gw, team_data=team_data)
    history = value_data.get("history", {})
    try:
        baseline_gw_num    = int(value_data.get("baseline_gw", current_gw))
        current_value_num  = float(value_data.get("current_value", squad_sell_value))
        baseline_value_num = float(value_data.get("baseline_value", current_value_num))
        total_change_num   = float(value_data.get("total_change", 0.0))
    except (TypeError, ValueError):
        baseline_gw_num = current_gw; current_value_num = squad_sell_value
        baseline_value_num = current_value_num; total_change_num = 0.0

    sign = "+" if total_change_num >= 0 else ""
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Squad Value", f"£{current_value_num:.1f}M", "Sell value (after FPL profit rule)")
    c2.metric("Baseline Value", f"£{baseline_value_num:.1f}M", f"GW{baseline_gw_num}")
    c3.metric("Total Change", f"{sign}{total_change_num:.1f}M")

    # Manager scorecard
    import json
    from pathlib import Path
    TRANSFER_LOG = "transfer_history.json"
    transfer_hit_rate = np.nan
    pred_calibration  = np.nan
    if Path(TRANSFER_LOG).exists():
        try:
            with open(TRANSFER_LOG, encoding="utf-8") as f:
                t_hist = json.load(f)
            t_df_score = pd.DataFrame(t_hist)
            if not t_df_score.empty and "evaluated" in t_df_score.columns:
                eval_df = t_df_score[t_df_score["evaluated"] == True].copy()
                if not eval_df.empty and {"actual_gain","predicted_gain"}.issubset(eval_df.columns):
                    transfer_hit_rate = float((eval_df["actual_gain"] > 0).mean() * 100.0)
                    pred_err = (eval_df["actual_gain"] - eval_df["predicted_gain"]).abs()
                    pred_calibration  = float(max(0.0, 100.0 - pred_err.mean() * 25.0))
        except Exception:
            pass

    value_eff  = float((current_value_num - baseline_value_num) / max(1.0, float(current_gw) - float(baseline_gw_num) + 1.0))
    hit_c      = float(transfer_hit_rate) if not np.isnan(transfer_hit_rate) else 50.0
    cal_c      = float(pred_calibration) if not np.isnan(pred_calibration) else 50.0
    val_c      = float(np.clip(50.0 + 40.0 * value_eff, 0.0, 100.0))
    mgr_score  = float(np.clip(0.40*hit_c + 0.35*cal_c + 0.25*val_c, 0, 100))
    render_stat_cards([
        {"label": "Manager Scorecard",       "value": f"{mgr_score:.0f}/100",                                              "delta": f"Hit {hit_c:.0f} | Cal {cal_c:.0f} | Value {val_c:.0f}", "tone": "positive" if mgr_score >= 70 else "warning" if mgr_score >= 50 else "danger"},
        {"label": "Transfer Hit Rate",       "value": f"{transfer_hit_rate:.1f}%" if not np.isnan(transfer_hit_rate) else "N/A", "delta": "Positive gains",                              "tone": "positive" if not np.isnan(transfer_hit_rate) and transfer_hit_rate >= 55 else "warning"},
        {"label": "Prediction Calibration",  "value": f"{pred_calibration:.1f}%" if not np.isnan(pred_calibration) else "N/A",  "delta": "Predicted vs realized",                       "tone": "positive" if not np.isnan(pred_calibration) and pred_calibration >= 65 else "warning"},
        {"label": "Value Growth Efficiency", "value": f"{value_eff:+.2f}M/GW",                                             "delta": "Sell-price per GW",                               "tone": "positive" if value_eff >= 0 else "danger"},
    ])

    if len(history) > 1:
        gws_hist  = sorted(history.keys(), key=int)
        vals_hist = [float(history[g]) for g in gws_hist]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[f"GW{g}" for g in gws_hist], y=vals_hist,
            mode="lines+markers",
            line=dict(color=PLOTLY_PRIMARY, width=2.5),
            marker=dict(size=8, color=PLOTLY_PRIMARY, line=dict(color="white", width=1.5)),
            fill="tozeroy", fillcolor="rgba(39,232,167,0.08)",
            hovertemplate="<b>%{x}</b><br>Sell Value: £%{y:.1f}M<extra></extra>",
            name="Squad Sell Value",
        ))
        fig.add_hline(y=baseline_value_num, line_dash="dash", line_color=PLOTLY_ACCENT,
                      annotation_text=f"Baseline £{baseline_value_num:.1f}M", annotation_font=dict(color=PLOTLY_ACCENT))
        fig.update_layout(**PLOTLY_THEME, height=300,
                          title=dict(text="Squad Sell Value Over Time", font=dict(color=PLOTLY_ACCENT,size=13,family="Space Mono")),
                          yaxis_title="Sell Value (£M)", margin=dict(l=10,r=10,t=50,b=30))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        st.caption("Values use sell_price (accounts for FPL's 50% profit rule on price rises).")
    else:
        st.info("Run across multiple gameweeks to see value trend.")

    st.divider()
    render_section_header("Transfer History & Model Accuracy")

    if Path(TRANSFER_LOG).exists():
        with open(TRANSFER_LOG, encoding="utf-8") as f:
            t_history = json.load(f)
        if t_history:
            t_df = pd.DataFrame(t_history)
            if "evaluated" not in t_df.columns:
                st.warning("Transfer history schema changed — missing 'evaluated' column.")
                evaluated = pd.DataFrame()
            else:
                evaluated = t_df[t_df["evaluated"] == True]
            if not evaluated.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=evaluated["player_in"], y=evaluated["predicted_gain"], name="Predicted", marker_color=PLOTLY_ACCENT))
                fig.add_trace(go.Bar(x=evaluated["player_in"], y=evaluated["actual_gain"],    name="Actual",    marker_color=PLOTLY_PRIMARY))
                fig.update_layout(**PLOTLY_THEME, barmode="group", height=320,
                                  title=dict(text="Transfer Prediction Accuracy",font=dict(color=PLOTLY_ACCENT,size=13,family="Space Mono")),
                                  xaxis_tickangle=-30, margin=dict(l=10,r=10,t=50,b=80))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
                eval_disp = evaluated[["gw","player_out","player_in","predicted_gain","actual_gain"]].copy()
                eval_disp["result"] = eval_disp.apply(lambda r: "✅ Good" if r["actual_gain"] >= r["predicted_gain"]*0.7 else "❌ Miss", axis=1)
                if "eval_window" in evaluated.columns:
                    eval_disp["window"] = evaluated["eval_window"]
                st.dataframe(eval_disp.rename(columns={"gw":"GW","player_out":"OUT","player_in":"IN","predicted_gain":"Predicted","actual_gain":"Actual","result":"Result","window":"Window"}).sort_values("GW", ascending=False), use_container_width=True, hide_index=True)
            else:
                pending = len(t_df[t_df["evaluated"] == False])
                st.info(f"{pending} transfer(s) logged, not yet evaluable (need ≥3 GWs after suggestion).")
            with st.expander("All logged suggestions"):
                wanted = ["gw","player_out","player_in","predicted_gain","evaluated"]
                safe_cols = [c for c in wanted if c in t_df.columns]
                if len(safe_cols) >= 2:
                    st.dataframe(t_df[safe_cols].rename(columns={"gw":"GW","player_out":"OUT","player_in":"IN","predicted_gain":"Predicted Gain","evaluated":"Evaluated"}), use_container_width=True, hide_index=True)
    else:
        st.info("No transfer history yet. Make transfers via the Transfer Planner and they'll be tracked here.")

    st.divider()
    render_section_header("Model Performance by Position")
    rmse_data = [{"Position": pos, "RMSE (pts)": round(rmse, 3),
                  "R²": round(data["models"].get(pos, {}).get("r2", 0), 3)}
                 for pos, rmse in rmse_map.items()]
    if rmse_data:
        rmse_df = pd.DataFrame(rmse_data)
        fig = px.bar(rmse_df, x="Position", y="RMSE (pts)",
                     color="RMSE (pts)", color_continuous_scale=PLOTLY_RMSE_SCALE,
                     text="RMSE (pts)")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(**PLOTLY_THEME, height=300, showlegend=False,
                          coloraxis_showscale=False, margin=dict(l=10,r=10,t=20,b=30))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        st.caption("RMSE = Root Mean Squared Error (Phase 1 v5 component-blend model). Lower = more accurate.")


# ─────────────────────────────────────────
# AI ANALYST PAGE  (unchanged — Phase 7 integration was already correct)
# ─────────────────────────────────────────

elif page == "AI Analyst":
    render_section_header("AI Analyst | Powered by Groq + Live Data")

    if not ANALYST_AVAILABLE:
        st.error(f"Phase 7 backend not available: {ANALYST_ERROR}")
        st.info("Run: pip install groq feedparser newsapi-python understat nest_asyncio")
        st.stop()

    for k in ["analyst_messages","analyst_sources","analyst_context"]:
        if k not in st.session_state:
            st.session_state[k] = [] if k != "analyst_context" else {}

    user_input = st.chat_input("Ask anything about your squad, transfers, captain, injuries...")

    with st.container():
        for msg in st.session_state["analyst_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources_display"):
                    with st.expander("Sources & Confidence", expanded=False):
                        st.markdown(msg["sources_display"])
                        conf_label = msg.get("confidence_label", "?")
                        conf_score = msg.get("confidence_score", 0)
                        conf_color = PLOTLY_PRIMARY if conf_label=="HIGH" else PLOTLY_WARNING if conf_label=="MEDIUM" else PLOTLY_DANGER
                        st.markdown(
                            f"<div class='kpi-block' style='display:inline-block; padding:0.5rem 1rem;'>"
                            f"<div class='kpi-label'>Source Confidence</div>"
                            f"<div style='color:{conf_color}; font-weight:800; font-size:1.2rem;'>"
                            f"{conf_label} ({conf_score:.0f}%)</div></div>",
                            unsafe_allow_html=True,
                        )

    render_section_header("Or try a quick question:")
    q_cols  = st.columns(4)
    quick_q = None
    for i, q in enumerate(QUICK_QUESTIONS[:4]):
        with q_cols[i]:
            if st.button(q, use_container_width=True, key=f"qq_{i}"):
                quick_q = q
    if len(QUICK_QUESTIONS) > 4:
        with st.expander("More questions", expanded=False):
            more_cols = st.columns(2)
            for i, q in enumerate(QUICK_QUESTIONS[4:]):
                with more_cols[i % 2]:
                    if st.button(q, use_container_width=True, key=f"qq_more_{i}"):
                        quick_q = q

    try:
        proactive_alerts = generate_proactive_alerts(
            my_team=my_team, xi_result=xi_result, news_map=news_map,
            chance_map=chance_map, chip_info=chip_info,
            deadline_status={}, current_gw=current_gw,
        )
    except Exception:
        proactive_alerts = []
    if proactive_alerts:
        render_section_header("Proactive Alerts")
        for alert in proactive_alerts[:5]:
            level   = str(alert.get("level","info")).lower()
            title   = str(alert.get("title","Alert"))
            message = str(alert.get("message",""))
            if level == "critical":   st.error(f"{title}: {message}")
            elif level == "warning":  st.warning(f"{title}: {message}")
            else:                     st.info(f"{title}: {message}")

    with st.expander("How it works", expanded=False):
        st.markdown("""
            <div class='fpl-card' style='border-color:#37b6ff40; margin:0;'>
                <div class='kpi-label'>HOW IT WORKS</div>
                <div style='font-size:0.9rem; color:#c8d8f0; margin-top:0.5rem; line-height:1.6;'>
                    The AI Analyst combines your live squad data (Phase 1-4 v5 outputs including
                    expected_pts, captain_ev, total_ev) with real-time news from multiple sources.
                    <br><br>
                    <b>Sources:</b> FPL API | API-Football lineups | NewsAPI | BBC Sport | Sky Sports |
                    Google News | Odds API | Understat xG
                </div>
            </div>""", unsafe_allow_html=True)

    question = quick_q or user_input
    if question:
        st.session_state["analyst_messages"].append({"role": "user", "content": question})
        st.session_state["analyst_messages"] = st.session_state["analyst_messages"][-30:]
        llm_history = [{"role": m["role"], "content": m["content"]}
                       for m in st.session_state["analyst_messages"][:-1]]
        try:
            c_ilp1  = cached_ilp_transfers(my_team, others, float(bank_balance), n_transfers=1)
            c_ilp2  = cached_ilp_transfers(my_team, others, float(bank_balance), n_transfers=2)
            c_roll  = get_rolling_transfer_advice(my_team, others, bank_balance, transfers_made, chip_info, current_gw, ilp_result=c_ilp1)
            c_hits  = get_hit_transfer_analysis(my_team, others, bank_balance, transfers_made)
        except Exception:
            c_ilp1 = c_ilp2 = c_roll = None; c_hits = []

        with st.spinner("Fetching live data and consulting the analyst..."):
            try:
                result = run_analyst(
                    question=question, my_team=my_team, others=others,
                    enriched_df=enriched_df, xi_result=xi_result,
                    bank_balance=bank_balance, transfers_made=transfers_made,
                    available_chips=available_chips, current_gw=current_gw,
                    news_map=news_map, chance_map=chance_map, bootstrap=bootstrap,
                    chat_history=llm_history,
                    ilp_1=c_ilp1, ilp_2=c_ilp2, roll_advice=c_roll, hit_transfers=c_hits,
                )
                conf_label, conf_score = result["confidence"]
                st.session_state["analyst_messages"].append({
                    "role": "assistant", "content": result["answer"],
                    "sources_display": result["source_display"],
                    "confidence_label": conf_label, "confidence_score": conf_score,
                })
            except Exception as e:
                st.session_state["analyst_messages"].append({
                    "role": "assistant", "content": f"Error running analyst: {str(e)}",
                    "sources_display": "", "confidence_label": "LOW", "confidence_score": 0,
                })
        st.rerun()

    if st.session_state["analyst_messages"]:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state["analyst_messages"] = []
            st.rerun()

    with st.expander("System Status", expanded=False):
        status_items = [
            ("LLM (Groq)", ANALYST_STATUS.get("groq", False), "Ready", "Not installed"),
            ("NewsAPI",    ANALYST_STATUS.get("newsapi", False), "Ready", "No key"),
            ("RSS Feeds",  ANALYST_STATUS.get("feedparser", False), "Ready", "Not installed"),
            ("Understat",  ANALYST_STATUS.get("understat", False), "Ready", "Not installed"),
            ("Odds API",   ANALYST_STATUS.get("odds_api", False), "Ready", "No key"),
        ]
        sc = st.columns(len(status_items))
        for i, (label, ok, ok_t, fail_t) in enumerate(status_items):
            sc[i].markdown(
                f"<div class='kpi-block'><div class='kpi-label'>{label}</div>"
                f"<div style='color:{'#27e8a7' if ok else '#ffb547'}; font-weight:800;'>"
                f"{ok_t if ok else fail_t}</div></div>", unsafe_allow_html=True)
        if ANALYST_STATUS.get("odds_api", False):
            st.caption(get_odds_usage_summary())

    st.divider()
    st.caption(
        "AI Analyst powered by Groq (Llama 3.3 70B) | "
        "Responses grounded in Phase 1-4 v5 data + live news | "
        "Always verify before deadline"
    )


# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────

if page != "AI Analyst":
    st.caption("FPL AI ASSISTANT v2 | PHASE 1-4 v5 BACKEND | BUILT WITH STREAMLIT + PLOTLY")
    st.caption("Always verify bank balance in the FPL app before confirming transfers.")