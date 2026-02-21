"""
FPL AI Assistant ” Phase 6: Streamlit Dashboard
================================================
Interactive web dashboard bringing all phases together.

6 Pages:
  1. My Squad      ” pitch layout, KPI cards, injury flags
  2. Fixture Planner ” interactive colour-coded heatmap
  3. Transfer Planner ” ILP suggestions, before/after XI preview
  4. Player Explorer  ” scatter plots, filters, side-by-side comparison
  5. Captain Picker   ” xPts ranking, chip detection, DGW awareness
  6. Season Tracker   ” squad value trend, transfer accuracy

Run:
  streamlit run fpl_dashboard.py

Install dependencies first:
  pip install streamlit plotly
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        print_transfer_history,
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

st.set_page_config(
    page_title="FPL AI Assistant",
    page_icon="?",
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

DIFFICULTY_COLORS = {
    1: "#1a7a4a", 2: "#2ecc71", 3: "#ffb547",
    4: "#e67e22", 5: "#ff5d73", 6: "#7f0000",
}


@st.cache_data(ttl=300, show_spinner=False)
def load_all_data(team_id: int, refresh: bool = False):
    """Load all data from FPL API and run full pipeline. Cached for 5 mins."""
    bootstrap   = fetch_bootstrap()
    fixtures_df = fetch_fixtures()
    current_gw  = fetch_current_gw(bootstrap)

    team_data     = fetch_my_team(team_id, current_gw)
    my_player_ids = [p["element"] for p in team_data["picks"]]

    transfer_info = fetch_transfer_info(team_id, current_gw)

    history_df = build_player_history_df(bootstrap, refresh=refresh)
    models     = train_models(history_df)
    rmse_map   = get_rmse_from_models(models)

    pred_df = build_current_features(
        bootstrap, fixtures_df, history_df,
        models, current_gw, my_player_ids=my_player_ids
    )

    custom_diff     = build_custom_difficulty(history_df, bootstrap)
    team_form_map   = build_team_form(history_df, bootstrap)
    opp_scoring_map = build_opponent_scoring_map(history_df)
    chip_info       = build_chip_status(team_id, bootstrap, fixtures_df, current_gw)

    fixture_run_df = build_fixture_run(
        bootstrap, fixtures_df, current_gw,
        custom_difficulty=custom_diff,
        gw_lookahead=FIXTURE_LOOKAHEAD
    )
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
        "transfer_info": transfer_info,
        "enriched_df":   enriched_df,
        "my_team":       my_team,
        "others":        others,
        "xi_result":     xi_result,
        "chip_info":     chip_info,
        "rmse_map":      rmse_map,
        "models":        models,
        "history_df":    history_df,
    }


@st.cache_data(show_spinner=False)
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
        f"<div style='font-size:0.74rem; color:#90a2be; margin-top:0.1rem;'>{subtitle_safe}</div>"
        if subtitle_safe else ""
    )

    return (
        f"<div class='entity-line'>"
        f"<img class='{face_class}' src='{face_final}' "
        "onerror=\"this.style.display='none'; this.nextElementSibling.style.display='flex';\" />"
        f"<div class='{face_class}' style='display:none; align-items:center; justify-content:center;"
        " color:#eaf2ff; font-weight:700; background:linear-gradient(145deg,#172a4a,#0d1730);'>"
        f"{fallback_initial}</div>"
        "<div style='min-width:0;'>"
        f"<div style='font-weight:700; font-size:0.96rem; line-height:1.1;'>{pname}</div>"
        "<div class='entity-line' style='margin-top:0.16rem; gap:0.35rem;'>"
        f"<img class='team-badge' src='{badge_final}' onerror=\"this.onerror=null;this.style.display='none';\" />"
        f"<span style='font-size:0.75rem; color:#9cb0d0;'>{tname}</span>"
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
        pos_df = pos_df.sort_values("predicted_pts", ascending=False)
        tiles = []
        for _, r in pos_df.iterrows():
            pid = int(r.get("player_id", 0))
            pts = float(r.get("predicted_pts", 0.0))
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
                " color:#eaf2ff; font-weight:700; background:linear-gradient(145deg,#172a4a,#0d1730);'>?</div>"
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
        f"<li style='margin:0.15rem 0; color:#b9cae6; font-size:0.82rem;'>{_safe_text(r)}</li>"
        for r in (reasons or [])
    )
    st.markdown(
        f"""
        <div class='rec-box {risk_cls}'>
            <div class='kpi-label'>{_safe_text(title)}</div>
            <div style='display:flex; justify-content:space-between; gap:0.8rem; align-items:flex-start;'>
                <div>
                    <div style='font-size:1.15rem; font-weight:800; color:#eaf2ff;'>{_safe_text(primary_action)}</div>
                    <div style='font-size:0.78rem; color:#90a2be; margin-top:0.2rem;'>Risk: {_safe_text(risk_level)}</div>
                </div>
                <div style='text-align:right;'>
                    <div style='font-size:0.68rem; color:#37b6ff; font-family:Space Mono;'>CONFIDENCE</div>
                    <div style='font-size:1.25rem; font-weight:800; color:#27e8a7;'>{float(confidence):.0f}%</div>
                </div>
            </div>
            <ul style='margin:0.55rem 0 0 1.0rem; padding:0;'>{reasons_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str):
    st.markdown(
        f"<div class='section-header'>{_safe_text(title)}</div>",
        unsafe_allow_html=True,
    )


def render_stat_cards(cards: list[dict], compact: bool = False):
    if not cards:
        return
    per_row = 2 if compact else min(5, len(cards))
    for i in range(0, len(cards), per_row):
        row = cards[i:i + per_row]
        cols = st.columns(len(row))
        for idx, card in enumerate(row):
            tone = card.get("tone", "neutral")
            tone_color = {
                "positive": "#27e8a7",
                "warning": "#ffb547",
                "danger": "#ff5d73",
                "neutral": "#37b6ff",
            }.get(tone, "#37b6ff")
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
        f"<li style='margin:0.12rem 0; color:#ffb2bf; font-size:0.76rem;'>{_safe_text(r)}</li>"
        for r in (risk_notes or [])
    )
    points = "".join(
        f"<li style='margin:0.12rem 0; color:#bdd3f2; font-size:0.76rem;'>{_safe_text(p)}</li>"
        for p in (supporting_points or [])
    )
    st.markdown(
        f"""
        <div class='transfer-card'>
            <div class='kpi-label'>Recommendation</div>
            <div style='font-size:1rem; font-weight:800; color:#eaf2ff; margin-bottom:0.35rem;'>{_safe_text(headline)}</div>
            <div style='display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:0.45rem;'>
                <span class='xi-role'>Next GW: {impact_now:+.2f}</span>
                <span class='xi-role'>5 GW: {impact_horizon:+.2f}</span>
                <span class='xi-role'>Confidence: {confidence:.0f}%</span>
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


def _normalize_1_5(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    vmin, vmax = float(values.min()), float(values.max())
    if abs(vmax - vmin) < 1e-9:
        return pd.Series([3.0] * len(values), index=values.index)
    return 1.0 + 4.0 * ((values - vmin) / (vmax - vmin))


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
    hit_count = len(hit_transfers or [])
    if rec == "USE NOW":
        score = base + 9.0 + 6.0 * min(gain_5, 3.0) + 4.0 * min(gain_1, 2.0) - 2.0 * hit_count
    elif rec == "BORDERLINE":
        score = base + 2.0 + 3.5 * min(gain_5, 2.0) + 2.0 * min(gain_1, 1.5) - 1.5 * hit_count
    else:
        score = base - 8.0 + 1.5 * min(gain_5, 1.0) - 2.0 * hit_count
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


@st.cache_data(ttl=300, show_spinner=False)
def build_ui_health_snapshot() -> dict:
    from pathlib import Path
    p = Path(__file__)
    txt = p.read_text(encoding="utf-8")
    unsafe_count = txt.count("unsafe_allow_html=True")
    plotly_calls = txt.count("st.plotly_chart(")
    plotly_cfg = txt.count('config={"displayModeBar": False, "responsive": True}')
    return {
        "unsafe_html_count": unsafe_count,
        "plotly_calls": plotly_calls,
        "plotly_configured": plotly_cfg,
    }


with st.sidebar:
    st.markdown("### ? FPL AI")
    st.caption("ASSISTANT")

    st.divider()

    team_id_input = st.number_input(
        "Your FPL Team ID",
        value=TEAM_ID if BACKEND_AVAILABLE else 9179961,
        step=1,
        help="Find your team ID in the FPL website URL"
    )

    bank_override = st.number_input(
        "Bank Balance Override (£M)",
        value=0.0, step=0.1, min_value=0.0,
        help="Override if API bank differs from FPL app"
    )

    refresh = st.toggle("Force Fresh Data", value=False,
                        help="Re-fetch all player history from API (~2 min)")

    if st.button("Run Analysis", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.session_state["run"] = True
        st.rerun()

    st.divider()

    page = st.radio(
        "Navigation",
        ["My Squad", "Fixture Planner", "Transfer Planner",
         "Player Explorer", "Captain Picker", "Season Tracker",
         "AI Analyst"],
        label_visibility="collapsed"
    )
    compact_layout = st.toggle(
        "Compact layout",
        value=False,
        help="Use stacked controls and fewer columns for smaller screens.",
    )
    dev_mode = os.getenv("FPL_DEBUG_UI", "0") == "1"
    show_qa_panel = st.toggle(
        "Show QA panel",
        value=False,
        help="Developer diagnostics for UI/runtime health.",
    ) if dev_mode else False

    st.divider()
    st.caption("Phases 1-4 backend active | ML | ILP | Fixtures | XI Optimizer")



if not BACKEND_AVAILABLE:
    st.error(f"Backend import failed: `{IMPORT_ERROR}`")
    st.info("Make sure all phase files (fpl_phase1_model.py through fpl_phase4_optimizer.py) "
            "and config.py are in the same directory as fpl_dashboard.py")
    st.stop()

if "run" not in st.session_state:
    st.session_state["run"] = False

skeleton_placeholder = st.empty()
with skeleton_placeholder.container():
    render_loading_skeleton()
with st.spinner("Loading data from FPL API and running analysis..."):
    try:
        data = load_all_data(int(team_id_input), refresh=refresh)
    except Exception as e:
        skeleton_placeholder.empty()
        st.error(f"Failed to load data: {e}")
        st.info("Check your team ID and internet connection, then click Run Analysis.")
        st.stop()
skeleton_placeholder.empty()

# Unpack
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

render_section_header(f"GW{current_gw+1} Decision Center")
ctx1, ctx2, ctx3 = st.columns(3)
ctx1.caption(f"Bank £{bank_balance:.1f}M")
ctx2.caption("1 FT" if transfers_made == 0 else "FT Used")
ctx3.caption(f"{len(available_chips)} chips available")

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

if show_qa_panel:
    snap = build_ui_health_snapshot()
    with st.expander("QA Panel", expanded=False):
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
            ],
            compact=compact_layout,
        )
        st.caption("Use compact layout for 768/430/390 widths during manual UI checks.")


if page == "My Squad":

    render_section_header(f"GW{current_gw} Completed -> Optimized for GW{current_gw+1}")

    squad_val   = my_team["price"].sum()
    pred_total  = (xi_result["starting_xi"]["predicted_pts"].sum()
                   if xi_result else my_team["predicted_pts"].sum())
    lo, hi = (compute_score_range(xi_result["starting_xi"], rmse_map)
              if xi_result else (0, 0))
    ft_label = "Free Transfer" if transfers_made == 0 else "Used"
    risk_players = int((my_team.get("blank_gws", 0) > 0).sum()) + int(
        (my_team["player_id"].map(chance_map).fillna(100) < 75).sum()
    )
    bench_cover = float(xi_result["bench"]["predicted_pts"].sum()) if xi_result and not xi_result["bench"].empty else 0.0
    confidence = float(np.clip(84 - risk_players * 5, 35, 90))
    render_decision_banner(
        title="My Squad Decision",
        primary_action=f"Start {xi_result.get('formation', 'Best XI') if xi_result else 'Best XI'}",
        confidence=confidence,
        reasons=[
            f"Projected XI score {pred_total:.1f} pts",
            f"Bench cover {bench_cover:.1f} pts",
            f"Risk flags on {risk_players} players",
        ],
        risk_level="Low" if risk_players <= 2 else "Medium" if risk_players <= 4 else "High",
    )
    render_stat_cards(
        [
            {"label": "Expected Pts", "value": f"{pred_total:.1f}", "delta": "Starting XI projection", "tone": "positive"},
            {"label": "Risk-Adjusted", "value": f"{max(pred_total - 0.4 * risk_players, 0):.1f}", "delta": "Availability-adjusted", "tone": "neutral"},
            {"label": "Players at Risk", "value": str(risk_players), "delta": "Blanks + low availability", "tone": "warning" if risk_players <= 3 else "danger"},
            {"label": "Bench Cover", "value": f"{bench_cover:.1f}", "delta": "First backup strength", "tone": "neutral"},
            {"label": "Bank", "value": f"£{bank_balance:.1f}M", "delta": ft_label, "tone": "positive"},
        ],
        compact=compact_layout,
    )

    st.divider()

    if xi_result:
        render_section_header("Optimal Starting XI")

        xi   = xi_result["starting_xi"]
        cap  = xi_result["captain"]
        vc   = xi_result["vice_captain"]
        bench = xi_result["bench"]

        xi_cards = xi.copy()
        if "player_face" not in xi_cards.columns:
            xi_cards["player_face"] = xi_cards["player_id"].map(player_face_map)
        if "team_badge" not in xi_cards.columns:
            xi_cards["team_badge"] = xi_cards["team_id"].map(team_badge_map)

        st.markdown(
            build_lineup_board_html(
                xi_cards,
                int(cap["player_id"]),
                int(vc["player_id"]),
            ),
            unsafe_allow_html=True,
        )

        if compact_layout:
            st.caption("8+ pts elite | 5-7 pts strong | 3-4 pts playable | <3 pts risky | C captain | VC vice")
        else:
            l1, l2, l3, l4 = st.columns(4)
            l1.markdown("`8+ pts` elite")
            l2.markdown("`5-7 pts` strong")
            l3.markdown("`3-4 pts` playable")
            l4.markdown("`<3 pts` risky · `C` captain · `VC` vice")
        alt_forms = score_all_formations(my_team)
        if alt_forms:
            alt_df = pd.DataFrame(alt_forms)[["formation", "pred_pts", "combined"]].head(5).copy()
            best_pts = float(alt_df["pred_pts"].max()) if not alt_df.empty else 0.0
            alt_df["delta_vs_best"] = alt_df["pred_pts"] - best_pts
            alt_df = alt_df.rename(columns={
                "formation": "Formation",
                "pred_pts": "Predicted Pts",
                "combined": "Combined Score",
                "delta_vs_best": "Delta vs Best",
            })
            render_section_header("Formation Alternatives")
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

    st.divider()

    if xi_result and not xi_result["bench"].empty:
        render_section_header("Bench")
        bench_cols = st.columns(2 if compact_layout else 4)
        for i, (_, row) in enumerate(xi_result["bench"].iterrows()):
            with bench_cols[i % len(bench_cols)]:
                chance = chance_map.get(int(row["player_id"]))
                badge  = ("Red" if chance is not None and chance < 75 else
                          "Amber" if chance is not None and chance < 100 else "Green")
                subtitle = (
                    f"{row['position']} · £{row['price']:.1f}M · {row['predicted_pts']} pts · "
                    f"{badge} {f'{chance}%' if chance is not None else '100%'}"
                )
                st.markdown(f"""
                <div class='fpl-card'>
                    <div style='font-size:0.7rem; color:#37b6ff;
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

    render_section_header("Full Squad Stats")

    disp = my_team[[
        "player_face", "team_badge", "player_name", "position", "team_name", "price",
        "predicted_pts", "combined_score", "avg_difficulty",
        "fixture_run_label", "blank_gws", "double_gws"
    ]].copy()
    disp["ownership"] = disp.index.map(
        lambda i: f"{ownership_map.get(int(my_team.loc[i,'player_id']), 0):.1f}%"
        if i in my_team.index else "?"
    )
    disp = disp.rename(columns={
        "player_face": "Face",
        "team_badge": "Badge",
        "player_name": "Player", "position": "Pos",
        "team_name": "Team", "price": "£",
        "predicted_pts": "Pred", "combined_score": "5GW Score",
        "avg_difficulty": "Avg Diff", "fixture_run_label": "Run",
        "blank_gws": "Blanks", "double_gws": "DGWs",
        "ownership": "Owned%"
    }).sort_values("Pred", ascending=False)

    st.dataframe(disp, use_container_width=True, hide_index=True,
                 column_config={
                     "Face": st.column_config.ImageColumn("Face", width="small"),
                     "Badge": st.column_config.ImageColumn("Badge", width="small"),
                     "Pred": st.column_config.NumberColumn(format="%.2f"),
                     "5GW Score": st.column_config.NumberColumn(format="%.2f"),
                     "Avg Diff": st.column_config.NumberColumn(format="%.1f"),
                     "£": st.column_config.NumberColumn(format="£%.1f"),
                 })

    injury_players = [
        (row["player_name"], chance_map.get(int(row["player_id"])),
         news_map.get(int(row["player_id"]), ""))
        for _, row in my_team.iterrows()
        if chance_map.get(int(row["player_id"])) is not None
        and chance_map.get(int(row["player_id"])) < 100
    ]
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


elif page == "Fixture Planner":

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
    if compact_layout:
        c1, c2 = st.columns(2)
        with c1:
            position_filter = st.multiselect(
                "Filter by position",
                ["Goalkeeper", "Defender", "Midfielder", "Forward"],
                default=["Midfielder", "Forward"],
            )
        with c2:
            show_all_teams = st.toggle("Show all 20 teams", value=False)
        c3, c4 = st.columns(2)
        with c3:
            gw_start, gw_end = st.select_slider(
                "GW range",
                options=future_gws,
                value=(default_start, default_end),
            )
        with c4:
            sort_mode = st.selectbox("Sort mode", ["Easiest overall", "Easiest next 2", "Blank risk"])
        preset = st.selectbox("Team preset", list(preset_map.keys()))
        custom_team_selection = st.multiselect(
            "Custom teams",
            all_team_names,
            default=preset_map.get(preset, []),
            disabled=show_all_teams,
        )
    else:
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

    # Build matrix with readable cell labels and ranking context
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
            gw_fix = fixtures_df[fixtures_df["event"] == gw]
            home = gw_fix[gw_fix["team_h"] == tid]
            away = gw_fix[gw_fix["team_a"] == tid]
            if not home.empty:
                opp_id = int(home.iloc[0]["team_a"])
                opp = team_name_map.get(opp_id, "?")
                opp_short = team_short_map.get(opp_id, opp[:3].upper())
                diff = int(home.iloc[0]["team_h_difficulty"])
                row_diffs.append(diff)
                row_labels.append(f"{opp_short} (H)<br><b>{diff}</b>")
                row_hover.append(f"vs {opp} (Home) · Difficulty {diff}")
                row_meta.append([opp, team_badge_map.get(opp_id, ""), "H", diff])
            elif not away.empty:
                opp_id = int(away.iloc[0]["team_h"])
                opp = team_name_map.get(opp_id, "?")
                opp_short = team_short_map.get(opp_id, opp[:3].upper())
                diff = int(away.iloc[0]["team_a_difficulty"])
                row_diffs.append(diff)
                row_labels.append(f"{opp_short} (A)<br><b>{diff}</b>")
                row_hover.append(f"vs {opp} (Away) · Difficulty {diff}")
                row_meta.append([opp, team_badge_map.get(opp_id, ""), "A", diff])
            else:
                row_diffs.append(0)
                row_labels.append("<b>Blank</b>")
                row_hover.append("Blank gameweek")
                row_meta.append(["Blank", "", "-", 0])
                blanks += 1

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

        if compact_layout:
            r1c1, r1c2 = st.columns(2)
            r1c1.metric("Best Run", display_teams[0], f"{avg_diffs[0]:.2f} avg diff")
            r1c2.metric("Worst Run", display_teams[-1], f"{avg_diffs[-1]:.2f} avg diff")
            r2c1, r2c2 = st.columns(2)
            r2c1.metric("Total Blanks", sum(blank_counts), f"{len(gws)} GW window")
            r2c2.metric("Teams Tracked", len(display_teams), "ranked easiest ? hardest")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Best Run", display_teams[0], f"{avg_diffs[0]:.2f} avg diff")
            k2.metric("Worst Run", display_teams[-1], f"{avg_diffs[-1]:.2f} avg diff")
            k3.metric("Total Blanks", sum(blank_counts), f"{len(gws)} GW window")
            k4.metric("Teams Tracked", len(display_teams), "ranked easiest ? hardest")

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
            textfont=dict(size=10, color="#101217", family="Space Mono"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{x}: %{customdata[0]} (%{customdata[2]})<br>"
                "Difficulty: %{customdata[3]}<br>"
                "<extra></extra>"
            ),
            colorscale=colorscale,
            zmin=0, zmax=5,
            xgap=2,
            ygap=2,
            showscale=True,
            colorbar=dict(
                title=dict(text="Difficulty", font=dict(color="#37b6ff")),
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["Easy", "2", "3", "4", "Hard"],
                tickfont=dict(color="#eaf2ff"),
            ),
        ))

        fig.update_layout(
            **PLOTLY_THEME,
            height=max(380, len(display_teams) * 44 + 120),
            margin=dict(l=10, r=10, t=52, b=30),
            title=dict(
                text=f"Fixture Radar · GW{gws[0]} to GW{gws[-1]}",
                font=dict(color="#37b6ff", size=13, family="Space Mono"),
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

    render_section_header(
        f"Bank: £{bank_balance:.1f}M | "
        f"{'1 Free Transfer' if transfers_made == 0 else 'Free Transfer Used'}"
        f"{' | Hit analysis available' if transfers_made > 0 else ''}"
    )

    with st.spinner("Computing optimal transfers..."):
        ilp_1 = cached_ilp_transfers(my_team, others, float(bank_balance), n_transfers=1)
        ilp_2 = cached_ilp_transfers(my_team, others, float(bank_balance), n_transfers=2)
        roll   = get_rolling_transfer_advice(
            my_team, others, bank_balance, transfers_made,
            chip_info, current_gw, ilp_result=ilp_1
        )
        hit_transfers = get_hit_transfer_analysis(
            my_team, others, bank_balance, transfers_made
        )

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

    stack_candidates = []
    if ilp_1.get("transfers"):
        t = ilp_1["transfers"][0]
        stack_candidates.append(
            {
                "label": "Safe move",
                "headline": f"{t['out_name']} -> {t['in_name']}",
                "now": float(ilp_1.get("total_next_gain", 0.0)),
                "horizon": float(ilp_1.get("total_gain", 0.0)),
                "cost": float(ilp_1.get("total_cost", 0.0)),
                "risk": ["Low variance role", "Single transfer only"],
                "why": [f"Run: {t.get('fixture_run','?')}", f"Position: {t.get('position','?')}"],
            }
        )
    if ilp_2.get("transfers"):
        stack_candidates.append(
            {
                "label": "Aggressive move",
                "headline": " + ".join([f"{x['out_name']} -> {x['in_name']}" for x in ilp_2["transfers"][:2]]),
                "now": float(ilp_2.get("total_next_gain", 0.0)),
                "horizon": float(ilp_2.get("total_gain", 0.0)),
                "cost": float(ilp_2.get("total_cost", 0.0)),
                "risk": ["Higher variance", "Two moves lock flexibility"],
                "why": ["Targets broader fixture turn", "Larger ceiling if both start"],
            }
        )
    diff_df = get_differential_picks(others, bootstrap, top_n=1)
    if not diff_df.empty:
        d = diff_df.iloc[0]
        stack_candidates.append(
            {
                "label": "Differential move",
                "headline": f"Buy {d['player_name']} ({d['ownership_pct']:.1f}% owned)",
                "now": float(d.get("predicted_pts", 0.0)),
                "horizon": float(d.get("combined_score", 0.0)),
                "cost": float(d.get("price", 0.0)),
                "risk": ["Low ownership volatility", "Minutes uncertainty possible"],
                "why": [f"Fixture run: {d.get('fixture_run_label','?')}", f"Differential score: {d.get('differential_score',0):.2f}"],
            }
        )
    if stack_candidates:
        render_section_header("Recommendation Stack")
        for c in stack_candidates[:3]:
            raw_conf = 60 + 7 * max(0.0, c["horizon"] - c["now"])
            conf = float(np.clip(raw_conf, 40, 88))
            render_recommendation_card(
                headline=f"{c['label']}: {c['headline']}",
                impact_now=c["now"],
                impact_horizon=c["horizon"],
                confidence=conf,
                risk_notes=c["risk"],
                supporting_points=[*c["why"], f"Net cost: {c['cost']:+.1f}M"],
            )

    tab1, tab2, tab3 = st.tabs(["1 FT", "2 FT", "-4 Hit"])

    def _transfer_card_meta(tr: dict, fallback_next: float = 0.0) -> tuple[float, float, float, str, list[str]]:
        next_gain = float(tr.get("next_gain", fallback_next) or 0.0)
        horizon_gain = float(tr.get("gain", 0.0) or 0.0)
        cost = float(tr.get("cost_diff", 0.0) or 0.0)
        is_blank = bool(tr.get("is_blank", False))
        has_dgw = float(tr.get("double_gws", 0) or 0) > 0
        confidence = float(np.clip(58 + 10 * horizon_gain + 5 * next_gain - (12 if is_blank else 0) - (4 if cost > 1.5 else 0), 35, 92))
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
        return next_gain, horizon_gain, confidence, risk_level, risks

    with tab1:
        if ilp_1.get("transfers"):
            t = ilp_1["transfers"][0]
            blank_note = " | BLANK" if t.get("is_blank") else ""
            dgw_note   = " | DGW"   if t.get("double_gws",0) > 0 else ""
            out_team = transfer_team_name_map.get(t['out_name'], "")
            in_team = transfer_team_name_map.get(t['in_name'], "")
            if compact_layout:
                st.markdown(f"""
                <div class='transfer-card'>
                    <div class='kpi-label'>TRANSFER OUT</div>
                    {player_identity_html(
                        t['out_name'],
                        out_team,
                        transfer_face_map.get(t['out_name'], ''),
                        transfer_badge_map.get(t['out_name'], ''),
                        f"{t['position']} | Gain: +{t['gain']:.2f}",
                        'player-face-sm'
                    )}
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class='transfer-card' style='border-color:#27e8a740;'>
                    <div class='kpi-label'>TRANSFER IN</div>
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
            else:
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
                            f"{t['position']} | Gain: +{t['gain']:.2f}",
                            'player-face-sm'
                        )}
                    </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""
                    <div style='text-align:center; padding-top:1.5rem;'>
                        <div class='transfer-gain'>+{t['gain']:.2f}</div>
                        <div style='font-size:0.65rem; color:#37b6ff;
                                    font-family:Space Mono; letter-spacing:0.1em;'>
                            5GW GAIN
                        </div>
                        <div style='font-size:0.8rem; color:#90a2be; margin-top:0.4rem;'>
                            Cost: {t['cost_diff']:+.1f}M
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_c:
                    st.markdown(f"""
                    <div class='transfer-card' style='border-color:#27e8a740;'>
                        <div class='kpi-label'>TRANSFER IN</div>
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

            t_next, t_horizon, t_conf, t_risk, t_risks = _transfer_card_meta(
                t,
                fallback_next=float(ilp_1.get("total_next_gain", 0.0)),
            )
            render_recommendation_card(
                headline=f"Transfer detail: {t['out_name']} -> {t['in_name']}",
                impact_now=t_next,
                impact_horizon=t_horizon,
                confidence=t_conf,
                risk_notes=t_risks,
                supporting_points=[
                    f"Cost delta: £{float(t.get('cost_diff', 0.0)):+.1f}M",
                    f"Fixture run: {t.get('fixture_run', '?')}",
                    f"Risk level: {t_risk}",
                ],
            )

            render_stat_cards(
                [
                    {"label": "5GW Combined Gain", "value": f"+{ilp_1['total_gain']:.2f}", "delta": "Horizon impact", "tone": "positive"},
                    {"label": "Next GW Gain", "value": f"+{ilp_1['total_next_gain']:.2f}", "delta": "Immediate impact", "tone": "neutral"},
                    {"label": "Net Cost", "value": f"£{ilp_1['total_cost']:+.1f}M", "delta": "Budget delta", "tone": "warning" if ilp_1["total_cost"] > 0 else "positive"},
                    {"label": "Solver", "value": str(ilp_1.get("solver_status", "ILP")), "delta": "Optimization status", "tone": "neutral"},
                ],
                compact=compact_layout,
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
            p1.markdown(f"**Before XI**  \n{', '.join(sorted(before_names))[:220]}")
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
            p3.markdown(f"**After XI**  \n{', '.join(sorted(after_names))[:220]}")
        else:
            st.info("No beneficial 1-transfer found within budget.")

    with tab2:
        if ilp_2.get("transfers") and len(ilp_2["transfers"]) == 2:
            for i, t in enumerate(ilp_2["transfers"], 1):
                blank_note = " | BLANK" if t.get("is_blank") else ""
                dgw_note   = " | DGW"   if t.get("double_gws",0) > 0 else ""
                out_team = transfer_team_name_map.get(t['out_name'], "")
                in_team = transfer_team_name_map.get(t['in_name'], "")
                if compact_layout:
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
                    st.markdown(f"""
                    <div class='transfer-card' style='border-color:#27e8a740;'>
                        <div class='kpi-label'>T{i} - IN</div>
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
                else:
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
                            <div class='transfer-gain'>+{t['gain']:.2f}</div>
                            <div style='font-size:0.65rem; color:#37b6ff; font-family:Space Mono;'>
                            5GW GAIN</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_c:
                        st.markdown(f"""
                        <div class='transfer-card' style='border-color:#27e8a740;'>
                            <div class='kpi-label'>T{i} - IN</div>
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

                t2_next, t2_horizon, t2_conf, t2_risk, t2_risks = _transfer_card_meta(
                    t,
                    fallback_next=float(ilp_2.get("total_next_gain", 0.0)) / max(len(ilp_2.get("transfers", [])), 1),
                )
                render_recommendation_card(
                    headline=f"T{i} detail: {t['out_name']} -> {t['in_name']}",
                    impact_now=t2_next,
                    impact_horizon=t2_horizon,
                    confidence=t2_conf,
                    risk_notes=t2_risks,
                    supporting_points=[
                        f"Cost delta: £{float(t.get('cost_diff', 0.0)):+.1f}M",
                        f"Fixture run: {t.get('fixture_run', '?')}",
                        f"Risk level: {t2_risk}",
                    ],
                )

            render_stat_cards(
                [
                    {"label": "Total 5GW Gain", "value": f"+{ilp_2['total_gain']:.2f}", "delta": "Combined horizon impact", "tone": "positive"},
                    {"label": "Total Next GW", "value": f"+{ilp_2['total_next_gain']:.2f}", "delta": "Immediate impact", "tone": "neutral"},
                    {"label": "Total Cost", "value": f"£{ilp_2['total_cost']:+.1f}M", "delta": "Budget delta", "tone": "warning" if ilp_2["total_cost"] > 0 else "positive"},
                ],
                compact=compact_layout,
            )
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
                if compact_layout:
                    r1c1, r1c2 = st.columns(2)
                    r1c1.markdown(f"**OUT** {h['replace']}")
                    r1c2.markdown(f"**IN** {h['player_in']} · {h['fixture_run']}")
                else:
                    c1, c2 = st.columns([3, 3])
                    c1.markdown(f"**OUT** {h['replace']}")
                    c2.markdown(f"**IN** {h['player_in']} · {h['fixture_run']}")
                render_recommendation_card(
                    headline=f"Hit detail: {h['replace']} -> {h['player_in']}",
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
                    compact=compact_layout,
                )


elif page == "Player Explorer":

    render_section_header("Player Explorer")

    if "px_pos_filter" not in st.session_state:
        st.session_state["px_pos_filter"] = ["Midfielder", "Forward"]
    if "px_price_range" not in st.session_state:
        st.session_state["px_price_range"] = (4.0, 13.0)
    if "px_min_pred" not in st.session_state:
        st.session_state["px_min_pred"] = 2.0
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

    r_mode, r_reset = st.columns([1, 1])
    with r_mode:
        mode = st.radio("Objective mode", ["Value", "Ceiling", "Safety"], horizontal=True, key="px_mode")
    with r_reset:
        if st.button("Reset filters", use_container_width=True):
            st.session_state["px_pos_filter"] = ["Midfielder", "Forward"]
            st.session_state["px_price_range"] = (4.0, 13.0)
            st.session_state["px_min_pred"] = 2.0
            st.session_state["px_search"] = ""
            st.session_state["px_pos_filter_widget"] = st.session_state["px_pos_filter"]
            st.session_state["px_price_range_widget"] = st.session_state["px_price_range"]
            st.session_state["px_min_pred_widget"] = st.session_state["px_min_pred"]
            st.session_state["px_search_widget"] = st.session_state["px_search"]
            st.rerun()

    if compact_layout:
        f1, f2 = st.columns(2)
        with f1:
            pos_filter = st.multiselect("Position",
                ["Goalkeeper","Defender","Midfielder","Forward"],
                key="px_pos_filter_widget")
        with f2:
            search = st.text_input("Search player name", st.session_state["px_search_widget"], key="px_search_widget")
        f3, f4 = st.columns(2)
        with f3:
            price_range = st.slider("Price Range (£M)", 3.5, 15.0, st.session_state["px_price_range_widget"], 0.5, key="px_price_range_widget")
        with f4:
            min_pred = st.slider("Min Predicted Pts", 0.0, 15.0, st.session_state["px_min_pred_widget"], 0.5, key="px_min_pred_widget")
    else:
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            pos_filter = st.multiselect("Position",
                ["Goalkeeper","Defender","Midfielder","Forward"],
                key="px_pos_filter_widget")
        with f2:
            price_range = st.slider("Price Range (£M)", 3.5, 15.0, st.session_state["px_price_range_widget"], 0.5, key="px_price_range_widget")
        with f3:
            min_pred = st.slider("Min Predicted Pts", 0.0, 15.0, st.session_state["px_min_pred_widget"], 0.5, key="px_min_pred_widget")
        with f4:
            search = st.text_input("Search player name", st.session_state["px_search_widget"], key="px_search_widget")

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

    tab_scatter, tab_bar, tab_compare, tab_table = st.tabs([
        "Pts vs Price", "Top Differentials", "Player Comparison", "Full Table"
    ])

    with tab_scatter:
        plot_df = pool.copy()
        if plot_df.empty:
            st.info("No players match the current filters.")
        else:
            plot_df["value_index"] = (
                plot_df["predicted_pts"] / plot_df["price"].replace(0, np.nan)
            ).fillna(0.0)
            if "chance_of_playing" in plot_df.columns:
                safety = plot_df["chance_of_playing"].fillna(100).astype(float) / 100.0
            else:
                safety = pd.Series([1.0] * len(plot_df), index=plot_df.index, dtype=float)
            plot_df["objective_score"] = (
                plot_df["value_index"] if mode == "Value"
                else plot_df["predicted_pts"] if mode == "Ceiling"
                else plot_df["predicted_pts"] * safety
            )
            x_med = float(plot_df["price"].median())
            y_med = float(plot_df["predicted_pts"].median())
            top_value = plot_df.nlargest(8, "objective_score")

            fig = px.scatter(
                plot_df,
                x="price", y="predicted_pts",
                color="position",
                size="value_index",
                size_max=26,
                opacity=0.78,
                hover_name="player_name",
                hover_data={
                    "team_name": True,
                    "price": ":.1f",
                    "predicted_pts": ":.2f",
                    "combined_score": ":.2f",
                    "fixture_run_label": True,
                    "ownership": ":.1f",
                    "value_index": ":.3f",
                },
                labels={"price": "Price (£M)", "predicted_pts": "Predicted Points"},
                color_discrete_map={
                    "Goalkeeper": "#27e8a7",
                    "Defender": "#37b6ff",
                    "Midfielder": "#ffb547",
                    "Forward": "#ff5d73",
                },
            )

            fig.add_hline(
                y=y_med, line_dash="dot", line_color="#37b6ff",
                annotation_text="Median xPts", annotation_position="top left",
            )
            fig.add_vline(
                x=x_med, line_dash="dot", line_color="#37b6ff",
                annotation_text="Median Price", annotation_position="top right",
            )

            fig.add_trace(go.Scatter(
                x=top_value["price"],
                y=top_value["predicted_pts"],
                mode="markers+text",
                text=top_value["player_name"].str.split().str[-1],
                textposition="top center",
                textfont=dict(size=10, color="#eaf2ff"),
                marker=dict(
                    size=12,
                    symbol="diamond",
                    color="rgba(39,232,167,0.15)",
                    line=dict(color="#27e8a7", width=1.8),
                ),
                name="Top Value",
                hoverinfo="skip",
            ))

            squad_pool = plot_df[plot_df["player_id"].isin(data["my_player_ids"])]
            if not squad_pool.empty:
                fig.add_trace(go.Scatter(
                    x=squad_pool["price"],
                    y=squad_pool["predicted_pts"],
                    mode="markers",
                    marker=dict(size=16, color="white", symbol="star",
                                line=dict(color="#27e8a7", width=2)),
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
                config={"displayModeBar": False, "responsive": True},
            )
            st.caption(f"Objective mode: {mode}. Diamonds mark top picks for the selected objective.")

            render_section_header(f"Top Picks by {mode}")
            top_strip = top_value.sort_values("objective_score", ascending=False).head(6).copy()
            strip_cols = st.columns(2 if compact_layout else 3)
            for i, (_, row) in enumerate(top_strip.iterrows(), 1):
                with strip_cols[(i - 1) % len(strip_cols)]:
                    objective_label = (
                        f"{float(row.get('value_index', 0.0)):.3f} value/£"
                        if mode == "Value"
                        else f"{float(row.get('predicted_pts', 0.0)):.2f} xPts"
                        if mode == "Ceiling"
                        else f"{float(row.get('reliability', row.get('chance_of_playing', 100) / 100)):.0%} reliability"
                    )
                    rank_tag = f"#{i}"
                    subtitle = f"{row['position']} | £{float(row['price']):.1f}M | {float(row['predicted_pts']):.2f} pts"
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
                            <div style='font-size:0.72rem; color:#9cb0d0; margin-top:0.35rem;'>
                                Run: {_safe_text(row.get('fixture_run_label', '?'))}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            shortlist = (
                top_value[[
                    "player_face", "team_badge", "player_name", "team_name", "position",
                    "price", "predicted_pts", "value_index", "objective_score", "fixture_run_label"
                ]]
                .rename(columns={
                    "player_face": "Face", "team_badge": "Badge", "player_name": "Player",
                    "team_name": "Team", "position": "Pos", "price": "£",
                    "predicted_pts": "xPts", "value_index": "Value/£",
                    "objective_score": "Objective", "fixture_run_label": "Run",
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
                        "Value/£": st.column_config.NumberColumn(format="%.3f"),
                        "Objective": st.column_config.NumberColumn(format="%.3f"),
                    },
                )

    with tab_bar:
        # Differentials - low ownership, high predicted pts
        diffs = get_differential_picks(others, bootstrap, top_n=15)
        if not diffs.empty:
            fig = px.bar(
                diffs,
                x="differential_score", y="player_name",
                orientation="h",
                color="position",
                hover_data=["team_name", "price", "predicted_pts",
                            "ownership_pct", "fixture_run_label"],
                color_discrete_map={
                    "Goalkeeper": "#27e8a7", "Defender": "#37b6ff",
                    "Midfielder": "#f6c90e", "Forward": "#ff5d73"
                },
                labels={"differential_score": "Differential Score",
                        "player_name": ""},
            )
            fig.update_layout(**PLOTLY_THEME, height=450,
                      margin=dict(l=10, r=10, t=30, b=30))
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
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
                p1_subtitle = f"£{p1['price']:.1f}M | {p1['predicted_pts']:.2f} pts"
                st.markdown(
                    f"<div class='fpl-card'>{player_identity_html(p1['player_name'], p1['team_name'], p1.get('player_face',''), p1.get('team_badge',''), p1_subtitle)}</div>",
                    unsafe_allow_html=True,
                )
            with id_col2:
                p2_subtitle = f"£{p2['price']:.1f}M | {p2['predicted_pts']:.2f} pts"
                st.markdown(
                    f"<div class='fpl-card'>{player_identity_html(p2['player_name'], p2['team_name'], p2.get('player_face',''), p2.get('team_badge',''), p2_subtitle)}</div>",
                    unsafe_allow_html=True,
                )

            metrics = [
                ("Predicted Pts",  p1["predicted_pts"],       p2["predicted_pts"]),
                ("Combined Score", p1["combined_score"],      p2["combined_score"]),
                ("Price £M",       p1["price"],               p2["price"]),
                ("Avg Difficulty", p1.get("avg_difficulty",3),p2.get("avg_difficulty",3)),
                ("Momentum",       p1.get("momentum_score",3),p2.get("momentum_score",3)),
                ("Value Score",    p1.get("value_score",0),   p2.get("value_score",0)),
                ("Blank GWs",      p1.get("blank_gws",0),     p2.get("blank_gws",0)),
                ("Double GWs",     p1.get("double_gws",0),    p2.get("double_gws",0)),
            ]

            labels  = [m[0] for m in metrics]
            vals_p1 = [float(m[1] or 0) for m in metrics]
            vals_p2 = [float(m[2] or 0) for m in metrics]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name=p1_name.split()[-1], x=labels, y=vals_p1,
                marker_color="#27e8a7",
            ))
            fig.add_trace(go.Bar(
                name=p2_name.split()[-1], x=labels, y=vals_p2,
                marker_color="#37b6ff",
            ))
            fig.update_layout(**PLOTLY_THEME, barmode="group", height=380,
                              margin=dict(l=10, r=10, t=30, b=60))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

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
            gap = p2_avg - p1_avg
            verdict = p1_name if gap > 0 else p2_name
            conf = float(np.clip(50 + abs(gap) * 18 + abs(p1_conf - p2_conf) * 0.2, 51, 93))
            reasons = [
                f"Avg difficulty delta: {gap:+.2f} (blank-aware)",
                f"Swing {p1_name.split()[-1]}: {p1_swing:+.2f} | {p2_name.split()[-1]}: {p2_swing:+.2f}",
                f"Blank penalty included ({p1_blanks} vs {p2_blanks})",
            ]
            render_decision_banner(
                title="Comparison Verdict",
                primary_action=f"Preferred: {verdict}",
                confidence=conf,
                reasons=reasons,
                risk_level="Medium" if abs(gap) < 0.35 else "Low",
            )

    with tab_table:
        disp_cols = ["player_face","team_badge","player_name","team_name","position","price",
                     "predicted_pts","combined_score","value_score",
                     "fixture_run_label","blank_gws","double_gws"]
        disp_pool = pool[disp_cols].rename(columns={
            "player_face":"Face","team_badge":"Badge",
            "player_name":"Player","team_name":"Team","position":"Pos",
            "price":"£","predicted_pts":"Pred","combined_score":"5GW",
            "value_score":"Val","fixture_run_label":"Run",
            "blank_gws":"Blanks","double_gws":"DGWs",
        }).sort_values("Pred", ascending=False)
        st.dataframe(
            disp_pool, use_container_width=True, hide_index=True,
            column_config={
                "Face": st.column_config.ImageColumn("Face", width="small"),
                "Badge": st.column_config.ImageColumn("Badge", width="small"),
            }
        )


elif page == "Captain Picker":

    render_section_header("Captain & Vice Captain Recommendation")

    # Chip status
    chips = ["Wildcard","Free Hit","Triple Captain","Bench Boost"]
    if compact_layout:
        for row in [chips[:2], chips[2:]]:
            row_cols = st.columns(2)
            for i, chip in enumerate(row):
                available = chip in available_chips
                row_cols[i].markdown(f"""
                <div class='kpi-block'>
                    <div class='kpi-label'>{chip}</div>
                    <div style='font-size:1.2rem; font-weight:800;
                                color:{"#27e8a7" if available else "#ff5d73"};'>
                        {'Available' if available else 'Used'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        chip_cols = st.columns(4)
        for i, chip in enumerate(chips):
            available = chip in available_chips
            chip_cols[i].markdown(f"""
            <div class='kpi-block'>
                <div class='kpi-label'>{chip}</div>
                <div style='font-size:1.2rem; font-weight:800;
                            color:{"#27e8a7" if available else "#ff5d73"};'>
                    {'Available' if available else 'Used'}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Compute captain scores
    cap_df = my_team.copy()
    cap_df["reliability"] = (
        cap_df["player_id"].astype("Int64").map(chance_map).fillna(100).astype(float) / 100.0
    )
    cap_df["upside"] = (
        cap_df["predicted_pts"].astype(float)
        + 0.5 * cap_df.get("double_gws", 0).fillna(0).astype(float)
        - 0.35 * cap_df.get("blank_gws", 0).fillna(0).astype(float)
    )
    cap_df["xpts_score"] = cap_df.apply(
        lambda r: xpts_captain_score(r, triple_captain), axis=1
    )
    blank_mask = (
        cap_df["is_blank_next_gw"].fillna(False).astype(bool)
        if "is_blank_next_gw" in cap_df.columns
        else pd.Series(False, index=cap_df.index)
    )
    cap_df["vc_score"] = (cap_df["predicted_pts"].astype(float) * cap_df["reliability"]).where(~blank_mask, 0.0)
    cap_df["captain_expected_return"] = cap_df["predicted_pts"].astype(float) * 2.0
    cap_df["captain_confidence"] = np.clip(45 + 30 * cap_df["reliability"] + 3 * cap_df["predicted_pts"], 40, 95)

    # Exclude blanks
    non_blank = cap_df[~blank_mask]

    top3_cap = non_blank.nlargest(3, "xpts_score")
    top_vc   = non_blank[~non_blank["player_id"].isin(
        top3_cap.iloc[:1]["player_id"]
    )].nlargest(1, "vc_score")

    if not top3_cap.empty:
        cap_row = top3_cap.iloc[0]
        vc_name = top_vc.iloc[0]["player_name"] if not top_vc.empty else "No clear VC"
        cap_conf = float(cap_row.get("captain_confidence", 60.0))
        cap_risk = (
            "Low" if float(cap_row.get("reliability", 1.0)) >= 0.85
            else "Medium" if float(cap_row.get("reliability", 1.0)) >= 0.70
            else "High"
        )
        dgw_note = "DGW upside available" if float(cap_row.get("double_gws", 0) or 0) > 0 else "No DGW boost"
        render_decision_banner(
            title="Captain Decision",
            primary_action=f"Captain {cap_row['player_name']} | VC {vc_name}",
            confidence=cap_conf,
            reasons=[
                f"Expected captain return: {float(cap_row.get('captain_expected_return', 0.0)):.1f}",
                f"Reliability: {float(cap_row.get('reliability', 1.0)):.0%} | Upside: {float(cap_row.get('upside', 0.0)):.1f}",
                dgw_note,
            ],
            risk_level=cap_risk,
        )

    # Captain podium
    st.markdown("**Captain Recommendations**")
    medals = ["Captain", "Vice Captain Option", "3rd Option"]
    cap_cols = st.columns(1 if compact_layout else 3)
    for i, (_, row) in enumerate(top3_cap.iterrows()):
        with cap_cols[i % len(cap_cols)]:
            mult = 3 if (triple_captain and i == 0) else 2
            dgw  = row.get("double_gws", 0) > 0
            cap_name = _safe_text(row.get("player_name", "Unknown"))
            cap_team = _safe_text(row.get("team_name", ""))
            cap_run = _safe_text(row.get("fixture_run_label", "?"))
            cap_face = _safe_text(row.get("player_face", ""))
            cap_badge = _safe_text(row.get("team_badge", ""))
            st.markdown(f"""
            <div class='fpl-card' style='border-color:{"#f6c90e" if i==0 else "#1f3459"};
                                         text-align:center;'>
                <div style='font-size:1.5rem; margin-bottom:0.5rem;'>
                    {medals[i]}
                </div>
                <div style='display:flex; justify-content:center; margin-bottom:0.4rem;'>
                    <img class='player-face' src='{cap_face}'
                         onerror="this.onerror=null;this.style.display='none';" />
                </div>
                <div style='display:flex; justify-content:center; align-items:center; gap:0.35rem;'>
                    <img class='team-badge' src='{cap_badge}'
                         onerror="this.onerror=null;this.style.display='none';" />
                    <div style='font-weight:800; font-size:1.05rem;'>
                        {cap_name}
                    </div>
                </div>
                <div style='color:#9cb0d0; font-size:0.8rem; margin-top:0.3rem;'>
                    {cap_team} | {cap_run}
                </div>
                <div style='font-family:Space Mono; font-size:1.4rem;
                            color:#27e8a7; margin-top:0.6rem;'>
                    {row["predicted_pts"]} pts
                </div>
                <div style='font-size:0.75rem; color:#37b6ff; margin-top:0.2rem;'>
                    = {round(row["predicted_pts"]*mult,2)} if captained
                    {"  DGW" if dgw else ""}
                    {"  TC 3x" if triple_captain and i==0 else ""}
                </div>
                <div style='display:flex; justify-content:center; gap:0.35rem; flex-wrap:wrap; margin-top:0.3rem;'>
                    <span class='xi-role'>Reliability {float(row.get("reliability",1.0))*100:.0f}%</span>
                    <span class='xi-role'>Upside {float(row.get("upside",0.0)):.1f}</span>
                    <span class='xi-role'>Conf {float(row.get("captain_confidence",60)):.0f}%</span>
                </div>
                <div style='font-size:0.7rem; color:#90a2be; margin-top:0.3rem;'>
                    xPts score: {row["xpts_score"]:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # VC recommendation
    if not top_vc.empty:
        vc_row = top_vc.iloc[0]
        chance = int(round(float(vc_row.get("reliability", 1.0) * 100)))
        st.markdown(f"""
        <div class='rec-box' style='margin-top:1rem;'>
            <div class='kpi-label'>VICE CAPTAIN (Reliability-Weighted)</div>
            <div style='font-weight:800; font-size:1.1rem; margin-top:0.3rem;'>
                {vc_row["player_name"]}
            </div>
            <div style='font-size:0.85rem; color:#b7c7df; margin-top:0.3rem;'>
                {vc_row["predicted_pts"]} predicted pts |
                {chance}% chance of playing |
                {vc_row.get("fixture_run_label","?")}
                - Chosen as most reliable backup in case captain doesn't play.
            </div>
            <div style='margin-top:0.45rem; color:#c9d8f0; font-size:0.8rem;'>
                Fallback logic: If captain fails to start, VC expected return = {float(vc_row.get("captain_expected_return",0.0)):.1f}.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    render_section_header("Full Squad xPts Ranking")

    cap_sorted = cap_df.sort_values("xpts_score", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cap_sorted["xpts_score"],
        y=cap_sorted["player_name"],
        orientation="h",
        marker=dict(
            color=cap_sorted["xpts_score"],
            colorscale=[[0, "#1f3459"], [0.5, "#37b6ff"], [1, "#27e8a7"]],
        ),
        hovertemplate="<b>%{y}</b><br>xPts Score: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_THEME, height=450,
        xaxis_title="xPts Captain Score",
        margin=dict(l=10, r=10, t=20, b=30),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    render_section_header("Captain Matrix: Upside vs Reliability")
    cap_matrix = cap_df.copy()
    fig2 = px.scatter(
        cap_matrix,
        x="reliability",
        y="upside",
        size="predicted_pts",
        color="position",
        hover_name="player_name",
        labels={"reliability": "Reliability", "upside": "Upside Score"},
        color_discrete_map={
            "Goalkeeper": "#27e8a7",
            "Defender": "#37b6ff",
            "Midfielder": "#ffb547",
            "Forward": "#ff5d73",
        },
    )
    fig2.update_layout(**PLOTLY_THEME, height=380, margin=dict(l=10, r=10, t=20, b=30))
    fig2.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    # Formation table
    st.divider()
    render_section_header("Formation Comparison")
    formations = score_all_formations(my_team)
    if formations:
        form_df = pd.DataFrame(formations).rename(columns={
            "formation": "Formation",
            "pred_pts": "Predicted Pts",
            "combined": "Combined Score"
        })
        form_df["Optimal"] = ["Best" if i == 0 else "" for i in range(len(form_df))]
        st.dataframe(form_df, use_container_width=True, hide_index=True)



elif page == "Season Tracker":

    render_section_header("Season Performance Tracker")

    value_data = track_squad_value(my_team, bootstrap, current_gw)
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
    if compact_layout:
        r1c1, r1c2 = st.columns(2)
        r1c1.metric("Current Squad Value", f"£{current_value_num:.1f}M")
        r1c2.metric("Baseline Value", f"£{baseline_value_num:.1f}M", f"GW{baseline_gw_num}")
        st.metric("Total Change", f"{change_sign}{total_change_num:.1f}M")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Squad Value", f"£{current_value_num:.1f}M")
        col2.metric("Baseline Value", f"£{baseline_value_num:.1f}M", f"GW{baseline_gw_num}")
        col3.metric("Total Change", f"{change_sign}{total_change_num:.1f}M")

    # Manager scorecard + period comparison
    gw_cols = sorted(
        [c for c in my_team.columns if c.startswith("gw") and c.endswith("_difficulty")],
        key=lambda c: int(c[2:].split("_")[0]),
    )
    window_vals = []
    for c in gw_cols:
        col = pd.to_numeric(my_team[c], errors="coerce").fillna(6.0)
        window_vals.append(float(col.mean()))
    if window_vals:
        last5 = window_vals[:5]
        prior5 = window_vals[5:10] if len(window_vals) >= 10 else window_vals[2:7]
        last5_avg = float(np.mean(last5)) if last5 else np.nan
        prior5_avg = float(np.mean(prior5)) if prior5 else np.nan
    else:
        last5_avg, prior5_avg = np.nan, np.nan

    transfer_file = "transfer_history.json"
    transfer_hit_rate = np.nan
    pred_calibration = np.nan
    try:
        import json
        from pathlib import Path
        if Path(transfer_file).exists():
            with open(transfer_file, encoding="utf-8") as f:
                t_hist = json.load(f)
            t_df_score = pd.DataFrame(t_hist)
            if not t_df_score.empty and "evaluated" in t_df_score.columns:
                eval_df = t_df_score[t_df_score["evaluated"] == True].copy()
                if not eval_df.empty and {"actual_gain", "predicted_gain"}.issubset(eval_df.columns):
                    transfer_hit_rate = float((eval_df["actual_gain"] > 0).mean() * 100.0)
                    pred_err = (eval_df["actual_gain"] - eval_df["predicted_gain"]).abs()
                    pred_calibration = float(max(0.0, 100.0 - pred_err.mean() * 25.0))
    except Exception:
        pass

    value_eff = float((current_value_num - baseline_value_num) / max(1.0, float(current_gw) - float(baseline_gw_num) + 1.0))
    render_stat_cards(
        [
            {"label": "Manager Scorecard", "value": "Decision Quality", "delta": "Transfer + calibration + value growth", "tone": "neutral"},
            {"label": "Transfer Hit Rate", "value": f"{transfer_hit_rate:.1f}%" if not np.isnan(transfer_hit_rate) else "N/A", "delta": "Evaluated positive gains", "tone": "positive" if (not np.isnan(transfer_hit_rate) and transfer_hit_rate >= 55) else "warning"},
            {"label": "Prediction Calibration", "value": f"{pred_calibration:.1f}%" if not np.isnan(pred_calibration) else "N/A", "delta": "Predicted vs realized", "tone": "positive" if (not np.isnan(pred_calibration) and pred_calibration >= 65) else "warning"},
            {"label": "Value Growth Efficiency", "value": f"{value_eff:+.2f}M/GW", "delta": "Squad value trend per GW", "tone": "positive" if value_eff >= 0 else "danger"},
        ],
        compact=compact_layout,
    )
    if not np.isnan(last5_avg) and not np.isnan(prior5_avg):
        render_section_header("Period Comparison: Last 5 vs Prior 5")
        cmp_df = pd.DataFrame(
            [
                {"Period": "Last 5 GWs", "Avg Fixture Difficulty": round(last5_avg, 2)},
                {"Period": "Prior 5 GWs", "Avg Fixture Difficulty": round(prior5_avg, 2)},
                {"Period": "Delta", "Avg Fixture Difficulty": round(last5_avg - prior5_avg, 2)},
            ]
        )
        render_insight_table(cmp_df, row_density="compact")
        worked = []
        hurt = []
        if last5_avg <= prior5_avg:
            worked.append("Fixture run improved versus prior period.")
        else:
            hurt.append("Fixture run hardened in the latest 5 GW window.")
        if not np.isnan(transfer_hit_rate):
            (worked if transfer_hit_rate >= 50 else hurt).append(f"Transfer hit rate at {transfer_hit_rate:.1f}%.")
        if value_eff >= 0:
            worked.append(f"Squad value trend positive at {value_eff:+.2f}M/GW.")
        else:
            hurt.append(f"Value trend negative at {value_eff:+.2f}M/GW.")
        render_section_header("What worked / what hurt")
        w_col, h_col = st.columns(2)
        with w_col:
            st.markdown("**Worked**")
            for w in worked or ["No clear positive pattern yet."]:
                st.markdown(f"- {w}")
        with h_col:
            st.markdown("**Hurt**")
            for h in hurt or ["No major drag detected yet."]:
                st.markdown(f"- {h}")

    history = value_data["history"]
    if len(history) > 1:
        gws_hist = sorted(history.keys(), key=int)
        vals_hist = [float(history[g]) for g in gws_hist]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[f"GW{g}" for g in gws_hist],
            y=vals_hist,
            mode="lines+markers",
            line=dict(color="#27e8a7", width=2.5),
            marker=dict(size=8, color="#27e8a7",
                        line=dict(color="white", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(56, 217, 150, 0.08)",
            hovertemplate="<b>%{x}</b><br>Value: £%{y:.1f}M<extra></extra>",
            name="Squad Value",
        ))
        fig.add_hline(
            y=baseline_value_num,
            line_dash="dash", line_color="#37b6ff",
            annotation_text=f"Baseline £{baseline_value_num:.1f}M",
            annotation_font=dict(color="#37b6ff"),
        )
        fig.update_layout(
            **PLOTLY_THEME, height=300,
            title=dict(text="Squad Value Over Time",
                       font=dict(color="#37b6ff", size=13, family="Space Mono")),
            yaxis_title="Value (£M)",
            margin=dict(l=10, r=10, t=50, b=30),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
    else:
        st.info("Run the app across multiple gameweeks to see your value trend.")

    st.divider()

    render_section_header("Transfer History & Model Accuracy")

    import json
    from pathlib import Path
    TRANSFER_LOG = "transfer_history.json"
    if Path(TRANSFER_LOG).exists():
        with open(TRANSFER_LOG, encoding="utf-8") as f:
            t_history = json.load(f)

        if t_history:
            t_df = pd.DataFrame(t_history)
            if "evaluated" not in t_df.columns:
                st.warning("Transfer history schema changed: missing 'evaluated' column.")
                evaluated = pd.DataFrame()
            else:
                evaluated = t_df[t_df["evaluated"] == True]

            if not evaluated.empty:
                # Accuracy chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=evaluated["player_in"],
                    y=evaluated["predicted_gain"],
                    name="Predicted", marker_color="#37b6ff",
                ))
                fig.add_trace(go.Bar(
                    x=evaluated["player_in"],
                    y=evaluated["actual_gain"],
                    name="Actual", marker_color="#27e8a7",
                ))
                fig.update_layout(
                    **PLOTLY_THEME, barmode="group", height=320,
                    title=dict(text="Transfer Prediction Accuracy",
                               font=dict(color="#37b6ff",size=13,family="Space Mono")),
                    xaxis_tickangle=-30,
                    margin=dict(l=10,r=10,t=50,b=80),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

                # Accuracy table
                evaluated_disp = evaluated[[
                    "gw","player_out","player_in",
                    "predicted_gain","actual_gain"
                ]].copy()
                evaluated_disp["result"] = evaluated_disp.apply(
                    lambda r: "Good" if r["actual_gain"] >= r["predicted_gain"]*0.7
                              else "Miss", axis=1
                )
                st.dataframe(
                    evaluated_disp.rename(columns={
                        "gw":"GW","player_out":"OUT","player_in":"IN",
                        "predicted_gain":"Predicted","actual_gain":"Actual",
                        "result":"Result"
                    }).sort_values("GW", ascending=False),
                    use_container_width=True, hide_index=True
                )
            else:
                pending = len(t_df[t_df["evaluated"] == False])
                st.info(f"{pending} transfer(s) logged but not yet evaluable "
                        f"(evaluation appears after enough completed GWs in the {FIXTURE_LOOKAHEAD}-GW horizon).")

            # All logged transfers
            with st.expander("All logged suggestions"):
                wanted = ["gw", "player_out", "player_in", "predicted_gain", "evaluated"]
                safe_cols = [c for c in wanted if c in t_df.columns]
                if len(safe_cols) >= 2:
                    st.dataframe(
                        t_df[safe_cols].rename(columns={
                            "gw":"GW","player_out":"OUT","player_in":"IN",
                            "predicted_gain":"Predicted Gain","evaluated":"Evaluated"
                        }),
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.warning("Transfer history schema is incomplete. Could not render full transfer log table.")
    else:
        st.info("No transfer history yet. Make some transfers via the Transfer Planner "
                "and they'll be tracked here automatically.")

    st.divider()

    render_section_header("Model Performance by Position")

    rmse_data = [{"Position": pos, "RMSE (pts)": round(rmse, 3),
                  "R2": round(data["models"].get(pos, {}).get("r2", 0), 3)}
                 for pos, rmse in rmse_map.items()]
    if rmse_data:
        rmse_df = pd.DataFrame(rmse_data)
        fig = px.bar(
            rmse_df, x="Position", y="RMSE (pts)",
            color="RMSE (pts)",
            color_continuous_scale=[[0,"#27e8a7"],[0.5,"#ffb547"],[1,"#ff5d73"]],
            text="RMSE (pts)",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(**PLOTLY_THEME, height=300, showlegend=False,
                          coloraxis_showscale=False,
                          margin=dict(l=10,r=10,t=20,b=30))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        st.caption("RMSE = Root Mean Squared Error. Lower = more accurate model. "
                   "GK models typically most accurate due to consistent playing time.")

elif page == "AI Analyst":

    render_section_header("AI Analyst | Powered by Groq + Live Data")

    if not ANALYST_AVAILABLE:
        st.error(f"Phase 7 backend not available: {ANALYST_ERROR}")
        st.info("Run: pip install groq feedparser newsapi-python understat nest_asyncio")
        st.stop()

    status_items = [
        ("LLM (Groq)", ANALYST_STATUS.get("groq", False), "Ready", "Not installed"),
        ("NewsAPI", ANALYST_STATUS.get("newsapi", False), "Ready", "No key"),
        ("RSS Feeds", ANALYST_STATUS.get("feedparser", False), "Ready", "Not installed"),
        ("Understat xG", ANALYST_STATUS.get("understat", False), "Ready", "Not installed"),
        ("The Odds API", ANALYST_STATUS.get("odds_api", False), "Ready", "No key"),
    ]
    status_cols = st.columns(len(status_items))
    for i, (label, ok, ok_text, fail_text) in enumerate(status_items):
        status_cols[i].markdown(
            f"<div class='kpi-block'><div class='kpi-label'>{label}</div>"
            f"<div style='color:{'#27e8a7' if ok else '#ffb547'};font-weight:800;'>"
            f"{ok_text if ok else fail_text}</div></div>",
            unsafe_allow_html=True,
        )

    if ANALYST_STATUS.get("odds_api", False):
        st.caption(get_odds_usage_summary())

    # Optional proactive alerts sourced from Phase 7 logic.
    try:
        proactive_alerts = generate_proactive_alerts(
            my_team=my_team,
            xi_result=xi_result,
            news_map=news_map,
            chance_map=chance_map,
            chip_info=chip_info,
            deadline_status={},
            current_gw=current_gw,
        )
    except Exception:
        proactive_alerts = []
    if proactive_alerts:
        render_section_header("Proactive Alerts")
        for alert in proactive_alerts[:5]:
            level = str(alert.get("level", "info")).lower()
            title = str(alert.get("title", "Alert"))
            message = str(alert.get("message", ""))
            if level == "critical":
                st.error(f"{title}: {message}")
            elif level == "warning":
                st.warning(f"{title}: {message}")
            else:
                st.info(f"{title}: {message}")

    st.divider()

    if "analyst_messages" not in st.session_state:
        st.session_state["analyst_messages"] = []
    if "analyst_sources" not in st.session_state:
        st.session_state["analyst_sources"] = []
    if "analyst_context" not in st.session_state:
        st.session_state["analyst_context"] = {}

    render_section_header("Quick Questions")
    q_cols = st.columns(2 if compact_layout else 4)
    quick_q = None
    for i, q in enumerate(QUICK_QUESTIONS[:8]):
        with q_cols[i % len(q_cols)]:
            if st.button(q, use_container_width=True, key=f"qq_{i}"):
                quick_q = q

    st.divider()

    with st.container():
        for msg in st.session_state["analyst_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources_display"):
                    with st.expander("Sources & Confidence", expanded=False):
                        st.markdown(msg["sources_display"])
                        conf_label = msg.get("confidence_label", "?")
                        conf_score = msg.get("confidence_score", 0)
                        conf_color = (
                            "#27e8a7" if conf_label == "HIGH"
                            else "#ffb547" if conf_label == "MEDIUM"
                            else "#ff5d73"
                        )
                        st.markdown(
                            f"<div class='kpi-block' style='display:inline-block; padding:0.5rem 1rem;'>"
                            f"<div class='kpi-label'>Source Confidence</div>"
                            f"<div style='color:{conf_color}; font-weight:800; font-size:1.2rem;'>"
                            f"{conf_label} ({conf_score:.0f}%)</div></div>",
                            unsafe_allow_html=True,
                        )

    st.divider()
    user_input = st.chat_input("Ask anything about your squad, transfers, captain, injuries...")

    question = quick_q or user_input
    if question:
        st.session_state["analyst_messages"].append({
            "role": "user",
            "content": question,
        })
        st.session_state["analyst_messages"] = st.session_state["analyst_messages"][-30:]

        llm_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state["analyst_messages"][:-1]
        ]

        try:
            cached_ilp_1 = cached_ilp_transfers(my_team, others, float(bank_balance), n_transfers=1)
            cached_ilp_2 = cached_ilp_transfers(my_team, others, float(bank_balance), n_transfers=2)
            cached_roll = get_rolling_transfer_advice(
                my_team, others, bank_balance, transfers_made,
                chip_info, current_gw, ilp_result=cached_ilp_1
            )
            cached_hits = get_hit_transfer_analysis(
                my_team, others, bank_balance, transfers_made
            )
        except Exception:
            cached_ilp_1 = cached_ilp_2 = cached_roll = None
            cached_hits = []

        with st.spinner("Fetching live data and consulting the analyst..."):
            try:
                result = run_analyst(
                    question=question,
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
                    chat_history=llm_history,
                    ilp_1=cached_ilp_1,
                    ilp_2=cached_ilp_2,
                    roll_advice=cached_roll,
                    hit_transfers=cached_hits,
                )
                conf_label, conf_score = result["confidence"]
                st.session_state["analyst_messages"].append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources_display": result["source_display"],
                    "confidence_label": conf_label,
                    "confidence_score": conf_score,
                })
            except Exception as e:
                st.session_state["analyst_messages"].append({
                    "role": "assistant",
                    "content": f"Error running analyst: {str(e)}",
                    "sources_display": "",
                    "confidence_label": "LOW",
                    "confidence_score": 0,
                })

        st.rerun()

    if st.session_state["analyst_messages"]:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state["analyst_messages"] = []
            st.rerun()

    if not st.session_state["analyst_messages"]:
        st.markdown(
            """
            <div class='fpl-card' style='border-color:#37b6ff40;'>
                <div class='kpi-label'>HOW IT WORKS</div>
                <div style='font-size:0.9rem; color:#c8d8f0; margin-top:0.5rem; line-height:1.6;'>
                    The AI Analyst combines your live squad data with real-time news from
                    multiple sources to give you grounded, data-driven FPL advice.
                    <br><br>
                    <b>Sources consulted:</b> FPL API | API-Football lineups | NewsAPI |
                    BBC Sport | Sky Sports | Google News | Odds API | Understat
                    <br><br>
                    <b>Click a quick question above or type your own below.</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption(
        "AI Analyst powered by Groq (Llama 3.3 70B) | "
        "Responses grounded in live FPL + news data | "
        "Always verify before deadline"
    )



st.caption("FPL AI ASSISTANT | PHASES 1-4 BACKEND | BUILT WITH STREAMLIT + PLOTLY")
st.caption("Always verify bank balance in the FPL app before confirming transfers.")





