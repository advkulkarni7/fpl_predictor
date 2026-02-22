# FPL Dashboard Redesign Spec

## Objective
Turn `fpl_dashboard.py` from a data-heavy prototype into a decision-first product with:
- clear hierarchy
- consistent components
- strong recommendation UX
- responsive behavior
- stable visual quality

Success metric: a user can answer "What should I do this GW?" in under 10 seconds on each main page.

---

## Design Principles
1. One page = one primary decision.
2. Show recommendation first, details second, raw data third.
3. Every recommendation must include `why`, `confidence`, `risk`.
4. Reuse components; no one-off styling blocks.
5. Keep interactions fast and predictable on desktop and mobile.

---

## Component Contracts

### `DecisionBanner`
Purpose: top-level action summary.
Props:
- `title: str`
- `primary_action: str`
- `confidence: float (0-100)`
- `reasons: list[str]`
- `risk_level: "Low" | "Medium" | "High"`

### `StatCard`
Purpose: standardized KPI card.
Props:
- `label: str`
- `value: str`
- `delta: str | None`
- `tone: "neutral" | "positive" | "warning" | "danger"`

### `PlayerChip`
Purpose: reusable player identity strip.
Props:
- `player_name: str`
- `team_name: str`
- `face_url: str`
- `badge_url: str`
- `subtitle: str`
- `tags: list[str]` (`C`, `VC`, `DGW`, `BLK`, etc.)

### `RecommendationCard`
Purpose: transfer/captain recommendations.
Props:
- `headline: str`
- `impact_now: float`
- `impact_horizon: float`
- `confidence: float`
- `risk_notes: list[str]`
- `supporting_points: list[str]`

### `InsightTable`
Purpose: standardized dataframes.
Props:
- `df: pd.DataFrame`
- `column_config: dict`
- `default_sort: tuple[column, asc]`
- `row_density: "compact" | "normal"`

---

## Page-by-Page Build Plan

## 1) My Squad
Primary question: "What is my best XI and how risky is it?"

Changes:
1. Add `DecisionBanner` at top:
- best formation
- expected points
- confidence/risk.
2. Replace current KPI row with `StatCard` set:
- Expected Pts
- Risk-Adjusted Pts
- Players at Risk
- Bench Cover Score
- Bank.
3. Keep lineup board, but add click-target details panel:
- next fixture
- minutes/chance
- backup options.
4. Add concise "Formation alternatives" mini-table with deltas.
5. Replace injury list with ranked urgency table.

Files:
- `fpl_dashboard.py` (My Squad block + helper render functions)

Acceptance:
- No raw HTML shown in UI.
- All currency symbols render as `£`.
- Page has one top recommendation and one clear fallback.

---

## 2) Fixture Planner
Primary question: "Which teams should I target/avoid in next 2-5 GWs?"

Changes:
1. Keep current heatmap core.
2. Add top action panel:
- `Target now` (top 3)
- `Avoid now` (bottom 3)
- `Best 2-GW swing`.
3. Add sort mode toggle:
- easiest overall
- easiest next 2
- blank risk.
4. Keep custom team selection and presets.
5. Add two-team compare mode (delta table).

Files:
- `fpl_dashboard.py` (Fixture Planner section)

Acceptance:
- Heatmap remains responsive with 20 teams.
- Custom team mode and default mode work identically.

---

## 3) Transfer Planner
Primary question: "Should I transfer now, and which move gives best net value?"

Changes:
1. Add recommendation stack:
- Safe move
- Aggressive move
- Differential move.
2. Every transfer card must show:
- next GW impact
- 5GW impact
- cost
- confidence
- key risks.
3. Add `Before vs After XI` compact preview.
4. Standardize tab naming:
- `1 FT`
- `2 FT`
- `-4 Hit`.

Files:
- `fpl_dashboard.py` (Transfer Planner block)
- optional: `fpl_phase3_constraints.py` (expose confidence/risk fields)

Acceptance:
- No duplicated/contradictory metric labels.
- Recommendation includes explanation bullets.

---

## 4) Player Explorer
Primary question: "Which player best fits my objective?"

Changes:
1. Add mode toggle:
- `Value`
- `Ceiling`
- `Safety`.
2. Replace generic scatter default with objective-aware chart config.
3. Add "Top picks by mode" card strip.
4. Keep comparison tab but add automatic winner verdict block.
5. Sticky filters + reset button.

Files:
- `fpl_dashboard.py` (Player Explorer block)

Acceptance:
- Filters persist in session state.
- Chart + shortlist always agree on top candidates.

---

## 5) Captain Picker
Primary question: "Who is captain and what is the fallback?"

Changes:
1. Top 3 options with:
- expected captain return
- reliability score
- upside score
- confidence.
2. Explicit fallback logic card:
- "If captain does not start → VC expected return".
3. Add captain matrix plot (upside vs reliability).

Files:
- `fpl_dashboard.py` (Captain Picker block)
- optional: `fpl_phase4_optimizer.py` (if adding confidence calc inputs)

Acceptance:
- One clear captain recommendation with confidence and reason.

---

## 6) Season Tracker
Primary question: "Am I improving decision quality over time?"

Changes:
1. Add Manager Scorecard:
- transfer hit rate
- prediction calibration
- value growth efficiency.
2. Add period comparison:
- last 5 GWs vs prior 5.
3. Add "What worked / what hurt" summary bullets.

Files:
- `fpl_dashboard.py` (Season Tracker block)
- optional: `fpl_phase3_constraints.py` (additional historical metrics)

Acceptance:
- Trend charts have clear narrative labels.
- Scorecard values map to visible data sources.

---

## Shared Engineering Tasks
1. Add render helper functions in `fpl_dashboard.py`:
- `render_decision_banner`
- `render_stat_cards`
- `render_recommendation_card`
- `render_insight_table`.
2. Replace inline one-off HTML snippets with helper usage.
3. Remove duplicated color literals; use CSS tokens only.
4. Ensure all plotly charts pass `config={"displayModeBar": False, "responsive": True}`.
5. Add encoding guard:
- open/save UTF-8 only
- eliminate mojibake fragments.

---

## Rollout Plan

## Phase 1 (Stability + System)
1. Encoding cleanup pass.
2. Shared component helpers.
3. Top context bar + DecisionBanner foundation.

## Phase 2 (High Impact UX)
1. My Squad redesign.
2. Transfer Planner redesign.

## Phase 3 (Discovery UX)
1. Player Explorer mode system.
2. Fixture Planner action panel + compare mode.

## Phase 4 (Decision Intelligence)
1. Captain confidence framework.
2. Season scorecard + calibration visuals.

---

## QA Checklist
1. Desktop widths: 1366, 1440.
2. Tablet width: 1024.
3. Mobile widths: 768, 430, 390.
4. No raw HTML visible anywhere.
5. No broken encoding symbols (`Â£`, `â€“`, etc.).
6. No runtime errors across all page tabs.
7. Time-to-first-interaction < 2s after cached load.

---

## Suggested Next Implementation Ticket
`Ticket-01`: Phase 1 foundation in `fpl_dashboard.py`
1. Add component render helpers.
2. Introduce global top context bar.
3. Replace existing KPI rows on My Squad + Transfer Planner with `StatCard`.
4. Add smoke tests by running app and checking all pages load without exceptions.
