import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from db.session import get_connection

READY_SNAPSHOT_RETENTION = 7
FAILED_SNAPSHOT_RETENTION = 10


def apply_snapshot_retention(conn, ready_keep: int = READY_SNAPSHOT_RETENTION, failed_keep: int = FAILED_SNAPSHOT_RETENTION) -> None:
    """Keep only the newest ready/failed snapshots."""
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM model_snapshots
            WHERE status = 'ready'
              AND id NOT IN (
                SELECT id FROM model_snapshots
                WHERE status = 'ready'
                ORDER BY created_at DESC
                LIMIT %s
              )
            """,
            (ready_keep,),
        )
        cur.execute(
            """
            DELETE FROM model_snapshots
            WHERE status = 'failed'
              AND id NOT IN (
                SELECT id FROM model_snapshots
                WHERE status = 'failed'
                ORDER BY created_at DESC
                LIMIT %s
              )
            """,
            (failed_keep,),
        )


def apply_schema(conn) -> None:
    schema_path = Path(__file__).resolve().parents[1] / "db" / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(schema_sql)


def create_snapshot_run(conn, *, season: str | None = None, current_gw: int | None = None) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO model_snapshots (
                status, season, current_gw, pipeline_version, model_version, feature_version
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            ("building", season, current_gw, "mvp-v1", "phase-models", "phase-features"),
        )
        return int(cur.fetchone()[0])


def mark_snapshot_ready(
    conn,
    snapshot_id: int,
    *,
    current_gw: int | None,
    build_duration_sec: int,
    row_count_players: int,
    row_count_player_fixture: int,
    row_count_team_fixture: int,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE model_snapshots
            SET status = 'ready',
                current_gw = COALESCE(%s, current_gw),
                build_duration_sec = %s,
                row_count_players = %s,
                row_count_player_fixture = %s,
                row_count_team_fixture = %s,
                error_message = NULL
            WHERE id = %s
            """,
            (
                current_gw,
                build_duration_sec,
                row_count_players,
                row_count_player_fixture,
                row_count_team_fixture,
                snapshot_id,
            ),
        )


def mark_snapshot_failed(conn, snapshot_id: int, *, error_message: str, build_duration_sec: int | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE model_snapshots
            SET status = 'failed',
                error_message = %s,
                build_duration_sec = COALESCE(%s, build_duration_sec)
            WHERE id = %s
            """,
            (error_message[:4000], build_duration_sec, snapshot_id),
        )


def _insert_many(conn, sql: str, rows: list[tuple]) -> None:
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(sql, rows)


def _to_none(val):
    if pd.isna(val):
        return None
    if hasattr(val, "item"):
        try:
            return val.item()
        except Exception:
            return val
    return val


def _safe_bool(val):
    if val is None or pd.isna(val):
        return None
    return bool(val)


def _build_player_prediction_rows(snapshot_id: int, enriched_df: pd.DataFrame) -> list[tuple]:
    rows: list[tuple] = []
    wanted = [
        "predicted_pts", "expected_pts", "pts_low", "pts_high", "captain_ev", "p_plays_full",
        "predicted_price_change", "combined_score", "value_score", "avg_difficulty",
        "fixture_run_label", "blank_gws", "double_gws", "is_blank_next_gw", "momentum_score",
    ]
    safe_cols = [c for c in wanted if c in enriched_df.columns]

    for _, r in enriched_df.iterrows():
        raw_payload = {c: _to_none(r[c]) for c in safe_cols}
        rows.append(
            (
                snapshot_id,
                int(r["player_id"]),
                _to_none(r.get("player_name")),
                int(r["team_id"]) if "team_id" in enriched_df.columns and not pd.isna(r["team_id"]) else None,
                _to_none(r.get("team_name")),
                _to_none(r.get("position")),
                _to_none(r.get("price")),
                _to_none(r.get("predicted_pts")),
                _to_none(r.get("expected_pts")),
                _to_none(r.get("pts_low")),
                _to_none(r.get("pts_high")),
                _to_none(r.get("captain_ev")),
                _to_none(r.get("p_plays_full")),
                _to_none(r.get("predicted_price_change")),
                _to_none(r.get("combined_score")),
                _to_none(r.get("value_score")),
                _to_none(r.get("avg_difficulty")),
                _to_none(r.get("fixture_run_label")),
                _to_none(r.get("blank_gws")),
                _to_none(r.get("double_gws")),
                _safe_bool(r.get("is_blank_next_gw")),
                _to_none(r.get("momentum_score")),
                json.dumps(raw_payload),
            )
        )
    return rows


def _build_team_fixture_rows(snapshot_id: int, fixture_run_df: pd.DataFrame) -> list[tuple]:
    rows: list[tuple] = []
    for _, r in fixture_run_df.iterrows():
        rows.append(
            (
                snapshot_id,
                int(r["team_id"]),
                int(r["gw"]),
                _to_none(r.get("opponent")),
                int(r["opponent_id"]) if "opponent_id" in fixture_run_df.columns and not pd.isna(r.get("opponent_id")) else None,
                _to_none(r.get("difficulty")),
                _safe_bool(r.get("is_home")),
                _safe_bool(r.get("is_blank")),
                _safe_bool(r.get("is_double")),
            )
        )
    return rows


def _build_player_fixture_rows(snapshot_id: int, enriched_df: pd.DataFrame, current_gw: int, gw_lookahead: int) -> list[tuple]:
    rows: list[tuple] = []
    for _, r in enriched_df.iterrows():
        pid = int(r["player_id"])
        for gw in range(current_gw + 1, current_gw + 1 + gw_lookahead):
            opp_col = f"gw{gw}_opponent"
            diff_col = f"gw{gw}_difficulty"
            home_col = f"gw{gw}_home"
            if opp_col not in enriched_df.columns and diff_col not in enriched_df.columns:
                continue
            opp = _to_none(r.get(opp_col))
            diff = _to_none(r.get(diff_col))
            home = _safe_bool(r.get(home_col)) if home_col in enriched_df.columns else None
            opp_text = str(opp).upper() if opp is not None else ""
            is_blank = opp_text in {"BLANK", "B"} or opp is None
            is_double = ("&" in str(opp)) if opp is not None else False
            rows.append(
                (
                    snapshot_id,
                    pid,
                    int(gw),
                    opp,
                    None,  # opponent_team_id not readily available from wide cols
                    diff,
                    home,
                    is_blank,
                    is_double,
                    None,  # fixture_pts_proxy optional
                )
            )
    return rows


def _build_model_metric_rows(snapshot_id: int, models: dict) -> list[tuple]:
    rows: list[tuple] = []
    for pos, info in (models or {}).items():
        rows.append(
            (
                snapshot_id,
                str(pos),
                _to_none(info.get("rmse")),
                _to_none(info.get("r2")),
                "xgboost" if info.get("model") is not None else None,
                None,
            )
        )
    return rows


def write_snapshot_tables(
    conn,
    *,
    snapshot_id: int,
    enriched_df: pd.DataFrame,
    fixture_run_df: pd.DataFrame,
    models: dict,
    current_gw: int,
    gw_lookahead: int,
) -> tuple[int, int, int]:
    pred_rows = _build_player_prediction_rows(snapshot_id, enriched_df)
    team_fix_rows = _build_team_fixture_rows(snapshot_id, fixture_run_df)
    player_fix_rows = _build_player_fixture_rows(snapshot_id, enriched_df, current_gw, gw_lookahead)
    metric_rows = _build_model_metric_rows(snapshot_id, models)

    _insert_many(
        conn,
        """
        INSERT INTO player_predictions_snapshot (
            snapshot_id, player_id, player_name, team_id, team_name, position, price,
            predicted_pts, expected_pts, pts_low, pts_high, captain_ev, p_plays_full,
            predicted_price_change, combined_score, value_score, avg_difficulty,
            fixture_run_label, blank_gws, double_gws, is_blank_next_gw, momentum_score, raw_json
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s::jsonb
        )
        """,
        pred_rows,
    )
    _insert_many(
        conn,
        """
        INSERT INTO team_fixture_run_snapshot (
            snapshot_id, team_id, gw, opponent, opponent_team_id, difficulty,
            is_home, is_blank, is_double
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        team_fix_rows,
    )
    _insert_many(
        conn,
        """
        INSERT INTO player_fixture_features_snapshot (
            snapshot_id, player_id, gw, opponent, opponent_team_id, difficulty,
            is_home, is_blank, is_double, fixture_pts_proxy
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        player_fix_rows,
    )
    _insert_many(
        conn,
        """
        INSERT INTO model_metrics_snapshot (
            snapshot_id, position, rmse, r2, model_name, n_train_rows
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """,
        metric_rows,
    )
    return len(pred_rows), len(player_fix_rows), len(team_fix_rows)


def run_global_pipeline():
    # Imports are local to keep CLI startup fast and surface import/config errors clearly.
    from fpl_phase1_model import (
        fetch_bootstrap,
        fetch_fixtures,
        fetch_current_gw,
        build_player_history_df,
        build_current_features,
        train_models,
        train_component_models,
        predict_component_pts,
        train_price_model,
        add_price_predictions,
        compute_expected_pts,
    )
    from fpl_phase2_fixtures import (
        FIXTURE_LOOKAHEAD,
        build_custom_difficulty,
        build_team_form,
        build_opponent_scoring_map,
        build_cs_probability_map,
        build_fixture_run,
        build_player_fixture_scores,
    )

    bootstrap = fetch_bootstrap()
    fixtures_df = fetch_fixtures()
    current_gw = fetch_current_gw(bootstrap)
    history_df = build_player_history_df(bootstrap, refresh=False)
    models = train_models(history_df)
    pred_df = build_current_features(
        bootstrap,
        fixtures_df,
        history_df,
        models,
        current_gw,
        # Include all players in the global snapshot so any user's squad can be resolved
        # even if a player is currently flagged/injured and would normally be filtered.
        my_player_ids=[int(p.get("id", 0)) for p in bootstrap.get("elements", [])],
    )

    # Advanced enrichments are optional: continue if any step fails.
    try:
        comp_models = train_component_models(history_df)
        pred_df = predict_component_pts(comp_models, pred_df)
    except Exception:
        pass
    try:
        pred_df = compute_expected_pts(pred_df)
    except Exception:
        pass
    try:
        price_model_info = train_price_model(history_df)
        pred_df = add_price_predictions(price_model_info, pred_df)
    except Exception:
        pass

    custom_diff = build_custom_difficulty(history_df, bootstrap)
    team_form_map = build_team_form(history_df, bootstrap)
    opp_scoring_map = build_opponent_scoring_map(history_df)
    try:
        cs_prob_map = build_cs_probability_map(history_df)
    except Exception:
        cs_prob_map = {}

    fixture_run_df = build_fixture_run(
        bootstrap,
        fixtures_df,
        current_gw,
        custom_difficulty=custom_diff,
        gw_lookahead=FIXTURE_LOOKAHEAD,
    )
    enriched_df = build_player_fixture_scores(
        pred_df,
        fixture_run_df,
        current_gw,
        team_form_map,
        opp_scoring_map,
        gw_lookahead=FIXTURE_LOOKAHEAD,
        cs_probability_map=cs_prob_map,
    )

    return {
        "bootstrap": bootstrap,
        "fixtures_df": fixtures_df,
        "current_gw": current_gw,
        "history_df": history_df,
        "models": models,
        "fixture_run_df": fixture_run_df,
        "enriched_df": enriched_df,
        "gw_lookahead": FIXTURE_LOOKAHEAD,
    }


def main() -> int:
    if not os.getenv("DATABASE_URL"):
        print("ERROR: DATABASE_URL is not set", file=sys.stderr)
        return 1

    started_at = time.time()
    print(f"[{datetime.now(timezone.utc).isoformat()}] Snapshot builder started")
    print("DATABASE_URL present: yes")
    print(f"Retention policy: ready={READY_SNAPSHOT_RETENTION}, failed={FAILED_SNAPSHOT_RETENTION}")

    conn = None
    snapshot_id = None
    try:
        conn = get_connection(autocommit=False)
        print("Connected to database")

        apply_schema(conn)
        conn.commit()
        print("Schema applied/verified successfully")

        pipeline = run_global_pipeline()
        current_gw = int(pipeline["current_gw"])
        season = str(datetime.now(timezone.utc).year)

        snapshot_id = create_snapshot_run(conn, season=season, current_gw=current_gw)
        conn.commit()
        print(f"Created snapshot run: id={snapshot_id} (status=building)")

        rc_players, rc_player_fixture, rc_team_fixture = write_snapshot_tables(
            conn,
            snapshot_id=snapshot_id,
            enriched_df=pipeline["enriched_df"],
            fixture_run_df=pipeline["fixture_run_df"],
            models=pipeline["models"],
            current_gw=current_gw,
            gw_lookahead=int(pipeline["gw_lookahead"]),
        )

        duration = int(time.time() - started_at)
        mark_snapshot_ready(
            conn,
            snapshot_id,
            current_gw=current_gw,
            build_duration_sec=duration,
            row_count_players=rc_players,
            row_count_player_fixture=rc_player_fixture,
            row_count_team_fixture=rc_team_fixture,
        )
        apply_snapshot_retention(conn)
        conn.commit()

        print(f"Snapshot {snapshot_id} marked ready")
        print(f"Wrote rows: players={rc_players}, player_fixture={rc_player_fixture}, team_fixture={rc_team_fixture}")
        print("Retention cleanup applied")
        return 0

    except Exception as e:
        duration = int(time.time() - started_at)
        if conn is not None:
            try:
                if snapshot_id is not None:
                    mark_snapshot_failed(conn, snapshot_id, error_message=str(e), build_duration_sec=duration)
                    apply_snapshot_retention(conn)
                    conn.commit()
                    print(f"Snapshot {snapshot_id} marked failed", file=sys.stderr)
                else:
                    conn.rollback()
            except Exception:
                conn.rollback()
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
