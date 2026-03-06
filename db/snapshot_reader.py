from __future__ import annotations

from typing import Any

import pandas as pd

from db.session import get_connection


def _fetch_all_df(conn, sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [d.name for d in cur.description] if cur.description else []
    return pd.DataFrame(rows, columns=cols)


def get_latest_ready_snapshot() -> dict | None:
    """Return metadata for the newest ready/degraded snapshot, or None."""
    conn = None
    try:
        conn = get_connection(autocommit=True)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, status, season, current_gw,
                       pipeline_version, model_version, feature_version,
                       build_duration_sec, error_message,
                       row_count_players, row_count_player_fixture, row_count_team_fixture
                FROM model_snapshots
                WHERE status IN ('ready', 'degraded')
                ORDER BY created_at DESC, CASE WHEN status = 'ready' THEN 0 ELSE 1 END
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [d.name for d in cur.description]
            return dict(zip(cols, row))
    except Exception:
        return None
    finally:
        if conn is not None:
            conn.close()


def load_snapshot_bundle(snapshot_id: int) -> dict:
    """Load snapshot tables needed by the dashboard."""
    conn = None
    try:
        conn = get_connection(autocommit=True)
        predictions_df = _fetch_all_df(
            conn,
            """
            SELECT *
            FROM player_predictions_snapshot
            WHERE snapshot_id = %s
            """,
            (int(snapshot_id),),
        )
        player_fixture_df = _fetch_all_df(
            conn,
            """
            SELECT *
            FROM player_fixture_features_snapshot
            WHERE snapshot_id = %s
            """,
            (int(snapshot_id),),
        )
        team_fixture_df = _fetch_all_df(
            conn,
            """
            SELECT *
            FROM team_fixture_run_snapshot
            WHERE snapshot_id = %s
            """,
            (int(snapshot_id),),
        )
        model_metrics_df = _fetch_all_df(
            conn,
            """
            SELECT snapshot_id, position, rmse, r2, naive_baseline_rmse, beats_baseline, model_name,
                   n_train_rows, shap_features
            FROM model_metrics_snapshot
            WHERE snapshot_id = %s
            """,
            (int(snapshot_id),),
        )
        return {
            "predictions_df": predictions_df,
            "player_fixture_df": player_fixture_df,
            "team_fixture_df": team_fixture_df,
            "model_metrics_df": model_metrics_df,
        }
    finally:
        if conn is not None:
            conn.close()


def load_latest_ready_snapshot_bundle() -> tuple[dict | None, dict | None]:
    """Return (snapshot_meta, bundle) for latest ready snapshot, or (None, None)."""
    meta = get_latest_ready_snapshot()
    if not meta:
        return None, None
    try:
        bundle = load_snapshot_bundle(int(meta["id"]))
    except Exception:
        return meta, None
    return meta, bundle
