import pandas as pd

from db.session import get_connection


def get_latest_ready_snapshot() -> dict | None:
    """
    Returns latest ready snapshot metadata row as a dict, or None if no ready snapshot exists.
    """
    conn = get_connection(autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    created_at,
                    status,
                    season,
                    current_gw,
                    pipeline_version,
                    model_version,
                    feature_version,
                    build_duration_sec,
                    row_count_players,
                    row_count_player_fixture,
                    row_count_team_fixture
                FROM model_snapshots
                WHERE status = 'ready'
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if not row:
                return None

            cols = [
                "id",
                "created_at",
                "status",
                "season",
                "current_gw",
                "pipeline_version",
                "model_version",
                "feature_version",
                "build_duration_sec",
                "row_count_players",
                "row_count_player_fixture",
                "row_count_team_fixture",
            ]
            return dict(zip(cols, row))
    finally:
        conn.close()


def load_snapshot_player_predictions(snapshot_id: int) -> pd.DataFrame:
    conn = get_connection(autocommit=False)
    try:
        query = """
            SELECT *
            FROM player_predictions_snapshot
            WHERE snapshot_id = %s
        """
        return pd.read_sql_query(query, conn, params=(snapshot_id,))
    finally:
        conn.close()


def load_snapshot_team_fixture_run(snapshot_id: int) -> pd.DataFrame:
    conn = get_connection(autocommit=False)
    try:
        query = """
            SELECT *
            FROM team_fixture_run_snapshot
            WHERE snapshot_id = %s
        """
        return pd.read_sql_query(query, conn, params=(snapshot_id,))
    finally:
        conn.close()


def load_snapshot_player_fixture_features(snapshot_id: int) -> pd.DataFrame:
    conn = get_connection(autocommit=False)
    try:
        query = """
            SELECT *
            FROM player_fixture_features_snapshot
            WHERE snapshot_id = %s
        """
        return pd.read_sql_query(query, conn, params=(snapshot_id,))
    finally:
        conn.close()


def load_snapshot_model_metrics(snapshot_id: int) -> pd.DataFrame:
    conn = get_connection(autocommit=False)
    try:
        query = """
            SELECT *
            FROM model_metrics_snapshot
            WHERE snapshot_id = %s
        """
        return pd.read_sql_query(query, conn, params=(snapshot_id,))
    finally:
        conn.close()