import json
from datetime import datetime, timezone

from db.session import get_connection


def get_cached_team_context(team_id: int, event_gw: int, max_age_minutes: int = 30) -> dict | None:
    """
    Return cached team picks/transfer info if cache row exists and is fresh enough.
    Returns dict with parsed JSON payloads, or None.
    """
    conn = get_connection(autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    team_id,
                    event_gw,
                    fetched_at,
                    picks_json,
                    transfer_info_json,
                    status,
                    error_message
                FROM fpl_team_cache
                WHERE team_id = %s
                  AND event_gw = %s
                  AND fetched_at >= (NOW() - (%s * INTERVAL '1 minute'))
                LIMIT 1
                """,
                (team_id, event_gw, max_age_minutes),
            )
            row = cur.fetchone()
            if not row:
                return None

            cols = [
                "team_id",
                "event_gw",
                "fetched_at",
                "picks_json",
                "transfer_info_json",
                "status",
                "error_message",
            ]
            out = dict(zip(cols, row))

            # psycopg may already return dict/json; handle both cases safely.
            if isinstance(out["picks_json"], str):
                out["picks_json"] = json.loads(out["picks_json"])
            if isinstance(out["transfer_info_json"], str):
                out["transfer_info_json"] = json.loads(out["transfer_info_json"])

            return out
    finally:
        conn.close()


def set_cached_team_context(
    team_id: int,
    event_gw: int,
    picks_json: dict | None,
    transfer_info_json: dict | None,
    *,
    status: str = "ok",
    error_message: str | None = None,
) -> None:
    """
    Upsert team cache row for a team + GW.
    """
    conn = get_connection(autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fpl_team_cache (
                    team_id, event_gw, fetched_at,
                    picks_json, transfer_info_json,
                    status, error_message
                )
                VALUES (%s, %s, NOW(), %s::jsonb, %s::jsonb, %s, %s)
                ON CONFLICT (team_id, event_gw)
                DO UPDATE SET
                    fetched_at = EXCLUDED.fetched_at,
                    picks_json = EXCLUDED.picks_json,
                    transfer_info_json = EXCLUDED.transfer_info_json,
                    status = EXCLUDED.status,
                    error_message = EXCLUDED.error_message
                """,
                (
                    team_id,
                    event_gw,
                    json.dumps(picks_json or {}),
                    json.dumps(transfer_info_json or {}),
                    status,
                    (error_message[:4000] if error_message else None),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def cache_age_minutes(fetched_at: datetime) -> float:
    """
    Helper for display/debugging if needed.
    """
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - fetched_at).total_seconds() / 60.0
