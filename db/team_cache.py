"""Optional team cache fallback for local development.

Provides no-op implementations used when DB cache is not configured.
"""

from __future__ import annotations


def get_team_cache(team_id: int, event_id: int) -> dict | None:
    """Return cached payload for (team, event) if available."""
    return None


def upsert_team_cache_ok(
    team_id: int,
    event_id: int,
    *,
    picks_json: dict | None = None,
    transfer_info_json: dict | None = None,
) -> None:
    """No-op cache writer for successful fetches."""
    return None


def upsert_team_cache_failed(team_id: int, event_id: int, error_msg: str) -> None:
    """No-op cache writer for failed fetches."""
    return None

