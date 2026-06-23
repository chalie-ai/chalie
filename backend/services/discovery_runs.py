"""Persistence + read helpers for proactive-research (discovery) runs.

One row per loop execution: the grounding it ran against (user summary +
compacted summary at execution time) and a preview of what it surfaced. The
loop's full output is NOT duplicated here — it lives in the transcript table
under the discovery channel/turn, joined live by :func:`get_run_detail` so the
loop stays inspectable from a single source of truth. The detail view relies on
those transcript rows never being compacted away: the discovery channel runs
with suppressed history and a muted source profile, so it never accumulates the
context that triggers transcript GC.

Callers: ``configs/channels/discovery.PersistDiscoveryRunHook`` (write) and
``api.system`` Auto Research observability (Brain read-only views).
"""

import logging
from typing import Optional, cast

from services.database_service import get_shared_db_service
from services.source_profiles import CHANNEL_DISCOVERY
from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[DISCOVERY]"


def record_run(
    turn_id: Optional[int],
    user_summary: str,
    compacted_summary: str,
    researched: str,
) -> Optional[int]:
    """Insert one discovery run, returning its id (None on DB error)."""
    try:
        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO discovery_runs "
                "(ran_at, turn_id, user_summary, compacted_summary, researched) "
                "VALUES (?, ?, ?, ?, ?)",
                (utc_now().isoformat(), turn_id, user_summary, compacted_summary, researched),
            )
            row_id = cursor.lastrowid
            cursor.close()
        return cast(int, row_id)
    except Exception as exc:
        logger.warning("%s record_run failed: %s", LOG_PREFIX, exc)
        return None


def list_runs(limit: int, offset: int = 0) -> list[dict[str, object]]:
    """Newest-first list of runs as ``{id, ran_at, researched}`` (raises on DB error)."""
    return get_shared_db_service().fetch_all(
        "SELECT id, ran_at, researched FROM discovery_runs "
        "ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )


def get_run_detail(run_id: int) -> Optional[dict[str, object]]:
    """One run with its full loop output, or None when the id is unknown.

    The transcript is read live from the transcript table — every assistant row
    of the run's turn, oldest-first, joined by blank lines into one blob. Tool
    calls and the input prompt are intentionally excluded: this view shows what
    the loop *said*, not how it got there.
    """
    rows = get_shared_db_service().fetch_all(
        "SELECT id, ran_at, turn_id, user_summary, compacted_summary "
        "FROM discovery_runs WHERE id = ? LIMIT 1",
        (run_id,),
    )
    if not rows:
        return None
    run = rows[0]
    return {
        "id": run["id"],
        "ran_at": run["ran_at"],
        "user_summary": run["user_summary"] or "",
        "compacted_summary": run["compacted_summary"] or "",
        "transcript": _transcript_blob(cast("Optional[int]", run["turn_id"])),
    }


def _transcript_blob(turn_id: Optional[int]) -> str:
    """Assistant rows of a discovery turn joined into one continuous blob."""
    if turn_id is None:
        return ""
    from services.transcript_service import Transcript
    rows = Transcript.by_turn(CHANNEL_DISCOVERY, turn_id)
    return "\n\n".join(
        cast("str", r["content"])
        for r in rows
        if r.get("role") == "assistant" and r.get("content")
    )
