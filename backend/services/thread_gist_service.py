"""ThreadGistService — persist + batch-read per-thread one-line labels.

Stores one current label per ``(channel, turn_id)`` in ``thread_gist``. The
label is a pure FE affordance shown on a collapsed thread pill — it carries no
memory or retrieval significance (no embedding, no search index).
"""

from __future__ import annotations

import logging
from typing import Optional, cast

from services.database_service import get_shared_db_service

logger = logging.getLogger(__name__)
LOG_PREFIX = "[THREAD GIST]"


class ThreadGistService:
    """Upsert + batch-read the one-line thread-label index."""

    def upsert(self, channel: str, turn_id: int, summary: str) -> None:
        """Persist (or replace) the label for one thread."""
        summary = (summary or "").strip()
        if not summary:
            return
        try:
            with get_shared_db_service().connection() as conn:
                conn.execute(
                    "INSERT INTO thread_gist (channel, turn_id, summary, updated_at) "
                    "VALUES (?, ?, ?, datetime('now')) "
                    "ON CONFLICT(channel, turn_id) DO UPDATE SET "
                    "summary = excluded.summary, updated_at = excluded.updated_at",
                    (channel, turn_id, summary),
                )
        except Exception as exc:
            logger.warning("%s upsert failed for %s turn=%s: %s", LOG_PREFIX, channel, turn_id, exc)

    def bulk_get(self, channel: str, turn_ids: list[int]) -> dict[int, str]:
        """Labels for a batch of turn_ids — the thread feed's collapsed summaries.

        Returns ``{turn_id: summary}`` for every turn_id that has a label.
        """
        if not turn_ids:
            return {}
        try:
            placeholders = ",".join("?" * len(turn_ids))
            with get_shared_db_service().connection() as conn:
                rows = conn.execute(
                    f"SELECT turn_id, summary FROM thread_gist "
                    f"WHERE channel = ? AND turn_id IN ({placeholders})",
                    (channel, *turn_ids),
                ).fetchall()
            return {int(r[0]): cast(str, r[1]) for r in rows}
        except Exception as exc:
            logger.debug("%s bulk_get failed for %s: %s", LOG_PREFIX, channel, exc)
            return {}


_instance: Optional[ThreadGistService] = None


def get_thread_gist_service() -> ThreadGistService:
    global _instance
    if _instance is None:
        _instance = ThreadGistService()
    return _instance
