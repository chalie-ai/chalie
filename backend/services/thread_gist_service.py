"""ThreadGistService — persist + batch-read per-thread topical labels.

Stores one terse 3-5 word topical label per ``(channel, turn_id)`` in
``thread_gist``, written once on the first reply past settle0. The label is a
pure FE affordance shown on a collapsed thread pill — it carries no memory or
retrieval significance (no embedding, no search index).
"""

from __future__ import annotations

import logging
from typing import Optional, cast

from services.database import Database
from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[THREAD GIST]"


class ThreadGistService:
    """Upsert + batch-read the per-thread topical-label index."""

    def upsert(self, channel: str, turn_id: int, gist: str) -> None:
        """Persist (or replace) the label for one thread (idempotent per turn)."""
        gist = (gist or "").strip()
        if not gist:
            return
        try:
            Database.conn().execute(
                "INSERT INTO thread_gist (channel, turn_id, gist, created_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(channel, turn_id) DO UPDATE SET gist = excluded.gist",
                (channel, turn_id, gist, utc_now().isoformat()),
            )
        except Exception as exc:
            logger.warning("%s upsert failed for %s turn=%s: %s", LOG_PREFIX, channel, turn_id, exc)

    def bulk_get(self, channel: str, turn_ids: list[int]) -> dict[int, str]:
        """Labels for a batch of turn_ids — the thread feed's collapsed labels.

        Returns ``{turn_id: gist}`` for every turn_id that has a label.
        """
        if not turn_ids:
            return {}
        try:
            placeholders = ",".join("?" * len(turn_ids))
            rows = Database.conn().execute(
                f"SELECT turn_id, gist FROM thread_gist "
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
