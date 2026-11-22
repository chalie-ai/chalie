"""GistService — persist + batch-read per-thread topical labels.

Coordinates each thread's ``summary``: one terse 3-5 word topical label per
thread, written once on the first reply past settle0 and stored on the
thread's own row. The label is a pure FE affordance shown
on a collapsed thread pill — it carries no memory or retrieval significance
(no embedding, no search index).

``channel``/``turn_id`` are taken as parameters rather than read off
``self.mp`` because both callers address a thread distinct from the mp
instance doing the addressing: the delegate gist processor labels the
*trigger* thread while running under its own config/channel, and the API
batch-reads labels for arbitrary threads on demand.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from models.thread import Thread

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

logger = logging.getLogger(__name__)
LOG_PREFIX = "[THREAD GIST]"


class GistService:
    """Upsert + batch-read the per-thread topical-label index."""

    def __init__(self, mp: MessageProcessor) -> None:
        self.mp = mp

    def upsert(self, channel: str, turn_id: int, gist: str) -> bool:
        """Persist (or replace) the label for one thread (idempotent per thread)
        and return whether it was stored.

        The label is stored on the thread's row; a thread with no rows has no
        such row to carry it, so the label is skipped and logged."""
        gist = (gist or "").strip()
        if not gist:
            return False
        try:
            stored = Thread.set_summary(channel, turn_id, gist)
        except Exception as exc:
            logger.warning("%s upsert failed for %s turn=%s: %s", LOG_PREFIX, channel, turn_id, exc)
            return False
        if not stored:
            logger.warning("%s no thread for %s turn=%s; label skipped", LOG_PREFIX, channel, turn_id)
        return stored

    def bulk_get(self, channel: str, turn_ids: list[int]) -> dict[int, str]:
        """Labels for a batch of turn_ids — the thread feed's collapsed labels.

        Returns ``{turn_id: gist}`` for every turn_id that has a label.
        """
        try:
            return Thread.summaries(channel, turn_ids)
        except Exception as exc:
            logger.debug("%s bulk_get failed for %s: %s", LOG_PREFIX, channel, exc)
            return {}
