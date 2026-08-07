"""
Scheduler Service — embedding helper for ``scheduled_items`` rows.

The minute-aligned poll-and-fire loop that used to live here was unified into
the ``cron`` package: ``cron.jobs.scheduled_items.ScheduledItemsDispatcherJob``
now owns the dumb-cron dispatch (fired by ``cron-runner``). What remains here is
the semantic-search embedding helper used when a schedule row is written —
``abilities.schedule`` and ``api.endpoints.scheduler`` call it to index a new
item.
"""

import logging

from models.scheduled_item import ScheduledItem
from services.embedding_utils import pack_embedding

logger = logging.getLogger(__name__)
LOG_PREFIX = "[SCHEDULER]"


class SchedulerService:
    """Embedding helper for scheduled_items rows."""

    @staticmethod
    def embed_scheduled_item(item_id: int, message: str) -> None:
        """Non-fatal: logs a warning and returns silently on any failure."""
        try:
            from services.embedding_service import get_embedding_service

            emb_service = get_embedding_service()
            embedding = emb_service.generate_embedding(message)
            if not embedding:
                return

            packed = pack_embedding(embedding)
            if packed is None:
                return
            ScheduledItem.write_embedding(item_id, packed)

            logger.debug(f"{LOG_PREFIX} Embedded scheduled item {item_id}")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to embed scheduled item {item_id}: {e}")
