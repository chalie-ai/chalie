"""Memory v3 migration — catch up the new memory system by replaying every
transcript turn through the consolidator.

A thin loop around :meth:`MemoryConsolidatorService.consolidate`: iterate every
``(channel, turn_id)`` in transcript order and fire the consolidator once each.
The consolidator distils each turn's transcript into Graph/Map rows (with
provenance), so by the end of the loop the new stores carry the full history.

Idempotent — ``consolidate`` skips turns already marked consolidated — so
re-running or resuming after an interrupt picks up where it left off. Transcript
rows are never deleted; this loop only populates the new stores.
"""

from __future__ import annotations

import logging
from typing import Iterator, cast

from configs.enums.channels import Channel
from services.database import Database
from services.memory_consolidator_service import MemoryConsolidatorService

logger = logging.getLogger(__name__)

# Same exclusions as the consolidator's tick(): delegates + skills_building
# surface their value through the parent chat channel; the consolidator never
# consolidates itself.
_EXCLUDED_PREFIXES = ("delegate:",)
_EXCLUDED_CHANNELS = {
    Channel.MEMORY_CONSOLIDATOR.value,
    Channel.SKILLS_BUILDING.value,
}


class MemoryV3Migration:
    """Replay every transcript turn through the consolidator to catch up Graph/Map."""

    def run(self) -> dict[str, int]:
        self._backfill_legacy_turn_id()
        svc = MemoryConsolidatorService()
        consolidated = 0
        for channel, turn_id in self._all_turns_in_order():
            if "consolidated" in svc.consolidate(channel, turn_id):
                consolidated += 1
        counts = {"turns_consolidated": consolidated}
        logger.info("[MEMORY V3 MIGRATION] %s", counts)
        return counts

    def _backfill_legacy_turn_id(self) -> None:
        """Repair legacy NULL turn_id (``-id``) so those rows join a turn."""
        Database.conn().execute(
            "UPDATE transcript SET turn_id = -id WHERE turn_id IS NULL"
        )

    def _all_turns_in_order(self) -> Iterator[tuple[str, int]]:
        rows = cast(
            "list[tuple[str, int]]",
            Database.conn().execute(
                "SELECT channel, turn_id FROM ("
                "  SELECT DISTINCT channel, turn_id FROM transcript "
                "  WHERE turn_id IS NOT NULL"
                ") ORDER BY channel, turn_id"
            ).fetchall(),
        )
        for channel, turn_id in rows:
            if channel in _EXCLUDED_CHANNELS or any(
                channel.startswith(prefix) for prefix in _EXCLUDED_PREFIXES
            ):
                continue
            yield channel, turn_id
