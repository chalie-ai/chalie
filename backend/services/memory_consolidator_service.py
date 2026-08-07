"""Memory v3 consolidator service — the background agentic pass.

For a given (channel, turn) it builds the input window (the thread's last
compaction + the turn's transcript rows past it, plus any map rows previously
written for the turn), drives the consolidator LLM (which uses the four memory
tools to write Graph/Map rows), and lets the tools stamp provenance
(``sourced_from``) off the config. ``tick`` walks every consolidated channel and
consolidates the most recent not-yet-consolidated turn.

Scheduling lives in ``cron.jobs.memory_consolidator`` (IdleGatedJob, 10-min idle
window). Re-fire on a growing thread is handled by re-injecting the map rows the
service already wrote for the turn (``_prior_map``).
"""

from __future__ import annotations

import logging
from typing import cast

from configs.channels.memory_consolidator import MemoryConsolidatorConfig
from configs.enums.channels import Channel
from models.compaction import Compaction
from services.database import Database

logger = logging.getLogger(__name__)

# Channels the consolidator never touches: delegates + skills_building surface
# their durable value through the parent chat channel; the consolidator never
# consolidates itself.
_EXCLUDED_PREFIXES = ("delegate:",)
_EXCLUDED_CHANNELS = {
    Channel.MEMORY_CONSOLIDATOR.value,
    Channel.SKILLS_BUILDING.value,
}


class MemoryConsolidatorService:
    """Drives the per-(channel, turn) consolidation pass."""

    def consolidate(self, channel: str, turn_id: int) -> str:
        """Consolidate one (channel, turn) window through the consolidator LLM."""
        conn = Database.conn()
        rows = cast(
            "list[tuple[int, str, str]]",
            conn.execute(
                "SELECT id, role, content FROM transcript "
                "WHERE channel = ? AND turn_id = ? ORDER BY id",
                (channel, turn_id),
            ).fetchall(),
        )
        if not rows:
            return f"{channel}:{turn_id} no rows"

        ids = [row[0] for row in rows]
        window = "\n".join(f"[{row[0]}] {row[1]}: {row[2]}" for row in rows)
        compaction = Compaction.latest_main(channel)
        config = MemoryConsolidatorConfig(
            target_channel=channel,
            compaction=compaction.content if compaction else "",
            window=window,
            prior_map=self._prior_map_text(ids),
            source_transcript_ids=ids,
        )

        # Lazy import: controllers pull in the full service graph.
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        try:
            MessageProcessor.process(config).result()
        except Exception:
            logger.exception(
                "[MEMORY CONSOLIDATOR] %s:%s failed", channel, turn_id
            )
            return f"{channel}:{turn_id} error"
        return f"{channel}:{turn_id} consolidated ({len(ids)} rows)"

    def tick(self) -> str:
        """Walk every consolidated channel; consolidate its most recent
        not-yet-consolidated turn. Returns a short status."""
        from models.transcript import Transcript  # noqa: PLC0415

        details: list[str] = []
        for channel in Transcript.distinct_channels():
            if channel in _EXCLUDED_CHANNELS or any(
                channel.startswith(prefix) for prefix in _EXCLUDED_PREFIXES
            ):
                continue
            turn_id = self._most_recent_unconsolidated_turn(channel)
            if turn_id is not None:
                details.append(self.consolidate(channel, turn_id))
        return "; ".join(details) if details else "nothing to consolidate"

    def _most_recent_unconsolidated_turn(self, channel: str) -> int | None:
        conn = Database.conn()
        turns = cast(
            "list[tuple[int]]",
            conn.execute(
                "SELECT DISTINCT turn_id FROM transcript "
                "WHERE channel = ? AND turn_id IS NOT NULL "
                "ORDER BY turn_id DESC LIMIT 20",
                (channel,),
            ).fetchall(),
        )
        for (turn_id,) in turns:
            ids = cast(
                "list[tuple[int]]",
                conn.execute(
                    "SELECT id FROM transcript WHERE channel = ? AND turn_id = ?",
                    (channel, turn_id),
                ).fetchall(),
            )
            row_ids = [i for (i,) in ids]
            if row_ids and not self._is_consolidated(row_ids):
                return turn_id
        return None

    def _is_consolidated(self, transcript_ids: list[int]) -> bool:
        """True if any memory_map row was already written from these transcript ids."""
        placeholders = ",".join("?" for _ in transcript_ids)
        row = Database.conn().execute(
            f"SELECT 1 FROM memory_map m, json_each(m.sourced_from) "
            f"WHERE json_each.value IN ({placeholders}) LIMIT 1",
            transcript_ids,
        ).fetchone()
        return row is not None

    def _prior_map_text(self, transcript_ids: list[int]) -> str:
        """Map rows previously written for any of these transcript ids (re-fire)."""
        if not transcript_ids:
            return ""
        placeholders = ",".join("?" for _ in transcript_ids)
        rows = cast(
            "list[tuple[int, int, str]]",
            Database.conn().execute(
                f"SELECT id, iteration, contents FROM memory_map m "
                f"WHERE EXISTS (SELECT 1 FROM json_each(m.sourced_from) "
                f"WHERE json_each.value IN ({placeholders})) "
                f"ORDER BY iteration DESC",
                transcript_ids,
            ).fetchall(),
        )
        return "\n".join(f"#{r[0]} (iter {r[1]}) {r[2]}" for r in rows)
