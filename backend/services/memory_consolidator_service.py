"""Memory v3 consolidator service — the background agentic pass.

The consolidator runs on a fixed 30-minute cron (see ``cron.jobs.memory_consolidator``).
On each tick it walks every channel, finds the oldest unconsolidated rows
(those with ``consolidated = 0``), builds a token-budgeted window, and drives
the consolidator LLM through the normal message-processor path. On success the
rows are stamped ``consolidated = 1`` so they are never re-processed.

No compaction-boundary or memory_map-readiness dependency: the ``consolidated``
flag on the transcript table is the sole progress tracker.
"""

from __future__ import annotations

import logging
from typing import cast

from configs.channels.memory_consolidator import MemoryConsolidatorConfig, descriptor_for
from configs.enums.channels import Channel
from services.database import Database
from services.time_utils import parse_utc

logger = logging.getLogger(__name__)

# Channels the consolidator never touches: delegates + skills_building surface
# their durable value through the parent chat channel; the consolidator never
# consolidates itself or the discovery channel (AutoResearch manages its own
# memory).
_EXCLUDED_PREFIXES = ("delegate:",)
_EXCLUDED_CHANNELS = {
    Channel.MEMORY_CONSOLIDATOR.value,
    Channel.SKILLS_BUILDING.value,
    Channel.DISCOVERY.value,
}

# Minimum unconsolidated rows before we bother consolidating a channel.
# Prevents trivial consolidation of one-off messages.
_MIN_ROWS = 10

# Token estimator: rough characters-per-token ratio for SQLite TEXT.
_CHARS_PER_TOKEN = 4

# Window budget fraction of the CHAT provider's context window.
_WINDOW_BUDGET_FRACTION = 0.70

# Fallback context window when the provider is unavailable.
_FALLBACK_CONTEXT_LIMIT = 8000


def _token_estimate(text: str) -> int:
    """Rough token count for ``text`` using the characters-per-token ratio."""
    return len(text) // _CHARS_PER_TOKEN


def _format_row_ts(created_at: str) -> str:
    """Format an ISO-created_at as ``yyyy-mm-dd HH:mm``, falling back to the
    raw string on unparseable input."""
    try:
        dt = parse_utc(created_at)
        # parse_utc returns PARSE_SENTINEL on failure; detect that.
        if dt.year == 1:
            return created_at
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return created_at


class MemoryConsolidatorService:
    """Drives the per-channel consolidation pass over unconsolidated rows."""

    def consolidate(self, channel: str) -> str:
        """Consolidate a window of unconsolidated rows for ``channel`` through
        the consolidator LLM. Returns a short status string."""
        conn = Database.conn()
        rows = cast(
            "list[tuple[int, str, str, str, str | None]]",
            conn.execute(
                "SELECT id, role, content, created_at, location_name "
                "FROM transcript "
                "WHERE channel = ? AND consolidated = 0 "
                "  AND (turn_id IS NULL OR EXISTS ("
                "    SELECT 1 FROM transcript t2 "
                "    WHERE t2.channel = transcript.channel "
                "      AND t2.turn_id = transcript.turn_id "
                "      AND t2.role = 'assistant' AND t2.settled = 1"
                "  )) "
                "ORDER BY id ASC",
                (channel,),
            ).fetchall(),
        )
        if len(rows) < _MIN_ROWS:
            return f"{channel}: <10 unconsolidated rows"

        window, batch_ids = self._build_window(rows, channel, self._token_budget())

        config = MemoryConsolidatorConfig(
            target_channel=channel,
            window=window,
            source_transcript_ids=batch_ids,
        )

        # Lazy import: controllers pull in the full service graph.
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        try:
            MessageProcessor.process(config).result()
        except Exception:
            logger.exception("[MEMORY CONSOLIDATOR] %s failed", channel)
            return f"{channel} error"

        self._mark_consolidated(batch_ids)
        return f"{channel} consolidated ({len(batch_ids)} rows)"

    @staticmethod
    def _build_window(
        rows: "list[tuple[int, str, str, str, str | None]]",
        channel: str,
        budget: int,
    ) -> "tuple[str, list[int]]":
        """Format the oldest rows that fit ``budget`` tokens into the consolidator
        window. Pure (no DB, no LLM): the batch selection + the ``## channel`` /
        ``### Description`` / ``### Exchanges`` format live here so they are
        testable without driving the model. At least the first row is always
        included; further rows are added oldest-first until the next would push
        the estimated window over the budget."""
        name, description = descriptor_for(channel)
        header = f"## {name}\n### Description\n{description}\n\n### Exchanges\n"
        budget_for_lines = max(budget - _token_estimate(header), 0)

        window_lines: list[str] = []
        batch_ids: list[int] = []
        used = 0
        for row_id, role, content, created_at, location_name in rows:
            if location_name:
                line = f"[{_format_row_ts(created_at)} @ {location_name}] {role}: {content}"
            else:
                line = f"[{_format_row_ts(created_at)}] {role}: {content}"
            line_tokens = _token_estimate(line)
            if window_lines and used + line_tokens > budget_for_lines:
                break
            window_lines.append(line)
            batch_ids.append(row_id)
            used += line_tokens

        return header + "\n".join(window_lines), batch_ids

    @staticmethod
    def _mark_consolidated(batch_ids: list[int]) -> None:
        """Stamp ``consolidated = 1`` on the consolidated batch."""
        if not batch_ids:
            return
        placeholders = ",".join("?" for _ in batch_ids)
        Database.conn().execute(
            f"UPDATE transcript SET consolidated = 1 WHERE id IN ({placeholders})",
            batch_ids,
        )

    def tick(self) -> str:
        """Walk every channel; consolidate each one that has enough unconsolidated
        rows. Returns a combined status string."""
        from models.transcript import Transcript  # noqa: PLC0415

        details: list[str] = []
        for channel in Transcript.distinct_channels():
            if channel in _EXCLUDED_CHANNELS or any(
                channel.startswith(prefix) for prefix in _EXCLUDED_PREFIXES
            ):
                continue
            details.append(self.consolidate(channel))
        return "; ".join(details) if details else "nothing to consolidate"

    @staticmethod
    def _token_budget() -> int:
        """The token budget for the consolidation window: 70% of the CHAT
        provider's context window, with a sane fallback."""
        try:
            from services.provider_db_service import ProviderDbService  # noqa: PLC0415
            from services.provider_cache_service import ProviderCacheService  # noqa: PLC0415

            selected = ProviderCacheService.get_selected_provider()
            if selected:
                config = dict(selected)
            else:
                providers = ProviderCacheService.get_providers()
                if not providers:
                    return _FALLBACK_CONTEXT_LIMIT
                config = dict(next(iter(providers.values())))

            window = ProviderDbService().pin_context_window(config)
            if window and window > 0:
                return int(window * _WINDOW_BUDGET_FRACTION)
        except Exception:
            logger.debug("[MEMORY CONSOLIDATOR] could not resolve context limit", exc_info=True)
        return _FALLBACK_CONTEXT_LIMIT
