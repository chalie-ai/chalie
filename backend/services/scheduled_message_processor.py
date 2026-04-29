"""
ScheduledMessageProcessor — Executes scheduled prompts via the standard tool loop.

MessageProcessor v2 subclass. CHANNEL is flat 'scheduled' (collapsed from
the old 'scheduled:{item_id}' format). item_id is stored in metadata only.

North star: /Volumes/llm/chalie-plans/message-processing.md
Plan: /Users/dylangrech/.claude/plans/joyful-cooking-riddle.md § Commit 9
"""

import logging

from services.message_processor import MessageProcessor
from services.system_message_prompt import ScheduledSystemMessagePrompt

logger = logging.getLogger(__name__)


class ScheduledMessageProcessor(MessageProcessor):
    """MessageProcessor subclass for scheduled prompt execution.

    Standard limits (30 iterations / 900 s). Excludes 'schedule' (prevent
    recursive scheduling) and 'subagent' (prevent background spawning).

    CHANNEL is flat 'scheduled' — item_id lives only in metadata, never
    in the channel string (channel collapse from Commit 9).
    """

    CHANNEL = 'scheduled'   # flat — collapsed from scheduled:{id}
    ROLE = 'scheduled'
    SYSTEM_PROMPT_CLASS = ScheduledSystemMessagePrompt
    ALWAYS_AVAILABLE: list[str] = [
        "document",
        "find_tools",
        "list",
        "memory",
        "read",
        "review_tool_calls",
    ]
    DISCOVERABLE: list[str] = [
        "browser",
        "code_eval",
        "news",
        "programming_docs_search",
        "search",
        "weather",
    ]

    def getUserDefinition(self) -> str:
        return (
            "The user is 'scheduled' — an automated background process "
            "executing a task that the user scheduled in advance."
        )

    def getUserPrompt(self) -> str:
        """Scheduled message is the raw input; prepend role prefix."""
        trail = self.getActLoopTrail()
        parts = [f"scheduled: {self._raw_input}"]
        if trail:
            parts.append(trail)
        return '\n'.join(parts)

    def postTurn(self) -> None:
        """Scheduled post-turn: metrics + mark item executed.

        item_id comes from metadata['item_id']. The mark_executed step is a
        named hook for the future ScheduledItemService and is best-effort —
        it must not raise.

        NOTE: ScheduledItemService does NOT exist yet. The mark-executed step
        is currently a placeholder; the scheduler_service caller updates
        scheduled_items.status in its own transaction.
        """
        try:
            from services.metrics_service import MetricsService
            m = MetricsService()
            m.record_counter('requests_total')
            m.record_counter('scheduled_turns_total')
        except Exception as e:
            logger.debug("[Scheduled.postTurn] Metrics failed: %s", e, exc_info=True)

        # Placeholder hook — see _mark_scheduled_item_executed() for the
        # caveat about scheduler_service timing.
        item_id = self._metadata.get('item_id')
        if item_id:
            self._mark_scheduled_item_executed(item_id)

    def _mark_scheduled_item_executed(self, item_id: str) -> None:
        """Named hook for the future ScheduledItemService wiring point.

        TIMING CAVEAT: The scheduler_service caller's `_fire_item()` spawns a
        daemon thread which calls `send()` → `postTurn()` → here. The caller
        only runs `UPDATE scheduled_items SET status='fired'` AFTER `_fire_item`
        returns, and the surrounding `conn.commit()` is even later. At the
        moment this method runs, `scheduled_items.status` may still be
        'pending'. Do NOT rely on the row already being marked 'fired'.

        Today this is a no-op. When ScheduledItemService is introduced it
        should mark the item executed itself rather than depending on the
        scheduler_service caller's UPDATE.

        Does NOT raise — any failure is logged at DEBUG level only.
        """
        logger.debug(
            "[Scheduled._mark_scheduled_item_executed] item_id=%s — placeholder "
            "(see TIMING CAVEAT in docstring; scheduler_service status update "
            "races with this hook)", item_id
        )
