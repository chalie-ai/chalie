"""
GoalPursuitProcessor — Background goal execution with extended limits.

MessageProcessor v2 subclass. CHANNEL is flat 'goal_pursuit' (collapsed from
the old 'goal_pursuit:{pursuit_id}' format). pursuit_id is stored in metadata
only, not embedded in the channel string.

North star: /Volumes/llm/chalie-plans/message-processing.md
Plan: /Users/dylangrech/.claude/plans/joyful-cooking-riddle.md § Commit 9
"""

import logging

from services.message_processor import MessageProcessor
from services.system_message_prompt import GoalPursuitSystemMessagePrompt
from services.innate_skills.registry import ALL_SKILL_NAMES

logger = logging.getLogger(__name__)


class GoalPursuitProcessor(MessageProcessor):
    """MessageProcessor subclass for background goal execution.

    Extended safety caps for long-running tasks:
      MAX_ITERATIONS = 50   (vs. base 30)
      MAX_TIMEOUT    = 7200 (2 hours, vs. base 900)

    NATIVE_TOOLS excludes 'goal_pursuit' itself to prevent recursive spawning.
    """

    CHANNEL = 'goal_pursuit'   # flat — collapsed from goal_pursuit:{id}
    ROLE = 'goal_pursuit'
    MAX_ITERATIONS = 50
    MAX_TIMEOUT = 7200  # 2 hours
    SYSTEM_PROMPT_CLASS = GoalPursuitSystemMessagePrompt
    NATIVE_TOOLS: list[str] = sorted(s for s in ALL_SKILL_NAMES if s != 'goal_pursuit')

    def getUserDefinition(self) -> str:
        return (
            "The user is 'goal_pursuit' — a background process that is "
            "autonomously pursuing a long-running goal on behalf of the user."
        )

    def getUserPrompt(self) -> str:
        """Goal string is the raw input; prepend role prefix."""
        trail = self.getActLoopTrail()
        parts = [f"goal_pursuit: {self._raw_input}"]
        if trail:
            parts.append(trail)
        return '\n'.join(parts)

    def postTurn(self) -> None:
        """Goal pursuit post-turn: interaction log (goal_pursuit_turn) + metrics only.

        Pursuit progress tracking lives in the goal_pursuit_skill layer, not here.
        """
        try:
            from services.interaction_log_service import InteractionLogService
            from services.database_service import get_shared_db_service
            log = InteractionLogService(get_shared_db_service())
            log.log_event(
                event_type='goal_pursuit_turn',
                payload={
                    'pursuit_id': self._metadata.get('pursuit_id', 'unknown'),
                },
                channel=self.CHANNEL,
                source='goal_pursuit',
                metadata=self._metadata,
            )
        except Exception as e:
            # WARNING (not DEBUG): audit-trail loss must be visible in production.
            logger.warning("[GoalPursuit.postTurn] Interaction log failed: %s", e, exc_info=True)

        try:
            from services.metrics_service import MetricsService
            m = MetricsService()
            m.record_counter('requests_total')
            m.record_counter('goal_pursuit_turns_total')
        except Exception as e:
            logger.debug("[GoalPursuit.postTurn] Metrics failed: %s", e, exc_info=True)
