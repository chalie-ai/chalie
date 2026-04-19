"""
DMNMessageProcessor — Background DMN execution with reduced limits.

MessageProcessor v2 subclass. Hardcodes CHANNEL='dmn', ROLE='proactive_thought'.
postTurn() increments metrics counters.

North star: /Volumes/llm/chalie-plans/message-processing.md
Plan: /Users/dylangrech/.claude/plans/joyful-cooking-riddle.md § Commit 9
"""

import logging

from services.message_processor import MessageProcessor
from services.system_message_prompt import DMNSystemMessagePrompt
from services.innate_skills.registry import ALL_SKILL_NAMES

logger = logging.getLogger(__name__)


class DMNMessageProcessor(MessageProcessor):
    """MessageProcessor subclass for background DMN proactive reviews.

    Reduced safety caps — DMN turns are best-effort, short-lived:
      MAX_ITERATIONS = 15  (vs. base 30)
      MAX_TIMEOUT    = 300 (5 minutes, vs. base 900)

    NATIVE_TOOLS excludes 'goal_pursuit' to prevent spawning background
    tasks from within a background process.
    """

    CHANNEL = 'dmn'
    ROLE = 'proactive_thought'
    MAX_ITERATIONS = 15
    MAX_TIMEOUT = 300  # 5 minutes
    SYSTEM_PROMPT_CLASS = DMNSystemMessagePrompt
    NATIVE_TOOLS: list[str] = sorted(s for s in ALL_SKILL_NAMES if s != 'goal_pursuit')

    def getUserDefinition(self) -> str:
        return (
            "The user is 'proactive_thought' — a special background process "
            "that represents your own reflections on recent activity."
        )

    def getUserPrompt(self) -> str:
        """DMN context is pre-built by DMNService — pass through with role prefix."""
        trail = self.getActLoopTrail()
        parts = [f"proactive_thought: {self._raw_input}"]
        if trail:
            parts.append(trail)
        return '\n'.join(parts)

    def postTurn(self) -> None:
        """DMN post-turn: metrics only.

        No phase update, no DMNService.on_turn()
        (that would be re-entrant — DMNService already owns the timer reset).
        """
        try:
            from services.metrics_service import MetricsService
            m = MetricsService()
            m.record_counter('requests_total')
            m.record_counter('dmn_turns_total')
        except Exception as e:
            logger.debug("[DMN.postTurn] Metrics failed: %s", e, exc_info=True)
