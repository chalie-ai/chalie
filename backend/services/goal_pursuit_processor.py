"""
GoalPursuitProcessor — Background goal execution with extended limits.

Simple MessageProcessor subclass that runs a goal in an isolated channel
with higher iteration and timeout limits. Used by the goal_pursuit skill
to execute long-running tasks in a daemon thread.
"""

import os
import logging

from services.message_processor import MessageProcessor

logger = logging.getLogger(__name__)

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'goal-pursuit.md')


def _load_system_prompt() -> str:
    """Load the goal-pursuit system prompt from disk."""
    try:
        with open(os.path.normpath(_PROMPT_PATH), 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.warning(f"[GOAL PURSUIT] Failed to load system prompt: {e}")
        return (
            "You are Chalie, a determined assistant. Pursue the goal provided to the best "
            "of your ability. If tool calls fail or errors occur, try alternatives. "
            "When complete, provide a clear summary of what you accomplished."
        )


class GoalPursuitProcessor(MessageProcessor):
    """MessageProcessor subclass for background goal execution.

    Overrides limits to allow long-running pursuits:
      - MAX_ITERATIONS: 50 (vs. 30 for normal turns)
      - MAX_TIMEOUT: 7200 seconds / 2 hours (vs. 900 for normal turns)
    """

    MAX_ITERATIONS = 50
    MAX_TIMEOUT = 7200  # 2 hours

    def _assemble_user_prompt(self, message, channel, metadata=None):
        """Goal pursuits pass the raw goal string — no prompt assembly needed."""
        return message

    def process(self, goal: str, pursuit_id: str) -> dict:
        """Execute the goal in an isolated channel.

        Args:
            goal: The goal string to pursue.
            pursuit_id: UUID hex string for channel isolation.

        Returns:
            Standard MessageProcessor result dict.
        """
        from services.tool_schema_service import get_skill_schemas
        from services.innate_skills.registry import ALL_SKILL_NAMES

        system_prompt = _load_system_prompt()
        channel = f'goal_pursuit:{pursuit_id}'

        # All skills except goal_pursuit itself (prevent recursion)
        tool_names = [s for s in ALL_SKILL_NAMES if s != 'goal_pursuit']
        tools = get_skill_schemas(tool_names)

        logger.info(f"[GOAL PURSUIT] Starting pursuit {pursuit_id[:8]}: '{goal[:80]}'")
        return self.send(
            goal,
            system_prompt,
            channel=channel,
            job='frontal-cortex-unified',
            tools=tools,
        )
