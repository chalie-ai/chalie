"""
ScheduledMessageProcessor — Executes scheduled prompts via the standard tool loop.

Thin MessageProcessor subclass. Same pattern as GoalPursuitProcessor: load a
system prompt, pass the message through, let the parent handle everything.
"""

import os
import logging

from services.message_processor import MessageProcessor

logger = logging.getLogger(__name__)

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'scheduled-prompt.md')

_EXCLUDED_SKILLS = frozenset({'schedule', 'goal_pursuit'})


def _load_system_prompt() -> str:
    """Load the scheduled-prompt system prompt from disk."""
    try:
        with open(os.path.normpath(_PROMPT_PATH), 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.warning(f"[SCHEDULED PROMPT] Failed to load system prompt: {e}")
        return (
            "You are Chalie, executing a scheduled task. The user set this up earlier "
            "and it is now due. Execute the task to the best of your ability using the "
            "tools available to you. Be concise and action-oriented in your response."
        )


class ScheduledMessageProcessor(MessageProcessor):
    """MessageProcessor subclass for scheduled prompt execution.

    Standard limits (30 iterations / 15 minutes). Excludes schedule and
    goal_pursuit skills to prevent recursion.
    """

    def process(self, message: str, item_id: str) -> dict:
        """Execute a scheduled prompt through the tool loop.

        Args:
            message: The scheduled message/task to execute.
            item_id: Scheduled item ID used for channel isolation.

        Returns:
            Standard MessageProcessor result dict.
        """
        from services.tool_schema_service import get_skill_schemas
        from services.innate_skills.registry import ALL_SKILL_NAMES
        from services.user_prompt_assembly_service import UserPromptAssemblyService

        system_prompt = _load_system_prompt()
        channel = f'scheduled:{item_id}'

        # Build user prompt with world state + episodic recall (same as UserMessageProcessor)
        user_prompt = UserPromptAssemblyService().build(
            user_message=message,
            channel=channel,
        ).to_provider()

        tool_names = [s for s in ALL_SKILL_NAMES if s not in _EXCLUDED_SKILLS]
        tools = get_skill_schemas(tool_names)

        logger.info(f"[SCHEDULED PROMPT] Starting item {item_id}: '{message[:80]}'")
        return self.send(user_prompt, system_prompt, channel=channel,
                         job='frontal-cortex-unified', tools=tools)
