"""
ScheduledMessageProcessor — Executes scheduled prompts via the standard tool loop.

Simple MessageProcessor subclass that runs a scheduled prompt in an isolated
channel. Used by scheduler_service._fire_item() when is_prompt=True.

Injects full user-prompt context (world state, episodic recall, system awareness)
via UserPromptAssemblyService so the LLM has the same memory context as a normal
user interaction.
"""

import os
import logging

from services.message_processor import MessageProcessor

logger = logging.getLogger(__name__)

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'scheduled-prompt.md')

# Skills that must not be available during scheduled prompt execution
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

    Limits match the standard tool loop — scheduled tasks should not need
    extended time beyond a normal interaction.
      - MAX_ITERATIONS: 30
      - MAX_TIMEOUT: 900 seconds / 15 minutes
    """

    MAX_ITERATIONS = 30
    MAX_TIMEOUT = 900

    def process(self, message: str, item_id: str, channel: str = None) -> dict:
        """Execute a scheduled prompt through the tool loop.

        Args:
            message: The scheduled message/task to execute.
            item_id: Scheduled item ID used for channel isolation.
            channel: Optional channel override (defaults to scheduled:{item_id}).

        Returns:
            Standard MessageProcessor result dict.
        """
        from services.tool_schema_service import get_skill_schemas
        from services.innate_skills.registry import ALL_SKILL_NAMES
        from services.user_prompt_assembly_service import UserPromptAssemblyService

        system_prompt = _load_system_prompt()
        _channel = channel or f'scheduled:{item_id}'

        # Frame as an execution instruction — prevents the LLM from reading the
        # raw scheduled message (e.g. "At 09:30 every morning, check …") and
        # treating it as a new scheduling request instead of executing the task.
        raw_instruction = (
            f"[SCHEDULED TASK — EXECUTE NOW]\n"
            f"This is a previously scheduled prompt that is now due. "
            f"Execute the task described below immediately. "
            f"Do NOT create a new schedule or reminder — it is already scheduled "
            f"and will fire again automatically.\n\n"
            f"{message}"
        )

        # Inject full user-prompt context (world state, episodic recall, system
        # awareness) so the LLM has the same memory context as a normal turn.
        user_prompt = UserPromptAssemblyService().build(
            user_message=raw_instruction,
            channel=_channel,
            thread_id=_channel,
        ).to_provider()

        # All skills except schedule (recursive) and goal_pursuit (background-only)
        tool_names = [s for s in ALL_SKILL_NAMES if s not in _EXCLUDED_SKILLS]
        tools = get_skill_schemas(tool_names)

        logger.info(f"[SCHEDULED PROMPT] Starting item {item_id}: '{message[:80]}'")
        return self.send(
            user_prompt,
            system_prompt,
            channel=_channel,
            job='frontal-cortex-unified',
            tools=tools,
        )
