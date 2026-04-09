"""
DMNMessageProcessor — Background DMN execution with reduced limits.

MessageProcessor subclass that runs a DMN proactive review in an isolated
channel with shorter iteration and timeout limits than normal turns.
Used by DMNService to process pre-built context strings through the full
tool-calling pipeline.
"""

import os
import logging

from services.message_processor import MessageProcessor

logger = logging.getLogger(__name__)

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'dmn.md')


def _load_system_prompt() -> str:
    """Load the DMN system prompt from disk."""
    try:
        with open(os.path.normpath(_PROMPT_PATH), 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.warning(f"[DMN] Failed to load system prompt: {e}")
        return (
            "You are Chalie, running a background review of recent activity. "
            "Based on the episodes provided, do you detect any patterns, unresolved threads, "
            "or opportunities to be proactive? If so, act on it using the tools available to you. "
            "If nothing actionable stands out, respond with exactly: DMN_NO_ACTION"
        )


class DMNMessageProcessor(MessageProcessor):
    """MessageProcessor subclass for background DMN proactive reviews.

    Overrides limits to prevent runaway tool chains:
      - MAX_ITERATIONS: 15 (vs. 30 for normal turns)
      - MAX_TIMEOUT: 300 seconds / 5 minutes (vs. 900 for normal turns)
    """

    MAX_ITERATIONS = 15
    MAX_TIMEOUT = 300  # 5 minutes

    def _assemble_user_prompt(self, message, channel, metadata=None):
        """DMN context is pre-built by DMNService — pass through unchanged."""
        return message

    def process(self, context: str, mode: str, channel: str) -> dict:
        """Run a DMN proactive review in the given channel.

        Args:
            context: Pre-assembled context string from DMNService.
            mode: DMN mode label ('recent' or 'salience') for logging.
            channel: Transcript channel to use (resolved by DMNService).

        Returns:
            Standard MessageProcessor result dict.
        """
        from services.tool_schema_service import get_skill_schemas
        from services.innate_skills.registry import ALL_SKILL_NAMES

        system_prompt = _load_system_prompt()

        # All skills except goal_pursuit (prevent spawning background tasks from DMN)
        tool_names = [s for s in ALL_SKILL_NAMES if s != 'goal_pursuit']
        tools = get_skill_schemas(tool_names)

        logger.info(f"[DMN] Starting {mode} review on channel '{channel}'")
        return self.send(
            context,
            system_prompt,
            channel=channel,
            job='frontal-cortex-unified',
            tools=tools,
        )
