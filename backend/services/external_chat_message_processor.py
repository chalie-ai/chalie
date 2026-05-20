# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
ExternalChatMessageProcessor — handles messages from external chat platforms (Discord).

Lifecycle: one instance per inbound Discord message. Instantiated by the Discord
capability's _on_discord_message handler, runs a full ACT loop, and returns the
LLM response text. The capability is responsible for sending the response back
to Discord.

Channel is fixed to 'discord' so all Discord turns share one transcript history,
giving Chalie cross-channel conversational context.
"""

import logging

from services.message_processor import MessageProcessor
from services.system_message_prompt import ExternalChatSystemMessagePrompt

logger = logging.getLogger(__name__)


class ExternalChatMessageProcessor(MessageProcessor):
    """MessageProcessor subclass for external chat platform messages.

    Runs a full ACT loop with policy-gated tools under the 'discord' policy
    context. Conversation history is shared across all Discord messages on the
    'discord' channel.

    When the inbound author matches the owner's Discord user ID, ROLE is set
    to 'user' (owner trust). Otherwise ROLE is 'third-party-user'.
    """

    ROLE = 'third-party-user'
    LOG_LABEL = 'discord'
    USAGE_CLASS = 'discord'
    SYSTEM_PROMPT_CLASS = ExternalChatSystemMessagePrompt
    MAX_ITERATIONS = 20
    CHANNEL = 'discord'

    ALWAYS_AVAILABLE: list[str] = [
        "find_tools",
        "memory",
        "read",
    ]
    DISCOVERABLE: list[str] = [
        "browser",
        "calendar",
        "contacts",
        "document",
        "email",
        "list",
        "news",
        "programming_docs_search",
        "search",
        "weather",
    ]

    def __init__(
        self,
        raw_input: str,
        metadata: dict | None,
        author_name: str,
        author_id: int,
        owner_user_id: int | None,
        guild_name: str,
        channel_name: str,
        custom_system_prompt: str,
    ):
        super().__init__(raw_input, metadata)
        self._author_name = author_name
        self._author_id = author_id
        self._owner_user_id = owner_user_id
        self._guild_name = guild_name
        self._channel_name = channel_name
        self._custom_system_prompt = custom_system_prompt
        self._final_response: str = ''

        if owner_user_id is not None and author_id == owner_user_id:
            self.ROLE = 'user'

    def get_user_definition(self) -> str:
        if self.ROLE == 'user':
            return (
                f"The user is your primary user, speaking via Discord "
                f"in {self._guild_name} #{self._channel_name}."
            )
        return (
            f"The user is {self._author_name}, a third-party user on Discord "
            f"in {self._guild_name} #{self._channel_name}."
        )

    def get_user_prompt(self) -> str:
        """Build user-message body for one ACT iteration.

        Stripped compared to UMP: no world state, no system awareness,
        no file tags, no nudge. Keeps: previousMessages, memory seed,
        current turn, ACT trail.
        """
        parts = []

        prev = self.get_previous_messages()
        if prev:
            parts.append(f"## Previous Messages\n{prev}")

        parts.append('')

        if self._memory_seed:
            parts.append(self._memory_seed)

        parts.append(f"{self._author_name}: {self._raw_input}")

        trail = self.get_act_loop_trail()
        if trail:
            parts.append(trail)

        return '\n'.join(parts)

    def get_system_prompt(self) -> str:
        """Build system prompt with template variables substituted."""
        body = self.SYSTEM_PROMPT_CLASS().get_prompt()
        body = self._substitute_provider_placeholders(body)

        user_name = self._resolve_user_name()
        owner_discord_id = str(self._owner_user_id) if self._owner_user_id is not None else ''

        body = (
            body
            .replace("{user_name}", user_name)
            .replace("{owner_discord_id}", owner_discord_id)
            .replace("{custom_instructions}", self._custom_system_prompt)
        )
        return f"{self.get_user_definition()}\n\n{body}"

    def store(self, llm_response: str) -> None:
        """Capture the final response text before delegating to the base store."""
        self._final_response = llm_response or ''
        super().store(llm_response)

    def _resolve_user_name(self) -> str:
        """Get the user's first name from data_graph, falling back to 'the user'."""
        try:
            from services.data_graph_service import get_data_graph_service
            dgs = get_data_graph_service()
            rows = dgs.fetch(kinds=['system'])
            for row in rows:
                key = row['key'] if hasattr(row, '__getitem__') else getattr(row, 'key', None)
                val = row['value'] if hasattr(row, '__getitem__') else getattr(row, 'value', None)
                if key == 'user_summary' and val:
                    first_word = val.split()[0] if val else ''
                    return first_word if first_word else 'the user'
        except Exception as e:
            logger.debug("[ECMP] _resolve_user_name failed: %s", e)
        return 'the user'
