# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
ExternalAgentMessageProcessor — handles messages from external agents via MCP.

Lifecycle: one instance per external agent message. Instantiated by the MCP
server's tool handler, runs a full ACT loop, and optionally enqueues a
proactive UMP turn to disclose the exchange to the user.
"""

import logging

from services.message_processor import MessageProcessor
from services.system_message_prompt import ExternalAgentSystemMessagePrompt

logger = logging.getLogger(__name__)


class ExternalAgentMessageProcessor(MessageProcessor):
    """MessageProcessor subclass for external agent communication.

    Runs a full ACT loop with policy-gated tools. Conversation history
    is isolated per-agent via dynamic channel naming.
    """

    ROLE = 'user'
    JOB = 'external-agent'
    USAGE_CLASS = 'external_agent'
    SYSTEM_PROMPT_CLASS = ExternalAgentSystemMessagePrompt
    MAX_ITERATIONS = 20

    ALWAYS_AVAILABLE: list[str] = [
        "document",
        "find_tools",
        "list",
        "memory",
        "read",
        "review_tool_calls",
        "review_transcript",
        "schedule",
    ]
    DISCOVERABLE: list[str] = [
        "browser",
        "calendar",
        "code_eval",
        "contacts",
        "email",
        "home_assistant",
        "news",
        "programming_docs_search",
        "search",
        "weather",
    ]

    def __init__(
        self,
        raw_input: str,
        agent_name: str,
        project_or_task_name: str,
        loop_in_human: bool = False,
        wrapper_id: str | None = None,
        metadata: dict | None = None,
    ):
        super().__init__(raw_input, metadata)
        self._agent_name = agent_name
        self._project_or_task_name = project_or_task_name
        self._loop_in_human = loop_in_human
        self._wrapper_id = wrapper_id
        self._final_response: str = ''
        # Channel namespaced by wrapper_id to prevent cross-agent transcript access.
        prefix = f"{wrapper_id}:" if wrapper_id else ""
        self.CHANNEL = f"external-agent:{prefix}{agent_name}"

    def getUserDefinition(self) -> str:
        return (
            f"The user is {self._agent_name}, an external agent. "
            f"This conversation is about: {self._project_or_task_name}."
        )

    def getUserPrompt(self) -> str:
        """Build user-message body for one ACT iteration.

        Stripped compared to UMP: no world state, no system awareness,
        no file tags, no nudge. Keeps: previousMessages, memory seed,
        current turn, ACT trail.
        """
        parts = []

        # Previous Messages
        prev = self.getPreviousMessages()
        if prev:
            parts.append(f"## Previous Messages\n{prev}")

        parts.append('')

        # Memory seed (set by pre_act)
        if self._memory_seed:
            parts.append(self._memory_seed)

        # Current turn line
        parts.append(f"user: {self._raw_input}")

        # Drain pending steers from async subagent completions
        if self._pending_steers:
            steers = self._pending_steers[:]
            self._pending_steers.clear()
            for steer in steers:
                self._act_trail.append(steer)

        # ACT loop trail
        trail = self.getActLoopTrail()
        if trail:
            parts.append(trail)

        return '\n'.join(parts)

    def getSystemPrompt(self) -> str:
        """Build system prompt with template variables substituted."""
        body = self.SYSTEM_PROMPT_CLASS().getPrompt()
        body = self._substitute_provider_placeholders(body)

        user_name = self._resolve_user_name()

        body = (
            body
            .replace("{user_name}", user_name)
            .replace("{agent_name}", self._agent_name)
            .replace("{project_or_task_name}", self._project_or_task_name)
        )
        return f"{self.getUserDefinition()}\n\n{body}"

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
            logger.debug("[EAMP] _resolve_user_name failed: %s", e)
        return 'the user'

    def postTurn(self) -> None:
        """After ACT loop completes, optionally trigger user disclosure.

        When loop_in_human=True, spawns a ScheduledPromptProcessor (UMP with
        hidden input + full personality) so the disclosure goes through the
        normal Chalie voice — not a raw dump.
        """
        if not self._loop_in_human:
            return

        import threading

        disclosure_input = (
            f"An external agent called '{self._agent_name}' just contacted you "
            f"about '{self._project_or_task_name}'. "
            f"Here's what they said:\n\n\"{self._raw_input[:500]}\"\n\n"
            f"You replied:\n\n\"{self._final_response[:500]}\"\n\n"
            f"Let the user know about this exchange in your own words."
        )

        def _run():
            from services.output_service import OutputService
            from services.user_message_processor import UserMessageProcessor

            try:
                proc = UserMessageProcessor(raw_input=disclosure_input)
                proc.SKIP_INPUT_ROW = True
                response_text = proc.send()
                response_text = (response_text or '').strip()
                if not response_text:
                    response_text = (
                        f"{self._agent_name} reached out about "
                        f"'{self._project_or_task_name}' — check your transcript."
                    )
            except Exception as exc:
                logger.warning("[EAMP] Disclosure UMP failed: %s", exc, exc_info=True)
                response_text = (
                    f"{self._agent_name} sent a message about "
                    f"'{self._project_or_task_name}' but I couldn't summarize it."
                )

            try:
                OutputService().enqueue_proactive(
                    topic='user',
                    response=response_text,
                    source='external_agent',
                )
                logger.info(
                    "[EAMP] loop_in_human: disclosure delivered for agent=%s project=%s",
                    self._agent_name, self._project_or_task_name,
                )
            except Exception as exc:
                logger.warning("[EAMP] Disclosure delivery failed: %s", exc, exc_info=True)

        threading.Thread(target=_run, daemon=True, name="eamp-disclosure").start()
