# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Frontal Cortex Service — thin facade delegating to PromptAssemblyService and
ResponseGenerationService (WS3 decomposition).

Re-exports ``ChatHistoryProcessor`` and ``_ONBOARDING_SCHEDULE`` so any
existing callers that import those names from this module continue to work.
"""

# WS4 batch-5: silent catches audited 2026-03-23 — zero try/except blocks in facade

from services.prompt_assembly_service import (
    PromptAssemblyService,
    _ONBOARDING_SCHEDULE,
)
from services.response_generation_service import (
    ResponseGenerationService,
    ChatHistoryProcessor,
)

__all__ = ['FrontalCortexService', 'ChatHistoryProcessor', '_ONBOARDING_SCHEDULE']


class FrontalCortexService:
    """Facade preserving the original FrontalCortexService public API.

    All prompt-assembly work is delegated to ``self._prompt_assembly``
    (:class:`PromptAssemblyService`) and all LLM/response-parsing work is
    delegated to ``self._response_gen`` (:class:`ResponseGenerationService`).
    """

    def __init__(self, config: dict):
        """Initialise sub-services with the given provider config dict.

        Args:
            config: Provider configuration dict (requires at least a
                ``platform`` key).  Forwarded unchanged to both sub-services.
        """
        self.config = config
        self._prompt_assembly = PromptAssemblyService(config)
        self._response_gen = ResponseGenerationService(config)

    # ── LLM capability ────────────────────────────────────────────────────────

    def get_context_limit(self) -> int:
        """Return the LLM provider's maximum context-window token count.

        Delegates to :meth:`ResponseGenerationService.get_context_limit`.
        """
        return self._response_gen.get_context_limit()

    def count_tokens(
        self, messages: list, system_prompt: str = '', tools: list = None
    ) -> int:
        """Count tokens for *messages* + *system_prompt* (+ optional *tools*).

        Delegates to :meth:`ResponseGenerationService.count_tokens`.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            system_prompt: System prompt string (may be empty).
            tools: Optional list of native tool schema dicts.

        Returns:
            Estimated integer token count.
        """
        return self._response_gen.count_tokens(messages, system_prompt, tools)

    # ── Prompt assembly ───────────────────────────────────────────────────────

    def build_system_prompt(
        self,
        system_prompt_template: str,
        original_prompt: str,
        classification: dict,
        chat_history: list,
        assembled_context: dict = None,
        relevant_tools: list = None,
        selected_tools: list = None,
        selected_skills: list = None,
        thread_id: str = None,
        returning_from_silence: bool = False,
        inclusion_map: dict = None,
    ) -> str:
        """Build the stable system prompt (act_history excluded).

        Used by the ACT loop — act_history travels in the message array to
        allow provider-side prompt caching on the stable system prefix.

        Delegates to :meth:`PromptAssemblyService.build_system_prompt`.
        """
        return self._prompt_assembly.build_system_prompt(
            system_prompt_template, original_prompt, classification, chat_history,
            assembled_context=assembled_context, relevant_tools=relevant_tools,
            selected_tools=selected_tools, selected_skills=selected_skills,
            thread_id=thread_id, returning_from_silence=returning_from_silence,
            inclusion_map=inclusion_map,
        )

    # ── Response generation ───────────────────────────────────────────────────

    def generate_response(
        self,
        system_prompt_template: str,
        original_prompt: str,
        classification: dict,
        chat_history: list,
        act_history: str = "",
        assembled_context: dict = None,
        relevant_tools: list = None,
        selected_tools: list = None,
        selected_skills: list = None,
        thread_id: str = None,
        returning_from_silence: bool = False,
        inclusion_map: dict = None,
    ) -> dict:
        """Assemble the full system prompt then invoke the LLM.

        Orchestrates :class:`PromptAssemblyService` (template injection) and
        :class:`ResponseGenerationService` (LLM call + JSON parsing).

        Args:
            system_prompt_template: Template with ``{{variable}}`` placeholders.
            original_prompt: The user's raw message.
            classification: Classification result dict (topic, confidence, …).
            chat_history: List of previous exchange dicts.
            act_history: ACT loop history string (default empty).
            assembled_context: Pre-assembled context dict.
            relevant_tools: Tools scored by embedding relevance.
            selected_tools: Triage-selected tool names.
            selected_skills: Triage-selected innate skill names.
            thread_id: Conversation thread identifier.
            returning_from_silence: True when user resumes after a long gap.
            inclusion_map: Context node inclusion decisions.

        Returns:
            Standard cortex result dict (``mode``, ``response``,
            ``actions``, ``confidence``, ``generation_time``, …).
        """
        system_prompt = self._prompt_assembly._inject_parameters(
            system_prompt_template, original_prompt, classification, chat_history,
            act_history=act_history, assembled_context=assembled_context,
            relevant_tools=relevant_tools, selected_tools=selected_tools,
            selected_skills=selected_skills, thread_id=thread_id,
            returning_from_silence=returning_from_silence, inclusion_map=inclusion_map,
        )
        return self._response_gen.generate_response(
            system_prompt, original_prompt, system_prompt_template,
        )

    def generate_response_appended(
        self,
        system_prompt: str,
        messages: list,
        cache_prefix: bool = False,
        tools: list = None,
    ) -> dict:
        """Multi-turn LLM call using a pre-built system prompt.

        Delegates to :meth:`ResponseGenerationService.generate_response_appended`.

        Args:
            system_prompt: Fully-injected system prompt (no act_history).
            messages: Growing list of ``{"role": ..., "content": ...}`` dicts.
            cache_prefix: Hint to provider to cache the system-prompt prefix.
            tools: Native tool schema dicts for tool-calling mode.

        Returns:
            Cortex result dict, plus ``raw_response`` key.
        """
        return self._response_gen.generate_response_appended(
            system_prompt, messages, cache_prefix, tools,
        )

    def _parse_response_text(
        self, response_text: str, generation_time: float
    ) -> dict:
        """Parse raw LLM text into the standard cortex result dict.

        Delegates to :meth:`ResponseGenerationService._parse_response_text`.

        Args:
            response_text: Raw text returned by the LLM provider.
            generation_time: Wall-clock seconds for the LLM call.

        Returns:
            Standard cortex result dict.
        """
        return self._response_gen._parse_response_text(response_text, generation_time)
