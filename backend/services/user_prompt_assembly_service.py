# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
User Prompt Assembly Service — builds the per-turn user message.

Assembles the full user-facing message sent to the LLM on each turn:
  1. World state header (time, calendar, weather)
  2. System awareness (degradation signals)
  3. Current turn: episodic auto-recall + user message + file tags + nudge

Conversation history is NOT included here. It is constructed directly from
the database by context_window_service.build_messages() and injected into
the messages array by MessageProcessor on every iteration.

This content changes every turn and is never cached.
"""

import logging

from services.prompt_assembly_contract import PromptAssemblyContract

logger = logging.getLogger(__name__)


class UserPromptAssemblyService(PromptAssemblyContract):
    """Builds the per-turn user message from world state, transcript, and memory.

    Usage:
        svc = UserPromptAssemblyService()
        svc.build(user_message, topic, ...)
        user_prompt = svc.to_provider()
    """

    def __init__(self):
        self._result = ''

    def build(self, user_message: str, channel: str, thread_id: str = None,
              metadata: dict = None) -> 'UserPromptAssemblyService':
        """Build the full user prompt for a single turn.

        Returns self for chaining: `svc.build(...).to_provider()`
        """
        parts = []

        # 1. World state header
        world_state = self._get_world_state(channel, thread_id)
        if world_state:
            parts.append(f"## World State\n{world_state}")

        # 1b. Voice mode instruction (per-turn — user switches between voice and text)
        if (metadata or {}).get('source') == 'voice':
            parts.append(
                'IMPORTANT: The user is in voice mode. Your response will be spoken aloud via TTS. '
                'Respond in plain conversational text only. No markdown formatting, code blocks, '
                'tables, bullet lists, links, or structured formatting. Write as you would speak.'
            )

        # 2. System awareness (degradation signals — strong section, before current turn)
        self_awareness = self._get_self_awareness()
        if self_awareness:
            parts.append(f"## System Awareness\n{self_awareness}")

        # 3. Current turn (includes file tags and nudge for this turn)
        file_tags = (metadata or {}).get('file_tags', [])
        nudge_tag = (metadata or {}).get('nudge_tag')
        current_turn = self._build_current_turn(user_message, channel, file_tags, nudge_tag)
        parts.append(current_turn)

        self._result = '\n\n'.join(parts)
        return self

    def to_provider(self) -> str:
        return self._result

    # ── Internal builders ────────────────────────────────────────────

    def _get_world_state(self, channel: str, thread_id: str = None) -> str:
        try:
            from services.world_state_service import WorldStateService
            svc = WorldStateService()
            return svc.get_world_state(channel, thread_id=thread_id)
        except Exception as e:
            logger.info(f"[USER PROMPT] World state unavailable: {e}")
            return ''

    def _build_current_turn(self, user_message: str, channel: str,
                            file_tags: list = None,
                            nudge_tag: str = None) -> str:
        parts = ["# Current Turn"]

        memories = self._get_episodic_recall(user_message)
        if memories:
            parts.append(f"### Related Memories\n{memories}")

        parts.append(f"## User Message\n{user_message}")

        if file_tags:
            for tag in file_tags:
                parts.append(tag)

        if nudge_tag:
            parts.append(nudge_tag)

        return '\n\n'.join(parts)

    def _get_self_awareness(self) -> str:
        """Get system health signals — only non-empty when degraded."""
        try:
            from services.self_model_service import SelfModelService
            return SelfModelService().format_for_prompt()
        except Exception as e:
            logger.info(f"[USER PROMPT] Self-awareness unavailable: {e}")
            return ''

    def _get_episodic_recall(self, query_text: str) -> str:
        # Seed recall routes through the memory skill's dynamic-radius path
        # so every seed call emits a memory_recall_log row with caller='seed'.
        # When a MessageProcessor v2 turn is bound, this also pre-populates
        # `_memory_query_history` so subsequent LLM-side memory calls can
        # compute drift vs the seed query. When no processor is bound (legacy
        # orchestrator), telemetry still fires; history is simply empty.
        try:
            from services.episodic_service import EpisodicService
            from services.database_service import get_shared_db_service
            from services.innate_skills import memory_skill

            db = get_shared_db_service()
            svc = EpisodicService(db)

            raw_episodes, _status = memory_skill.recall_episodes(
                channel='',  # legacy path has no channel at this point
                query=query_text,
                caller='seed',
                baseline_radius=memory_skill.SEED_RADIUS_BASELINE,
                return_raw=True,
            )

            if not raw_episodes:
                return ''

            return svc.format_for_prompt(raw_episodes)

        except Exception as e:
            logger.info(f"[USER PROMPT] Episodic recall failed: {e}")
            return ''
