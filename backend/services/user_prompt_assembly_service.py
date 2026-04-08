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
  2. Previous messages (transcript with interleaved tool_calls and memory tags)
  3. Compaction summary (when older turns have been compacted)
  4. Current turn: episodic auto-recall + user message

This content changes every turn and is never cached.
"""

import logging
from typing import List, Dict

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

        # 2. Previous messages (compaction + transcript + tool calls)
        conversation = self._get_conversation_context(channel)
        if conversation:
            parts.append(conversation)

        # 3. System awareness (degradation signals — strong section, before current turn)
        self_awareness = self._get_self_awareness()
        if self_awareness:
            parts.append(f"## System Awareness\n{self_awareness}")

        # 4. Current turn (includes file tags and nudge for this turn)
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

    def _get_conversation_context(self, channel: str) -> str:
        """Build conversation context from compaction + recent transcript + tool calls."""
        try:
            from services import compaction_service, transcript_service
            from services.database_service import get_shared_db_service

            compaction = compaction_service.get_compaction(channel)
            watermark = compaction['compacted_up_to_id'] if compaction else 0

            entries = transcript_service.get_recent(channel, limit=50, since_id=watermark)

            parts = []

            if compaction and compaction.get('compacted_text'):
                parts.append(f"## Context\n{compaction['compacted_text']}")

            if entries:
                tool_calls = self._get_tool_calls_for_entries(entries)
                lines = ["## Previous Messages"]
                for entry in entries:
                    role = 'User' if entry.get('role') == 'user' else 'System'
                    created = str(entry.get('created_at', ''))[:16]
                    content = entry.get('content', '')
                    lines.append(f"[{created}] {role}: {content}")

                    entry_id = entry.get('id')
                    if entry_id and entry_id in tool_calls:
                        for tc in tool_calls[entry_id]:
                            tool_name = tc.get('tool_name', '')
                            result = tc.get('result', '')
                            if tool_name == 'memory' and result:
                                lines.append(f"[memory] {result}")
                            elif tool_name in ('file', 'nudge') and result:
                                lines.append(result)
                            elif result:
                                lines.append(f"[tool:{tool_name}] {result}")

                parts.append('\n'.join(lines))

            if not parts:
                return ''

            return '\n\n'.join(parts)

        except Exception as e:
            logger.info(f"[USER PROMPT] Conversation context failed: {e}")
            return ''

    def _get_tool_calls_for_entries(self, entries: list) -> Dict[int, List[dict]]:
        """Fetch tool_calls rows for the given transcript entries, grouped by transcript_id."""
        try:
            from services.database_service import get_shared_db_service

            entry_ids = [e.get('id') for e in entries if e.get('id') is not None]
            if not entry_ids:
                return {}

            db = get_shared_db_service()
            with db.connection() as conn:
                placeholders = ','.join('?' for _ in entry_ids)
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT transcript_id, tool_name, result FROM tool_calls "
                    f"WHERE transcript_id IN ({placeholders}) AND ephemeral = 0 ORDER BY created_at",
                    entry_ids,
                )
                rows = cursor.fetchall()

            grouped: Dict[int, List[dict]] = {}
            for row in rows:
                tid = row[0]
                if tid not in grouped:
                    grouped[tid] = []
                grouped[tid].append({
                    'tool_name': row[1],
                    'result': row[2],
                })
            return grouped

        except Exception as e:
            logger.info(f"[USER PROMPT] Tool calls fetch failed: {e}")
            return {}

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
        try:
            from services.episodic_service import EpisodicService
            from services.database_service import get_shared_db_service

            db = get_shared_db_service()
            svc = EpisodicService(db)
            episodes = svc.retrieve_episodes(query_text=query_text, radius=0.2)

            if not episodes:
                return ''

            return svc.format_for_prompt(episodes)

        except Exception as e:
            logger.info(f"[USER PROMPT] Episodic recall failed: {e}")
            return ''
