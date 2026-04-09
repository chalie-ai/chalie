# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
UserMessageProcessor — entry point for all user-initiated messages.

Builds user + system prompts, applies voice-mode tool filtering, then sends via
MessageProcessor.send() which handles transcript persistence, LLM invocation,
and the full tool-calling loop. Returns a normalized result dict.
"""

import logging
import threading

from services.message_processor import MessageProcessor

logger = logging.getLogger(__name__)


class UserMessageProcessor(MessageProcessor):
    """Processes user messages: build prompts → send → return.

    Usage:
        result = UserMessageProcessor.instance().process(prompt, metadata={...})
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def process(self, prompt, metadata=None):
        """Full user message pipeline. Returns result dict.

        Args:
            prompt: Raw user message text.
            metadata: Optional dict with keys: channel, source, image_ids, uuid, etc.

        Returns:
            dict with keys: response, channel, generation_time, model, provider,
                            tokens_input, tokens_output, stop_reason
        """
        from services.system_prompt_assembly_service import SystemPromptAssemblyService

        # 1. Channel from metadata (default: 'user')
        channel = (metadata or {}).get('channel', 'user')

        # 2. Build user prompt (per-turn, volatile)
        user_prompt = self._assemble_user_prompt(prompt, channel, metadata)

        # 3. Build system prompt (stable, cacheable)
        system_prompt = SystemPromptAssemblyService().build(
            type='unified',
            original_prompt=prompt,
        ).to_provider()

        # 4. Filter tools for voice mode — exclude rich_render (not speakable)
        tools = None
        if (metadata or {}).get('source') == 'voice':
            from services.tool_schema_service import get_skill_schemas
            from services.innate_skills.registry import ALL_SKILL_NAMES
            tools = get_skill_schemas([s for s in ALL_SKILL_NAMES if s != 'rich_render'])

        # 5. Thread request_id and narration callback for tool loop
        request_id = (metadata or {}).get('uuid')

        def _on_narration(text, step=0):
            """Publish per-iteration synthesis text to the per-request SSE channel."""
            if not request_id or not text:
                return
            try:
                from uuid import uuid4
                import json as _json
                from services.memory_client import MemoryClientService
                store = MemoryClientService.create_connection()
                narration_id = f"narr_{uuid4().hex[:12]}"
                store.set(f"output:{narration_id}", _json.dumps({
                    'type': 'act_narration',
                    'text': text,
                    'step': step,
                }), ex=300)
                store.publish(f"sse:{request_id}", narration_id)
            except Exception as e:
                logger.debug(f"[USER MSG] Narration publish failed: {e}")

        # 6. Send (MessageProcessor.send handles transcript + LLM call + tool loop)
        result = self.send(
            user_prompt, system_prompt,
            channel=channel,
            job='frontal-cortex-unified',
            tools=tools,
            request_id=request_id,
            on_narration=_on_narration,
        )

        result['channel'] = channel
        return result
