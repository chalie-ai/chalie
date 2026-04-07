# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
UserMessageProcessor — entry point for all user-initiated messages.

Builds user + system prompts, applies voice-mode tool filtering, sends via
MessageProcessor.send(), parks ACT tool calls (task-ddffe1), and returns
a normalized result dict.
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
        from services.user_prompt_assembly_service import UserPromptAssemblyService
        from services.system_prompt_assembly_service import SystemPromptAssemblyService

        # 1. Channel from metadata (default: 'user')
        channel = (metadata or {}).get('channel', 'user')

        # 2. Build user prompt (per-turn, volatile)
        user_prompt = UserPromptAssemblyService().build(
            user_message=prompt,
            channel=channel,
            metadata=metadata,
        ).to_provider()

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

        # 5. Send (MessageProcessor.send handles transcript + LLM call)
        result = self.send(
            user_prompt, system_prompt,
            channel=channel,
            job='frontal-cortex-unified',
            tools=tools,
        )

        # 6. ACT parked (task-ddffe1): if LLM returned tool calls, log warning
        #    and return narration text as the response. Tool execution is deferred.
        if result.get('actions'):
            logger.warning(
                f"[USER MSG] LLM returned {len(result['actions'])} tool call(s) "
                f"but ACT is parked (task-ddffe1). Returning narration as response."
            )
            result['response'] = result.get('narration', '') or result.get('response', '')
            result['actions'] = None
            result['tool_calls'] = None

        result['channel'] = channel
        return result
