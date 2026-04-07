# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Providers — thin singleton wrapping provider resolution and LLM send.

Resolves the correct LLM provider for a given job (e.g. 'frontal-cortex-unified'),
sends messages, and returns a raw LLMResponse. No response parsing here.
"""

import threading
import logging

logger = logging.getLogger(__name__)


class Providers:
    """Singleton provider gateway. Resolves provider by job, sends messages."""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def send(self, user_prompt, system_prompt, job='unified', tools=None, cache_prefix=True):
        """Sync send. Returns LLMResponse."""
        if tools is None:
            tools = self._get_tools(job)
        provider = self._resolve(job)
        messages = [{"role": "user", "content": user_prompt}]
        return provider.send_messages(system_prompt, messages, cache_prefix=cache_prefix, tools=tools)

    def send_async(self, user_prompt, system_prompt, job='unified', tools=None, callback=None):
        """Fire-and-forget in daemon thread. Calls callback(LLMResponse) when done."""
        def _run():
            try:
                result = self.send(user_prompt, system_prompt, job=job, tools=tools)
                if callback:
                    callback(result)
            except Exception as e:
                logger.error(f"[Providers] send_async failed: {e}", exc_info=True)
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def _resolve(self, job):
        """Resolve LLM provider for a job. Uses ConfigService → create_llm_service."""
        from services.config_service import ConfigService
        from services.llm_service import create_llm_service
        config = ConfigService.resolve_agent_config(job)
        return create_llm_service(config)

    def _get_tools(self, job):
        """Get native tool schemas for a job. Default: all innate skills."""
        from services.tool_schema_service import get_skill_schemas
        from services.innate_skills.registry import ALL_SKILL_NAMES
        return get_skill_schemas(list(ALL_SKILL_NAMES))

    def get_context_limit(self, job='unified'):
        """Delegate to resolved provider."""
        return self._resolve(job).get_context_limit()

    def count_tokens(self, messages, system_prompt='', tools=None, job='unified'):
        """Delegate to resolved provider."""
        return self._resolve(job).count_tokens(messages, system_prompt, tools)
