# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
MessageProcessor — base class for all message processing pipelines.

Handles two responsibilities:
  1. Transcript persistence (user turn in, assistant turn out)
  2. LLM invocation via Providers singleton

Subclasses build the prompts and call self.send().
"""

import time
import logging

logger = logging.getLogger(__name__)


class MessageProcessor:
    """Base class for all message processing pipelines.

    Handles: transcript persistence (in + out) and LLM invocation via Providers.
    """

    def send(self, user_prompt, system_prompt, channel, job='unified', tools=None):
        """Append user prompt to transcript → send to LLM → append response to transcript.

        Args:
            user_prompt: Assembled user-turn content (includes context, world state, etc.)
            system_prompt: Assembled system prompt (identity, directives, etc.)
            channel: Transcript channel identifier (e.g. 'user', 'system', interface name)
            job: Provider job name used to resolve the LLM config
            tools: Tool schemas to inject. If None, Providers resolves defaults.

        Returns:
            dict with keys: response, generation_time, model, provider, tokens_input,
                            tokens_output, stop_reason, tool_calls, actions
        """
        from services.providers import Providers
        from services import transcript_service

        # Persist user turn
        transcript_service.append(channel, 'user', user_prompt)

        # LLM call
        start = time.time()
        llm_response = Providers.instance().send(
            user_prompt, system_prompt, job=job, tools=tools
        )
        generation_time = time.time() - start

        result = self._normalize_response(llm_response, generation_time)

        # Persist assistant turn
        if result.get('response'):
            transcript_service.append(channel, 'assistant', result['response'])

        return result

    def _normalize_response(self, llm_response, generation_time):
        """Convert raw LLMResponse to standard result dict."""
        result = {
            'response': llm_response.text or '',
            'generation_time': generation_time,
            'model': llm_response.model,
            'provider': llm_response.provider,
            'tokens_input': llm_response.tokens_input,
            'tokens_output': llm_response.tokens_output,
            'stop_reason': llm_response.stop_reason,
            'tool_calls': llm_response.tool_calls,
            'actions': None,
        }

        if llm_response.tool_calls:
            result['actions'] = [
                {'type': tc['name'], 'tool_call_id': tc.get('id'), **tc.get('input', {})}
                for tc in llm_response.tool_calls
            ]
            result['narration'] = llm_response.text or ''

        return result
