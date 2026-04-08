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
  2. LLM invocation via Providers singleton, with full tool loop execution

Subclasses build the prompts and call self.send().
"""

import time
import logging

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 30
MAX_TIMEOUT = 900  # 15 minutes


class MessageProcessor:
    """Base class for all message processing pipelines.

    Handles: transcript persistence (in + out), LLM invocation via Providers,
    and the full tool-calling while loop.
    """

    def send(self, user_prompt, system_prompt, channel, job='unified', tools=None,
             request_id=None, on_narration=None):
        """Append user prompt to transcript → send to LLM → run tool loop → append response.

        Args:
            user_prompt: Assembled user-turn content (includes context, world state, etc.)
            system_prompt: Assembled system prompt (identity, directives, etc.)
            channel: Transcript channel identifier (e.g. 'user', 'system', interface name)
            job: Provider job name used to resolve the LLM config
            tools: Tool schemas to inject. If None, Providers resolves defaults.
            request_id: Per-request UUID used for steering queue (metadata['uuid']).
            on_narration: Optional callback(text, step) invoked when the LLM returns text
                          alongside tool calls mid-loop. Used to keep the WebSocket alive.

        Returns:
            dict with keys: response, generation_time, model, provider, tokens_input,
                            tokens_output, stop_reason, tool_calls, actions
        """
        from services.providers import Providers
        from services import transcript_service
        from services.tool_call_service import ToolCallService
        from services.act_dispatcher_service import ActDispatcherService

        # Persist user turn and get the transcript entry ID
        transcript_id = transcript_service.append(channel, 'user', user_prompt)

        # Resolve base tools once — these are the starting set before compounding
        if tools is None:
            from services.tool_schema_service import get_skill_schemas
            from services.innate_skills.registry import ALL_SKILL_NAMES
            tools = get_skill_schemas(list(ALL_SKILL_NAMES))

        base_tools = list(tools)
        current_tools = list(tools)

        # First LLM call (single user message)
        start = time.time()
        llm_response = Providers.instance().send(
            user_prompt, system_prompt, job=job, tools=current_tools
        )
        generation_time = time.time() - start

        if not llm_response.tool_calls:
            # Direct response — no tool loop needed
            if llm_response.text:
                transcript_service.append(channel, 'assistant', llm_response.text)
            return self._normalize_response(llm_response, generation_time)

        # --- Tool loop ---
        dispatcher = ActDispatcherService(execution_gate=False)
        tool_call_svc = ToolCallService()

        messages = [{"role": "user", "content": user_prompt}]
        iteration = 0
        timed_out = False
        loop_start = time.time()

        while llm_response.tool_calls and iteration < MAX_ITERATIONS:
            if time.time() - loop_start > MAX_TIMEOUT:
                logger.warning(f"[TOOL LOOP] Timeout after {iteration} iterations for channel={channel!r}")
                timed_out = True
                break

            # Emit per-iteration synthesis text (keeps WebSocket alive during long loops)
            if llm_response.text and on_narration:
                try:
                    on_narration(llm_response.text, iteration)
                except Exception as e:
                    logger.debug(f"[TOOL LOOP] on_narration callback failed: {e}")

            # Store synthesis text as ephemeral tool_synthesis if present
            if llm_response.text and transcript_id:
                tool_call_svc.store(
                    transcript_id, 'tool_synthesis', {}, llm_response.text,
                    invoked_by='llm', ephemeral=True,
                )

            # Translate LLM tool_calls to dispatcher format and execute
            dispatch_results = []
            for tc in llm_response.tool_calls:
                action = {'type': tc['name'], **tc.get('input', {})}
                try:
                    result = dispatcher.dispatch_action(channel, action)
                except Exception as e:
                    logger.error(f"[TOOL LOOP] Dispatch failed for {tc['name']}: {e}")
                    result = {
                        'action_type': tc['name'],
                        'status': 'error',
                        'result': f'Dispatch failed: {e}',
                        'execution_time': 0.0,
                    }
                dispatch_results.append(result)
                logger.debug(f"[TOOL LOOP] {tc['name']} → status={result.get('status')}")

            # Store results as ephemeral tool call records
            if transcript_id:
                tool_call_svc.store_batch(
                    transcript_id, llm_response.tool_calls, dispatch_results,
                    invoked_by='llm', ephemeral=True,
                )

            # Build assistant message with tool_calls for the messages array
            assistant_msg = {"role": "assistant", "content": llm_response.text or ""}
            if llm_response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.get('id', ''),
                        "name": tc['name'],
                        "input": tc.get('input', {}),
                    }
                    for tc in llm_response.tool_calls
                ]
            messages.append(assistant_msg)

            # Build tool result messages for the messages array
            for tc, r in zip(llm_response.tool_calls, dispatch_results):
                result_text = str(r.get('result', ''))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get('id', ''),
                    "content": result_text,
                    "name": tc['name'],  # Gemini requires this field
                })

            # Dynamic tool injection: compound find_tools results across iterations
            if transcript_id:
                current_tools = self._compound_tools(transcript_id, list(base_tools))

            # Steering: drain mid-turn user messages from MemoryStore
            steer = self._drain_steering(request_id)
            if steer:
                messages.append({"role": "user", "content": steer})
                if transcript_id:
                    tool_call_svc.store(
                        transcript_id, 'user_steer', {}, steer,
                        invoked_by='system', ephemeral=True,
                    )
                logger.debug(f"[TOOL LOOP] Steering injected: {steer[:60]!r}")

            # Next LLM call with full growing messages array
            llm_response = Providers.instance().send_messages(
                system_prompt, messages, job=job, tools=current_tools
            )
            iteration += 1

        generation_time = time.time() - start
        logger.info(f"[TOOL LOOP] Completed after {iteration} iteration(s) for channel={channel!r}")

        # Persist assistant turn (the final text response — not stored in tool_calls)
        if llm_response.text:
            transcript_service.append(channel, 'assistant', llm_response.text)

        return self._normalize_response(llm_response, generation_time, tool_loop_ran=(iteration > 0 or timed_out))

    def _compound_tools(self, transcript_id, base_tools):
        """Merge base tools with all find_tools results from this transcript.

        Tools list grows as find_tools discovers new capabilities — never shrinks
        within a turn.
        """
        from services.tool_call_service import ToolCallService
        from services.tool_schema_service import get_external_tool_schemas

        discovered = ToolCallService().get_find_tools_results(transcript_id)
        if not discovered:
            return base_tools

        new_schemas = get_external_tool_schemas(discovered)
        existing = {t['name'] for t in base_tools}
        for schema in new_schemas:
            if schema['name'] not in existing:
                base_tools.append(schema)
                existing.add(schema['name'])
        return base_tools

    def _drain_steering(self, request_id):
        """Drain steering queue from MemoryStore. Returns concatenated steer text or None."""
        if not request_id:
            return None
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            key = f"steer:{request_id}"
            steers = store.lrange(key, 0, -1)
            if steers:
                store.delete(key)
                parts = [s if isinstance(s, str) else s.decode() for s in steers]
                return '\n'.join(parts)
        except Exception as e:
            logging.debug(f"[MSG PROCESSOR] Steering check failed: {e}")
        return None

    def _normalize_response(self, llm_response, generation_time, tool_loop_ran=False):
        """Convert raw LLMResponse to standard result dict."""
        result = {
            'response': llm_response.text or '',
            'generation_time': generation_time,
            'model': llm_response.model,
            'provider': llm_response.provider,
            'tokens_input': llm_response.tokens_input,
            'tokens_output': llm_response.tokens_output,
            'stop_reason': llm_response.stop_reason,
            'tool_calls': None,
            'actions': None,
        }

        # If the tool loop ran to completion, all tool_calls were executed.
        # The final LLM response is text-only — nothing to surface as pending actions.
        if tool_loop_ran:
            return result

        # First-call tool_calls that were not yet processed (no loop entered)
        if llm_response.tool_calls:
            result['tool_calls'] = llm_response.tool_calls
            result['actions'] = [
                {'type': tc['name'], 'tool_call_id': tc.get('id'), **tc.get('input', {})}
                for tc in llm_response.tool_calls
            ]
            result['narration'] = llm_response.text or ''

        return result
