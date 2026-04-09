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
  1. Transcript persistence (every turn stored to DB)
  2. LLM invocation with DB-backed context window construction

The context window is ALWAYS reconstructed from the database. Nothing
accumulates in memory except transcript IDs. Compaction triggers at 80%
of the provider's context limit.
"""

import time
import logging

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 30
MAX_TIMEOUT = 900  # 15 minutes


class MessageProcessor:
    """Base class for all message processing pipelines.

    Handles: transcript persistence (in + out), LLM invocation via Providers,
    and the full tool-calling while loop with DB-backed context windows.
    """

    def send(self, user_prompt, system_prompt, channel, job='unified', tools=None,
             request_id=None, on_narration=None):
        """Persist user prompt → build context from DB → send to LLM → run tool loop.

        The context window is always constructed from the database on every
        iteration. Nothing accumulates in memory. Compaction triggers at 80%
        of the provider's context limit.

        Args:
            user_prompt: Assembled user-turn content (world state, current message, etc.)
            system_prompt: Assembled system prompt (identity, directives, etc.)
            channel: Transcript channel identifier (e.g. 'user', interface name)
            job: Provider job name used to resolve the LLM config
            tools: Tool schemas to inject. If None, resolved from ALL_SKILL_NAMES.
            request_id: Per-request UUID used for the steering queue.
            on_narration: Optional callback(text, step) invoked when the LLM returns
                          text alongside tool calls mid-loop (keeps WebSocket alive).

        Returns:
            dict with keys: response, generation_time, model, provider, tokens_input,
                            tokens_output, stop_reason, tool_calls, actions
        """
        from services.providers import Providers
        from services import transcript_service
        from services.tool_call_service import ToolCallService
        from services.act_dispatcher_service import ActDispatcherService
        from services import context_window_service

        # 1. Persist user turn
        transcript_id = transcript_service.append(channel, 'user', user_prompt)

        # Resolve base tools once
        if tools is None:
            from services.tool_schema_service import get_skill_schemas
            from services.innate_skills.registry import ALL_SKILL_NAMES
            tools = get_skill_schemas(list(ALL_SKILL_NAMES))

        base_tools = list(tools)
        current_tools = list(tools)

        # Get context limit once for the lifetime of this request
        context_limit = Providers.instance().get_context_limit(job)

        # 2. Pre-call compaction check
        context_window_service.check_and_compact(channel, context_limit, job)

        # 3. Build messages from DB and make first LLM call
        start = time.time()
        messages = context_window_service.build_messages(channel)
        llm_response = Providers.instance().send_messages(
            system_prompt, messages, job=job, tools=current_tools
        )
        generation_time = time.time() - start

        if not llm_response.tool_calls:
            if llm_response.text:
                transcript_service.append(channel, 'assistant', llm_response.text)
            return self._normalize_response(llm_response, generation_time)

        # --- Tool loop ---
        dispatcher = ActDispatcherService(execution_gate=False)
        tool_call_svc = ToolCallService()

        # Track all transcript IDs from this turn (for find_tools compounding)
        turn_transcript_ids = [transcript_id] if transcript_id else []
        iteration = 0
        timed_out = False
        loop_start = time.time()
        max_iter = getattr(self, 'MAX_ITERATIONS', MAX_ITERATIONS)
        max_timeout = getattr(self, 'MAX_TIMEOUT', MAX_TIMEOUT)

        while llm_response.tool_calls and iteration < max_iter:
            if time.time() - loop_start > max_timeout:
                logger.warning(
                    f"[TOOL LOOP] Timeout after {iteration} iterations for channel={channel!r}"
                )
                timed_out = True
                break

            # Store assistant response to transcript
            asst_id = transcript_service.append(channel, 'assistant', llm_response.text or '')
            if asst_id:
                turn_transcript_ids.append(asst_id)

            # Narration callback — keeps WebSocket alive during long loops
            if llm_response.text and on_narration:
                try:
                    on_narration(llm_response.text, iteration)
                except Exception as e:
                    logger.debug(f"[TOOL LOOP] on_narration failed: {e}")

            # Execute tools and persist results
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

                result_text = str(result.get('result', ''))
                logger.debug(f"[TOOL LOOP] {tc['name']} → status={result.get('status')}")

                # Check for overflow BEFORE storing: would this result exceed the context limit?
                # If it would, compact now (result stored in overflow_content). After compaction,
                # the result is stored to transcript (id > watermark) and build_messages() will
                # emit: [overflow_content, compacted_text, tool_result_entry].
                # The overflow_content IS the tool result content placed before compacted_text.
                context_window_service.check_and_compact(
                    channel, context_limit, job,
                    pending_content=result_text,
                    is_tool_triggered=True,
                )

                # Store tool result to transcript
                tool_tid = transcript_service.append(
                    channel, 'tool', result_text,
                    tool_call_id=tc.get('id', ''),
                    tool_name=tc['name'],
                )
                if tool_tid:
                    turn_transcript_ids.append(tool_tid)

                # Store to tool_calls table linked to the assistant entry
                if asst_id:
                    tool_call_svc.store(
                        asst_id, tc['name'], tc.get('input', {}), result_text,
                        invoked_by='llm', tool_call_id=tc.get('id'),
                    )

            # Dynamic tool injection: compound find_tools results across iterations
            current_tools = self._compound_tools(turn_transcript_ids, list(base_tools))

            # Steering: drain mid-turn user messages from MemoryStore
            steer = self._drain_steering(request_id)
            if steer:
                steer_id = transcript_service.append(channel, 'user', steer)
                if steer_id:
                    turn_transcript_ids.append(steer_id)
                logger.debug(f"[TOOL LOOP] Steering injected: {steer[:60]!r}")

            # Post-iteration compaction check at 80% threshold
            context_window_service.check_and_compact(channel, context_limit, job)

            # Rebuild messages from DB and send next iteration
            messages = context_window_service.build_messages(channel)
            llm_response = Providers.instance().send_messages(
                system_prompt, messages, job=job, tools=current_tools
            )
            iteration += 1

        generation_time = time.time() - start
        logger.info(
            f"[TOOL LOOP] Completed after {iteration} iteration(s) for channel={channel!r}"
        )

        # Store final assistant response
        if llm_response.text:
            transcript_service.append(channel, 'assistant', llm_response.text)

        return self._normalize_response(
            llm_response, generation_time, tool_loop_ran=(iteration > 0 or timed_out)
        )

    def _compound_tools(self, transcript_ids, base_tools):
        """Merge base tools with find_tools results from any of the given transcript IDs."""
        from services.tool_call_service import ToolCallService
        from services.tool_schema_service import get_external_tool_schemas

        discovered = ToolCallService().get_find_tools_results(transcript_ids)
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
            logger.debug(f"[MSG PROCESSOR] Steering check failed: {e}")
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
