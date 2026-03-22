# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Frontal Cortex Service — LLM response generation and context assembly.

Assembles the full system prompt from 60+ context injection placeholders
(memory, identity, user traits, calendar, tool results, etc.) and invokes
the configured LLM provider to generate a structured JSON response.

Also hosts :class:`ChatHistoryProcessor`, a lightweight helper that
truncates and serialises chat history for prompt injection.
"""

import re
import time
import json
from typing import Optional
import logging

# ─────────────────────────────────────────────────────────────────────────────
# Onboarding schedule — defines which identity traits to elicit, at what turn
# count, and with what behavioural hint to the LLM. Traits are elicited in list
# order; later entries can be added here without touching any other code.
# ─────────────────────────────────────────────────────────────────────────────
_ONBOARDING_SCHEDULE = [
    # ── Phase 0: First impression ─────────────────────────────────────────────
    {
        'trait': 'name',
        'min_turn': 3,
        'cooldown_turns': 6,
        'max_attempts': 2,
        'hint': (
            "You haven't learned the user's name yet. "
            "Find a natural moment in your response to ask what they'd like to be called. "
            "Keep it casual — don't make it the focus of the response."
        ),
    },
    # ── Phase 1: Core identity ────────────────────────────────────────────────
    {
        'trait': 'age_range',
        'min_turn': 5,
        'cooldown_turns': 8,
        'max_attempts': 1,
        'hint': (
            "If it comes up naturally, ask their general age range (e.g. 20s, 30s). "
            "Don't press if it doesn't fit — it helps adapt communication style."
        ),
    },
    {
        'trait': 'occupation',
        'min_turn': 8,
        'cooldown_turns': 10,
        'max_attempts': 1,
        'hint': (
            "If relevant to the conversation, ask what they do — "
            "engineer, designer, student, etc. "
            "Helps understand context and time constraints."
        ),
    },
    # ── Phase 2: Communication & work ────────────────────────────────────────
    {
        'trait': 'timezone',
        'min_turn': 15,
        'cooldown_turns': 10,
        'max_attempts': 1,
        'hint': (
            "If relevant, ask where the user is based — "
            "it helps with scheduling context."
        ),
    },
    {
        'trait': 'communication_preference',
        'min_turn': 18,
        'cooldown_turns': 12,
        'max_attempts': 1,
        'hint': (
            "Ask whether they prefer detailed explanations or quick summaries. "
            "Do they like theory first, or jump straight to examples?"
        ),
    },
    {
        'trait': 'interests',
        'min_turn': 20,
        'cooldown_turns': 15,
        'max_attempts': 1,
        'hint': (
            "If the conversation touches on hobbies or interests, "
            "ask what they enjoy working on or exploring."
        ),
    },
    {
        'trait': 'work_schedule',
        'min_turn': 22,
        'cooldown_turns': 12,
        'max_attempts': 1,
        'hint': (
            "If scheduling or availability comes up, ask when they're usually free. "
            "Helps with proactive outreach timing."
        ),
    },
    {
        'trait': 'primary_goal',
        'min_turn': 25,
        'cooldown_turns': 15,
        'max_attempts': 1,
        'hint': (
            "Ask what they're mainly trying to accomplish right now. "
            "Helps prioritise what matters most."
        ),
    },
    # ── Phase 3: Refinement ───────────────────────────────────────────────────
    {
        'trait': 'learning_style',
        'min_turn': 30,
        'cooldown_turns': 20,
        'max_attempts': 1,
        'hint': (
            "Ask whether they prefer examples, step-by-step guides, "
            "or high-level conceptual overviews."
        ),
    },
]


class ChatHistoryProcessor:
    """Processes chat history for context injection into prompts."""

    def __init__(self, max_exchanges: int = None, max_tokens: int = None):
        """Initialize the chat history processor with optional window limits.

        Args:
            max_exchanges: Maximum number of most-recent exchanges to include.
                ``None`` means no exchange-count limit.
            max_tokens: Maximum token budget for the serialised history.
                ``None`` means no token limit (not yet enforced).
        """
        self.max_exchanges = max_exchanges
        self.max_tokens = max_tokens

    def process(self, chat_history: list) -> str:
        """Serialise chat history into a plain-text prompt snippet.

        Args:
            chat_history: List of exchange dicts, each with a ``'prompt'``
                sub-dict containing ``'message'`` and an optional ``'response'``
                value (dict or string).

        Returns:
            A newline-joined string of ``"User: …"`` / ``"Assistant: …"`` lines,
            or ``"No previous conversation"`` when the list is empty.
        """
        if not chat_history:
            return "No previous conversation"

        limited_history = self._apply_limits(chat_history)

        lines = []
        for exchange in limited_history:
            if 'prompt' in exchange and 'message' in exchange['prompt']:
                lines.append(f"User: {exchange['prompt']['message']}")
            if 'response' in exchange:
                if isinstance(exchange['response'], dict):
                    if 'message' in exchange['response']:
                        lines.append(f"Assistant: {exchange['response']['message']}")
                    elif 'error' in exchange['response']:
                        lines.append(f"Assistant: [Error: {exchange['response']['error']}]")
                else:
                    lines.append(f"Assistant: {exchange['response']}")

        return "\n".join(lines) if lines else "No previous conversation"

    def _apply_limits(self, chat_history: list) -> list:
        """Trim the history list to ``max_exchanges`` most-recent entries.

        Args:
            chat_history: Full chat history list.

        Returns:
            A (possibly truncated) list containing at most ``max_exchanges``
            entries from the tail of ``chat_history``.
        """
        if self.max_exchanges and len(chat_history) > self.max_exchanges:
            chat_history = chat_history[-self.max_exchanges:]
        return chat_history


def _extract_response_from_broken_json(text: str) -> str | None:
    """
    Extract the 'response' value from broken JSON where inner quotes are unescaped.

    Strategy: find '"response"' key, then locate the value boundary by searching
    backwards from the next known sibling key ('"modifiers"', '"mode"', etc.)
    or from the last '}' if no sibling is found.
    """
    # Known sibling keys that appear after "response" in frontal cortex output
    sibling_keys = ['"modifiers"', '"mode"', '"actions"', '"confidence"',
                    '"alternative_paths"', '"downstream_mode"']

    resp_marker = '"response"'
    idx = text.find(resp_marker)
    if idx == -1:
        return None

    # Find the colon and opening quote after "response"
    colon_idx = text.find(':', idx + len(resp_marker))
    if colon_idx == -1:
        return None
    open_quote = text.find('"', colon_idx + 1)
    if open_quote == -1:
        return None

    # Find the earliest sibling key after the opening quote
    value_start = open_quote + 1
    end_boundary = len(text)
    for key in sibling_keys:
        pos = text.find(key, value_start)
        if pos != -1 and pos < end_boundary:
            end_boundary = pos

    # Walk backwards from boundary to find the closing pattern: ", or "}
    segment = text[value_start:end_boundary]
    # Strip trailing whitespace, comma, and quote
    segment = segment.rstrip()
    if segment.endswith(','):
        segment = segment[:-1].rstrip()
    if segment.endswith('"'):
        segment = segment[:-1]

    return segment if segment else None


class FrontalCortexService:
    """Service for generating contextual responses using LLM."""

    def __init__(self, config: dict):
        """Initialize the frontal cortex service with LLM and world state components.

        Args:
            config: Provider configuration dict passed to
                :func:`~services.llm_service.create_llm_service`.  Must include
                at least a ``platform`` key.
        """
        from services.llm_service import create_llm_service
        from services.world_state_service import WorldStateService

        self.config = config
        self.llm = create_llm_service(config)
        self.world_state_service = WorldStateService()

    def get_context_limit(self) -> int:
        """Delegate to underlying LLM service."""
        return self.llm.get_context_limit()

    def count_tokens(self, messages: list, system_prompt: str = '', tools: list = None) -> int:
        """Delegate to underlying LLM service."""
        return self.llm.count_tokens(messages, system_prompt, tools)

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
        """
        Generate a response based on the prompt, classification, and history.

        Args:
            system_prompt_template: Template with {{variable}} placeholders
            original_prompt: The user's original message
            classification: Dict containing topic, confidence, etc.
            chat_history: List of previous exchanges
            act_history: Formatted ACT loop history (defaults to empty)

        Returns:
            dict: {
                'mode': str,
                'modifiers': list,
                'response': str,
                'generation_time': float,
                'actions': list|None
            }
        """
        start_time = time.time()

        # Inject parameters into template
        system_prompt = self._inject_parameters(
            system_prompt_template,
            original_prompt,
            classification,
            chat_history,
            act_history,
            assembled_context=assembled_context,
            relevant_tools=relevant_tools,
            selected_tools=selected_tools,
            selected_skills=selected_skills,
            thread_id=thread_id,
            returning_from_silence=returning_from_silence,
            inclusion_map=inclusion_map,
        )

        # Generate response from LLM
        try:
            response_text = self.llm.send_message(system_prompt, original_prompt).text
        except Exception as e:
            # Re-raise with context for upstream handling
            raise Exception(f"LLM generation failed: {str(e)}") from e

        generation_time = time.time() - start_time

        # Diagnostic: log raw ACT response for debugging tool invocation issues
        if 'act' in system_prompt_template.lower()[:200]:
            # Log full response for short ones, truncated for long ones
            _trunc = response_text if len(response_text) <= 800 else response_text[:800]
            logging.info(
                f"[CORTEX ACT RAW] LLM response ({len(response_text)} chars): "
                f"{_trunc}"
            )

        return self._parse_response_text(response_text, generation_time)

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
        """Build the stable system prompt WITHOUT act_history.

        Called once at the start of the ACT loop in append mode.  The
        act_history is kept out of the system prompt deliberately — it is
        instead passed as part of the growing message array so that
        provider-side prompt caching can treat the system prompt as a stable
        prefix across all loop iterations.

        Args:
            system_prompt_template: Template with {{variable}} placeholders
            original_prompt: The user's original message
            classification: Dict containing topic, confidence, etc.
            chat_history: List of previous exchanges
            assembled_context: Pre-assembled context dict
            relevant_tools: Tools scored by embedding relevance
            selected_tools: Triage-selected tools
            selected_skills: Triage-selected innate skills
            thread_id: Conversation thread identifier
            returning_from_silence: Whether the user is returning after a gap
            inclusion_map: Context node inclusion decisions

        Returns:
            str: Fully injected system prompt (act_history left empty)
        """
        return self._inject_parameters(
            system_prompt_template,
            original_prompt,
            classification,
            chat_history,
            act_history="",  # act_history goes in the message array, not the system prompt
            assembled_context=assembled_context,
            relevant_tools=relevant_tools,
            selected_tools=selected_tools,
            selected_skills=selected_skills,
            thread_id=thread_id,
            returning_from_silence=returning_from_silence,
            inclusion_map=inclusion_map,
        )

    def generate_response_appended(
        self,
        system_prompt: str,
        messages: list,
        cache_prefix: bool = False,
        tools: list = None,
    ) -> dict:
        """Multi-turn LLM call using a pre-built system prompt.

        Used by the ACT loop in append mode: the system prompt is built once
        at loop start via :meth:`build_system_prompt` and the act_history
        grows as a message array across iterations, enabling provider-level
        prompt caching on the stable system prefix.

        Args:
            system_prompt: Pre-built system prompt (no act_history placeholder)
            messages: Growing list of {"role": ..., "content": ...} dicts
            cache_prefix: When True, hint to the provider to cache the system
                prompt prefix (Anthropic / OpenAI prompt-caching APIs)
            tools: List of native tool schema dicts. When provided, the LLM
                uses native tool calling instead of JSON text output.

        Returns:
            Same dict structure as :meth:`generate_response`, plus
            ``raw_response`` containing the unmodified LLM text (needed by
            the orchestrator to append the assistant turn to the message array).
            When native tool calling is used, ``tool_calls`` contains the
            structured tool invocations and ``actions`` is derived from them.
        """
        start_time = time.time()

        try:
            llm_response = self.llm.send_messages(
                system_prompt, messages, cache_prefix, tools=tools,
            )
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}") from e

        generation_time = time.time() - start_time

        # Native tool calling path — structured response, no parsing needed
        if llm_response.tool_calls:
            actions = [
                {
                    'type': tc['name'],
                    'tool_call_id': tc['id'],
                    **tc['input'],
                }
                for tc in llm_response.tool_calls
            ]
            return {
                'mode': 'ACT',
                'modifiers': [],
                'response': '',
                'generation_time': generation_time,
                'actions': actions,
                'confidence': 0.9,
                'alternative_paths': [],
                'downstream_mode': 'UNIFIED',
                'narrated': False,
                'narration': llm_response.text or '',
                'raw_response': llm_response.text or '',
                'tool_calls': llm_response.tool_calls,
                'stop_reason': llm_response.stop_reason,
            }

        # Text-only response — either direct reply or legacy JSON parsing
        if tools:
            # Model chose to respond with text instead of calling tools
            return {
                'mode': 'UNIFIED',
                'modifiers': [],
                'response': llm_response.text,
                'generation_time': generation_time,
                'actions': None,
                'confidence': 0.8,
                'alternative_paths': [],
                'downstream_mode': 'UNIFIED',
                'narrated': False,
                'narration': '',
                'raw_response': llm_response.text,
                'tool_calls': None,
                'stop_reason': llm_response.stop_reason,
            }

        # Legacy path (no tools parameter) — parse JSON from text
        result = self._parse_response_text(llm_response.text, generation_time)
        result['raw_response'] = llm_response.text
        return result

    def _parse_response_text(self, response_text: str, generation_time: float) -> dict:
        """Parse a raw LLM response string into the standard cortex result dict.

        Shared by both :meth:`generate_response` (single-turn) and
        :meth:`generate_response_appended` (multi-turn append mode) so that
        all JSON recovery logic lives in exactly one place.

        Args:
            response_text: Raw text returned by the LLM provider
            generation_time: Wall-clock seconds for the LLM call

        Returns:
            dict: {mode, modifiers, response, generation_time, actions,
                   confidence, alternative_paths, downstream_mode} plus
                   optional narrated/narration pass-through keys.

        Raises:
            Exception: When JSON parsing fails after all recovery layers.
        """
        # Parse JSON response (format: "json" is set in config)
        # Strip markdown code fences if the model wrapped the JSON
        try:
            stripped = response_text.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("\n", 1)[-1]  # remove opening fence line
                stripped = stripped.rsplit("```", 1)[0]  # remove closing fence

            # Multi-layer JSON parse with progressive recovery.
            # Layer 0: direct parse
            # Layer 0b: multi-object JSON (GPT-5.4 returns {…}\n{…})
            # Layer 1: fix invalid escape sequences (\$ etc.)
            # Layer 1b: multi-object on escape-fixed text
            # Layer 2: extract {…} from prose wrapper
            # Layer 3: fix literal newlines in string values
            # Layer 4: extract "response" from broken JSON
            # Layer 5: wrap prose as UNIFIED
            response_data = None

            # Helper: try raw_decode to extract just the first JSON object
            def _try_raw_decode(text):
                try:
                    obj, _end = json.JSONDecoder().raw_decode(text)
                    logging.warning(
                        f"[FRONTAL CORTEX] Parsed first JSON object from multi-object "
                        f"response (used {_end}/{len(text)} chars)"
                    )
                    return obj
                except (json.JSONDecodeError, ValueError):
                    return None

            # Layer 0: direct parse
            try:
                response_data = json.loads(stripped)
            except json.JSONDecodeError as _first_err:
                # Layer 0b: multi-object JSON ("Extra data" = valid first object + trailing data)
                if 'Extra data' in str(_first_err):
                    response_data = _try_raw_decode(stripped)

            # Layer 1: fix invalid escape sequences
            if response_data is None:
                fixed = re.sub(r'\\([^"\\/bfnrtu])', r'\1', stripped)
                try:
                    response_data = json.loads(fixed)
                except json.JSONDecodeError as _fixed_err:
                    # Layer 1b: multi-object on escape-fixed text
                    if 'Extra data' in str(_fixed_err):
                        response_data = _try_raw_decode(fixed)

            # Layer 2: extract {…} from prose wrapper
            if response_data is None:
                fixed = re.sub(r'\\([^"\\/bfnrtu])', r'\1', stripped)
                brace_start = fixed.find('{')
                brace_end = fixed.rfind('}')
                if brace_start != -1 and brace_end > brace_start:
                    candidate = fixed[brace_start:brace_end + 1]
                    try:
                        response_data = json.loads(candidate)
                        logging.warning(
                            "[FRONTAL CORTEX] Extracted JSON from prose response "
                            f"(offset {brace_start}–{brace_end})"
                        )
                    except json.JSONDecodeError:
                        pass

            # Layer 3: fix literal newlines inside JSON string values
            if response_data is None:
                try:
                    src = candidate if (brace_start != -1 and brace_end > brace_start) else fixed
                    buf, in_str, esc = [], False, False
                    for ch in src:
                        if esc:
                            buf.append(ch); esc = False
                        elif ch == '\\':
                            buf.append(ch); esc = True
                        elif ch == '"':
                            in_str = not in_str; buf.append(ch)
                        elif in_str and ch == '\n':
                            buf.append('\\n')
                        elif in_str and ch == '\r':
                            buf.append('\\r')
                        elif in_str and ch == '\t':
                            buf.append('\\t')
                        else:
                            buf.append(ch)
                    response_data = json.loads(''.join(buf))
                    logging.warning(
                        "[FRONTAL CORTEX] Fixed literal newlines in JSON string values"
                    )
                except (json.JSONDecodeError, Exception):
                    pass

            # Layer 4: extract "response" from broken JSON (unescaped inner quotes)
            if response_data is None:
                prose = stripped.strip() or response_text.strip()
                if prose and prose.lstrip().startswith('{'):
                    extracted = _extract_response_from_broken_json(prose)
                    if extracted:
                        logging.warning(
                            "[FRONTAL CORTEX] Extracted response from broken JSON "
                            f"(first 80 chars): {extracted[:80]!r}"
                        )
                        response_data = {"response": extracted, "modifiers": []}

            # Layer 5: pure prose — wrap as UNIFIED
            if response_data is None:
                prose = stripped.strip() or response_text.strip()
                if prose:
                    logging.warning(
                        "[FRONTAL CORTEX] LLM returned prose instead of JSON — "
                        "wrapping as UNIFIED (first 80 chars): "
                        f"{prose[:80]!r}"
                    )
                    response_data = {"response": prose, "modifiers": []}
                else:
                    raise Exception(f"Empty LLM response — no JSON to parse")

            # Guard: response_data must be a dict — LLM may return a bare JSON
            # array (e.g. [{...}]) which passes json.loads but has no .get()
            if not isinstance(response_data, dict):
                if isinstance(response_data, list):
                    dict_items = [x for x in response_data if isinstance(x, dict)]
                    if dict_items:
                        logging.warning(
                            "[FRONTAL CORTEX] LLM returned JSON array — using first dict element"
                        )
                        response_data = dict_items[0]
                    else:
                        response_data = {"response": str(response_data), "modifiers": []}
                else:
                    response_data = {"response": str(response_data) if response_data else "", "modifiers": []}

            # Extract fields (mode-specific prompts produce simpler output)
            mode = response_data.get('mode', 'UNIFIED')
            modifiers = response_data.get('modifiers', [])
            user_response = response_data.get('response', '')
            # Guard: ensure response is always a string (LLM may return nested object)
            if not isinstance(user_response, str):
                user_response = str(user_response)
            actions = response_data.get('actions')
            confidence = response_data.get('confidence', 0.5)
            alternative_paths = response_data.get('alternative_paths', [])

            # Infer ACT mode if actions are present but mode wasn't explicit
            # (ACT prompt output contract returns actions without a mode field)
            if actions and isinstance(actions, list) and len(actions) > 0 and mode != 'ACT':
                mode = 'ACT'

            # Validate mode
            valid_modes = ['ACT', 'UNIFIED', 'IGNORE']
            if mode not in valid_modes:
                logging.warning(f"Invalid mode '{mode}', defaulting to UNIFIED")
                mode = 'UNIFIED'

            # Validate confidence range
            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.5

            # Validate alternative paths structure (needed by act_loop decision gate)
            validated_alternatives = []
            for i, path in enumerate(alternative_paths):
                if not isinstance(path, dict) or 'mode' not in path:
                    continue
                if 'expected_confidence' not in path:
                    path['expected_confidence'] = 0.5
                try:
                    path['expected_confidence'] = float(path['expected_confidence'])
                    path['expected_confidence'] = max(0.0, min(1.0, path['expected_confidence']))
                except (ValueError, TypeError):
                    path['expected_confidence'] = 0.5
                # Validate downstream_mode for ACT paths
                if mode == 'ACT' and path.get('mode') == 'ACT':
                    valid_terminal = ['UNIFIED', 'IGNORE']
                    if path.get('downstream_mode') not in valid_terminal:
                        path['downstream_mode'] = 'UNIFIED'
                validated_alternatives.append(path)
            alternative_paths = validated_alternatives

            # Validate actions for ACT mode
            if mode == 'ACT' and actions and isinstance(actions, list):
                actions = [a for a in actions if isinstance(a, dict) and 'type' in a]
                if not actions:
                    actions = None
            elif mode != 'ACT':
                actions = None

            # Normalize empty actions to None
            if not actions:
                actions = None

            logging.info(f"[MODE:{mode}] Cortex response: mode={mode}, confidence={confidence:.2f}, "
                        f"actions={len(actions) if actions else 0}, alternatives={len(alternative_paths)}")

        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON response: {str(e)}\nRaw response: {response_text[:200]}") from e

        result = {
            'mode': mode,
            'modifiers': modifiers,
            'response': user_response,
            'generation_time': generation_time,
            'actions': actions,
            'confidence': confidence,
            'alternative_paths': alternative_paths,
            'downstream_mode': response_data.get('downstream_mode', 'UNIFIED'),
        }

        # Pass through ACT narration fields (used by ACTOrchestrator for live progress)
        if 'narrated' in response_data:
            result['narrated'] = response_data['narrated']
        if 'narration' in response_data:
            result['narration'] = response_data['narration']

        return result

    def _inject_parameters(
        self,
        template: str,
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
    ) -> str:
        """
        Replace {{variable}} placeholders in template with actual values.

        Args:
            template: System prompt template
            original_prompt: User's message
            classification: Classification result
            chat_history: Previous exchanges (unused - kept for compatibility)
            act_history: ACT loop history (defaults to empty)
            assembled_context: Pre-assembled context dict from ContextAssemblyService
            inclusion_map: Context node inclusion decisions from ContextRelevanceService
                If None, includes all nodes (backward compatible)

        Returns:
            str: Processed system prompt
        """
        # Determine the topic (classifier may return similar_topic, topic_update, or topic)
        topic = classification.get('similar_topic') or \
                classification.get('topic_update') or \
                classification.get('topic', 'unknown')

        confidence = classification.get('confidence', 0)

        # Helper to check if a node should be included
        _include = lambda node: (inclusion_map or {}).get(node, True)

        _ctx = assembled_context or {}
        episodic_context = _ctx.get('episodes', '') if _include('episodic_memory') else ''
        working_memory_context = _ctx.get('working_memory', '') if _include('working_memory') else ''
        concepts_context = _ctx.get('concepts', '') if _include('concepts') else ''
        _msg_emb = _ctx.get('message_embedding') if _ctx else None
        world_state = (
            self.world_state_service.get_world_state(
                topic, thread_id=thread_id, message_embedding=_msg_emb
            )
            if _include('world_state')
            else ''
        )

        # Replace placeholders
        try:
            from services.time_utils import utc_now
            _now = utc_now()
            _current_datetime = _now.strftime('%A, %Y-%m-%d %H:%M UTC')
            _current_date = _now.strftime('%A, %Y-%m-%d')
        except Exception:
            from datetime import datetime, timezone
            _now = datetime.now(timezone.utc)
            _current_datetime = _now.strftime('%A, %Y-%m-%d %H:%M UTC')
            _current_date = _now.strftime('%A, %Y-%m-%d')
        result = template.replace('{{current_datetime}}', _current_datetime)
        result = result.replace('{{current_date}}', _current_date)

        # Consolidated user state (identity + telemetry) — always injected
        user_state = self._get_user_state()
        result = result.replace('{{user_state}}', user_state)
        result = result.replace('{{original_prompt}}', original_prompt)
        result = result.replace('{{topic}}', str(topic))
        result = result.replace('{{confidence}}', str(confidence))
        result = result.replace('{{world_state}}', world_state if _include('world_state') else '')
        result = result.replace('{{episodic_memory}}', episodic_context if _include('episodic_memory') else '')

        # Situation directive — plain-language behavioral guidance derived from the
        # situation model.  Returns None when the situation is unremarkable; the
        # placeholder is then replaced with empty string so zero tokens are added.
        situation_directive = ''
        if '{{situation}}' in result:
            try:
                from services.situation_model_service import get_situation_model_service
                _directive = get_situation_model_service().get_directive()
                if _directive:
                    situation_directive = f"## Situation\n{_directive}"
            except Exception:
                pass
            result = result.replace('{{situation}}', situation_directive)
        result = result.replace('{{semantic_concepts}}', concepts_context)
        result = result.replace('{{act_history}}', act_history)
        result = result.replace('{{working_memory}}', working_memory_context if _include('working_memory') else '')

        goal_context = ''
        if _include('active_goals'):
            goal_context = self._get_goal_context()
        result = result.replace('{{active_goals}}', goal_context)

        # Phase 3 — Response weaving: inject contradiction context when flagged
        contradiction_ctx = _ctx.get('contradiction_context')
        if contradiction_ctx and '{{contradiction_context}}' in result:
            mem_a = contradiction_ctx.get('memory_a_text', '')
            mem_b = contradiction_ctx.get('memory_b_text', '')
            classification = contradiction_ctx.get('classification', '')
            reasoning = contradiction_ctx.get('reasoning', '')
            if mem_a and mem_b:
                hint = (
                    f"\n\n## Memory Conflict Detected\n"
                    f"The current message may contradict an existing memory:\n"
                    f"- Existing: {mem_b}\n"
                    f"- Current context suggests: {mem_a}\n"
                    f"- Classification: {classification}\n"
                    f"- Reasoning: {reasoning}\n\n"
                    f"If relevant to the response, weave this naturally into your reply "
                    f"(e.g. noting a change, gently asking for clarification). "
                    f"Do NOT present it as a system alert. Omit entirely if not relevant."
                )
            else:
                hint = ''
            result = result.replace('{{contradiction_context}}', hint)
        else:
            result = result.replace('{{contradiction_context}}', '')

        # Inject visual context from attached images
        visual_context_raw = _ctx.get('visual_context', '')
        if visual_context_raw:
            visual_block = f"\n\n## Visual Context (attached images)\n{visual_context_raw}"
        else:
            visual_block = ''
        result = result.replace('{{visual_context}}', visual_block)

        # Template integrity guard — warn when skills were selected but template has no placeholder
        if selected_skills and '{{injected_skills}}' not in template:
            logging.error("[FRONTAL CORTEX] ACT template missing {{injected_skills}} placeholder — skill docs will not be injected")

        # Inject skill docs for the provided list.
        # When native tool calling is active (append mode with all skills), inject
        # a minimal note instead of full docs — the tool definitions are in the
        # `tools` API parameter and shouldn't be duplicated in the prompt text.
        _use_native_tools = self.config.get('append_mode', False) and selected_skills
        if _use_native_tools and len(selected_skills or []) > 4:
            # All skills available as native tools — don't repeat docs in prompt
            injected_skills = (
                "All cognitive skills are available as callable tools. "
                "Call them directly by name — their descriptions and parameters "
                "are provided via the tools interface."
            )
        else:
            injected_skills = self._get_injected_skills(selected_skills or [])
        result = result.replace('{{injected_skills}}', injected_skills)

        # Inject registered tool name index (lightweight, no docs)
        if '{{registered_tool_names}}' in result:
            tool_names = self._get_registered_tool_names()
            result = result.replace('{{registered_tool_names}}', tool_names)

        # Strategy hints from procedural memory (learned action reliability)
        strategy_hints = ''
        if _include('strategy_hints'):
            strategy_hints = self._get_strategy_hints(topic)
        result = result.replace('{{strategy_hints}}', strategy_hints)

        # Constraint context — gate rejection patterns visible to LLM
        constraint_context = ''
        if _include('constraint_context') and '{{constraint_context}}' in result:
            try:
                from services.constraint_memory_service import ConstraintMemoryService
                cms = ConstraintMemoryService()
                mode_name = classification.get('mode', 'respond').lower()
                constraint_context = cms.format_for_prompt(mode=mode_name)
            except Exception:
                pass
        result = result.replace('{{constraint_context}}', constraint_context)

        # Identity modulation (voice mapper)
        if _include('identity_modulation'):
            identity_modulation = self._get_identity_modulation()
        else:
            identity_modulation = ''
        result = result.replace('{{identity_modulation}}', identity_modulation)


        # Onboarding nudge — elicit missing identity traits progressively
        if _include('onboarding_nudge'):
            onboarding_nudge = self._get_onboarding_nudge(thread_id, classification)
        else:
            onboarding_nudge = ''
        result = result.replace('{{onboarding_nudge}}', onboarding_nudge)


        # User traits (known facts about the user)
        # Lower injection threshold when returning from silence to surface more context
        if _include('user_traits'):
            user_traits = self._get_user_traits(
                original_prompt, topic,
                injection_threshold_override=0.2 if returning_from_silence else None,
            )
        else:
            user_traits = ''
        result = result.replace('{{user_traits}}', user_traits)

        # Adaptive response directives (style-driven behavioral hints)
        if _include('adaptive_directives'):
            adaptive_directives = self._get_adaptive_directives(
                original_prompt=original_prompt,
                thread_id=thread_id,
            )
        else:
            adaptive_directives = ''
        result = result.replace('{{adaptive_directives}}', adaptive_directives)


        # Focus session (current declared or inferred focus)
        if _include('focus'):
            focus_context = self._get_focus_context(thread_id)
        else:
            focus_context = ''
        result = result.replace('{{focus}}', focus_context)


        # temporal_rhythm — removed from unified prompt; behavioral traits surface
        # via salient traits selector instead. Keep no-op for old templates.
        result = result.replace('{{temporal_rhythm}}', '')

        # Self-awareness (interoception — only injected when noteworthy)
        if _include('self_awareness'):
            self_awareness = self._get_self_awareness()
        else:
            self_awareness = ''
        result = result.replace('{{self_awareness}}', self_awareness)

        # Legacy placeholders — still present in old UNIFIED/ACT templates used by background workers
        for legacy in ('{{identity_context}}', '{{client_context}}', '{{available_skills}}',
                       '{{available_tools}}', '{{active_goals}}', '{{active_lists}}', '{{warm_return_hint}}'):
            result = result.replace(legacy, '')

        return result

    def _get_identity_modulation(self) -> str:
        """Retrieve identity modulation text from the voice mapper for prompt injection.

        Returns:
            Formatted modulation string from
            :meth:`~services.voice_mapper_service.VoiceMapperService.generate_modulation`,
            or a safe fallback string on error.
        """
        try:
            from services.identity_service import IdentityService
            from services.voice_mapper_service import VoiceMapperService
            from services.database_service import get_shared_db_service

            db_service = get_shared_db_service()
            identity = IdentityService(db_service)
            mapper = VoiceMapperService()

            vectors = identity.get_vectors()
            identity.check_coherence()
            modulation = mapper.generate_modulation(vectors)
            return modulation if modulation else "Engage naturally as a peer."
        except Exception as e:
            logging.warning(f"Identity modulation unavailable: {e}")
            return "Engage naturally as a peer."

    def _get_user_traits(
        self,
        prompt: str,
        topic: str,
        injection_threshold_override: Optional[float] = None,
    ) -> str:
        """
        Get user traits formatted for prompt injection.

        Retrieves core traits (always) and contextually relevant traits
        (by embedding similarity), capped at 6 total.

        Args:
            prompt: Current user message (for contextual retrieval)
            topic: Current topic
            injection_threshold_override: When set, overrides the default
                INJECTION_THRESHOLD (e.g., 0.2 when returning from silence).

        Returns:
            str: Formatted user traits section or empty string
        """
        try:
            from services.user_trait_service import UserTraitService, INJECTION_THRESHOLD
            from services.database_service import get_shared_db_service

            db_service = get_shared_db_service()
            trait_service = UserTraitService(db_service)
            effective_threshold = (
                injection_threshold_override
                if injection_threshold_override is not None
                else INJECTION_THRESHOLD
            )
            return trait_service.get_traits_for_prompt(
                prompt, injection_threshold=effective_threshold
            )
        except ImportError:
            return ""
        except Exception as e:
            logging.debug(f"User traits not available: {e}")
            return ""

    def _get_user_state(self) -> str:
        """
        Consolidated telemetry: time, location, device, energy, attention, mobility.

        Pure telemetry — no identity data. Always injected, no gating. ~25 tokens.
        """
        try:
            return self._build_user_state_line()
        except Exception as e:
            logging.warning(f"[FRONTAL CORTEX] User state failed: {e}")
            return ''

    def _build_user_state_line(self) -> str:
        """Build the user state line — pure telemetry, no identity. Raises on failure."""
        from services.client_context_service import ClientContextService
        from services.ambient_inference_service import AmbientInferenceService
        from services.place_learning_service import PlaceLearningService
        from services.database_service import DatabaseService
        from zoneinfo import ZoneInfo
        from datetime import datetime

        parts = []

        ctx = ClientContextService().get()
        if not ctx:
            return ''

        # Local time
        if timezone := ctx.get('timezone'):
            user_dt = datetime.now(ZoneInfo(timezone))
            day_str = user_dt.strftime('%a %d %b')
            time_str = user_dt.strftime('%H:%M')
            tz_abbr = user_dt.strftime('%Z') or timezone.split('/')[-1]
            parts.append(f"{day_str}, {time_str} {tz_abbr}")

        # Location
        location_name = ctx.get('location_name', '')

        # Device + battery
        device = ctx.get('device', {})
        if device_class := device.get('class'):
            battery = ctx.get('battery', {})
            battery_level = battery.get('level') if isinstance(battery, dict) else None
            if battery_level is not None:
                parts.append(f"{device_class} {int(battery_level)}%")
            else:
                parts.append(device_class)

        # Ambient inferences
        db = DatabaseService()
        place_learning = PlaceLearningService(db)
        inferences = AmbientInferenceService(
            place_learning_service=place_learning
        ).infer(ctx)

        place = inferences.get('place')
        if location_name and place:
            parts.append(f"{location_name} ({place})")
        elif location_name:
            parts.append(location_name)
        elif place:
            parts.append(place)

        if energy := inferences.get('energy'):
            parts.append(f"Energy: {energy}")

        attention_labels = {
            'deep_focus': 'Focus: deep',
            'focused': 'Focus: active',
            'casual': 'Casual',
            'distracted': 'Distracted',
            'away': 'Away',
        }
        if attention := inferences.get('attention'):
            if label := attention_labels.get(attention):
                parts.append(label)

        if mobility := inferences.get('mobility'):
            if mobility != 'stationary':
                parts.append(mobility.capitalize())

        return ' \u00b7 '.join(parts) if parts else ''

    def _get_onboarding_nudge(
        self,
        thread_id: str,
        classification: dict = None,
    ) -> str:
        """
        Return a behavioural nudge for missing identity traits, or '' if none due.

        Checks:
        - Exchange count vs min_turn threshold
        - Whether the trait already exists in IdentityStateService
        - Nudge cooldown and max_attempts from _ONBOARDING_SCHEDULE

        Skips when classification signals high urgency or tool use.
        Updates onboarding state in IdentityStateService on nudge emission.

        Never raises.
        """
        if not thread_id:
            return ""
        try:
            # Don't nudge during urgent or tool-heavy exchanges
            if classification:
                if classification.get('needs_tools') or classification.get('urgency') == 'high':
                    return ""

            from services.identity_state_service import IdentityStateService
            from services.memory_client import MemoryClientService
            from services.database_service import get_shared_db_service
            from services.user_trait_service import UserTraitService

            # Read current exchange count from thread hash
            r = MemoryClientService.create_connection()
            exchange_count = int(r.hget(f"thread:{thread_id}", "exchange_count") or 0)

            identity_svc = IdentityStateService()
            identity_state = identity_svc.get_all()
            onboarding_state = identity_state.get('_onboarding', {})

            # Build set of trait keys already known in permanent storage
            # (user_traits survives MemoryStore TTL expiry)
            try:
                db = get_shared_db_service()
                trait_svc = UserTraitService(db)
                known_trait_keys = {
                    t['trait_key'] for t in trait_svc.get_all_traits()
                    if t.get('confidence', 0) >= 0.5
                }
            except Exception:
                known_trait_keys = set()

            for entry in _ONBOARDING_SCHEDULE:
                trait = entry['trait']

                # Already have this trait in MemoryStore — skip
                trait_data = identity_state.get(trait)
                if trait_data and trait_data.get('value'):
                    continue

                # Already have this trait in permanent storage — skip
                if trait in known_trait_keys:
                    continue

                # Too early in the conversation
                if exchange_count < entry['min_turn']:
                    continue

                # Check nudge history for this trait
                nudge_info = onboarding_state.get(trait, {})
                attempts = nudge_info.get('attempts', 0)
                last_nudge_turn = nudge_info.get('nudged_at_turn', 0)

                # Backed off — user declined implicitly
                if attempts >= entry['max_attempts']:
                    continue

                # Cooldown — too recent
                if attempts > 0 and (exchange_count - last_nudge_turn) < entry['cooldown_turns']:
                    continue

                # Emit nudge and record it
                nudge_info['nudged_at_turn'] = exchange_count
                nudge_info['attempts'] = attempts + 1
                onboarding_state[trait] = nudge_info
                identity_svc.set_onboarding_state(onboarding_state)

                logging.debug(
                    f"[FRONTAL CORTEX] Onboarding nudge: trait='{trait}' "
                    f"attempt={nudge_info['attempts']} at turn {exchange_count}"
                )
                return f"\n**Onboarding note:** {entry['hint']}\n"

            return ""
        except Exception as e:
            logging.debug(f"[FRONTAL CORTEX] Onboarding nudge unavailable: {e}")
            return ""

    def _get_focus_context(self, thread_id: str = None) -> str:
        """
        Get current focus session formatted for prompt injection.

        Args:
            thread_id: Thread identifier

        Returns:
            str: Formatted focus section or empty string
        """
        if not thread_id:
            return ""
        try:
            from services.focus_session_service import FocusSessionService
            return FocusSessionService().get_focus_for_prompt(thread_id)
        except Exception as e:
            logging.debug(f"Focus context not available: {e}")
            return ""

    def _get_adaptive_directives(self, original_prompt: str = "", thread_id: str = None) -> str:
        """
        Get adaptive response directives based on detected user interaction style.

        Returns:
            str: Formatted directive block or empty string
        """
        try:
            from services.adaptive_layer_service import AdaptiveLayerService
            from services.database_service import get_shared_db_service
            from services.working_memory_service import WorkingMemoryService

            db_service = get_shared_db_service()
            service = AdaptiveLayerService(db_service)

            # Get raw working memory turns for cognitive load estimation
            working_memory_turns = []
            if thread_id:
                try:
                    wm_service = WorkingMemoryService()
                    working_memory_turns = wm_service.get_recent_turns(thread_id) or []
                except Exception:
                    pass

            # Build current signals — start with prompt length, augment from MemoryStore snapshot
            current_signals = {
                'prompt_token_count': len(original_prompt.split()) if original_prompt else 0,
            }
            if thread_id:
                try:
                    from services.memory_client import MemoryClientService
                    import json as _json
                    _store = MemoryClientService.create_connection()
                    _snapshot_raw = _store.get(f"adaptive_signals:{thread_id}")
                    if _snapshot_raw:
                        _snapshot = _json.loads(_snapshot_raw)
                        current_signals.update(_snapshot)
                except Exception:
                    pass

            return service.generate_directives(
                thread_id=thread_id,
                working_memory_turns=working_memory_turns,
                current_signals=current_signals,
                current_message=original_prompt,
            )
        except Exception as e:
            logging.debug(f"Adaptive directives not available: {e}")
            return ""


    def _get_temporal_rhythm(self) -> str:
        """Get rhythm context from temporal pattern mining.

        Returns max 3 most salient lines, sanitized, total < 200 chars.
        Returns empty string if no patterns available.
        """
        try:
            from services.temporal_pattern_service import TemporalPatternService
            from services.database_service import get_shared_db_service

            db = get_shared_db_service()
            service = TemporalPatternService(db)
            summary = service.get_rhythm_summary()
            return summary if summary else ''
        except Exception as e:
            logging.debug(f"Temporal rhythm not available: {e}")
            return ''

    def _get_self_awareness(self) -> str:
        """Get self-awareness context from self-model (only when noteworthy).

        Returns empty string when all systems are healthy (zero token cost).
        When degraded, includes signals AND behavioral directives.
        """
        try:
            from services.self_model_service import SelfModelService
            return SelfModelService().format_for_prompt()
        except Exception as e:
            logging.debug(f"Self-awareness not available: {e}")
            return ""

    def get_skill_docs(self, skills: list) -> str:
        """Public API for loading skill documentation by name.

        Used by the orchestrator for mid-loop skill escalation.
        """
        return self._get_injected_skills(skills)

    def _get_injected_skills(self, skills: list) -> str:
        """
        Build skill documentation from TOOL_SCHEMA dicts on handler modules.

        Each skill handler module exposes a TOOL_SCHEMA dict with name,
        description, and input_schema. This method formats them into a
        compact prompt-ready block.

        Args:
            skills: List of innate skill names (e.g. ['recall', 'memorize', 'schedule'])

        Returns:
            str: Formatted skill docs, or '' if empty.
        """
        if not skills:
            return ''
        from services.innate_skills import get_skill_handler
        import importlib, inspect
        parts = []
        for skill in skills:
            handler = get_skill_handler(skill)
            if not handler:
                logging.warning(f"[FRONTAL CORTEX] No handler for skill '{skill}'")
                continue
            module = inspect.getmodule(handler)
            schema = getattr(module, 'TOOL_SCHEMA', None)
            if not schema:
                logging.warning(f"[FRONTAL CORTEX] No TOOL_SCHEMA for skill '{skill}'")
                continue
            parts.append(self._format_skill_doc(schema))
        return '\n\n'.join(parts)

    @staticmethod
    def _format_skill_doc(schema: dict) -> str:
        """Format a TOOL_SCHEMA dict into a compact prompt-ready block."""
        import json
        name = schema.get('name', '?')
        desc = schema.get('description', '')
        input_schema = schema.get('input_schema', {})
        props = input_schema.get('properties', {})
        required = input_schema.get('required', [])

        lines = [f"### {name}", desc, "", "Parameters:"]
        for pname, pdef in props.items():
            req_marker = " (required)" if pname in required else ""
            ptype = pdef.get('type', 'any')
            pdesc = pdef.get('description', '')
            enum = pdef.get('enum')
            enum_str = f" [{', '.join(str(e) for e in enum)}]" if enum else ""
            lines.append(f"- `{pname}` ({ptype}{enum_str}){req_marker}: {pdesc}")
        return '\n'.join(lines)

    def _get_registered_tool_names(self) -> str:
        """Return descriptors of registered external tools from the database."""
        try:
            from services.database_service import DatabaseService
            db = DatabaseService()
            rows = db.fetch_all(
                "SELECT tool_name, descriptor FROM tool_capability_profiles "
                "WHERE tool_type = 'tool' ORDER BY tool_name"
            )
            parts = [r['descriptor'] or r['tool_name'] for r in (rows or [])]
            return ', '.join(parts) if parts else '(none registered)'
        except Exception as e:
            logging.debug(f"[FRONTAL CORTEX] Failed to fetch tool names: {e}")
            return '(none registered)'

    def _get_available_skills(self) -> str:
        """
        Get available innate skills from the dispatcher for prompt injection.

        Returns innate skills list only. External tools are discovered
        dynamically via the find_tools innate skill during ACT execution.

        Returns:
            Formatted available skills string or empty string
        """
        try:
            from services.act_dispatcher_service import ActDispatcherService
            dispatcher = ActDispatcherService()
            innate = ["recall", "memorize", "introspect", "associate", "autobiography", "focus", "list", "schedule", "persistent_task", "read", "goals"]
            available = [s for s in innate if s in dispatcher.handlers]
            if available:
                return "Available skills: " + ", ".join(available)
            return ""
        except Exception as e:
            logging.debug(f"Skill registry not available: {e}")
            return ""

    def _get_performance_hint(self, tool_name: str) -> str:
        """Build a compact performance hint string for a tool in the ACT prompt.

        Args:
            tool_name: Registered tool name to look up statistics for.

        Returns:
            Single-line performance annotation string (e.g.
            ``'[perf: reliable • 92% success • 430ms • 15 uses]'``), or
            empty string when fewer than 3 uses have been recorded.
        """
        try:
            from services.tool_performance_service import ToolPerformanceService
            stats = ToolPerformanceService().get_tool_stats(tool_name)
            if stats.get('total', 0) < 3:
                return ''
            success_pct = int(stats['success_rate'] * 100)
            avg_ms = int(stats['avg_latency'])
            label = 'reliable' if success_pct >= 80 else 'moderate' if success_pct >= 50 else 'unreliable'
            return f"[perf: {label} • {success_pct}% success • {avg_ms}ms • {stats['total']} uses]"
        except Exception:
            return ''

    def _get_strategy_hints(self, topic: str) -> str:
        """Build compact strategy hints from procedural memory for the ACT prompt.

        Args:
            topic: Current conversation topic (currently unused but reserved for
                future topic-scoped strategy filtering).

        Returns:
            Formatted ``## Strategy Hints`` section string listing up to 8
            action-reliability signals, or empty string when no data is available.
        """
        try:
            from services.procedural_memory_service import ProceduralMemoryService
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            proc = ProceduralMemoryService(db)

            hints = []  # (extremity_score, hint_text) tuples
            for action_name in proc.get_all_policy_weights():
                stats = proc.get_action_stats(action_name)
                if not stats or stats.get('total_attempts', 0) < 5:
                    continue
                attempts = stats['total_attempts']
                successes = stats.get('total_successes', 0)
                reliability = (successes + 1) / (attempts + 2)
                extremity = abs(reliability - 0.5)
                if reliability < 0.4:
                    hints.append((extremity, f"{action_name}: low reliability ({int(reliability*100)}% over {attempts} uses)"))
                elif reliability > 0.85:
                    hints.append((extremity, f"{action_name}: reliable ({int(reliability*100)}% over {attempts} uses)"))

            if not hints:
                return ''
            # Sort by extremity (strongest signals first) for cognitive prioritization
            hints.sort(key=lambda h: h[0], reverse=True)
            lines = [h[1] for h in hints[:8]]
            return "## Strategy Hints (from experience)\n" + "\n".join(f"- {l}" for l in lines)
        except Exception:
            return ''

    def _get_goal_context(self, limit: int = 3) -> str:
        """Compact goal stack for prompt injection (~150 tokens)."""
        try:
            from services.goal_ecology_service import GoalEcologyService
            ecology = GoalEcologyService()
            goals = ecology.get_active_stack(limit=limit)
            if not goals:
                return ''
            lines = ['\n## Active Goals']
            for g in goals:
                sal = f"{g['salience']:.0%}"
                conf = f"{g['confidence']:.0%}"
                status_hint = f" (strategy: {g['strategy'][:60]})" if g.get('strategy') else ''
                lines.append(
                    f"- [{g['type']}] {g['description'][:100]} "
                    f"(salience={sal}, confidence={conf}){status_hint}"
                )
                # Show parent relationship if exists
                if g.get('lineage_parent_id'):
                    try:
                        parent = ecology.get_goal(g['lineage_parent_id'])
                        if parent:
                            lines.append(
                                f"  ^ serves: [{parent['type']}] {parent['description'][:80]}"
                            )
                    except Exception:
                        pass
            return '\n'.join(lines)
        except Exception as e:
            logging.debug(f"[FRONTAL CORTEX] Goal context unavailable: {e}")
            return ''

    def _get_available_tools(self, selected_tools: list = None, relevant_tools: list = None) -> str:
        """
        Get tool profiles for ACT prompt injection.

        When selected_tools is provided (from CognitiveTriageService), injects
        full profiles for those specific tools from tool_capability_profiles table.
        Falls back to manifest-based summaries if profiles unavailable.
        """
        try:
            from services.tool_registry_service import ToolRegistryService
            registry = ToolRegistryService()

            # Prefer triage-selected tools (Wave 2)
            tool_names = None
            if selected_tools:
                tool_names = [t for t in selected_tools if t in registry.tools]
            elif relevant_tools:
                tool_names = [
                    item['name'] for item in relevant_tools
                    if item.get('type') == 'tool' and item['name'] in registry.tools
                ][:5]

            if tool_names:
                # Try to get rich profiles first (inject full_profile for triage-selected tools)
                try:
                    from services.tool_profile_service import ToolProfileService
                    profiles = ToolProfileService().get_profiles_for_tools(tool_names)
                    if profiles:
                        lines = []
                        for p in profiles:
                            name = p.get('tool_name', '')
                            full_profile = p.get('full_profile', '') or p.get('short_summary', '')
                            perf_hint = self._get_performance_hint(name)
                            suffix = f"\n{perf_hint}" if perf_hint else ''
                            lines.append(f"### {name}\n{full_profile}{suffix}")
                        return "\n\n".join(lines)
                except Exception:
                    pass

                # Fallback to manifest-based summaries
                lines = []
                for name in tool_names:
                    manifest = registry.tools[name]['manifest']
                    desc = manifest.get('description', name)
                    params = manifest.get('parameters', {})
                    param_str = f" ({', '.join(list(params.keys()))})" if params else ""
                    perf_hint = self._get_performance_hint(name)
                    suffix = f" {perf_hint}" if perf_hint else ''
                    lines.append(f"- {name}{param_str}: {desc}{suffix}")
                return "\n".join(lines)

            # No selected tools — return all registered tools
            summaries = registry.get_tool_prompt_summaries()
            return summaries if summaries else "(no tools loaded)"
        except Exception as e:
            logging.debug(f"Tool registry not available for prompt: {e}")
            return "(no tools loaded)"


