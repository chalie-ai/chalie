# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Response Generation Service — LLM invocation and raw response parsing.

Handles the runtime LLM calls (single-turn and multi-turn append mode) and
all JSON recovery layers that translate a raw LLM text response into the
standard cortex result dict.

Also hosts :class:`ChatHistoryProcessor`, a lightweight helper that truncates
and serialises chat history for prompt injection.

This module is extracted from :mod:`services.frontal_cortex_service` as part
of the WS3 decomposition.  The :class:`FrontalCortexService` facade delegates
all response-generation calls here.
"""

import re
import time
import json
import logging
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helper
# ─────────────────────────────────────────────────────────────────────────────

def _extract_response_from_broken_json(text: str) -> str | None:
    """Extract the 'response' value from broken JSON where inner quotes are unescaped.

    Strategy: find ``"response"`` key, then locate the value boundary by
    searching backwards from the next known sibling key (``"modifiers"``,
    ``"mode"``, etc.) or from the last ``}`` if no sibling is found.

    Args:
        text: Raw JSON-like string that failed standard parsing.

    Returns:
        The extracted response string, or ``None`` if extraction fails.
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


# ─────────────────────────────────────────────────────────────────────────────
# ChatHistoryProcessor
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# ResponseGenerationService
# ─────────────────────────────────────────────────────────────────────────────

class ResponseGenerationService:
    """Handles LLM invocation and raw response parsing for the frontal cortex.

    Encapsulates the single-turn (:meth:`generate_response`) and multi-turn
    append-mode (:meth:`generate_response_appended`) LLM call paths, plus all
    JSON recovery layers that normalise a raw LLM text response into the
    standard cortex result dict.
    """

    def __init__(self, config: dict):
        """Initialise the service by creating the underlying LLM provider.

        Args:
            config: Provider configuration dict passed to
                :func:`~services.llm_service.create_llm_service`.  Must include
                at least a ``platform`` key.
        """
        from services.llm_service import create_llm_service

        self.config = config
        self.llm = create_llm_service(config)

    def get_context_limit(self) -> int:
        """Return the maximum context window size for the configured LLM provider.

        Returns:
            Integer token count representing the LLM's context limit.
        """
        return self.llm.get_context_limit()

    def count_tokens(self, messages: list, system_prompt: str = '', tools: list = None) -> int:
        """Count the tokens for the given messages and system prompt.

        Delegates directly to the underlying LLM provider's token counter.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` message dicts.
            system_prompt: System prompt string (may be empty).
            tools: Optional list of native tool schema dicts.

        Returns:
            Estimated token count as an integer.
        """
        return self.llm.count_tokens(messages, system_prompt, tools)

    def generate_response(
        self,
        system_prompt: str,
        original_prompt: str,
        system_prompt_template: str = "",
    ) -> dict:
        """Perform a single-turn LLM call and parse the response.

        Args:
            system_prompt: Fully injected system prompt string.
            original_prompt: The user's original message.
            system_prompt_template: Raw template (used only for ACT diagnostic
                logging — presence of ``'act'`` in the first 200 chars triggers
                full response logging).

        Returns:
            dict: Standard cortex result dict with keys ``mode``, ``modifiers``,
                ``response``, ``generation_time``, ``actions``, ``confidence``,
                ``alternative_paths``, ``downstream_mode``.

        Raises:
            Exception: When the LLM call fails or JSON parsing fails after all
                recovery layers.
        """
        start_time = time.time()

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

    def generate_response_appended(
        self,
        system_prompt: str,
        messages: list,
        cache_prefix: bool = False,
        tools: list = None,
    ) -> dict:
        """Multi-turn LLM call using a pre-built system prompt.

        Used by the ACT loop in append mode: the system prompt is built once
        at loop start and the act_history grows as a message array across
        iterations, enabling provider-level prompt caching on the stable system
        prefix.

        Args:
            system_prompt: Pre-built system prompt (no act_history placeholder).
            messages: Growing list of ``{"role": ..., "content": ...}`` dicts.
            cache_prefix: When ``True``, hint to the provider to cache the system
                prompt prefix (Anthropic / OpenAI prompt-caching APIs).
            tools: List of native tool schema dicts.  When provided, the LLM
                uses native tool calling instead of JSON text output.

        Returns:
            Same dict structure as :meth:`generate_response`, plus
            ``raw_response`` containing the unmodified LLM text.  When native
            tool calling is used, ``tool_calls`` contains the structured tool
            invocations and ``actions`` is derived from them.

        Raises:
            Exception: When the LLM call fails.
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

        Applies up to 6 progressive JSON recovery layers:

        - Layer 0: direct ``json.loads``
        - Layer 0b: multi-object JSON (``raw_decode`` extracts first object)
        - Layer 1: fix invalid escape sequences (``\\$`` etc.)
        - Layer 1b: multi-object on escape-fixed text
        - Layer 2: extract ``{…}`` from prose wrapper
        - Layer 3: fix literal newlines inside JSON string values
        - Layer 4: extract ``"response"`` from broken JSON (unescaped inner quotes)
        - Layer 5: wrap pure prose as UNIFIED

        Args:
            response_text: Raw text returned by the LLM provider.
            generation_time: Wall-clock seconds for the LLM call.

        Returns:
            dict: Keys ``mode``, ``modifiers``, ``response``, ``generation_time``,
                ``actions``, ``confidence``, ``alternative_paths``,
                ``downstream_mode``.  Optional ``narrated``/``narration`` keys
                are included when present in the parsed data.

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
                """Attempt to decode the first JSON object from a multi-object string.

                Args:
                    text: String potentially containing multiple JSON objects.

                Returns:
                    The first decoded object, or ``None`` on failure.
                """
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
