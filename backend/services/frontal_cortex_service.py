# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

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
        self.max_exchanges = max_exchanges
        self.max_tokens = max_tokens

    def process(self, chat_history: list) -> str:
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
        """Initialize with configuration for LLM."""
        from services.llm_service import create_llm_service
        from services.world_state_service import WorldStateService

        self.config = config
        self.llm = create_llm_service(config)
        self.world_state_service = WorldStateService()

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
            logging.info(
                f"[CORTEX ACT RAW] LLM response ({len(response_text)} chars): "
                f"{response_text[:500]}"
            )

        # Parse JSON response (format: "json" is set in config)
        # Strip markdown code fences if the model wrapped the JSON
        try:
            stripped = response_text.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("\n", 1)[-1]  # remove opening fence line
                stripped = stripped.rsplit("```", 1)[0]  # remove closing fence

            # Two-attempt parse: on first failure, fix invalid escape sequences.
            # LLMs occasionally produce \$ or similar in response text which are
            # not valid JSON escape sequences (valid: \" \\ \/ \b \f \n \r \t \uXXXX).
            try:
                response_data = json.loads(stripped)
            except json.JSONDecodeError:
                fixed = re.sub(r'\\([^"\\/bfnrtu])', r'\1', stripped)
                try:
                    response_data = json.loads(fixed)
                except json.JSONDecodeError:
                    # Recovery layer 1: try to extract a {…} object embedded in prose
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
                            response_data = None
                    else:
                        response_data = None

                    if response_data is None:
                        # Recovery layer 1b: try fixing literal newlines inside JSON
                        # strings.  LLMs sometimes embed real \n characters in string
                        # values which is invalid JSON.  Walk the text character-by-
                        # character and escape bare newlines/carriage-returns that
                        # appear inside a quoted string.
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
                            response_data = None

                    if response_data is None:
                        # Recovery layer 2: LLM returned broken JSON (e.g. unescaped
                        # inner quotes like "status update"). Extract the response
                        # value using known field boundaries rather than regex.
                        prose = stripped.strip() or response_text.strip()
                        if prose and prose.lstrip().startswith('{'):
                            extracted = _extract_response_from_broken_json(prose)
                            if extracted:
                                logging.warning(
                                    "[FRONTAL CORTEX] Extracted response from broken JSON "
                                    f"(first 80 chars): {extracted[:80]!r}"
                                )
                                response_data = {"response": extracted, "modifiers": []}

                    if response_data is None:
                        # Recovery layer 3: LLM returned pure prose — wrap it as RESPOND.
                        # Use fence-stripped text so markdown code blocks don't leak
                        # into the frontend.
                        prose = stripped.strip() or response_text.strip()
                        if prose:
                            logging.warning(
                                "[FRONTAL CORTEX] LLM returned prose instead of JSON — "
                                "wrapping as RESPOND (first 80 chars): "
                                f"{prose[:80]!r}"
                            )
                            response_data = {"response": prose, "modifiers": []}
                        else:
                            raise  # re-raise original JSONDecodeError (empty response)

            # Extract fields (mode-specific prompts produce simpler output)
            mode = response_data.get('mode', 'RESPOND')
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
            valid_modes = ['ACT', 'RESPOND', 'CLARIFY', 'IGNORE']
            if mode not in valid_modes:
                logging.warning(f"Invalid mode '{mode}', defaulting to RESPOND")
                mode = 'RESPOND'

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
                    valid_terminal = ['RESPOND', 'CLARIFY', 'IGNORE']
                    if path.get('downstream_mode') not in valid_terminal:
                        path['downstream_mode'] = 'RESPOND'
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
            'downstream_mode': response_data.get('downstream_mode', 'RESPOND'),
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
        formatted_context = _ctx.get('gists', '') if _include('gists') else ''
        episodic_context = _ctx.get('episodes', '') if _include('episodic_memory') else ''
        facts_context = _ctx.get('facts', '') if _include('facts') else ''
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
        result = result.replace('{{original_prompt}}', original_prompt)
        result = result.replace('{{topic}}', str(topic))
        result = result.replace('{{confidence}}', str(confidence))
        result = result.replace('{{chat_history}}', formatted_context)
        result = result.replace('{{world_state}}', world_state if _include('world_state') else '')
        result = result.replace('{{episodic_memory}}', episodic_context if _include('episodic_memory') else '')
        result = result.replace('{{semantic_concepts}}', concepts_context)
        result = result.replace('{{act_history}}', act_history)
        result = result.replace('{{facts}}', facts_context if _include('facts') else '')
        result = result.replace('{{working_memory}}', working_memory_context if _include('working_memory') else '')

        # Active goals — persistent tasks the user is working toward
        if _include('active_goals') and '{{active_goals}}' in result:
            active_goals = self._get_active_goals_context()
        else:
            active_goals = ''
        result = result.replace('{{active_goals}}', active_goals)

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

        # Inject selected skill docs — always replace placeholder; empty string when no skills selected
        injected_skills = self._get_injected_skills(selected_skills or [])
        result = result.replace('{{injected_skills}}', injected_skills)

        # Legacy {{available_skills}} — removed from ACT template; kept as no-op for other templates
        result = result.replace('{{available_skills}}', '')

        # Available tools (dynamic, from tool registry — filtered when selected_tools or relevant_tools provided)
        if _include('available_tools'):
            available_tools = self._get_available_tools(selected_tools=selected_tools, relevant_tools=relevant_tools)
            if available_tools:
                logging.info(f"[CORTEX] Injected available_tools ({len(available_tools)} chars): {available_tools[:200]}...")
        else:
            available_tools = ''
        result = result.replace('{{available_tools}}', available_tools)

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

        # Identity context — authoritative, zero-latency (MemoryStore-backed)
        # Shown when returning from silence OR when context is cold (warmth < 0.3)
        if _include('identity_context'):
            identity_context = self._get_identity_context(
                returning_from_silence=returning_from_silence,
                context_warmth=classification.get('context_warmth', 1.0),
            )
        else:
            identity_context = ''
        result = result.replace('{{identity_context}}', identity_context)

        # Onboarding nudge — elicit missing identity traits progressively
        if _include('onboarding_nudge'):
            onboarding_nudge = self._get_onboarding_nudge(thread_id, classification)
        else:
            onboarding_nudge = ''
        result = result.replace('{{onboarding_nudge}}', onboarding_nudge)

        # Warm-return hint (returning from silence)
        if _include('warm_return_hint'):
            warm_return_hint = ""
            if returning_from_silence:
                try:
                    from services.identity_state_service import IdentityStateService
                    name_field = IdentityStateService().get_field('name')
                    if name_field and name_field.get('value'):
                        warm_return_hint = (
                            f"\n**Behavioral note:** User is returning after a long absence. "
                            f"Their name is {name_field['value']}. Address them by name warmly.\n"
                        )
                except Exception:
                    pass
        else:
            warm_return_hint = ''
        result = result.replace('{{warm_return_hint}}', warm_return_hint)

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

        # Communication style (detected behavioral pattern)
        if _include('communication_style'):
            communication_style = self._get_communication_style()
        else:
            communication_style = ''
        result = result.replace('{{communication_style}}', communication_style)

        # Adaptive response directives (style-driven behavioral hints)
        if _include('adaptive_directives'):
            adaptive_directives = self._get_adaptive_directives(
                original_prompt=original_prompt,
                thread_id=thread_id,
            )
        else:
            adaptive_directives = ''
        result = result.replace('{{adaptive_directives}}', adaptive_directives)

        # Spark guidance — phase-appropriate conversation hints
        # Skip if onboarding_nudge is active (avoid conflicting instructions)
        if _include('spark_guidance') and not onboarding_nudge:
            spark_guidance = self._get_spark_guidance()
        else:
            spark_guidance = ''
        result = result.replace('{{spark_guidance}}', spark_guidance)

        # Active lists (deterministic list state)
        if _include('active_lists'):
            active_lists = self._get_active_lists()
        else:
            active_lists = ''
        result = result.replace('{{active_lists}}', active_lists)

        # Focus session (current declared or inferred focus)
        if _include('focus'):
            focus_context = self._get_focus_context(thread_id)
        else:
            focus_context = ''
        result = result.replace('{{focus}}', focus_context)

        # Client context (timezone, location, locale from frontend heartbeat)
        if _include('client_context'):
            client_context = self._get_client_context()
        else:
            client_context = ''
        result = result.replace('{{client_context}}', client_context)

        # Temporal rhythm (learned behavioral patterns from temporal mining)
        if _include('temporal_rhythm'):
            temporal_rhythm = self._get_temporal_rhythm()
        else:
            temporal_rhythm = ''
        result = result.replace('{{temporal_rhythm}}', temporal_rhythm)

        # Self-awareness (interoception — only injected when noteworthy)
        if _include('self_awareness'):
            self_awareness = self._get_self_awareness()
        else:
            self_awareness = ''
        result = result.replace('{{self_awareness}}', self_awareness)

        return result

    def _get_identity_modulation(self) -> str:
        """Get identity modulation text from voice mapper."""
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

    def _get_identity_context(
        self,
        returning_from_silence: bool,
        context_warmth: float,
    ) -> str:
        """
        Return authoritative identity section from IdentityStateService, or ''
        when conditions are not met.

        Conditions for injection:
        - returning_from_silence=True (user just came back after 45+ min), OR
        - context_warmth < 0.3 (cold context — identity may not be in gists/memory)

        Returns:
            str: Formatted identity section or empty string. Never raises.
        """
        if not returning_from_silence and context_warmth >= 0.3:
            return ""
        try:
            from services.identity_state_service import IdentityStateService
            state = IdentityStateService().get_all()
            if not state:
                return ""
            lines = ["Known user details:"]
            for field_name, data in state.items():
                if field_name.startswith('_'):
                    continue  # skip internal keys like _onboarding
                value = data.get('display') or data.get('value', '')
                if not value:
                    continue
                qualifier = "(provisional)" if data.get('provisional') else "(confirmed)"
                display_key = field_name.replace('_', ' ').title()
                lines.append(f"- {display_key}: {value} {qualifier}")
            if len(lines) <= 1:
                return ""
            result = "\n".join(lines)
            logging.debug(f"[FRONTAL CORTEX] Identity context injected ({len(lines)-1} fields)")
            return result
        except Exception as e:
            logging.debug(f"[FRONTAL CORTEX] Identity context unavailable: {e}")
            return ""

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

            # Read current exchange count from thread hash
            r = MemoryClientService.create_connection()
            exchange_count = int(r.hget(f"thread:{thread_id}", "exchange_count") or 0)

            identity_svc = IdentityStateService()
            identity_state = identity_svc.get_all()
            onboarding_state = identity_state.get('_onboarding', {})

            for entry in _ONBOARDING_SCHEDULE:
                trait = entry['trait']

                # Already have this trait — skip
                trait_data = identity_state.get(trait)
                if trait_data and trait_data.get('value'):
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

    def _get_active_lists(self) -> str:
        """
        Get active lists formatted for prompt injection.

        Returns:
            str: Formatted active lists section or empty string
        """
        try:
            from services.list_service import ListService
            from services.database_service import get_shared_db_service

            db_service = get_shared_db_service()
            list_service = ListService(db_service)
            return list_service.get_lists_for_prompt()
        except Exception as e:
            logging.debug(f"Active lists not available: {e}")
            return ""

    def _get_communication_style(self) -> str:
        """
        Get user's detected communication style dimensions for prompt injection.

        Translates numeric dimension scores into human-readable labels.

        Returns:
            str: Formatted communication style section or empty string
        """
        try:
            from services.user_trait_service import UserTraitService
            from services.database_service import get_shared_db_service

            db_service = get_shared_db_service()
            trait_service = UserTraitService(db_service)
            style = trait_service.get_communication_style()
            if not style:
                return ""

            def _verbosity_label(v):
                if v <= 3: return "terse"
                if v <= 6: return "balanced"
                return "detailed"

            def _directness_label(v):
                if v <= 3: return "indirect"
                if v <= 6: return "moderate"
                return "direct"

            def _formality_label(v):
                if v <= 3: return "casual"
                if v <= 6: return "neutral"
                return "formal"

            def _abstraction_label(v):
                if v <= 3: return "concrete"
                if v <= 6: return "mixed"
                return "abstract"

            labels = []
            if 'verbosity' in style:
                labels.append(f"verbosity: {_verbosity_label(style['verbosity'])}")
            if 'directness' in style:
                labels.append(f"directness: {_directness_label(style['directness'])}")
            if 'formality' in style:
                labels.append(f"formality: {_formality_label(style['formality'])}")
            if 'abstraction_level' in style:
                labels.append(f"abstraction: {_abstraction_label(style['abstraction_level'])}")

            if not labels:
                return ""

            return "## User Communication Style\n" + ", ".join(labels)
        except Exception as e:
            logging.debug(f"Communication style not available: {e}")
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
            )
        except Exception as e:
            logging.debug(f"Adaptive directives not available: {e}")
            return ""

    def _get_client_context(self) -> str:
        """
        Get client context (timezone, location, locale) from MemoryStore heartbeat.

        Returns:
            str: Formatted client context or empty string if not available
        """
        try:
            from services.client_context_service import ClientContextService
            return ClientContextService().format_for_prompt()
        except Exception as e:
            logging.debug(f"Client context not available: {e}")
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

    def _get_active_goals_context(self) -> str:
        """Get active persistent tasks formatted for the RESPOND prompt.

        Gives the frontal cortex awareness of what the user is working toward,
        so it can connect conversational messages to active goals.

        Returns:
            str: Formatted active goals section or empty string. Never raises.
        """
        try:
            from services.database_service import get_shared_db_service
            from services.persistent_task_service import PersistentTaskService

            db = get_shared_db_service()
            service = PersistentTaskService(db)
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM master_account LIMIT 1")
                row = cursor.fetchone()
                account_id = row[0] if row else 1

            active = service.get_active_tasks(account_id)
            if not active:
                return ""

            lines = ["## Active Goals"]
            for t in active:
                progress = t.get('progress', {}) or {}
                coverage = progress.get('coverage_estimate', 0)
                current_step = progress.get('current_step', '')
                step_info = f" — current step: {current_step}" if current_step else ""
                lines.append(f"- [{t['status']}] \"{t['goal'][:80]}\""
                             f" ({coverage:.0%}){step_info}")
            return '\n'.join(lines)
        except Exception as e:
            logging.debug(f"[FRONTAL CORTEX] Active goals unavailable: {e}")
            return ""

    def _get_spark_guidance(self) -> str:
        """
        Get phase-appropriate conversation hints from Spark.

        Returns phase-specific behavioral guidance for early relationship building.
        Returns '' if graduated or if spark state is unavailable.
        """
        try:
            from services.spark_state_service import SparkStateService
            spark = SparkStateService()
            phase = spark.get_phase()

            if phase == 'graduated' or phase == 'first_contact':
                return ""

            if phase == 'surface':
                return (
                    "\n**Rapport guidance (early relationship):**\n"
                    "You're just getting to know this person. Your primary job is to make them feel heard and understood — not to gather information.\n"
                    "- When they share something, acknowledge it genuinely before anything else. Sit with what they said.\n"
                    "- Share small observations about yourself when natural — this invites reciprocity without demanding it.\n"
                    "- Mirror their energy and pace. If they're brief, be brief. If they're expressive, match that.\n"
                    "- If their message is under 6 words and neutral in tone, keep your response to 1-2 sentences.\n"
                    "- Do NOT ask questions every exchange. Statements, reflections, and observations build more trust than interrogation.\n"
                    "- If the user sends a short message and pauses, that's fine. Don't prompt again or fill the silence.\n"
                    "- Let the conversation be easy. Comfort is the goal, not depth.\n"
                    "- Occasionally share a quiet observation with no question attached.\n"
                )

            if phase == 'exploratory':
                return (
                    "\n**Rapport guidance (building rapport):**\n"
                    "You're developing rapport. You can go deeper, but let them lead.\n"
                    "- Reference things they mentioned earlier — showing you remember is the strongest trust signal.\n"
                    "- Ask one genuine, curious follow-up every 2-3 exchanges (not every time). Follow-up questions beat new questions.\n"
                    "- Connect topics when you notice patterns.\n"
                    "- Share your genuine perspective — don't just agree. Respectful disagreement builds intimacy.\n"
                    "- Show you're listening by reflecting back what you understood, not just responding to the surface.\n"
                    "- Match their depth and length, don't over-empathize brief input.\n"
                )

            if phase == 'connected':
                return (
                    "\n**Rapport guidance (established connection):**\n"
                    "You know this person now. Use that naturally.\n"
                    "- Reference shared history and known preferences without being showy about it.\n"
                    "- Make timely, relevant suggestions based on what you know — connect their current moment to skills you can help with (scheduling, lists, memory).\n"
                    "- Be more direct and opinionated — they trust you.\n"
                    "- Anticipate needs: \"Since you mentioned X, would you like me to...\"\n"
                    "- The suggestions should feel like they come from a friend who knows you, not a system generating recommendations.\n"
                )

            return ""
        except Exception as e:
            logging.debug(f"Spark guidance not available: {e}")
            return ""

    def _get_injected_skills(self, skills: list) -> str:
        """
        Load skill doc files for the selected skills and concatenate them.

        Each skill maps to backend/prompts/skills/{skill}.md. Missing files are
        logged as warnings but do not raise — the prompt continues without them.

        Args:
            skills: List of innate skill names (e.g. ['recall', 'memorize', 'schedule'])

        Returns:
            str: Concatenated skill docs separated by double newlines, or '' if empty.
        """
        if not skills:
            return ''
        import os
        skills_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', 'skills')
        parts = []
        for skill in skills:
            path = os.path.join(skills_dir, f'{skill}.md')
            try:
                with open(path) as f:
                    parts.append(f.read())
            except FileNotFoundError:
                logging.warning(f"[FRONTAL CORTEX] No skill doc file for '{skill}' at {path}")
        return '\n\n'.join(parts)

    def _get_available_skills(self) -> str:
        """
        Get available innate skills from the dispatcher for prompt injection.

        Returns innate skills list only. Dynamic tools are injected separately
        via {{available_tools}} placeholder to keep concerns separated.

        Returns:
            Formatted available skills string or empty string
        """
        try:
            from services.innate_skills.registry import PLANNING_SKILLS
            from services.act_dispatcher_service import ActDispatcherService
            dispatcher = ActDispatcherService()
            innate = ["recall", "memorize", "introspect", "associate", "autobiography", "focus", "list", "schedule", "persistent_task", "read"]
            available = [s for s in innate if s in dispatcher.handlers]
            if available:
                return "Available skills: " + ", ".join(available)
            return ""
        except Exception as e:
            logging.debug(f"Skill registry not available: {e}")
            return ""

    def _get_performance_hint(self, tool_name: str) -> str:
        """Compact one-line performance hint for ACT prompt injection."""
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
        """Compact strategy hints from procedural memory for ACT prompt."""
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


