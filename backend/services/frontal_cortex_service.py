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
from typing import List, Dict, Optional
from services.gist_storage_service import GistStorageService
import logging

# ─────────────────────────────────────────────────────────────────────────────
# Onboarding schedule — defines which identity traits to elicit, at what turn
# count, and with what behavioural hint to the LLM. Traits are elicited in list
# order; later entries can be added here without touching any other code.
# ─────────────────────────────────────────────────────────────────────────────
_ONBOARDING_SCHEDULE = [
    {
        'trait': 'name',
        'min_turn': 5,
        'cooldown_turns': 8,
        'max_attempts': 2,
        'hint': (
            "You haven't learned the user's name yet. "
            "Find a natural moment in your response to ask what they'd like to be called. "
            "Keep it casual — don't make it the focus of the response."
        ),
    },
    # Future additions (uncomment to activate):
    # {
    #     'trait': 'timezone',
    #     'min_turn': 20,
    #     'cooldown_turns': 15,
    #     'max_attempts': 1,
    #     'hint': (
    #         "If relevant, ask where the user is based — "
    #         "it helps with scheduling context."
    #     ),
    # },
    # {
    #     'trait': 'interests',
    #     'min_turn': 30,
    #     'cooldown_turns': 20,
    #     'max_attempts': 1,
    #     'hint': (
    #         "If the conversation touches on hobbies or interests, "
    #         "ask what they enjoy."
    #     ),
    # },
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


class FrontalCortexService:
    """Service for generating contextual responses using LLM."""

    def __init__(self, config: dict):
        """Initialize with configuration for LLM."""
        from services.llm_service import create_llm_service
        from services.world_state_service import WorldStateService
        from services.config_service import ConfigService
        from services.database_service import DatabaseService
        from services.episodic_retrieval_service import EpisodicRetrievalService

        self.config = config
        self.llm = create_llm_service(config)
        self.world_state_service = WorldStateService()

        # Initialize gist storage service
        attention_span_minutes = config.get('attention_span_minutes', 30)
        min_confidence = config.get('min_gist_confidence', 7)
        max_gists = config.get('max_gists', 8)
        self.gist_storage = GistStorageService(
            attention_span_minutes=attention_span_minutes,
            min_confidence=min_confidence,
            max_gists=max_gists
        )

        # Initialize episodic memory retrieval
        try:
            from services.database_service import get_merged_db_config

            episodic_config = ConfigService.resolve_agent_config("episodic-memory")
            db_config = get_merged_db_config()
            database_service = DatabaseService(db_config)
            self.episodic_retrieval = EpisodicRetrievalService(database_service, episodic_config)
            logging.info("Episodic memory retrieval initialized")
        except Exception as e:
            logging.warning(f"Episodic memory service not available: {e}")
            self.episodic_retrieval = None

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
                response_data = json.loads(fixed)  # propagates if still invalid

            # Extract fields (mode-specific prompts produce simpler output)
            mode = response_data.get('mode', 'RESPOND')
            modifiers = response_data.get('modifiers', [])
            user_response = response_data.get('response', '')
            actions = response_data.get('actions')
            confidence = response_data.get('confidence', 0.5)
            alternative_paths = response_data.get('alternative_paths', [])

            # Infer ACT mode if actions are present but mode wasn't explicit
            # (ACT prompt output contract returns actions without a mode field)
            if actions and isinstance(actions, list) and len(actions) > 0 and mode != 'ACT':
                mode = 'ACT'

            # Validate mode
            valid_modes = ['ACT', 'RESPOND', 'CLARIFY', 'ACKNOWLEDGE', 'IGNORE']
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
                    valid_terminal = ['RESPOND', 'CLARIFY', 'ACKNOWLEDGE', 'IGNORE']
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

        return {
            'mode': mode,
            'modifiers': modifiers,
            'response': user_response,
            'generation_time': generation_time,
            'actions': actions,
            'confidence': confidence,
            'alternative_paths': alternative_paths,
            'downstream_mode': response_data.get('downstream_mode', 'RESPOND')
        }

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

        # Use pre-assembled context if available, otherwise build internally
        if assembled_context:
            formatted_context = assembled_context.get('gists', '') if _include('gists') else ''
            if not formatted_context and _include('gists'):
                formatted_context = self._get_gist_context(topic)
            episodic_context = assembled_context.get('episodes', '') if _include('episodic_memory') else ''
            facts_context = assembled_context.get('facts', '') if _include('facts') else ''
            working_memory_context = assembled_context.get('working_memory', '') if _include('working_memory') else ''
            world_state = self.world_state_service.get_world_state(topic, thread_id=thread_id) if _include('world_state') else ''
        else:
            # Parallelize independent I/O calls for faster context assembly
            # Only submit futures for nodes that are included
            from concurrent.futures import ThreadPoolExecutor, as_completed

            context_results = {}
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {}
                if _include('gists'):
                    futures[executor.submit(self._get_gist_context, topic)] = 'gists'
                if _include('world_state'):
                    futures[executor.submit(self.world_state_service.get_world_state, topic, thread_id)] = 'world_state'
                if _include('episodic_memory'):
                    futures[executor.submit(self._get_episodic_context, original_prompt, topic, act_history)] = 'episodes'
                if _include('facts'):
                    futures[executor.submit(self._get_facts_context, topic)] = 'facts'
                if _include('working_memory'):
                    futures[executor.submit(self._get_working_memory_context, topic, thread_id)] = 'working_memory'

                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        context_results[key] = future.result()
                    except Exception as e:
                        logging.warning(f"[FRONTAL CORTEX] Parallel context failed for {key}: {e}")
                        context_results[key] = ''

            formatted_context = context_results.get('gists', '')
            world_state = context_results.get('world_state', '')
            episodic_context = context_results.get('episodes', '')
            facts_context = context_results.get('facts', '')
            working_memory_context = context_results.get('working_memory', '')

        # Replace placeholders
        result = template.replace('{{original_prompt}}', original_prompt)
        result = result.replace('{{topic}}', str(topic))
        result = result.replace('{{confidence}}', str(confidence))
        result = result.replace('{{chat_history}}', formatted_context)
        result = result.replace('{{world_state}}', world_state if _include('world_state') else '')
        result = result.replace('{{episodic_memory}}', episodic_context if _include('episodic_memory') else '')
        result = result.replace('{{act_history}}', act_history)
        result = result.replace('{{facts}}', facts_context if _include('facts') else '')
        result = result.replace('{{working_memory}}', working_memory_context if _include('working_memory') else '')

        # Get available skills (only if included)
        if _include('available_skills'):
            available_skills = self._get_available_skills()
        else:
            available_skills = ''
        result = result.replace('{{available_skills}}', available_skills)

        # Available tools (dynamic, from tool registry — filtered when selected_tools or relevant_tools provided)
        if _include('available_tools'):
            available_tools = self._get_available_tools(selected_tools=selected_tools, relevant_tools=relevant_tools)
        else:
            available_tools = ''
        result = result.replace('{{available_tools}}', available_tools)

        # Identity modulation (voice mapper)
        if _include('identity_modulation'):
            identity_modulation = self._get_identity_modulation()
        else:
            identity_modulation = ''
        result = result.replace('{{identity_modulation}}', identity_modulation)

        # Identity context — authoritative, zero-latency (Redis-backed)
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

        # Warm-return hint (ACKNOWLEDGE mode only)
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

        # Active goals (persistent directional goals)
        if _include('active_goals'):
            active_goals = self._get_active_goals(topic)
        else:
            active_goals = ''
        result = result.replace('{{active_goals}}', active_goals)

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

        return result

    def _get_identity_modulation(self) -> str:
        """Get identity modulation text from voice mapper."""
        try:
            from services.identity_service import IdentityService
            from services.voice_mapper_service import VoiceMapperService
            from services.database_service import DatabaseService, get_merged_db_config

            db_config = get_merged_db_config()
            db_service = DatabaseService(db_config)
            try:
                identity = IdentityService(db_service)
                mapper = VoiceMapperService()

                vectors = identity.get_vectors()
                identity.check_coherence()
                modulation = mapper.generate_modulation(vectors)
                return modulation if modulation else "Engage naturally as a peer."
            finally:
                db_service.close_pool()
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
            from services.database_service import DatabaseService, get_merged_db_config

            db_config = get_merged_db_config()
            db_service = DatabaseService(db_config)
            try:
                trait_service = UserTraitService(db_service)
                effective_threshold = (
                    injection_threshold_override
                    if injection_threshold_override is not None
                    else INJECTION_THRESHOLD
                )
                return trait_service.get_traits_for_prompt(
                    prompt, injection_threshold=effective_threshold
                )
            finally:
                db_service.close_pool()
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
            from services.redis_client import RedisClientService

            # Read current exchange count from thread hash
            r = RedisClientService.create_connection()
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

    def _get_active_goals(self, topic: str = "") -> str:
        """
        Get active goals formatted for prompt injection.

        Args:
            topic: Current topic for relevance filtering

        Returns:
            str: Formatted active goals section or empty string
        """
        try:
            from services.goal_service import GoalService
            from services.database_service import DatabaseService, get_merged_db_config

            db_config = get_merged_db_config()
            db_service = DatabaseService(db_config)
            try:
                goal_service = GoalService(db_service)
                return goal_service.get_goals_for_prompt(topic=topic)
            finally:
                db_service.close_pool()
        except Exception as e:
            logging.debug(f"Active goals not available: {e}")
            return ""

    def _get_active_lists(self) -> str:
        """
        Get active lists formatted for prompt injection.

        Returns:
            str: Formatted active lists section or empty string
        """
        try:
            from services.list_service import ListService
            from services.database_service import DatabaseService, get_merged_db_config

            db_config = get_merged_db_config()
            db_service = DatabaseService(db_config)
            try:
                list_service = ListService(db_service)
                return list_service.get_lists_for_prompt()
            finally:
                db_service.close_pool()
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
            from services.database_service import DatabaseService, get_merged_db_config

            db_config = get_merged_db_config()
            db_service = DatabaseService(db_config)
            try:
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
            finally:
                db_service.close_pool()
        except Exception as e:
            logging.debug(f"Communication style not available: {e}")
            return ""

    def _get_client_context(self) -> str:
        """
        Get client context (timezone, location, locale) from Redis heartbeat.

        Returns:
            str: Formatted client context or empty string if not available
        """
        try:
            from services.client_context_service import ClientContextService
            return ClientContextService().format_for_prompt()
        except Exception as e:
            logging.debug(f"Client context not available: {e}")
            return ""

    def _get_gist_context(self, topic: str) -> str:
        """
        Get gists from Redis or fallback to last message.

        Args:
            topic: Topic name

        Returns:
            str: Formatted gist context or fallback message
        """
        # Try to get gists from Redis
        gists = self.gist_storage.get_latest_gists(topic)

        if gists:
            # Filter out cold_start gists — they're internal metadata, not conversation context
            real_gists = [g for g in gists if g.get('type') != 'cold_start']
            if real_gists:
                lines = ["## Recent Conversation Gists"]
                for gist in real_gists:
                    content = gist['content']
                    gist_type = gist['type']
                    confidence = gist['confidence']
                    lines.append(f"- [{gist_type}] {content} (confidence: {confidence})")
                return "\n".join(lines)

        # Fallback to last message if no gists available
        last_message = self.gist_storage.get_last_message(topic)
        if last_message:
            return f"## Last Exchange\nUser: {last_message['prompt']}\nAssistant: {last_message['response']}"

        return "No previous conversation context available"

    def _get_available_skills(self) -> str:
        """
        Get available innate skills from the dispatcher for prompt injection.

        Returns innate skills list only. Dynamic tools are injected separately
        via {{available_tools}} placeholder to keep concerns separated.

        Returns:
            Formatted available skills string or empty string
        """
        try:
            from services.act_dispatcher_service import ActDispatcherService
            dispatcher = ActDispatcherService()
            innate = ["recall", "memorize", "introspect", "associate", "autobiography", "goal", "focus", "list"]
            available = [s for s in innate if s in dispatcher.handlers]
            if available:
                return "Available skills: " + ", ".join(available)
            return ""
        except Exception as e:
            logging.debug(f"Skill registry not available: {e}")
            return ""

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
                # Try to get rich profiles first
                try:
                    from services.tool_profile_service import ToolProfileService
                    profiles = ToolProfileService().get_profiles_for_tools(tool_names)
                    if profiles:
                        lines = []
                        for p in profiles:
                            name = p.get('tool_name', '')
                            summary = p.get('short_summary', '')
                            lines.append(f"- {name}: {summary}")
                        return "\n".join(lines)
                except Exception:
                    pass

                # Fallback to manifest-based summaries
                lines = []
                for name in tool_names:
                    manifest = registry.tools[name]['manifest']
                    desc = manifest.get('description', name)
                    params = manifest.get('parameters', {})
                    param_str = f" ({', '.join(list(params.keys()))})" if params else ""
                    lines.append(f"- {name}{param_str}: {desc}")
                return "\n".join(lines)

            # No selected tools — return all registered tools
            summaries = registry.get_tool_prompt_summaries()
            return summaries if summaries else "(no tools loaded)"
        except Exception as e:
            logging.debug(f"Tool registry not available for prompt: {e}")
            return "(no tools loaded)"

    def _get_facts_context(self, topic: str) -> str:
        """
        Get facts from Redis formatted for prompt injection.

        Args:
            topic: Topic name

        Returns:
            Formatted facts context or empty string
        """
        try:
            from services.fact_store_service import FactStoreService
            fact_store = FactStoreService()
            return fact_store.get_facts_formatted(topic)
        except Exception as e:
            logging.warning(f"Fact store not available: {e}")
            return ""

    def _get_working_memory_context(self, topic: str, thread_id: str = None) -> str:
        """
        Get working memory formatted for prompt injection.

        Args:
            topic: Topic name (legacy fallback key)
            thread_id: Thread ID (preferred key, overrides topic)

        Returns:
            Formatted working memory context or empty string
        """
        try:
            from services.working_memory_service import WorkingMemoryService
            max_turns = self.config.get('max_working_memory_turns', 10)
            working_memory = WorkingMemoryService(max_turns=max_turns)
            identifier = thread_id if thread_id else topic
            return working_memory.get_formatted_context(identifier)
        except Exception as e:
            logging.warning(f"Working memory not available: {e}")
            return ""

    def _extract_semantic_from_history(self, act_history: str) -> List[Dict]:
        """
        Parse semantic_query results from act_history string.

        Returns list of {"name": "...", "definition": "..."} dicts.
        """
        import re

        concepts = []
        # Match both old format "(strength: 0.XX)" and innate skills format "(confidence=X, strength=Y)"
        pattern = r'-\s+([^:]+):\s+([^(]+)\s+\((?:strength[:=]|confidence=)'
        matches = re.findall(pattern, act_history)

        for name, definition in matches:
            concepts.append({
                'name': name.strip(),
                'definition': definition.strip()
            })

        return concepts

    def _get_episodic_context(self, prompt: str, topic: str, act_history: str = "") -> str:
        """
        Retrieve relevant episodes and format as context.

        Args:
            prompt: User's current prompt
            topic: Current topic
            act_history: ACT history to extract semantic concepts from

        Returns:
            str: Formatted episodic memory context or empty string
        """
        if not self.episodic_retrieval:
            return ""

        try:
            # Extract semantic concepts from act_history if present
            semantic_concepts = self._extract_semantic_from_history(act_history)

            # Retrieve episodes with semantic boost
            episodes = self.episodic_retrieval.retrieve_episodes(
                query_text=prompt,
                topic=topic,
                limit=3,
                semantic_concepts=semantic_concepts if semantic_concepts else None
            )

            if not episodes:
                return ""

            lines = ["\n## Relevant Past Experiences"]
            for i, ep in enumerate(episodes, 1):
                lines.append(f"{i}. {ep['gist']}")
                lines.append(f"   - Outcome: {ep['outcome']}")
                lines.append(f"   - Salience: {ep['salience']}/10")

            return "\n".join(lines)

        except Exception as e:
            logging.error(f"Failed to retrieve episodic memories: {e}")
            return ""

