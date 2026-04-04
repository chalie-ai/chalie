# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Prompt Assembly Service — system-prompt construction and placeholder injection.

Assembles the full system prompt from 60+ context injection placeholders
(memory, identity, user traits, calendar, tool results, etc.) by replacing
``{{variable}}`` tokens with live runtime values.

This module is extracted from :mod:`services.frontal_cortex_service` as part
of the WS3 decomposition.  The :class:`FrontalCortexService` facade delegates
all prompt-assembly calls here.
"""

import logging
from typing import Optional

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

# ─────────────────────────────────────────────────────────────────────────────
# Capability connect nudge — injected when no life integrations are configured.
# Separate from the trait schedule because it checks capability state, not
# identity traits.
# ─────────────────────────────────────────────────────────────────────────────
_CAPABILITY_NUDGE = {
    'key': '_capability_connect',
    'min_turn': 4,
    'cooldown_turns': 10,
    'max_attempts': 2,
    'hint': (
        "No life integrations are connected yet. "
        "Naturally mention that connecting a calendar would let you help with "
        "scheduling, conflicts, and daily briefings. "
        "Keep it brief and inviting — don't push."
    ),
}


class PromptAssemblyService:
    """Assembles system prompts from templates and live runtime context.

    All ``{{variable}}`` placeholder injection logic lives here.  Response
    generation (LLM calls, JSON parsing) is handled by
    :class:`~services.response_generation_service.ResponseGenerationService`.
    """

    def __init__(self, config: dict):
        """Initialise the prompt assembly service.

        Args:
            config: Provider configuration dict (same shape passed to
                :func:`~services.llm_service.create_llm_service`).  Used to
                read feature-flag keys such as ``append_mode`` that influence
                how skill docs are injected.
        """
        from services.world_state_service import WorldStateService

        self.config = config
        self.world_state_service = WorldStateService()

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
            system_prompt_template: Template with ``{{variable}}`` placeholders.
            original_prompt: The user's original message.
            classification: Dict containing topic, confidence, etc.
            chat_history: List of previous exchanges.
            assembled_context: Pre-assembled context dict.
            relevant_tools: Tools scored by embedding relevance.
            selected_tools: Triage-selected tools.
            selected_skills: Triage-selected innate skills.
            thread_id: Conversation thread identifier.
            returning_from_silence: Whether the user is returning after a gap.
            inclusion_map: Context node inclusion decisions.

        Returns:
            str: Fully injected system prompt (act_history left empty).
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
        """Replace ``{{variable}}`` placeholders in *template* with actual values.

        Args:
            template: System prompt template.
            original_prompt: User's message.
            classification: Classification result dict.
            chat_history: Previous exchanges (unused — kept for compatibility).
            act_history: ACT loop history string (defaults to empty).
            assembled_context: Pre-assembled context dict from
                :class:`~services.context_assembly_service.ContextAssemblyService`.
            relevant_tools: Tools scored by embedding relevance.
            selected_tools: Triage-selected tool names.
            selected_skills: Triage-selected innate skill names.
            thread_id: Conversation thread identifier.
            returning_from_silence: When ``True`` lowers user-trait injection
                threshold to surface more context.
            inclusion_map: Context node inclusion decisions from
                :class:`~services.context_relevance_service.ContextRelevanceService`.
                When ``None`` all nodes are included (backward compatible).

        Returns:
            str: Processed system prompt with all placeholders replaced.
        """
        # Determine the topic (classifier may return similar_topic, topic_update, or topic)
        topic = classification.get('similar_topic') or \
                classification.get('topic_update') or \
                classification.get('topic', 'unknown')

        confidence = classification.get('confidence', 0)

        # Helper to check if a node should be included
        def _include(node):
            return (inclusion_map or {}).get(node, True)

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

        # Replace placeholders — use user's local timezone when available
        try:
            from services.time_utils import utc_now, get_user_tz
            _tz = get_user_tz()
            _now = utc_now().astimezone(_tz)
            _tz_abbr = _now.strftime('%Z') or _tz.key
            _current_datetime = _now.strftime(f'%A, %Y-%m-%d %H:%M {_tz_abbr}')
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

        # Methodology guidance from learned goal-cluster patterns
        methodology_guidance = ''
        if _include('strategy_hints'):
            methodology_guidance = self._get_methodology_guidance(original_prompt)
        result = result.replace('{{strategy_hints}}', methodology_guidance)

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
                _injection_threshold_override=0.2 if returning_from_silence else None,
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
        _injection_threshold_override: Optional[float] = None,
    ) -> str:
        """Get the pre-synthesized user sentence for prompt injection.

        Returns the single natural-language sentence produced by the
        trait-synthesis LLM call after trait extraction, cached in the
        MemoryStore.  Returns an empty string when no sentence is available yet.

        Args:
            prompt: Unused (kept for call-site compatibility).
            topic: Unused (kept for call-site compatibility).
            injection_threshold_override: Unused (kept for call-site compatibility).

        Returns:
            str: ``## About the User`` section with the synthesized sentence,
                or empty string when unavailable.
        """
        try:
            from services.database_service import get_shared_db_service
            from services.knowledge_service import KnowledgeService
            ks = KnowledgeService(get_shared_db_service())
            entry = ks.get('system', 'user_summary')
            if not entry or not entry.get('value'):
                return ""
            return "## About the User\n" + entry['value']
        except Exception as e:
            logging.debug(f"User traits not available: {e}")
            return ""

    def _get_user_state(self) -> str:
        """Return consolidated telemetry: time, location, device, energy, attention, mobility.

        Pure telemetry — no identity data. Always injected, no gating. ~25 tokens.

        Returns:
            str: Formatted user-state line, or empty string on failure.
        """
        try:
            return self._build_user_state_line()
        except Exception as e:
            logging.warning(f"[FRONTAL CORTEX] User state failed: {e}")
            return ''

    def _build_user_state_line(self) -> str:
        """Build the user state line — pure telemetry, no identity.

        Returns:
            str: Middle-dot-joined string of telemetry parts (e.g.
            ``'Mon 23 Mar, 14:32 UTC · mobile 82% · home (office) · Energy: high'``),
            or empty string when no client context is available.

        Raises:
            Exception: Propagates any unexpected service failure to the caller
                (:meth:`_get_user_state` catches and suppresses it).
        """
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
        """Return a behavioural nudge for missing identity traits, or ``''`` if none due.

        Checks:

        - Exchange count vs ``min_turn`` threshold.
        - Whether the trait already exists in ``IdentityStateService``.
        - Nudge cooldown and ``max_attempts`` from :data:`_ONBOARDING_SCHEDULE`.

        Skips when *classification* signals high urgency or tool use.
        Updates onboarding state in ``IdentityStateService`` on nudge emission.

        Args:
            thread_id: Conversation thread identifier.
            classification: Current classification dict (used to skip nudges
                during urgent or tool-heavy exchanges).

        Returns:
            str: Formatted onboarding hint string, or empty string.

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
            from services.knowledge_service import KnowledgeService

            # Read current exchange count from thread hash
            r = MemoryClientService.create_connection()
            exchange_count = int(r.hget(f"thread:{thread_id}", "exchange_count") or 0)

            identity_svc = IdentityStateService()
            identity_state = identity_svc.get_all()
            onboarding_state = identity_state.get('_onboarding', {})

            # Build set of trait keys already known in permanent storage
            # (knowledge table survives MemoryStore TTL expiry)
            try:
                db = get_shared_db_service()
                ks = KnowledgeService(db)
                known_traits = ks.get_by_kind('trait', entity='user', min_confidence=0.5)
                known_trait_keys = {t['key'] for t in known_traits if t.get('key')}
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

            # ── Capability connect nudge ─────────────────────────────────
            # If no trait nudge is due, check whether any life capability
            # has been configured.  If not, suggest connecting one.
            try:
                from capabilities import load_capabilities
                from services.tool_config_service import ToolConfigService

                caps = load_capabilities()
                any_configured = any(c._connected for c in caps.values())
                if not any_configured:
                    tcs = ToolConfigService(get_shared_db_service())
                    any_configured = any(tcs.get_tool_config(c.get_id()) for c in caps.values())

                if not any_configured and exchange_count >= _CAPABILITY_NUDGE['min_turn']:
                    cap_nudge = onboarding_state.get(_CAPABILITY_NUDGE['key'], {})
                    cap_attempts = cap_nudge.get('attempts', 0)
                    cap_last_turn = cap_nudge.get('nudged_at_turn', 0)
                    cooldown_ok = cap_attempts == 0 or (exchange_count - cap_last_turn) >= _CAPABILITY_NUDGE['cooldown_turns']

                    if cap_attempts < _CAPABILITY_NUDGE['max_attempts'] and cooldown_ok:
                        cap_nudge['nudged_at_turn'] = exchange_count
                        cap_nudge['attempts'] = cap_attempts + 1
                        onboarding_state[_CAPABILITY_NUDGE['key']] = cap_nudge
                        identity_svc.set_onboarding_state(onboarding_state)

                        logging.debug(
                            "[FRONTAL CORTEX] Capability connect nudge "
                            "attempt=%d at turn %d",
                            cap_nudge['attempts'], exchange_count,
                        )
                        return f"\n**Onboarding note:** {_CAPABILITY_NUDGE['hint']}\n"
            except Exception:
                pass  # capability check is best-effort

            return ""
        except Exception as e:
            logging.debug(f"[FRONTAL CORTEX] Onboarding nudge unavailable: {e}")
            return ""

    def _get_adaptive_directives(self, original_prompt: str = "", thread_id: str = None) -> str:
        """Get adaptive response directives based on detected user interaction style.

        Args:
            original_prompt: The user's current message (used for token-count
                signal and cognitive-load estimation).
            thread_id: Conversation thread identifier.

        Returns:
            str: Formatted directive block, or empty string when unavailable.
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

    def _get_self_awareness(self) -> str:
        """Get self-awareness context from self-model (only when noteworthy).

        Returns empty string when all systems are healthy (zero token cost).
        When degraded, includes signals AND behavioral directives.

        Returns:
            str: Self-awareness section, or empty string when all systems nominal.
        """
        try:
            from services.self_model_service import SelfModelService
            return SelfModelService().format_for_prompt()
        except Exception as e:
            logging.debug(f"Self-awareness not available: {e}")
            return ""

    def get_skill_docs(self, skills: list) -> str:
        """Return formatted skill documentation for the given skill names.

        Public API for loading skill documentation by name.
        Used by the orchestrator for mid-loop skill escalation.

        Args:
            skills: List of innate skill names (e.g. ``['recall', 'schedule']``).

        Returns:
            str: Formatted skill documentation block, or empty string.
        """
        return self._get_injected_skills(skills)

    def _get_injected_skills(self, skills: list) -> str:
        """Build skill documentation from ``TOOL_SCHEMA`` dicts on handler modules.

        Each skill handler module exposes a ``TOOL_SCHEMA`` dict with ``name``,
        ``description``, and ``input_schema``. This method formats them into a
        compact prompt-ready block.

        Args:
            skills: List of innate skill names (e.g.
                ``['recall', 'memorize', 'schedule']``).

        Returns:
            str: Formatted skill docs joined by double newlines, or ``''`` if empty.
        """
        if not skills:
            return ''
        from services.innate_skills import get_skill_handler
        import inspect
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
        """Format a ``TOOL_SCHEMA`` dict into a compact prompt-ready block.

        Args:
            schema: Tool schema dict containing at minimum ``name``,
                ``description``, and ``input_schema`` keys.

        Returns:
            str: Multi-line markdown-style skill documentation string.
        """
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
        """Return descriptors of registered external tools from the database.

        Returns:
            str: Comma-separated tool descriptors, or ``'(none registered)'``
            when the table is empty or the query fails.
        """
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

    def _get_methodology_guidance(self, request_text: str) -> str:
        """Retrieve methodology guidance from learned methodology records via KNN.

        Each methodology record was written by PostLoopReflectionService and already
        passed the outcome_quality >= 0.35 gate at write time — no further filtering
        needed. Among records within 0.05 of the best match distance, prefer the
        most recently created (it has the most compounded knowledge).

        Returns top-2 guidance paragraphs as a formatted section, or '' if none found.
        """
        try:
            from services.embedding_service import get_embedding_service
            from services.embedding_utils import pack_embedding
            from services.database_service import get_shared_db_service

            db = get_shared_db_service()
            emb_service = get_embedding_service()

            query_embedding = emb_service.generate_embedding(request_text)
            if query_embedding is None:
                return ''

            blob = pack_embedding(query_embedding)
            if not blob:
                return ''

            matches = []
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT v.rowid, v.distance
                    FROM knowledge_vec v
                    WHERE v.embedding MATCH ? AND k = ?
                    ORDER BY v.distance
                """, (blob, 8))
                vec_results = cursor.fetchall()

                candidates = []  # (distance, created_at, value)
                for rid, distance in vec_results:
                    cursor.execute("""
                        SELECT value, created_at
                        FROM knowledge
                        WHERE rowid = ? AND entity = 'methodology' AND kind = 'procedure'
                          AND deleted_at IS NULL AND value != ''
                    """, (rid,))
                    row = cursor.fetchone()
                    if row:
                        candidates.append((distance, row[1] or '', row[0]))

                # Deduplicate into top-2: for each group of records within 0.05 of
                # each other, keep only the most recent. Then take the top-2 groups.
                selected = []
                for dist, created_at, value in candidates:
                    if not selected:
                        selected.append((dist, created_at, value))
                        continue
                    last_dist = selected[0][0]
                    if dist <= last_dist + 0.05:
                        # Same cluster — replace with this one if it's newer
                        if created_at > selected[-1][1]:
                            selected[-1] = (dist, created_at, value)
                    else:
                        selected.append((dist, created_at, value))
                    if len(selected) >= 2:
                        break

                matches = [v for _, _, v in selected]
                cursor.close()

            if not matches:
                return ''

            parts = ["## Methodology Guidance"]
            parts.extend(matches)
            return '\n\n'.join(parts)

        except Exception:
            return ''

    def _get_goal_context(self, limit: int = 3) -> str:
        """Build compact goal stack for prompt injection (~150 tokens).

        Args:
            limit: Maximum number of active goals to include.

        Returns:
            str: Formatted ``## Active Goals`` section string, or empty string
            when no goals are active or the service is unavailable.
        """
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

