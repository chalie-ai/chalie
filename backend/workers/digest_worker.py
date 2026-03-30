# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Digest Worker — Fast-path message processing pipeline for the assistant.

This module implements the primary worker function (``digest_worker``) that
consumes items from the ``prompt-queue`` and orchestrates the full LLM
response cycle: topic classification, context assembly, unified generation
(single LLM call with discoverable skills/tools), memory chunking,
and output delivery.

It also provides lightweight helpers used by cron-tool and other workers for
interactive tool dialog (``process_tool_dialog``, ``store_tool_dialog_memory``)
and a lazy singleton accessor for the shared ``ContextAssemblyService``
(``get_context_assembly_service``).
"""

import json
import re
import time
import logging

logger = logging.getLogger(__name__)
from services import ConfigService, FrontalCortexService, OrchestratorService, SessionService
from services.llm_service import create_llm_service
from services.world_state_service import WorldStateService
from services.working_memory_service import WorkingMemoryService
from services.interaction_log_service import InteractionLogService
from services.event_bus_service import EventBusService
from services.metrics_service import MetricsService
from services.mode_router_service import ModeRouterService, collect_routing_signals, compute_nlp_signals
from services.intent_classifier_service import IntentClassifierService
from services.thread_service import get_thread_service
from services.thread_conversation_service import ThreadConversationService
from services.context_relevance_service import ContextRelevanceService
from services.context_assembly_service import ContextAssemblyService
from services.innate_skills.registry import ALL_SKILL_NAMES

# Global session service instance (shared across worker invocations)
_session_service = None

# Global mode router instance (shared across invocations)
_mode_router = None

# Global intent classifier instance
_intent_classifier = None

# Global orchestrator instance
_orchestrator = None

# Global thread conversation service
_thread_conv_service = None

# Global context relevance service
_context_relevance_service = None


def _resolve_image_contexts(image_ids: list, timeout: int = 30) -> list:
    """Resolve image IDs to analysis results from MemoryStore.

    Polls with 1-second backoff up to *timeout* seconds per image.  The WS
    handler passes ``image_ids`` through in metadata without blocking; this
    function is where we actually wait for the vision analysis background thread
    to finish before the LLM calls need the visual context.

    Args:
        image_ids: List of image ID strings (max 3, enforced by WS handler).
        timeout: Maximum seconds to wait per image before giving up.

    Returns:
        List of image context dicts (``description``, ``ocr_text``, etc.).
        Images that time out or fail to parse are silently skipped.
    """
    if not image_ids:
        return []
    from services.memory_client import MemoryClientService
    store = MemoryClientService.create_connection()
    contexts = []
    for img_id in image_ids:
        key = f'chat_image_result:{img_id}'
        deadline = time.time() + timeout
        while time.time() < deadline:
            raw = store.get(key)
            if raw:
                try:
                    contexts.append(json.loads(raw))
                except json.JSONDecodeError as e:
                    logger.debug(f"[DIGEST] Failed to parse image context JSON for {img_id!r}: {e}", exc_info=True)
                break
            time.sleep(1)
        else:
            logging.debug(f"[DIGEST] Image resolution timed out for {img_id!r} after {timeout}s")
    return contexts


def get_context_relevance_service():
    """Get or create global ContextRelevanceService instance."""
    global _context_relevance_service
    if _context_relevance_service is None:
        _context_relevance_service = ContextRelevanceService()
    return _context_relevance_service


_context_assembly_service = None


def get_context_assembly_service():
    """Return the module-level singleton ``ContextAssemblyService`` instance.

    The service is created lazily on first access and reused for the lifetime
    of the worker process, avoiding repeated initialisation overhead across
    queue items.

    Returns:
        ContextAssemblyService: Shared context assembly service instance
            initialised with an empty configuration override dict.
    """
    global _context_assembly_service
    if _context_assembly_service is None:
        _context_assembly_service = ContextAssemblyService({})
    return _context_assembly_service


def get_orchestrator():
    """Get or create global OrchestratorService instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorService()
    return _orchestrator


def get_thread_conv_service() -> ThreadConversationService:
    """Get or create global ThreadConversationService instance."""
    global _thread_conv_service
    if _thread_conv_service is None:
        _thread_conv_service = ThreadConversationService()
    return _thread_conv_service


def get_session_service():
    """Get or create global session service instance."""
    global _session_service
    if _session_service is None:
        episodic_config = ConfigService.resolve_agent_config("episodic-memory")
        inactivity_timeout = episodic_config.get('inactivity_timeout', 600)
        _session_service = SessionService(inactivity_timeout=inactivity_timeout)
    return _session_service


def get_intent_classifier():
    """Get or create global intent classifier instance."""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifierService()
    return _intent_classifier


def get_mode_router():
    """Get or create global mode router instance."""
    global _mode_router
    if _mode_router is None:
        import os
        # Prefer generated config (from stability regulator) over base config
        generated_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs", "generated", "mode_router_config.json"
        )
        if os.path.exists(generated_path):
            try:
                with open(generated_path, 'r') as f:
                    router_config = json.load(f)
                logging.info("[DIGEST] Loaded generated mode router config")
            except Exception as e:
                logger.debug(f"[DIGEST] Failed to load generated mode router config, using default: {e}")
                router_config = ConfigService.get_agent_config("mode-router")
        else:
            router_config = ConfigService.get_agent_config("mode-router")
        _mode_router = ModeRouterService(router_config)
    return _mode_router


def enqueue_trait_extraction(prompt_message: str, metadata: dict = None, thread_id: str = None):
    """Enqueue lightweight trait extraction for a user message.

    Messages over 300 chars are skipped entirely — long messages are almost
    certainly pasted content (transcripts, plans, articles, code) and the
    introductory framing is where third-party names appear.
    """
    try:
        import threading

        if len(prompt_message) > 300:
            logging.debug("[TraitExtraction] Skipping — message too long (%d chars), likely pasted content", len(prompt_message))
            return

        def _extract_traits():
            try:
                import json
                from services.knowledge_service import KnowledgeService
                from services.database_service import get_shared_db_service
                from services.provider_cache_service import ProviderCacheService

                # ONNX gate: skip LLM call if classifier says no trait present
                try:
                    from services.onnx_inference_service import get_onnx_inference_service
                    onnx_svc = get_onnx_inference_service()
                    if onnx_svc.is_available("trait-detector"):
                        gate_input = f"{prompt_message}\nTrait:"
                        label, confidence = onnx_svc.predict("trait-detector", gate_input)
                        if label == "false" and confidence >= 0.85:
                            logging.debug(
                                f"[TRAIT_EXTRACT] ONNX gate: no trait detected "
                                f"(confidence={confidence:.3f}), skipping LLM"
                            )
                            return
                except Exception as e:
                    logging.debug(f"[TRAIT_EXTRACT] ONNX gate unavailable: {e}")

                provider_config = ProviderCacheService.resolve_for_job('trait-extraction')
                if not provider_config:
                    return

                prompt_text = _load_trait_prompt(prompt_message)
                if not prompt_text:
                    return

                llm = create_llm_service(provider_config)
                llm_resp = llm.send_message(
                    prompt_text,
                    "Extract traits as JSON. Values must be clean noun phrases — the entity itself, no pronouns, articles, or conjunctions. Return only valid JSON."
                )
                result = llm_resp.text if llm_resp else None
                if not result:
                    return

                # Strip markdown fences if present
                import re
                fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', result, re.DOTALL)
                if fence_match:
                    result = fence_match.group(1).strip()

                parsed = json.loads(result)
                traits = parsed.get('traits', [])

                CONFIDENCE_MAP = {'high': 0.85, 'medium': 0.55, 'low': 0.35}
                CORE_KEYS = {
                    'name', 'age', 'gender', 'occupation', 'nationality', 'language',
                    'education', 'culture_region', 'language_preference',
                    'relationship_status', 'ethnicity', 'birthday', 'location',
                }

                db = get_shared_db_service()
                ks = KnowledgeService(db)

                # Stop words to strip from trait values (LLMs sometimes
                # include pronouns/conjunctions like "alex and i")
                _VALUE_STRIP = {
                    'i', 'me', 'my', 'and', 'or', 'the', 'a', 'an', 'is',
                    'am', 'are', 'was', 'it', 'to', 'in', 'of', 'for',
                }

                def _clean_value(v: str) -> str:
                    """Strip leading/trailing stop words from an extracted value."""
                    words = v.split()
                    # Trim stop words from both ends
                    while words and words[0].lower() in _VALUE_STRIP:
                        words.pop(0)
                    while words and words[-1].lower() in _VALUE_STRIP:
                        words.pop()
                    return ' '.join(words) if words else v

                _DECAY_MAP = {'core': 'permanent', 'behavioral': 'slow'}

                for trait in traits:
                    key = trait.get('key', '').lower().strip()
                    value = _clean_value(trait.get('value', '').strip())
                    conf_label = trait.get('confidence', 'low')

                    if not key or not value:
                        continue

                    confidence = CONFIDENCE_MAP.get(conf_label, 0.35)
                    category = 'core' if key in CORE_KEYS else 'preference'

                    ks.store(
                        kind='trait', entity='user', key=key, value=value,
                        data={'category': category},
                        decay_class=_DECAY_MAP.get(category, 'standard'),
                        confidence=confidence, source='llm_extraction',
                    )

            except Exception as e:
                logging.debug(f"[TRAIT_EXTRACT] Failed: {e}")

        def _load_trait_prompt(message: str) -> str:
            try:
                import os
                prompt_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), 'prompts', 'trait-extraction.md'
                )
                with open(prompt_path, 'r') as f:
                    template = f.read()
                return template.replace('{{message}}', message)
            except Exception as e:
                logger.debug(f"[TRAIT_EXTRACT] Failed to load trait prompt template: {e}", exc_info=True)
                return None

        t = threading.Thread(target=_extract_traits, daemon=True)
        t.start()

    except Exception as e:
        logging.debug(f"[TRAIT_EXTRACT] Enqueue failed: {e}")


def load_configs():
    """Load frontal cortex mode-specific prompts and configurations."""
    soul_prompt = ConfigService.get_agent_prompt("soul")
    identity_prompt = ConfigService.get_agent_prompt("identity-core")
    cortex_config = ConfigService.resolve_agent_config("frontal-cortex")

    # Mode-specific prompts: soul → identity → mode prompt (instincts + context + contract)
    # Ordering: values first, then voice, then behavioral nudges closest to generation
    # ACT does NOT get identity — reasoning stays pure
    act_prompt = ConfigService.get_agent_prompt("frontal-cortex-act")
    unified_prompt = soul_prompt + "\n\n" + identity_prompt + "\n\n" + ConfigService.get_agent_prompt("frontal-cortex-unified")

    return {
        'cortex': {
            'config': cortex_config,
            'prompt_map': {
                'ACT': act_prompt,
                'UNIFIED': unified_prompt,
            }
        },
    }


def calculate_context_warmth(working_memory_len: int, world_state_nonempty: bool, gists: list = None) -> float:
    """
    Calculate context warmth signal (0.0-1.0) for scaling uncertainty cost.
    """
    wm_score = min(working_memory_len / 4, 1.0)
    world_score = 1.0 if world_state_nonempty else 0.0
    warmth = (wm_score + world_score) / 2
    return warmth


def _format_visual_context(image_contexts: list) -> str:
    """Format image analysis results as a markdown section for the prompt."""
    parts = []
    for i, ctx in enumerate(image_contexts, 1):
        label = f"Image {i}" if len(image_contexts) > 1 else "Attached image"
        desc = ctx.get('description', '').strip()
        ocr = ctx.get('ocr_text', '').strip()
        part = f"**{label}:** {desc}" if desc else f"**{label}:** (no description)"
        if ocr:
            part += f"\n> Extracted text: {ocr[:500]}"
        parts.append(part)
    return "\n\n".join(parts)


def unified_generate(topic, text, classification, thread_conv_service,
                     cortex_config, cortex_prompt_map, signals,
                     metadata=None, thread_id=None,
                     returning_from_silence=False, message_embedding=None,
                     topic_context=None):
    """
    Unified generation — single LLM call with all innate skills available.

    The LLM decides whether to respond directly (Format B) or invoke
    skills/tools first (Format A). No mode routing.

    Returns:
        tuple: (response_data dict, routing_result dict)
    """
    from services.config_service import ConfigService

    prompt = cortex_prompt_map['UNIFIED']

    # Load unified config
    try:
        config = ConfigService.resolve_agent_config("frontal-cortex-unified")
        logging.info(f"[UNIFIED] Using config frontal-cortex-unified.json with model {config.get('model')}")
    except Exception as e:
        logging.warning(f"[UNIFIED] Config frontal-cortex-unified.json not found, using base config: {e}")
        config = cortex_config

    # Compute context inclusion map
    inclusion_map = None
    try:
        context_relevance_service = get_context_relevance_service()
        inclusion_map = context_relevance_service.compute_inclusion_map(
            mode='UNIFIED',
            signals=signals or {},
            classification=classification,
            returning_from_silence=returning_from_silence,
        )
    except Exception as e:
        logging.warning(f"[UNIFIED] Context relevance computation failed: {e}, proceeding without inclusion_map")

    # Assemble context
    assembled_context = None
    try:
        assembled_context = get_context_assembly_service().assemble(
            prompt=text,
            topic=topic,
            thread_id=thread_id,
            message_embedding=message_embedding,
            context=topic_context,
        )
        if topic_context and topic_context.failed_sections:
            logging.warning(
                f"[UNIFIED] Context assembly had failures: {topic_context.failed_sections}"
            )
    except Exception as e:
        logging.warning(f"[UNIFIED] Context assembly failed: {e}")

    # Contradiction detection — always run for unified (every message may be conversational)
    if assembled_context is not None:
        try:
            from services.contradiction_classifier_service import ContradictionClassifierService
            from services.uncertainty_service import UncertaintyService
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            classifier = ContradictionClassifierService(db_service=db)
            conflict = classifier.check_ingestion(text)
            if conflict:
                conflict_type = conflict.get('classification')
                mem_a = conflict.get('memory_a', {})
                mem_b = conflict.get('memory_b', {})
                unc_svc = UncertaintyService(db)

                if conflict_type == 'temporal_change' and conflict.get('temporal_signal'):
                    if mem_b.get('id') and not classifier.pair_already_tracked('incoming', mem_b['id']):
                        unc_id = unc_svc.create_uncertainty(
                            memory_a_type=mem_b['type'],
                            memory_a_id=mem_b['id'],
                            uncertainty_type='contradiction',
                            detection_context='ingestion',
                            reasoning=conflict.get('reasoning'),
                            temporal_signal=True,
                        )
                        unc_svc.resolve_uncertainty(
                            uncertainty_id=unc_id,
                            strategy='temporal_supersede',
                            detail=conflict.get('reasoning', ''),
                        )
                elif conflict_type in ('true_contradiction', 'context_dependent'):
                    assembled_context['contradiction_context'] = {
                        'classification': conflict_type,
                        'memory_a_text': mem_a.get('text', ''),
                        'memory_b_text': mem_b.get('text', ''),
                        'reasoning': conflict.get('reasoning', ''),
                        'surface_context': conflict.get('surface_context', ''),
                    }
                    if mem_b.get('id') and not classifier.pair_already_tracked('incoming', mem_b['id']):
                        unc_svc.create_uncertainty(
                            memory_a_type=mem_b['type'],
                            memory_a_id=mem_b['id'],
                            uncertainty_type='contradiction',
                            detection_context='ingestion',
                            reasoning=conflict.get('reasoning'),
                            temporal_signal=False,
                            surface_context=conflict.get('surface_context'),
                        )
        except Exception as e:
            logging.debug(f"[UNIFIED] Ingestion contradiction check skipped: {e}")

    # Inject visual context from attached images
    image_contexts = (metadata or {}).get('image_contexts', [])
    if image_contexts:
        if assembled_context is None:
            assembled_context = {}
        assembled_context['visual_context'] = _format_visual_context(image_contexts)

    # Propagate message embedding for WorldStateService semantic scoring
    if message_embedding is not None:
        if assembled_context is None:
            assembled_context = {}
        assembled_context['message_embedding'] = message_embedding

    cortex_service = FrontalCortexService(config)
    chat_history = thread_conv_service.get_conversation_history(thread_id) if thread_id else []

    # Build system prompt and native tool schemas for the first call
    all_skills = list(ALL_SKILL_NAMES)
    # Voice mode: exclude visual-only skills (rich_render outputs blocks unusable via TTS)
    if (metadata or {}).get('source') == 'voice':
        all_skills = [s for s in all_skills if s != 'rich_render']
    from services.tool_schema_service import get_skill_schemas
    native_tools = get_skill_schemas(all_skills)

    system_prompt = cortex_service.build_system_prompt(
        system_prompt_template=prompt,
        original_prompt=text,
        classification=classification,
        chat_history=chat_history,
        assembled_context=assembled_context,
        selected_skills=all_skills,
        thread_id=thread_id,
        returning_from_silence=returning_from_silence,
        inclusion_map=inclusion_map,
    )

    # Voice mode: inject plain-text-only instruction for TTS-friendly responses
    if (metadata or {}).get('source') == 'voice':
        system_prompt = system_prompt.replace('{{voice_mode_instruction}}',
            '\n\nIMPORTANT: The user is in voice mode. Your response will be spoken aloud via TTS. '
            'Respond in plain conversational text only. No markdown formatting, code blocks, '
            'tables, bullet lists, links, or structured formatting. Write as you would speak.')
    else:
        system_prompt = system_prompt.replace('{{voice_mode_instruction}}', '')

    first_messages = [{"role": "user", "content": text}]

    # First LLM call with native tools — model either responds with text or calls tools
    first_response = cortex_service.generate_response_appended(
        system_prompt=system_prompt,
        messages=first_messages,
        cache_prefix=True,
        tools=native_tools,
    )

    routing_result = {
        'mode': 'UNIFIED',
        'router_confidence': 1.0,
        'routing_source': 'unified',
        'routing_time_ms': 0.0,
        'scores': {'UNIFIED': 1.0},
    }

    # Check if the model wants to invoke skills/tools (tool_calls or legacy actions)
    if not first_response.get('actions'):
        # Direct response — no ACT loop needed
        first_response['mode'] = 'UNIFIED'
        if not first_response.get('response', '').strip():
            first_response['response'] = "I understand. Let me think about that."
        return first_response, routing_result

    # Format A — enter ACT loop with unified prompt
    from services.act_orchestrator_service import ACTOrchestrator

    act_cumulative_timeout = config.get('act_cumulative_timeout', cortex_config.get('act_cumulative_timeout', 60.0))
    act_per_action_timeout = config.get('act_per_action_timeout', cortex_config.get('act_per_action_timeout', 10.0))
    max_act_iterations = config.get('max_act_iterations', cortex_config.get('max_act_iterations', 5))

    # Reuse the inclusion_map and assembled_context from the initial call —
    # UNIFIED mask is already the union of all modes, no need to recompute

    orchestrator = ACTOrchestrator(
        config=cortex_config,
        max_iterations=max_act_iterations,
        cumulative_timeout=act_cumulative_timeout,
        per_action_timeout=act_per_action_timeout,
        critic_enabled=True,
        smart_repetition=True,
        escalation_hints=True,
        persistent_task_exit=True,
        execution_gate=False,  # User explicitly requested this action — skip autonomous gate
    )

    # Narration callback: stream progress to user via WebSocket
    request_uuid = (metadata or {}).get('uuid', '')

    def _narration_callback(narration_text, step):
        """Publish narration to the per-request SSE channel."""
        logging.info(f"[UNIFIED] Narration callback fired: step={step}, text={narration_text[:60]}")
        if not request_uuid:
            logging.warning("[UNIFIED] Narration skipped — no request_uuid")
            return
        try:
            import json as _json
            from uuid import uuid4
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            narration_id = f"narr_{uuid4().hex[:12]}"
            store.set(f"output:{narration_id}", _json.dumps({
                'type': 'act_narration',
                'text': narration_text,
                'step': step,
            }), ex=300)
            store.publish(f"sse:{request_uuid}", narration_id)
            logging.info(f"[UNIFIED] Narration published: {narration_id} → sse:{request_uuid[:12]}...")

            # Emit show_narration intent for wrapper contract
            try:
                from services.intent_service import IntentService, CognitiveIntent, _make_intent_id
                intent = CognitiveIntent(
                    intent_id=_make_intent_id(),
                    intent_type='show_narration',
                    target_wrapper='__chat_ui__',
                    payload={
                        'request_id': request_uuid,
                        'text': narration_text,
                        'step': step,
                    },
                )
                IntentService().emit(intent)
            except Exception as e:
                logger.debug(f"[UNIFIED] Intent emit for narration failed (non-critical): {e}", exc_info=True)

        except Exception as _e:
            logging.error(f"[UNIFIED] Narration publish failed: {_e}", exc_info=True)

    result = orchestrator.run(
        topic=topic,
        text=text,
        cortex_service=cortex_service,
        act_prompt=prompt,  # Unified prompt (not ACT-specific)
        classification=classification,
        chat_history=chat_history,
        relevant_tools=None,
        selected_skills=list(ALL_SKILL_NAMES),
        selected_tools=None,
        assembled_context=assembled_context,
        inclusion_map=inclusion_map,
        on_narration=_narration_callback,
        session_id='digest_unified',
        exchange_id=(metadata or {}).get('exchange_id', 'unknown'),
        request_id=request_uuid,
    )

    logging.info(
        f"[UNIFIED] ACT loop complete: {len(result.act_history)} actions, "
        f"response={len(result.final_response)} chars"
    )

    # Card-only detection
    _history = result.act_history

    def _is_card_result(r):
        rt = r.get('result')
        return rt == '__CARD_ONLY__' or (isinstance(rt, str) and rt.startswith('__CARD_EMITTED__\n'))

    _all_card_only = (
        bool(_history)
        and all(r.get('status') == 'success' for r in _history)
        and all(_is_card_result(r) for r in _history)
    )

    _has_card_text = any(
        isinstance(r.get('result'), str) and r['result'].startswith('__CARD_EMITTED__\n')
        for r in _history
    ) if _all_card_only else False

    if _all_card_only and not _has_card_text:
        logging.info(
            "[UNIFIED] All actions emitted cards (no text) — skipping text response"
        )
        terminal_response = {
            'mode': 'IGNORE',
            'modifiers': [],
            'response': '',
            'generation_time': 0.0,
            'actions': None,
            'confidence': 1.0,
        }
    else:
        # Use the model's final response from the ACT loop — it already has full
        # context from tool results via native tool calling. No synthesis LLM call needed.
        terminal_response = {
            'mode': 'UNIFIED',
            'modifiers': [],
            'response': result.final_response or "I understand. Let me think about that.",
            'generation_time': 0.0,
            'actions': None,
            'confidence': 0.8,
        }

    # Enqueue tool reflection
    try:
        from services.act_reflection_service import enqueue_tool_reflection
        enqueue_tool_reflection(result.act_history, topic, text)
    except Exception as _e:
        logging.debug(f"[UNIFIED] Reflection enqueue skipped: {_e}")

    # Carry over action history
    terminal_response['actions'] = [
        {'type': r['action_type'], 'status': r['status'], 'result': r['result']}
        for r in result.act_history
    ] if result.act_history else None

    # Extract reply_actions from the last successful skill result (for UI buttons)
    for entry in reversed(result.act_history):
        if entry.get('status') == 'success' and entry.get('reply_actions'):
            terminal_response['reply_actions'] = entry['reply_actions']
            break

    routing_result['mode'] = terminal_response.get('mode', 'UNIFIED')
    return terminal_response, routing_result


def route_and_generate(topic, text, classification, thread_conv_service, cortex_config, cortex_prompt_map,
                       mode_router, signals, metadata=None, context_warmth=1.0,
                       _pre_routing_result=None, relevant_tools=None, selected_tools=None,
                       selected_skills=None, thread_id=None, returning_from_silence=False,
                       message_embedding=None):
    """
    Thin wrapper around unified_generate() — kept for backward compatibility.

    Called by process_tool_dialog() and _handle_proactive_drift().
    The mode_router and _pre_routing_result parameters are accepted but unused —
    unified_generate() handles routing internally.

    Returns:
        tuple: (response_data dict, routing_result dict)
    """
    # Trait extraction — runs in background thread, non-blocking
    enqueue_trait_extraction(
        prompt_message=text[:1000],
        metadata={'source': 'chat'},
        thread_id=thread_id,
    )

    response_data, routing_result = unified_generate(
        topic=topic,
        text=text,
        classification=classification,
        thread_conv_service=thread_conv_service,
        cortex_config=cortex_config,
        cortex_prompt_map=cortex_prompt_map,
        signals=signals,
        metadata=metadata,
        thread_id=thread_id,
        returning_from_silence=returning_from_silence,
        message_embedding=message_embedding,
    )

    # Store response
    thread_conv_service.add_response(
        thread_id,
        response_data['response'],
        response_data['generation_time']
    )

    # Route through orchestrator
    if metadata:
        try:
            orchestrator = get_orchestrator()

            context = {
                'topic': topic,
                'response': response_data.get('response', ''),
                'confidence': response_data.get('confidence', 0.0),
                'generation_time': response_data.get('generation_time', 0.0),
                'destination': metadata.get('destination', 'web'),
                'metadata': metadata,
                'actions': response_data.get('actions', []),
                'clarification_question': None,
                'reply_actions': response_data.get('reply_actions'),
            }

            mode = response_data.get('mode', 'UNIFIED')
            logging.info(f"[FRONTAL CORTEX] Routing through orchestrator: {mode}")

            orchestrator_result = orchestrator.route_path(mode=mode, context=context)

            if orchestrator_result['status'] == 'error':
                logging.error(f"[ORCHESTRATOR] Error: {orchestrator_result['message']}")
                try:
                    from services.output_service import OutputService
                    OutputService().enqueue_text(
                        topic=topic,
                        response=response_data.get('response') or "I ran into an issue. Please try again.",
                        mode='UNIFIED',
                        confidence=response_data.get('confidence', 0.0),
                        generation_time=response_data.get('generation_time', 0.0),
                        original_metadata=metadata,
                    )
                except Exception as oe:
                    logging.warning(f"[ORCHESTRATOR] Failed to surface error to user: {oe}")
            else:
                logging.info(f"[ORCHESTRATOR] Executed {mode}: {orchestrator_result.get('result', {})}")
        except Exception as e:
            logging.error(f"[ORCHESTRATOR] Failed: {e}")

    # Signal completion for IGNORE/card-only mode on sync WebSocket channels.
    if response_data.get('mode') == 'IGNORE' and metadata and metadata.get('uuid'):
        try:
            from services.output_service import OutputService
            OutputService().enqueue_text(
                topic=topic,
                response='',
                mode='ACT',
                confidence=response_data.get('confidence', 1.0),
                generation_time=response_data.get('generation_time', 0.0),
                original_metadata=metadata,
                reply_actions=response_data.get('reply_actions'),
            )
        except Exception as e:
            logging.warning(f"[IGNORE] Failed to publish empty-text message event: {e}")

    return response_data, routing_result


def process_tool_dialog(text: str, tool_name: str, trigger_prompt: str) -> str:
    """
    Process tool data through Chalie's full cognitive pipeline (including ACT loop).

    Called synchronously during an interactive tool↔Chalie dialog. Returns response text
    to be written back to the tool container's stdin. Does NOT surface to the user.

    Memory is stored with memory_durability='tool_internal' — weaker than cron_tool,
    so the user can ask "did that tool ask you something?" but the tool dialog doesn't
    alter long-term behavioral patterns.

    Args:
        text: Tool data text (prefixed by trigger_prompt from caller)
        tool_name: Tool name for memory tagging
        trigger_prompt: Tool's trigger.prompt from manifest (system context for Chalie)

    Returns:
        Chalie's response text (not surfaced to user).
    """
    try:
        configs = load_configs()
        cortex_config = configs['cortex']['config']
        cortex_prompt_map = configs['cortex']['prompt_map']

        topic = f'tool_dialog:{tool_name}'

        thread_service = get_thread_service()
        resolution = thread_service.resolve_thread('default',f'tool_dialog:{tool_name}')
        thread_id = resolution.thread_id

        thread_conv_service = get_thread_conv_service()

        classification = {
            'topic': topic,
            'confidence': 10,
            'similar_topic': '',
            'topic_update': '',
        }

        # Build minimal routing signals — tool dialogs don't need full signal collection
        signals = {'_prompt_text': text}

        try:
            from services.topic_context import TopicContext
            _tool_ctx = TopicContext(topic=topic, thread_id=thread_id)
            get_context_assembly_service().assemble(
                prompt=text, topic=topic, thread_id=thread_id,
                context=_tool_ctx,
            )
            if _tool_ctx.failed_sections:
                logging.warning(f"[TOOL DIALOG] Context assembly had failures: {_tool_ctx.failed_sections}")
        except Exception as e:
            logging.warning(f"[TOOL DIALOG] Context assembly failed for '{tool_name}': {e}")

        mode_router = get_mode_router()

        # Route through full pipeline (may engage ACT loop if mode router selects ACT)
        # metadata=None means orchestrator does NOT deliver to user
        response_data, _ = route_and_generate(
            topic=topic,
            text=text,
            classification=classification,
            thread_conv_service=thread_conv_service,
            cortex_config=cortex_config,
            cortex_prompt_map=cortex_prompt_map,
            mode_router=mode_router,
            signals=signals,
            metadata=None,
            context_warmth=0.5,
            thread_id=thread_id,
        )

        response = response_data.get('response', '')

        # Store response in conversation history (weak signal)
        if thread_id:
            thread_conv_service.add_response(thread_id, response, response_data.get('generation_time', 0.0))

        logging.info(
            f"[TOOL DIALOG] '{tool_name}' processed: mode={response_data.get('mode')} "
            f"({response_data.get('generation_time', 0):.2f}s)"
        )
        return response

    except Exception as e:
        logging.error(f"[TOOL DIALOG] Failed for '{tool_name}': {e}")
        return f"(analysis unavailable: {str(e)[:100]})"


def store_tool_dialog_memory(tool_name: str, turns: list):
    """
    Store final-turn-only memory after interactive tool dialog completes.

    Stores ONE memory entry regardless of dialog length:
    - 1-2 turns: full exchange
    - >2 turns: first request + final response (summary)

    Tagged with memory_durability='tool_internal' for weak persistence.
    """
    if not turns:
        return

    # Build compact exchange summary
    if len(turns) <= 2:
        prompt_msg = turns[0].get('request', '')
    else:
        prompt_msg = turns[0].get('request', '') + f'\n[{len(turns) - 2} intermediate turns omitted]'

    try:
        enqueue_trait_extraction(
            prompt_message=prompt_msg[:1000],
            metadata={
                'source': f'tool_dialog:{tool_name}',
                'tool_name': tool_name,
            },
        )
    except Exception as e:
        logging.warning(f"[TOOL DIALOG] Memory storage failed for '{tool_name}': {e}")


def _handle_cron_tool_result(text: str, metadata: dict) -> str:
    """
    Pipeline for scheduled (cron) tool results.

    Goes directly to response generation (no mode routing or user input logging).
    Tool has already formatted the prompt with its data.
    Enqueues trait extraction with memory_durability: 'cron_tool' for 3x decay.
    """
    try:
        from services.config_service import ConfigService

        configs = load_configs()
        cortex_config = configs['cortex']['config']

        tool_name = metadata.get('tool_name', 'unknown')
        priority = metadata.get('priority', 'normal')
        destination = metadata.get('destination', 'web')

        # Resolve thread
        thread_service = get_thread_service()
        thread_id = metadata.get('thread_id')
        if not thread_id:
            platform = metadata.get('source', 'cron_tool')
            resolution = thread_service.resolve_thread('default',platform)
            thread_id = resolution.thread_id

        thread_conv_service = get_thread_conv_service()
        working_memory = WorkingMemoryService(
            max_turns=cortex_config.get('max_working_memory_turns', 10)
        )

        # Get recent chat history for context
        chat_history = thread_conv_service.get_conversation_history(thread_id) if thread_id else []

        # Load scheduled tool prompt template
        scheduled_tool_template = ConfigService.get_agent_prompt("frontal-cortex-scheduled-tool")

        try:
            scheduled_tool_config = ConfigService.resolve_agent_config("frontal-cortex-scheduled-tool")
        except Exception as e:
            logging.warning(f"[CRON TOOL] frontal-cortex-scheduled-tool.json not found, using frontal-cortex config: {e}")
            scheduled_tool_config = ConfigService.resolve_agent_config("frontal-cortex")

        _cron_classification = {
            'topic': f'cron_tool:{tool_name}',
            'confidence': 10,
            'similar_topic': '',
            'topic_update': '',
        }
        inclusion_map = None
        try:
            inclusion_map = get_context_relevance_service().compute_inclusion_map(
                mode='UNIFIED', signals={}, classification=_cron_classification,
            )
        except Exception as e:
            logging.warning(f"[CRON TOOL] Context relevance failed: {e}")

        assembled_context = None
        try:
            from services.topic_context import TopicContext
            _cron_ctx = TopicContext(topic=f'cron_tool:{tool_name}', thread_id=thread_id)
            assembled_context = get_context_assembly_service().assemble(
                prompt=text, topic=f'cron_tool:{tool_name}', thread_id=thread_id,
                context=_cron_ctx,
            )
            if _cron_ctx.failed_sections:
                logging.warning(f"[CRON TOOL] Context assembly had failures: {_cron_ctx.failed_sections}")
        except Exception as e:
            logging.warning(f"[CRON TOOL] Context assembly failed: {e}")

        cortex_service = FrontalCortexService(scheduled_tool_config)

        # Generate response using the scheduled tool prompt
        response_data = cortex_service.generate_response(
            system_prompt_template=scheduled_tool_template,
            original_prompt=text,
            classification=_cron_classification,
            chat_history=chat_history,
            thread_id=thread_id,
            inclusion_map=inclusion_map,
            assembled_context=assembled_context,
        )

        # Always UNIFIED mode for scheduled tools (bypass mode routing)
        response_data['mode'] = 'UNIFIED'

        if not response_data.get('response', '').strip():
            logging.info(f"[CRON TOOL] {tool_name}: Empty response generated — skipping delivery")
            return f"Tool '{tool_name}' | Mode: UNIFIED | Empty response (no updates)"

        # Append assistant turn to working memory
        working_memory.append_turn(thread_id or f'cron_tool:{tool_name}', 'assistant', response_data['response'])

        # Store response in conversation history
        if thread_id:
            thread_conv_service.add_response(
                thread_id, response_data['response'], response_data.get('generation_time', 0.0)
            )

        # Route through orchestrator for delivery
        try:
            orchestrator = get_orchestrator()

            context = {
                'topic': f'cron_tool:{tool_name}',
                'response': response_data['response'],
                'confidence': response_data.get('confidence', 0.5),
                'generation_time': response_data.get('generation_time', 0.0),
                'destination': destination,
                'metadata': metadata,
                'actions': [],
            }
            orchestrator.route_path(mode='UNIFIED', context=context)
        except Exception as e:
            logging.error(f"[CRON TOOL] {tool_name}: Orchestrator failed: {e}")

        try:
            enqueue_trait_extraction(
                prompt_message=text,
                metadata={
                    'source': f'cron_tool:{tool_name}',
                    'priority': priority,
                },
                thread_id=thread_id,
            )
        except Exception as e:
            logging.warning(f"[CRON TOOL] {tool_name}: Trait extraction enqueue failed: {e}")

        # Log the cron tool execution
        try:
            from services.database_service import get_shared_db_service
            from services.interaction_log_service import InteractionLogService
            db_service = get_shared_db_service()
            log_service = InteractionLogService(db_service)
            log_service.log_event(
                event_type='cron_tool_executed',
                payload={
                    'tool_name': tool_name,
                    'priority': priority,
                    'response': response_data['response'][:500],
                    'generation_time': response_data.get('generation_time', 0),
                },
                topic=f'cron_tool:{tool_name}',
                source='cron_tool',
                metadata=metadata,
            )
        except Exception as e:
            logger.debug(f"[CRON TOOL] Failed to log cron tool execution event: {e}", exc_info=True)

        logging.info(
            f"[CRON TOOL] {tool_name} delivered: priority={priority} "
            f"({response_data.get('generation_time', 0):.2f}s)"
        )

        return (
            f"Tool '{tool_name}' | Mode: UNIFIED | "
            f"Response generated in {response_data.get('generation_time', 0):.2f}s"
        )

    except Exception as e:
        logging.error(f"[CRON TOOL] Failed: {e}")
        tool_name = metadata.get('tool_name', 'unknown')
        return f"Tool '{tool_name}' | ERROR: cron_tool - {e}"


def _handle_proactive_drift(text: str, metadata: dict) -> str:
    """
    Pipeline for proactive drift messages (system-initiated outreach).

    Goes through full mode routing (router as final judge) but skips:
    - User input logging (no user input)
    - Topic classification (topic provided in metadata)
    - Reward evaluation (no previous exchange to evaluate)

    The mode router may select IGNORE for a weak thought — that's a feature.
    """
    configs = load_configs()
    cortex_config = configs['cortex']['config']

    topic = metadata.get('related_topic', 'general')
    drift_gist = metadata.get('drift_gist', text)
    drift_type = metadata.get('drift_type', 'reflection')
    proactive_id = metadata.get('proactive_id', '')
    destination = metadata.get('destination', 'web')

    # Resolve thread for proactive drift
    thread_conv_service = get_thread_conv_service()
    thread_id = metadata.get('thread_id')
    if not thread_id:
        platform = metadata.get('source', 'unknown')
        resolution = get_thread_service().resolve_thread('default',platform)
        thread_id = resolution.thread_id

    working_memory = WorkingMemoryService(
        max_turns=cortex_config.get('max_working_memory_turns', 10)
    )
    world_state_service = WorldStateService()

    # Build classification stub (topic is pre-determined)
    classification = {
        'topic': topic,
        'confidence': 10,
        'similar_topic': '',
        'topic_update': '',
    }

    # Collect routing signals for the mode router (full router run)
    mode_router = get_mode_router()
    context_warmth = 0.0
    try:
        wm_turns = working_memory.get_recent_turns(thread_id or topic)
        world_state = world_state_service.get_world_state(
            topic, thread_id=thread_id, message_embedding=None
        )
        context_warmth = calculate_context_warmth(
            working_memory_len=len(wm_turns),
            world_state_nonempty=bool(world_state)
        )
    except Exception as e:
        logger.debug(f"[PROACTIVE] Context warmth calculation failed: {e}", exc_info=True)

    # Collect signals for mode routing
    try:
        session_service = get_session_service()

        # The prompt to the router is the drift thought itself
        classification_result = {'confidence': 1.0, 'is_new_topic': False}

        signals = collect_routing_signals(
            text=drift_gist,
            topic=topic,
            context_warmth=context_warmth,
            working_memory=working_memory,
            world_state_service=world_state_service,
            classification_result=classification_result,
            session_service=session_service,
        )
    except Exception as e:
        logging.warning(f"[PROACTIVE] Signal collection failed: {e}")
        signals = {}

    # Route through mode router (unbiased — doesn't know this is proactive)
    try:
        routing_result = mode_router.route(signals, drift_gist)
        selected_mode = routing_result['mode']
        logging.info(
            f"[PROACTIVE] Router selected: {selected_mode} "
            f"(confidence={routing_result.get('router_confidence', 0):.3f})"
        )
    except Exception as e:
        logging.warning(f"[PROACTIVE] Routing failed, defaulting to UNIFIED: {e}")
        selected_mode = 'UNIFIED'
        routing_result = {'mode': 'UNIFIED', 'router_confidence': 0.5, 'routing_time_ms': 0.0}

    # If router says IGNORE, respect it — the thought wasn't worth sharing
    if selected_mode == 'IGNORE':
        logging.info("[PROACTIVE] Router selected IGNORE — thought filtered")
        # Record as router_ignored for circuit breaker
        try:
            from services.autonomous_actions.engagement_tracker import EngagementTracker
            tracker = EngagementTracker()
            tracker._update_engagement_state(proactive_id, 'router_ignored', -0.3)
        except Exception as e:
            logger.debug(f"[PROACTIVE] Failed to record router_ignored engagement state: {e}", exc_info=True)
        return f"Topic '{topic}' | Mode: PROACTIVE_IGNORED | Router filtered thought"

    # ACT mode doesn't make sense for proactive thoughts — fall back to UNIFIED
    if selected_mode == 'ACT':
        selected_mode = 'UNIFIED'

    # Generate proactive outreach using dedicated prompt template
    try:
        from services.config_service import ConfigService

        proactive_template = ConfigService.get_agent_prompt("frontal-cortex-proactive")

        try:
            proactive_config = ConfigService.resolve_agent_config("frontal-cortex-proactive")
        except Exception as e:
            logging.warning(f"[PROACTIVE] frontal-cortex-proactive.json not found, falling back to frontal-cortex config: {e}")
            proactive_config = ConfigService.resolve_agent_config("frontal-cortex")

        inclusion_map = None
        try:
            inclusion_map = get_context_relevance_service().compute_inclusion_map(
                mode='UNIFIED', signals={}, classification=classification,
            )
        except Exception as e:
            logging.warning(f"[PROACTIVE] Context relevance failed: {e}")

        assembled_context = None
        try:
            from services.topic_context import TopicContext
            _drift_ctx = TopicContext(topic=topic, thread_id=thread_id)
            assembled_context = get_context_assembly_service().assemble(
                prompt=drift_gist, topic=topic, thread_id=thread_id,
                context=_drift_ctx,
            )
            if _drift_ctx.failed_sections:
                logging.warning(f"[PROACTIVE] Context assembly had failures: {_drift_ctx.failed_sections}")
        except Exception as e:
            logging.warning(f"[PROACTIVE] Context assembly failed: {e}")

        cortex_service = FrontalCortexService(proactive_config)
        chat_history = thread_conv_service.get_conversation_history(thread_id) if thread_id else []

        response_data = cortex_service.generate_response(
            system_prompt_template=proactive_template,
            original_prompt=drift_gist,
            classification=classification,
            chat_history=chat_history,
            thread_id=thread_id,
            inclusion_map=inclusion_map,
            assembled_context=assembled_context,
        )

        # Proactive messages always deliver as UNIFIED
        response_data['mode'] = 'UNIFIED'
        selected_mode = 'UNIFIED'

        if not response_data.get('response', '').strip():
            logging.info("[PROACTIVE] Empty response generated — skipping delivery")
            return f"Topic '{topic}' | Mode: PROACTIVE_EMPTY | No response generated"

        # Append assistant turn to working memory
        working_memory.append_turn(thread_id or topic, 'assistant', response_data['response'])

        # Store response in conversation history
        if thread_id:
            thread_conv_service.add_response(
                thread_id, response_data['response'], response_data['generation_time']
            )

        # Store goal_id in MemoryStore for engagement correlation
        goal_id = metadata.get('goal_id') or proactive_id
        if goal_id:
            try:
                from services.memory_client import MemoryClientService
                store = MemoryClientService.create_connection()
                store.setex(f"proactive_response_tag:{topic}", 14400, goal_id)
            except Exception as e:
                logger.debug(f"[PROACTIVE] Failed to store response tag in MemoryStore: {e}", exc_info=True)

        # Route through orchestrator for delivery
        try:
            orchestrator = get_orchestrator()

            context = {
                'topic': topic,
                'response': response_data['response'],
                'confidence': response_data.get('confidence', 0.5),
                'generation_time': response_data.get('generation_time', 0.0),
                'destination': destination,
                'metadata': metadata,
                'actions': [],
            }
            orchestrator.route_path(mode=selected_mode, context=context)
        except Exception as e:
            logging.error(f"[PROACTIVE] Orchestrator failed: {e}")

        # Log the proactive send
        try:
            from services.database_service import get_shared_db_service
            from services.interaction_log_service import InteractionLogService
            db_service = get_shared_db_service()
            log_service = InteractionLogService(db_service)
            log_service.log_event(
                event_type='proactive_sent',
                payload={
                    'response': response_data['response'][:500],
                    'mode': selected_mode,
                    'drift_type': drift_type,
                    'proactive_id': proactive_id,
                    'router_confidence': routing_result.get('router_confidence', 0),
                },
                topic=topic,
                source='proactive_drift',
                metadata=metadata,
            )
        except Exception as e:
            logger.debug(f"[PROACTIVE] Failed to log proactive_sent event: {e}", exc_info=True)

        logging.info(
            f"[PROACTIVE] Delivered: [{drift_type}] → {selected_mode} "
            f"(proactive_id={proactive_id[:8] if proactive_id else '?'})"
        )

        return (
            f"Topic '{topic}' | Mode: PROACTIVE_{selected_mode} | "
            f"Response generated in {response_data.get('generation_time', 0):.2f}s"
        )

    except Exception as e:
        logging.error(f"[PROACTIVE] Failed: {e}")
        return f"Topic '{topic}' | ERROR: proactive - {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Immediate Identity Promotion (IIP)
#
# Deterministic regex patterns for detecting explicit name statements.
# Written synchronously (before any LLM call) to MemoryStore + SQLite so the name
# is available within the same request cycle. Target: <5ms. No LLM, no embeddings.
# ─────────────────────────────────────────────────────────────────────────────

# Capture group: one or two tokens, each allowing Unicode letters, apostrophes, hyphens.
# [^\W\d_] = any Unicode letter (standard re module — no external packages).
# Accepts any case — casing is normalised on write.
_IIP_NAME_CAPTURE = (
    r"([^\W\d_](?:[^\W\d_]|['\-]){0,39}"
    r"(?:\s+[^\W\d_](?:[^\W\d_]|['\-]){0,39})?)"
)

_IIP_PATTERNS = [
    re.compile(r"\bcall me\s+" + _IIP_NAME_CAPTURE, re.IGNORECASE),
    re.compile(r"\bmy name is\s+" + _IIP_NAME_CAPTURE, re.IGNORECASE),
    re.compile(r"\bi go by\s+" + _IIP_NAME_CAPTURE, re.IGNORECASE),
    re.compile(r"\byou can call me\s+" + _IIP_NAME_CAPTURE, re.IGNORECASE),
    re.compile(r"\bi'?m known as\s+" + _IIP_NAME_CAPTURE, re.IGNORECASE),
    re.compile(r"\brefer to me as\s+" + _IIP_NAME_CAPTURE, re.IGNORECASE),
]

_IIP_STOPWORDS = frozenset([
    'a', 'an', 'the', 'i', 'me', 'my', 'we', 'you', 'your', 'he', 'she',
    'they', 'it', 'this', 'that', 'here', 'there', 'done', 'fine', 'good',
    'okay', 'ok', 'sure', 'yes', 'no', 'maybe', 'later', 'anything',
    'something', 'nothing', 'everything',
])


def _run_iip_hook(text: str, database_service) -> None:
    """
    Detect explicit name statements and write to MemoryStore + SQLite synchronously.

    Deterministic regex only — no LLM, no embedding. Target: <5ms. Never raises.
    Preserves user's mixed-case input (McDonald, O'Brien); only title-cases
    when input is all-lowercase.
    """
    try:
        matched_name = None
        for pattern in _IIP_PATTERNS:
            m = pattern.search(text)
            if m:
                candidate = m.group(1).strip()
                # Reject stopwords (case-insensitive) and single-char matches
                if candidate.lower() not in _IIP_STOPWORDS and len(candidate) >= 2:
                    matched_name = candidate.title() if candidate.islower() else candidate
                    break

        if not matched_name:
            return

        from services.identity_state_service import IdentityStateService
        IdentityStateService().set_field(
            'name', matched_name, confidence=0.95, provisional=False
        )

        from services.knowledge_service import KnowledgeService
        KnowledgeService(database_service).store(
            kind='trait', entity='user', key='name', value=matched_name,
            data={'category': 'core'},
            decay_class='permanent', confidence=0.95, source='iip_hook',
        )
        logging.info(f"[IIP] Promoted name='{matched_name}' → MemoryStore + SQLite")

    except Exception as e:
        logging.warning(f"[IIP] Hook failed (non-fatal): {e}")


# Belief correction patterns — detect explicit trait corrections/negations
_BELIEF_CORRECTION_PATTERNS = [
    # Direct negation: "I don't like X", "I'm not a Y"
    re.compile(r"\b(?:I\s+(?:don'?t|do\s+not|never)\s+(?:like|enjoy|want|eat|drink|use|have|prefer|need))\s+(.+)", re.IGNORECASE),
    re.compile(r"\b(?:I'?m\s+not\s+(?:a\s+)?|I\s+am\s+not\s+(?:a\s+)?)(.+)", re.IGNORECASE),

    # Explicit correction: "actually my X is Y", "my name is actually Y"
    re.compile(r"\b(?:actually,?\s+)?my\s+(\w+(?:\s+\w+)?)\s+is\s+(?:actually\s+)?(.+)", re.IGNORECASE),

    # Belief correction: "that's wrong about me", "you're wrong about"
    re.compile(r"\b(?:that'?s\s+(?:wrong|incorrect|not\s+(?:true|right|correct))\s+(?:about\s+me|about\s+that))", re.IGNORECASE),
    re.compile(r"\b(?:you(?:'re|\s+are)\s+wrong\s+about)", re.IGNORECASE),

    # Retraction: "I never said I liked X", "I didn't say"
    re.compile(r"\b(?:I\s+never\s+said|I\s+didn'?t\s+say|I\s+didn'?t\s+tell\s+you)", re.IGNORECASE),

    # Stop assuming: "stop assuming", "don't assume"
    re.compile(r"\b(?:(?:stop|don'?t)\s+(?:assuming|thinking)\s+(?:I|that\s+I))", re.IGNORECASE),
]


def _run_belief_correction_hook(text: str, thread_id: str = None):
    """
    Detect explicit belief corrections and update/delete traits.
    Runs synchronously in Phase A before LLM trait injection.
    Precision-first: better to miss a correction than delete the wrong trait.
    """
    if not any(p.search(text) for p in _BELIEF_CORRECTION_PATTERNS):
        return

    text_lower = text.lower()

    # GUARDRAIL 1: Require explicit self-reference before any mutation
    # Prevents "sushi is terrible" from deleting a food preference
    if not re.search(r"\b(i|me|my|about me)\b", text_lower):
        return

    try:
        from services.knowledge_service import KnowledgeService
        from services.database_service import get_shared_db_service
        ks = KnowledgeService(get_shared_db_service())

        traits = ks.get_by_kind('trait', entity='user', limit=100)
        if not traits:
            return

        for trait in traits:
            key = trait.get('key', '')
            value = trait.get('value', '')
            confidence = trait.get('confidence', 0)

            # GUARDRAIL 2: Skip low-confidence traits — don't churn noisy data
            if confidence < 0.4:
                continue

            # GUARDRAIL 3: Skip empty trait values — "" is substring of everything
            if not value or not value.strip():
                continue

            # Check if the user's message negates this specific trait value
            escaped_value = re.escape(value.lower())
            if value.lower() in text_lower:
                negation_near_value = re.search(
                    rf"\b(?:not|don'?t|never|no longer|isn'?t|aren'?t|wasn'?t|wrong)\b.{{0,30}}\b{escaped_value}\b|"
                    rf"\b{escaped_value}\b.{{0,30}}\b(?:is wrong|is incorrect|is not right|isn'?t right)\b",
                    text_lower
                )
                if negation_near_value:
                    ks.forget('user', key)
                    logging.info(f"[BELIEF CORRECTION] Deleted trait '{key}={value}' — user negated it")
                    continue

            # Check for "actually my X is Y" pattern (value replacement)
            # Cap capture at 3 words to avoid trailing clauses
            replacement_match = re.search(
                rf"(?:actually,?\s+)?my\s+{re.escape(key.replace('_', ' '))}\s+is\s+(?:actually\s+)?(.+?)(?:\.|,|!|\?|$)",
                text_lower
            )
            if replacement_match:
                raw_value = replacement_match.group(1).strip()
                # Cap at 3 words to avoid trailing clause capture
                new_value = " ".join(raw_value.split()[:3])
                if new_value and new_value.lower() != value.lower():
                    ks.update('user', key, value=new_value)
                    logging.info(f"[BELIEF CORRECTION] Corrected trait '{key}': '{value}' → '{new_value}'")

    except Exception as e:
        logging.warning(f"[BELIEF CORRECTION] Hook failed (non-fatal): {e}")


def _classify_engagement(text: str) -> str:
    """
    Classify user engagement with a proactive message.
    Deterministic, no LLM. Pattern-based classification.
    Returns: engaged|acknowledged|rejected|ignored
    """
    import re
    text_lower = text.strip().lower()

    if not text_lower:
        return 'ignored'

    # Acknowledgment patterns (short, non-substantive) — check first
    ack = re.compile(
        r'^(ok|okay|sure|thanks|cool|got it|noted|yep|yeah|alright|'
        r'fine|roger|k|ty|thx|ack)\s*[.!]*$', re.IGNORECASE
    )
    if ack.match(text_lower):
        return 'acknowledged'

    if len(text_lower) < 3:
        return 'ignored'

    # For substantive messages (>20 chars), check engagement BEFORE rejection
    # This handles "I don't think that's right but tell me more" correctly
    engagement_pattern = re.compile(
        r'\b(yes|please|tell me|show|how|what|why|when|do it|go ahead|'
        r'more|explain|help|interesting|continue|elaborate)\b', re.IGNORECASE
    )

    rejection_pattern = re.compile(
        r'\b(stop|don.t|no thanks|not interested|shut up|leave me|go away|'
        r'not now|quit|enough|annoying)\b', re.IGNORECASE
    )

    has_engagement = engagement_pattern.search(text_lower)
    has_rejection = rejection_pattern.search(text_lower)

    # For longer messages with both signals, engagement wins
    # (user is still engaging even if they disagree)
    if len(text_lower) > 20 and has_engagement and has_rejection:
        # Check if rejection is followed by a "but" clause with engagement
        if re.search(r'\b(but|however|though|although)\b', text_lower):
            return 'engaged'
        # For longer messages, engagement signal wins by default
        return 'engaged'

    # Pure rejection (short or unambiguous)
    if has_rejection:
        return 'rejected'

    # Engaged: longer response, question, or action words
    if len(text_lower) > 20 or '?' in text or has_engagement:
        return 'engaged'

    return 'acknowledged'


def _try_proactive_engagement_correlation(text: str, topic: str):
    """
    Check if user is responding to a proactive message and classify engagement.
    Uses proactive_response_tag stored in MemoryStore (4h TTL).
    """
    try:
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()

        tag_key = f"proactive_response_tag:{topic}"
        goal_id = store.get(tag_key)
        if not goal_id:
            return

        # Consume the tag (one-shot correlation)
        store.delete(tag_key)

        response_type = _classify_engagement(text)

        from services.goal_ecology_service import GoalEcologyService
        ecology = GoalEcologyService()
        ecology.record_outcome(goal_id, response_type)

        # Track ignored count for social cost calculation
        if response_type == 'ignored':
            ignored_key = 'goal:recent_ignored_count'
            current = int(store.get(ignored_key) or 0)
            store.setex(ignored_key, 86400, str(current + 1))
        elif response_type in ('engaged', 'acknowledged'):
            store.delete('goal:recent_ignored_count')

        logging.info(
            f"[DIGEST] Proactive engagement: {response_type} "
            f"for goal {goal_id[:8]}"
        )
    except Exception as e:
        logging.debug(f"[DIGEST] Proactive engagement correlation failed: {e}")


_FORK_RESPONSE_PATTERNS = {
    'prefers_concise': [r'\b(quick|short|brief|summary|tldr|just.{0,10}(main|key|quick))\b'],
    'prefers_depth': [r'\b(deep|deeper|detail|more|elaborate|explore|full|thorough)\b'],
    'enjoys_challenge': [r'\b(challenge|push back|harder|stress.test|poke holes|counterpoint|disagree)\b'],
}


def _detect_fork_response(text: str, thread_id: str):
    """
    Detect if the user's message is a response to a previously offered fork.

    If a fork was pending (adaptive_fork_pending:{thread_id} MemoryStore key exists),
    pattern-match the user's reply and store the corresponding micro-preference.
    """
    import re as _re
    try:
        from services.memory_client import MemoryClientService
        from services.database_service import get_shared_db_service
        from services.knowledge_service import KnowledgeService

        store = MemoryClientService.create_connection()
        fork_type = store.get(f"adaptive_fork_pending:{thread_id}")
        if not fork_type:
            return

        # Match user response to a micro-preference
        text_lower = text.lower()
        for pref_key, patterns in _FORK_RESPONSE_PATTERNS.items():
            if any(_re.search(p, text_lower) for p in patterns):
                db_service = get_shared_db_service()
                ks = KnowledgeService(db_service)
                ks.store(
                    kind='trait', entity='user', key=pref_key, value='true',
                    data={'category': 'preference'},
                    decay_class='standard', confidence=0.75,
                    source='fork_response',
                )
                logging.info(f"[DIGEST] Fork response detected → stored micro-preference: {pref_key}")
                # Clear the pending key
                store.delete(f"adaptive_fork_pending:{thread_id}")
                break
    except Exception as e:
        logging.debug(f"[DIGEST] Fork response detection failed: {e}")


def _store_adaptive_signals(thread_id: str, text: str, signals: dict = None):
    """
    Store a minimal snapshot of current exchange signals to MemoryStore for use by
    AdaptiveLayerService (energy mirroring, cognitive load).

    Key: adaptive_signals:{thread_id}, TTL: 300s
    """
    import json as _json
    try:
        from services.memory_client import MemoryClientService

        store = MemoryClientService.create_connection()
        snapshot = {
            'prompt_token_count': len(text.split()) if text else 0,
            'explicit_feedback': signals.get('explicit_feedback') if signals else None,
        }
        store.setex(
            f"adaptive_signals:{thread_id}",
            300,
            _json.dumps(snapshot),
        )
    except Exception as e:
        logging.debug(f"[DIGEST] Adaptive signal storage failed: {e}")


def digest_worker(text: str, metadata: dict = None) -> str:
    """
    Main worker function that processes prompts through classification and response generation.

    Pipeline: Phase A (immediate commit) → Phase B (retrieval) → Phase C (route + generate)
              → Phase D (post-response commit) → Phase E (async follow-up)

    Proactive drift messages go through full routing but skip user input logging.
    """
    metadata = metadata or {}

    # Proactive drift shortcut: system-initiated outreach from cognitive drift engine
    if metadata.get('type') == 'proactive_drift':
        return _handle_proactive_drift(text, metadata)

    # Cron tool shortcut: background scheduled tool result (not a conversational turn)
    if metadata.get('source', '').startswith('cron_tool:'):
        return _handle_cron_tool_result(text, metadata)

    # Step 1: Load configurations
    configs = load_configs()
    cortex_config = configs['cortex']['config']
    cortex_prompt_map = configs['cortex']['prompt_map']

    # Step 1b: Resolve image_ids → image_contexts (WS4b).
    # The WS handler now passes image_ids in metadata without blocking.
    # We resolve them here, before any LLM call, with a 30-second timeout so
    # vision analysis has time to complete even if the user sends very quickly.
    _image_ids = metadata.get('image_ids', [])
    _image_contexts = metadata.get('image_contexts', [])  # backward-compat
    if _image_ids and not _image_contexts:
        _image_contexts = _resolve_image_contexts(_image_ids)
        if _image_contexts:
            metadata['image_contexts'] = _image_contexts
        logging.debug(
            f"[DIGEST] Resolved {len(_image_contexts)}/{len(_image_ids)} image(s) from MemoryStore"
        )

    # Step 2: Resolve thread
    thread_service = get_thread_service()
    platform = metadata.get('source', 'unknown')
    thread_resolution = thread_service.resolve_thread('default',platform)
    thread_id = thread_resolution.thread_id
    metadata['thread_id'] = thread_id

    # Mark thread as busy — prevents observer from trimming WM mid-response
    from services.memory_client import MemoryClientService
    _busy_store = MemoryClientService.create_connection()
    _busy_store.setex(f"thread_busy:{thread_id}", 30, "1")

    # Step 2a: Initialize services
    thread_conv_service = ThreadConversationService()
    world_state_service = WorldStateService()

    # Initialize working memory (keyed by thread_id)
    max_working_memory_turns = cortex_config.get('max_working_memory_turns', 10)
    working_memory = WorkingMemoryService(max_turns=max_working_memory_turns)

    # Hydrate working memory from SQLite if empty (e.g. after container restart)
    working_memory.hydrate_from_db(thread_id)

    # Initialize interaction log
    interaction_log = None
    try:
        from services.database_service import get_shared_db_service
        interaction_db = get_shared_db_service()
        interaction_log = InteractionLogService(interaction_db)
    except Exception as e:
        logging.warning(f"[DIGEST] Interaction log not available: {e}")

    # Initialize event bus
    EventBusService()

    # Initialize metrics
    metrics = MetricsService()
    trace_id = metrics.start_trace()
    metrics.record_counter('requests_total')
    request_start_time = time.time()

    # ═══════════════════════════════════════════════════════════
    # PHASE A: IMMEDIATE COMMIT (before any LLM call)
    # ═══════════════════════════════════════════════════════════

    # Step 3: Derive context topic from thread_id (thread-scoped, no classifier needed)
    context_topic = thread_id
    source = metadata.get('source', 'unknown') if metadata else 'unknown'

    # Step 3a: Immediate commit - append user turn to working memory (keyed by thread_id)
    # If images are attached, annotate the turn with their visual descriptions so
    # subsequent turns have context about what was shared.
    _image_ctxs_wm = (metadata or {}).get('image_contexts', [])
    _wm_text = text
    if _image_ctxs_wm:
        _visual_note = '; '.join(
            ctx.get('description', '') for ctx in _image_ctxs_wm if ctx.get('description')
        )
        if _visual_note:
            _wm_text = f"{text}\n[Attached: {_visual_note}]" if text else f"[Attached: {_visual_note}]"
    working_memory.append_turn(thread_id, 'user', _wm_text)

    # Persist user turn to topic transcript (durable, searchable)
    try:
        from services import transcript_service
        transcript_service.append(context_topic, 'user', _wm_text)
    except Exception as e:
        logging.debug(f"[DIGEST] Transcript append (user) failed: {e}", exc_info=True)

    # Situational intelligence — update conversation phase and situation model for this
    # user message.  Both calls are non-blocking, fail-open, and write to MemoryStore
    # only.  The updated state is ready when the frontal cortex generates its response.
    try:
        from services.conversation_phase_service import get_conversation_phase_service
        _phase_svc = get_conversation_phase_service()
        _phase_svc.update(thread_id, text, is_user=True, topic=context_topic)
    except Exception as e:
        logging.debug(f"[DIGEST] Phase update failed: {e}", exc_info=True)

    try:
        from services.situation_model_service import get_situation_model_service
        get_situation_model_service().update_on_message(thread_id)
    except Exception as e:
        logging.debug(f"[DIGEST] Situation update failed: {e}", exc_info=True)

    # IIP: Immediate Identity Promotion — synchronous, before any LLM call
    # Detects explicit name statements and writes to MemoryStore + SQLite immediately.
    try:
        from services.database_service import get_shared_db_service
        _run_iip_hook(text, get_shared_db_service())
    except Exception as _iip_e:
        logging.debug(f"[IIP] Skipped: {_iip_e}")

    # Step 3a.2: Belief correction hook — detect and apply explicit trait corrections
    # Runs before frontal cortex retrieves traits (Phase C), so corrected traits are
    # already in the database when get_traits_for_prompt() is called.
    _run_belief_correction_hook(text, thread_id=thread_id)

    # Step 3b: Immediate commit - log user input event (pre-classification)
    if interaction_log:
        interaction_log.log_event(
            event_type='user_input',
            payload={'message': text},
            topic=context_topic or 'unknown',
            source=source,
            metadata=metadata,
            thread_id=thread_id,
        )

    # Step 3b.0: Track message pace for proactive timing
    try:
        from services.time_utils import utc_now as _pace_utc_now
        _busy_store.set('last_user_message_ts', _pace_utc_now().isoformat())
        _current_count = int(_busy_store.get('recent_message_count_5min') or 0)
        _busy_store.setex('recent_message_count_5min', 300, str(_current_count + 1))
    except Exception as e:
        logger.debug(f"[DIGEST] Message pace tracking failed: {e}", exc_info=True)

    # Step 3b.1: Check for save trigger (completion/deferral signal)
    try:
        from services.save_suggestion_service import SaveSuggestionService
        _save_svc = SaveSuggestionService()
        _save_flag = _save_svc.get_saveable_flag(thread_id)
        if _save_flag:
            _trigger = _save_svc.detect_save_trigger(text)
            if _trigger:
                _save_svc.emit_save_card(
                    thread_id,
                    _save_flag.get('topic', context_topic or 'unknown'),
                    _save_flag['content_type'],
                )
                _save_svc.clear_flag(thread_id)
    except Exception as _save_e:
        logging.debug(f"[DIGEST] Save trigger check skipped: {_save_e}")

    # Step 3d: Proactive engagement correlation
    if context_topic:
        _try_proactive_engagement_correlation(text, context_topic)

    # Step 3e: Record user interaction timestamp (embedding stored after classification in Phase C)
    try:
        from services.autonomous_actions.communicate_action import CommunicateAction
        communicate = CommunicateAction()
        communicate.record_user_interaction()
    except Exception as e:
        logger.debug(f"[DIGEST] User interaction timestamp recording failed: {e}", exc_info=True)

    # Step 3f: Detect fork responses and store adaptive signals
    _detect_fork_response(text, thread_id)
    _store_adaptive_signals(thread_id, text)

    # ═══════════════════════════════════════════════════════════
    # PHASE B: RETRIEVAL (context assembly)
    # ═══════════════════════════════════════════════════════════

    world_state = (
        world_state_service.get_world_state(
            context_topic, thread_id=thread_id, message_embedding=None
        )
        if context_topic
        else ""
    )

    # Step 4b: Calculate context warmth for cost scaling
    wm_turns = working_memory.get_recent_turns(thread_id)
    context_warmth = calculate_context_warmth(
        working_memory_len=len(wm_turns),
        gists=[],
        world_state_nonempty=bool(world_state)
    )
    logging.info(f"[DIGEST] Context warmth for '{context_topic}': {context_warmth:.2f}")

    # ═══════════════════════════════════════════════════════════
    # PHASE C: CLASSIFICATION + ROUTING + RESPONSE
    # ═══════════════════════════════════════════════════════════

    # Step 6: Compute message embedding directly — thread_id is the topic key
    _embed_start = time.time()
    msg_embedding = None
    try:
        from services.embedding_service import EmbeddingService
        msg_embedding = EmbeddingService().generate_embedding(text)
        try:
            from services.autonomous_actions.communicate_action import CommunicateAction
            communicate = CommunicateAction()
            communicate.record_user_interaction(message_embedding=msg_embedding)
        except Exception as e:
            logging.debug(f"[DIGEST] Failed to store message embedding for proactive: {e}")
    except Exception as e:
        logging.debug(f"[DIGEST] Embedding computation failed: {e}")
    embedding_time = time.time() - _embed_start

    # Use thread_id as the topic key; supply static defaults for downstream signal consumers
    topic = thread_id
    classification_result = {'confidence': 1.0, 'is_new_topic': False}
    classification = {
        'topic': topic,
        'confidence': 10,
        'similar_topic': '',
        'topic_update': '',
        'context_warmth': context_warmth,
    }

    metrics.record_timing(trace_id, 'embedding', embedding_time * 1000)
    metrics.record_counter('embeddings_total')

    # Step 6b: Create TopicContext — single source of truth for this message's topic identity
    from services.topic_context import TopicContext
    topic_ctx = TopicContext(topic=topic, thread_id=thread_id, message_embedding=msg_embedding)

    # Step 7: Add exchange to thread conversation
    exchange_id = thread_conv_service.add_exchange(thread_id, topic, {
        "message": text,
        "embedding_time": embedding_time,
    })

    # Inject exchange_id into metadata so it flows through to SSE output
    metadata['exchange_id'] = exchange_id

    # Update thread with current topic
    thread_service.update_topic(thread_id, topic)
    thread_service.increment_exchange_count(thread_id)

    # Step 7a: Log classification event (with resolved topic)
    if interaction_log:
        interaction_log.log_event(
            event_type='classification',
            payload=classification,
            topic=topic,
            exchange_id=exchange_id,
            source=source,
            metadata={'embedding_time': embedding_time}
        )

    # Step 7d: Focus auto-inference and distraction check
    try:
        from services.focus_session_service import FocusSessionService
        focus_service = FocusSessionService()

        # Count consecutive exchanges on current topic for auto-inference
        try:
            from services.memory_client import MemoryClientService
            _store = MemoryClientService.create_connection()
            _streak_key = f"topic_streak:{thread_id}"
            _streak_raw = _store.get(_streak_key)
            _streak_data = json.loads(_streak_raw) if _streak_raw else {}

            if _streak_data.get('topic') == topic:
                _streak_count = _streak_data.get('count', 0) + 1
            else:
                _streak_count = 1

            _store.setex(_streak_key, 7200, json.dumps({'topic': topic, 'count': _streak_count}))

            # Auto-infer focus after consecutive exchanges on same topic
            focus_service.maybe_infer_focus(thread_id, topic, _streak_count)
        except Exception as _se:
            logging.debug(f"[DIGEST] Topic streak tracking failed: {_se}")

        # Distraction check if focus is active and message embedding available
        try:
            if msg_embedding is not None:
                distraction = focus_service.check_distraction(thread_id, msg_embedding)
                if distraction.get('is_distraction'):
                    logging.info(
                        f"[DIGEST] Focus distraction detected: "
                        f"similarity={distraction['similarity_to_focus']:.3f} "
                        f"to '{distraction['focus_description'][:50]}'"
                    )
        except Exception as _de:
            logging.debug(f"[DIGEST] Distraction check failed: {_de}")
    except Exception as _fe:
        logging.debug(f"[DIGEST] Focus services failed: {_fe}")

    # Step 9: Track session and check for episode generation
    session_service = get_session_service()
    session_service.set_thread(thread_id)

    # Returning-from-silence detection — must be BEFORE track_classification()
    # updates last_activity_time so the gap is measured against prior activity.
    _session_silence = session_service.is_returning_from_silence(threshold_seconds=2700)
    # silence_seconds > 0 means returning; keep raw value for future tiered-warmth use
    silence_seconds = _session_silence if _session_silence > 0 else 0.0
    returning_from_silence = silence_seconds > 0
    if returning_from_silence:
        logging.info(f"[DIGEST] Returning from silence: {silence_seconds:.0f}s gap detected")

    is_new_topic = classification.get('is_new_topic', False)
    session_service.track_classification(topic, is_new_topic, time.time())

    exchange_data = {
        'exchange_id': exchange_id,
        'prompt': {'message': text},
        'timestamp': time.time()
    }

    if session_service.check_topic_switch(topic):
        # Episodic consolidation handled by EpisodicMemoryObserver (signal density scan)
        session_service.reset_session()
        session_service.mark_topic_switch(topic)

        # Conditional WM reset: full clear if old topic consolidated, else keep 2-turn bridge
        try:
            from services.memory_client import MemoryClientService
            _wm_store = MemoryClientService.create_connection()
            wm_identifier = thread_id or topic
            consolidation_ts = _wm_store.get(f"last_consolidation:{wm_identifier}")
            if consolidation_ts is not None:
                working_memory.clear(wm_identifier)
                logging.info(f"[DIGEST] Full WM clear on topic switch (consolidated) for '{wm_identifier}'")
            else:
                wm_key = working_memory._get_memory_key(wm_identifier)
                working_memory.store.ltrim(wm_key, -4, -1)  # Keep last 2 turns (4 entries)
                logging.info(f"[DIGEST] WM trimmed to 2-turn bridge on topic switch for '{wm_identifier}'")
        except Exception as _wm_e:
            logging.debug(f"[DIGEST] WM topic-switch reset failed (non-fatal): {_wm_e}")

    # Tool relevance scoring removed — handled by ACT loop

    # Step 9c: Compute memory_confidence before intent classifier
    # Use FOK + context warmth + working memory depth as density proxy
    from services.memory_client import MemoryClientService
    store = MemoryClientService.create_connection()
    raw_fok = store.get(f"fok:{topic}") if topic else None
    fok = float(raw_fok) if raw_fok else 0.0
    fok_score = min(1.0, fok / 5.0)

    wm_depth_score = min(1.0, len(wm_turns) / 6.0)

    memory_confidence = (
        0.4 * fok_score
        + 0.4 * context_warmth
        + 0.2 * wm_depth_score
    )
    if classification_result.get('is_new_topic', False):
        memory_confidence *= 0.7
    memory_confidence = round(memory_confidence, 3)

    # Get working memory turn count
    working_memory_turns = len(wm_turns) if wm_turns else 0

    # Step 9d: Intent classification (~5ms, deterministic, no LLM)
    intent_classifier = get_intent_classifier()
    intent = intent_classifier.classify(
        text=text,
        topic=topic,
        context_warmth=context_warmth,
        memory_confidence=memory_confidence,
        working_memory_turns=working_memory_turns,
    )
    logging.info(
        f"[DIGEST] Intent: type={intent['intent_type']}, "
        f"complexity={intent['complexity']}, confidence={intent['confidence']:.2f}"
    )

    # Step 9e: Enqueue trait extraction (fire-and-forget, daemon thread)
    enqueue_trait_extraction(text, metadata=metadata, thread_id=thread_id)

    # Step 10: Unified generation (no gate, no mode routing)
    routing_result = None
    try:
        _nlp = compute_nlp_signals(text, intent)
        _signals = {
            'context_warmth': context_warmth,
            'working_memory_turns': working_memory_turns,
            'gist_count': 0,
            'fact_count': 0,
            'fact_keys': [],
            'world_state_present': bool(world_state and world_state.strip()),
            'topic_confidence': classification_result.get('confidence', 0.5),
            'is_new_topic': classification_result.get('is_new_topic', False),
            'session_exchange_count': getattr(session_service, 'topic_exchange_count', 0) if session_service else 0,
            'memory_confidence': memory_confidence,
        }
        _signals.update(_nlp)
        _signals['_prompt_text'] = text

        _store_adaptive_signals(thread_id, text, signals=_signals)

        response_data, routing_result = unified_generate(
            topic, text, classification, thread_conv_service,
            cortex_config, cortex_prompt_map, _signals,
            metadata=metadata, thread_id=thread_id,
            returning_from_silence=returning_from_silence,
            message_embedding=msg_embedding,
            topic_context=topic_ctx,
        )

    except Exception as _gen_ex:
        logging.error(f"[DIGEST] Unified generation failed: {_gen_ex}", exc_info=True)
        # Fallback: minimal UNIFIED
        response_data = {
            'mode': 'UNIFIED',
            'modifiers': [],
            'response': "I ran into an issue processing that. Could you try again?",
            'generation_time': 0.0,
            'actions': None,
            'confidence': 0.0,
        }
        routing_result = {
            'mode': 'UNIFIED',
            'router_confidence': 0.0,
            'routing_source': 'fallback',
            'routing_time_ms': 0.0,
            'scores': {},
        }

    # Store response in thread conversation history
    thread_conv_service.add_response(
        thread_id,
        response_data['response'],
        response_data['generation_time']
    )

    # Route through orchestrator (delivers response to WebSocket / output queue)
    if metadata:
        try:
            orchestrator = get_orchestrator()

            context = {
                'topic': topic,
                'response': response_data.get('response', ''),
                'confidence': response_data.get('confidence', 0.0),
                'generation_time': response_data.get('generation_time', 0.0),
                'destination': metadata.get('destination', 'web'),
                'metadata': metadata,
                'actions': response_data.get('actions', []),
                'clarification_question': None,
                'reply_actions': response_data.get('reply_actions'),
            }

            mode = response_data.get('mode', 'UNIFIED')
            logging.info(f"[FRONTAL CORTEX] Routing through orchestrator: {mode}")

            orchestrator_result = orchestrator.route_path(mode=mode, context=context)

            if orchestrator_result['status'] == 'error':
                logging.error(f"[ORCHESTRATOR] Error: {orchestrator_result['message']}")
                try:
                    from services.output_service import OutputService
                    OutputService().enqueue_text(
                        topic=topic,
                        response=response_data.get('response') or "I ran into an issue. Please try again.",
                        mode='UNIFIED',
                        confidence=response_data.get('confidence', 0.0),
                        generation_time=response_data.get('generation_time', 0.0),
                        original_metadata=metadata,
                    )
                except Exception as oe:
                    logging.warning(f"[ORCHESTRATOR] Failed to surface error to user: {oe}")
            else:
                logging.info(f"[ORCHESTRATOR] Executed {mode}: {orchestrator_result.get('result', {})}")
        except Exception as e:
            logging.error(f"[ORCHESTRATOR] Failed: {e}")

    # Signal completion for IGNORE/card-only mode on sync WebSocket channels.
    if response_data.get('mode') == 'IGNORE' and metadata and metadata.get('uuid'):
        try:
            from services.output_service import OutputService
            OutputService().enqueue_text(
                topic=topic,
                response='',
                mode='ACT',
                confidence=response_data.get('confidence', 1.0),
                generation_time=response_data.get('generation_time', 0.0),
                original_metadata=metadata,
                reply_actions=response_data.get('reply_actions'),
            )
        except Exception as e:
            logging.warning(f"[IGNORE] Failed to publish empty-text message event: {e}")

    # Add response to exchange data
    exchange_data['response'] = {'message': response_data['response']}
    if response_data.get('actions'):
        exchange_data['steps'] = response_data['actions']

    # Add complete exchange to session
    session_service.add_exchange(exchange_data)

    # ═══════════════════════════════════════════════════════════
    # PHASE D: POST-RESPONSE COMMIT
    # ═══════════════════════════════════════════════════════════

    # Step 11a: Append assistant turn to working memory (keyed by thread_id)
    working_memory.append_turn(thread_id, 'assistant', response_data['response'])

    # Persist assistant turn to topic transcript (durable, searchable)
    try:
        from services import transcript_service
        transcript_service.append(topic, 'assistant', response_data['response'])
    except Exception as e:
        logging.debug(f"[DIGEST] Transcript append (assistant) failed: {e}", exc_info=True)

    # Fire compaction if context is approaching budget
    try:
        from services import compaction_service
        _ctx_budget = 32000
        try:
            from services.frontal_cortex_service import FrontalCortexService
            _fc = FrontalCortexService(cortex_config, cortex_prompt_map)
            _ctx_limit = _fc.get_context_limit()
            _ctx_budget = min(int(_ctx_limit * 0.6), 150_000)
        except Exception as e:
            logger.debug(f"[DIGEST] Context limit resolution failed, using default: {e}", exc_info=True)
        compaction_service.check_and_compact(topic, _ctx_budget)
    except Exception as e:
        logging.debug(f"[DIGEST] Compaction check failed: {e}", exc_info=True)

    # Update conversation phase with Chalie's response so momentum and direction
    # reflect the full exchange, not just the user turn.
    try:
        from services.conversation_phase_service import get_conversation_phase_service
        _phase_svc_resp = get_conversation_phase_service()
        _phase_svc_resp.update(thread_id, response_data['response'], is_user=False, topic=topic)
    except Exception as e:
        logger.debug(f"[DIGEST] Conversation phase update (assistant response) failed: {e}", exc_info=True)

    # Step 11b: Log system response event
    if interaction_log:
        interaction_log.log_event(
            event_type='system_response',
            payload={
                'message': response_data['response'],
                'mode': response_data.get('mode', 'UNIFIED'),
                'confidence': response_data.get('confidence', 0.0),
                'generation_time': response_data.get('generation_time', 0.0)
            },
            topic=topic,
            exchange_id=exchange_id,
            source=source,
            metadata=metadata,
            thread_id=thread_id,
        )


    # Step 11e: Detect saveable content in response
    try:
        from services.save_suggestion_service import SaveSuggestionService
        _save_svc = SaveSuggestionService()
        # Text-heavy images (receipt, document photo) → trigger save suggestion
        _image_ctxs_save = (metadata or {}).get('image_contexts', [])
        _image_saveable = None
        for _ictx in _image_ctxs_save:
            if _ictx.get('has_text') and len(_ictx.get('ocr_text', '')) > 200:
                _image_saveable = {'content_type': 'image_document', 'topic': 'Captured document'}
                break
        _saveable = _image_saveable or _save_svc.detect_saveable_content(
            response_data['response'], topic, thread_id,
        )
        if _saveable:
            _save_svc.flag_saveable(
                thread_id, topic, _saveable['content_type'], exchange_id,
            )
    except Exception as _save_e:
        logging.debug(f"[DIGEST] Saveable content detection skipped: {_save_e}")

    # ── Phase D Step 11f: Goal signal extraction ──
    try:
        from services.goal_signal_service import extract_and_route_signals
        _goal_classification = dict(classification or {})
        _goal_classification['intent_type'] = (intent or {}).get('intent_type', '')
        extract_and_route_signals(topic, text, _goal_classification)
    except Exception as e:
        logging.debug(f"[DIGEST] Goal signal extraction non-fatal: {e}")

    # ═══════════════════════════════════════════════════════════
    # PHASE E: ASYNC FOLLOW-UP
    # ═══════════════════════════════════════════════════════════

    # Step 12: Episodic consolidation handled by EpisodicMemoryObserver (60s scan)

    # Print the actual response to stdout for the user
    logging.info(f"\n{'='*60}")
    logging.info(f"Topic: {topic}")
    _rc = routing_result.get('router_confidence', 0.0) if routing_result else 0.0
    logging.info(f"Mode: {response_data['mode']} (router confidence: {_rc:.3f})")
    logging.info(f"{'='*60}")
    logging.info(response_data['response'])
    logging.info(f"{'='*60}\n")

    # Record metrics
    metrics.record_timing(trace_id, 'response_generation', response_data['generation_time'] * 1000)
    metrics.record_timing(trace_id, 'total_request', (time.time() - request_start_time) * 1000)
    metrics.record_counter('responses_total')

    # Clear thread-busy flag — observer can now safely scan this thread
    try:
        _busy_store.delete(f"thread_busy:{thread_id}")
    except Exception as e:
        logger.debug(f"[DIGEST] Failed to clear thread-busy flag: {e}", exc_info=True)

    return f"Topic '{topic}' | Mode: {response_data['mode']} | Response generated in {response_data['generation_time']:.2f}s"
