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
interactive tool dialog (``process_tool_dialog``, ``store_tool_dialog_memory``).
"""

import json
import os
import time
import logging

from services import FrontalCortexService
from services.llm_service import create_llm_service
from services.world_state_service import WorldStateService
from services.working_memory_service import WorkingMemoryService
from services.interaction_log_service import InteractionLogService
from services.event_bus_service import EventBusService
from services.metrics_service import MetricsService
from services.mode_router_service import compute_nlp_signals
from services.innate_skills.registry import ALL_SKILL_NAMES

# Singleton getters — canonical implementations live in digest_singletons.py;
# re-exported here so existing ``from workers.digest_worker import get_*`` still works.
from workers.digest_singletons import (          # noqa: F401
    get_context_relevance_service,
    get_orchestrator,
    get_mode_router,
    load_configs,
)

# Post-exchange hooks — canonical implementations live in post_exchange_hooks.py;
# re-exported here so existing ``from workers.digest_worker import _run_*`` still works.
from workers.post_exchange_hooks import (         # noqa: F401
    _run_iip_hook,
    _run_belief_correction_hook,
    _classify_engagement,
    _detect_fork_response,
    _store_adaptive_signals,
)

logger = logging.getLogger(__name__)
_LOG_PROMPTS = os.environ.get('CHALIE_LOG_PROMPTS') == '1'


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
        List of image context dicts (``ocr_text``, ``has_text``,
        ``analysis_time_ms``, ``error``).  Always same length as *image_ids*
        in the same order.  Timed-out or unparseable images get an ``error``
        key describing what happened.
    """
    if not image_ids:
        return []
    from services.memory_client import MemoryClientService
    store = MemoryClientService.create_connection()
    contexts = []
    for img_id in image_ids:
        key = f'chat_image_result:{img_id}'
        deadline = time.time() + timeout
        resolved = False
        while time.time() < deadline:
            raw = store.get(key)
            if raw:
                try:
                    contexts.append(json.loads(raw))
                except json.JSONDecodeError as e:
                    contexts.append({'error': f'failed to parse analysis result: {e}'})
                    logger.debug(f"[DIGEST] Failed to parse image context JSON for {img_id!r}: {e}", exc_info=True)
                resolved = True
                break
            time.sleep(1)
        if not resolved:
            contexts.append({'error': f'timed out trying to process image after {timeout} seconds'})
            logging.info(f"[DIGEST] Image resolution timed out for {img_id!r} after {timeout}s")
    return contexts




def _store_image_tool_calls(transcript_id: int, image_ids: list, image_contexts: list) -> list:
    """Store a tool_call row for each uploaded image, linking it to the transcript entry.

    Fetches document metadata (mime_type, file_path) from the documents table and
    pairs it with the analysis results from image_contexts (ocr_text, has_text,
    analysis_time_ms).  The result is a pre-formatted [file()] tag that
    UserPromptAssemblyService renders verbatim.

    Returns the list of [file()] tag strings so the caller can inject them into
    the current turn without a DB round-trip.
    """
    from urllib.parse import quote

    tags = []
    try:
        from services.database_service import get_shared_db_service
        from services.document_service import DocumentService

        db = get_shared_db_service()
        doc_svc = DocumentService(db)

        # Build a lookup from image_id → analysis context.
        # _resolve_image_contexts iterates image_ids sequentially, preserving
        # order, but skips timed-out images — so the list may be shorter.
        # Include the image_id in each context dict for safe correlation.
        ctx_by_id = {}
        for img_id, ctx in zip(image_ids, image_contexts):
            ctx_by_id[img_id] = ctx

        # Fetch all document records before opening the write connection
        docs_by_id = {}
        for img_id in image_ids:
            docs_by_id[img_id] = doc_svc.get_document(img_id)

        for img_id in image_ids:
            doc = docs_by_id.get(img_id)
            ctx = ctx_by_id.get(img_id)

            # Build description from all available analysis features
            if ctx is None:
                description = 'image failed to process'
            elif ctx.get('error'):
                description = ctx['error']
            else:
                ocr = ctx.get('ocr_text', '').strip()
                description = f"Contains text: {ocr}" if ocr else 'image'

            mime = doc.get('mime_type', 'image/unknown') if doc else 'image/unknown'
            path = quote(doc.get('file_path', ''), safe='/') if doc else ''

            tag = f"[file(id:{img_id},type:{mime},path:{path})] {description}"
            tags.append(tag)

        # Persist all file tags via ToolCallService
        from services.tool_call_service import ToolCallService
        _tc_svc = ToolCallService()
        for tag in tags:
            _tc_svc.store(transcript_id, 'file', {}, tag, invoked_by='system')

    except Exception as e:
        logging.info(f"[DIGEST] Failed to store image tool_calls: {e}", exc_info=True)

    return tags


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
                    while words and words[0].lower() in _VALUE_STRIP:
                        words.pop(0)
                    while words and words[-1].lower() in _VALUE_STRIP:
                        words.pop()
                    return ' '.join(words) if words else v

                stored_count = 0
                for trait in traits:
                    key = trait.get('key', '').lower().strip()
                    value = _clean_value(trait.get('value', '').strip())
                    conf_label = trait.get('confidence', 'low')
                    is_permanent = trait.get('permanent', False)

                    if not key or not value:
                        continue

                    confidence = CONFIDENCE_MAP.get(conf_label, 0.35)
                    decay_class = 'permanent' if is_permanent else 'standard'

                    stored_entry = ks.store(
                        kind='trait', entity='user', key=key, value=value,
                        data={'category': 'core' if is_permanent else 'preference'},
                        decay_class=decay_class,
                        confidence=confidence, source='llm_extraction',
                    )
                    stored_count += 1

                    # Inline contradiction check — uses rowid from returned dict
                    if stored_entry:
                        new_id = stored_entry.get('rowid') or stored_entry.get('id')
                        if new_id:
                            _check_trait_contradiction(ks, new_id, key, value, confidence, thread_id, source='chat')

                if stored_count > 0:
                    _synthesize_user_sentence(db, provider_config)

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

        def _synthesize_user_sentence(db, provider_config) -> None:
            """Fire a second LLM call to produce a single natural-language sentence
            summarising ALL stored user traits, then persist it in the settings table."""
            try:
                import os

                rows = db.fetch_all(
                    "SELECT key, value, confidence, decay_class "
                    "FROM knowledge "
                    "WHERE entity = 'user' AND kind = 'trait' AND deleted_at IS NULL "
                    "ORDER BY decay_class DESC, confidence DESC"
                )
                if not rows:
                    return

                trait_lines = []
                for row in rows:
                    key = row['key'] if isinstance(row, dict) else row[0]
                    value = row['value'] if isinstance(row, dict) else row[1]
                    confidence = row['confidence'] if isinstance(row, dict) else row[2]
                    decay_class = row['decay_class'] if isinstance(row, dict) else row[3]
                    trait_lines.append(f"{key}: {value} (confidence: {confidence:.2f}, {decay_class})")

                prompt_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), 'prompts', 'trait-synthesis.md'
                )
                with open(prompt_path, 'r') as f:
                    template = f.read()
                prompt_text = template.replace('{{traits}}', '\n'.join(trait_lines))

                llm = create_llm_service(provider_config)
                llm_resp = llm.send_message(
                    prompt_text,
                    "Output only the sentence. No preamble, no explanation."
                )
                sentence = llm_resp.text.strip() if llm_resp and llm_resp.text else None
                if not sentence:
                    return

                from services.knowledge_service import KnowledgeService
                ks = KnowledgeService(db)
                ks.store(
                    kind='fact', entity='system', key='user_summary',
                    value=sentence, decay_class='permanent',
                    confidence=1.0, source='trait_synthesis',
                )
                logging.debug("[TRAIT_SYNTH] Sentence stored: %s", sentence)
            except Exception as e:
                logging.warning("[TRAIT_SYNTH] Failed: %s", e)

        t = threading.Thread(target=_extract_traits, daemon=True)
        t.start()

    except Exception as e:
        logging.debug(f"[TRAIT_EXTRACT] Enqueue failed: {e}")


def _check_trait_contradiction(ks, new_id: int, key: str, value: str, confidence: float, thread_id: str, source: str = 'chat'):
    """
    Runs synchronously after a trait is stored.
    - temporal_change + source=chat → hard-delete old trait
    - true_contradiction OR source=ambient → reduce confidences, create pending record, push question
    """
    try:
        from services.embedding_service import get_embedding_service
        from services.contradiction_classifier_service import ContradictionClassifierService
        from services.pending_contradiction_service import PendingContradictionService
        from services.database_service import get_shared_db_service
        from services.output_service import OutputService

        emb_svc = get_embedding_service()
        new_text = f"{key}: {value}"
        embedding = emb_svc.generate_embedding(new_text)
        if embedding is None:
            return

        similar = ks.find_similar_traits(embedding, exclude_id=new_id)
        if not similar:
            return

        existing = similar[0]  # top candidate only
        existing_text = f"{existing['key']}: {existing['value']}"

        classifier = ContradictionClassifierService()
        result = classifier.check_new_trait(new_text, existing_text, source=source)
        if result is None:
            return

        classification = result.get('classification', 'compatible')

        if classification == 'temporal_change' and source == 'chat':
            # Auto-overwrite: hard-delete old trait
            ks.hard_delete_by_id(existing['id'])
            logging.info(f"[CONTRADICTION] temporal_change: deleted old trait id={existing['id']} key={existing['key']}")

        elif classification in ('true_contradiction', 'ambiguous') or source == 'ambient':
            # Ask user: reduce confidences, create pending record, push message
            ks.update_confidence(existing['id'], existing['confidence'] * 0.75)
            ks.update_confidence(new_id, 0.5)

            question = (
                f"I'm seeing two conflicting things about {key}: "
                f"'{existing['value']}' (existing) and '{value}' (just mentioned). "
                f"Which is correct? Just tell me and I'll update."
            )

            db = get_shared_db_service()
            pending_svc = PendingContradictionService(db)
            pending_svc.create(new_id, existing['id'], question, source)

            if thread_id:
                OutputService().enqueue_proactive(
                    topic=thread_id,
                    response=question,
                    source='contradiction',
                )
                logging.info(f"[CONTRADICTION] Surfaced to user: {question[:80]}")

    except Exception as e:
        logging.debug(f"[CONTRADICTION] Check failed: {e}")




def calculate_context_warmth(working_memory_len: int, world_state_nonempty: bool, gists: list = None) -> float:
    """
    Calculate context warmth signal (0.0-1.0) for scaling uncertainty cost.
    """
    wm_score = min(working_memory_len / 4, 1.0)
    world_score = 1.0 if world_state_nonempty else 0.0
    warmth = (wm_score + world_score) / 2
    return warmth




def unified_generate(channel, text, classification, thread_conv_service,
                     cortex_config, cortex_prompt_map, signals,
                     metadata=None, thread_id=None,
                     returning_from_silence=False, message_embedding=None,
                     proactive: bool = False):
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

    cortex_service = FrontalCortexService(config)

    # Build native tool schemas
    all_skills = list(ALL_SKILL_NAMES)
    # Voice mode: exclude visual-only skills (rich_render outputs blocks unusable via TTS)
    if (metadata or {}).get('source') == 'voice':
        all_skills = [s for s in all_skills if s != 'rich_render']
    from services.tool_schema_service import get_skill_schemas
    native_tools = get_skill_schemas(all_skills)

    # ── System prompt (stable, cacheable) ─────────────────────────
    from services.system_prompt_assembly_service import SystemPromptAssemblyService
    system_prompt_svc = SystemPromptAssemblyService(config)
    system_prompt_svc.build(
        template=prompt,
        classification=classification,
        selected_skills=all_skills,
        thread_id=thread_id,
        returning_from_silence=returning_from_silence,
        inclusion_map=inclusion_map,
    )

    system_prompt = system_prompt_svc.to_provider()

    # ── User prompt (per-turn, never cached) ──────────────────────
    from services.user_prompt_assembly_service import UserPromptAssemblyService
    user_prompt_svc = UserPromptAssemblyService()
    user_prompt_svc.build(
        user_message=text,
        channel=channel,
        thread_id=thread_id,
        metadata=metadata,
    )
    user_prompt = user_prompt_svc.to_provider()

    # Prompt tracing
    if _LOG_PROMPTS:
        try:
            from services.database_service import get_shared_db_service
            from services.interaction_log_service import InteractionLogService
            _log_svc = InteractionLogService(get_shared_db_service())
            _log_svc.log_event(
                event_type='llm_prompt',
                payload={
                    'system_prompt': system_prompt,
                    'user_message': user_prompt,
                    'skills': all_skills,
                    'tool_count': len(native_tools),
                },
                topic=channel,
                exchange_id=(metadata or {}).get('exchange_id'),
                source='unified_generate',
                metadata=metadata,
                thread_id=thread_id,
            )
        except Exception as _lp_e:
            logging.debug(f"[DIGEST] Prompt logging failed: {_lp_e}")

    first_messages = [{"role": "user", "content": user_prompt}]

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

    # TODO: migrate to MessageProcessor tool loop
    # ACTOrchestrator has been deleted. digest_worker is a legacy path that will be
    # removed once all traffic moves through UserMessageProcessor.process().
    # For now, return the first_response as-is (tool calls unexecuted).
    logging.warning("[UNIFIED] digest_worker ACT path reached but ACTOrchestrator is gone. Returning direct response.")
    first_response['mode'] = 'UNIFIED'
    routing_result['mode'] = 'UNIFIED'
    return first_response, routing_result



def process_tool_dialog(text: str, tool_name: str, trigger_prompt: str) -> str:
    """
    Process tool data through Chalie's full cognitive pipeline (including ACT loop).

    Called synchronously during an interactive tool-Chalie dialog. Returns response text
    to be written back to the tool container's stdin. Does NOT surface to the user.

    Memory is stored with memory_durability='tool_internal' -- weaker than cron_tool,
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

        channel = f'tool_dialog:{tool_name}'
        thread_id = channel

        classification = {
            'topic': channel,
            'confidence': 10,
            'similar_topic': '',
            'topic_update': '',
        }

        # Build minimal routing signals -- tool dialogs don't need full signal collection
        signals = {'_prompt_text': text}

        # Trait extraction -- runs in background thread, non-blocking
        enqueue_trait_extraction(
            prompt_message=text[:1000],
            metadata={'source': 'chat'},
            thread_id=thread_id,
        )

        # Generate response via unified pipeline (may engage ACT loop)
        # metadata=None means orchestrator does NOT deliver to user
        response_data, _ = unified_generate(
            channel=channel,
            text=text,
            classification=classification,
            thread_conv_service=None,
            cortex_config=cortex_config,
            cortex_prompt_map=cortex_prompt_map,
            signals=signals,
            metadata=None,
            thread_id=thread_id,
            returning_from_silence=False,
            message_embedding=None,
        )

        response = response_data.get('response', '')

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


def _run_response_pipeline(
    *,
    text,
    channel,
    classification,
    thread_id,
    metadata,
    cortex_config,
    prompt_template,
    generation_config,
    destination='web',
    trait_extraction_meta=None,
    log_event_type=None,
    log_payload=None,
    log_source=None,
    log_tag='PIPELINE',
    working_memory=None,
    wm_key=None,
):
    """Shared response pipeline for cron-tool and proactive-drift handlers.

    Covers the common steps: context assembly, FrontalCortexService generation,
    empty-response check, working-memory append, conversation history store,
    orchestrator routing, trait extraction, and interaction logging.

    Args:
        text: The prompt text for generation.
        channel: Channel key for context assembly and orchestrator.
        classification: Classification dict for context/generation.
        thread_id: Resolved thread ID.
        metadata: Full metadata dict (passed to orchestrator).
        cortex_config: Base cortex config dict.
        prompt_template: System prompt template string for FrontalCortexService.
        generation_config: Config dict for FrontalCortexService (may differ from cortex_config).
        destination: Delivery destination (default 'web').
        trait_extraction_meta: Dict passed as ``metadata`` to ``enqueue_trait_extraction``.
        log_event_type: Event type string for interaction log (None to skip).
        log_payload: Payload dict for interaction log (response text is injected automatically).
        log_source: Source string for interaction log.
        log_tag: Tag for log messages (e.g. 'CRON TOOL', 'PROACTIVE').
        working_memory: Optional WorkingMemoryService instance for WM append.
        wm_key: Key for working memory append (defaults to ``thread_id or channel``).

    Returns:
        tuple: (response_data dict, is_empty bool) where is_empty is True
            when the LLM produced an empty response and the caller should
            return an early-exit status string.
    """
    # Context relevance
    inclusion_map = None
    try:
        inclusion_map = get_context_relevance_service().compute_inclusion_map(
            mode='UNIFIED', signals={}, classification=classification,
        )
    except Exception as e:
        logging.warning(f"[{log_tag}] Context relevance failed: {e}")

    # ── User prompt (per-turn) ────────────────────────────────────
    from services.user_prompt_assembly_service import UserPromptAssemblyService
    user_prompt_svc = UserPromptAssemblyService()
    user_prompt_svc.build(user_message=text, channel=channel, thread_id=thread_id)
    user_prompt = user_prompt_svc.to_provider()

    # Generation
    cortex_service = FrontalCortexService(generation_config)

    response_data = cortex_service.generate_response(
        system_prompt_template=prompt_template,
        original_prompt=user_prompt,
        classification=classification,
        chat_history=[],
        thread_id=thread_id,
        inclusion_map=inclusion_map,
    )

    # Force UNIFIED mode (both cron and proactive bypass mode routing)
    response_data['mode'] = 'UNIFIED'

    # Empty response check
    if not response_data.get('response', '').strip():
        logging.info(f"[{log_tag}] Empty response generated -- skipping delivery")
        return response_data, True

    # Working memory append
    if working_memory is not None:
        _wm_key = wm_key or thread_id or channel
        working_memory.append_turn(_wm_key, 'assistant', response_data['response'])

    # Orchestrator routing
    try:
        orchestrator = get_orchestrator()
        context = {
            'topic': channel,
            'response': response_data['response'],
            'confidence': response_data.get('confidence', 0.5),
            'generation_time': response_data.get('generation_time', 0.0),
            'destination': destination,
            'metadata': metadata,
            'actions': [],
        }
        orchestrator.route_path(mode='UNIFIED', context=context)
    except Exception as e:
        logging.error(f"[{log_tag}] Orchestrator failed: {e}")

    # Trait extraction
    if trait_extraction_meta is not None:
        try:
            enqueue_trait_extraction(
                prompt_message=text,
                metadata=trait_extraction_meta,
                thread_id=thread_id,
            )
        except Exception as e:
            logging.warning(f"[{log_tag}] Trait extraction enqueue failed: {e}")

    # Interaction logging
    if log_event_type:
        try:
            from services.database_service import get_shared_db_service
            from services.interaction_log_service import InteractionLogService
            db_service = get_shared_db_service()
            log_service = InteractionLogService(db_service)
            payload = dict(log_payload) if log_payload else {}
            payload.setdefault('response', response_data['response'][:500])
            payload.setdefault('generation_time', response_data.get('generation_time', 0))
            log_service.log_event(
                event_type=log_event_type,
                payload=payload,
                topic=channel,
                source=log_source or log_tag.lower(),
                metadata=metadata,
            )
        except Exception as e:
            logger.debug(f"[{log_tag}] Failed to log {log_event_type} event: {e}", exc_info=True)

    return response_data, False


def _handle_persistent_task_result(text: str, metadata: dict) -> str:
    """Surface a completed persistent task result to the user.

    Follows the same pattern as _handle_cron_tool_result — inject the result
    into the response pipeline so the LLM produces a natural summary.
    """
    try:
        from services.config_service import ConfigService

        configs = load_configs()
        cortex_config = configs['cortex']['config']

        task_id = metadata.get('task_id')
        thread_id = metadata.get('thread_id') or 'persistent_task'
        goal = metadata.get('goal', '')

        working_memory = WorkingMemoryService(
            max_turns=cortex_config.get('max_working_memory_turns', 10)
        )

        try:
            generation_config = ConfigService.resolve_agent_config("frontal-cortex-scheduled-tool")
        except Exception:
            generation_config = ConfigService.resolve_agent_config("frontal-cortex")

        prompt_template = ConfigService.get_agent_prompt("frontal-cortex-scheduled-tool")

        channel = f'persistent_task:{task_id}'
        classification = {
            'topic': channel,
            'confidence': 10,
            'similar_topic': '',
            'topic_update': '',
        }

        injected_text = f"Background task completed.\n\nGoal: {goal}\n\nResult:\n{text}"

        response_data, is_empty = _run_response_pipeline(
            text=injected_text,
            channel=channel,
            classification=classification,
            thread_id=thread_id,
            metadata=metadata,
            cortex_config=cortex_config,
            prompt_template=prompt_template,
            generation_config=generation_config,
            destination=metadata.get('destination', 'web'),
            trait_extraction_meta={'source': f'persistent_task:{task_id}'},
            log_event_type='persistent_task_completed',
            log_payload={'task_id': task_id, 'goal': goal},
            log_source='persistent_task',
            log_tag='PERSISTENT TASK',
            working_memory=working_memory,
            wm_key=thread_id or channel,
        )

        if is_empty:
            return f"Task {task_id} | Empty response"
        return f"Task {task_id} | Surfaced in {response_data.get('generation_time', 0):.2f}s"

    except Exception as e:
        logging.error(f"[PERSISTENT TASK] Failed to surface result: {e}", exc_info=True)
        return f"Task {metadata.get('task_id', 'unknown')} | ERROR: {e}"


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

        thread_id = metadata.get('thread_id') or metadata.get('source', 'cron_tool')

        working_memory = WorkingMemoryService(
            max_turns=cortex_config.get('max_working_memory_turns', 10)
        )

        # Load scheduled tool prompt template and config
        scheduled_tool_template = ConfigService.get_agent_prompt("frontal-cortex-scheduled-tool")
        try:
            scheduled_tool_config = ConfigService.resolve_agent_config("frontal-cortex-scheduled-tool")
        except Exception as e:
            logging.warning(f"[CRON TOOL] frontal-cortex-scheduled-tool.json not found, using frontal-cortex config: {e}")
            scheduled_tool_config = ConfigService.resolve_agent_config("frontal-cortex")

        channel = f'cron_tool:{tool_name}'
        classification = {
            'topic': channel,
            'confidence': 10,
            'similar_topic': '',
            'topic_update': '',
        }

        response_data, is_empty = _run_response_pipeline(
            text=text,
            channel=channel,
            classification=classification,
            thread_id=thread_id,
            metadata=metadata,
            cortex_config=cortex_config,
            prompt_template=scheduled_tool_template,
            generation_config=scheduled_tool_config,
            destination=destination,
            trait_extraction_meta={
                'source': f'cron_tool:{tool_name}',
                'priority': priority,
            },
            log_event_type='cron_tool_executed',
            log_payload={
                'tool_name': tool_name,
                'priority': priority,
            },
            log_source='cron_tool',
            log_tag='CRON TOOL',
            working_memory=working_memory,
            wm_key=thread_id or channel,
        )

        if is_empty:
            return f"Tool '{tool_name}' | Mode: UNIFIED | Empty response (no updates)"

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


def digest_worker(text: str, metadata: dict = None) -> str:
    """
    Main worker function that processes prompts through classification and response generation.

    Pipeline: Phase A (immediate commit) → Phase B (retrieval) → Phase C (route + generate)
              → Phase D (post-response commit) → Phase E (async follow-up)

    Proactive drift messages go through full routing but skip user input logging.
    """
    metadata = metadata or {}

    # Persistent task result shortcut: background task completed (not a conversational turn)
    if metadata.get('source') == 'persistent_task_result':
        return _handle_persistent_task_result(text, metadata)

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

    # Step 2: Resolve channel
    from services.memory_client import MemoryClientService
    _channel_store = MemoryClientService.create_connection()
    thread_id = _channel_store.get('active_channel:default') or 'web:default:1'
    if isinstance(thread_id, bytes):
        thread_id = thread_id.decode()
    metadata['thread_id'] = thread_id

    # Mark channel as busy — prevents observer from trimming WM mid-response
    _busy_store = _channel_store
    _busy_store.setex(f"thread_busy:{thread_id}", 30, "1")

    # Step 2a: Initialize services
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
    working_memory.append_turn(thread_id, 'user', text)

    # Persist user turn to topic transcript (durable, searchable)
    _user_transcript_id = None
    try:
        from services import transcript_service
        _user_transcript_id = transcript_service.append(context_topic, 'user', text)
    except Exception as e:
        logging.debug(f"[DIGEST] Transcript append (user) failed: {e}", exc_info=True)

    # Store uploaded images as tool_calls linked to this transcript entry.
    # Each image is already persisted in the documents table (source_type='chat_image')
    # by the /chat/image endpoint.  We record a tool_call so UserPromptAssemblyService
    # renders the [file()] tag alongside the user's message.
    # Tags are also returned so we can inject them into the current turn directly
    # (the tool_calls DB round-trip only serves previous turns).
    _file_tags = []
    if _user_transcript_id and _image_ids:
        _file_tags = _store_image_tool_calls(_user_transcript_id, _image_ids, _image_contexts)
    if _file_tags:
        metadata['file_tags'] = _file_tags

    # Onboarding nudge — check if we should prompt the LLM to elicit a missing trait.
    # Stored as a tool_call so it appears in previous turns, and passed via metadata
    # for the current turn.
    if _user_transcript_id:
        try:
            from services.prompt_assembly_service import PromptAssemblyService
            _nudge_svc = PromptAssemblyService(cortex_config)
            _nudge_text = _nudge_svc._get_onboarding_nudge(thread_id)
            if _nudge_text:
                _nudge_tag = f"[nudge] {_nudge_text}"
                metadata['nudge_tag'] = _nudge_tag
                from services.tool_call_service import ToolCallService
                ToolCallService().store(
                    _user_transcript_id, 'nudge', {}, _nudge_tag, invoked_by='system'
                )
        except Exception as e:
            logging.debug(f"[DIGEST] Onboarding nudge failed: {e}", exc_info=True)

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
        _busy_store.set('last_user_message_ts', _pace_utc_now().isoformat(), ex=86400)
        _current_count = int(_busy_store.get('recent_message_count_5min') or 0)
        _busy_store.setex('recent_message_count_5min', 300, str(_current_count + 1))
    except Exception as e:
        logger.debug(f"[DIGEST] Message pace tracking failed: {e}", exc_info=True)

    # Step 3b.1: Reset DMN idle timer
    try:
        from services.dmn_service import get_dmn_service
        get_dmn_service().on_turn()
    except Exception as _e:
        logging.debug(f"[DIGEST] DMN on_turn failed: {_e}")

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
    except Exception as e:
        logging.debug(f"[DIGEST] Embedding computation failed: {e}")
    embedding_time = time.time() - _embed_start

    # Use thread_id as the channel key; supply static defaults for downstream signal consumers
    channel = thread_id
    classification_result = {'confidence': 1.0, 'is_new_topic': False}
    classification = {
        'topic': channel,
        'confidence': 10,
        'similar_topic': '',
        'topic_update': '',
        'context_warmth': context_warmth,
    }

    metrics.record_timing(trace_id, 'embedding', embedding_time * 1000)
    metrics.record_counter('embeddings_total')

    # Step 7: Generate exchange_id for this turn
    import uuid as _uuid_mod
    exchange_id = str(_uuid_mod.uuid4())

    # Inject exchange_id into metadata so it flows through to SSE output
    metadata['exchange_id'] = exchange_id

    # Step 7a: Log classification event (with resolved channel)
    if interaction_log:
        interaction_log.log_event(
            event_type='classification',
            payload=classification,
            topic=channel,
            exchange_id=exchange_id,
            source=source,
            metadata={'embedding_time': embedding_time}
        )

    # Step 9: Returning-from-silence detection via MemoryStore last activity timestamp
    _last_activity_raw = _busy_store.get('last_activity_ts')
    returning_from_silence = False
    silence_seconds = 0.0
    if _last_activity_raw:
        try:
            from services.time_utils import parse_utc, utc_now as _utc_now_silence
            _last_ts = parse_utc(_last_activity_raw.decode() if isinstance(_last_activity_raw, bytes) else _last_activity_raw)
            silence_seconds = (_utc_now_silence() - _last_ts).total_seconds()
            returning_from_silence = silence_seconds > 2700
            if returning_from_silence:
                logging.info(f"[DIGEST] Returning from silence: {silence_seconds:.0f}s gap detected")
        except Exception as _sil_e:
            logging.debug(f"[DIGEST] Silence detection failed: {_sil_e}")
    try:
        from services.time_utils import utc_now as _utc_now_act
        _busy_store.set('last_activity_ts', _utc_now_act().isoformat(), ex=86400)
    except Exception:
        pass

    # Step 9c: Compute memory_confidence
    # Use FOK + context warmth + working memory depth as density proxy
    store = _busy_store
    raw_fok = store.get(f"fok:{channel}") if channel else None
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

    # Step 9e: Enqueue trait extraction (fire-and-forget, daemon thread)
    enqueue_trait_extraction(text, metadata=metadata, thread_id=thread_id)

    # Step 10: Unified generation (no gate, no mode routing)
    routing_result = None
    try:
        _nlp = compute_nlp_signals(text)
        _signals = {
            'context_warmth': context_warmth,
            'working_memory_turns': working_memory_turns,
            'gist_count': 0,
            'fact_count': 0,
            'fact_keys': [],
            'world_state_present': bool(world_state and world_state.strip()),
            'topic_confidence': classification_result.get('confidence', 0.5),
            'is_new_topic': classification_result.get('is_new_topic', False),
            'session_exchange_count': 0,
            'memory_confidence': memory_confidence,
        }
        _signals.update(_nlp)
        _signals['_prompt_text'] = text

        _store_adaptive_signals(thread_id, text, signals=_signals)

        response_data, routing_result = unified_generate(
            channel, text, classification, None,
            cortex_config, cortex_prompt_map, _signals,
            metadata=metadata, thread_id=thread_id,
            returning_from_silence=returning_from_silence,
            message_embedding=msg_embedding,
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

    # Route through orchestrator (delivers response to WebSocket / output queue)
    if metadata:
        try:
            orchestrator = get_orchestrator()

            context = {
                'topic': channel,
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
                        topic=channel,
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
                topic=channel,
                response='',
                mode='ACT',
                confidence=response_data.get('confidence', 1.0),
                generation_time=response_data.get('generation_time', 0.0),
                original_metadata=metadata,
                reply_actions=response_data.get('reply_actions'),
            )
        except Exception as e:
            logging.warning(f"[IGNORE] Failed to publish empty-text message event: {e}")

    # ═══════════════════════════════════════════════════════════
    # PHASE D: POST-RESPONSE COMMIT
    # ═══════════════════════════════════════════════════════════

    # Step 11a: Append assistant turn to working memory (keyed by thread_id)
    working_memory.append_turn(thread_id, 'assistant', response_data['response'])

    # Persist assistant turn to topic transcript (durable, searchable)
    try:
        from services import transcript_service
        transcript_service.append(channel, 'assistant', response_data['response'])
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
        compaction_service.check_and_compact(channel, _ctx_budget)
    except Exception as e:
        logging.debug(f"[DIGEST] Compaction check failed: {e}", exc_info=True)

    # Update conversation phase with Chalie's response so momentum and direction
    # reflect the full exchange, not just the user turn.
    try:
        from services.conversation_phase_service import get_conversation_phase_service
        _phase_svc_resp = get_conversation_phase_service()
        _phase_svc_resp.update(thread_id, response_data['response'], is_user=False, topic=channel)
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
            topic=channel,
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
            response_data['response'], channel, thread_id,
        )
        if _saveable:
            _save_svc.flag_saveable(
                thread_id, channel, _saveable['content_type'], exchange_id,
            )
    except Exception as _save_e:
        logging.debug(f"[DIGEST] Saveable content detection skipped: {_save_e}")

    # ═══════════════════════════════════════════════════════════
    # PHASE E: ASYNC FOLLOW-UP
    # ═══════════════════════════════════════════════════════════

    # Print the actual response to stdout for the user
    logging.info(f"\n{'='*60}")
    logging.info(f"Channel: {channel}")
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

    return f"Channel '{channel}' | Mode: {response_data['mode']} | Response generated in {response_data['generation_time']:.2f}s"
