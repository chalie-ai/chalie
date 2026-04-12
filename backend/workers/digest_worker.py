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

It also provides lightweight helpers for interactive tool dialog
(``process_tool_dialog``, ``store_tool_dialog_memory``).

.. deprecated::
    This module is scheduled for removal. The new message-processing model
    (see ``/Volumes/llm/chalie-plans/message-processing.md``) splits this
    5-phase pipeline into per-channel ``MessageProcessor`` subclasses. Each
    subclass owns its own pre-turn preparation, runs the shared
    ``MessageProcessor.send()`` (LLM + ACT loop + compaction + atomic store),
    and fires its own ``postTurn()`` service fan-out (contradiction detection
    via memory skill, phase updates, metrics, …). Nothing remains
    centralised. This file will be deleted once every channel has been
    migrated. Do not add new callers or extend the existing pipeline here.
"""

import json
import os
import time
import logging

from services import FrontalCortexService
from services.world_state_service import WorldStateService
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
    load_configs,
)

# Post-exchange hooks — canonical implementations live in post_exchange_hooks.py;
# re-exported here so existing ``from workers.digest_worker import _run_*`` still works.
from workers.post_exchange_hooks import (         # noqa: F401
    _run_iip_hook,
    _run_belief_correction_hook,
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

    Memory is stored with memory_durability='tool_internal' so the user can ask
    "did that tool ask you something?" but the tool dialog doesn't alter long-term
    behavioral patterns.

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
    No-op — retained only for the webhook tool path in ``api/tools.py`` which
    still calls this function after an interactive tool dialog completes.

    Historically this routed the final-turn summary into
    ``enqueue_trait_extraction`` — the only storage path it had. That pipeline
    was deleted on 2026-04-11 (trait-extraction RIP); there is no replacement
    writer because the LLM-native memory skill already captures tool-initiated
    traits inline. This function is kept as a deliberate no-op so the webhook
    path does not need to change in the narrow trait-extraction rip. It will
    be deleted alongside the full ``digest_worker`` rip.
    """
    return


def digest_worker(text: str, metadata: dict = None) -> str:
    """
    Main worker function that processes prompts through classification and response generation.

    Pipeline: Phase A (immediate commit) → Phase B (retrieval) → Phase C (route + generate)
              → Phase D (post-response commit) → Phase E (async follow-up)

    Proactive drift messages go through full routing but skip user input logging.
    """
    metadata = metadata or {}

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

    # Mark channel as busy
    _busy_store = _channel_store
    _busy_store.setex(f"thread_busy:{thread_id}", 30, "1")

    # Step 2a: Initialize services
    world_state_service = WorldStateService()

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
    context_warmth = calculate_context_warmth(
        working_memory_len=0,
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
    store = _busy_store
    raw_fok = store.get(f"fok:{channel}") if channel else None
    fok = float(raw_fok) if raw_fok else 0.0
    fok_score = min(1.0, fok / 5.0)

    memory_confidence = (
        0.4 * fok_score
        + 0.4 * context_warmth
    )
    if classification_result.get('is_new_topic', False):
        memory_confidence *= 0.7
    memory_confidence = round(memory_confidence, 3)

    # Step 10: Unified generation (no gate, no mode routing)
    routing_result = None
    try:
        _nlp = compute_nlp_signals(text)
        _signals = {
            'context_warmth': context_warmth,
            'working_memory_turns': 0,
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
