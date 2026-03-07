"""
Episodic Memory Worker - Generate episodes from conversation sessions.
Responsibility: Episode generation only (SRP).
"""

import json
import time
import re
from datetime import datetime, timezone
from services.time_utils import utc_now
from services import ConfigService, DatabaseService, EpisodicStorageService, SalienceService
from services.llm_service import create_llm_service
from services.thread_conversation_service import ThreadConversationService
import logging


def check_readiness(topic: str, thread_id: str = None, min_exchanges: int = 3, timeout_minutes: int = 10) -> tuple[bool, str, list]:
    """
    Check if topic is ready for episodic memory consolidation.

    Conditions:
    - 3+ exchanges with memory_chunk OR
    - 10+ minutes since earliest exchange

    Args:
        topic: Topic name
        thread_id: Thread ID for conversation history lookup
        min_exchanges: Minimum exchanges to trigger (default 3)
        timeout_minutes: Timeout in minutes to trigger (default 10)

    Returns:
        Tuple of (ready: bool, reason: str, exchanges: list)
    """
    if not thread_id:
        return False, "No thread_id provided", []
    conversation_service = ThreadConversationService()
    exchanges = conversation_service.get_conversation_history(thread_id)

    # Filter only enriched exchanges (guard against None entries from storage)
    enriched = [e for e in exchanges if e and e.get('memory_chunk')]

    # Condition 1: 3+ enriched exchanges
    if len(enriched) >= min_exchanges:
        return True, f"{len(enriched)} enriched exchanges ready", enriched

    # Condition 2: 10+ minutes since earliest exchange — check ALL exchanges, not just enriched,
    # so a topic with 0 enriched exchanges can still time out rather than retrying forever.
    earliest_time = None
    for exchange in exchanges:
        resp = exchange.get('response')
        response_time_str = resp.get('time') if isinstance(resp, dict) else None
        if response_time_str:
            # Parse time (format: "2026-01-22 04:52")
            try:
                exchange_time = datetime.strptime(response_time_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                if earliest_time is None or exchange_time < earliest_time:
                    earliest_time = exchange_time
            except ValueError:
                continue

    if earliest_time:
        elapsed = utc_now() - earliest_time
        if elapsed.total_seconds() >= timeout_minutes * 60:
            return True, f"timeout reached, {len(enriched)} enriched", enriched

    if not enriched:
        return False, "No enriched exchanges", []

    return False, f"Not ready: {len(enriched)} exchanges, waiting for more or timeout", []


def _extract_json(text: str) -> str:
    """
    Extract JSON from text, handling markdown code fences.

    Strips markdown fences (```json ... ``` or ``` ... ```), handles commentary
    before/after JSON, and multiple fenced blocks (takes first).

    Args:
        text: Text potentially containing fenced JSON

    Returns:
        Cleaned JSON string (may still need json.loads() validation)
    """
    text = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    return match.group(1).strip() if match else text


def _safe_json_load(text: str) -> dict | None:
    """
    Safely load JSON with graceful fallback for parse errors.

    Extracts JSON from markdown fences and attempts parsing.
    On failure, logs error and returns None instead of crashing.

    Args:
        text: Text potentially containing JSON (may be fenced)

    Returns:
        Parsed dict on success, None on parse failure
    """
    cleaned = _extract_json(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logging.error("[EPISODIC] Failed to parse JSON from LLM output")
        logging.debug(f"[EPISODIC] Raw output: {cleaned[:500]}")
        return None


def episodic_memory_worker(job_data: dict) -> str:
    """
    Generate episodic memory from a conversation topic.

    Called by EpisodicMemoryObserver (primary) or thread_expiry_service (fallback).
    The caller has already verified readiness via signal density or timeout.
    The worker still runs check_readiness() as a safety check but does not requeue.

    Args:
        job_data: {
            'topic': str,
            'thread_id': str (optional)
        }

    Returns:
        Result string with episode ID or skip reason
    """
    topic = job_data['topic']
    thread_id = job_data.get('thread_id') or None
    logging.info(f"Episodic memory worker: Processing topic '{topic}' (thread: {thread_id})")

    try:
        # Safety check — caller should have verified, but guard against edge cases
        ready, reason, exchanges = check_readiness(topic, thread_id=thread_id)
        if not ready:
            logging.info(f"Topic '{topic}' not ready (safety check): {reason}")
            return f"Skipped - {reason}"

        logging.info(f"Topic '{topic}' ready for consolidation: {reason}, {len(exchanges)} exchanges")

        if not exchanges:
            logging.warning(f"[EPISODIC] Topic '{topic}' timeout reached with 0 enriched exchanges — skipping episode creation")
            return "Skipped — timeout reached with 0 enriched exchanges"

        # Load configs
        from services.database_service import get_lightweight_db_service

        config = ConfigService.resolve_agent_config("episodic-memory")
        prompt_template = ConfigService.get_agent_prompt("episodic-memory")

        # Initialize services
        database_service = get_lightweight_db_service()
        storage_service = EpisodicStorageService(database_service)
        ollama_service = create_llm_service(config)
        salience_service = SalienceService(config)
        conversation_service = ThreadConversationService()

        # Build session data from exchanges
        start_time = (exchanges[0].get('prompt') or {}).get('time', datetime.now().isoformat())
        end_time = (exchanges[-1].get('response') or {}).get('time', datetime.now().isoformat())

        session_data = {
            'topic': topic,
            'exchanges': exchanges,
            'start_time': start_time,
            'end_time': end_time
        }

        # Prepare context for LLM
        conversation_context = _format_session_for_llm(session_data)

        # Inject variables into prompt
        prompt = prompt_template.replace('{{session_context}}', conversation_context)

        # Generate episode structure using LLM
        logging.info("Generating episode structure with LLM")
        response = ollama_service.send_message("", prompt).text

        # Parse JSON response with safe fallback
        episode_data = _safe_json_load(response)
        if not episode_data:
            logging.warning(f"[EPISODIC] Skipping episode for topic '{topic}' — invalid LLM JSON")
            return "Skipped — LLM returned unparseable JSON"

        # Validate episode structure
        required_fields = ['intent', 'context', 'action', 'emotion', 'outcome', 'gist', 'salience_factors']
        for field in required_fields:
            if field not in episode_data:
                raise ValueError(f"Missing required field: {field}")

        # Extract salience factors
        salience_factors = episode_data.get('salience_factors', {})

        # Calculate salience from LLM-provided factors (0-3 scale normalized to 0-1)
        logging.debug("Computing salience score from LLM factors")
        salience_float = salience_service.calculate_salience(salience_factors)

        # Scale from [0.1, 1.0] to [1, 10] and convert to integer for database
        salience = max(1, min(10, round(salience_float * 10)))
        episode_data['salience'] = salience

        # Set initial freshness to salience (will be computed dynamically on retrieval)
        episode_data['freshness'] = salience

        # Generate embedding for episode (gist + intent + outcome + emotion)
        logging.debug("Generating embedding for episode")
        embedding_model = config.get('embedding_model', 'nomic-embed-text')
        embedding_dimensions = config.get('embedding_dimensions', 256)

        # Construct rich embedding text
        intent = episode_data.get('intent', {})
        emotion = episode_data.get('emotion', {})
        embedding_text_parts = [
            f"Gist: {episode_data['gist']}",
            f"Intent: {intent.get('type', 'unknown')} ({intent.get('direction', 'unknown')})",
            f"Outcome: {episode_data.get('outcome', 'none')}",
            f"Emotion: {emotion.get('valence', 'neutral')} ({emotion.get('intensity', 'low')})"
        ]
        embedding_text = " | ".join(embedding_text_parts)

        logging.debug(f"Embedding text: {embedding_text}")
        from services.embedding_service import get_embedding_service
        emb_service = get_embedding_service()
        embedding = emb_service.generate_embedding(embedding_text)
        episode_data['embedding'] = embedding

        # Add metadata
        episode_data['topic'] = session_data['topic']
        episode_data['exchange_id'] = session_data.get('exchange_id')

        # Store episode
        episode_id = storage_service.store_episode(episode_data)

        logging.info(f"Generated and stored episode {episode_id} for topic '{topic}'")

        # Feed the semantic consolidation tracker so concepts get created
        try:
            from services.semantic_consolidation_tracker import SemanticConsolidationTracker
            tracker = SemanticConsolidationTracker()
            tracker.increment_episode_count()
            tracker.record_episode_salience(salience_float)

            should_trigger, trigger_reason = tracker.should_trigger_consolidation(salience_float)
            if should_trigger:
                from services.config_service import ConfigService as _CS
                _sem_config = _CS.connections()
                _sem_queue_name = _sem_config.get("memory", {}).get("queues", {}).get(
                    "semantic_consolidation_queue", {}
                ).get("name", "semantic_consolidation_queue")
                _sem_store = MemoryClientService.create_connection(decode_responses=False)
                _sem_queue = Queue(_sem_queue_name, connection=_sem_store)
                _sem_queue.enqueue(
                    'workers.semantic_consolidation_worker.semantic_consolidation_worker',
                    {
                        "type": "batch_consolidation",
                        "trigger": trigger_reason,
                        "timestamp": time.time(),
                    }
                )
                tracker.reset_episode_count()
                logging.info(f"[EPISODIC] Enqueued semantic consolidation (trigger={trigger_reason})")
        except Exception as tracker_err:
            logging.warning(f"[EPISODIC] Consolidation tracker error (non-fatal): {tracker_err}")

        # Cleanup: Remove consolidated exchanges from thread conversation
        if thread_id:
            exchange_ids = [e.get('id') or e.get('prompt', {}).get('id') for e in exchanges]
            exchange_ids = [eid for eid in exchange_ids if eid]
            conversation_service.remove_exchanges(thread_id, exchange_ids)
            logging.info(f"Removed {len(exchange_ids)} consolidated exchanges from thread '{thread_id}'")

            # Trim working memory — consolidated turns now captured in episode
            try:
                from services.working_memory_service import WorkingMemoryService
                WM_BASELINE = 4  # Keep last 4 turns post-consolidation
                wm = WorkingMemoryService()
                wm_key = wm._get_memory_key(thread_id)
                wm.store.ltrim(wm_key, -(WM_BASELINE * 2), -1)
                logging.info(f"Trimmed working memory for thread '{thread_id}' after consolidation")
            except Exception as wm_err:
                logging.debug(f"[EPISODIC] WM trim after consolidation failed (non-fatal): {wm_err}")

        # Close database pool
        database_service.close_pool()

        return f"Episode {episode_id} created and {len(exchange_ids)} exchanges consolidated"

    except Exception as e:
        logging.error(f"Episodic memory worker failed: {e}", exc_info=True)
        raise




def _format_session_for_llm(session_data: dict) -> str:
    """
    Format session data for LLM prompt using memory_chunk enrichments only.

    Args:
        session_data: Dict with topic, exchanges (with memory_chunk enrichments), start_time, end_time

    Returns:
        Formatted string for LLM prompt with memory chunk data
    """
    lines = []
    lines.append(f"Session Duration: {session_data['start_time']} to {session_data.get('end_time', 'now')}")
    lines.append("\nMemory Chunks:")

    for i, exchange in enumerate(session_data['exchanges'], 1):
        memory_chunk = exchange.get('memory_chunk', {})

        if not memory_chunk:
            logging.warning(f"Exchange {i} missing memory_chunk - should have been checked earlier")
            continue

        lines.append(f"\n--- Exchange {i} Memory Chunk ---")

        # Include scope
        scope = memory_chunk.get('scope', 'N/A')
        lines.append(f"Scope: {scope}")

        # Include emotion
        emotion = memory_chunk.get('emotion', {})
        if emotion:
            emotion_type = emotion.get('type', 'N/A')
            emotion_intensity = emotion.get('intensity', 'N/A')
            lines.append(f"Emotion: {emotion_type} (intensity: {emotion_intensity})")

        # Include gists
        gists = memory_chunk.get('gists', [])
        if gists:
            lines.append("Gists:")
            for gist in gists:
                gist_type = gist.get('type', 'unknown')
                content = gist.get('content', '')
                confidence = gist.get('confidence', 0)
                lines.append(f"  - [{gist_type}] {content} (confidence: {confidence})")

        # Include steps if present
        if 'steps' in exchange:
            lines.append(f"Actions: {exchange['steps']}")

    return "\n".join(lines)
