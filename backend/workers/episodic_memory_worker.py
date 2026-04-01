"""
Episodic Memory Worker - Generate episodes from conversation sessions.
Responsibility: Episode generation only (SRP).
"""

import json
import re
from datetime import datetime, timezone
from services.time_utils import utc_now
from services import ConfigService, EpisodicService, SalienceService
from services.llm_service import create_llm_service
from services.thread_conversation_service import ThreadConversationService
import logging


def check_readiness(topic: str, thread_id: str = None, min_exchanges: int = 3, timeout_minutes: int = 10) -> tuple[bool, str, list]:
    """
    Check if topic is ready for episodic memory consolidation.

    Conditions:
    - 3+ raw conversation exchanges OR
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
    exchanges = [e for e in conversation_service.get_conversation_history(thread_id) if e]

    # Condition 1: enough raw turns
    if len(exchanges) >= min_exchanges:
        return True, f"{len(exchanges)} exchanges ready", exchanges

    # Condition 2: 10+ minutes since earliest exchange
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
            return True, f"timeout reached, {len(exchanges)} exchanges", exchanges

    if not exchanges:
        return False, "No exchanges", []

    return False, f"Not ready: {len(exchanges)} exchanges, waiting for more or timeout", []


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
            logging.warning(f"[EPISODIC] Topic '{topic}' timeout reached with 0 exchanges — skipping episode creation")
            return "Skipped — timeout reached with 0 exchanges"

        # Load configs
        from services.database_service import get_shared_db_service

        config = ConfigService.resolve_agent_config("episodic-memory")
        prompt_template = ConfigService.get_agent_prompt("episodic-memory")

        # Initialize services
        database_service = get_shared_db_service()
        storage_service = EpisodicService(database_service)
        ollama_service = create_llm_service(config)
        salience_service = SalienceService(config)
        conversation_service = ThreadConversationService()

        # Build session data from exchanges
        start_time = (exchanges[0].get('prompt') or {}).get('time', utc_now().isoformat())
        end_time = (exchanges[-1].get('response') or {}).get('time', utc_now().isoformat())

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

        # Goal emergence check — does this episode cluster with others?
        try:
            _check_goal_emergence(episode_id, episode_data.get('gist', ''), embedding, topic)
        except Exception as e:
            logging.debug(f"[EPISODIC] Goal emergence check non-fatal: {e}")

        try:
            from services.cognitive_drift_engine import emit_reasoning_signal, ReasoningSignal
            emit_reasoning_signal(ReasoningSignal(
                signal_type='episode_created',
                source='episodic_memory_worker',
                topic=topic,
                content=episode_data.get('gist', '')[:200],
                activation_energy=0.5,
            ))
        except Exception:
            pass

        # Feed the semantic consolidation tracker so concepts get created
        try:
            from services.semantic_consolidation_tracker import SemanticConsolidationTracker
            tracker = SemanticConsolidationTracker()
            tracker.increment_episode_count()
            tracker.record_episode_salience(salience_float)

            tracker.should_trigger_consolidation(salience_float)
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
    Format session data for LLM prompt using raw conversation turns.

    Args:
        session_data: Dict with topic, exchanges, start_time, end_time

    Returns:
        Formatted string for LLM prompt with user/assistant conversation turns
    """
    lines = []
    lines.append(f"Session Duration: {session_data['start_time']} to {session_data.get('end_time', 'now')}")
    lines.append("\n## Conversation from Session")

    for i, exchange in enumerate(session_data['exchanges'], 1):
        prompt = exchange.get('prompt', {})
        response = exchange.get('response', {})

        user_msg = prompt.get('message', '') if isinstance(prompt, dict) else str(prompt)
        assistant_msg = response.get('message', '') if isinstance(response, dict) else str(response)

        if not user_msg and not assistant_msg:
            continue

        lines.append(f"\n--- Exchange {i} ---")
        if user_msg:
            lines.append(f"User: {user_msg}")
        if assistant_msg:
            lines.append(f"Assistant: {assistant_msg}")

        # Include tool steps if present
        if exchange.get('steps'):
            lines.append(f"Actions: {exchange['steps']}")

    return "\n".join(lines)


def _check_goal_emergence(episode_id: str, gist: str, embedding: list, topic: str):
    """
    Check if a newly created episode clusters with existing episodes,
    suggesting an emergent goal.

    Flow:
    1. KNN search episode embedding against episodes_vec (top 10)
    2. Apply temporal decay + salience weighting
    3. If cluster found, check if an existing goal already covers these episodes
       - Yes → reinforce that goal + append this episode to derived_from
       - No → if cluster >= 2 episodes, run tiny LLM to assess goal emergence
    """
    import math

    if not embedding or not gist:
        return

    from services.embedding_utils import pack_embedding
    from services.database_service import DatabaseService
    from services.time_utils import utc_now, parse_utc

    db = DatabaseService()
    packed = pack_embedding(embedding)

    # 1. KNN search against episodes_vec
    with db.connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT e.id, e.gist, e.salience, e.created_at, v.distance
            FROM episodes_vec v
            JOIN episodes e ON e.id = v.rowid
            WHERE v.embedding MATCH ? AND k = 10
              AND e.deleted_at IS NULL
              AND e.id != ?
            ORDER BY v.distance
        """, (packed, episode_id))
        rows = cursor.fetchall()
        cursor.close()

    if not rows:
        return

    # 2. Score with temporal decay and salience
    now = utc_now()
    scored = []
    SIMILARITY_THRESHOLD = 0.60

    for row in rows:
        ep_id, ep_gist, salience, created_at_str, distance = row
        similarity = max(0.0, 1.0 - (distance or 1.0))
        if similarity < SIMILARITY_THRESHOLD:
            continue

        try:
            created_at = parse_utc(created_at_str)
            days_old = (now - created_at).total_seconds() / 86400
        except Exception:
            days_old = 7.0  # default if parse fails

        # Temporal decay: 14-day half-life
        decay = math.exp(-days_old / 14.0)
        # Normalize salience from 1-10 scale to 0-1
        sal_norm = (salience or 5) / 10.0
        score = similarity * sal_norm * decay

        scored.append({
            'id': ep_id,
            'gist': ep_gist,
            'similarity': similarity,
            'score': score,
        })

    if not scored:
        return

    # Sort by score descending, cap at 5
    scored.sort(key=lambda x: x['score'], reverse=True)
    cluster = scored[:5]
    cluster_episode_ids = [ep['id'] for ep in cluster]

    # 3. Check if an existing goal already covers any of these episodes
    from services.goal_ecology_service import GoalEcologyService
    ecology = GoalEcologyService()

    existing_goals = ecology.find_goals_by_episode_ids(cluster_episode_ids)
    if existing_goals:
        # Reinforce the best-matching goal and append this episode
        goal = existing_goals[0]
        ecology.add_evidence(
            goal_id=goal['id'],
            signal_type='episode_cluster',
            content=gist[:500],
            source=f'episode:{episode_id}',
            strength=0.8,
        )
        ecology.append_derived_from(goal['id'], episode_id)
        logging.info(
            f"[EPISODIC] Reinforced goal {goal['id'][:8]} with episode {episode_id[:8]}"
        )
        return

    # 4. No existing goal covers this cluster — need >= 2 episodes to synthesize
    if len(cluster) < 2:
        return

    # Tiny LLM call to assess goal emergence
    from services.background_llm_queue import create_background_llm_proxy

    gists = [f"- {ep['gist']}" for ep in cluster]
    gists.append(f"- {gist}")  # include current episode
    gists_text = "\n".join(gists)

    llm = create_background_llm_proxy("goal-emergence")
    response = llm.send_message(
        system_prompt=(
            "You analyze episode clusters for emergent goals. "
            "An episode is a narrative summary of a conversation segment. "
            "Respond with EXACTLY this format:\n"
            "GOAL: <one-sentence goal description>\n"
            "CONFIDENCE: <low|medium|high>\n\n"
            "If no coherent goal emerges, respond with:\n"
            "GOAL: none\nCONFIDENCE: low"
        ),
        user_message=(
            f"Do these episodes suggest an emerging goal?\n\n{gists_text}"
        ),
    )

    if not response or not response.text:
        return

    # Parse response
    text = response.text.strip()
    goal_line = None
    confidence = 'low'

    for line in text.split('\n'):
        line = line.strip()
        if line.upper().startswith('GOAL:'):
            goal_line = line[5:].strip()
        elif line.upper().startswith('CONFIDENCE:'):
            confidence = line[11:].strip().lower()

    if not goal_line or goal_line.lower() == 'none' or confidence == 'low':
        return

    # Create the emergent goal
    derived_from_ids = cluster_episode_ids + [episode_id]
    goal = ecology.create_goal(
        description=goal_line,
        type='emergent',
        derived_from=derived_from_ids,
    )

    logging.info(
        f"[EPISODIC] Created emergent goal '{goal_line[:60]}' "
        f"(confidence={confidence}, episodes={len(derived_from_ids)})"
    )
