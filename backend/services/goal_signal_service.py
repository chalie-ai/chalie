"""
Goal Signal Service — Post-message signal extraction and routing.

Two entry points:

1. extract_and_route_signals() — Called from digest_worker after response.
   Uses existing intent classification + embedding match. No LLM calls.

2. route_cognitive_signal() — Called from reasoning loop for cognitive signals
   (memory_pressure, new_knowledge, etc.).
"""

import json
import logging
import uuid
from typing import Dict, List, Optional

from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[GOAL SIGNAL]"

# Intent types from IntentClassifierService that indicate explicit user intent.
# These are computed upstream by the digest worker (deterministic, ~1ms) and passed
# in via the classification dict — no regex needed.
INTENT_TYPES = frozenset({'command', 'action'})

# Minimum text length to process (skip trivial messages like "ok", "thanks")
MIN_TEXT_LENGTH = 10

# Embedding similarity threshold for matching existing goals
GOAL_MATCH_THRESHOLD = 0.65
# Lower threshold for candidate goals — they need MORE evidence to mature,
# so they should be EASIER to match (not harder)
CANDIDATE_MATCH_THRESHOLD = 0.55
# Evidence count required for a candidate goal to graduate to active
CANDIDATE_GRADUATION_EVIDENCE = 3


def extract_and_route_signals(
    topic: str,
    user_text: str,
    classification: Optional[Dict] = None,
) -> List[Dict]:
    """
    Extract goal signals from a user message and route them.

    Called from digest_worker post-response. Lightweight, no LLM.

    Uses the intent classification already computed by IntentClassifierService
    (passed in via classification dict) to detect explicit user intent.
    Falls back to treating all non-trivial messages as potential signals
    for embedding matching against existing goals.

    Returns list of signals extracted (for observability).
    """
    if not user_text or len(user_text.strip()) < MIN_TEXT_LENGTH:
        return []

    signals = []

    # 1. Explicit intent detection (from upstream IntentClassifierService)
    # The digest worker already classifies intent as command/action/question/statement.
    # 'command' and 'action' indicate the user wants something done.
    intent_type = (classification or {}).get('intent_type', '')
    is_explicit_intent = intent_type in INTENT_TYPES

    if is_explicit_intent:
        signals.append({
            'signal_type': 'explicit_statement',
            'content': user_text.strip(),
            'source': f'topic:{topic}',
            'strength': 1.0,
        })

    # 2. Topic matching (embedding similarity against existing goals)
    # Use the lower candidate threshold to also match emergent candidate goals
    try:
        from services.goal_ecology_service import GoalEcologyService
        ecology = GoalEcologyService()
        matches = ecology.find_matching_goals(user_text, threshold=CANDIDATE_MATCH_THRESHOLD)

        for match in matches:
            # Apply stricter threshold for non-candidate goals
            if match.get('status') != 'candidate' and match.get('similarity', 0) < GOAL_MATCH_THRESHOLD:
                continue

            ecology.add_evidence(
                goal_id=match['id'],
                signal_type='topic_recurrence',
                content=user_text[:500],
                source=f'topic:{topic}',
                strength=match.get('similarity', 0.7),
            )
            signals.append({
                'signal_type': 'topic_recurrence',
                'content': user_text[:500],
                'goal_id': match['id'],
                'similarity': match.get('similarity', 0.0),
            })

            # Graduate candidate goals with enough evidence
            if match.get('status') == 'candidate':
                _try_graduate_candidate(ecology, match['id'])

    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Goal matching failed: {e}")

    # 2b. Enrich signals with ambient context
    _enrich_with_ambient_context(signals)

    # 3. Handle unmatched explicit signals — create stated goal or store for clustering
    if signals and signals[0]['signal_type'] == 'explicit_statement':
        # Check if any goal matched
        matched_goal = any(s.get('goal_id') for s in signals)

        if not matched_goal:
            # Explicit intent with no matching goal — create a stated goal
            try:
                from services.goal_ecology_service import GoalEcologyService
                ecology = GoalEcologyService()
                goal = ecology.create_goal(
                    description=user_text.strip()[:500],
                    type='stated',
                    timescale='short_term',
                )
                ecology.add_evidence(
                    goal_id=goal['id'],
                    signal_type='explicit_statement',
                    content=user_text.strip()[:500],
                    source=f'topic:{topic}',
                    strength=1.0,
                )
                signals.append({
                    'signal_type': 'goal_created',
                    'goal_id': goal['id'],
                    'goal_type': 'stated',
                })
                logger.info(
                    f"{LOG_PREFIX} Created stated goal from explicit intent: "
                    f"'{user_text[:60]}'"
                )
            except Exception as e:
                logger.debug(f"{LOG_PREFIX} Failed to create stated goal: {e}")

    elif not signals:
        # No explicit intent and no goal match — store as unmatched for pattern detection
        _store_unmatched_signal(user_text, topic)

    if signals:
        logger.debug(
            f"{LOG_PREFIX} Extracted {len(signals)} signals from message "
            f"(topic={topic})"
        )

    # Observability
    try:
        from services.metrics_service import MetricsService
        MetricsService().record_counter('goal_signals_extracted', len(signals))
    except Exception:
        pass

    return signals


def route_cognitive_signal(signal) -> None:
    """
    Route a cognitive signal (from reasoning loop) to the goal ecology.

    Converts signal content to goal evidence if it matches existing goals.
    Otherwise stores as unmatched signal for pattern detection.

    Args:
        signal: A ReasoningSignal-like object or dict with 'signal_type' and 'payload'.
    """
    try:
        # Handle both dataclass and dict formats
        if hasattr(signal, 'signal_type'):
            signal_type = signal.signal_type
            payload = signal.payload if hasattr(signal, 'payload') else {}
            source_id = signal.source_id if hasattr(signal, 'source_id') else None
        elif isinstance(signal, dict):
            signal_type = signal.get('signal_type', 'unknown')
            payload = signal.get('payload', {})
            source_id = signal.get('source_id')
        else:
            return

        # Extract meaningful content from signal
        content = _extract_signal_content(signal_type, payload)
        if not content:
            return

        # Try to match against existing goals (use lower threshold to include candidates)
        from services.goal_ecology_service import GoalEcologyService
        ecology = GoalEcologyService()
        matches = ecology.find_matching_goals(content, threshold=CANDIDATE_MATCH_THRESHOLD)

        # Filter: non-candidate goals must meet the stricter threshold
        filtered = [
            m for m in matches
            if m.get('status') == 'candidate' or m.get('similarity', 0) >= GOAL_MATCH_THRESHOLD
        ]

        if filtered:
            for match in filtered[:2]:  # Route to top 2 matches
                ecology.add_evidence(
                    goal_id=match['id'],
                    signal_type=_map_signal_type(signal_type),
                    content=content[:500],
                    source=source_id,
                    strength=match.get('similarity', 0.7),
                )
                # Graduate candidate goals with enough evidence
                if match.get('status') == 'candidate':
                    _try_graduate_candidate(ecology, match['id'])
            logger.debug(
                f"{LOG_PREFIX} Cognitive signal '{signal_type}' routed to "
                f"{len(filtered)} goals"
            )
        else:
            # Store as unmatched
            _store_unmatched_signal(content, signal_type, source_id)

    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Cognitive signal routing failed: {e}")


def _try_graduate_candidate(ecology, goal_id: str) -> None:
    """
    Check if a candidate goal has enough evidence to graduate to active.

    Graduation requires CANDIDATE_GRADUATION_EVIDENCE (3) evidence entries.
    On graduation, confidence is bumped and status transitions to 'active'.
    """
    try:
        goal = ecology.get_goal(goal_id)
        if not goal or goal.get('status') != 'candidate':
            return

        evidence_count = goal.get('evidence_count', 0)
        if evidence_count >= CANDIDATE_GRADUATION_EVIDENCE:
            from services.database_service import get_lightweight_db_service
            from services.time_utils import utc_now as _utc_now
            db = ecology.db
            now = _utc_now().isoformat()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE goals
                    SET status = 'active',
                        confidence = MAX(confidence, 0.5),
                        updated_at = ?
                    WHERE id = ? AND status = 'candidate'
                """, (now, goal_id))
                cursor.close()
            logger.info(
                f"{LOG_PREFIX} Candidate goal {goal_id[:8]} graduated to active "
                f"({evidence_count} evidence entries)"
            )
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Candidate graduation check failed: {e}")


def _extract_signal_content(signal_type: str, payload: Dict) -> Optional[str]:
    """Extract meaningful text content from a signal payload."""
    if isinstance(payload, str):
        return payload if len(payload) >= MIN_TEXT_LENGTH else None

    if isinstance(payload, dict):
        # Try common payload fields
        for key in ('content', 'text', 'message', 'description', 'gist', 'summary', 'topic'):
            val = payload.get(key)
            if val and isinstance(val, str) and len(val) >= MIN_TEXT_LENGTH:
                return val

        # Try to compose from multiple fields
        parts = []
        for key in ('topic', 'action', 'result'):
            val = payload.get(key)
            if val and isinstance(val, str):
                parts.append(val)
        if parts:
            composed = ' — '.join(parts)
            if len(composed) >= MIN_TEXT_LENGTH:
                return composed

    return None


def _map_signal_type(cognitive_signal_type: str) -> str:
    """Map cognitive signal types to goal evidence signal types."""
    mapping = {
        'new_knowledge': 'behavioral',
        'novel_observation': 'behavioral',
        'memory_pressure': 'ambient',
        'schedule_fired': 'temporal',
        'trait_changed': 'behavioral',
        'task_state_changed': 'behavioral',
        'thread_expired': 'ambient',
        'user_message': 'topic_recurrence',
    }
    return mapping.get(cognitive_signal_type, 'behavioral')


def _enrich_with_ambient_context(signals: List[Dict]) -> None:
    """
    Enrich extracted signals with current ambient context.

    Deep focus boosts signal strength (user is concentrating = higher value signal).
    Casual attention keeps default strength.
    Away/distracted has no effect (those messages are rare).
    """
    if not signals:
        return

    try:
        from services.ambient_inference_service import AmbientInferenceService
        inference = AmbientInferenceService()

        from services.client_context_service import ClientContextService
        ctx_service = ClientContextService()
        ctx = ctx_service.get()
        if not ctx:
            return

        state = inference.infer(ctx)
        attention = state.get('attention')
        tempo = state.get('tempo')

        for signal in signals:
            # Deep focus = higher signal strength (user is concentrating on this)
            if attention == 'deep_focus':
                signal['strength'] = min(1.0, signal.get('strength', 0.5) * 1.3)
                signal['ambient_context'] = 'deep_focus'
            elif attention == 'casual':
                signal['ambient_context'] = 'casual'

            # Routine tempo suggests ongoing interest (not a one-off)
            if tempo == 'routine':
                signal['ambient_context'] = signal.get('ambient_context', '') + ':routine'

    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Ambient enrichment skipped: {e}")


def _store_unmatched_signal(
    content: str,
    source: str = 'message',
    source_id: Optional[str] = None,
) -> None:
    """Store an unmatched signal in MemoryStore for later pattern detection."""
    try:
        from services.memory_store import get_shared_store
        from services.goal_ecology_service import UNMATCHED_SIGNAL_PREFIX, UNMATCHED_SIGNAL_TTL

        store = MemoryClientService.create_connection()
        signal_id = str(uuid.uuid4())[:12]
        key = f"{UNMATCHED_SIGNAL_PREFIX}{signal_id}"

        store.setex(key, UNMATCHED_SIGNAL_TTL, json.dumps({
            'content': content[:500],
            'signal_type': 'topic_recurrence',
            'source': source_id or source,
            'strength': 0.5,
            'timestamp': utc_now().isoformat(),
        }))

    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Failed to store unmatched signal: {e}")
