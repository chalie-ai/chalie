"""
Cognitive Reflex Service — learned fast path via semantic abstraction.

Mirrors human automaticity: repeated simple queries build semantic clusters
(rolling-average centroids in pgvector). Once a cluster has enough evidence,
future similar queries skip the full pipeline and respond via a lightweight LLM call.

Learning cycle:
  1. OBSERVATION — heuristic pre-screen identifies candidates, full pipeline runs,
     outcome recorded (was the pipeline actually useful?)
  2. CLUSTERING — similar queries merge into clusters via rolling-average centroid
  3. ACTIVATION — cluster confidence > 0.85 + enough observations → fast path fires
  4. SELF-CORRECTION — user corrections / shadow validation disable bad clusters
"""

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

CLUSTER_DISTANCE_THRESHOLD = 0.35   # cosine distance; similarity > 0.65
ROLLING_AVG_CAP = 20               # cap on rolling average denominator
MAX_SAMPLE_QUERIES = 5             # keep last N queries per cluster for observability
SHADOW_VALIDATION_RATE = 0.10      # 10% of activations get shadow validation
PENDING_VALIDATION_TTL = 300       # 5 minutes TTL for pending validation

# Activation thresholds (per-cluster)
MIN_CONFIDENCE = 0.85              # 85%+ of cluster queries didn't need pipeline
MIN_OBSERVATIONS = 5               # enough data points
MIN_SUCCESSES = 3                  # at least 3 validated successes
MAX_FAILURE_RATE = 0.20            # fast path isn't producing bad results
MAX_STALE_DAYS = 30                # cluster must have been seen within 30 days

# ─── Heuristic pre-screen patterns ───────────────────────────────────────────

_ANAPHORIC = re.compile(
    r'\b(it|that|this|those|these|them)\b', re.IGNORECASE
)
_PERSONAL_MEMORY = re.compile(
    r'\b(my|I|me|we|our|mine)\b', re.IGNORECASE
)
_CONVERSATIONAL_REF = re.compile(
    r'\b(you said|earlier|before|last time|remember when)\b', re.IGNORECASE
)
_TOOL_ACTION = re.compile(
    r'\b(search|find|look up|remind|schedule|set a|create a|add a|delete|remove)\b', re.IGNORECASE
)
_FRESHNESS = re.compile(
    r'\b(latest|current|today|now|recent|news|tonight|yesterday|tomorrow)\b', re.IGNORECASE
)
_REASONING = re.compile(
    r'\b(why|explain|how does|what causes|prove|derive|analyze|evaluate)\b', re.IGNORECASE
)
_COMPARATIVE = re.compile(
    r'\b(difference between|compare|vs\.?|better than|worse than|pros and cons)\b', re.IGNORECASE
)
_UNCERTAINTY = re.compile(
    r'\b(estimate|approximate|roughly|about how)\b', re.IGNORECASE
)
_MULTI_CLAUSE = re.compile(
    r'\b(and also|but also|then|however|additionally|furthermore)\b', re.IGNORECASE
)
_URL_PRESENT = re.compile(r'https?://', re.IGNORECASE)

# Max word count for candidate queries
_MAX_WORDS = 15
# Max context warmth for candidate queries
_MAX_WARMTH = 0.5

# Correction / rephrase patterns (reused from triage_calibration_service)
_CORRECTION_PATTERNS = [
    re.compile(r'\b(no,?\s+(that\'?s|I meant|what I want)|not what I|wrong|incorrect|try again)\b', re.IGNORECASE),
    re.compile(r'\b(I said|that\'?s not|you misunderstood|not quite|you missed)\b', re.IGNORECASE),
]
_REPHRASE_PATTERNS = [
    re.compile(r'\b(same question|asked (you |this )?before|what I (just |already )?said|I already told you)\b', re.IGNORECASE),
    re.compile(r'\b(what I meant (was|is)|let me rephrase|what I\'m saying is|to clarify)\b', re.IGNORECASE),
]


@dataclass
class ReflexResult:
    """Result of a reflex check."""
    is_candidate: bool        # Passed heuristic pre-screen
    can_activate: bool        # Semantic match found with sufficient confidence
    confidence: float         # Cluster confidence (0.0-1.0)
    cluster_id: Optional[int] # Matched cluster ID (for validation tracking)
    observations: int         # Cluster's times_seen (for logging)
    embedding: Optional[list] # Pre-computed embedding (reusable downstream)
    reasoning: str            # Human-readable explanation


class CognitiveReflexService:
    """
    Learned fast-path service for self-contained queries.

    Uses pgvector HNSW index for O(1) nearest-neighbor lookup against
    rolling-average centroids. Same proven pattern as TopicClassifierService.
    """

    def __init__(self, db=None, redis=None):
        """
        Args:
            db: DatabaseService instance (uses shared if None)
            redis: Redis connection (uses shared if None)
        """
        if db is None:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
        self.db = db

        if redis is None:
            from services.redis_client import RedisClientService
            redis = RedisClientService.create_connection()
        self.redis = redis

    # ─── Public interface ─────────────────────────────────────────────────

    def check(self, text: str, context_warmth: float) -> ReflexResult:
        """
        Check if a query can use the reflex fast path.

        1. Heuristic pre-screen (~1ms) — reject obvious non-candidates
        2. Compute embedding (~10-50ms) — for semantic similarity lookup
        3. Semantic reflex lookup (~5-20ms, pgvector) — find nearest cluster
        4. Check cluster confidence — enough evidence to trust fast path?

        Args:
            text: User message text
            context_warmth: Current conversation warmth (0.0-1.0)

        Returns:
            ReflexResult with activation decision + pre-computed embedding
        """
        # 1. Heuristic pre-screen
        if not self._is_candidate(text, context_warmth):
            return ReflexResult(
                is_candidate=False, can_activate=False,
                confidence=0.0, cluster_id=None,
                observations=0, embedding=None,
                reasoning='Failed heuristic pre-screen',
            )

        # 2. Compute embedding
        try:
            from services.embedding_service import get_embedding_service
            embedding = get_embedding_service().generate_embedding(text)
        except Exception as e:
            logger.debug(f"[REFLEX] Embedding generation failed: {e}")
            return ReflexResult(
                is_candidate=True, can_activate=False,
                confidence=0.0, cluster_id=None,
                observations=0, embedding=None,
                reasoning=f'Embedding generation failed: {e}',
            )

        # 3. Semantic reflex lookup
        match = self._find_matching_reflex(embedding)
        if not match:
            return ReflexResult(
                is_candidate=True, can_activate=False,
                confidence=0.0, cluster_id=None,
                observations=0, embedding=embedding,
                reasoning='No matching reflex cluster found',
            )

        # 4. Check cluster confidence
        confidence = match['times_unnecessary'] / max(match['times_seen'], 1)
        times_activated = match['times_activated'] or 0
        failure_rate = match['times_failed'] / max(times_activated, 1)
        last_seen = match['last_seen']
        last_seen_days = (datetime.now(timezone.utc) - last_seen).days if last_seen else 999

        can_activate = (
            confidence >= MIN_CONFIDENCE
            and match['times_seen'] >= MIN_OBSERVATIONS
            and match['times_succeeded'] >= MIN_SUCCESSES
            and failure_rate <= MAX_FAILURE_RATE
            and last_seen_days <= MAX_STALE_DAYS
        )

        if can_activate:
            reasoning = (
                f"Cluster {match['id']} activated: confidence={confidence:.2f}, "
                f"seen={match['times_seen']}, succeeded={match['times_succeeded']}, "
                f"failure_rate={failure_rate:.2f}"
            )
        else:
            parts = []
            if confidence < MIN_CONFIDENCE:
                parts.append(f"confidence={confidence:.2f}<{MIN_CONFIDENCE}")
            if match['times_seen'] < MIN_OBSERVATIONS:
                parts.append(f"seen={match['times_seen']}<{MIN_OBSERVATIONS}")
            if match['times_succeeded'] < MIN_SUCCESSES:
                parts.append(f"succeeded={match['times_succeeded']}<{MIN_SUCCESSES}")
            if failure_rate > MAX_FAILURE_RATE:
                parts.append(f"failure_rate={failure_rate:.2f}>{MAX_FAILURE_RATE}")
            if last_seen_days > MAX_STALE_DAYS:
                parts.append(f"stale={last_seen_days}d>{MAX_STALE_DAYS}d")
            reasoning = f"Cluster {match['id']} not ready: {', '.join(parts)}"

        return ReflexResult(
            is_candidate=True,
            can_activate=can_activate,
            confidence=confidence,
            cluster_id=match['id'],
            observations=match['times_seen'],
            embedding=embedding,
            reasoning=reasoning,
        )

    def record_observation(self, text: str, embedding: list, was_useful: bool):
        """
        Learn from a full-pipeline run. Updates or creates a reflex cluster.

        Called after the full pipeline completes for a reflex candidate.
        If the pipeline was unnecessary → the cluster evidence grows.

        Args:
            text: Original query text
            embedding: Pre-computed embedding (from check())
            was_useful: True if the full pipeline actually added value
        """
        match = self._find_matching_reflex(embedding)

        if match:
            self._merge_into_cluster(match, text, embedding, was_useful)
        else:
            self._create_cluster(text, embedding, was_useful)

    def record_activation(self, cluster_id: int):
        """Record that the fast path was activated for a cluster."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE cognitive_reflexes
                    SET times_activated = times_activated + 1,
                        last_activated = NOW()
                    WHERE id = %s
                """, (cluster_id,))
                cursor.close()
        except Exception as e:
            logger.warning(f"[REFLEX] Failed to record activation: {e}")

    def set_pending_validation(self, thread_id: str, cluster_id: int):
        """Store pending validation state — next user message checked for correction."""
        try:
            self.redis.setex(
                f"reflex:pending:{thread_id}",
                PENDING_VALIDATION_TTL,
                str(cluster_id),
            )
        except Exception as e:
            logger.debug(f"[REFLEX] Failed to set pending validation: {e}")

    def check_pending_validation(self, thread_id: str, next_message: str):
        """
        Check if the previous reflex response needs correction.

        Called at the start of each digest cycle — if a pending validation
        exists for this thread, examine the new message for correction signals.
        """
        try:
            key = f"reflex:pending:{thread_id}"
            cluster_id_str = self.redis.get(key)
            if not cluster_id_str:
                return

            # Consume the pending validation (one-shot)
            self.redis.delete(key)
            cluster_id = int(cluster_id_str)

            is_correction = self._is_correction(next_message)

            with self.db.connection() as conn:
                cursor = conn.cursor()
                if is_correction:
                    cursor.execute("""
                        UPDATE cognitive_reflexes
                        SET times_failed = times_failed + 1
                        WHERE id = %s
                    """, (cluster_id,))
                    logger.info(f"[REFLEX] Correction detected → cluster {cluster_id} times_failed++")
                else:
                    cursor.execute("""
                        UPDATE cognitive_reflexes
                        SET times_succeeded = times_succeeded + 1
                        WHERE id = %s
                    """, (cluster_id,))
                cursor.close()

        except Exception as e:
            logger.debug(f"[REFLEX] Pending validation check failed: {e}")

    def evaluate_pipeline_utility(self, triage_result, assembled_context) -> bool:
        """
        Determine whether the full pipeline was actually useful for this query.

        Pipeline was UNNECESSARY (returns False) if ALL of:
          - triage mode is RESPOND (not ACT, CLARIFY, etc.)
          - no tools or skills selected
          - high internal confidence (≥ 0.8)
          - sparse/empty assembled context (< 100 tokens)

        Args:
            triage_result: TriageResult from cognitive triage
            assembled_context: Dict from context assembly service

        Returns:
            bool: True if pipeline was useful, False if it was unnecessary
        """
        if triage_result is None:
            return True  # Can't evaluate — assume useful

        # Check mode
        mode = getattr(triage_result, 'mode', None) or triage_result.get('mode', '') if isinstance(triage_result, dict) else getattr(triage_result, 'mode', '')
        if mode != 'RESPOND':
            return True  # Non-RESPOND modes always need the pipeline

        # Check tools/skills
        tools = getattr(triage_result, 'tools', None) or (triage_result.get('tools', []) if isinstance(triage_result, dict) else [])
        skills = getattr(triage_result, 'skills', None) or (triage_result.get('skills', []) if isinstance(triage_result, dict) else [])
        if tools or skills:
            return True  # Tool/skill selection needed the pipeline

        # Check internal confidence
        conf = getattr(triage_result, 'confidence_internal', None)
        if conf is None and isinstance(triage_result, dict):
            conf = triage_result.get('confidence_internal', 0.0)
        if conf is not None and conf < 0.8:
            return True  # Low confidence → pipeline was useful for deliberation

        # Check context sparsity
        if assembled_context and isinstance(assembled_context, dict):
            total_tokens = assembled_context.get('total_tokens_est', 0)
            if total_tokens >= 100:
                return True  # Substantial context was retrieved → pipeline was useful

        return False  # All checks passed → pipeline was unnecessary

    def maybe_queue_shadow_validation(
        self, text: str, metadata: dict, thread_id: str,
        reflex_response: str, cluster_id: int,
    ):
        """
        Probabilistically queue full-pipeline run for quality comparison.

        ~10% of reflex activations get a shadow run through the full pipeline.
        The shadow comparison detects silent correctness failures.
        """
        if random.random() > SHADOW_VALIDATION_RATE:
            return

        try:
            shadow_data = json.dumps({
                'text': text,
                'reflex_response': reflex_response,
                'cluster_id': cluster_id,
                'queued_at': time.time(),
            })
            self.redis.setex(
                f"reflex:shadow:{thread_id}",
                600,  # 10 min TTL
                shadow_data,
            )
            logger.info(f"[REFLEX] Shadow validation queued for cluster {cluster_id}")
        except Exception as e:
            logger.debug(f"[REFLEX] Shadow validation queue failed: {e}")

    def process_shadow_result(self, thread_id: str, full_pipeline_response: str):
        """
        Compare shadow pipeline response with the reflex response.

        Called after a shadow full-pipeline run completes.
        """
        try:
            key = f"reflex:shadow:{thread_id}"
            shadow_data_str = self.redis.get(key)
            if not shadow_data_str:
                return

            self.redis.delete(key)
            shadow_data = json.loads(shadow_data_str)
            cluster_id = shadow_data['cluster_id']
            reflex_response = shadow_data['reflex_response']

            # Compare responses via embedding similarity
            from services.embedding_service import get_embedding_service
            emb_service = get_embedding_service()
            reflex_emb = np.array(emb_service.generate_embedding(reflex_response))
            pipeline_emb = np.array(emb_service.generate_embedding(full_pipeline_response))

            # Cosine distance (embeddings are L2-normalized)
            distance = 1.0 - float(np.dot(reflex_emb, pipeline_emb))

            with self.db.connection() as conn:
                cursor = conn.cursor()
                if distance >= 0.3:
                    # Divergent responses → flag quality concern
                    cursor.execute("""
                        UPDATE cognitive_reflexes
                        SET times_failed = times_failed + 1
                        WHERE id = %s
                    """, (cluster_id,))
                    logger.warning(
                        f"[REFLEX] Shadow DIVERGENT for cluster {cluster_id}: "
                        f"distance={distance:.3f}"
                    )
                else:
                    # Responses agree → validates fast path
                    cursor.execute("""
                        UPDATE cognitive_reflexes
                        SET times_succeeded = times_succeeded + 1
                        WHERE id = %s
                    """, (cluster_id,))
                    logger.info(
                        f"[REFLEX] Shadow AGREED for cluster {cluster_id}: "
                        f"distance={distance:.3f}"
                    )
                cursor.close()

        except Exception as e:
            logger.debug(f"[REFLEX] Shadow result processing failed: {e}")

    def get_stats(self) -> dict:
        """Get aggregate reflex statistics for observability."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Total clusters
                cursor.execute("SELECT COUNT(*) FROM cognitive_reflexes")
                total_clusters = cursor.fetchone()[0]

                # Active clusters (meet activation criteria)
                cursor.execute("""
                    SELECT COUNT(*) FROM cognitive_reflexes
                    WHERE times_seen >= %s
                      AND times_succeeded >= %s
                      AND last_seen >= NOW() - INTERVAL '%s days'
                      AND (times_unnecessary::float / GREATEST(times_seen, 1)) >= %s
                      AND (times_failed::float / GREATEST(times_activated, 1)) <= %s
                """, (MIN_OBSERVATIONS, MIN_SUCCESSES, MAX_STALE_DAYS,
                      MIN_CONFIDENCE, MAX_FAILURE_RATE))
                active_clusters = cursor.fetchone()[0]

                # Aggregate counters
                cursor.execute("""
                    SELECT
                        COALESCE(SUM(times_seen), 0),
                        COALESCE(SUM(times_activated), 0),
                        COALESCE(SUM(times_succeeded), 0),
                        COALESCE(SUM(times_failed), 0),
                        COALESCE(SUM(times_unnecessary), 0)
                    FROM cognitive_reflexes
                """)
                row = cursor.fetchone()
                total_seen, total_activated, total_succeeded, total_failed, total_unnecessary = row

                # New clusters in last 24h
                cursor.execute("""
                    SELECT COUNT(*) FROM cognitive_reflexes
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                """)
                new_24h = cursor.fetchone()[0]

                # Recent clusters for detail view
                cursor.execute("""
                    SELECT id, sample_queries, times_seen, times_unnecessary,
                           times_activated, times_succeeded, times_failed,
                           created_at, last_seen, last_activated
                    FROM cognitive_reflexes
                    ORDER BY last_seen DESC NULLS LAST
                    LIMIT 10
                """)
                recent = []
                for r in cursor.fetchall():
                    confidence = r[3] / max(r[2], 1)
                    failure_rate = r[6] / max(r[4], 1)
                    recent.append({
                        'id': r[0],
                        'sample_queries': r[1] or [],
                        'times_seen': r[2],
                        'times_unnecessary': r[3],
                        'times_activated': r[4],
                        'times_succeeded': r[5],
                        'times_failed': r[6],
                        'confidence': round(confidence, 3),
                        'failure_rate': round(failure_rate, 3),
                        'created_at': str(r[7]) if r[7] else None,
                        'last_seen': str(r[8]) if r[8] else None,
                        'last_activated': str(r[9]) if r[9] else None,
                    })

                cursor.close()

                # Compute rates
                activation_rate = total_activated / max(total_seen, 1)
                success_rate = total_succeeded / max(total_activated, 1)

                return {
                    'total_clusters': total_clusters,
                    'active_clusters': active_clusters,
                    'total_observations': total_seen,
                    'total_activations': total_activated,
                    'total_succeeded': total_succeeded,
                    'total_failed': total_failed,
                    'total_unnecessary': total_unnecessary,
                    'activation_rate': round(activation_rate, 4),
                    'success_rate': round(success_rate, 4),
                    'new_clusters_24h': new_24h,
                    'recent_clusters': recent,
                }
        except Exception as e:
            logger.error(f"[REFLEX] Stats query failed: {e}")
            return {'error': str(e)}

    # ─── Internal methods ─────────────────────────────────────────────────

    def _is_candidate(self, text: str, context_warmth: float) -> bool:
        """
        Heuristic pre-screen (~1ms). Reject obvious non-candidates
        BEFORE embedding computation.
        """
        # Empty or whitespace-only
        stripped = text.strip()
        if not stripped:
            return False

        # Too long
        word_count = len(stripped.split())
        if word_count > _MAX_WORDS:
            return False

        # High warmth — active conversation, risky to skip
        if context_warmth > _MAX_WARMTH:
            return False

        # Pattern-based rejection
        if _ANAPHORIC.search(stripped):
            return False
        if _PERSONAL_MEMORY.search(stripped):
            return False
        if _CONVERSATIONAL_REF.search(stripped):
            return False
        if _TOOL_ACTION.search(stripped):
            return False
        if _FRESHNESS.search(stripped):
            return False
        if _REASONING.search(stripped):
            return False
        if _COMPARATIVE.search(stripped):
            return False
        if _UNCERTAINTY.search(stripped):
            return False
        if _MULTI_CLAUSE.search(stripped):
            return False
        if _URL_PRESENT.search(stripped):
            return False

        return True

    def _find_matching_reflex(self, embedding: list) -> Optional[dict]:
        """
        Find the nearest reflex cluster via pgvector cosine similarity.

        Returns the nearest cluster dict if within distance threshold, else None.
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, embedding, times_seen, times_unnecessary,
                           times_activated, times_succeeded, times_failed,
                           sample_queries, last_seen, last_activated,
                           (embedding <=> %s::vector) AS distance
                    FROM cognitive_reflexes
                    WHERE (embedding <=> %s::vector) < %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT 1
                """, (embedding, embedding, CLUSTER_DISTANCE_THRESHOLD, embedding))

                row = cursor.fetchone()
                cursor.close()

                if not row:
                    return None

                return {
                    'id': row[0],
                    'embedding': row[1],
                    'times_seen': row[2],
                    'times_unnecessary': row[3],
                    'times_activated': row[4],
                    'times_succeeded': row[5],
                    'times_failed': row[6],
                    'sample_queries': row[7],
                    'last_seen': row[8],
                    'last_activated': row[9],
                    'distance': row[10],
                }
        except Exception as e:
            logger.warning(f"[REFLEX] Semantic lookup failed: {e}")
            return None

    def _merge_into_cluster(self, match: dict, text: str, embedding: list, was_useful: bool):
        """Merge a new observation into an existing reflex cluster."""
        try:
            # Rolling average centroid (same pattern as TopicClassifierService._update_topic)
            old_embedding = match['embedding']
            if isinstance(old_embedding, str):
                old_embedding = json.loads(old_embedding)
            old_embedding = np.array(old_embedding, dtype=np.float32)
            new_embedding = np.array(embedding, dtype=np.float32)

            n = min(match['times_seen'], ROLLING_AVG_CAP)
            updated_centroid = (old_embedding * n + new_embedding) / (n + 1)

            # L2-normalize to keep cosine similarity stable
            norm = np.linalg.norm(updated_centroid)
            if norm > 0:
                updated_centroid = updated_centroid / norm

            # Update sample_queries (keep last MAX_SAMPLE_QUERIES)
            sample_queries = list(match.get('sample_queries') or [])
            sample_queries.append(text)
            sample_queries = sample_queries[-MAX_SAMPLE_QUERIES:]

            unnecessary_inc = 0 if was_useful else 1

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE cognitive_reflexes
                    SET embedding = %s::vector,
                        times_seen = times_seen + 1,
                        times_unnecessary = times_unnecessary + %s,
                        sample_queries = %s,
                        last_seen = NOW()
                    WHERE id = %s
                """, (
                    updated_centroid.tolist(),
                    unnecessary_inc,
                    sample_queries,
                    match['id'],
                ))
                cursor.close()

            logger.info(
                f"[REFLEX] Merged into cluster {match['id']} "
                f"(seen={match['times_seen'] + 1}, useful={was_useful})"
            )
        except Exception as e:
            logger.warning(f"[REFLEX] Cluster merge failed: {e}")

    def _create_cluster(self, text: str, embedding: list, was_useful: bool):
        """Create a new reflex cluster seeded by this query."""
        try:
            unnecessary = 0 if was_useful else 1

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO cognitive_reflexes
                        (embedding, sample_queries, times_seen, times_unnecessary)
                    VALUES (%s::vector, %s, 1, %s)
                    RETURNING id
                """, (embedding, [text], unnecessary))
                new_id = cursor.fetchone()[0]
                cursor.close()

            logger.info(f"[REFLEX] Created cluster {new_id} (useful={was_useful})")
        except Exception as e:
            logger.warning(f"[REFLEX] Cluster creation failed: {e}")

    def _is_correction(self, text: str) -> bool:
        """Check if text contains correction/rephrase signals."""
        for pattern in _CORRECTION_PATTERNS:
            if pattern.search(text):
                return True
        for pattern in _REPHRASE_PATTERNS:
            if pattern.search(text):
                return True
        return False
