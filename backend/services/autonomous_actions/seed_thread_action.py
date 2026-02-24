"""
SeedThreadAction — Seeds a new curiosity thread from a drift insight.

Priority: 6 (above REFLECT=5, below COMMUNICATE=10).

Eligibility gates:
  1. Drift thought type must be 'insight'
  2. No active thread for the same seed_topic
  3. Activation energy >= 0.6
  4. Max 1 new thread per 24h window (Redis rate key)
  5. Total active threads < 5
  6. Salience alignment: episodic + semantic gates must pass
"""

import logging
from typing import Optional, Tuple, Dict, Any

from services.redis_client import RedisClientService

from .base import AutonomousAction, ActionResult, ThoughtContext

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SEED THREAD]"

# Behavioral seed topic keywords — abstract/meta concepts
_BEHAVIORAL_KEYWORDS = frozenset({
    'concise', 'coherent', 'engaged', 'clear', 'tone', 'style',
    'explain', 'respond', 'communicate', 'approach', 'pattern',
    'behavior', 'behaviour', 'habit', 'tendency', 'reflection',
})

SEED_COOLDOWN_KEY = "curiosity:seed_cooldown"
SEED_COOLDOWN_TTL = 86400  # 24 hours


class SeedThreadAction(AutonomousAction):
    """Seeds curiosity threads from drift insights that pass salience gates."""

    def __init__(self, config: dict = None):
        super().__init__(name='SEED_THREAD', enabled=True, priority=6)
        config = config or {}
        self.redis = RedisClientService.create_connection()
        self.min_activation = config.get('min_activation', 0.6)
        self.episodic_similarity_threshold = config.get('episodic_similarity_threshold', 0.55)
        self.episodic_min_matches = config.get('episodic_min_matches', 2)
        self.semantic_min_strength = config.get('semantic_min_strength', 0.5)

    def should_execute(self, thought: ThoughtContext) -> Tuple[float, bool]:
        """
        Evaluate eligibility. All gates must pass.

        Returns:
            (score, eligible)
        """
        # Gate 1: Must come from an insight seed strategy (topic 3+ times in 7 days)
        if thought.extra.get('seed_type') != 'insight':
            return (0.0, False)

        # Gate 2: No active thread for same seed_topic
        from services.curiosity_thread_service import CuriosityThreadService
        thread_service = CuriosityThreadService()

        active_threads = thread_service.get_active_threads()
        for t in active_threads:
            if t['seed_topic'] == thought.seed_topic:
                logger.debug(f"{LOG_PREFIX} Dedup: active thread exists for '{thought.seed_topic}'")
                return (0.0, False)

        # Gate 3: Activation energy threshold
        if thought.activation_energy < self.min_activation:
            return (0.0, False)

        # Gate 4: 24h seed cooldown
        if self.redis.exists(SEED_COOLDOWN_KEY):
            logger.debug(f"{LOG_PREFIX} Seed cooldown active, skipping")
            return (0.0, False)

        # Gate 5: Max active threads
        if thread_service.count_active() >= CuriosityThreadService.MAX_ACTIVE_THREADS:
            logger.debug(f"{LOG_PREFIX} Max active threads reached")
            return (0.0, False)

        # Gate 6: Salience alignment
        if not self._check_salience(thought):
            return (0.0, False)

        # Score = activation energy (higher insight activation = higher priority)
        score = thought.activation_energy * 0.8
        return (score, True)

    def execute(self, thought: ThoughtContext) -> ActionResult:
        """
        Create a new curiosity thread from the insight.
        """
        from services.curiosity_thread_service import CuriosityThreadService

        thread_service = CuriosityThreadService()

        # Classify thread type
        thread_type = self._classify_type(thought)

        # Generate title and rationale
        if thread_type == 'learning':
            title = f"Explore {thought.seed_topic} further"
            rationale = "This keeps coming up in conversation"
        else:
            title = f"Be more curious about {thought.seed_topic}"
            rationale = "I've been noticing this pattern"

        thread_id = thread_service.create_thread(
            title=title,
            rationale=rationale,
            thread_type=thread_type,
            seed_topic=thought.seed_topic,
        )

        if not thread_id:
            return ActionResult(
                action_name='SEED_THREAD',
                success=False,
                details={'reason': 'create_failed_or_dedup'}
            )

        # Set 24h cooldown
        self.redis.setex(SEED_COOLDOWN_KEY, SEED_COOLDOWN_TTL, '1')

        # Log to interaction_log
        self._log_seeded(thought, thread_id, thread_type)

        logger.info(
            f"{LOG_PREFIX} Seeded thread {thread_id}: "
            f"type={thread_type}, topic='{thought.seed_topic}'"
        )

        return ActionResult(
            action_name='SEED_THREAD',
            success=True,
            details={
                'thread_id': thread_id,
                'thread_type': thread_type,
                'seed_topic': thought.seed_topic,
                'title': title,
            }
        )

    def _classify_type(self, thought: ThoughtContext) -> str:
        """
        Classify thread type by heuristic.

        Concrete/technical/proper-noun terms → learning.
        Abstract/behavioral/meta concepts → behavioral.
        """
        topic_lower = thought.seed_topic.lower()
        words = set(topic_lower.split())

        # Check for behavioral keywords
        if words & _BEHAVIORAL_KEYWORDS:
            return 'behavioral'

        # Short, abstract topics are more likely behavioral
        if len(words) <= 2 and not any(c.isupper() for c in thought.seed_topic):
            # Check if it's a concrete noun (has digits, special chars, or is capitalized)
            if topic_lower.isalpha() and topic_lower in _BEHAVIORAL_KEYWORDS:
                return 'behavioral'

        return 'learning'

    def _check_salience(self, thought: ThoughtContext) -> bool:
        """
        Salience alignment check — ensures threads reflect what the human
        actually cares about, not random noise.

        Both episodic AND semantic gates must pass.
        """
        episodic_pass = self._check_episodic_salience(thought)
        if not episodic_pass:
            logger.debug(f"{LOG_PREFIX} Episodic salience gate failed for '{thought.seed_topic}'")
            return False

        semantic_pass = self._check_semantic_salience(thought)
        if not semantic_pass:
            logger.debug(f"{LOG_PREFIX} Semantic salience gate failed for '{thought.seed_topic}'")
            return False

        return True

    def _check_episodic_salience(self, thought: ThoughtContext) -> bool:
        """
        Check episodic salience: seed_topic embedding vs recent user-generated
        episodes (last 72h). Mean cosine similarity must be >= 0.55 with at
        least 2 matching episodes.
        """
        if not thought.thought_embedding:
            return False

        try:
            from services.database_service import get_shared_db_service
            import math

            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()

                # Get user-generated episodes from last 72h with embeddings
                cursor.execute("""
                    SELECT embedding FROM episodes
                    WHERE created_at > NOW() - INTERVAL '72 hours'
                      AND deleted_at IS NULL
                      AND embedding IS NOT NULL
                      AND (
                          salience_factors->>'source' IS NULL
                          OR salience_factors->>'source' NOT IN (
                              'tool_reflection', 'pursuit', 'drift', 'curiosity_thread'
                          )
                      )
                    ORDER BY created_at DESC
                    LIMIT 50
                """)

                rows = cursor.fetchall()
                cursor.close()

                if len(rows) < self.episodic_min_matches:
                    return False

                # Compute similarities
                similarities = []
                for row in rows:
                    ep_embedding = row[0]
                    if ep_embedding:
                        sim = self._cosine_similarity(thought.thought_embedding, ep_embedding)
                        if sim >= self.episodic_similarity_threshold:
                            similarities.append(sim)

                if len(similarities) < self.episodic_min_matches:
                    return False

                mean_sim = sum(similarities) / len(similarities)
                return mean_sim >= self.episodic_similarity_threshold

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Episodic salience check failed: {e}")
            return False

    def _check_semantic_salience(self, thought: ThoughtContext) -> bool:
        """
        Check semantic salience: at least 1 semantic concept matching the
        seed_topic with concept strength >= 0.5.
        """
        try:
            from services.database_service import get_shared_db_service

            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT COUNT(*) FROM semantic_concepts
                    WHERE LOWER(concept_name) LIKE %s
                      AND strength >= %s
                      AND deleted_at IS NULL
                """, (f'%{thought.seed_topic.lower()}%', self.semantic_min_strength))

                count = cursor.fetchone()[0]
                cursor.close()
                return count >= 1

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Semantic salience check failed: {e}")
            return False

    def _log_seeded(self, thought: ThoughtContext, thread_id: str, thread_type: str):
        """Log thread seeding to interaction_log."""
        try:
            from services.database_service import get_shared_db_service
            from services.interaction_log_service import InteractionLogService

            db = get_shared_db_service()
            log_service = InteractionLogService(db)
            log_service.log_event(
                event_type='curiosity_thread_seeded',
                payload={
                    'thread_id': thread_id,
                    'thread_type': thread_type,
                    'seed_topic': thought.seed_topic,
                    'activation_energy': thought.activation_energy,
                    'thought_content': thought.thought_content[:200],
                },
                topic=thought.seed_topic,
                source='seed_thread_action',
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to log seeding: {e}")

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        if not a or not b:
            return 0.0

        # Handle pgvector string format
        if isinstance(a, str):
            a = [float(x) for x in a.strip('[]').split(',')]
        if isinstance(b, str):
            b = [float(x) for x in b.strip('[]').split(',')]

        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
