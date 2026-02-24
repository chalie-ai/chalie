"""
CuriosityThreadService — Manages Chalie's self-directed exploration threads.

Curiosity threads are open-ended explorations seeded from cognitive drift.
They are NOT goals (no completion framing) — they're living investigations
that grow from noticed patterns and naturally fade when interest wanes.

Two types:
  - learning: researches a topic the user keeps mentioning
  - behavioral: gentle reflection on Chalie's own patterns
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[CURIOSITY THREAD]"


class CuriosityThreadService:
    """CRUD and lifecycle management for curiosity threads."""

    # Exploration interval constants
    BASE_EXPLORE_INTERVAL = 8 * 3600  # 8 hours
    MIN_EXPLORE_INTERVAL = 3 * 3600   # 3 hours floor

    # Surfacing interval constants
    BASE_SURFACE_INTERVAL = 72 * 3600  # 72 hours

    # Limits
    MAX_ACTIVE_THREADS = 5
    DORMANCY_DAYS = 45
    ABANDON_DAYS = 60
    ABANDON_ENGAGEMENT = 0.2
    MAX_DAILY_REINFORCEMENT = 0.2

    def __init__(self, db_service=None):
        if db_service is None:
            from services.database_service import get_shared_db_service
            db_service = get_shared_db_service()
        self.db = db_service

    def create_thread(
        self,
        title: str,
        rationale: str,
        thread_type: str,
        seed_topic: str,
    ) -> Optional[str]:
        """
        Create a new curiosity thread.

        Dedup: if an active thread exists for the same seed_topic, returns None.

        Returns:
            8-char hex thread ID, or None if deduplicated.
        """
        if thread_type not in ('learning', 'behavioral'):
            logger.warning(f"{LOG_PREFIX} Invalid thread_type: {thread_type}")
            return None

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Dedup check
                cursor.execute(
                    "SELECT id FROM curiosity_threads WHERE seed_topic = %s AND status = 'active'",
                    (seed_topic,)
                )
                if cursor.fetchone():
                    logger.info(f"{LOG_PREFIX} Dedup: active thread already exists for '{seed_topic}'")
                    cursor.close()
                    return None

                thread_id = os.urandom(4).hex()

                cursor.execute("""
                    INSERT INTO curiosity_threads (id, title, rationale, thread_type, seed_topic)
                    VALUES (%s, %s, %s, %s, %s)
                """, (thread_id, title, rationale, thread_type, seed_topic))

                cursor.close()

                logger.info(
                    f"{LOG_PREFIX} Created thread {thread_id}: "
                    f"type={thread_type}, topic='{seed_topic}'"
                )
                return thread_id

        except Exception as e:
            logger.error(f"{LOG_PREFIX} create_thread failed: {e}")
            return None

    def get_threads_for_exploration(self, limit: int = 1) -> List[Dict]:
        """
        Get active threads eligible for exploration.

        Skips threads whose last_explored_at is within their effective
        exploration interval (adaptive based on engagement).
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT id, title, rationale, thread_type, seed_topic,
                           learning_notes, last_explored_at, exploration_count,
                           last_surfaced_at, engagement_score, created_at
                    FROM curiosity_threads
                    WHERE status = 'active'
                    ORDER BY last_explored_at NULLS FIRST, created_at ASC
                """)

                candidates = []
                now = datetime.now(timezone.utc)

                for row in cursor.fetchall():
                    thread = self._row_to_dict(row)
                    effective_interval = self.get_effective_explore_interval(thread)

                    if thread['last_explored_at'] is None:
                        candidates.append(thread)
                    else:
                        elapsed = (now - thread['last_explored_at']).total_seconds()
                        if elapsed >= effective_interval:
                            candidates.append(thread)

                    if len(candidates) >= limit:
                        break

                cursor.close()
                return candidates

        except Exception as e:
            logger.error(f"{LOG_PREFIX} get_threads_for_exploration failed: {e}")
            return []

    def get_active_threads(self) -> List[Dict]:
        """Get all active threads."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, rationale, thread_type, seed_topic,
                           learning_notes, last_explored_at, exploration_count,
                           last_surfaced_at, engagement_score, created_at
                    FROM curiosity_threads
                    WHERE status = 'active'
                    ORDER BY created_at ASC
                """)
                threads = [self._row_to_dict(row) for row in cursor.fetchall()]
                cursor.close()
                return threads
        except Exception as e:
            logger.error(f"{LOG_PREFIX} get_active_threads failed: {e}")
            return []

    def add_learning_note(
        self,
        thread_id: str,
        note: str,
        source: str = 'pursuit',
    ) -> bool:
        """
        Append a learning note to the thread's learning_notes JSONB array.

        Args:
            thread_id: Thread identifier
            note: The learning note text
            source: Origin of the note ('pursuit', 'user', etc.)

        Returns:
            True if successful
        """
        try:
            entry = json.dumps({
                "note": note,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": source,
            })

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE curiosity_threads
                    SET learning_notes = learning_notes || %s::jsonb,
                        exploration_count = exploration_count + 1,
                        last_explored_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s
                """, (f'[{entry}]', thread_id))
                updated = cursor.rowcount > 0
                cursor.close()

                if updated:
                    logger.info(f"{LOG_PREFIX} Added learning note to thread {thread_id}")
                return updated

        except Exception as e:
            logger.error(f"{LOG_PREFIX} add_learning_note failed: {e}")
            return False

    def get_surfacing_candidate(self, thread_id: str) -> Optional[str]:
        """
        Get last 2-3 learning notes as formatted summary for surfacing.

        Returns None if too thin (<50 chars total).
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT learning_notes FROM curiosity_threads WHERE id = %s",
                    (thread_id,)
                )
                row = cursor.fetchone()
                cursor.close()

                if not row or not row[0]:
                    return None

                notes = row[0] if isinstance(row[0], list) else json.loads(row[0])
                if not notes:
                    return None

                # Take last 2-3 notes
                recent = notes[-3:]
                summary_parts = [n.get('note', '') for n in recent if n.get('note')]
                summary = " ".join(summary_parts)

                if len(summary) < 50:
                    return None

                return summary

        except Exception as e:
            logger.error(f"{LOG_PREFIX} get_surfacing_candidate failed: {e}")
            return None

    def update_engagement(self, thread_id: str, score: float) -> bool:
        """
        Update engagement score with rolling average (0.3 weight on new).

        If score drops below 0.2, thread goes dormant.
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT engagement_score FROM curiosity_threads WHERE id = %s",
                    (thread_id,)
                )
                row = cursor.fetchone()
                if not row:
                    cursor.close()
                    return False

                current = row[0]
                new_score = (current * 0.7) + (score * 0.3)
                new_score = max(0.0, min(1.0, new_score))

                new_status = 'dormant' if new_score < 0.2 else 'active'

                cursor.execute("""
                    UPDATE curiosity_threads
                    SET engagement_score = %s,
                        status = %s,
                        updated_at = NOW()
                    WHERE id = %s
                """, (new_score, new_status, thread_id))

                cursor.close()

                if new_status == 'dormant':
                    logger.info(
                        f"{LOG_PREFIX} Thread {thread_id} → dormant "
                        f"(engagement={new_score:.2f})"
                    )

                return True

        except Exception as e:
            logger.error(f"{LOG_PREFIX} update_engagement failed: {e}")
            return False

    def reinforce_from_conversation(self, seed_topic: str) -> bool:
        """
        Boost engagement_score by +0.1 (capped at 1.0).

        Rate-limited to max +0.2/day per thread via Redis daily counter.
        Called when user-generated episodes align with a thread's seed_topic.
        """
        try:
            from services.redis_client import RedisClientService
            redis = RedisClientService.create_connection()

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, engagement_score FROM curiosity_threads "
                    "WHERE seed_topic = %s AND status = 'active'",
                    (seed_topic,)
                )
                row = cursor.fetchone()
                if not row:
                    cursor.close()
                    return False

                thread_id = row[0]
                current_score = row[1]

                # Daily reinforcement cap via Redis counter
                daily_key = f"curiosity:reinforce:{thread_id}:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
                current_daily = float(redis.get(daily_key) or 0)

                if current_daily >= self.MAX_DAILY_REINFORCEMENT:
                    logger.debug(
                        f"{LOG_PREFIX} Daily reinforcement cap reached for thread {thread_id}"
                    )
                    cursor.close()
                    return False

                boost = min(0.1, self.MAX_DAILY_REINFORCEMENT - current_daily)
                new_score = min(1.0, current_score + boost)

                cursor.execute(
                    "UPDATE curiosity_threads SET engagement_score = %s, updated_at = NOW() WHERE id = %s",
                    (new_score, thread_id)
                )
                cursor.close()

                # Update daily counter
                pipe = redis.pipeline()
                pipe.incrbyfloat(daily_key, boost)
                pipe.expire(daily_key, 86400)
                pipe.execute()

                logger.info(
                    f"{LOG_PREFIX} Reinforced thread {thread_id} "
                    f"(+{boost:.1f} → {new_score:.2f})"
                )
                return True

        except Exception as e:
            logger.error(f"{LOG_PREFIX} reinforce_from_conversation failed: {e}")
            return False

    def apply_dormancy(self) -> int:
        """
        Apply dormancy rules:
          - Active threads not explored in 45 days → dormant
          - Dormant + engagement < 0.2 + dormant > 60 days → abandoned

        Returns:
            Number of threads transitioned
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                count = 0

                # Active → dormant (not explored in 45 days)
                cursor.execute("""
                    UPDATE curiosity_threads
                    SET status = 'dormant', updated_at = NOW()
                    WHERE status = 'active'
                      AND last_explored_at IS NOT NULL
                      AND last_explored_at < NOW() - INTERVAL '%s days'
                """ % self.DORMANCY_DAYS)
                count += cursor.rowcount

                # Dormant → abandoned (engagement < 0.2 AND dormant > 60 days)
                cursor.execute("""
                    UPDATE curiosity_threads
                    SET status = 'abandoned', updated_at = NOW()
                    WHERE status = 'dormant'
                      AND engagement_score < %s
                      AND updated_at < NOW() - INTERVAL '%s days'
                """ % (self.ABANDON_ENGAGEMENT, self.ABANDON_DAYS))
                count += cursor.rowcount

                cursor.close()

                if count > 0:
                    logger.info(f"{LOG_PREFIX} Dormancy applied to {count} threads")
                return count

        except Exception as e:
            logger.error(f"{LOG_PREFIX} apply_dormancy failed: {e}")
            return 0

    def count_active(self) -> int:
        """Count active threads for rate limiting."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM curiosity_threads WHERE status = 'active'"
                )
                count = cursor.fetchone()[0]
                cursor.close()
                return count
        except Exception as e:
            logger.error(f"{LOG_PREFIX} count_active failed: {e}")
            return 0

    def get_effective_explore_interval(self, thread: Dict) -> float:
        """
        Adaptive exploration interval based on engagement.

        High engagement → explore more often (down to 3h).
        Low engagement → slow down (up to 12h).
        Default: 8h.
        """
        engagement = thread.get('engagement_score', 0.5)

        if engagement > 0.7:
            factor = 1.0 + (engagement - 0.5) * 2  # 1.4 – 2.0
            effective = self.BASE_EXPLORE_INTERVAL / factor
        elif engagement < 0.3:
            effective = self.BASE_EXPLORE_INTERVAL * 1.5
        else:
            effective = self.BASE_EXPLORE_INTERVAL

        return max(effective, self.MIN_EXPLORE_INTERVAL)

    def get_effective_surface_interval(self, thread: Dict) -> float:
        """
        Adaptive surfacing interval modulated by user conversation activity.

        When user talks about the topic more → surface sooner.
        High engagement → 30% faster surfacing.
        """
        engagement = thread.get('engagement_score', 0.5)
        seed_topic = thread.get('seed_topic', '')
        effective = float(self.BASE_SURFACE_INTERVAL)

        # Count user-generated episodes about this topic in last 24h
        topic_episodes = self._count_user_episodes_for_topic(seed_topic)
        if topic_episodes > 0:
            reduction = topic_episodes * 0.2
            effective = self.BASE_SURFACE_INTERVAL * max(0.33, 1.0 - reduction)

        # High engagement shortens interval
        if engagement > 0.7:
            effective *= 0.7

        return effective

    def get_fatigue_budget(self, thread: Dict) -> float:
        """
        Adaptive ACT budget for pursuit based on engagement.

        High engagement → invest more (7.0).
        Low engagement → lighter touch (3.0).
        Default: 5.0 (vs 10.0 for user ACT).
        """
        engagement = thread.get('engagement_score', 0.5)

        if engagement > 0.7:
            return 7.0
        elif engagement < 0.3:
            return 3.0
        return 5.0

    def mark_explored(self, thread_id: str) -> bool:
        """Set last_explored_at = NOW() immediately (prevent double-pickup)."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE curiosity_threads SET last_explored_at = NOW() WHERE id = %s",
                    (thread_id,)
                )
                updated = cursor.rowcount > 0
                cursor.close()
                return updated
        except Exception as e:
            logger.error(f"{LOG_PREFIX} mark_explored failed: {e}")
            return False

    def mark_surfaced(self, thread_id: str) -> bool:
        """Set last_surfaced_at = NOW()."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE curiosity_threads SET last_surfaced_at = NOW() WHERE id = %s",
                    (thread_id,)
                )
                updated = cursor.rowcount > 0
                cursor.close()
                return updated
        except Exception as e:
            logger.error(f"{LOG_PREFIX} mark_surfaced failed: {e}")
            return False

    def _count_user_episodes_for_topic(self, seed_topic: str, hours: int = 24) -> int:
        """
        Count user-generated episodes matching the seed topic in last N hours.

        CRITICAL: excludes episodes from pursuit/tool_reflection/drift/curiosity_thread
        to prevent self-reinforcement loops.
        """
        if not seed_topic:
            return 0

        try:
            from services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            topic_embedding = embedding_service.generate_embedding(seed_topic)

            if topic_embedding is None:
                return 0

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM episodes
                    WHERE created_at > NOW() - INTERVAL '%s hours'
                      AND deleted_at IS NULL
                      AND (
                          salience_factors->>'source' IS NULL
                          OR salience_factors->>'source' NOT IN (
                              'tool_reflection', 'pursuit', 'drift', 'curiosity_thread'
                          )
                      )
                      AND embedding IS NOT NULL
                      AND (1 - (embedding <=> %s::vector)) >= 0.5
                """ % hours, (topic_embedding,))

                count = cursor.fetchone()[0]
                cursor.close()
                return count

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} _count_user_episodes_for_topic failed: {e}")
            return 0

    def _row_to_dict(self, row) -> Dict:
        """Convert a database row tuple to a thread dict."""
        return {
            'id': row[0],
            'title': row[1],
            'rationale': row[2],
            'thread_type': row[3],
            'seed_topic': row[4],
            'learning_notes': row[5] if isinstance(row[5], list) else json.loads(row[5] or '[]'),
            'last_explored_at': row[6],
            'exploration_count': row[7],
            'last_surfaced_at': row[8],
            'engagement_score': row[9],
            'created_at': row[10],
        }
