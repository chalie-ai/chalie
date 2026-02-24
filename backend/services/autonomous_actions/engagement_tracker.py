"""
EngagementTracker — Correlates user responses with proactive messages.

Called by the digest worker on each incoming user message. Checks if
there's a pending proactive message awaiting response, and scores the
user's engagement using embedding similarity.
"""

import json
import math
import time
import logging
from typing import Optional, Dict, Any

from services.redis_client import RedisClientService
from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

LOG_PREFIX = "[ENGAGEMENT]"

_NS = "proactive"


def _key(user_id: str, suffix: str) -> str:
    return f"{_NS}:{user_id}:{suffix}"


class EngagementTracker:
    """Tracks and scores user engagement with proactive messages."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.redis = RedisClientService.create_connection()
        self.user_id = config.get('user_id', 'default')

        # Similarity thresholds
        self.engaged_similarity = config.get('engaged_similarity', 0.35)
        self.dismissed_similarity = config.get('dismissed_similarity', 0.2)
        self.min_engaged_words = config.get('min_engaged_words', 3)

        self._embedding_service = None

    @property
    def embedding_service(self):
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def check_and_score(
        self,
        user_message: str,
        user_embedding: Optional[list] = None,
        topic: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Check if there's a pending proactive message and score the user's response.

        Called by digest worker on each incoming user message.

        Args:
            user_message: The user's message text
            user_embedding: Pre-computed embedding of user message (optional)
            topic: Current conversation topic

        Returns:
            Engagement result dict if a proactive response was correlated, else None.
        """
        pending_id = self.redis.get(_key(self.user_id, 'pending_response'))
        if not pending_id:
            return None

        # Get the proactive thought content for comparison
        pending_content = self.redis.get(_key(self.user_id, 'pending_content'))
        pending_embedding = None
        if pending_content:
            try:
                pending_data = json.loads(pending_content)
                pending_embedding = pending_data.get('embedding')
                pending_text = pending_data.get('content', '')
            except (json.JSONDecodeError, TypeError):
                pending_text = ''
        else:
            pending_text = ''

        # Compute similarity
        similarity = 0.0
        if user_embedding and pending_embedding:
            similarity = self._cosine_similarity(user_embedding, pending_embedding)
        elif pending_text and user_message:
            # Fallback to Jaccard if no embeddings
            similarity = self._jaccard_similarity(user_message, pending_text)

        # Score the engagement
        word_count = len(user_message.strip().split())
        outcome, score = self._classify_response(similarity, word_count)

        # Record the outcome
        result = {
            'proactive_id': pending_id,
            'outcome': outcome,
            'score': score,
            'similarity': similarity,
            'word_count': word_count,
        }

        self._update_engagement_state(pending_id, outcome, score)

        # Route feedback to curiosity thread if this was a thread surfacing
        self._route_thread_feedback(outcome, score)

        logger.info(
            f"{LOG_PREFIX} Proactive response scored: {outcome} "
            f"(similarity={similarity:.3f}, words={word_count}, score={score:.2f})"
        )

        return result

    def _classify_response(self, similarity: float, word_count: int) -> tuple:
        """
        Classify user response to proactive message.

        Returns:
            (outcome_label, score)
        """
        if similarity > self.engaged_similarity and word_count >= self.min_engaged_words:
            return ('engaged', 1.0)
        elif similarity > self.engaged_similarity or word_count >= self.min_engaged_words:
            return ('acknowledged', 0.5)
        elif similarity < self.dismissed_similarity:
            return ('dismissed', 0.0)
        else:
            return ('acknowledged', 0.5)

    def _update_engagement_state(self, proactive_id: str, outcome: str, score: float):
        """Update all engagement-related Redis state."""
        now = time.time()

        # Recent outcomes (circuit breaker)
        outcomes_key = _key(self.user_id, 'recent_outcomes')
        entry = json.dumps({
            'id': proactive_id,
            'outcome': outcome,
            'score': score,
            'ts': now,
        })
        self.redis.lpush(outcomes_key, entry)
        self.redis.ltrim(outcomes_key, 0, 2)

        # Engagement history (rolling 10)
        history_key = _key(self.user_id, 'engagement_history')
        self.redis.lpush(history_key, entry)
        self.redis.ltrim(history_key, 0, 9)

        # Recompute engagement score
        self._recompute_engagement_score()

        # Adjust backoff
        if outcome in ('engaged', 'acknowledged'):
            self.redis.set(_key(self.user_id, 'backoff_multiplier'), 1)

            # If auto-paused, check if engagement is high enough to resume
            if self.redis.get(_key(self.user_id, 'paused')) == '1':
                engagement = float(self.redis.get(_key(self.user_id, 'engagement_score')) or 0)
                if engagement >= 0.5:
                    self.redis.delete(_key(self.user_id, 'paused'))
                    self.redis.delete(_key(self.user_id, 'paused_since'))
                    logger.info(f"{LOG_PREFIX} Auto-pause lifted via positive engagement")
        elif outcome in ('ignored', 'dismissed'):
            current = int(self.redis.get(_key(self.user_id, 'backoff_multiplier')) or 1)
            new_backoff = min(current * 2, 16)
            self.redis.set(_key(self.user_id, 'backoff_multiplier'), new_backoff)

        # Clear pending
        self.redis.delete(_key(self.user_id, 'pending_response'))
        self.redis.delete(_key(self.user_id, 'pending_content'))

    def _recompute_engagement_score(self):
        """Recompute rolling engagement score from history."""
        history_key = _key(self.user_id, 'engagement_history')
        raw = self.redis.lrange(history_key, 0, 9)

        if not raw:
            self.redis.set(_key(self.user_id, 'engagement_score'), '1.0')
            return

        total = 0.0
        count = 0
        for entry_raw in raw:
            try:
                entry = json.loads(entry_raw)
                outcome = entry.get('outcome', 'unknown')
                if outcome == 'engaged':
                    total += 1.0
                elif outcome == 'acknowledged':
                    total += 0.5
                elif outcome == 'dismissed':
                    total += 0.0
                elif outcome == 'ignored':
                    total += -0.5
                count += 1
            except (json.JSONDecodeError, TypeError):
                continue

        if count > 0:
            engagement = max(0.0, (total / count + 0.5) / 1.5)
        else:
            engagement = 1.0

        self.redis.set(_key(self.user_id, 'engagement_score'), str(round(engagement, 3)))

    def _route_thread_feedback(self, outcome: str, score: float):
        """
        If the pending proactive message was from a curiosity thread,
        route the engagement feedback to CuriosityThreadService.

        Score mapping:
          - engaged (1.0) → thread reinforced
          - acknowledged (0.5) → mild positive
          - dismissed (0.0) → mild negative signal
          - ignored → 0.0 (neutral, not punitive)
        """
        try:
            thread_id = self.redis.get(_key(self.user_id, 'pending_thread_id'))
            if not thread_id:
                return

            # Map outcome to thread engagement score
            # ignored = 0.0 (neutral) per plan: absence of signal, not negative
            thread_score_map = {
                'engaged': 1.0,
                'acknowledged': 0.5,
                'dismissed': 0.0,
                'ignored': 0.0,
            }
            mapped = thread_score_map.get(outcome, 0.5)

            from services.curiosity_thread_service import CuriosityThreadService
            CuriosityThreadService().update_engagement(thread_id, mapped)

            # Clean up
            self.redis.delete(_key(self.user_id, 'pending_thread_id'))

            logger.info(
                f"{LOG_PREFIX} Routed feedback to thread {thread_id}: "
                f"{outcome} → {mapped:.1f}"
            )

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Thread feedback routing failed: {e}")

    def store_pending_content(self, proactive_id: str, content: str, embedding: list = None):
        """
        Store proactive message content for later engagement scoring.

        Called after a proactive message is successfully enqueued.
        """
        data = {
            'content': content,
            'embedding': embedding,
            'ts': time.time(),
        }
        self.redis.setex(
            _key(self.user_id, 'pending_content'),
            14400,  # 4h TTL (same as pending timeout)
            json.dumps(data)
        )

    @staticmethod
    def _cosine_similarity(a: list, b: list) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)
