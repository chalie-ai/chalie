"""
Focus Session Service - Redis-backed focus session management.

Tracks declared or inferred focus sessions per thread. Raises boundary
thresholds during focus, detects distraction gently, and auto-infers focus
from sustained topic engagement.

UX principle: Distraction detection is a signal, not a correction.
The system anchors gently — never blocks or admonishes.
"""

import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Focus session Redis key prefix
_KEY_PREFIX = "focus_session"

# Focus session TTL: 2 hours
_SESSION_TTL = 7200

# Distraction similarity threshold: below this = off-focus message
_DISTRACTION_THRESHOLD = 0.35

# Auto-infer focus after this many consecutive exchanges on same topic
_INFER_AFTER_EXCHANGES = 5

# Boundary modifier values by source
_BOUNDARY_MODIFIER = {
    'explicit': 1.0,
    'inferred': 0.5,
}


class FocusSessionService:
    """Manages per-thread focus sessions backed by Redis."""

    def set_focus(
        self,
        thread_id: str,
        description: str,
        topic: str = "",
        goal_id: Optional[str] = None,
        source: str = 'explicit',
    ) -> bool:
        """
        Set a focus session for a thread.

        Args:
            thread_id: Thread identifier
            description: What the user is focusing on
            topic: Topic associated with focus
            goal_id: Optional goal this focus session relates to
            source: 'explicit' (user declared) or 'inferred' (auto-detected)

        Returns:
            True if stored successfully
        """
        try:
            from services.redis_client import RedisClientService

            redis = RedisClientService.create_connection()

            # Generate embedding for distraction detection
            embedding = self._generate_embedding(description + " " + topic)

            session = {
                "description": description,
                "topic": topic,
                "goal_id": goal_id,
                "source": source,
                "embedding": embedding,
            }

            key = f"{_KEY_PREFIX}:{thread_id}"
            redis.setex(key, _SESSION_TTL, json.dumps(session))

            logger.info(
                f"[FOCUS] Set focus for thread {thread_id}: "
                f"'{description[:50]}' (source={source})"
            )
            return True

        except Exception as e:
            logger.error(f"[FOCUS] set_focus failed: {e}")
            return False

    def get_focus(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve current focus session for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            Focus session dict or None if no active focus
        """
        try:
            from services.redis_client import RedisClientService

            redis = RedisClientService.create_connection()
            raw = redis.get(f"{_KEY_PREFIX}:{thread_id}")
            if not raw:
                return None
            return json.loads(raw)

        except Exception as e:
            logger.debug(f"[FOCUS] get_focus failed: {e}")
            return None

    def clear_focus(self, thread_id: str) -> bool:
        """
        Clear focus session for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            True if cleared (or already absent)
        """
        try:
            from services.redis_client import RedisClientService

            redis = RedisClientService.create_connection()
            redis.delete(f"{_KEY_PREFIX}:{thread_id}")
            logger.info(f"[FOCUS] Cleared focus for thread {thread_id}")
            return True

        except Exception as e:
            logger.debug(f"[FOCUS] clear_focus failed: {e}")
            return False

    def check_distraction(
        self,
        thread_id: str,
        message_embedding,
    ) -> Dict[str, Any]:
        """
        Check if the current message is off-focus (distraction signal).

        Computes cosine similarity of message embedding to focus embedding.
        Returns a gentle signal — never used to block the user.

        Args:
            thread_id: Thread identifier
            message_embedding: Embedding of the current message (numpy array or list)

        Returns:
            Dict with is_distraction, similarity_to_focus, focus_description
        """
        focus = self.get_focus(thread_id)
        if not focus or not focus.get('embedding'):
            return {"is_distraction": False, "similarity_to_focus": 1.0, "focus_description": ""}

        try:
            import numpy as np

            focus_emb = np.array(focus['embedding'], dtype=float)
            msg_emb = np.array(message_embedding, dtype=float)

            # Cosine similarity
            norm_f = np.linalg.norm(focus_emb)
            norm_m = np.linalg.norm(msg_emb)
            if norm_f < 1e-8 or norm_m < 1e-8:
                return {"is_distraction": False, "similarity_to_focus": 1.0,
                        "focus_description": focus.get('description', '')}

            similarity = float(np.dot(focus_emb, msg_emb) / (norm_f * norm_m))

            is_distraction = similarity < _DISTRACTION_THRESHOLD

            if is_distraction:
                logger.debug(
                    f"[FOCUS] Distraction detected for thread {thread_id}: "
                    f"similarity={similarity:.3f} (threshold={_DISTRACTION_THRESHOLD})"
                )

            return {
                "is_distraction": is_distraction,
                "similarity_to_focus": round(similarity, 3),
                "focus_description": focus.get('description', ''),
            }

        except Exception as e:
            logger.debug(f"[FOCUS] check_distraction failed: {e}")
            return {"is_distraction": False, "similarity_to_focus": 1.0,
                    "focus_description": focus.get('description', '')}

    def maybe_infer_focus(
        self,
        thread_id: str,
        topic: str,
        consecutive_count: int,
    ) -> bool:
        """
        Auto-infer focus when the same topic spans 5+ consecutive exchanges.

        Will not override an explicit focus session.

        Args:
            thread_id: Thread identifier
            topic: Current topic
            consecutive_count: Number of consecutive exchanges on this topic

        Returns:
            True if focus was inferred and set
        """
        if consecutive_count < _INFER_AFTER_EXCHANGES:
            return False

        # Don't override explicit focus
        existing = self.get_focus(thread_id)
        if existing and existing.get('source') == 'explicit':
            return False

        # Set inferred focus
        description = f"Deep work on: {topic}"
        return self.set_focus(
            thread_id=thread_id,
            description=description,
            topic=topic,
            source='inferred',
        )

    def get_focus_for_prompt(self, thread_id: str) -> str:
        """
        Format current focus session for prompt injection.

        Args:
            thread_id: Thread identifier

        Returns:
            Formatted focus string or empty string
        """
        focus = self.get_focus(thread_id)
        if not focus:
            return ""

        source_label = "(declared)" if focus.get('source') == 'explicit' else "(inferred)"
        return f"## Current Focus\nPrimary: {focus['description']} {source_label}"

    def get_boundary_modifier(self, thread_id: str) -> float:
        """
        Get boundary modifier for adaptive boundary detector.

        Returns:
            0.0 (no focus), 0.5 (inferred focus), 1.0 (explicit focus)
        """
        focus = self.get_focus(thread_id)
        if not focus:
            return 0.0
        return _BOUNDARY_MODIFIER.get(focus.get('source', 'inferred'), 0.5)

    def _generate_embedding(self, text: str) -> Optional[list]:
        """Generate embedding as list for JSON serialization."""
        try:
            from services.embedding_service import get_embedding_service
            emb_service = get_embedding_service()
            emb = emb_service.generate_embedding(text)
            return [float(x) for x in emb]
        except Exception as e:
            logger.debug(f"[FOCUS] Embedding generation failed: {e}")
            return None
