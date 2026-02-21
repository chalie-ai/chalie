"""
Goal Service - Persistent directional goals with lifecycle management.

Stores user goals with status transitions, progress tracking, and
dormancy based on inactivity. Integrates with autobiography for inference
and decay engine for dormancy cleanup.
"""

import json
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

# Valid status transitions
_TRANSITIONS = {
    'active':      {'progressing', 'dormant', 'abandoned'},
    'progressing': {'achieved', 'active', 'dormant', 'abandoned'},
    'dormant':     {'active'},
    'achieved':    set(),   # terminal
    'abandoned':   set(),   # terminal
}

# Dormancy threshold: goals not mentioned in 30 days move to dormant
DORMANCY_DAYS = 30

# Deduplication similarity threshold for autobiography inference
DEDUP_SIMILARITY_THRESHOLD = 0.8


class GoalService:
    """Manages user goals with lifecycle, prompt injection, and decay."""

    def __init__(self, db_service):
        """
        Initialize goal service.

        Args:
            db_service: DatabaseService instance
        """
        self.db = db_service

    def create_goal(
        self,
        title: str,
        description: str = "",
        priority: int = 5,
        source: str = 'inferred',
        related_topics: Optional[List[str]] = None,
        user_id: str = 'primary',
    ) -> str:
        """
        Create a new goal.

        Args:
            title: Goal title (max 200 chars)
            description: Optional detailed description
            priority: Priority 1-10 (default 5)
            source: 'explicit', 'inferred', or 'autobiography'
            related_topics: List of related topic strings
            user_id: User identifier

        Returns:
            goal_id (8-char hex string)
        """
        goal_id = secrets.token_hex(4)  # 8-char hex
        related_topics = related_topics or []
        priority = max(1, min(10, priority))

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO goals
                        (id, user_id, title, description, status, priority, source,
                         progress_notes, related_topics, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, 'active', %s, %s, '[]', %s, NOW(), NOW())
                """, (
                    goal_id, user_id, title[:200], description,
                    priority, source, related_topics,
                ))
                cursor.close()

            logger.info(
                f"[GOALS] Created goal '{title[:50]}' "
                f"(id={goal_id}, priority={priority}, source={source})"
            )
            return goal_id

        except Exception as e:
            logger.error(f"[GOALS] Failed to create goal: {e}")
            raise

    def update_status(
        self,
        goal_id: str,
        new_status: str,
        note: str = "",
        user_id: str = 'primary',
    ) -> bool:
        """
        Update goal status with transition validation.

        Args:
            goal_id: Goal identifier
            new_status: Target status
            note: Optional progress note to append
            user_id: User identifier

        Returns:
            True if updated, False if transition invalid or goal not found
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT status FROM goals WHERE id = %s AND user_id = %s
                """, (goal_id, user_id))
                row = cursor.fetchone()

                if not row:
                    logger.warning(f"[GOALS] Goal {goal_id} not found")
                    cursor.close()
                    return False

                current_status = row[0]
                allowed = _TRANSITIONS.get(current_status, set())

                if new_status not in allowed:
                    logger.warning(
                        f"[GOALS] Invalid transition {current_status} → {new_status} "
                        f"for goal {goal_id}"
                    )
                    cursor.close()
                    return False

                if note:
                    cursor.execute("""
                        UPDATE goals
                        SET status = %s,
                            updated_at = NOW(),
                            progress_notes = progress_notes || %s::jsonb
                        WHERE id = %s AND user_id = %s
                    """, (
                        new_status,
                        json.dumps([{"note": note, "timestamp": datetime.utcnow().isoformat()}]),
                        goal_id, user_id,
                    ))
                else:
                    cursor.execute("""
                        UPDATE goals
                        SET status = %s, updated_at = NOW()
                        WHERE id = %s AND user_id = %s
                    """, (new_status, goal_id, user_id))

                cursor.close()

            logger.info(f"[GOALS] Goal {goal_id}: {current_status} → {new_status}")
            return True

        except Exception as e:
            logger.error(f"[GOALS] update_status failed: {e}")
            return False

    def add_progress_note(
        self,
        goal_id: str,
        note: str,
        user_id: str = 'primary',
    ) -> bool:
        """
        Append a progress note to a goal.

        Args:
            goal_id: Goal identifier
            note: Progress note text
            user_id: User identifier

        Returns:
            True if appended, False on failure
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE goals
                    SET progress_notes = progress_notes || %s::jsonb,
                        updated_at = NOW()
                    WHERE id = %s AND user_id = %s
                """, (
                    json.dumps([{"note": note, "timestamp": datetime.utcnow().isoformat()}]),
                    goal_id, user_id,
                ))
                updated = cursor.rowcount > 0
                cursor.close()
            return updated
        except Exception as e:
            logger.error(f"[GOALS] add_progress_note failed: {e}")
            return False

    def get_active_goals(
        self,
        user_id: str = 'primary',
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get active and progressing goals sorted by priority.

        Args:
            user_id: User identifier
            limit: Maximum number of goals to return

        Returns:
            List of goal dicts
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, description, status, priority, source,
                           related_topics, last_mentioned, created_at, progress_notes
                    FROM goals
                    WHERE user_id = %s AND status IN ('active', 'progressing')
                    ORDER BY priority DESC, created_at ASC
                    LIMIT %s
                """, (user_id, limit))
                rows = cursor.fetchall()
                cursor.close()

            return [
                {
                    "id": row[0],
                    "title": row[1],
                    "description": row[2],
                    "status": row[3],
                    "priority": row[4],
                    "source": row[5],
                    "related_topics": row[6] or [],
                    "last_mentioned": row[7],
                    "created_at": row[8],
                    "progress_notes": row[9] or [],
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"[GOALS] get_active_goals failed: {e}")
            return []

    def get_goals_for_prompt(
        self,
        topic: str = "",
        user_id: str = 'primary',
        limit: int = 3,
    ) -> str:
        """
        Format active goals for prompt injection.

        Topic-related goals are prioritized. Returns formatted string or empty.

        Args:
            topic: Current topic for relevance filtering
            user_id: User identifier
            limit: Maximum goals to inject

        Returns:
            Formatted goals string or empty string
        """
        goals = self.get_active_goals(user_id, limit=10)
        if not goals:
            return ""

        # Prioritize topic-related goals
        if topic:
            topic_lower = topic.lower()
            topic_goals = [
                g for g in goals
                if topic_lower in (g.get('title') or '').lower()
                or any(topic_lower in t.lower() for t in (g.get('related_topics') or []))
            ]
            other_goals = [g for g in goals if g not in topic_goals]
            ordered = (topic_goals + other_goals)[:limit]
        else:
            ordered = goals[:limit]

        if not ordered:
            return ""

        priority_label = {
            range(1, 4): "low",
            range(4, 7): "medium",
            range(7, 11): "high",
        }

        def _priority_str(p):
            if p <= 3: return "low"
            if p <= 6: return "medium"
            return "high"

        lines = ["## Active Goals"]
        for g in ordered:
            lines.append(
                f"- [{_priority_str(g['priority'])}] {g['title']} ({g['status']})"
            )

        return "\n".join(lines)

    def touch_goal(
        self,
        goal_id: str,
        topic: str = "",
        user_id: str = 'primary',
    ) -> bool:
        """
        Update last_mentioned and add topic to related_topics.

        Args:
            goal_id: Goal identifier
            topic: Topic to add to related_topics
            user_id: User identifier

        Returns:
            True if updated
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                if topic:
                    cursor.execute("""
                        UPDATE goals
                        SET last_mentioned = NOW(),
                            related_topics = ARRAY(
                                SELECT DISTINCT unnest(related_topics || %s::text[])
                            ),
                            updated_at = NOW()
                        WHERE id = %s AND user_id = %s
                    """, ([topic], goal_id, user_id))
                else:
                    cursor.execute("""
                        UPDATE goals
                        SET last_mentioned = NOW(), updated_at = NOW()
                        WHERE id = %s AND user_id = %s
                    """, (goal_id, user_id))
                updated = cursor.rowcount > 0
                cursor.close()
            return updated
        except Exception as e:
            logger.error(f"[GOALS] touch_goal failed: {e}")
            return False

    def apply_dormancy(self, user_id: str = 'primary') -> int:
        """
        Move goals not mentioned in DORMANCY_DAYS days to dormant status.

        Args:
            user_id: User identifier

        Returns:
            Number of goals moved to dormant
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE goals
                    SET status = 'dormant', updated_at = NOW()
                    WHERE user_id = %s
                      AND status IN ('active', 'progressing')
                      AND (
                          last_mentioned IS NULL AND created_at < NOW() - INTERVAL '%s days'
                          OR last_mentioned < NOW() - INTERVAL '%s days'
                      )
                """, (user_id, DORMANCY_DAYS, DORMANCY_DAYS))
                count = cursor.rowcount
                cursor.close()

            if count > 0:
                logger.info(f"[GOALS] Moved {count} goals to dormant (inactive for {DORMANCY_DAYS}+ days)")
            return count
        except Exception as e:
            logger.error(f"[GOALS] apply_dormancy failed: {e}")
            return 0

    def infer_goals_from_autobiography(
        self,
        narrative: str,
        user_id: str = 'primary',
    ) -> int:
        """
        Extract structured goals from the autobiography's Values And Goals section.

        Deduplicates via embedding similarity before inserting.

        Args:
            narrative: Full autobiography narrative text
            user_id: User identifier

        Returns:
            Number of new goals created
        """
        if not narrative:
            return 0

        # Extract Values And Goals section
        import re
        match = re.search(
            r'##\s+Values\s+And\s+Goals\s*\n(.*?)(?=##|\Z)',
            narrative,
            re.IGNORECASE | re.DOTALL
        )
        if not match:
            return 0

        section_text = match.group(1).strip()
        if not section_text:
            return 0

        # Extract goal candidates from section text (bullet points, numbered items)
        candidates = []
        for line in section_text.split('\n'):
            line = line.strip().lstrip('-*•0123456789.) ').strip()
            if len(line) > 10:
                candidates.append(line)

        if not candidates:
            return 0

        # Get existing active goals for dedup
        existing_goals = self.get_active_goals(user_id, limit=50)

        # Load embedding service for similarity check
        try:
            from services.embedding_service import get_embedding_service
            import numpy as np

            emb_service = get_embedding_service()

            # Build existing goal embeddings
            existing_embeddings = []
            for g in existing_goals:
                try:
                    emb = emb_service.generate_embedding(g['title'])
                    existing_embeddings.append((g['title'], emb))
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"[GOALS] Embedding service unavailable for dedup: {e}")
            existing_embeddings = []

        created = 0
        for candidate in candidates[:10]:  # Max 10 goals from autobiography
            # Dedup check via embedding similarity
            is_duplicate = False
            if existing_embeddings:
                try:
                    candidate_emb = emb_service.generate_embedding(candidate)
                    for _, existing_emb in existing_embeddings:
                        sim = float(np.dot(candidate_emb, existing_emb) / (
                            np.linalg.norm(candidate_emb) * np.linalg.norm(existing_emb) + 1e-8
                        ))
                        if sim >= DEDUP_SIMILARITY_THRESHOLD:
                            is_duplicate = True
                            break
                except Exception:
                    pass

            if not is_duplicate:
                try:
                    self.create_goal(
                        title=candidate[:200],
                        source='autobiography',
                        priority=5,
                        user_id=user_id,
                    )
                    created += 1
                except Exception as e:
                    logger.warning(f"[GOALS] Failed to create inferred goal: {e}")

        if created > 0:
            logger.info(f"[GOALS] Inferred {created} goals from autobiography")
        return created

    def get_all_goals(
        self,
        user_id: str = 'primary',
        include_terminal: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get all goals for a user.

        Args:
            user_id: User identifier
            include_terminal: Include achieved/abandoned goals

        Returns:
            List of goal dicts
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                if include_terminal:
                    cursor.execute("""
                        SELECT id, title, description, status, priority, source,
                               related_topics, last_mentioned, created_at, progress_notes
                        FROM goals
                        WHERE user_id = %s
                        ORDER BY status, priority DESC, created_at ASC
                    """, (user_id,))
                else:
                    cursor.execute("""
                        SELECT id, title, description, status, priority, source,
                               related_topics, last_mentioned, created_at, progress_notes
                        FROM goals
                        WHERE user_id = %s AND status NOT IN ('achieved', 'abandoned')
                        ORDER BY status, priority DESC, created_at ASC
                    """, (user_id,))
                rows = cursor.fetchall()
                cursor.close()

            return [
                {
                    "id": row[0],
                    "title": row[1],
                    "description": row[2],
                    "status": row[3],
                    "priority": row[4],
                    "source": row[5],
                    "related_topics": row[6] or [],
                    "last_mentioned": row[7],
                    "created_at": row[8],
                    "progress_notes": row[9] or [],
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"[GOALS] get_all_goals failed: {e}")
            return []
