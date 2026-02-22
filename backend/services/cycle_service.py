"""
Cycle Service — Lifecycle management for message cycles.

Every message (user input, fast response, tool result, delegate result, proactive drift)
gets its own cycle record. Cycles form a tree via parent_cycle_id, with root_cycle_id
tracing back to the original user message.

Enforces throttling rules: max depth, max follow-ups per root, max active background cycles.
"""

import logging
import time
from typing import Dict, Any, Optional, List

from services.database_service import DatabaseService

logger = logging.getLogger(__name__)

CYCLE_LIMITS = {
    'max_followups_per_root': 2,
    'max_active_background_cycles': 3,
    'max_depth': 5,
}

VALID_TYPES = {
    'user_input', 'fast_response', 'tool_work', 'tool_result',
    'proactive_drift', 'act_followup',
}
VALID_STATUSES = {'pending', 'processing', 'completed', 'failed', 'cancelled', 'suppressed', 'expired'}


class CycleService:
    """Manages message cycle lifecycle in PostgreSQL."""

    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service

    def create_cycle(
        self,
        content: str,
        topic: str,
        cycle_type: str,
        source: str,
        parent_cycle_id: Optional[str] = None,
        intent: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Create a new message cycle.

        Args:
            content: Message content
            topic: Topic name
            cycle_type: One of VALID_TYPES
            source: Origin ('user', 'system', 'tool_worker', 'drift_engine')
            parent_cycle_id: Parent cycle UUID (None for root cycles)
            intent: Classified intent metadata
            metadata: Additional metadata

        Returns:
            cycle_id UUID string, or None on failure
        """
        try:
            # Determine root and depth
            root_cycle_id = None
            depth = 0

            if parent_cycle_id:
                parent = self.get_cycle(parent_cycle_id)
                if parent:
                    root_cycle_id = parent['root_cycle_id']
                    depth = parent['depth'] + 1
                else:
                    logger.warning(f"[CYCLE] Parent cycle {parent_cycle_id} not found")
                    return None

            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                if root_cycle_id is None:
                    # Root cycle — cycle_id will be set as root_cycle_id after insert
                    cursor.execute("""
                        INSERT INTO message_cycles
                            (topic, cycle_type, source, content, intent, metadata, status, depth, root_cycle_id)
                        VALUES (%s, %s, %s, %s, %s, %s, 'pending', %s, gen_random_uuid())
                        RETURNING cycle_id
                    """, (
                        topic, cycle_type, source, content,
                        _json_or_none(intent),
                        _json_or_none(metadata or {}),
                        depth,
                    ))
                    row = cursor.fetchone()
                    cycle_id = str(row[0])

                    # Set root_cycle_id = cycle_id for root cycles
                    cursor.execute(
                        "UPDATE message_cycles SET root_cycle_id = cycle_id WHERE cycle_id = %s",
                        (cycle_id,)
                    )
                else:
                    cursor.execute("""
                        INSERT INTO message_cycles
                            (parent_cycle_id, root_cycle_id, topic, cycle_type, source,
                             content, intent, metadata, status, depth)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s)
                        RETURNING cycle_id
                    """, (
                        parent_cycle_id, root_cycle_id, topic, cycle_type, source,
                        content, _json_or_none(intent), _json_or_none(metadata or {}), depth,
                    ))
                    row = cursor.fetchone()
                    cycle_id = str(row[0])

                cursor.close()

            logger.info(f"[CYCLE] Created {cycle_type} cycle {cycle_id[:8]} (depth={depth}, topic={topic})")
            return cycle_id

        except Exception as e:
            logger.error(f"[CYCLE] Failed to create cycle: {e}")
            return None

    def complete_cycle(self, cycle_id: str, status: str = 'completed') -> bool:
        """Mark a cycle as completed (or failed/cancelled/suppressed)."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE message_cycles
                    SET status = %s, completed_at = NOW()
                    WHERE cycle_id = %s
                """, (status, cycle_id))
                cursor.close()
            logger.info(f"[CYCLE] Cycle {cycle_id[:8]} → {status}")
            return True
        except Exception as e:
            logger.error(f"[CYCLE] Failed to complete cycle {cycle_id[:8]}: {e}")
            return False

    def update_cycle_status(self, cycle_id: str, status: str) -> bool:
        """Update cycle status without setting completed_at."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE message_cycles SET status = %s WHERE cycle_id = %s",
                    (status, cycle_id)
                )
                cursor.close()
            return True
        except Exception as e:
            logger.error(f"[CYCLE] Failed to update status: {e}")
            return False

    def get_cycle(self, cycle_id: str) -> Optional[Dict[str, Any]]:
        """Get a single cycle by ID."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT cycle_id, parent_cycle_id, root_cycle_id, topic, cycle_type,
                           source, content, intent, metadata, status, depth,
                           created_at, completed_at
                    FROM message_cycles WHERE cycle_id = %s
                """, (cycle_id,))
                row = cursor.fetchone()
                cursor.close()
                return _row_to_dict(row) if row else None
        except Exception as e:
            logger.error(f"[CYCLE] Failed to get cycle: {e}")
            return None

    def get_cycle_chain(self, root_cycle_id: str) -> List[Dict[str, Any]]:
        """Get all cycles in a lineage, ordered by creation time."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT cycle_id, parent_cycle_id, root_cycle_id, topic, cycle_type,
                           source, content, intent, metadata, status, depth,
                           created_at, completed_at
                    FROM message_cycles
                    WHERE root_cycle_id = %s
                    ORDER BY created_at ASC
                """, (root_cycle_id,))
                rows = cursor.fetchall()
                cursor.close()
                return [_row_to_dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[CYCLE] Failed to get chain: {e}")
            return []

    def get_recent_cycles(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent cycles for a topic."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT cycle_id, parent_cycle_id, root_cycle_id, topic, cycle_type,
                           source, content, intent, metadata, status, depth,
                           created_at, completed_at
                    FROM message_cycles
                    WHERE topic = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (topic, limit))
                rows = cursor.fetchall()
                cursor.close()
                return [_row_to_dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[CYCLE] Failed to get recent cycles: {e}")
            return []

    def get_active_cycles(
        self,
        topic: Optional[str] = None,
        cycle_type: Optional[str] = None,
        status: str = 'processing',
    ) -> List[Dict[str, Any]]:
        """Get active cycles, optionally filtered by topic and type."""
        try:
            conditions = ["status = %s"]
            params = [status]

            if topic:
                conditions.append("topic = %s")
                params.append(topic)
            if cycle_type:
                conditions.append("cycle_type = %s")
                params.append(cycle_type)

            where = " AND ".join(conditions)

            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT cycle_id, parent_cycle_id, root_cycle_id, topic, cycle_type,
                           source, content, intent, metadata, status, depth,
                           created_at, completed_at
                    FROM message_cycles
                    WHERE {where}
                    ORDER BY created_at DESC
                """, params)
                rows = cursor.fetchall()
                cursor.close()
                return [_row_to_dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[CYCLE] Failed to get active cycles: {e}")
            return []

    def can_spawn_child(self, parent_cycle_id: str) -> bool:
        """
        Check if a child cycle can be spawned from this parent.

        Enforces:
        - max_depth (prevent recursion)
        - max_followups_per_root (limit follow-ups)
        - max_active_background_cycles (global throttle)
        """
        try:
            parent = self.get_cycle(parent_cycle_id)
            if not parent:
                return False

            # Depth check
            if parent['depth'] >= CYCLE_LIMITS['max_depth']:
                logger.info(f"[CYCLE] Spawn blocked: depth {parent['depth']} >= max {CYCLE_LIMITS['max_depth']}")
                return False

            # Follow-up count under this root
            root_id = parent['root_cycle_id']
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT COUNT(*) FROM message_cycles
                    WHERE root_cycle_id = %s
                    AND cycle_type IN ('tool_result', 'act_followup')
                """, (root_id,))
                followup_count = cursor.fetchone()[0]

                if followup_count >= CYCLE_LIMITS['max_followups_per_root']:
                    logger.info(f"[CYCLE] Spawn blocked: {followup_count} follow-ups >= max {CYCLE_LIMITS['max_followups_per_root']}")
                    cursor.close()
                    return False

                # Active background cycle count
                cursor.execute("""
                    SELECT COUNT(*) FROM message_cycles
                    WHERE status = 'processing'
                    AND cycle_type IN ('tool_work')
                """)
                active_bg = cursor.fetchone()[0]
                cursor.close()

                if active_bg >= CYCLE_LIMITS['max_active_background_cycles']:
                    logger.info(f"[CYCLE] Spawn blocked: {active_bg} active bg cycles >= max {CYCLE_LIMITS['max_active_background_cycles']}")
                    return False

            return True

        except Exception as e:
            logger.error(f"[CYCLE] Spawn check failed: {e}")
            return False

    def get_defer_count(self, cycle_id: str) -> int:
        """Get the number of times a cycle has been deferred."""
        cycle = self.get_cycle(cycle_id)
        if cycle and cycle.get('metadata'):
            return cycle['metadata'].get('defer_count', 0)
        return 0

    def increment_defer_count(self, cycle_id: str) -> bool:
        """Increment the defer count in cycle metadata."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE message_cycles
                    SET metadata = jsonb_set(
                        COALESCE(metadata, '{}'::jsonb),
                        '{defer_count}',
                        (COALESCE((metadata->>'defer_count')::int, 0) + 1)::text::jsonb
                    )
                    WHERE cycle_id = %s
                """, (cycle_id,))
                cursor.close()
            return True
        except Exception as e:
            logger.error(f"[CYCLE] Failed to increment defer count: {e}")
            return False

    def expire_stale_cycles(self) -> int:
        """Mark stale cycles as expired. Returns count of expired cycles."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE message_cycles
                    SET status = 'expired', completed_at = NOW()
                    WHERE status IN ('pending', 'processing')
                    AND created_at < NOW() - INTERVAL '10 minutes'
                    RETURNING cycle_id
                """)
                expired = cursor.rowcount
                cursor.close()

            if expired > 0:
                logger.info(f"[CYCLE] Expired {expired} stale cycles")
            return expired
        except Exception as e:
            logger.error(f"[CYCLE] Failed to expire stale cycles: {e}")
            return 0


def _json_or_none(obj):
    """Convert dict to JSON string for PostgreSQL JSONB, or None."""
    import json
    if obj is None:
        return None
    return json.dumps(obj)


def _row_to_dict(row) -> Dict[str, Any]:
    """Convert a database row to a dict."""
    return {
        'cycle_id': str(row[0]),
        'parent_cycle_id': str(row[1]) if row[1] else None,
        'root_cycle_id': str(row[2]),
        'topic': row[3],
        'cycle_type': row[4],
        'source': row[5],
        'content': row[6],
        'intent': row[7],
        'metadata': row[8] or {},
        'status': row[9],
        'depth': row[10],
        'created_at': row[11],
        'completed_at': row[12],
    }
