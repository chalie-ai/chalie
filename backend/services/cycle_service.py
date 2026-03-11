"""
Cycle Service — Lifecycle management for message cycles.

Every message (user input, fast response, tool result, delegate result, proactive drift)
gets its own cycle record. Cycles form a tree via parent_cycle_id, with root_cycle_id
tracing back to the original user message.

Enforces throttling rules: max depth, max follow-ups per root, max active background cycles.
"""

import json
import uuid
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
    """Manages message cycle lifecycle in SQLite."""

    def __init__(self, db_service: DatabaseService):
        """Initialize the cycle service.

        Args:
            db_service: DatabaseService instance used for all SQLite operations.
        """
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

            cycle_id = str(uuid.uuid4())

            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                if root_cycle_id is None:
                    # Root cycle — root_cycle_id = cycle_id
                    cursor.execute("""
                        INSERT INTO message_cycles
                            (cycle_id, root_cycle_id, topic, cycle_type, source,
                             content, intent, metadata, status, depth)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                    """, (
                        cycle_id, cycle_id, topic, cycle_type, source, content,
                        _json_or_none(intent),
                        _json_or_none(metadata or {}),
                        depth,
                    ))
                else:
                    cursor.execute("""
                        INSERT INTO message_cycles
                            (cycle_id, parent_cycle_id, root_cycle_id, topic, cycle_type, source,
                             content, intent, metadata, status, depth)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                    """, (
                        cycle_id, parent_cycle_id, root_cycle_id, topic, cycle_type, source,
                        content, _json_or_none(intent), _json_or_none(metadata or {}), depth,
                    ))

                cursor.close()

            logger.info(f"[CYCLE] Created {cycle_type} cycle {cycle_id[:8]} (depth={depth}, topic={topic})")
            return cycle_id

        except Exception as e:
            logger.error(f"[CYCLE] Failed to create cycle: {e}")
            return None

    def complete_cycle(self, cycle_id: str, status: str = 'completed') -> bool:
        """Mark a cycle as completed, failed, cancelled, or suppressed.

        Sets the completed_at timestamp and updates status in a single query.

        Args:
            cycle_id: UUID string of the cycle to complete.
            status: Terminal status to set (default: 'completed').

        Returns:
            True on success, False if the database update fails.
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE message_cycles
                    SET status = ?, completed_at = datetime('now')
                    WHERE cycle_id = ?
                """, (status, cycle_id))
                cursor.close()
            logger.info(f"[CYCLE] Cycle {cycle_id[:8]} -> {status}")
            return True
        except Exception as e:
            logger.error(f"[CYCLE] Failed to complete cycle {cycle_id[:8]}: {e}")
            return False

    def update_cycle_status(self, cycle_id: str, status: str) -> bool:
        """Update cycle status without setting the completed_at timestamp.

        Args:
            cycle_id: UUID string of the cycle to update.
            status: New status string to set.

        Returns:
            True on success, False if the database update fails.
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE message_cycles SET status = ? WHERE cycle_id = ?",
                    (status, cycle_id)
                )
                cursor.close()
            return True
        except Exception as e:
            logger.error(f"[CYCLE] Failed to update status: {e}")
            return False

    def get_cycle(self, cycle_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single cycle record by its ID.

        Args:
            cycle_id: UUID string of the cycle to retrieve.

        Returns:
            Cycle dict, or None if not found or on error.
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT cycle_id, parent_cycle_id, root_cycle_id, topic, cycle_type,
                           source, content, intent, metadata, status, depth,
                           created_at, completed_at
                    FROM message_cycles WHERE cycle_id = ?
                """, (cycle_id,))
                row = cursor.fetchone()
                cursor.close()
                return _row_to_dict(row) if row else None
        except Exception as e:
            logger.error(f"[CYCLE] Failed to get cycle: {e}")
            return None

    def get_cycle_chain(self, root_cycle_id: str) -> List[Dict[str, Any]]:
        """Get all cycles belonging to a lineage tree, ordered by creation time.

        Args:
            root_cycle_id: UUID string of the root cycle whose lineage to retrieve.

        Returns:
            List of cycle dicts ordered by created_at ascending, or [] on error.
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT cycle_id, parent_cycle_id, root_cycle_id, topic, cycle_type,
                           source, content, intent, metadata, status, depth,
                           created_at, completed_at
                    FROM message_cycles
                    WHERE root_cycle_id = ?
                    ORDER BY created_at ASC
                """, (root_cycle_id,))
                rows = cursor.fetchall()
                cursor.close()
                return [_row_to_dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[CYCLE] Failed to get chain: {e}")
            return []

    def get_recent_cycles(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent cycles for a given topic.

        Args:
            topic: Topic name to filter by.
            limit: Maximum number of cycles to return (default: 10).

        Returns:
            List of cycle dicts ordered by created_at descending, or [] on error.
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT cycle_id, parent_cycle_id, root_cycle_id, topic, cycle_type,
                           source, content, intent, metadata, status, depth,
                           created_at, completed_at
                    FROM message_cycles
                    WHERE topic = ?
                    ORDER BY created_at DESC
                    LIMIT ?
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
        """Get cycles matching a given status, optionally filtered by topic and type.

        Args:
            topic: Optional topic name to filter by.
            cycle_type: Optional cycle type to filter by.
            status: Status string to match (default: 'processing').

        Returns:
            List of matching cycle dicts ordered by created_at descending, or [] on error.
        """
        try:
            conditions = ["status = ?"]
            params = [status]

            if topic:
                conditions.append("topic = ?")
                params.append(topic)
            if cycle_type:
                conditions.append("cycle_type = ?")
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
                    WHERE root_cycle_id = ?
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
        """Get the number of times a cycle has been deferred.

        Args:
            cycle_id: UUID string of the cycle to check.

        Returns:
            Defer count integer, or 0 if the cycle is not found.
        """
        cycle = self.get_cycle(cycle_id)
        if cycle and cycle.get('metadata'):
            return cycle['metadata'].get('defer_count', 0)
        return 0

    def increment_defer_count(self, cycle_id: str) -> bool:
        """Increment the defer count stored in the cycle's metadata JSON.

        Args:
            cycle_id: UUID string of the cycle to update.

        Returns:
            True on success, False if the cycle is not found or on error.
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # Fetch current metadata, update in Python, write back
                cursor.execute(
                    "SELECT metadata FROM message_cycles WHERE cycle_id = ?",
                    (cycle_id,)
                )
                row = cursor.fetchone()
                if not row:
                    cursor.close()
                    return False

                metadata = row[0] if isinstance(row[0], dict) else (json.loads(row[0]) if row[0] else {})
                metadata['defer_count'] = metadata.get('defer_count', 0) + 1

                cursor.execute(
                    "UPDATE message_cycles SET metadata = ? WHERE cycle_id = ?",
                    (json.dumps(metadata), cycle_id)
                )
                cursor.close()
            return True
        except Exception as e:
            logger.error(f"[CYCLE] Failed to increment defer count: {e}")
            return False

    def expire_stale_cycles(self) -> int:
        """Mark stale pending/processing cycles older than 10 minutes as expired.

        Returns:
            Count of cycles transitioned to 'expired' status.
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE message_cycles
                    SET status = 'expired', completed_at = datetime('now')
                    WHERE status IN ('pending', 'processing')
                    AND created_at < datetime('now', '-10 minutes')
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
    """Convert a dict to a JSON string for SQLite TEXT storage, or return None.

    Args:
        obj: Dict to serialise, or None.

    Returns:
        JSON string representation of obj, or None if obj is None.
    """
    if obj is None:
        return None
    return json.dumps(obj)


def _row_to_dict(row) -> Dict[str, Any]:
    """Convert a database row tuple to a cycle dict.

    Args:
        row: Tuple of column values from the message_cycles SELECT query.

    Returns:
        Dict with cycle fields keyed by name.
    """
    return {
        'cycle_id': str(row[0]),
        'parent_cycle_id': str(row[1]) if row[1] else None,
        'root_cycle_id': str(row[2]),
        'topic': row[3],
        'cycle_type': row[4],
        'source': row[5],
        'content': row[6],
        'intent': row[7] if isinstance(row[7], dict) else (json.loads(row[7]) if row[7] else None),
        'metadata': row[8] if isinstance(row[8], dict) else (json.loads(row[8]) if row[8] else {}),
        'status': row[9],
        'depth': row[10],
        'created_at': row[11],
        'completed_at': row[12],
    }
