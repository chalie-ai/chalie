"""
Interaction Log Service - Immutable audit trail of all raw events.

Append-only SQLite-backed log of user inputs, classifications, and system responses.
Follows EpisodicService pattern (DatabaseService injection, get_connection/release_connection).
"""

import json
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from services.database_service import DatabaseService



class InteractionLogService:
    """Manages append-only interaction event logging."""

    def __init__(self, database_service: DatabaseService = None):
        """
        Initialize interaction log service.

        Args:
            database_service: DatabaseService instance for connection management.
                If None, uses the shared database service instance.
        """
        if database_service is None:
            from services.database_service import get_shared_db_service
            database_service = get_shared_db_service()
        self.db_service = database_service

    def log_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        channel: str = None,
        exchange_id: str = None,
        session_id: str = None,
        source: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Optional[str]:
        """Log an event to the interaction log."""
        try:
            event_id = str(uuid.uuid4())

            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO interaction_log (
                        id, event_type, channel, exchange_id, session_id, source,
                        payload, metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_id,
                    event_type,
                    channel,
                    exchange_id,
                    session_id,
                    source,
                    json.dumps(payload),
                    json.dumps(metadata or {}),
                ))

                cursor.close()

                logging.info(f"[INTERACTION LOG] Logged {event_type} event {event_id} for channel '{channel}'")
                return event_id

        except Exception as e:
            logging.error(f"[INTERACTION LOG] Failed to log event: {e}")
            return None

    def get_events_by_topic(
        self,
        topic: str,
        event_type: str = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve events for a topic, optionally filtered by event type.

        Args:
            topic: Topic to query
            event_type: Optional event type filter
            limit: Maximum number of events to return

        Returns:
            List of event dicts ordered by created_at ascending
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                if event_type:
                    cursor.execute("""
                        SELECT id, event_type, channel, exchange_id, session_id, source,
                               payload, metadata, created_at
                        FROM interaction_log
                        WHERE channel = ? AND event_type = ?
                        ORDER BY created_at ASC
                        LIMIT ?
                    """, (topic, event_type, limit))
                else:
                    cursor.execute("""
                        SELECT id, event_type, channel, exchange_id, session_id, source,
                               payload, metadata, created_at
                        FROM interaction_log
                        WHERE channel = ?
                        ORDER BY created_at ASC
                        LIMIT ?
                    """, (topic, limit))

                rows = cursor.fetchall()
                cursor.close()

                return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            logging.error(f"[INTERACTION LOG] Failed to get events by topic: {e}")
            return []

    def get_events_by_exchange(self, exchange_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all events for a specific exchange.

        Args:
            exchange_id: Exchange identifier

        Returns:
            List of event dicts ordered by created_at ascending
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT id, event_type, channel, exchange_id, session_id, source,
                           payload, metadata, created_at
                    FROM interaction_log
                    WHERE exchange_id = ?
                    ORDER BY created_at ASC
                """, (exchange_id,))

                rows = cursor.fetchall()
                cursor.close()

                return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            logging.error(f"[INTERACTION LOG] Failed to get events by exchange: {e}")
            return []

    def get_transcript(
        self,
        topic: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get a transcript of user_input and system_response events for a topic.

        Args:
            topic: Topic to query
            limit: Maximum number of events

        Returns:
            List of user_input and system_response events in chronological order
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT id, event_type, channel, exchange_id, session_id, source,
                           payload, metadata, created_at
                    FROM interaction_log
                    WHERE channel = ? AND event_type IN ('user_input', 'system_response')
                    ORDER BY created_at ASC
                    LIMIT ?
                """, (topic, limit))

                rows = cursor.fetchall()
                cursor.close()

                return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            logging.error(f"[INTERACTION LOG] Failed to get transcript: {e}")
            return []

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert an interaction_log table row to an event dict.

        Args:
            row: sqlite3 row (sequence) with positional columns matching the
                SELECT column order used in this service's queries.

        Returns:
            Event dict with keys ``id``, ``event_type``, ``topic``,
            ``exchange_id``, ``session_id``, ``source``, ``payload``
            (parsed from JSON), ``metadata`` (parsed from JSON), and
            ``created_at``.
        """
        return {
            'id': str(row[0]),
            'event_type': row[1],
            'channel': row[2],
            'exchange_id': row[3],
            'session_id': row[4],
            'source': row[5],
            'payload': row[6] if isinstance(row[6], dict) else (json.loads(row[6]) if row[6] else {}),
            'metadata': row[7] if isinstance(row[7], dict) else (json.loads(row[7]) if row[7] else {}),
            'created_at': row[8]
        }
