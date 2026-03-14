"""
Interaction Log Service - Immutable audit trail of all raw events.

Append-only SQLite-backed log of user inputs, classifications, and system responses.
Follows EpisodicStorageService pattern (DatabaseService injection, get_connection/release_connection).
"""

import json
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from services.database_service import DatabaseService


# Activity-relevant event types (autonomous actions only, not conversation).
# proactive_candidate intentionally excluded — those are thoughts Chalie
# decided NOT to send; surfacing them undermines the judgment layer.
_ACTIVITY_EVENT_TYPES = (
    'proactive_sent',
    'act_loop_telemetry', 'cron_tool_executed',
    'plan_proposed', 'curiosity_thread_seeded',
    'place_transition',
)


def _summarize_event(event_type: str, payload: dict) -> str:
    """One-line human-readable summary of an autonomous event.

    Args:
        event_type: The event type string (e.g. ``'proactive_sent'``).
        payload: Event payload dict; may be ``None`` or empty.

    Returns:
        A concise human-readable string describing the event.
    """
    p = payload or {}
    summaries = {
        'proactive_sent': lambda: f"Shared a thought: {p.get('response', '')[:80]}",
        'act_loop_telemetry': lambda: f"Ran {p.get('actions_total', 0)} actions ({p.get('termination_reason', 'completed')})",
        'cron_tool_executed': lambda: f"Ran {p.get('tool_name', 'tool')} in background",
        'plan_proposed': lambda: f"Proposed background task: {p.get('topic', 'unknown')}",
        'curiosity_thread_seeded': lambda: "Started exploring a new curiosity thread",
        'place_transition': lambda: "Noticed a location change",
    }
    fn = summaries.get(event_type, lambda: event_type.replace('_', ' ').title())
    try:
        return fn()
    except Exception:
        return event_type.replace('_', ' ').title()


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
        topic: str = None,
        exchange_id: str = None,
        session_id: str = None,
        source: str = None,
        metadata: Dict[str, Any] = None,
        thread_id: str = None
    ) -> Optional[str]:
        """
        Log an event to the interaction log.

        Args:
            event_type: Type of event (user_input, classification, system_response, etc.)
            payload: Event-specific data
            topic: Conversation topic
            exchange_id: Exchange identifier
            session_id: Session identifier
            source: Event source (telegram, rest_api, etc.)
            metadata: Optional metadata dict

        Returns:
            UUID of the created log entry, or None on failure
        """
        try:
            event_id = str(uuid.uuid4())

            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO interaction_log (
                        id, event_type, topic, exchange_id, session_id, source,
                        payload, metadata, thread_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_id,
                    event_type,
                    topic,
                    exchange_id,
                    session_id,
                    source,
                    json.dumps(payload),
                    json.dumps(metadata or {}),
                    thread_id,
                ))

                cursor.close()

                logging.info(f"[INTERACTION LOG] Logged {event_type} event {event_id} for topic '{topic}'")
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
                        SELECT id, event_type, topic, exchange_id, session_id, source,
                               payload, metadata, created_at
                        FROM interaction_log
                        WHERE topic = ? AND event_type = ?
                        ORDER BY created_at ASC
                        LIMIT ?
                    """, (topic, event_type, limit))
                else:
                    cursor.execute("""
                        SELECT id, event_type, topic, exchange_id, session_id, source,
                               payload, metadata, created_at
                        FROM interaction_log
                        WHERE topic = ?
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
                    SELECT id, event_type, topic, exchange_id, session_id, source,
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
                    SELECT id, event_type, topic, exchange_id, session_id, source,
                           payload, metadata, created_at
                    FROM interaction_log
                    WHERE topic = ? AND event_type IN ('user_input', 'system_response')
                    ORDER BY created_at ASC
                    LIMIT ?
                """, (topic, limit))

                rows = cursor.fetchall()
                cursor.close()

                return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            logging.error(f"[INTERACTION LOG] Failed to get transcript: {e}")
            return []

    def get_activity_feed(self, since_hours: int = 24, limit: int = 50, offset: int = 0) -> dict:
        """
        Unified activity feed: what Chalie did autonomously.
        Aggregates interaction_log + persistent_tasks + scheduled_items
        into a single chronological feed.

        Args:
            since_hours: How far back to look (default 24h, max 168h/7 days)
            limit: Maximum items to return
            offset: Pagination offset

        Returns:
            dict with 'items', 'total', and 'since_hours'
        """
        since_hours = max(1, min(since_hours, 168))
        since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        since_str = since.isoformat()
        items = []

        # SQL-level cap prevents memory exhaustion on long time windows.
        # We fetch more than `limit` to allow cross-source sorting, but bound it.
        sql_cap = min((offset + limit) * 3, 1000)

        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # 1. Autonomous events from interaction_log
                placeholders = ','.join(['?'] * len(_ACTIVITY_EVENT_TYPES))
                cursor.execute(
                    f"SELECT id, event_type, topic, payload, source, created_at "
                    f"FROM interaction_log "
                    f"WHERE created_at > ? AND event_type IN ({placeholders}) "
                    f"ORDER BY created_at DESC LIMIT ?",
                    (since_str, *_ACTIVITY_EVENT_TYPES, sql_cap)
                )
                for row in cursor.fetchall():
                    payload = row[3] if isinstance(row[3], dict) else (json.loads(row[3]) if row[3] else {})
                    items.append({
                        'source': 'autonomous',
                        'type': row[1],
                        'topic': row[2],
                        'summary': _summarize_event(row[1], payload),
                        'occurred_at': row[5],
                    })

                # 2. Persistent tasks that changed in the window
                try:
                    cursor.execute(
                        "SELECT id, goal, status, progress, updated_at "
                        "FROM persistent_tasks "
                        "WHERE updated_at > ? "
                        "ORDER BY updated_at DESC LIMIT ?",
                        (since_str, sql_cap)
                    )
                    for row in cursor.fetchall():
                        progress = row[3] if isinstance(row[3], dict) else (json.loads(row[3]) if row[3] else {})
                        items.append({
                            'source': 'background_task',
                            'type': f'task_{row[2]}',
                            'topic': row[1],
                            'summary': progress.get('last_summary', f'Task {row[2]}'),
                            'detail': {
                                'task_id': row[0],
                                'status': row[2],
                                'coverage': progress.get('coverage_estimate'),
                                'cycles': progress.get('cycles_completed'),
                            },
                            'occurred_at': row[4],
                        })
                except Exception as e:
                    logging.debug(f"[ACTIVITY FEED] persistent_tasks unavailable: {e}")

                # 3. Fired scheduled items
                try:
                    cursor.execute(
                        "SELECT id, message, item_type, topic, last_fired_at "
                        "FROM scheduled_items "
                        "WHERE status = 'fired' AND last_fired_at > ? "
                        "ORDER BY last_fired_at DESC LIMIT ?",
                        (since_str, sql_cap)
                    )
                    for row in cursor.fetchall():
                        items.append({
                            'source': 'scheduler',
                            'type': 'reminder_fired',
                            'topic': row[3],
                            'summary': row[1],
                            'detail': {'item_type': row[2], 'schedule_id': row[0]},
                            'occurred_at': row[4],
                        })
                except Exception as e:
                    logging.debug(f"[ACTIVITY FEED] scheduled_items unavailable: {e}")

        except Exception as e:
            logging.error(f"[ACTIVITY FEED] Feed query failed: {e}")

        # Sort all items chronologically (newest first) and apply pagination
        items.sort(key=lambda x: x.get('occurred_at') or '', reverse=True)
        total = len(items)
        items = items[offset:offset + limit]

        return {'items': items, 'total': total, 'since_hours': since_hours}

    def get_events_by_types(
        self,
        event_types: List[str],
        since_hours: int = 24,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve events matching any of the given event types within a time window.

        Args:
            event_types: List of event_type strings to match
            since_hours: How far back to look
            limit: Maximum number of events to return

        Returns:
            List of event dicts ordered by created_at descending (newest first)
        """
        if not event_types:
            return []
        try:
            since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
            since_str = since.isoformat()

            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                placeholders = ','.join(['?'] * len(event_types))
                cursor.execute(
                    f"SELECT id, event_type, topic, exchange_id, session_id, source, "
                    f"       payload, metadata, created_at "
                    f"FROM interaction_log "
                    f"WHERE event_type IN ({placeholders}) AND created_at > ? "
                    f"ORDER BY created_at DESC LIMIT ?",
                    (*event_types, since_str, limit),
                )
                rows = cursor.fetchall()
                cursor.close()
                return [self._row_to_dict(row) for row in rows]
        except Exception as e:
            logging.error(f"[INTERACTION LOG] Failed to get events by types: {e}")
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
            'topic': row[2],
            'exchange_id': row[3],
            'session_id': row[4],
            'source': row[5],
            'payload': row[6] if isinstance(row[6], dict) else (json.loads(row[6]) if row[6] else {}),
            'metadata': row[7] if isinstance(row[7], dict) else (json.loads(row[7]) if row[7] else {}),
            'created_at': row[8]
        }
