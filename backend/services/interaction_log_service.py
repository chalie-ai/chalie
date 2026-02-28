"""
Interaction Log Service - Immutable audit trail of all raw events.

Append-only PostgreSQL-backed log of user inputs, classifications, and system responses.
Follows EpisodicStorageService pattern (DatabaseService injection, get_connection/release_connection).
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from services.database_service import DatabaseService


# Activity-relevant event types (autonomous actions only, not conversation)
_ACTIVITY_EVENT_TYPES = (
    'proactive_sent', 'proactive_candidate',
    'act_loop_telemetry', 'cron_tool_executed',
    'plan_proposed', 'curiosity_thread_seeded',
    'spark_nurture_sent', 'spark_suggestion_sent',
    'spark_phase_change', 'place_transition',
)


def _summarize_event(event_type: str, payload: dict) -> str:
    """One-line human-readable summary of an autonomous event."""
    p = payload or {}
    summaries = {
        'proactive_sent': lambda: f"Shared a thought: {p.get('response', '')[:80]}",
        'act_loop_telemetry': lambda: f"Ran {p.get('actions_total', 0)} actions ({p.get('termination_reason', 'completed')})",
        'cron_tool_executed': lambda: f"Ran {p.get('tool_name', 'tool')} in background",
        'plan_proposed': lambda: f"Proposed background task: {p.get('topic', 'unknown')}",
        'curiosity_thread_seeded': lambda: "Started exploring a new curiosity thread",
        'spark_nurture_sent': lambda: "Sent a relationship check-in",
        'spark_suggestion_sent': lambda: f"Suggested: {p.get('skill_name', 'something')}",
        'spark_phase_change': lambda: "Relationship phase updated",
        'place_transition': lambda: "Noticed a location change",
        'proactive_candidate': lambda: f"Considered sharing: {p.get('thought_content', '')[:60]}",
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
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO interaction_log (
                        event_type, topic, exchange_id, session_id, source,
                        payload, metadata, thread_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    event_type,
                    topic,
                    exchange_id,
                    session_id,
                    source,
                    json.dumps(payload),
                    json.dumps(metadata or {}),
                    thread_id,
                ))

                event_id = cursor.fetchone()[0]
                cursor.close()

                logging.info(f"[INTERACTION LOG] Logged {event_type} event {event_id} for topic '{topic}'")
                return str(event_id)

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
                        WHERE topic = %s AND event_type = %s
                        ORDER BY created_at ASC
                        LIMIT %s
                    """, (topic, event_type, limit))
                else:
                    cursor.execute("""
                        SELECT id, event_type, topic, exchange_id, session_id, source,
                               payload, metadata, created_at
                        FROM interaction_log
                        WHERE topic = %s
                        ORDER BY created_at ASC
                        LIMIT %s
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
                    WHERE exchange_id = %s
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
                    WHERE topic = %s AND event_type IN ('user_input', 'system_response')
                    ORDER BY created_at ASC
                    LIMIT %s
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
        since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        items = []

        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # 1. Autonomous events from interaction_log
                placeholders = ','.join(['%s'] * len(_ACTIVITY_EVENT_TYPES))
                cursor.execute(
                    f"SELECT id, event_type, topic, payload, source, created_at "
                    f"FROM interaction_log "
                    f"WHERE created_at > %s AND event_type IN ({placeholders}) "
                    f"ORDER BY created_at DESC",
                    (since, *_ACTIVITY_EVENT_TYPES)
                )
                for row in cursor.fetchall():
                    items.append({
                        'source': 'autonomous',
                        'type': row[1],
                        'topic': row[2],
                        'summary': _summarize_event(row[1], row[3]),
                        'detail': row[3],
                        'occurred_at': row[5].isoformat() if row[5] else None,
                    })

                # 2. Persistent tasks that changed in the window
                try:
                    cursor.execute(
                        "SELECT id, goal, status, progress, updated_at "
                        "FROM persistent_tasks "
                        "WHERE updated_at > %s "
                        "ORDER BY updated_at DESC",
                        (since,)
                    )
                    for row in cursor.fetchall():
                        progress = row[3] or {}
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
                            'occurred_at': row[4].isoformat() if row[4] else None,
                        })
                except Exception as e:
                    logging.debug(f"[ACTIVITY FEED] persistent_tasks unavailable: {e}")

                # 3. Fired scheduled items
                try:
                    cursor.execute(
                        "SELECT id, message, item_type, topic, last_fired_at "
                        "FROM scheduled_items "
                        "WHERE status = 'fired' AND last_fired_at > %s "
                        "ORDER BY last_fired_at DESC",
                        (since,)
                    )
                    for row in cursor.fetchall():
                        items.append({
                            'source': 'scheduler',
                            'type': 'reminder_fired',
                            'topic': row[3],
                            'summary': row[1],
                            'detail': {'item_type': row[2], 'schedule_id': row[0]},
                            'occurred_at': row[4].isoformat() if row[4] else None,
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

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert a database row to a dict."""
        return {
            'id': str(row[0]),
            'event_type': row[1],
            'topic': row[2],
            'exchange_id': row[3],
            'session_id': row[4],
            'source': row[5],
            'payload': row[6],
            'metadata': row[7],
            'created_at': row[8]
        }
