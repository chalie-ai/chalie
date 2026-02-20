"""
Event Bus Service - Formal emit/subscribe pattern for async events.

Thin coordination layer using Redis lists for durable delivery.
Event handlers translate events into existing RQ queue enqueue operations.
"""

import json
import time
import logging
from typing import Callable, Dict, Any, List, Optional
from services.redis_client import RedisClientService

# Event type constants
ENCODE_EVENT = 'encode_event'
UPDATE_POLICY = 'update_policy'
HANDLE_ACTION = 'handle_action'
INTERACTION_LOGGED = 'interaction_logged'


class EventBusService:
    """Manages event emission and subscription via Redis lists."""

    def __init__(self):
        """Initialize event bus with Redis connection."""
        self.redis = RedisClientService.create_connection()
        self._handlers: Dict[str, List[Callable]] = {}

    def _get_event_queue_key(self, event_type: str) -> str:
        """Generate Redis key for an event queue."""
        return f"event_bus:{event_type}"

    def emit(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """
        Emit an event to the event bus.

        Args:
            event_type: Type of event (ENCODE_EVENT, UPDATE_POLICY, etc.)
            payload: Event payload dict

        Returns:
            True if event was emitted successfully
        """
        event = {
            'event_type': event_type,
            'payload': payload,
            'timestamp': time.time()
        }

        queue_key = self._get_event_queue_key(event_type)

        try:
            self.redis.rpush(queue_key, json.dumps(event))
            logging.debug(f"[EVENT BUS] Emitted {event_type}")
            return True
        except Exception as e:
            logging.error(f"[EVENT BUS] Failed to emit {event_type}: {e}")
            return False

    def subscribe(self, event_type: str, handler: Callable):
        """
        Register a handler for an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Callable that accepts (event_type, payload) args
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logging.info(f"[EVENT BUS] Subscribed handler to {event_type}")

    def process_events(self, event_type: str, batch_size: int = 10) -> int:
        """
        Process pending events for a given event type.

        Args:
            event_type: Type of events to process
            batch_size: Maximum events to process in one call

        Returns:
            Number of events processed
        """
        handlers = self._handlers.get(event_type, [])
        if not handlers:
            return 0

        queue_key = self._get_event_queue_key(event_type)
        processed = 0

        for _ in range(batch_size):
            event_json = self.redis.lpop(queue_key)
            if not event_json:
                break

            try:
                event = json.loads(event_json)
                payload = event.get('payload', {})

                for handler in handlers:
                    try:
                        handler(event_type, payload)
                    except Exception as e:
                        logging.error(
                            f"[EVENT BUS] Handler error for {event_type}: {e}"
                        )

                processed += 1
            except json.JSONDecodeError:
                logging.error(f"[EVENT BUS] Invalid event JSON: {event_json}")

        if processed > 0:
            logging.debug(f"[EVENT BUS] Processed {processed} {event_type} events")

        return processed

    def get_pending_count(self, event_type: str) -> int:
        """
        Get number of pending events for an event type.

        Args:
            event_type: Event type to check

        Returns:
            Number of pending events
        """
        queue_key = self._get_event_queue_key(event_type)
        return self.redis.llen(queue_key)

    def emit_and_handle(self, event_type: str, payload: Dict[str, Any]):
        """
        Emit an event and immediately process it with registered handlers.
        Useful for synchronous event handling without going through Redis.

        Args:
            event_type: Type of event
            payload: Event payload
        """
        handlers = self._handlers.get(event_type, [])
        if not handlers:
            # No handlers registered, emit to Redis for later processing
            self.emit(event_type, payload)
            return

        for handler in handlers:
            try:
                handler(event_type, payload)
            except Exception as e:
                logging.error(f"[EVENT BUS] Handler error for {event_type}: {e}")
