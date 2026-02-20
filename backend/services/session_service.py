"""
Session Service - Track conversation sessions and detect episode generation triggers.

When a thread_id is provided, session state is stored in the Redis thread hash
(survives restarts). Otherwise falls back to in-memory state (legacy behavior).
"""

import logging
import time
from typing import Tuple, List


class SessionService:
    """Manages conversation sessions and episode generation triggers."""

    def __init__(self, inactivity_timeout: int = 600):
        """
        Initialize session service.

        Args:
            inactivity_timeout: Seconds of inactivity before triggering episode (default 600 = 10 min)
        """
        self.inactivity_timeout = inactivity_timeout

        # In-memory state (legacy, used when no thread_id)
        self.current_topic: str = None
        self.session_exchanges: List[dict] = []
        self.session_start_time: float = None
        self.last_activity_time: float = None
        self.topic_exchange_count: int = 0
        self.global_exchange_count: int = 0

        # Thread-backed state
        self._thread_id: str = None
        self._redis = None

    def set_thread(self, thread_id: str):
        """Bind this session to a thread for persistent state."""
        self._thread_id = thread_id
        if thread_id:
            try:
                from services.redis_client import RedisClientService
                self._redis = RedisClientService.create_connection()
            except Exception:
                self._redis = None

    def _get_thread_field(self, field: str, default: str = "0") -> str:
        """Read a field from the thread hash."""
        if self._thread_id and self._redis:
            try:
                val = self._redis.hget(f"thread:{self._thread_id}", field)
                return val if val is not None else default
            except Exception:
                pass
        return default

    def _set_thread_field(self, field: str, value: str):
        """Write a field to the thread hash."""
        if self._thread_id and self._redis:
            try:
                self._redis.hset(f"thread:{self._thread_id}", field, value)
            except Exception:
                pass

    def track_classification(self, topic: str, is_new_topic: bool, timestamp: float):
        """Track a classification result and update session state."""
        self.last_activity_time = timestamp

        if self._thread_id:
            self._set_thread_field("last_activity", str(timestamp))

        if self.session_start_time is None:
            self.session_start_time = timestamp
            self.current_topic = topic
            logging.info(f"Session started with topic: {topic}")
            return

        self.current_topic = topic

    def should_generate_episode(self) -> Tuple[bool, str]:
        """Check if an episode should be generated based on session state."""
        if self.session_start_time is None or not self.session_exchanges:
            return (False, "")

        topic_count = self.topic_exchange_count
        global_count = self.global_exchange_count

        # If thread-backed, read from Redis
        if self._thread_id:
            try:
                topic_count = int(self._get_thread_field("topic_exchange_count", "0"))
                global_count = int(self._get_thread_field("exchange_count", "0"))
            except (ValueError, TypeError):
                pass

        if topic_count >= 3:
            logging.info(f"Topic exchange threshold reached: {topic_count} exchanges in '{self.current_topic}'")
            return (True, f"topic_exchange_threshold ({topic_count} exchanges in topic)")

        if global_count >= 5:
            logging.info(f"Global exchange threshold reached: {global_count} exchanges across all topics")
            return (True, f"global_exchange_threshold ({global_count} exchanges)")

        if self.last_activity_time:
            inactive_seconds = time.time() - self.last_activity_time
            if inactive_seconds >= self.inactivity_timeout:
                logging.info(f"Inactivity timeout reached: {inactive_seconds:.0f}s")
                return (True, "inactivity")

        return (False, "")

    def add_exchange(self, exchange: dict):
        """Add an exchange to the current session."""
        self.session_exchanges.append(exchange)
        self.topic_exchange_count += 1
        self.global_exchange_count += 1

        if self._thread_id:
            self._set_thread_field("topic_exchange_count", str(self.topic_exchange_count))

    def get_session_data(self) -> dict:
        """Get current session data for episode generation."""
        return {
            'topic': self.current_topic,
            'exchanges': self.session_exchanges.copy(),
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S',
                                       time.localtime(self.session_start_time)) if self.session_start_time else None
        }

    def reset_session(self):
        """Reset session state after episode generation."""
        logging.info(f"Resetting session (topic: {self.current_topic})")
        self.session_exchanges = []
        self.session_start_time = time.time()
        self.last_activity_time = time.time()
        self.topic_exchange_count = 0
        self.global_exchange_count = 0

        if self._thread_id:
            self._set_thread_field("topic_exchange_count", "0")

    def get_last_activity_time(self) -> float:
        """Get the timestamp of last activity."""
        return self.last_activity_time

    def check_topic_switch(self, new_topic: str) -> bool:
        """Check if a topic switch has occurred."""
        if self.current_topic is None:
            return True
        return new_topic.lower() != self.current_topic.lower()

    def mark_topic_switch(self, new_topic: str):
        """Mark that a topic switch has occurred."""
        logging.info(f"Topic switch detected: {self.current_topic} -> {new_topic}")
        self.current_topic = new_topic
        self.topic_exchange_count = 0

        if self._thread_id:
            self._set_thread_field("topic_exchange_count", "0")
