"""
Thread Service - Thread resolution and lifecycle management.

Provides user+channel scoped conversation threads with temporal lifecycle
(soft/hard expiry), replacing the global topic-only scoping.

Thread ID format: {platform}:{user_id}:{channel_id}:{sequence}
"""

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List

from services.redis_client import RedisClientService
from services.config_service import ConfigService


@dataclass
class ThreadResolution:
    thread_id: str
    is_new: bool
    is_resumed: bool
    resume_gap_minutes: float = 0.0
    previous_thread_id: Optional[str] = None
    recent_visible_context: Optional[List[dict]] = None


class ThreadService:
    """Manages thread resolution and lifecycle."""

    def __init__(self, soft_expiry_minutes: int = 30, hard_expiry_minutes: int = 240):
        self.redis = RedisClientService.create_connection()
        self.soft_expiry_seconds = soft_expiry_minutes * 60
        self.hard_expiry_seconds = hard_expiry_minutes * 60

    @classmethod
    def from_config(cls):
        """Create ThreadService from frontal-cortex config."""
        try:
            config = ConfigService.resolve_agent_config("frontal-cortex")
            thread_config = config.get("thread", {})
            return cls(
                soft_expiry_minutes=thread_config.get("soft_expiry_minutes", 30),
                hard_expiry_minutes=thread_config.get("hard_expiry_minutes", 240),
            )
        except Exception:
            return cls()

    def resolve_thread(self, user_id: str, channel_id: str, platform: str = "unknown") -> ThreadResolution:
        """
        Resolve the active thread for a user+channel pair.

        Algorithm:
        1. Check for active thread pointer
        2. If exists, check last_activity against expiry thresholds
        3. If expired or missing, create new thread

        Returns:
            ThreadResolution with thread state information
        """
        pointer_key = f"active_thread:{user_id}:{channel_id}"
        active_thread_id = self.redis.get(pointer_key)

        if active_thread_id:
            thread_data = self._get_thread_hash(active_thread_id)
            if thread_data and thread_data.get("state") != "expired":
                last_activity = float(thread_data.get("last_activity", 0))
                gap_seconds = time.time() - last_activity

                if gap_seconds < self.soft_expiry_seconds:
                    # Seamless continuation
                    self._update_activity(active_thread_id)
                    return ThreadResolution(
                        thread_id=active_thread_id,
                        is_new=False,
                        is_resumed=False,
                    )

                if gap_seconds < self.hard_expiry_seconds:
                    # Soft resume
                    self._update_activity(active_thread_id)
                    resume_count = int(thread_data.get("resume_count", 0)) + 1
                    self.redis.hset(f"thread:{active_thread_id}", mapping={
                        "resume_count": str(resume_count),
                        "last_resume_at": str(time.time()),
                    })
                    return ThreadResolution(
                        thread_id=active_thread_id,
                        is_new=False,
                        is_resumed=True,
                        resume_gap_minutes=gap_seconds / 60.0,
                    )

                # Hard expiry — expire old thread and create new
                previous_thread_id = active_thread_id
                recent_context = self._get_recent_visible_context(active_thread_id)
                self.expire_thread(active_thread_id)
                new_resolution = self._create_new_thread(user_id, channel_id, platform)
                new_resolution.previous_thread_id = previous_thread_id
                new_resolution.recent_visible_context = recent_context
                return new_resolution

        # No active thread — create new
        return self._create_new_thread(user_id, channel_id, platform)

    def _create_new_thread(self, user_id: str, channel_id: str, platform: str) -> ThreadResolution:
        """Create a new thread with SETNX race condition protection."""
        seq_key = f"thread_seq:{user_id}:{channel_id}"
        sequence = self.redis.incr(seq_key)

        thread_id = f"{platform}:{user_id}:{channel_id}:{sequence}"
        pointer_key = f"active_thread:{user_id}:{channel_id}"

        # SETNX to prevent duplicate thread creation from concurrent messages
        was_set = self.redis.setnx(pointer_key, thread_id)
        if not was_set:
            # Another message already created a thread — use that one
            existing_thread_id = self.redis.get(pointer_key)
            if existing_thread_id:
                self._update_activity(existing_thread_id)
                return ThreadResolution(
                    thread_id=existing_thread_id,
                    is_new=False,
                    is_resumed=False,
                )

        # Set pointer TTL
        self.redis.expire(pointer_key, 86400)  # 24h

        # Create thread hash
        now = str(time.time())
        self.redis.hset(f"thread:{thread_id}", mapping={
            "state": "active",
            "user_id": user_id,
            "channel_id": channel_id,
            "platform": platform,
            "current_topic": "",
            "topic_history": "[]",
            "created_at": now,
            "last_activity": now,
            "last_exchange_at": "",
            "exchange_count": "0",
            "resume_count": "0",
            "last_resume_at": "",
        })
        self.redis.expire(f"thread:{thread_id}", 86400)

        # Write to PostgreSQL for durable tracking
        self._persist_thread_created(thread_id, user_id, channel_id, platform)

        logging.info(f"[THREAD] Created new thread: {thread_id}")
        return ThreadResolution(
            thread_id=thread_id,
            is_new=True,
            is_resumed=False,
        )

    def _update_activity(self, thread_id: str):
        """Update last_activity timestamp and refresh TTLs."""
        now = str(time.time())
        pipe = self.redis.pipeline()
        pipe.hset(f"thread:{thread_id}", "last_activity", now)
        pipe.expire(f"thread:{thread_id}", 86400)
        pipe.execute()

        # Also refresh the pointer TTL
        thread_data = self._get_thread_hash(thread_id)
        if thread_data:
            user_id = thread_data.get("user_id", "")
            channel_id = thread_data.get("channel_id", "")
            pointer_key = f"active_thread:{user_id}:{channel_id}"
            self.redis.expire(pointer_key, 86400)

    def update_activity(self, thread_id: str, topic: str = None):
        """Public: update thread activity and optionally set current topic."""
        self._update_activity(thread_id)
        if topic:
            self.update_topic(thread_id, topic)

    def update_topic(self, thread_id: str, topic: str):
        """Update current topic and append to topic history if new."""
        thread_key = f"thread:{thread_id}"
        current = self.redis.hget(thread_key, "current_topic")

        if current != topic:
            # Append to topic history
            history_raw = self.redis.hget(thread_key, "topic_history") or "[]"
            try:
                history = json.loads(history_raw)
            except (json.JSONDecodeError, TypeError):
                history = []

            if topic not in history:
                history.append(topic)

            self.redis.hset(thread_key, mapping={
                "current_topic": topic,
                "topic_history": json.dumps(history),
            })

    def increment_exchange_count(self, thread_id: str):
        """Increment exchange count and update last_exchange_at."""
        pipe = self.redis.pipeline()
        pipe.hincrby(f"thread:{thread_id}", "exchange_count", 1)
        pipe.hset(f"thread:{thread_id}", "last_exchange_at", str(time.time()))
        pipe.execute()

    def expire_thread(self, thread_id: str):
        """Expire a thread — mark as expired, clear pointer."""
        thread_key = f"thread:{thread_id}"
        thread_data = self._get_thread_hash(thread_id)

        if not thread_data or thread_data.get("state") == "expired":
            return  # Already expired or doesn't exist

        # Mark as expired
        self.redis.hset(thread_key, mapping={
            "state": "expired",
            "expired_at": str(time.time()),
        })

        # Clear the active pointer
        user_id = thread_data.get("user_id", "")
        channel_id = thread_data.get("channel_id", "")
        pointer_key = f"active_thread:{user_id}:{channel_id}"

        # Only delete pointer if it still points to this thread
        current_pointer = self.redis.get(pointer_key)
        if current_pointer == thread_id:
            self.redis.delete(pointer_key)

        # Persist expiry to PostgreSQL
        self._persist_thread_expired(thread_id, thread_data)

        logging.info(f"[THREAD] Expired thread: {thread_id}")

    def get_thread(self, thread_id: str) -> Optional[dict]:
        """Get full thread data from Redis hash."""
        return self._get_thread_hash(thread_id)

    def get_active_thread_id(self, user_id: str, channel_id: str) -> Optional[str]:
        """Get the active thread ID for a user+channel pair."""
        pointer_key = f"active_thread:{user_id}:{channel_id}"
        return self.redis.get(pointer_key)

    def _get_thread_hash(self, thread_id: str) -> Optional[dict]:
        """Read thread hash from Redis."""
        data = self.redis.hgetall(f"thread:{thread_id}")
        return data if data else None

    def _get_recent_visible_context(self, thread_id: str) -> Optional[List[dict]]:
        """
        Get last 1-2 exchanges from an expiring thread for visual continuity.

        Returns list of exchange dicts or None if unavailable.
        """
        try:
            conv_key = f"thread_conv:{thread_id}"
            # Get last 2 exchanges
            raw_exchanges = self.redis.lrange(conv_key, -2, -1)
            if not raw_exchanges:
                return None

            exchanges = []
            for raw in raw_exchanges:
                try:
                    exchange = json.loads(raw)
                    # Only include exchanges that have both prompt and response
                    if exchange.get("prompt") and exchange.get("response"):
                        exchanges.append({
                            "prompt": exchange["prompt"].get("message", ""),
                            "response": exchange["response"].get("message", ""),
                        })
                except (json.JSONDecodeError, TypeError):
                    continue

            return exchanges if exchanges else None
        except Exception as e:
            logging.debug(f"[THREAD] Failed to get recent context for {thread_id}: {e}")
            return None

    def _persist_thread_created(self, thread_id: str, user_id: str, channel_id: str, platform: str):
        """Write thread creation to PostgreSQL."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO threads (thread_id, user_id, channel_id, platform, state)
                    VALUES (%s, %s, %s, %s, 'active')
                    ON CONFLICT (thread_id) DO NOTHING
                """, (thread_id, user_id, channel_id, platform))
                cursor.close()
        except Exception as e:
            logging.debug(f"[THREAD] PostgreSQL persist failed (non-critical): {e}")

    def _persist_thread_expired(self, thread_id: str, thread_data: dict):
        """Update thread record in PostgreSQL on expiry."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE threads SET
                        state = 'expired',
                        current_topic = %s,
                        topic_history = %s,
                        exchange_count = %s,
                        expired_at = NOW()
                    WHERE thread_id = %s
                """, (
                    thread_data.get("current_topic", ""),
                    thread_data.get("topic_history", "[]"),
                    int(thread_data.get("exchange_count", 0)),
                    thread_id,
                ))
                cursor.close()
        except Exception as e:
            logging.debug(f"[THREAD] PostgreSQL expire persist failed (non-critical): {e}")


# Singleton accessor
_thread_service = None


def get_thread_service() -> ThreadService:
    """Get or create global ThreadService instance."""
    global _thread_service
    if _thread_service is None:
        _thread_service = ThreadService.from_config()
    return _thread_service
