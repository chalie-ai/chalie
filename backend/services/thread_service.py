"""
Thread Service - Thread resolution and lifecycle management.

Provides channel-scoped conversation threads with temporal lifecycle
(soft/hard expiry). Single-user system — no user_id scoping.

Thread ID format: {platform}:{channel_id}:{sequence}
"""

import json
import time
import logging
from dataclasses import dataclass
from typing import Optional, List

from services.memory_client import MemoryClientService
from services.config_service import ConfigService


@dataclass
class ThreadResolution:
    """Outcome of a thread-resolution request for a channel.

    Attributes:
        thread_id: The resolved thread identifier in the format
            ``{platform}:{channel_id}:{sequence}``.
        is_new: ``True`` when a brand-new thread was created for this channel.
        is_resumed: ``True`` when an existing thread was continued after a gap
            that exceeded the soft-expiry threshold but not the hard-expiry.
        resume_gap_minutes: How many minutes elapsed since the last activity,
            populated only when *is_resumed* is ``True``.
        previous_thread_id: The thread ID that was hard-expired to make room
            for the new thread, if applicable.
        recent_visible_context: Last 1–2 exchanges from the expired thread,
            provided for visual continuity in the UI after a hard expiry.
    """

    thread_id: str
    is_new: bool
    is_resumed: bool
    resume_gap_minutes: float = 0.0
    previous_thread_id: Optional[str] = None
    recent_visible_context: Optional[List[dict]] = None


class ThreadService:
    """Manages thread resolution and lifecycle."""

    def __init__(self, soft_expiry_minutes: int = 30, hard_expiry_minutes: int = 240):
        """Initialise the service with expiry thresholds and a MemoryStore connection.

        Args:
            soft_expiry_minutes: Minutes of inactivity after which a thread is
                considered *resumed* (gap acknowledged) rather than seamlessly
                continued.  Defaults to 30 minutes.
            hard_expiry_minutes: Minutes of inactivity after which a thread is
                fully expired and a new thread is created.  Must be greater than
                *soft_expiry_minutes*.  Defaults to 240 minutes (4 hours).
        """
        self.store = MemoryClientService.create_connection()
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

    def resolve_thread(self, channel_id: str, platform: str = "unknown") -> ThreadResolution:
        """
        Resolve the active thread for a channel.

        Algorithm:
        1. Check for active thread pointer
        2. If exists, check last_activity against expiry thresholds
        3. If expired or missing, create new thread

        Returns:
            ThreadResolution with thread state information
        """
        pointer_key = f"active_thread:{channel_id}"
        active_thread_id = self.store.get(pointer_key)

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
                    self.store.hset(f"thread:{active_thread_id}", mapping={
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
                new_resolution = self._create_new_thread(channel_id, platform)
                new_resolution.previous_thread_id = previous_thread_id
                new_resolution.recent_visible_context = recent_context
                return new_resolution

        # No active thread in MemoryStore — check SQLite (post-restart recovery)
        recovered = self._recover_thread_from_sqlite(channel_id, platform)
        if recovered:
            return recovered

        # Truly new conversation — create fresh thread
        return self._create_new_thread(channel_id, platform)

    def _create_new_thread(self, channel_id: str, platform: str) -> ThreadResolution:
        """Create a new thread with SETNX race condition protection."""
        seq_key = f"thread_seq:{channel_id}"
        sequence = self.store.incr(seq_key)

        thread_id = f"{platform}:{channel_id}:{sequence}"
        pointer_key = f"active_thread:{channel_id}"

        # SETNX to prevent duplicate thread creation from concurrent messages
        was_set = self.store.setnx(pointer_key, thread_id)
        if not was_set:
            # Another message already created a thread — use that one
            existing_thread_id = self.store.get(pointer_key)
            if existing_thread_id:
                self._update_activity(existing_thread_id)
                return ThreadResolution(
                    thread_id=existing_thread_id,
                    is_new=False,
                    is_resumed=False,
                )

        # Set pointer TTL
        self.store.expire(pointer_key, 86400)  # 24h

        # Create thread hash
        now = str(time.time())
        self.store.hset(f"thread:{thread_id}", mapping={
            "state": "active",
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
        self.store.expire(f"thread:{thread_id}", 86400)

        # Write to SQLite for durable tracking
        self._persist_thread_created(thread_id, channel_id, platform)

        logging.info(f"[THREAD] Created new thread: {thread_id}")
        return ThreadResolution(
            thread_id=thread_id,
            is_new=True,
            is_resumed=False,
        )

    def _update_activity(self, thread_id: str):
        """Update last_activity timestamp and refresh TTLs."""
        now = str(time.time())
        pipe = self.store.pipeline()
        pipe.hset(f"thread:{thread_id}", "last_activity", now)
        pipe.expire(f"thread:{thread_id}", 86400)
        pipe.execute()

        # Also refresh the pointer TTL
        thread_data = self._get_thread_hash(thread_id)
        if thread_data:
            channel_id = thread_data.get("channel_id", "")
            pointer_key = f"active_thread:{channel_id}"
            self.store.expire(pointer_key, 86400)

        # Write-through to SQLite so last_activity survives restarts
        self._persist_activity(thread_id)

    def _persist_activity(self, thread_id: str):
        """Write-through: update last_activity in SQLite."""
        try:
            from services.database_service import get_shared_db_service
            from services.time_utils import utc_now
            db = get_shared_db_service()
            db.execute(
                "UPDATE threads SET last_activity = ? WHERE thread_id = ?",
                (utc_now().isoformat(), thread_id)
            )
        except Exception:
            pass

    def update_activity(self, thread_id: str, topic: str = None):
        """Public: update thread activity and optionally set current topic."""
        self._update_activity(thread_id)
        if topic:
            self.update_topic(thread_id, topic)

    def update_topic(self, thread_id: str, topic: str):
        """Update current topic and append to topic history if new."""
        thread_key = f"thread:{thread_id}"
        current = self.store.hget(thread_key, "current_topic")

        if current != topic:
            # Append to topic history
            history_raw = self.store.hget(thread_key, "topic_history") or "[]"
            try:
                history = json.loads(history_raw)
            except (json.JSONDecodeError, TypeError):
                history = []

            if topic not in history:
                history.append(topic)

            self.store.hset(thread_key, mapping={
                "current_topic": topic,
                "topic_history": json.dumps(history),
            })

    def increment_exchange_count(self, thread_id: str):
        """Increment exchange count and update last_exchange_at."""
        pipe = self.store.pipeline()
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
        self.store.hset(thread_key, mapping={
            "state": "expired",
            "expired_at": str(time.time()),
        })

        # Clear the active pointer
        channel_id = thread_data.get("channel_id", "")
        pointer_key = f"active_thread:{channel_id}"

        # Only delete pointer if it still points to this thread
        current_pointer = self.store.get(pointer_key)
        if current_pointer == thread_id:
            self.store.delete(pointer_key)

        # Persist expiry to SQLite
        self._persist_thread_expired(thread_id, thread_data)

        logging.info(f"[THREAD] Expired thread: {thread_id}")

    def get_thread(self, thread_id: str) -> Optional[dict]:
        """Get full thread data from MemoryStore hash."""
        return self._get_thread_hash(thread_id)

    def get_active_thread_id(self, channel_id: str) -> Optional[str]:
        """Get the active thread ID for a channel."""
        pointer_key = f"active_thread:{channel_id}"
        return self.store.get(pointer_key)

    def _get_thread_hash(self, thread_id: str) -> Optional[dict]:
        """Read thread hash from MemoryStore."""
        data = self.store.hgetall(f"thread:{thread_id}")
        return data if data else None

    def _get_recent_visible_context(self, thread_id: str) -> Optional[List[dict]]:
        """
        Get last 1-2 exchanges from an expiring thread for visual continuity.

        Returns list of exchange dicts or None if unavailable.
        """
        try:
            conv_key = f"thread_conv:{thread_id}"
            # Get last 2 exchanges
            raw_exchanges = self.store.lrange(conv_key, -2, -1)
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

    def _recover_thread_from_sqlite(self, channel_id: str, platform: str) -> Optional[ThreadResolution]:
        """After restart, recover the last active thread from SQLite and rehydrate MemoryStore.

        Checks thread_exchanges for actual last activity (more reliable than
        threads.last_activity which may be stale). Applies the same soft/hard
        expiry logic as the MemoryStore path.
        """
        try:
            from services.database_service import get_shared_db_service
            from services.time_utils import parse_utc, utc_now
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()

                # Find most recent active thread for this channel
                cursor.execute("""
                    SELECT thread_id, current_topic, topic_history, exchange_count
                    FROM threads
                    WHERE channel_id = ? AND state = 'active'
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (channel_id,))
                row = cursor.fetchone()
                if not row:
                    # No active thread — still restore seq counter to prevent
                    # _create_new_thread from reusing old thread IDs
                    self._restore_seq_counter(channel_id, conn)
                    cursor.close()
                    return None

                thread_id = row[0]
                current_topic = row[1] or ""
                topic_history = row[2] or "[]"
                exchange_count = row[3] or 0

                # Get actual last activity from exchanges (most reliable after restart)
                cursor.execute("""
                    SELECT MAX(COALESCE(response_time, prompt_time)) AS last_time
                    FROM thread_exchanges
                    WHERE thread_id = ?
                """, (thread_id,))
                ts_row = cursor.fetchone()
                cursor.close()

                if not ts_row or not ts_row[0]:
                    # No exchanges — still restore seq counter
                    self._restore_seq_counter(channel_id, conn)
                    return None

                last_activity = parse_utc(ts_row[0])
                gap_seconds = (utc_now() - last_activity).total_seconds()

                if gap_seconds >= self.hard_expiry_seconds:
                    # Thread too old — expire in SQLite and let caller create new
                    db.execute(
                        "UPDATE threads SET state = 'expired', expired_at = ? WHERE thread_id = ?",
                        (utc_now().isoformat(), thread_id)
                    )
                    # Restore sequence counter even on expiry to prevent
                    # _create_new_thread from reusing old thread IDs
                    self._restore_seq_counter(channel_id, conn)
                    return None

                # Rehydrate MemoryStore — restore pointer, thread hash, and sequence counter
                now = str(time.time())
                created_ts = str(last_activity.timestamp())

                pointer_key = f"active_thread:{channel_id}"
                self.store.set(pointer_key, thread_id)
                self.store.expire(pointer_key, 86400)

                self.store.hset(f"thread:{thread_id}", mapping={
                    "state": "active",
                    "channel_id": channel_id,
                    "platform": platform,
                    "current_topic": current_topic,
                    "topic_history": topic_history,
                    "created_at": created_ts,
                    "last_activity": now,
                    "last_exchange_at": now,
                    "exchange_count": str(exchange_count),
                    "resume_count": "0",
                    "last_resume_at": "",
                })
                self.store.expire(f"thread:{thread_id}", 86400)

                # Restore sequence counter to avoid thread_id collisions
                parts = thread_id.rsplit(":", 1)
                if len(parts) == 2:
                    try:
                        seq = int(parts[1])
                        seq_key = f"thread_seq:{channel_id}"
                        current_seq = self.store.get(seq_key)
                        if not current_seq or int(current_seq) < seq:
                            self.store.set(seq_key, str(seq))
                    except ValueError:
                        pass

                is_resumed = gap_seconds >= self.soft_expiry_seconds
                logging.info(f"[THREAD] Recovered thread from SQLite: {thread_id} (gap={gap_seconds:.0f}s)")
                return ThreadResolution(
                    thread_id=thread_id,
                    is_new=False,
                    is_resumed=is_resumed,
                    resume_gap_minutes=gap_seconds / 60.0 if is_resumed else 0.0,
                )
        except Exception as e:
            logging.debug(f"[THREAD] SQLite recovery failed: {e}")
            return None

    def _restore_seq_counter(self, channel_id: str, conn=None):
        """Restore the MemoryStore sequence counter from SQLite max thread sequence.

        Prevents _create_new_thread from generating thread IDs that collide
        with existing threads after a MemoryStore reset (restart / TTL expiry).
        """
        try:
            if conn is None:
                from services.database_service import get_shared_db_service
                db = get_shared_db_service()
                conn = db.connection().__enter__()

            cursor = conn.cursor()
            cursor.execute(
                "SELECT thread_id FROM threads WHERE channel_id = ? ORDER BY created_at DESC",
                (channel_id,),
            )
            max_seq = 0
            for row in cursor.fetchall():
                parts = row[0].rsplit(":", 1)
                if len(parts) == 2:
                    try:
                        seq = int(parts[1])
                        if seq > max_seq:
                            max_seq = seq
                    except ValueError:
                        pass
            cursor.close()

            if max_seq > 0:
                seq_key = f"thread_seq:{channel_id}"
                current_seq = self.store.get(seq_key)
                if not current_seq or int(current_seq) < max_seq:
                    self.store.set(seq_key, str(max_seq))
                    logging.debug(f"[THREAD] Restored seq counter for {channel_id}: {max_seq}")
        except Exception as e:
            logging.debug(f"[THREAD] Seq counter restore failed: {e}")

    def _persist_thread_created(self, thread_id: str, channel_id: str, platform: str):
        """Write thread creation to SQLite."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO threads (thread_id, channel_id, platform, state)
                    VALUES (?, ?, ?, 'active')
                    ON CONFLICT (thread_id) DO UPDATE SET
                        state = 'active', expired_at = NULL
                """, (thread_id, channel_id, platform))
                cursor.close()
        except Exception as e:
            logging.error(f"[THREAD] SQLite thread persist FAILED for {thread_id}: {e}", exc_info=True)

    def _persist_thread_expired(self, thread_id: str, thread_data: dict):
        """Update thread record in SQLite on expiry."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE threads SET
                        state = 'expired',
                        current_topic = ?,
                        topic_history = ?,
                        exchange_count = ?,
                        expired_at = datetime('now')
                    WHERE thread_id = ?
                """, (
                    thread_data.get("current_topic", ""),
                    thread_data.get("topic_history", "[]"),
                    int(thread_data.get("exchange_count", 0)),
                    thread_id,
                ))
                cursor.close()
        except Exception as e:
            logging.error(f"[THREAD] SQLite expire persist FAILED for {thread_id}: {e}", exc_info=True)


# Singleton accessor
_thread_service = None


def get_thread_service() -> ThreadService:
    """Get or create global ThreadService instance."""
    global _thread_service
    if _thread_service is None:
        _thread_service = ThreadService.from_config()
    return _thread_service
