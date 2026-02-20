"""
Thread Expiry Service - Background service for thread lifecycle management.

Runs on a 5-minute cycle, scanning for threads past hard expiry.
Expires them, persists to PostgreSQL, and triggers episodic summarization.
"""

import time
import logging
from typing import Optional

from .redis_client import RedisClientService
from .config_service import ConfigService

logger = logging.getLogger(__name__)


class ThreadExpiryService:
    """Background service that expires stale threads and triggers summarization."""

    def __init__(self, check_interval: int = 300, hard_expiry_seconds: int = 14400):
        """
        Initialize thread expiry service.

        Args:
            check_interval: Seconds between scan cycles (default: 300 = 5 minutes)
            hard_expiry_seconds: Hard expiry threshold in seconds (default: 14400 = 4 hours)
        """
        self.check_interval = check_interval
        self.hard_expiry_seconds = hard_expiry_seconds

        logger.info(
            f"[THREAD EXPIRY] Initialized "
            f"(interval={check_interval}s, hard_expiry={hard_expiry_seconds}s)"
        )

    def run(self, shared_state: Optional[dict] = None) -> None:
        """Main service loop."""
        logger.info("[THREAD EXPIRY] Service started")

        while True:
            try:
                time.sleep(self.check_interval)
                self._run_expiry_cycle()
            except KeyboardInterrupt:
                logger.info("[THREAD EXPIRY] Service shutting down...")
                break
            except Exception as e:
                logger.error(f"[THREAD EXPIRY] Error: {e}", exc_info=True)
                time.sleep(60)

    def _run_expiry_cycle(self):
        """Scan for and expire stale threads."""
        try:
            redis = RedisClientService.create_connection()
        except Exception as e:
            logger.debug(f"[THREAD EXPIRY] Redis unavailable: {e}")
            return

        expired_count = 0
        now = time.time()

        # Scan for active_thread:* pointer keys to find all active threads
        cursor = 0
        while True:
            cursor, keys = redis.scan(cursor, match="active_thread:*", count=100)

            for pointer_key in keys:
                try:
                    thread_id = redis.get(pointer_key)
                    if not thread_id:
                        continue

                    thread_data = redis.hgetall(f"thread:{thread_id}")
                    if not thread_data:
                        # Orphan pointer — clean up
                        redis.delete(pointer_key)
                        continue

                    if thread_data.get("state") == "expired":
                        continue

                    last_activity = float(thread_data.get("last_activity", 0))
                    gap_seconds = now - last_activity

                    if gap_seconds >= self.hard_expiry_seconds:
                        self._expire_thread(redis, thread_id, thread_data, pointer_key)
                        expired_count += 1

                except Exception as e:
                    logger.debug(f"[THREAD EXPIRY] Error checking thread: {e}")
                    continue

            if cursor == 0:
                break

        if expired_count > 0:
            logger.info(f"[THREAD EXPIRY] Cycle complete: expired {expired_count} thread(s)")

    def _expire_thread(self, redis, thread_id: str, thread_data: dict, pointer_key: str):
        """Expire a single thread and trigger episodic summarization."""
        # Mark as expired in Redis
        redis.hset(f"thread:{thread_id}", mapping={
            "state": "expired",
            "expired_at": str(time.time()),
        })

        # Clear the active pointer
        current_pointer = redis.get(pointer_key)
        if current_pointer == thread_id:
            redis.delete(pointer_key)

        # Persist to PostgreSQL
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
            logger.debug(f"[THREAD EXPIRY] PostgreSQL persist failed: {e}")

        # Trigger episodic summarization if enough exchanges
        exchange_count = int(thread_data.get("exchange_count", 0))
        if exchange_count >= 3:
            self._trigger_episodic_summarization(thread_id, thread_data)

        logger.info(
            f"[THREAD EXPIRY] Expired: {thread_id} "
            f"(exchanges={exchange_count}, "
            f"topic={thread_data.get('current_topic', '?')})"
        )

    def _trigger_episodic_summarization(self, thread_id: str, thread_data: dict):
        """Enqueue episodic memory job for the expired thread."""
        try:
            from services.prompt_queue import enqueue_episodic_memory
            topic = thread_data.get("current_topic", "general")
            enqueue_episodic_memory({
                "topic": topic,
                "thread_id": thread_id,
            })
            logger.info(f"[THREAD EXPIRY] Enqueued episodic job for thread {thread_id}")
        except Exception as e:
            logger.debug(f"[THREAD EXPIRY] Failed to enqueue episodic job: {e}")


def thread_expiry_worker(shared_state=None):
    """
    Module-level wrapper for multiprocessing.
    Instantiates the service inside the child process.
    """
    try:
        config = ConfigService.resolve_agent_config("frontal-cortex")
        thread_config = config.get("thread", {})
        hard_expiry_minutes = thread_config.get("hard_expiry_minutes", 240)
        hard_expiry_seconds = hard_expiry_minutes * 60
    except Exception:
        hard_expiry_seconds = 14400

    service = ThreadExpiryService(
        check_interval=300,
        hard_expiry_seconds=hard_expiry_seconds,
    )
    service.run(shared_state)
