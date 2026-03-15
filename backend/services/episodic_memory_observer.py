"""
Episodic Memory Observer — periodic thread scanner for episodic consolidation.

Scans active threads every 60 seconds and triggers episodic consolidation when
a thread has accumulated enough raw conversation turns or has gone idle.

Design:
  - 60s scan cycle, ~200ms budget per scan
  - Consolidation requires >= 5 raw turns OR >= 10min idle with any turns present
  - thread_busy:{thread_id} check prevents trimming mid-response
  - last_consolidation:{thread_id} dedup prevents double-fire within 1h
  - Richness gate: skips scan on cold systems (richness < 0.05)
"""

import json
import logging
import time

from services.memory_client import MemoryClientService

logger = logging.getLogger(__name__)
LOG_PREFIX = "[EPISODIC OBSERVER]"

# Scan & trigger constants
SCAN_INTERVAL = 60            # seconds between scans
MIN_TURNS = 5                 # minimum raw turns to trigger consolidation
IDLE_TIMEOUT = 600            # 10 minutes — data loss prevention fallback
COOLDOWN_KEY_TTL = 3600       # 1 hour — dedup window
SCAN_BUDGET_MS = 200          # max scan time per cycle (ms)
RICHNESS_GATE = 0.05          # skip scan on cold systems


class EpisodicMemoryObserver:
    """Periodic thread scanner that triggers episodic consolidation based on turn count."""

    def __init__(self, scan_interval=SCAN_INTERVAL):
        """Initialize the episodic memory observer.

        Args:
            scan_interval: Seconds between scan cycles (default: ``SCAN_INTERVAL`` = 60).
        """
        self.scan_interval = scan_interval
        self.store = MemoryClientService.create_connection()

    def run(self, shared_state=None):
        """Main service loop — 60s scan cycle."""
        logger.info(f"{LOG_PREFIX} Started (interval={self.scan_interval}s, "
                     f"min_turns={MIN_TURNS})")

        while True:
            try:
                # Richness gate — skip on cold systems
                try:
                    from services.self_model_service import SelfModelService
                    richness = SelfModelService().get_memory_richness()
                    if richness < RICHNESS_GATE:
                        logger.debug(f"{LOG_PREFIX} Richness {richness:.3f} below {RICHNESS_GATE}, skipping scan")
                        time.sleep(self.scan_interval)
                        continue
                except Exception:
                    pass  # Fail-open: scan anyway if telemetry unavailable

                self._scan_threads()
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Scan cycle error: {e}")

            time.sleep(self.scan_interval)

    def _scan_threads(self):
        """Scan thread_conv:* keys, check turn count, trigger if ready."""
        scan_start = time.time()
        scanned = 0
        triggered = 0

        cursor = 0
        while True:
            # Budget check — break early if scan takes too long
            elapsed_ms = (time.time() - scan_start) * 1000
            if elapsed_ms > SCAN_BUDGET_MS:
                logger.warning(f"{LOG_PREFIX} Scan budget exceeded ({elapsed_ms:.0f}ms > {SCAN_BUDGET_MS}ms), "
                               f"remaining threads deferred to next cycle")
                break

            cursor, keys = self.store.scan(cursor, match="thread_conv:*", count=50)

            for key in keys:
                try:
                    thread_id = key.split(":", 1)[1] if ":" in key else key
                    scanned += 1

                    # Skip if on cooldown (already consolidated recently)
                    if self._is_on_cooldown(thread_id):
                        continue

                    # Skip if thread is busy (digest worker mid-response)
                    if self._is_thread_busy(thread_id):
                        continue

                    # Read thread metadata
                    thread_data = self.store.hgetall(f"thread:{thread_id}")
                    if not thread_data:
                        continue

                    # Skip expired threads — thread_expiry handles those
                    if thread_data.get("state") == "expired":
                        continue

                    topic = thread_data.get("current_topic", "general")

                    # Read raw exchange list and count turns
                    raw_exchanges = self.store.lrange(key, 0, -1)
                    exchanges = []
                    for raw in raw_exchanges:
                        try:
                            exchanges.append(json.loads(raw))
                        except (json.JSONDecodeError, TypeError):
                            continue

                    if not exchanges:
                        continue

                    # Trigger condition 1: turn count threshold
                    if self.should_consolidate(len(exchanges)):
                        self._trigger_consolidation(thread_id, topic, exchanges)
                        triggered += 1
                        continue

                    # Trigger condition 2: idle timeout (data loss prevention)
                    last_activity = float(thread_data.get("last_activity", 0))
                    idle_seconds = time.time() - last_activity if last_activity > 0 else 0
                    if idle_seconds >= IDLE_TIMEOUT and len(exchanges) > 0:
                        self._trigger_consolidation(thread_id, topic, exchanges)
                        triggered += 1

                except Exception as e:
                    logger.debug(f"{LOG_PREFIX} Error scanning thread: {e}")
                    continue

            if cursor == 0:
                break

        if scanned > 0 or triggered > 0:
            logger.info(f"{LOG_PREFIX} Scan complete: {scanned} threads scanned, "
                        f"{triggered} consolidations triggered")

    def should_consolidate(self, turn_count: int) -> bool:
        """Return True when the thread has accumulated enough turns for consolidation.

        Args:
            turn_count: Number of raw conversation exchanges in the thread.

        Returns:
            ``True`` if ``turn_count >= MIN_TURNS``.
        """
        return turn_count >= MIN_TURNS

    def _trigger_consolidation(self, thread_id: str, topic: str, exchanges: list):
        """Trigger episodic memory consolidation for a thread via PromptQueue.

        Enqueues a consolidation job and sets a cooldown key to prevent double-firing
        within the same hour.

        Args:
            thread_id: Conversation thread identifier.
            topic: Current topic string for the consolidation job payload.
            exchanges: List of exchange dicts (used for logging only).
        """
        try:
            from workers.episodic_memory_worker import episodic_memory_worker
            from services.prompt_queue import PromptQueue

            job_data = {
                'topic': topic,
                'thread_id': thread_id,
            }

            queue = PromptQueue(
                queue_name="episodic-memory-queue",
                worker_func=episodic_memory_worker,
            )
            queue.enqueue(job_data)

            # Set cooldown (timestamp value for future time-based logic)
            self.store.setex(
                f"last_consolidation:{thread_id}",
                COOLDOWN_KEY_TTL,
                str(time.time()),
            )

            logger.info(f"{LOG_PREFIX} Triggered consolidation for thread '{thread_id}' "
                        f"(topic='{topic}', turns={len(exchanges)})")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Consolidation trigger failed: {e}")

    def _is_on_cooldown(self, thread_id: str) -> bool:
        """Check whether a thread is within the post-consolidation cooldown window.

        Args:
            thread_id: Conversation thread identifier to check.

        Returns:
            ``True`` if a consolidation cooldown key exists for the thread.
        """
        return self.store.get(f"last_consolidation:{thread_id}") is not None

    def _is_thread_busy(self, thread_id: str) -> bool:
        """Check whether a digest worker is actively processing this thread.

        Args:
            thread_id: Conversation thread identifier to check.

        Returns:
            ``True`` if a ``thread_busy`` key exists in the MemoryStore for the thread.
        """
        return self.store.get(f"thread_busy:{thread_id}") is not None


def episodic_memory_observer_worker(shared_state=None):
    """Entry point for thread spawn — instantiates and runs EpisodicMemoryObserver.

    Args:
        shared_state: Optional shared state dict passed by the consumer thread
            harness.  Currently unused but accepted for interface compatibility.
    """
    observer = EpisodicMemoryObserver()
    observer.run(shared_state)
