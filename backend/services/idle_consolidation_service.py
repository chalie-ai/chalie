import time
import logging
from typing import Optional

from .redis_client import RedisClientService
from .config_service import ConfigService
from .semantic_consolidation_tracker import SemanticConsolidationTracker

try:
    from rq import Queue
except ImportError:
    Queue = None


logger = logging.getLogger(__name__)


class IdleConsolidationService:
    """
    Background service that triggers batch consolidation during idle periods.

    Monitors memory worker queues and triggers consolidation when:
    - All memory queues are empty (idle state)
    - 1 hour has passed since last consolidation
    """

    def __init__(self, check_interval: int = 300):
        """
        Initialize the idle consolidation service.

        Args:
            check_interval: Seconds between idle checks (default: 300 = 5 minutes)
        """
        self.redis = RedisClientService.create_connection()
        self.tracker = SemanticConsolidationTracker()
        self.check_interval = check_interval

        # Load queue names from config
        config = ConfigService.connections()
        topics = config.get("redis", {}).get("topics", {})

        self.prompt_queue = topics.get("prompt_queue", "prompt-queue")
        self.memory_queue = topics.get("memory_chunker", "memory-chunker-queue")
        self.episodic_queue = topics.get("episodic_memory", "episodic-memory-queue")
        self.semantic_queue = config.get("redis", {}).get("queues", {}).get(
            "semantic_consolidation_queue", {}
        ).get("name", "semantic_consolidation_queue")

        logger.info(
            f"[IDLE CONSOLIDATION] Service initialized "
            f"(check_interval={check_interval}s, idle_threshold=900s)"
        )

    def run(self, shared_state: Optional[dict] = None) -> None:
        """
        Main service loop - checks for idle state and triggers consolidation.

        Args:
            shared_state: Optional shared state dict (for consumer integration)
        """
        logger.info("[IDLE CONSOLIDATION] Service started")

        while True:
            try:
                # Sleep for check interval
                time.sleep(self.check_interval)

                # Check if workers are idle
                if not self._are_workers_idle():
                    logger.debug("[IDLE CONSOLIDATION] Workers not idle, skipping")
                    continue

                # Check if 1 hour has passed since last consolidation
                state = self.tracker.get_state()
                time_since_last = time.time() - state['last_consolidation_time']

                if time_since_last >= 900:  # 15 minutes (fallback; primary trigger is episode-count)
                    logger.info(
                        f"[IDLE CONSOLIDATION] System idle for {time_since_last/60:.1f} minutes, "
                        f"triggering batch consolidation"
                    )
                    self._trigger_consolidation()
                else:
                    logger.debug(
                        f"[IDLE CONSOLIDATION] Idle but only {time_since_last/60:.1f} minutes "
                        f"since last consolidation (need 15 min)"
                    )

            except KeyboardInterrupt:
                logger.info("[IDLE CONSOLIDATION] Service shutting down...")
                break
            except Exception as e:
                logger.error(f"[IDLE CONSOLIDATION] Error: {e}", exc_info=True)
                logger.info("[IDLE CONSOLIDATION] Waiting 1 minute before retry...")
                time.sleep(60)

    def _are_workers_idle(self) -> bool:
        """
        Check if all memory worker queues are empty.

        Returns:
            bool: True if all queues are empty (idle), False otherwise
        """
        queues = [
            self.prompt_queue,
            self.memory_queue,
            self.episodic_queue
        ]

        for queue_name in queues:
            queue_length = self.redis.llen(queue_name)
            if queue_length > 0:
                logger.debug(
                    f"[IDLE CONSOLIDATION] Queue '{queue_name}' has {queue_length} items, "
                    f"not idle"
                )
                return False

        return True

    def _trigger_consolidation(self) -> None:
        """Trigger batch consolidation by enqueuing a batch job."""
        if Queue is None:
            logger.error("[IDLE CONSOLIDATION] RQ not available, cannot enqueue batch job")
            return

        try:
            redis_connection = RedisClientService.create_connection(decode_responses=False)
            queue = Queue(self.semantic_queue, connection=redis_connection)

            batch_job = {
                "type": "batch_consolidation",
                "trigger": "idle_sleep",
                "timestamp": time.time()
            }

            queue.enqueue(
                'workers.semantic_consolidation_worker.semantic_consolidation_worker',
                batch_job
            )

            logger.info("[IDLE CONSOLIDATION] Enqueued batch consolidation job")

            # Reset tracker after successful enqueue
            self.tracker.reset_episode_count()

        except Exception as e:
            logger.error(f"[IDLE CONSOLIDATION] Failed to enqueue batch job: {e}", exc_info=True)


def idle_consolidation_process(shared_state):
    """Top-level entry point for multiprocessing spawn.

    Creates the service instance inside the child process to avoid
    pickling Redis connections (which contain _thread.lock objects).
    """
    service = IdleConsolidationService()
    service.run(shared_state)
