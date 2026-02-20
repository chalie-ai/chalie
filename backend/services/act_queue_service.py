import json
import logging
from typing import Dict, Any

from .redis_client import RedisClientService
from .config_service import ConfigService


logger = logging.getLogger(__name__)


class ActQueueService:
    """Service for enqueuing output messages to the act-queue."""

    def __init__(self):
        """Initialize the ActQueueService with Redis connection and config."""
        self._redis = RedisClientService.create_connection()
        self._config = ConfigService.connections()

        # Get queue name from config or use default
        queues_config = self._config.get("redis", {}).get("queues", {})
        self._queue_name = queues_config.get("act_queue", {}).get("name", "act-queue")

        logger.info(f"ActQueueService initialized with queue: {self._queue_name}")

    def enqueue_output(self, message: Dict[str, Any]) -> None:
        """
        Enqueue an output message to the act-queue.

        Args:
            message: Message dict created by ActQueueMessage.create_output_message()
        """
        try:
            serialized = json.dumps(message)
            self._redis.rpush(self._queue_name, serialized)

            logger.info(
                f"Enqueued message to {self._queue_name}: "
                f"type={message.get('type')}, "
                f"destination={message.get('destination')}, "
                f"topic={message.get('topic')}"
            )
        except Exception as e:
            logger.error(f"Failed to enqueue message to {self._queue_name}: {e}")
            raise
