import json
from pathlib import Path
from datetime import datetime

try:
    import redis  # type: ignore
except ImportError:
    # If redis package is not available, raise a clear error when used.
    redis = None  # type: ignore

from .config_service import ConfigService

class RedisClientService:

    def __init__(self):
        self._config = ConfigService.connections()
        self._client = None

    @staticmethod
    def create_connection(decode_responses=True):
        """Create a new Redis connection using resolved hostname from ConfigService.

        Args:
            decode_responses: Whether to decode responses to strings (default: True)

        Returns:
            redis.Redis: A new Redis connection instance
        """
        if redis is None:
            raise ImportError("redis package is not installed. Please pip install redis")

        redis_conf = ConfigService.connections().get("redis", {})
        hostname = redis_conf.get("host", "localhost")
        port = redis_conf.get("port", 6379)

        # Get pre-resolved hostname from ConfigService (avoids DNS lookups in child processes)
        resolved_host = ConfigService.get_resolved_host(hostname)

        return redis.Redis(host=resolved_host, port=port, decode_responses=decode_responses)

    def get_topic(self, key: str) -> str:
        """Return the topic name for a given key.

        Parameters
        ----------
        key: str
            The key name defined in the config under `topics`.
        """
        topics = self._config.get("redis", {}).get("topics", {})
        return topics.get(key, key)
