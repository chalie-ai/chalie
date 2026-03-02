"""
Redis Client Service — now backed by in-memory MemoryStore.

All callers that use RedisClientService.create_connection() get a shared
MemoryStore singleton. This propagates to all 85+ files with zero per-file changes.
"""

import json
from pathlib import Path
from datetime import datetime

from .config_service import ConfigService

# Shared singleton MemoryStore instance
_store = None
_store_lock = None

def _get_store():
    """Get or create the shared MemoryStore singleton (thread-safe)."""
    global _store, _store_lock
    import threading
    if _store_lock is None:
        _store_lock = threading.Lock()
    if _store is None:
        with _store_lock:
            if _store is None:
                from .memory_store import MemoryStore
                _store = MemoryStore()
    return _store


class RedisClientService:

    def __init__(self):
        self._config = ConfigService.connections()
        self._client = None

    @staticmethod
    def create_connection(decode_responses=True):
        """Return the shared MemoryStore instance.

        Args:
            decode_responses: Ignored (MemoryStore always returns strings)

        Returns:
            MemoryStore: Thread-safe in-memory store with Redis-compatible API
        """
        return _get_store()

    def get_topic(self, key: str) -> str:
        """Return the topic name for a given key.

        Parameters
        ----------
        key: str
            The key name defined in the config under `topics`.
        """
        topics = self._config.get("redis", {}).get("topics", {})
        return topics.get(key, key)
