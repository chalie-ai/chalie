"""
Memory Client Service — backward-compatible shim.

Delegates to ``memory_store.get_shared_store()`` which owns the singleton.
Kept so existing ``from services.memory_client import MemoryClientService``
imports continue to work without touching 100+ call sites.
"""

from .memory_store import get_shared_store


class MemoryClientService:
    """Thin facade that delegates to the shared ``MemoryStore`` singleton."""

    @staticmethod
    def create_connection(_decode_responses=True):
        """Return the shared MemoryStore instance.

        Args:
            decode_responses: Ignored (MemoryStore always returns strings).

        Returns:
            MemoryStore: Thread-safe in-memory store.
        """
        return get_shared_store()
