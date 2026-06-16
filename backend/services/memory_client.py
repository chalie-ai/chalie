"""Memory Client Service — backward-compatible shim.

Delegates to ``memory_store.get_shared_store()`` which owns the singleton.
Kept so existing ``from services.memory_client import MemoryClientService``
imports continue to work without touching 100+ call sites.
"""

from .memory_store import get_shared_store


class MemoryClientService:
    """Thin facade that delegates to the shared ``MemoryStore`` singleton."""

    @staticmethod
    def create_connection(_decode_responses=True):
        """``_decode_responses`` is ignored (MemoryStore always returns strings)."""
        return get_shared_store()
