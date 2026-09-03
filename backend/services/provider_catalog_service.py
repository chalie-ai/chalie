"""Provider catalog — the setup wizard's list of platforms, derived from the registry.

Every entry is a real Chalie platform, and the provider is created through the
ordinary create path with no catalog special-casing. Nothing is listed here by
hand: the name, the pre-filled host and whether a key or a host is required are
read off each client class, so a preset cannot describe a provider differently
from the client that will serve it.

Adding a provider is a new module plus a line in
``services.llm_clients.registry`` — this file needs no edit.
"""

from __future__ import annotations

from services.llm_clients.registry import PROVIDER_CLASSES


def get_catalog() -> list[dict[str, object]]:
    """The platforms offered during provider setup, in registry order."""
    return [
        {
            "id": client.PLATFORM,
            "name": client.LABEL,
            "platform": client.PLATFORM,
            "host": client.DEFAULT_BASE_URL,
            "needs_key": client.REQUIRES_KEY,
            "needs_host": client.REQUIRES_HOST,
        }
        for client in PROVIDER_CLASSES
    ]
