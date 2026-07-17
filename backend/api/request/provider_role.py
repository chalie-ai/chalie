"""Inbound DTOs for the provider-role actions (selected, vision, delegate)."""

from __future__ import annotations

from .request import Request


class ProviderRef(Request):
    """Body for POST /api/providers/selected — provider_id is required and non-null."""

    provider_id: int


class NullableProviderRef(Request):
    """Body for POST /api/providers/vision and /api/providers/delegate.

    ``provider_id`` is required as a key but nullable — ``null`` clears the pin.
    """

    provider_id: int | None
