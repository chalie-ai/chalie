"""DTOs for provider-role endpoints (selected, vision, delegate)."""

from __future__ import annotations

from .base import DTO
from .provider import Provider


class ProviderRef(DTO):
    """Body for PUT /selected — provider_id is required and non-null."""

    provider_id: int


class NullableProviderRef(DTO):
    """Body for PUT /vision and PUT /delegate — provider_id may be null to clear."""

    provider_id: int | None


class ProviderRole(DTO):
    """Response shape for GET/PUT vision and delegate endpoints."""

    provider: Provider | None
    source: str
