"""DTOs for the providers resource — read shape and write bodies."""

from __future__ import annotations

from pydantic import Field

from .base import DTO


class Provider(DTO):
    """Read shape for one provider — api_key is never included."""

    id: int
    name: str
    platform: str
    model: str
    host: str | None
    dimensions: int | None
    timeout: int | None
    supports_vision: bool
    max_tokens: int | None


class ProviderCreate(DTO):
    """Inbound body to create a provider."""

    name: str = Field(..., min_length=1, max_length=200)
    platform: str
    model: str = Field(..., min_length=1, max_length=200)
    host: str | None = None
    api_key: str | None = None
    dimensions: int | None = None
    timeout: int = 120


class ProviderUpdate(DTO):
    """Partial update of a provider; every field optional."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    platform: str | None = None
    model: str | None = Field(default=None, min_length=1, max_length=200)
    host: str | None = None
    api_key: str | None = None
    dimensions: int | None = None
    timeout: int | None = None
