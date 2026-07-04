"""DTOs for provider model-listing and connectivity-test endpoints."""

from __future__ import annotations

from .base import DTO


class ModelInfo(DTO):
    """One model entry returned by the list-models endpoint."""

    id: str
    display_name: str | None = None


class ListModelsRequest(DTO):
    """Inbound body for POST /list-models."""

    platform: str
    host: str | None = None
    api_key: str | None = None


class ListModelsResult(DTO):
    """Response from POST /list-models — both success and upstream failure."""

    models: list[ModelInfo]
    error: str | None = None


class ProviderTestRequest(DTO):
    """Inbound body for POST /test."""

    provider_id: int | None = None
    platform: str | None = None
    model: str | None = None
    host: str | None = None
    api_key: str | None = None


class ProviderTestResult(DTO):
    """Response from POST /test — both success and failure paths."""

    success: bool
    model: str | None = None
    latency_ms: int | None = None
    message: str | None = None
    error: str | None = None
    hint: str | None = None
