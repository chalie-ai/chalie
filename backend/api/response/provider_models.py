"""Response DTOs for the provider model-listing and connectivity-test actions."""

from __future__ import annotations

from .response import Response


class ModelInfo(Response):
    """One model entry returned by the list-models action."""

    id: str
    display_name: str | None = None


class ListModelsResult(Response):
    """Result of listing a platform's available models — success and upstream-failure both."""

    models: list[ModelInfo]
    error: str | None = None


class ProviderTestResult(Response):
    """Result of a connectivity test — success and failure paths both."""

    success: bool
    model: str | None = None
    latency_ms: int | None = None
    message: str | None = None
    error: str | None = None
    hint: str | None = None
