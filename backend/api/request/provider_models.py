"""Inbound DTOs for the provider model-listing and connectivity-test actions."""

from __future__ import annotations

from .request import Request


class ListModelsRequest(Request):
    """Inbound body for POST /api/providers/list-models."""

    platform: str
    host: str | None = None
    api_key: str | None = None


class ProviderTestRequest(Request):
    """Inbound body for POST /api/providers/test.

    Every field is optional: ``provider_id`` alone tests a stored provider,
    inline fields alone test an unsaved configuration, and both together
    layer inline overrides on top of the stored provider's config.
    """

    provider_id: int | None = None
    platform: str | None = None
    model: str | None = None
    host: str | None = None
    api_key: str | None = None
