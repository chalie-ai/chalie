"""DTOs for the health probe — the liveness response + the loose heartbeat ingest body."""

from __future__ import annotations

from pydantic import ConfigDict

from .base import DTO


class Health(DTO):
    """Liveness response — status + running app version."""

    status: str
    version: str


class ClientTelemetry(DTO):
    """Loose inbound body for ``POST /health`` heartbeat ingest.

    The client telemetry shape is arbitrary and best-effort; ``extra="allow"`` lets
    any keys through so a heartbeat never errors on an unknown field.
    """

    model_config = ConfigDict(extra="allow")