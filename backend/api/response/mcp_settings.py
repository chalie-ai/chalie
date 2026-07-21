"""Response DTOs for the MCP server settings endpoint.

Mirrors the field shape of the DTOs this endpoint group replaced, so the
wire format stays identical. ``token`` is the raw bearer credential —
owner-visible on read and on regenerate (preserved current behaviour);
there is NO write path for it. ``wrapper_id`` is the opaque identifier
for the wrapper token record.
"""

from __future__ import annotations

from .response import Response


class McpServerSettings(Response):
    """Read shape for the MCP server settings record."""

    enabled: bool
    port: int
    token: str | None
    wrapper_id: str | None


class RegenerateTokenResult(Response):
    """Result of regenerating the MCP server bearer token."""

    token: str
    wrapper_id: str
