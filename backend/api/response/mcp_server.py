"""Response DTOs for the MCP clients endpoint group.

Mirrors the field shape of the DTOs this endpoint group replaced, so the
wire format stays identical. ``headers`` is a write-only credential
carrier — it lives on the request DTO only and never appears here.
"""

from __future__ import annotations

from datetime import datetime

from .response import Response


class McpServer(Response):
    """Read shape for an outbound MCP server. ``headers`` is write-only — never returned."""

    id: str
    name: str
    host: str
    enabled: bool
    status: str
    last_pinged_at: datetime | None
    created_at: datetime
    updated_at: datetime


class McpTestResult(Response):
    """Result of pinging+syncing one server (status, tool_count, reachable, error)."""

    status: str
    tool_count: int
    reachable: bool
    error: str | None
