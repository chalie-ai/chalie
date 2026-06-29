"""DTOs for the MCP clients resource — the servers HTTP contract.

One class per file is the namespace convention; ``headers`` is a write-only
credential carrier and lives only on the create/update request DTOs (it never
appears on ``McpServer``, the read shape). Datetimes serialize as ISO-8601 UTC
via :class:`backend.api.dto.base.DTO`.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .base import DTO


class McpServer(DTO):
    """Read shape for an outbound MCP server. ``headers`` is write-only — never returned."""

    id: str
    name: str
    host: str
    enabled: bool
    status: str
    last_pinged_at: datetime | None
    created_at: datetime
    updated_at: datetime


class McpServerCreate(DTO):
    """Inbound body to add a server. ``headers`` carries credentials to the remote MCP server."""

    name: str = Field(..., min_length=1, max_length=200)
    host: str = Field(..., min_length=1)
    headers: dict[str, str] = {}
    enabled: bool = True


class McpServerUpdate(DTO):
    """Partial update of a server by id; every field optional."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    host: str | None = Field(default=None, min_length=1)
    headers: dict[str, str] | None = None
    enabled: bool | None = None


class McpTestResult(DTO):
    """Result of pinging+syncing one server (status, tool_count, reachable, error)."""

    status: str
    tool_count: int
    reachable: bool
    error: str | None
