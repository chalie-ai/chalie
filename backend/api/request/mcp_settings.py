"""Inbound DTO for the MCP server settings endpoint.

One partial-update request covers the singleton settings record: every field is
optional — ``None`` means "leave unchanged". ``enabled`` toggles the MCP server
on/off; ``port`` is the listening port (1024–65535). The raw ``token`` is the
server's bearer credential — owner-visible on read and on regenerate (preserved
current behaviour); there is NO write path for it.
"""

from __future__ import annotations

from pydantic import Field

from .request import Request


class McpSettingsUpdate(Request):
    """Partial update of the MCP server settings; every field optional."""

    enabled: bool | None = None
    port: int | None = Field(default=None, ge=1024, le=65535)
