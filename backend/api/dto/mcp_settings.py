"""DTOs for the MCP-server-settings namespace — the single settings record contract.

The MCP server is configured by a fixed key-set in the ``settings`` table (no
resource id, no collection): one read shape, one partial-update request, and one
action result. The ``token`` is the server's raw bearer credential — owner-visible
on read and on regenerate (preserved current behaviour); there is NO write path
for it (regeneration is the only way to mint one), so no update field carries it.
Port range validation lives on the ``Field`` constraint, so out-of-range/non-int
now fails at the boundary (422) instead of an inline 400.
"""

from __future__ import annotations

from pydantic import Field

from .base import DTO


class McpServerSettings(DTO):
    """Read shape for the MCP server settings record."""

    enabled: bool
    port: int
    token: str | None
    wrapper_id: str | None


class McpSettingsUpdate(DTO):
    """Partial update of the MCP server settings; every field optional."""

    enabled: bool | None = None
    port: int | None = Field(default=None, ge=1024, le=65535)


class RegenerateTokenResult(DTO):
    """Result of regenerating the MCP server bearer token."""

    token: str
    wrapper_id: str