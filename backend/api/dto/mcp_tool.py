"""DTOs for the MCP tools sub-resource — the per-server tool inventory.

``server_id`` is never on a tool DTO: it lives in the URL (the parent segment of
``/mcp-clients/<server_id>/tools``). One class per file, per the namespace
convention.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field

from .base import DTO


class McpTool(DTO):
    """Read shape for one synced tool of an MCP server.

    The wire field is ``schema``; the attribute is ``schema_`` to avoid shadowing
    ``BaseModel.schema()``. ``serialize_by_alias`` re-emits it as ``schema`` through
    the foundation's plain ``model_dump`` (no ``by_alias`` at the call site).
    """

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    tool_name: str
    summary: str
    schema_: dict[str, object] = Field(alias="schema")


class DiscoverableTools(DTO):
    """Flat list of discoverable tool names (enabled+online servers) plus the count."""

    tools: list[str]
    count: int
