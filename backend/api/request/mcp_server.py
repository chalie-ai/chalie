"""Inbound DTO for the MCP clients endpoint group.

One request DTO covers both create and update: every field is optional —
``None`` means "leave unchanged" on update, while the endpoint's ``post()``
enforces ``name`` and ``host`` are required on create and defaults
``enabled`` to ``True`` when omitted. ``headers`` carries credentials to the
remote MCP server and is a write-only field — it lives here only and never
appears on any response DTO.
"""

from __future__ import annotations

from pydantic import Field

from .request import Request


class McpServerRequest(Request):
    """Inbound body for MCP server create/update."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    host: str | None = Field(default=None, min_length=1)
    headers: dict[str, str] | None = None
    enabled: bool | None = None
