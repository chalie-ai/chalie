"""Response DTO for the MCP server settings endpoint.

``enabled`` and ``port`` are the stored intent; ``listening``,
``listening_port`` and ``error`` are the listener's live state at read time,
so a toggle or port change is confirmed from the same record it was written to.
"""

from __future__ import annotations

from .response import Response


class McpServerSettings(Response):
    """Read shape for the MCP server settings record."""

    enabled: bool
    port: int
    listening: bool
    listening_port: int | None
    error: str | None
