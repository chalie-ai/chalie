"""MCP discoverable-tools action — nested resource under /api/mcp-clients/discoverable.

Covers:
- GET /api/mcp-clients/discoverable  → get (online, enabled server tool names)

Only the sentinel/id-less form is meaningful — an id-addressed call 404s,
since discoverable tools are not addressed by a single server id.
"""

from __future__ import annotations

from typing import ClassVar

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from exceptions import NotFoundError
from api.response.mcp_tool import DiscoverableTools
from services.mcp_client_service import McpClientService


class McpDiscoverable(Action):
    """Action listing every discoverable (enabled+online) MCP tool name."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get"})
    response_dto = {"get": DocumentedResponse(DiscoverableTools)}

    def get(self, id: int | str) -> ResponseReturnValue:
        if not self.is_create(id):
            raise NotFoundError("Not found")
        names = McpClientService().get_online_mcp_tool_names()
        return DiscoverableTools(tools=names, count=len(names)).single()
