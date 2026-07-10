"""MCP server tools action — nested resource under /api/mcp-clients/tools.

Covers:
- GET /api/mcp-clients/tools/<id>  → get (one server's synced tool inventory)
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse, NotFoundError
from api.response.mcp_tool import McpTool
from services.mcp_client_service import McpClientService


class McpTools(Action):
    """Action listing the synced tool inventory of a single outbound MCP server."""

    id_type: ClassVar[type[int] | type[str]] = str
    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get"})
    response_dto = {"get": DocumentedResponse(McpTool, listing=True)}

    def slug(self) -> str:
        return "mcp-clients"

    def verb(self) -> str:
        return "tools"

    def get(self, id: int | str) -> ResponseReturnValue:
        svc = McpClientService()
        server_id = cast(str, id)
        if svc.get_server(server_id) is None:
            raise NotFoundError("Server not found")
        items = [self._to_response(t) for t in svc.get_server_tools(server_id)]
        return McpTool.listing(items, page=1, limit=max(len(items), 1), total=len(items))

    def _to_response(self, row: dict[str, object]) -> McpTool:
        return McpTool(
            tool_name=cast(str, row["tool_name"]),
            summary=cast(str, row["summary"]),
            schema=cast("dict[str, object]", row["schema"]),
        )
