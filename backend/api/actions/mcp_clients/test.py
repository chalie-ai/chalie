"""MCP server test action — nested resource under /api/mcp-clients/test.

Covers:
- POST /api/mcp-clients/test/<id>  → post (ping + sync one server on demand)

The base auto-documents 404 for get/delete handlers but not for post, so this
class declares it explicitly via ``extras`` to keep swagger truthful.
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from exceptions import NotFoundError
from api.request import Request
from api.response.mcp_server import McpTestResult
from services.mcp_client_service import McpClientService


class McpTest(Action):
    """Action for pinging + syncing a single outbound MCP server on demand."""

    id_type: ClassVar[type[int] | type[str]] = str
    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"post"})
    response_dto = {
        "post": DocumentedResponse(McpTestResult, extras=((404, "Server not found"),)),
    }

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        svc = McpClientService()
        server_id = cast(str, id)
        if svc.get_server(server_id) is None:
            raise NotFoundError("Server not found")
        result = svc.ping_and_sync(server_id)
        return McpTestResult(
            connected=cast(bool, result["connected"]),
            tool_count=cast(int, result["tool_count"]),
            error=cast("str | None", result["error"]),
        ).single()
