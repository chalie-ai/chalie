"""MCP Clients namespace — outbound MCP server management.

CRUD plus discover/test/tools endpoints, all DTO-typed through the foundation
boundary decorators. ``headers`` is a write-only credential carrier: it lives on
the create/update request DTOs only and is never present on a response.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import cast

from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from services.mcp_client_service import McpClientService
from .auth import require_session
from .dto import Error, expects, register_dto, responds
from .dto.boundary import error
from .dto.mcp_server import McpServer, McpServerCreate, McpServerUpdate, McpTestResult
from .dto.mcp_tool import DiscoverableTools, McpTool

logger = logging.getLogger(__name__)

mcp_clients_ns = Namespace(
    "mcp_clients", description="Outbound MCP server management", path="/api/mcp-clients"
)

register_dto(
    mcp_clients_ns, McpServer, McpServerCreate, McpServerUpdate,
    McpTool, DiscoverableTools, McpTestResult, Error,
)

_M = mcp_clients_ns.models

_NOT_FOUND = "Server not found"


def _svc() -> McpClientService:
    return McpClientService()


def _server_dto(row: dict[str, object]) -> McpServer:
    return McpServer(
        id=cast(str, row["id"]),
        name=cast(str, row["name"]),
        host=cast(str, row["host"]),
        enabled=cast(bool, row["enabled"]),
        status=cast(str, row["status"]),
        last_pinged_at=cast("datetime | None", row["last_pinged_at"]),
        created_at=cast(datetime, row["created_at"]),
        updated_at=cast(datetime, row["updated_at"]),
    )


def _tool_dto(row: dict[str, object]) -> McpTool:
    return McpTool(
        tool_name=cast(str, row["tool_name"]),
        summary=cast(str, row["summary"]),
        schema=cast("dict[str, object]", row["schema"]),
    )


# ---------------------------------------------------------------------------
# /api/mcp-clients/  — server collection
# ---------------------------------------------------------------------------


@mcp_clients_ns.route("/")
class ServerListResource(Resource):
    @require_session
    @mcp_clients_ns.response(200, "All servers", model=_M["McpServer"])
    @mcp_clients_ns.response(500, "Internal error", model=_M["Error"])
    @responds(McpServer, code=200)
    def get(self) -> list[McpServer]:
        return [_server_dto(s) for s in _svc().list_servers()]

    @require_session
    @mcp_clients_ns.expect(_M["McpServerCreate"])
    @mcp_clients_ns.response(201, "Created", model=_M["McpServer"])
    @mcp_clients_ns.response(422, "Validation failed", model=_M["Error"])
    @responds(McpServer, code=201)
    @expects(McpServerCreate)
    def post(self, dto: McpServerCreate) -> McpServer | ResponseReturnValue:
        svc = _svc()
        server = svc.add_server(
            name=dto.name, host=dto.host, headers=dto.headers, enabled=dto.enabled,
        )
        try:
            sync_result = svc.ping_and_sync(cast(str, server["id"]))
            if sync_result.get("reachable"):
                svc.embed_server_tools(cast(str, server["id"]))
            refreshed = svc.get_server(cast(str, server["id"]))
            if refreshed:
                server = refreshed
        except Exception as exc:
            logger.warning("[MCP CLIENTS] Initial ping failed for %r: %s", dto.name, exc)
        return _server_dto(server)


# ---------------------------------------------------------------------------
# /api/mcp-clients/discoverable
# ---------------------------------------------------------------------------


@mcp_clients_ns.route("/discoverable")
class DiscoverableResource(Resource):
    @require_session
    @mcp_clients_ns.response(200, "Discoverable tool names", model=_M["DiscoverableTools"])
    @mcp_clients_ns.response(500, "Internal error", model=_M["Error"])
    @responds(DiscoverableTools, code=200)
    def get(self) -> DiscoverableTools:
        names = _svc().get_online_mcp_tool_names()
        return DiscoverableTools(tools=names, count=len(names))


# ---------------------------------------------------------------------------
# /api/mcp-clients/<server_id>
# ---------------------------------------------------------------------------


@mcp_clients_ns.route("/<server_id>")
class ServerResource(Resource):
    @require_session
    @mcp_clients_ns.param("server_id", "Server id")
    @mcp_clients_ns.expect(_M["McpServerUpdate"])
    @mcp_clients_ns.response(200, "Updated", model=_M["McpServer"])
    @mcp_clients_ns.response(404, _NOT_FOUND, model=_M["Error"])
    @mcp_clients_ns.response(422, "Validation failed", model=_M["Error"])
    @responds(McpServer, code=200)
    @expects(McpServerUpdate)
    def put(self, server_id: str, dto: McpServerUpdate) -> McpServer | ResponseReturnValue:
        updates: dict[str, object] = {k: v for k, v in dto.model_dump().items() if v is not None}
        try:
            server = _svc().update_server(server_id, updates)
        except LookupError:
            return error(_NOT_FOUND, 404)
        return _server_dto(server)

    @require_session
    @mcp_clients_ns.param("server_id", "Server id")
    @mcp_clients_ns.response(204, "Deleted")
    @mcp_clients_ns.response(404, _NOT_FOUND, model=_M["Error"])
    @responds(code=204)
    def delete(self, server_id: str) -> None | ResponseReturnValue:
        try:
            _svc().delete_server(server_id)
        except LookupError:
            return error(_NOT_FOUND, 404)
        return None


# ---------------------------------------------------------------------------
# /api/mcp-clients/<server_id>/test
# ---------------------------------------------------------------------------


@mcp_clients_ns.route("/<server_id>/test")
class ServerTestResource(Resource):
    @require_session
    @mcp_clients_ns.param("server_id", "Server id")
    @mcp_clients_ns.response(200, "Test result", model=_M["McpTestResult"])
    @mcp_clients_ns.response(404, _NOT_FOUND, model=_M["Error"])
    @responds(McpTestResult, code=200)
    def post(self, server_id: str) -> McpTestResult | ResponseReturnValue:
        svc = _svc()
        if svc.get_server(server_id) is None:
            return error(_NOT_FOUND, 404)
        result = svc.ping_and_sync(server_id)
        return McpTestResult(
            status=cast(str, result["status"]),
            tool_count=cast(int, result["tool_count"]),
            reachable=cast(bool, result["reachable"]),
            error=cast("str | None", result["error"]),
        )


# ---------------------------------------------------------------------------
# /api/mcp-clients/<server_id>/tools
# ---------------------------------------------------------------------------


@mcp_clients_ns.route("/<server_id>/tools")
class ServerToolsResource(Resource):
    @require_session
    @mcp_clients_ns.param("server_id", "Server id")
    @mcp_clients_ns.response(200, "Server tools", model=_M["McpTool"])
    @mcp_clients_ns.response(404, _NOT_FOUND, model=_M["Error"])
    @responds(McpTool, code=200)
    def get(self, server_id: str) -> list[McpTool] | ResponseReturnValue:
        svc = _svc()
        if svc.get_server(server_id) is None:
            return error(_NOT_FOUND, 404)
        return [_tool_dto(t) for t in svc.get_server_tools(server_id)]
