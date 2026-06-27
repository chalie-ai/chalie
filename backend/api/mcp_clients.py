"""
MCP Clients namespace — outbound MCP server management API.

Provides CRUD + test/sync endpoints for the mcp_client_servers table.
Auto-registered by api/__init__._register_namespaces() — no edits to
__init__.py required.

All endpoints return 200 (not 201 for create) per the design contract
locked by an end-to-end scenario.
"""

import logging
from typing import cast

from flask import request
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from .auth import require_session
from services.mcp_client_service import McpClientService

logger = logging.getLogger(__name__)

mcp_clients_bp = Namespace("mcp_clients", description="Outbound MCP server management", path="/api/mcp-clients")


def _svc() -> McpClientService:
    return McpClientService()


# ── GET/POST /api/mcp-clients ─────────────────────────────────────────────────


@mcp_clients_bp.route("/")
class ServerListResource(Resource):
    @require_session
    @mcp_clients_bp.response(200, "Success")
    def get(self) -> ResponseReturnValue:
        servers = _svc().list_servers()
        return servers, 200

    @require_session
    @mcp_clients_bp.response(200, "Success")
    @mcp_clients_bp.response(400, "Bad request")
    def post(self) -> ResponseReturnValue:
        body = request.get_json(silent=True) or {}
        name = (body.get("name") or "").strip()
        host = (body.get("host") or "").strip()
        if not name or not host:
            return {"error": "name and host are required"}, 400
        headers = body.get("headers") or {}
        if not isinstance(headers, dict):
            return {"error": "headers must be a JSON object"}, 400
        enabled = body.get("enabled", True)

        svc = _svc()
        server = svc.add_server(name=name, host=host, headers=headers, enabled=bool(enabled))
        # Trigger an immediate ping+sync so tools are indexed promptly.
        # Errors are swallowed — the row is persisted regardless.
        try:
            sync_result = svc.ping_and_sync(cast(str, server["id"]))
            if sync_result.get("reachable"):
                # Build vector embeddings for the newly-synced tools so semantic
                # queries can reach them immediately.  Add-only — not called on the
                # /test endpoint or heartbeat so the sync path stays zero-cost.
                svc.embed_server_tools(cast(str, server["id"]))
            # Reload to get the updated status.
            refreshed = svc.get_server(cast(str, server["id"]))
            if refreshed:
                server = refreshed
        except Exception as exc:
            logger.warning("[MCP CLIENTS] Initial ping failed for %r: %s", name, exc)

        return server, 200


# ── GET /api/mcp-clients/discoverable ────────────────────────────────────────


@mcp_clients_bp.route("/discoverable")
class DiscoverableResource(Resource):
    @require_session
    @mcp_clients_bp.response(200, "Success")
    def get(self) -> ResponseReturnValue:
        names = _svc().get_online_mcp_tool_names()
        return {"tools": names, "count": len(names)}, 200


# ── PUT/DELETE /api/mcp-clients/<id> ─────────────────────────────────────────


@mcp_clients_bp.route("/<server_id>")
class ServerResource(Resource):
    @require_session
    @mcp_clients_bp.response(200, "Success")
    @mcp_clients_bp.response(400, "Bad request")
    @mcp_clients_bp.response(404, "Server not found")
    def put(self, server_id: str) -> ResponseReturnValue:
        body = request.get_json(silent=True) or {}
        updates = {k: v for k, v in body.items() if k in ("name", "host", "headers", "enabled")}
        if not updates:
            return {"error": "No valid fields to update"}, 400
        try:
            server = _svc().update_server(server_id, updates)
        except LookupError:
            return {"error": f"Server not found: {server_id}"}, 404
        return server, 200

    @require_session
    @mcp_clients_bp.response(200, "Success")
    @mcp_clients_bp.response(404, "Server not found")
    def delete(self, server_id: str) -> ResponseReturnValue:
        try:
            _svc().delete_server(server_id)
        except LookupError:
            return {"error": f"Server not found: {server_id}"}, 404
        return {"deleted": server_id}, 200


# ── POST /api/mcp-clients/<id>/test ──────────────────────────────────────────


@mcp_clients_bp.route("/<server_id>/test")
class ServerTestResource(Resource):
    @require_session
    @mcp_clients_bp.response(200, "Success")
    @mcp_clients_bp.response(404, "Server not found")
    def post(self, server_id: str) -> ResponseReturnValue:
        svc = _svc()
        if svc.get_server(server_id) is None:
            return {"error": f"Server not found: {server_id}"}, 404
        result = svc.ping_and_sync(server_id)
        return result, 200


# ── GET /api/mcp-clients/<id>/tools ──────────────────────────────────────────


@mcp_clients_bp.route("/<server_id>/tools")
class ServerToolsResource(Resource):
    @require_session
    @mcp_clients_bp.response(200, "Success")
    @mcp_clients_bp.response(404, "Server not found")
    def get(self, server_id: str) -> ResponseReturnValue:
        svc = _svc()
        if svc.get_server(server_id) is None:
            return {"error": f"Server not found: {server_id}"}, 404
        tools = svc.get_server_tools(server_id)
        return {"tools": tools, "tool_count": len(tools)}, 200