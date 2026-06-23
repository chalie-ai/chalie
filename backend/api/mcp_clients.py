"""
MCP Clients blueprint — outbound MCP server management API.

Provides CRUD + test/sync endpoints for the mcp_client_servers table.
Auto-registered by api/__init__._register_blueprints() — no edits to
__init__.py required.

All endpoints return 200 (not 201 for create) per the design contract
locked by an end-to-end scenario.
"""

import logging
from typing import TYPE_CHECKING, cast

from flask import Blueprint, jsonify, request

from .auth import require_session
from services.mcp_client_service import McpClientService

if TYPE_CHECKING:
    from flask.typing import ResponseReturnValue

logger = logging.getLogger(__name__)

mcp_clients_bp = Blueprint("mcp_clients", __name__, url_prefix="/api/mcp-clients")


def _svc() -> McpClientService:
    return McpClientService()


# ── GET /api/mcp-clients ─────────────────────────────────────────────────────


@mcp_clients_bp.route("/", methods=["GET"])
@require_session
def list_servers() -> "ResponseReturnValue":
    servers = _svc().list_servers()
    return jsonify(servers), 200


# ── POST /api/mcp-clients — add a server ─────────────────────────────────────


@mcp_clients_bp.route("/", methods=["POST"])
@require_session
def add_server() -> "ResponseReturnValue":
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    host = (body.get("host") or "").strip()
    if not name or not host:
        return jsonify({"error": "name and host are required"}), 400
    headers = body.get("headers") or {}
    if not isinstance(headers, dict):
        return jsonify({"error": "headers must be a JSON object"}), 400
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

    return jsonify(server), 200


# ── GET /api/mcp-clients/discoverable ────────────────────────────────────────


@mcp_clients_bp.route("/discoverable", methods=["GET"])
@require_session
def get_discoverable() -> "ResponseReturnValue":
    names = _svc().get_online_mcp_tool_names()
    return jsonify({"tools": names, "count": len(names)}), 200


# ── PUT /api/mcp-clients/<id> ─────────────────────────────────────────────────


@mcp_clients_bp.route("/<server_id>", methods=["PUT"])
@require_session
def update_server(server_id: str) -> "ResponseReturnValue":
    body = request.get_json(silent=True) or {}
    updates = {k: v for k, v in body.items() if k in ("name", "host", "headers", "enabled")}
    if not updates:
        return jsonify({"error": "No valid fields to update"}), 400
    try:
        server = _svc().update_server(server_id, updates)
    except LookupError:
        return jsonify({"error": f"Server not found: {server_id}"}), 404
    return jsonify(server), 200


# ── DELETE /api/mcp-clients/<id> ─────────────────────────────────────────────


@mcp_clients_bp.route("/<server_id>", methods=["DELETE"])
@require_session
def delete_server(server_id: str) -> "ResponseReturnValue":
    try:
        _svc().delete_server(server_id)
    except LookupError:
        return jsonify({"error": f"Server not found: {server_id}"}), 404
    return jsonify({"deleted": server_id}), 200


# ── POST /api/mcp-clients/<id>/test ──────────────────────────────────────────


@mcp_clients_bp.route("/<server_id>/test", methods=["POST"])
@require_session
def test_server(server_id: str) -> "ResponseReturnValue":
    svc = _svc()
    if svc.get_server(server_id) is None:
        return jsonify({"error": f"Server not found: {server_id}"}), 404
    result = svc.ping_and_sync(server_id)
    return jsonify(result), 200


# ── GET /api/mcp-clients/<id>/tools ──────────────────────────────────────────


@mcp_clients_bp.route("/<server_id>/tools", methods=["GET"])
@require_session
def get_server_tools(server_id: str) -> "ResponseReturnValue":
    svc = _svc()
    if svc.get_server(server_id) is None:
        return jsonify({"error": f"Server not found: {server_id}"}), 404
    tools = svc.get_server_tools(server_id)
    return jsonify({"tools": tools, "tool_count": len(tools)}), 200
