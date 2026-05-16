# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
MCP Server Settings API — manage MCP server enable/disable, port, and token.

Routes:
  GET  /api/mcp-server          — get current MCP server settings + token
  PUT  /api/mcp-server          — update enabled/port settings
  POST /api/mcp-server/regenerate-token — revoke old token and generate new one
"""

import logging

from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

mcp_settings_bp = Blueprint("mcp_settings", __name__, url_prefix="/api/mcp-server")


def _get_services():
    from services.database_service import get_shared_db_service
    from services.settings_service import SettingsService
    from services.wrapper_auth_service import WrapperAuthService

    db = get_shared_db_service()
    return SettingsService(db), WrapperAuthService(db), db


@mcp_settings_bp.route("", methods=["GET"])
@require_session
def get_mcp_settings():
    """Return current MCP server settings and connection token."""
    settings, auth_svc, _ = _get_services()

    enabled = settings.get("mcp_server_enabled")
    port = settings.get("mcp_server_port") or "8462"
    wrapper_id = settings.get("mcp_server_token_wrapper_id")

    token_display = None
    if wrapper_id:
        wrapper = auth_svc.get_wrapper(wrapper_id)
        if wrapper:
            token_display = _get_stored_token(settings)

    return jsonify({
        "enabled": enabled is None or str(enabled).lower() not in ("false", "0", "no"),
        "port": int(port) if port and port.isdigit() else 8462,
        "token": token_display,
        "wrapper_id": wrapper_id,
    })


@mcp_settings_bp.route("", methods=["PUT"])
@require_session
def update_mcp_settings():
    """Update MCP server enabled state and/or port."""
    settings, _, _ = _get_services()
    data = request.get_json(silent=True) or {}

    if "enabled" in data:
        settings.set("mcp_server_enabled", "true" if data["enabled"] else "false")

    if "port" in data:
        port_val = data["port"]
        try:
            port_int = int(port_val)
            if 1024 <= port_int <= 65535:
                settings.set("mcp_server_port", str(port_int))
            else:
                return jsonify({"error": "Port must be between 1024 and 65535"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid port number"}), 400

    return jsonify({"success": True})


@mcp_settings_bp.route("/regenerate-token", methods=["POST"])
@require_session
def regenerate_token():
    """Revoke the current MCP token and generate a new one."""
    settings, auth_svc, _ = _get_services()

    old_wrapper_id = settings.get("mcp_server_token_wrapper_id")
    if old_wrapper_id:
        auth_svc.revoke(old_wrapper_id)

    raw_token, wrapper_id = auth_svc.create_token(
        name="MCP Server (External Agents)",
        capabilities={"signals": [], "intents": ["talk_to_chalie"]},
        permissions={"query": ["*"], "update": ["*"], "broadcast": False},
        wrapper_id_override=f"__mcp_server_{_short_id()}__",
    )

    settings.set("mcp_server_token_wrapper_id", wrapper_id)
    settings.set("mcp_server_token", raw_token)

    return jsonify({
        "token": raw_token,
        "wrapper_id": wrapper_id,
    })


def _get_stored_token(settings):
    """Retrieve the stored raw token (set at generation time)."""
    return settings.get("mcp_server_token")


def _short_id():
    """Generate a short unique suffix for wrapper_id on regeneration."""
    import uuid
    return uuid.uuid4().hex[:8]
