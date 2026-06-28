# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging
from typing import TYPE_CHECKING

from flask import request
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from .auth import require_session

if TYPE_CHECKING:
    from services.settings_service import SettingsService
    from services.wrapper_auth_service import WrapperAuthService

logger = logging.getLogger(__name__)

mcp_settings_ns = Namespace("mcp_settings", description="MCP server settings", path="/api/mcp-server")


def _get_services() -> "tuple[SettingsService, WrapperAuthService, object]":
    from services.database_service import get_shared_db_service
    from services.settings_service import SettingsService
    from services.wrapper_auth_service import WrapperAuthService

    db = get_shared_db_service()
    return SettingsService(db), WrapperAuthService(db), db


@mcp_settings_ns.route("")
class McpSettingsResource(Resource):
    @require_session
    @mcp_settings_ns.response(200, "Success")
    def get(self) -> ResponseReturnValue:
        settings, auth_svc, _ = _get_services()

        enabled = settings.get("mcp_server_enabled")
        port = settings.get("mcp_server_port") or "8462"
        wrapper_id = settings.get("mcp_server_token_wrapper_id")

        token_display = None
        if wrapper_id:
            wrapper = auth_svc.get_wrapper(wrapper_id)
            if wrapper:
                token_display = _get_stored_token(settings)

        return {
            "enabled": enabled is None or str(enabled).lower() not in ("false", "0", "no"),
            "port": int(port) if port and port.isdigit() else 8462,
            "token": token_display,
            "wrapper_id": wrapper_id,
        }

    @require_session
    @mcp_settings_ns.response(200, "Success")
    @mcp_settings_ns.response(400, "Bad request")
    def put(self) -> ResponseReturnValue:
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
                    return {"error": "Port must be between 1024 and 65535"}, 400
            except (ValueError, TypeError):
                return {"error": "Invalid port number"}, 400

        return {"success": True}


@mcp_settings_ns.route("/regenerate-token")
class RegenerateTokenResource(Resource):
    @require_session
    @mcp_settings_ns.response(200, "Success")
    def post(self) -> ResponseReturnValue:
        settings, auth_svc, _ = _get_services()

        old_wrapper_id = settings.get("mcp_server_token_wrapper_id")
        if old_wrapper_id:
            auth_svc.revoke(old_wrapper_id)

        raw_token, wrapper_id = auth_svc.create_token(
            name="MCP Server (External Agents)",
            capabilities={"signals": []},
            permissions={"query": ["*"], "broadcast": False},
            wrapper_id_override=f"__mcp_server_{_short_id()}__",
        )

        settings.set("mcp_server_token_wrapper_id", wrapper_id)
        settings.set("mcp_server_token", raw_token)

        return {
            "token": raw_token,
            "wrapper_id": wrapper_id,
        }


def _get_stored_token(settings: "SettingsService") -> str | None:
    return settings.get("mcp_server_token")


def _short_id() -> str:
    import uuid
    return uuid.uuid4().hex[:8]