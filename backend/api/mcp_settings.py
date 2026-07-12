# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MCP-server-settings namespace — the single MCP server settings record + token action.

Not pure CRUD: one settings record (a fixed key-set in the ``settings`` table,
no resource id) plus a regenerate-token action that mints a new credential. Read
responses are DTO-typed through the foundation boundary; the settings update is a
non-CRUD action returning 204 no-body; token regeneration creates a credential
resource and returns 201. The raw ``token`` is owner-visible on read and on
regenerate (preserved current behaviour); there is no write path for it.
"""

import logging
from typing import TYPE_CHECKING

from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from models.setting import Setting
from .auth import require_session
from .dto import Error, expects, register_dto, responds
from .dto.boundary import error
from .dto.mcp_settings import McpServerSettings, McpSettingsUpdate, RegenerateTokenResult

if TYPE_CHECKING:
    from services.wrapper_auth_service import WrapperAuthService

logger = logging.getLogger(__name__)

_ERR_READ = "Failed to read MCP server settings"
_ERR_UPDATE = "Failed to update MCP server settings"
_ERR_REGEN = "Failed to regenerate MCP server token"
_DEFAULT_PORT = 8462
_TOKEN_NAME = "MCP Server (External Agents)"

mcp_settings_ns = Namespace("mcp_settings", description="MCP server settings", path="/api/mcp-server")

register_dto(
    mcp_settings_ns,
    McpServerSettings,
    McpSettingsUpdate,
    RegenerateTokenResult,
    Error,
)

_M = mcp_settings_ns.models


def _get_auth_service() -> "WrapperAuthService":
    from services.wrapper_auth_service import WrapperAuthService

    return WrapperAuthService()


def _get_stored_token() -> str | None:
    return Setting.get("mcp_server_token")


def _short_id() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


def _settings_dto(auth_svc: "WrapperAuthService") -> McpServerSettings:
    """Build the read DTO from the settings keys, preserving the truthiness/port parsing."""
    enabled = Setting.get("mcp_server_enabled")
    port = Setting.get("mcp_server_port") or str(_DEFAULT_PORT)
    wrapper_id = Setting.get("mcp_server_token_wrapper_id")

    token_display = None
    if wrapper_id and auth_svc.get_wrapper(wrapper_id):
        token_display = _get_stored_token()

    return McpServerSettings(
        enabled=enabled is None or str(enabled).lower() not in ("false", "0", "no"),
        port=int(port) if port and port.isdigit() else _DEFAULT_PORT,
        token=token_display,
        wrapper_id=wrapper_id,
    )


@mcp_settings_ns.route("")
class McpSettingsResource(Resource):
    @require_session
    @mcp_settings_ns.response(200, "MCP server settings", model=_M["McpServerSettings"])
    @mcp_settings_ns.response(500, _ERR_READ, model=_M["Error"])
    @responds(McpServerSettings, code=200)
    def get(self) -> McpServerSettings | ResponseReturnValue:
        """Read the MCP server settings record."""
        try:
            auth_svc = _get_auth_service()
            return _settings_dto(auth_svc)
        except Exception as exc:
            logger.exception("[MCP SETTINGS] GET failed: %s", exc)
            return error(_ERR_READ, 500)

    @require_session
    @mcp_settings_ns.expect(_M["McpSettingsUpdate"])
    @mcp_settings_ns.response(204, "Settings updated")
    @mcp_settings_ns.response(422, "Validation failed", model=_M["Error"])
    @mcp_settings_ns.response(500, _ERR_UPDATE, model=_M["Error"])
    @responds(code=204)
    @expects(McpSettingsUpdate)
    def put(self, dto: McpSettingsUpdate) -> None | ResponseReturnValue:
        """Partially update the MCP server settings (only provided fields are written)."""
        try:
            if dto.enabled is not None:
                Setting.set("mcp_server_enabled", "true" if dto.enabled else "false")
            if dto.port is not None:
                Setting.set("mcp_server_port", str(dto.port))
            return None
        except Exception as exc:
            logger.exception("[MCP SETTINGS] PUT failed: %s", exc)
            return error(_ERR_UPDATE, 500)


@mcp_settings_ns.route("/regenerate-token")
class RegenerateTokenResource(Resource):
    @require_session
    @mcp_settings_ns.response(201, "Token regenerated", model=_M["RegenerateTokenResult"])
    @mcp_settings_ns.response(500, _ERR_REGEN, model=_M["Error"])
    @responds(RegenerateTokenResult, code=201)
    def post(self) -> RegenerateTokenResult | ResponseReturnValue:
        """Mint a fresh MCP server bearer token, revoking the prior one if present."""
        try:
            auth_svc = _get_auth_service()

            old_wrapper_id = Setting.get("mcp_server_token_wrapper_id")
            if old_wrapper_id:
                auth_svc.revoke(old_wrapper_id)

            raw_token, wrapper_id = auth_svc.create_token(
                name=_TOKEN_NAME,
                wrapper_id_override=f"__mcp_server_{_short_id()}__",
            )

            Setting.set("mcp_server_token_wrapper_id", wrapper_id)
            Setting.set("mcp_server_token", raw_token)

            return RegenerateTokenResult(token=raw_token, wrapper_id=wrapper_id)
        except Exception as exc:
            logger.exception("[MCP SETTINGS] regenerate-token failed: %s", exc)
            return error(_ERR_REGEN, 500)