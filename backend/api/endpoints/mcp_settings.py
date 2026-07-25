"""MCP server settings endpoint — singleton settings record.

Not pure CRUD: one settings record (a fixed key-set in the ``settings`` table,
no resource id). Read responses are DTO-typed through the foundation boundary;
the settings update is a non-CRUD action returning 204 no-body. The raw
``token`` is owner-visible on read and on regenerate (preserved current
behaviour); there is no write path for it.

Covers the singleton read/update routes the legacy module exposed:
- GET    /api/mcp-server/all   → get_all (one-item listing, singleton record)
- POST   /api/mcp-server/-1    → post (partial update, legacy PUT → POST)

The legacy module had no id-addressed routes, so ``get``/``delete`` stay the
base 405 stubs; a real id on ``post`` is a 404 (nothing lives at that id).

The ``/regenerate-token`` verb lives in
``api/actions/mcp_server/regenerate_token.py`` as a separate Action subclass.
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.endpoint import DocumentedResponse, Endpoint
from api.request import Request
from api.request.mcp_settings import McpSettingsUpdate
from api.response.mcp_settings import McpServerSettings
from exceptions import NotFoundError
from models.setting import Setting
from services.wrapper_auth_service import WrapperAuthService


class McpSettingsEndpoint(Endpoint):
    """Singleton settings record for the MCP server — no per-resource id."""

    id_type: ClassVar[type[int] | type[str]] = str
    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get_all", "post"})
    request_dto: ClassVar[type[Request] | None] = McpSettingsUpdate
    # The update POST never creates — the singleton record always exists.
    _post_may_create: ClassVar[bool] = False
    response_dto = {
        "get_all": DocumentedResponse(McpServerSettings, listing=True),
        "post": DocumentedResponse(extras=((404, "No per-resource id"),)),
    }

    def get_all(self, page: int = 1, limit: int = 20) -> ResponseReturnValue:
        """Read the singleton settings record."""
        dto = self._settings_dto()
        total = 1
        return McpServerSettings.listing([dto], page=page, limit=limit, total=total)

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Partial update of the singleton settings (id=-1 only).

        Args:
            id: Must be the create sentinel ``-1``; a real id is a structural 404.
            data: McpSettingsUpdate — only provided fields are written.

        Returns:
            204 No Content on success.

        Raises:
            NotFoundError: If id is not the create sentinel — the singleton
                has no per-resource id, so nothing lives at a real id.
        """
        if not self.is_create(id):
            raise NotFoundError("Singleton settings record has no per-resource id")
        self._update(cast(McpSettingsUpdate, data))
        return "", 204

    def _settings_dto(self) -> McpServerSettings:
        """Build the read DTO from the settings keys, preserving the truthiness/port parsing."""
        enabled = Setting.get_value("mcp_server_enabled")
        port = Setting.get_value("mcp_server_port") or str(8462)
        wrapper_id = Setting.get_value("mcp_server_token_wrapper_id")

        auth_svc = WrapperAuthService()
        token_display = None
        if wrapper_id and auth_svc.get_wrapper(wrapper_id):
            token_display = Setting.get_value("mcp_server_token")

        return McpServerSettings(
            enabled=enabled is None or str(enabled).lower() not in ("false", "0", "no"),
            port=int(port) if port and port.isdigit() else 8462,
            token=token_display,
            wrapper_id=wrapper_id,
        )

    def _update(self, dto: McpSettingsUpdate) -> None:
        """Apply partial update to the singleton settings."""
        if dto.enabled is not None:
            Setting.set("mcp_server_enabled", "true" if dto.enabled else "false")
        if dto.port is not None:
            Setting.set("mcp_server_port", str(dto.port))
