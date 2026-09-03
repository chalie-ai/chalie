"""MCP server settings endpoint — singleton settings record.

Not pure CRUD: one settings record (a fixed key-set in the ``settings`` table,
no resource id). Reads are DTO-typed through the foundation boundary and carry
the listener's live state next to the stored intent; the update is a non-CRUD
action returning 204 no-body, applied to the running listener before the
response goes out.

- GET    /api/mcp-server/all   → get_all (one-item listing, singleton record)
- POST   /api/mcp-server/-1    → post (partial update)

There are no id-addressed routes, so ``get``/``delete`` stay the base 405
stubs; a real id on ``post`` is a 404 (nothing lives at that id).
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.endpoint import DocumentedResponse, Endpoint
from api.request import Request
from api.request.mcp_settings import McpSettingsUpdate
from api.response.mcp_settings import McpServerSettings
from exceptions import NotFoundError
from mcp_server.server import listener
from models.setting import Setting


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
        """Read the singleton settings record together with the live listener state."""
        enabled, port = listener.desired_state()
        dto = McpServerSettings(
            enabled=enabled,
            port=port,
            listening=listener.listening,
            listening_port=listener.listening_port,
            error=listener.error,
        )
        return McpServerSettings.listing([dto], page=page, limit=limit, total=1)

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Partial update of the singleton settings (id=-1 only), applied live.

        Args:
            id: Must be the create sentinel ``-1``; a real id is a structural 404.
            data: McpSettingsUpdate — only provided fields are written.

        Returns:
            204 No Content once the listener matches the new settings.

        Raises:
            NotFoundError: If id is not the create sentinel — the singleton
                has no per-resource id, so nothing lives at a real id.
        """
        if not self.is_create(id):
            raise NotFoundError("Singleton settings record has no per-resource id")
        dto = cast(McpSettingsUpdate, data)
        if dto.enabled is not None:
            Setting.set("mcp_server_enabled", "true" if dto.enabled else "false")
        if dto.port is not None:
            Setting.set("mcp_server_port", str(dto.port))
        listener.reconcile()
        return "", 204
