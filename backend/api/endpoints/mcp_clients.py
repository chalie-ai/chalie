"""MCP clients endpoint — migrated from the legacy ``mcp_clients`` namespace.

Outbound MCP server management: Chalie connects OUT to remote MCP servers.
``headers`` is a write-only credential carrier — it lives on the request DTO
only and is never present on a response.

Covers the CRUD routes the legacy module exposed for the server collection:
- GET    /api/mcp-clients/all   → get_all (paginated listing)
- POST   /api/mcp-clients/<id>  → post (create when id=-1, update otherwise)
- DELETE /api/mcp-clients/<id>  → delete

The legacy module had no get-by-id route, so ``get`` is intentionally left
unoverridden (stays a 405 stub). The legacy PUT /api/mcp-clients/<id> is
mapped to POST with id != -1. ``/discoverable``, ``/<id>/test``, and
``/<id>/tools`` move to Action verbs under ``api/actions/mcp_clients/``.

Every implemented handler is cookie-only (``cookie_only_methods``), matching
the legacy module's blanket ``require_session`` on every route — a bearer
wrapper must never manage outbound MCP credentials on its own behalf.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.endpoint import DocumentedResponse, Endpoint
from exceptions import EndpointError, NotFoundError
from api.request import Request
from api.request.mcp_server import McpServerRequest
from api.response.mcp_server import McpServer
from services.mcp_client_service import McpClientService

logger = logging.getLogger(__name__)


class McpClients(Endpoint):
    """CRUD endpoint for outbound MCP server connections."""

    id_type: ClassVar[type[int] | type[str]] = str
    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get_all", "post", "delete"})
    request_dto: ClassVar[type[Request] | None] = McpServerRequest
    response_dto = {
        "get_all": DocumentedResponse(McpServer, listing=True),
        "post": DocumentedResponse(McpServer, extras=((404, "Server not found"),)),
    }

    def slug(self) -> str:
        return "mcp-clients"

    def _service(self) -> McpClientService:
        return McpClientService()

    def get_all(self, page: int = 1, limit: int = 20) -> ResponseReturnValue:
        """List all outbound MCP servers, oldest-created first."""
        items = [self._to_response(s) for s in self._service().list_servers()]
        total = len(items)
        start = (page - 1) * limit
        return McpServer.listing(items[start : start + limit], page=page, limit=limit, total=total)

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Add a server (create sentinel) or apply a partial update.

        Args:
            id: ``-1`` to create, otherwise the server id to update.
            data: McpServerRequest — create requires name+host; update applies
                only the non-None fields.

        Returns:
            McpServer, 201 on create / 200 on update.

        Raises:
            EndpointError: name/host missing on create.
            NotFoundError: server id not found on update.
        """
        dto = cast(McpServerRequest, data)
        if self.is_create(id):
            return self._create(dto)
        return self._update(cast(str, id), dto)

    def delete(self, id: int | str) -> ResponseReturnValue:
        """Delete a server and purge its tools + policy rows.

        Raises:
            NotFoundError: If server not found.

        Returns:
            204 No Content on success (no body).
        """
        try:
            self._service().delete_server(cast(str, id))
        except LookupError:
            raise NotFoundError("Server not found") from None
        return "", 204

    def _create(self, dto: McpServerRequest) -> ResponseReturnValue:
        if dto.name is None:
            raise EndpointError("name is required")
        if dto.host is None:
            raise EndpointError("host is required")

        svc = self._service()
        server = svc.add_server(
            name=dto.name,
            host=dto.host,
            headers=dto.headers or {},
            enabled=dto.enabled if dto.enabled is not None else True,
        )
        try:
            sync_result = svc.ping_and_sync(cast(str, server["id"]))
            if sync_result.get("reachable"):
                svc.embed_server_tools(cast(str, server["id"]))
            refreshed = svc.get_server(cast(str, server["id"]))
            if refreshed:
                server = refreshed
        except Exception as exc:
            # Deliberate swallow — initial ping failure must not fail creation.
            logger.warning("[MCP CLIENTS] Initial ping failed for %r: %s", dto.name, exc)
        return self._to_response(server).single(), 201

    def _update(self, server_id: str, dto: McpServerRequest) -> ResponseReturnValue:
        updates: dict[str, object] = {k: v for k, v in dto.model_dump().items() if v is not None}
        try:
            server = self._service().update_server(server_id, updates)
        except LookupError:
            raise NotFoundError("Server not found") from None
        return self._to_response(server).single()

    def _to_response(self, row: dict[str, object]) -> McpServer:
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
