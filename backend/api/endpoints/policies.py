"""Policies endpoint — migrated from the legacy policies namespace.

Covers the CRUD routes that the legacy module exposed:
- GET    /api/policies/all        → get_all (list all policies)
- POST   /api/policies/<id>       → post (upsert when id=-1, not allowed otherwise)

The legacy PUT /api/policies is mapped to POST with id=-1 (create sentinel).
"""

from __future__ import annotations

from typing import ClassVar, cast

import logging

from flask.typing import ResponseReturnValue

from api.endpoint import DocumentedResponse, Endpoint
from api.request import Request
from api.request.policies import PolicyUpsertRequest
from api.response.policies import PolicyResponse
from api.response.response import Response
from services.mcp_client_service import McpClientService, humanize_segment
from services.policy_manager import PolicyManager

logger = logging.getLogger(__name__)


class PoliciesEndpoint(Endpoint):
    """REST endpoints for per-action permission control."""

    id_type: ClassVar[type[str]] = str
    request_dto = PolicyUpsertRequest
    response_dto = {"get_all": DocumentedResponse(PolicyResponse, listing=True)}

    def slug(self) -> str:
        return "policies"

    def _service(self) -> PolicyManager:
        return PolicyManager()

    def get_all(self, page: int = 1, limit: int = 20) -> ResponseReturnValue:
        """List all non-internal policies with display labels."""
        rows = self._service().get_all()
        enriched = self._tag_display_rows(rows)
        total = len(enriched)
        start = (page - 1) * limit
        end = start + limit
        page_rows = enriched[start:end]
        dtos = [PolicyResponse(**row) for row in page_rows]
        return Response.listing(dtos, page=page, limit=limit, total=total)

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Upsert a single policy cell. Only accepts create (id=-1)."""
        if not self.is_create(id):
            return self.not_allowed()
        dto = cast(PolicyUpsertRequest, data)
        self._service().upsert(dto.channel, dto.permission, dto.setting)
        return "", 204

    def _tag_display_rows(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        """Annotate every row with display label and group (MCP-aware).

        MCP resolution is guarded: a misconfigured/absent MCP store must never
        break the page — MCP rows then fall through to the native branch.
        """
        mcp_labels: dict[str, dict[str, str]] = {}
        try:
            mcp_labels = McpClientService().label_mcp_permissions(
                [r["permission"] for r in rows]
            )
        except Exception as exc:  # noqa: BLE001 — display enrichment must not 500 the page
            logger.warning("MCP row tagging skipped: %s", exc)
        for r in rows:
            info = mcp_labels.get(r["permission"])
            if info:
                r["group"] = info["group"]
                r["label"] = info["label"]
            else:
                _base, _, action = r["permission"].partition(".")
                r["label"] = humanize_segment(action or _base)
        return rows
