"""Vision-provider role action — nested resource under /api/providers/vision.

Covers:
- GET  /api/providers/vision  → get  (resolved vision provider + resolution source)
- POST /api/providers/vision  → post (pin/clear the vision provider; legacy PUT →
  POST, the established contract reshape — see api/endpoints/providers.py)

``source`` is ``'explicit'`` (pinned here), ``'auto'`` (fell back to the main
provider, which must itself support vision), or ``'none'``. Cache invalidation
runs only on the explicit-pin branch, matching the legacy resource exactly —
clearing the pin does not invalidate (the cache holds provider rows, not role
pins, so a clear has nothing to invalidate). Only the sentinel/id-less form is
meaningful — an id-addressed call 404s, matching the discoverable-tools
precedent (api/actions/mcp_clients/discoverable.py).
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from exceptions import EndpointError, NotFoundError
from api.request import Request
from api.request.provider_role import NullableProviderRef
from api.response.provider import Provider
from api.response.provider_role import ProviderRole
from services.provider_db_service import ProviderDbService
from services.provider_probe import invalidate_provider_cache


class ProviderVision(Action):
    """Action for reading/pinning the vision-capable provider role."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get", "post"})
    request_dto: ClassVar[type[Request] | None] = NullableProviderRef
    response_dto = {
        "get": DocumentedResponse(ProviderRole),
        "post": DocumentedResponse(
            ProviderRole,
            extras=((400, "Provider does not support vision"), (404, "Provider not found")),
        ),
    }

    def get(self, id: int | str) -> ResponseReturnValue:
        if not self.is_create(id):
            raise NotFoundError("Not found")
        status = ProviderDbService().get_vision_provider_status()
        row = cast("dict[str, object] | None", status['provider'])
        return ProviderRole(
            provider=Provider.from_row(row) if row else None,
            source=cast(str, status['source']),
        ).single()

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        if not self.is_create(id):
            raise NotFoundError("Not found")
        dto = cast(NullableProviderRef, data)
        service = ProviderDbService()

        if dto.provider_id is None:
            service.set_vision_provider(None)
            return ProviderRole(provider=None, source='none').single()

        row = service.get_provider_by_id(dto.provider_id)
        if not row:
            raise NotFoundError("Provider not found")
        if not row.get('supports_vision'):
            raise EndpointError("Provider does not support vision")

        service.set_vision_provider(dto.provider_id)
        invalidate_provider_cache()
        return ProviderRole(provider=Provider.from_row(row), source='explicit').single()
