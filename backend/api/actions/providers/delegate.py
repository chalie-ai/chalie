"""Delegate-provider role action — nested resource under /api/providers/delegate.

Covers:
- GET  /api/providers/delegate  → get  (resolved delegate provider + resolution source)
- POST /api/providers/delegate  → post (pin/clear the delegate provider; legacy
  PUT → POST, the established contract reshape — see api/endpoints/providers.py)

``source`` is ``'explicit'`` (pinned here) or ``'auto'`` (fell back to the
main provider) — unlike vision there is no ``'none'`` clear state: clearing
the pin always resolves back to the main provider (or ``'none'`` if no main
provider is selected either). Cache invalidation runs only on the
explicit-pin branch, matching the legacy resource exactly. Only the
sentinel/id-less form is meaningful — an id-addressed call 404s, matching the
discoverable-tools precedent (api/actions/mcp_clients/discoverable.py).
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse, NotFoundError
from api.request import Request
from api.request.provider_role import NullableProviderRef
from api.response.provider import Provider
from api.response.provider_role import ProviderRole
from services.provider_db_service import ProviderDbService
from services.provider_probe import invalidate_provider_cache


class ProviderDelegate(Action):
    """Action for reading/pinning the delegate (subagent) provider role."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get", "post"})
    request_dto: ClassVar[type[Request] | None] = NullableProviderRef
    response_dto = {
        "get": DocumentedResponse(ProviderRole),
        "post": DocumentedResponse(ProviderRole, extras=((404, "Provider not found"),)),
    }

    def slug(self) -> str:
        return "providers"

    def verb(self) -> str:
        return "delegate"

    def get(self, id: int | str) -> ResponseReturnValue:
        if not self.is_create(id):
            raise NotFoundError("Not found")
        status = ProviderDbService().get_delegate_provider_status()
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
            service.set_delegate_provider(None)
            status = service.get_delegate_provider_status()
            row = cast("dict[str, object] | None", status['provider'])
            return ProviderRole(
                provider=Provider.from_row(row) if row else None,
                source=cast(str, status['source']),
            ).single()

        row = service.get_provider_by_id(dto.provider_id)
        if not row:
            raise NotFoundError("Provider not found")

        service.set_delegate_provider(dto.provider_id)
        invalidate_provider_cache()
        return ProviderRole(provider=Provider.from_row(row), source='explicit').single()
