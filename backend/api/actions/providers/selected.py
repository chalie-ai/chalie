"""Selected (main) provider action — nested resource under /api/providers/selected.

Covers:
- GET  /api/providers/selected  → get  (currently active main provider)
- POST /api/providers/selected  → post (pin the main provider; legacy PUT →
  POST, the established contract reshape — see api/endpoints/providers.py)

Reshaped to the ProviderRole{provider, source} envelope shared with vision/
delegate for a uniform role-action wire shape — legacy returned a bare
Provider|null here. ``source`` is always ``'explicit'`` once set, ``'none'``
when unset (selected has no auto-fallback, unlike vision/delegate). Only the
sentinel/id-less form is meaningful — an id-addressed call 404s, matching the
discoverable-tools precedent (api/actions/mcp_clients/discoverable.py).
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from exceptions import NotFoundError
from api.request import Request
from api.request.provider_role import ProviderRef
from api.response.provider import Provider
from api.response.provider_role import ProviderRole
from services.provider_db_service import ProviderDbService
from services.provider_probe import invalidate_provider_cache


class ProviderSelected(Action):
    """Action for reading/pinning the active main provider."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get", "post"})
    request_dto: ClassVar[type[Request] | None] = ProviderRef
    response_dto = {
        "get": DocumentedResponse(ProviderRole),
        "post": DocumentedResponse(ProviderRole, extras=((404, "Provider not found"),)),
    }

    def slug(self) -> str:
        return "providers"

    def verb(self) -> str:
        return "selected"

    def get(self, id: int | str) -> ResponseReturnValue:
        if not self.is_create(id):
            raise NotFoundError("Not found")
        row = ProviderDbService().get_selected_provider()
        return ProviderRole(
            provider=Provider.from_row(row) if row else None,
            source="explicit" if row else "none",
        ).single()

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        if not self.is_create(id):
            raise NotFoundError("Not found")
        dto = cast(ProviderRef, data)
        service = ProviderDbService()
        row = service.get_provider_by_id(dto.provider_id)
        if not row:
            raise NotFoundError("Provider not found")
        service.set_selected_provider(dto.provider_id)
        invalidate_provider_cache()
        return ProviderRole(provider=Provider.from_row(row), source="explicit").single()
