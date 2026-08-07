"""Providers endpoint — migrated from the legacy ``providers`` namespace.

LLM provider configuration CRUD. Covers the routes the legacy module exposed
for the provider collection:
- GET    /api/providers/all   → get_all (paginated listing)
- GET    /api/providers/<id>  → get
- POST   /api/providers/<id>  → post (create when id=-1, update otherwise)
- DELETE /api/providers/<id>  → delete

The legacy PUT /api/providers/<id> is mapped to POST with id != -1 (the same
contract reshape as every other migrated CRUD group). ``/catalog``,
``/list-models``, ``/test``, ``/selected``, ``/vision``, and ``/delegate``
move to Action verbs under ``api/actions/providers/``.

Every implemented handler is cookie-only (``cookie_only_methods``), matching
the legacy module's blanket ``require_session`` on every route.
"""

from __future__ import annotations

import sqlite3
from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.endpoint import DocumentedResponse, Endpoint
from exceptions import EndpointError, NotFoundError
from api.request import Request
from api.request.provider import ProviderRequest
from api.response.provider import Provider
from services.provider_db_service import ProviderDbService
from services.provider_probe import DUPLICATE_NAME_MSG, ProviderProbe

_PROVIDER_NOT_FOUND = "Provider not found"


class Providers(Endpoint):
    """CRUD endpoint for LLM provider configuration."""

    id_type: ClassVar[type[int] | type[str]] = int
    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get_all", "get", "post", "delete"})
    request_dto: ClassVar[type[Request] | None] = ProviderRequest
    response_dto = {
        "get_all": DocumentedResponse(Provider, listing=True),
        "get": DocumentedResponse(Provider),
        "post": DocumentedResponse(
            Provider,
            extras=(
                (404, "Provider not found"),
                (409, "A provider with that name already exists"),
            ),
        ),
        "delete": DocumentedResponse(
            extras=((409, "Provider is in use"),), not_found=False
        ),  # idempotent: 204 even when already gone
    }

    def _service(self) -> ProviderDbService:
        return ProviderDbService()

    def get_all(self, page: int = 1, limit: int = 20) -> ResponseReturnValue:
        """List every configured provider (api_key never included)."""
        items = [Provider.from_row(row) for row in self._service().list_providers_summary()]
        total = len(items)
        start = (page - 1) * limit
        return Provider.listing(items[start : start + limit], page=page, limit=limit, total=total)

    def get(self, id: int | str) -> ResponseReturnValue:
        """Fetch one provider by id (api_key never included)."""
        row = self._service().get_provider_by_id(cast(int, id))
        if row is None:
            raise NotFoundError(_PROVIDER_NOT_FOUND)
        return Provider.from_row(row).single()

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Create (id -1) or update a provider.

        Raises:
            EndpointError: name/platform missing on create, or a validation
                failure surfaced by the provider service (safe-listed message).
            NotFoundError: provider id not found on update.

        Returns:
            Provider, 201 on create / 200 on update; 409 on a duplicate name.
        """
        dto = cast(ProviderRequest, data)
        if self.is_create(id):
            return self._create(dto)
        return self._update(cast(int, id), dto)

    def delete(self, id: int | str) -> ResponseReturnValue:
        """Delete a provider (idempotent — a missing id is a no-op, matching the service).

        Returns:
            204 No Content on success; 409 with the in-use message when the
            provider still holds a role (selected/vision/delegate).
        """
        try:
            self._service().delete_provider(cast(int, id))
        except ValueError as exc:
            return Provider.failure(ProviderProbe.safe_validation_msg(exc)), 409
        ProviderProbe.invalidate_provider_cache()
        return "", 204

    def _create(self, dto: ProviderRequest) -> ResponseReturnValue:
        if dto.name is None:
            raise EndpointError("name is required")
        if dto.platform is None:
            raise EndpointError("platform is required")
        try:
            # exclude_none: an explicit null on create means "use the default"
            # (the service fills absent keys), never "store SQL NULL".
            row = cast(
                "dict[str, object]",
                self._service().create_provider(dto.model_dump(mode="json", exclude_none=True)),
            )
        except ValueError as exc:
            return Provider.failure(ProviderProbe.safe_validation_msg(exc)), 400
        except sqlite3.IntegrityError:
            return Provider.failure(DUPLICATE_NAME_MSG), 409
        ProviderProbe.invalidate_provider_cache()
        return Provider.from_row(row).single(), 201

    def _update(self, provider_id: int, dto: ProviderRequest) -> ResponseReturnValue:
        try:
            row = self._service().update_provider(provider_id, dto.model_dump(mode="json", exclude_unset=True))
        except ValueError as exc:
            return Provider.failure(ProviderProbe.safe_validation_msg(exc)), 400
        except sqlite3.IntegrityError:
            return Provider.failure(DUPLICATE_NAME_MSG), 409
        if row is None:
            raise NotFoundError(_PROVIDER_NOT_FOUND)
        ProviderProbe.invalidate_provider_cache()
        return Provider.from_row(row).single()
