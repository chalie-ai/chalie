"""Provider catalog action — nested resource under /api/providers/catalog.

Covers:
- GET /api/providers/catalog  → get (curated presets for the setup wizard)

Only the sentinel/id-less form is meaningful — an id-addressed call 404s,
matching the discoverable-tools precedent (api/actions/mcp_clients/discoverable.py).
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from exceptions import NotFoundError
from api.response.provider_catalog import CatalogEntry
from services.provider_catalog_service import get_catalog


class ProviderCatalog(Action):
    """Action listing the curated provider presets for the setup wizard."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get"})
    response_dto = {"get": DocumentedResponse(CatalogEntry, listing=True)}

    def get(self, id: int | str) -> ResponseReturnValue:
        if not self.is_create(id):
            raise NotFoundError("Not found")
        items = [
            CatalogEntry(
                id=cast(str, entry["id"]),
                name=cast(str, entry["name"]),
                platform=cast(str, entry["platform"]),
                host=cast(str, entry["host"]),
                needs_key=cast(bool, entry["needs_key"]),
            )
            for entry in get_catalog()
        ]
        return CatalogEntry.listing(items, page=1, limit=max(len(items), 1), total=len(items))
