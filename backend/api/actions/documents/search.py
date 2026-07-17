"""Document search action — id-less resource under /api/documents/search.

Covers:
- GET /api/documents/search  → semantic search across document artifacts
- GET /api/documents/search/<id>  → 404 (search takes no id)
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request.document import SearchQuery
from api.response.document import SearchResponse, SearchResult
from exceptions import NotFoundError


class DocumentsSearch(Action):
    """Action for semantic search across document artifacts in data_graph."""

    id_type: ClassVar[type[int] | type[str]] = str
    response_dto = {
        "get": DocumentedResponse(
            SearchResponse, extras=((422, "Validation failed"),)
        ),
    }

    def slug(self) -> str:
        return "documents"

    def verb(self) -> str:
        return "search"

    def get(self, id: int | str) -> ResponseReturnValue:
        if not self.is_create(id):
            raise NotFoundError("Not found")
        from flask import request

        dto = SearchQuery.model_validate(request.args.to_dict())
        from services.memory_recall_service import recall

        results = recall(dto.q, kinds=["document"], limit=dto.limit)

        return SearchResponse(
            results=[_search_result(row) for row in results],
            query=dto.q,
        ).single()


def _search_result(row: dict[str, object]) -> SearchResult:
    """Project a recall row onto a :class:`SearchResult` DTO."""
    source = cast(str, row.get("source", "") or "")
    return SearchResult(
        document_id=source.split(":", 1)[1] if source.startswith("document:") else "",
        key=cast(str, row.get("key", "")),
        content=cast(str, row.get("value", "")),
        source=source,
    )
