"""Document confirmation action — nested resource under /api/documents/confirm.

Covers:
- POST /api/documents/confirm/<id>  → confirm a document after synthesis review

The base auto-documents 404 only for get/delete handlers, so this post
declares its 404 and 400 explicitly via ``extras`` to keep swagger truthful.
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from exceptions import EndpointError, NotFoundError
from services.document_service import DocumentService

_NOT_FOUND = "Not found"


class DocumentsConfirm(Action):
    """Action to confirm a document after synthesis review."""

    id_type: ClassVar[type[int] | type[str]] = str
    response_dto = {
        "post": DocumentedResponse(
            extras=((404, _NOT_FOUND), (400, "Not awaiting confirmation")),
        )
    }

    def slug(self) -> str:
        return "documents"

    def verb(self) -> str:
        return "confirm"

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        if self.is_create(id):
            raise NotFoundError(_NOT_FOUND)
        svc = DocumentService()
        doc = svc.get_document(cast(str, id))
        if not doc:
            raise NotFoundError(_NOT_FOUND)
        if doc["status"] != "awaiting_confirmation":
            raise EndpointError("Document is not awaiting confirmation")
        svc.update_status(
            cast(str, id),
            "ready",
            chunk_count=cast(int, doc.get("chunk_count", 0)),
        )
        return "", 204
