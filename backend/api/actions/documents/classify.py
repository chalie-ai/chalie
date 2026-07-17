"""Document classification action — nested resource under /api/documents/classify.

Covers:
- POST /api/documents/classify/<id>  → partial classification update

The base auto-documents 404 only for get/delete handlers, so this post
declares its 404 explicitly via ``extras`` to keep swagger truthful.
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from api.request.document import ClassifyRequest
from exceptions import NotFoundError
from services.document_service import DocumentService

_NOT_FOUND = "Not found"


class DocumentsClassify(Action):
    """Action for partial classification update of a document."""

    id_type: ClassVar[type[int] | type[str]] = str
    request_dto = ClassifyRequest
    response_dto = {"post": DocumentedResponse(extras=((404, _NOT_FOUND),))}

    def slug(self) -> str:
        return "documents"

    def verb(self) -> str:
        return "classify"

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        if self.is_create(id):
            raise NotFoundError(_NOT_FOUND)
        dto = cast(ClassifyRequest, data)
        svc = DocumentService()
        if not svc.get_document(cast(str, id)):
            raise NotFoundError(_NOT_FOUND)
        svc.update_classification(
            cast(str, id),
            category=dto.category,
            project=dto.project,
            doc_date=dto.date,
            lock=True,
        )
        return "", 204
