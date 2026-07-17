"""Document supersede action — nested resource under /api/documents/supersede.

Covers:
- POST /api/documents/supersede/<id>  → mark the new document as superseding a prior one
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from api.request.document import SupersedeRequest
from api.response.document import SupersedeResponse
from exceptions import NotFoundError
from services.document_service import DocumentService


class DocumentsSupersede(Action):
    """Action marking the new document as superseding a prior one."""

    id_type: ClassVar[type[int] | type[str]] = str
    request_dto = SupersedeRequest
    response_dto = {
        "post": DocumentedResponse(
            SupersedeResponse,
            extras=((404, "New/Old document not found"),),
        ),
    }

    def slug(self) -> str:
        return "documents"

    def verb(self) -> str:
        return "supersede"

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        if self.is_create(id):
            raise NotFoundError("Not found")
        dto = cast(SupersedeRequest, data)
        svc = DocumentService()
        if not svc.get_document(cast(str, id)):
            raise NotFoundError("New document not found")
        if not svc.get_document(dto.old_id):
            raise NotFoundError("Old document not found")
        svc.set_supersedes(cast(str, id), dto.old_id)
        svc.soft_delete(dto.old_id)
        return SupersedeResponse(ok=True, supersedes_id=dto.old_id).single()
