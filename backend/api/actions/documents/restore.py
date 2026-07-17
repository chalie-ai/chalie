"""Document restore action — nested resource under /api/documents/restore.

Covers:
- POST /api/documents/restore/<id>  → undo soft delete

The base auto-documents 404 only for get/delete handlers, so this post
declares its 404 explicitly via ``extras`` to keep swagger truthful.
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from exceptions import NotFoundError
from services.document_service import DocumentService


class DocumentsRestore(Action):
    """Action to restore a previously soft-deleted document."""

    id_type: ClassVar[type[int] | type[str]] = str
    response_dto = {"post": DocumentedResponse(extras=((404, "Not found or not deleted"),))}

    def slug(self) -> str:
        return "documents"

    def verb(self) -> str:
        return "restore"

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        if self.is_create(id):
            raise NotFoundError("Not found")
        if not DocumentService().restore(cast(str, id)):
            raise NotFoundError("Not found or not deleted")
        return "", 204
