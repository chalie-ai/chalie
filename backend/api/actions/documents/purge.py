"""Document purge action — nested resource under /api/documents/purge.

Covers:
- DELETE /api/documents/purge/<id>  → hard delete a document

The base auto-documents 404 for delete handlers, so ``not_found=True`` on the
``DocumentedResponse`` is enough to keep swagger truthful here.
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from exceptions import NotFoundError
from services.document_service import DocumentService


class DocumentsPurge(Action):
    """Action to immediately hard delete a document."""

    id_type: ClassVar[type[int] | type[str]] = str
    response_dto = {"delete": DocumentedResponse(not_found=True)}

    def slug(self) -> str:
        return "documents"

    def verb(self) -> str:
        return "purge"

    def delete(self, id: int | str) -> ResponseReturnValue:
        if self.is_create(id):
            raise NotFoundError("Not found")
        if not DocumentService().hard_delete(cast(str, id)):
            raise NotFoundError("Not found")
        return "", 204
