"""Document content action — nested resource under /api/documents/content.

Covers:
- GET /api/documents/content/<id>  → full extracted text reconstructed from artifacts
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.response.document import ArtifactContent, DocumentContent
from exceptions import NotFoundError
from services.document_service import DocumentService


class DocumentsContent(Action):
    """Action returning full document text reconstructed from data_graph artifacts."""

    id_type: ClassVar[type[int] | type[str]] = str
    response_dto = {"get": DocumentedResponse(DocumentContent)}

    def slug(self) -> str:
        return "documents"

    def verb(self) -> str:
        return "content"

    def get(self, id: int | str) -> ResponseReturnValue:
        if self.is_create(id):
            raise NotFoundError("Not found")
        svc = DocumentService()
        if not svc.get_document(cast(str, id)):
            raise NotFoundError("Not found")
        from models.document import DocumentRow

        fragments = DocumentRow.for_document_id(cast(str, id)).get()
        return DocumentContent(
            document_id=cast(str, id),
            total_artifacts=len(fragments),
            artifacts=[
                ArtifactContent(key=f.key, content=f.value or "") for f in fragments
            ],
        ).single()
