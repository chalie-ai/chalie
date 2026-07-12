"""Document preview action — nested resource under /api/documents/preview.

Covers:
- GET  /api/documents/preview/<id>  → get (inline browser preview)

Identical to download except ``as_attachment=False`` for inline rendering.
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask import send_file
from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from exceptions import NotFoundError
from services.document_service import DocumentService


class DocumentsPreview(Action):
    """Action streaming the original file for inline browser preview."""

    id_type: ClassVar[type[int] | type[str]] = str
    response_dto = {"get": DocumentedResponse(not_found=True)}

    def slug(self) -> str:
        return "documents"

    def verb(self) -> str:
        return "preview"

    def get(self, id: int | str) -> ResponseReturnValue:
        if self.is_create(id):
            raise NotFoundError("Not found")

        import os

        svc = DocumentService()
        doc = svc.get_document(cast(str, id))
        if not doc:
            raise NotFoundError("Not found")

        full_path = DocumentService.resolve_file_path(doc)
        if full_path is None or not os.path.exists(full_path):
            raise NotFoundError("File not found")

        return send_file(
            full_path,
            mimetype=cast(str, doc["mime_type"]),
            as_attachment=False,
            download_name=cast(str, doc["original_name"]),
        )
