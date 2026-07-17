"""Document download action — nested resource under /api/documents/download.

Covers:
- GET  /api/documents/download/<id>  → get (download original file as attachment)

The success body is binary (no JSON DTO); only the 404 is documented via
``not_found=True``. The 200 shows as 204/No-Content in swagger — an accepted
cosmetic gap for binary downloads.
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask import send_file
from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from exceptions import NotFoundError
from services.document_service import DocumentService


class DocumentsDownload(Action):
    """Action downloading the original file for a document as an attachment."""

    id_type: ClassVar[type[int] | type[str]] = str
    response_dto = {"get": DocumentedResponse(not_found=True)}

    def slug(self) -> str:
        return "documents"

    def verb(self) -> str:
        return "download"

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
            as_attachment=True,
            download_name=cast(str, doc["original_name"]),
        )
