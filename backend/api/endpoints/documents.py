"""Documents endpoint — migrated from the legacy ``documents`` namespace.

Provides the read-side CRUD routes for the document collection:
- GET    /api/documents/all   → get_all (paginated listing)
- GET    /api/documents/<id>  → get (metadata + artifact previews)
- DELETE /api/documents/<id>  → delete (soft delete)

The legacy module exposed many more routes (upload, search, content, download,
preview, classify, groups, restore, purge, confirm, augment, supersede); those
stay in ``api/documents.py`` until the orchestrator deletes it in a later step.
This file replaces only the list/get/delete trio.

No create/update body is accepted — ``post`` stays the base 405 stub because
the legacy module had no ``POST /api/documents/<id>``.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar, TypeAlias, cast

from flask import request
from flask.typing import ResponseReturnValue

from api.endpoint import DocumentedResponse, Endpoint
from api.response.document import ArtifactPreview, Document, DocumentDetail
from exceptions import NotFoundError
from services.document_service import DocumentService

_STR_OR_NONE: TypeAlias = "str | None"


class Documents(Endpoint):
    """CRUD endpoint for the document collection."""

    id_type: ClassVar[type[int] | type[str]] = str
    response_dto = {
        "get_all": DocumentedResponse(Document, listing=True),
        "get": DocumentedResponse(DocumentDetail, not_found=True),
        "delete": DocumentedResponse(not_found=True),
    }

    def slug(self) -> str:
        return "documents"

    def _service(self) -> DocumentService:
        return DocumentService()

    def get_all(self, page: int = 1, limit: int = 20) -> ResponseReturnValue:
        """List all documents, optionally including soft-deleted ones."""
        include_deleted = (
            request.args.get("include_deleted", "false").lower() == "true"
        )
        rows = self._service().get_all_documents(include_deleted=include_deleted)
        docs = [self._to_response(row) for row in rows]
        start = (page - 1) * limit
        page_slice = docs[start : start + limit]
        return Document.listing(page_slice, page=page, limit=limit, total=len(docs))

    def get(self, id: int | str) -> ResponseReturnValue:
        """Get document metadata + first N data_graph artifact previews."""
        doc = self._service().get_document(cast(str, id))
        if not doc:
            raise NotFoundError("Not found")
        from models.document import DocumentRow

        fragments = DocumentRow.for_document_id(cast(str, id)).limit(5).get()
        return DocumentDetail(
            **self._to_response(doc).model_dump(mode="python"),
            artifacts=[
                ArtifactPreview(key=f.key, preview=(f.value or "")[:200])
                for f in fragments
            ],
        ).single()

    def delete(self, id: int | str) -> ResponseReturnValue:
        """Soft delete a document.

        Raises:
            NotFoundError: If the document is missing or already gone.

        Returns:
            204 No Content on success (no body).
        """
        if not self._service().soft_delete(cast(str, id)):
            raise NotFoundError("Not found")
        return "", 204

    def _to_response(self, row: dict[str, object]) -> Document:
        return Document(
            id=cast(str, row["id"]),
            original_name=cast(str, row["original_name"]),
            mime_type=cast(str, row["mime_type"]),
            file_size_bytes=cast(int, row["file_size_bytes"]),
            file_hash=cast(str, row["file_hash"]),
            page_count=cast("int | None", row.get("page_count")),
            status=cast(str, row["status"]),
            error_message=cast(_STR_OR_NONE, row.get("error_message")),
            chunk_count=cast("int | None", row.get("chunk_count")),
            source_type=cast(str, row["source_type"]),
            tags=cast("list[str]", row.get("tags")),
            summary=cast(_STR_OR_NONE, row.get("summary")),
            extracted_metadata=cast("dict[str, object]", row.get("extracted_metadata")),
            supersedes_id=cast(_STR_OR_NONE, row.get("supersedes_id")),
            language=cast(_STR_OR_NONE, row.get("language")),
            fingerprint=cast(_STR_OR_NONE, row.get("fingerprint")),
            doc_category=cast(_STR_OR_NONE, row.get("doc_category")),
            doc_project=cast(_STR_OR_NONE, row.get("doc_project")),
            doc_date=cast(_STR_OR_NONE, row.get("doc_date")),
            meta_locked=cast("bool", row.get("meta_locked")),
            watched_folder_id=cast(_STR_OR_NONE, row.get("watched_folder_id")),
            created_at=cast(datetime, row["created_at"]),
            updated_at=cast(datetime, row["updated_at"]),
            deleted_at=cast("datetime | None", row.get("deleted_at")),
            purge_after=cast("datetime | None", row.get("purge_after")),
        )
