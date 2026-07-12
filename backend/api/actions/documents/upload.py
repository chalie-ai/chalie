"""Document upload action — id-less POST at /api/documents/upload.

Covers:
- POST /api/documents/upload               → post (multipart file upload)

Upload validates the extension and size via typed ``EndpointError`` raises
(400), surfaces genuine ingest failures as ``RuntimeError`` (500, past
``_guard``), and always cleans up the temp file in a ``finally`` block.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import ClassVar, cast

from flask import request
from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from api.response.document import DuplicateRef, UploadResponse
from exceptions import EndpointError, NotFoundError
from services.document_service import DocumentService

MAX_FILE_SIZE = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md",
    ".css", ".csv", ".xml", ".json",
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rs", ".rb",
    ".jpg", ".jpeg", ".png", ".webp", ".gif",
}
_FALLBACK_DOCUMENT_NAME = "unnamed_document"


class DocumentsUpload(Action):
    """Action uploading and ingesting a document file."""

    id_type: ClassVar[type[int] | type[str]] = str
    _post_may_create: ClassVar[bool] = True
    response_dto = {
        "post": DocumentedResponse(
            UploadResponse,
            extras=(
                (400, "No file provided"),
                (400, "File has no extension"),
                (400, "File type not supported"),
                (400, "File exceeds 50MB limit"),
                (400, "File is empty"),
            ),
        )
    }

    def slug(self) -> str:
        return "documents"

    def verb(self) -> str:
        return "upload"

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        if not self.is_create(id):
            raise NotFoundError("Not found")

        from services.filename_utils import safe_filename
        from services.tmp_storage import new_tmp_path

        upload = request.files.get("file")
        if upload is None:
            raise EndpointError("No file provided")

        original_name = safe_filename(upload.filename or "") or _FALLBACK_DOCUMENT_NAME
        ext = os.path.splitext(original_name)[1].lower()
        if not ext:
            raise EndpointError("File has no extension")
        if ext not in ALLOWED_EXTENSIONS:
            raise EndpointError(f"File type '{ext}' is not supported")

        tmp_path = new_tmp_path(f"{uuid.uuid4().hex[:8]}_{original_name}")
        try:
            upload.save(tmp_path)

            size = os.path.getsize(tmp_path)
            if size > MAX_FILE_SIZE:
                raise EndpointError(f"File exceeds {MAX_FILE_SIZE // 1024 // 1024}MB limit")
            if size == 0:
                raise EndpointError("File is empty")

            from abilities.document import ingest_file

            svc = DocumentService()
            result = ingest_file(svc, tmp_path, name=original_name)
            if result.get("error"):
                raise RuntimeError(f"Upload failed: {result['error']}")

            doc_id = cast(str, result["id"])
            file_hash = cast(str, result["hash"])
            duplicates = svc.find_duplicates(file_hash, None, 0, exclude_id=doc_id)

            return UploadResponse(
                id=doc_id,
                original_name=cast(str, result["name"]),
                status=cast(str, result.get("status") or "pending"),
                file_size=cast(int, result["size"]),
                file_hash=file_hash,
                duplicates=[
                    DuplicateRef(
                        id=cast(str, d["id"]),
                        original_name=cast(str, d["original_name"]),
                        match_type=cast(str, d["match_type"]),
                        created_at=cast(datetime | None, d.get("created_at")),
                    )
                    for d in duplicates
                ] or None,
            ).single(), 201
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
