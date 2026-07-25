"""File preview action — nested resource under /api/files/preview.

Covers:
- GET  /api/files/preview/<relpath>  → get (inline browser preview)

Serves attachment bytes inline for files linked to transcripts via the
transcript_files table.
"""

from __future__ import annotations

import mimetypes
import os
from typing import ClassVar, cast

from flask import send_file
from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from exceptions import NotFoundError
from services.file_mapper_service import FileMapperService


class FilesPreview(Action):
    """Action streaming file bytes for inline browser preview."""

    id_type: ClassVar[type[int] | type[str]] = str
    id_converter: ClassVar[str] = "path"
    response_dto = {"get": DocumentedResponse(not_found=True)}

    def slug(self) -> str:
        return "files"

    def verb(self) -> str:
        return "preview"

    def get(self, id: int | str) -> ResponseReturnValue:
        if self.is_create(id):
            raise NotFoundError("Not found")

        relpath = cast(str, id)
        documents_root = FileMapperService.get_documents_path()
        full_path = os.path.realpath(os.path.join(documents_root, relpath))

        # One message for escape and absence alike — never reveal which.
        if not FileMapperService.validate_document_path(full_path) or not os.path.isfile(full_path):
            raise NotFoundError("File not found")

        mime = mimetypes.guess_type(full_path)[0] or "application/octet-stream"
        basename = os.path.basename(full_path)

        return send_file(
            full_path,
            mimetype=mime,
            as_attachment=False,
            download_name=basename,
        )