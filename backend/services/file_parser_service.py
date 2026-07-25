"""FileParserService — ingest a file into the documents store.

Replaces the document-pipeline plumbing used by screenshots and vision.
Extracts content first (by MIME), then copies the file to the documents
store and indexes it.  A failure raises ``ValueError`` — no file is copied
and nothing is indexed on failure.
"""

from __future__ import annotations

import logging
import mimetypes as _mimetypes
import os as _os
import shutil as _shutil

from services.file_index_service import FileIndexService
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)


class FileParserService:
    """Ingest a file into the documents store with extracted text content."""

    def ingest(
        self,
        src_path: str,
        *,
        name: str | None = None,
        subdir: str | None = None,
    ) -> tuple[str, str]:
        """Ingest a file into the documents store.

        Args:
            src_path: Path to the source file to ingest.
            name: Optional stored filename. Defaults to the source basename.
            subdir: Optional subdirectory under the documents root.

        Returns:
            A tuple of (saved_absolute_path, extracted_text).

        Raises:
            ValueError: If the file is missing, extraction fails, or the
                extracted content is empty.
        """
        # Validate source file exists.
        if not src_path or not _os.path.isfile(src_path):
            raise ValueError(f"No file at path: {src_path}")

        # Extract content FIRST by MIME (before any copy).
        mime = _mimetypes.guess_type(src_path)[0] or "application/octet-stream"
        try:
            if mime.startswith("image/"):
                from services.image_description import ImageDescription, RICH_INDEX_PROMPT

                text = ImageDescription(src_path, RICH_INDEX_PROMPT).get_value()
            else:
                from services.text_reader import TextReader

                text = TextReader(src_path).get_value()
        except Exception as exc:
            raise ValueError(f"Extraction failed for {src_path!r}: {exc}") from exc

        if not text or not text.strip():
            raise ValueError(f"No content could be extracted from {src_path!r}.")

        # Sanitise the stored name.
        from services.filename_utils import safe_filename

        stored_name = safe_filename(name or _os.path.basename(src_path)) or "unnamed_file"
        # subdir must survive sanitisation unchanged — rejects traversal
        # segments and separators rather than silently rewriting them.
        if subdir is not None and subdir != safe_filename(subdir):
            raise ValueError(f"Invalid subdir: {subdir!r}")

        # Copy the file to the documents store.
        saved_path = self._copy_to_store(src_path, stored_name, subdir)

        # Index the content.
        FileIndexService().upsert_content(saved_path, text)

        return saved_path, text

    def _copy_to_store(
        self, src_path: str, stored_name: str, subdir: str | None
    ) -> str:
        """Copy *src_path* to the documents store at *stored_name* under *subdir*.

        Returns the absolute path of the copied file.  On filename collision
        appends ``_1``, ``_2``, … before the extension.
        """
        base_dir = FileMapperService.get_documents_path(subdir) if subdir else FileMapperService.get_documents_path()
        _os.makedirs(base_dir, exist_ok=True)

        stem, ext = _os.path.splitext(stored_name)
        candidate = _os.path.join(base_dir, stored_name)
        counter = 1
        while _os.path.exists(candidate):
            candidate = _os.path.join(base_dir, f"{stem}_{counter}{ext}")
            counter += 1

        _shutil.copyfile(src_path, candidate)
        return candidate
