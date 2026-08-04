"""FileParserService — ingest a file into the documents store.

Provides three primitives split by concern:

* :meth:`_extract` pulls readable text out of a file (MIME-aware: image
  descriptions via vision, otherwise raw text via a text reader);
* :meth:`place` copies a source file into the documents store under a
  chosen basename;
* :meth:`index` extracts text from a previously-placed file and writes it
  to the file-index store.

:meth:`ingest` composes all three in the documented order (extract -> copy ->
index) so callers get both the stored path and extracted text. Failure raises
``ValueError`` — no file is copied and nothing is indexed on failure.
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

    def _extract(self, path: str) -> str:
        """Return extracted text for *path*, selected by MIME.

        Image sources route through :class:`ImageDescription`; everything else
        uses :class:`TextReader`. Failures and empty extractions both raise
        ``ValueError``.
        """
        mime = _mimetypes.guess_type(path)[0] or "application/octet-stream"
        try:
            if mime.startswith("image/"):
                from services.image_description import ImageDescription, RICH_INDEX_PROMPT

                text = ImageDescription(path, RICH_INDEX_PROMPT).get_value()
            else:
                from services.text_reader import TextReader

                text = TextReader(path).get_value()
        except Exception as exc:
            raise ValueError(f"Extraction failed for {path!r}: {exc}") from exc

        if not text or not text.strip():
            raise ValueError(f"No content could be extracted from {path!r}.")
        return text

    def place(
        self,
        src_path: str,
        *,
        name: str | None = None,
        subdir: str | None = None,
    ) -> str:
        """Copy *src_path* into the documents store and return its absolute path.

        Validates the source exists, derives a safe stored name, and rejects
        traversal segments in *subdir*. No extraction or indexing occurs.
        """
        if not src_path or not _os.path.isfile(src_path):
            raise ValueError(f"No file at path: {src_path}")

        from services.filename_utils import safe_filename

        stored_name = safe_filename(name or _os.path.basename(src_path)) or "unnamed_file"
        if subdir is not None and subdir != safe_filename(subdir):
            raise ValueError(f"Invalid subdir: {subdir!r}")

        return self._copy_to_store(src_path, stored_name, subdir)

    def index(self, saved_path: str) -> str:
        """Extract text from *saved_path* and write it to the file index.

        Returns the extracted text. A failure raises ``ValueError`` so the
        caller can decide whether to surface it.
        """
        text = self._extract(saved_path)
        FileIndexService().upsert_content(saved_path, text)
        return text

    def ingest(
        self,
        src_path: str,
        *,
        name: str | None = None,
        subdir: str | None = None,
    ) -> tuple[str, str]:
        """Ingest a file into the documents store.

        Extract first (so nothing is copied or indexed on extraction failure —
        ``abilities/browser.py`` relies on that), then copy flat to the
        documents store, then index. It composes ``_extract`` + ``place`` +
        ``upsert_content`` rather than ``place`` + :meth:`index` because
        ``index`` extracts again: composing it would run every screenshot
        through vision twice.

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
        if not src_path or not _os.path.isfile(src_path):
            raise ValueError(f"No file at path: {src_path}")
        text = self._extract(src_path)
        saved_path = self.place(src_path, name=name, subdir=subdir)
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
