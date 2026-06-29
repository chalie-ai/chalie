"""DTOs for the documents namespace — the upload/search/lifecycle HTTP contract.

Documents are not pure CRUD (file I/O, semantic search, and lifecycle
transitions sit alongside id-addressed reads), so this file owns both the read
shapes and the per-operation request/response bodies. Validation (non-empty,
min-length, limit caps) lives on the ``pydantic.Field`` constraints, so handlers
never hand-validate. Datetimes serialize as ISO-8601 UTC via
:class:`backend.api.dto.base.DTO`.

The document store persists ``tags`` and ``extracted_metadata`` as JSON strings;
field validators lift those into native types so the wire shape is always a real
list/dict, never a string.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import cast

from pydantic import Field, field_validator

from .base import DTO
from .boundary import File


def _as_object(value: object) -> object:
    """Parse a JSON string into its native value; pass non-strings through."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


class Document(DTO):
    """Read shape for a document, exposed in lists and detail.

    ``clean_text`` is intentionally absent — it is too large for list/detail
    responses and lives only on the content sub-resource.
    """

    id: str
    original_name: str
    mime_type: str
    file_size_bytes: int
    file_hash: str
    page_count: int | None
    status: str
    error_message: str | None
    chunk_count: int | None
    source_type: str
    tags: list[str]
    summary: str | None
    extracted_metadata: dict[str, object]
    supersedes_id: str | None
    language: str | None
    fingerprint: str | None
    doc_category: str | None
    doc_project: str | None
    doc_date: str | None
    meta_locked: bool
    watched_folder_id: str | None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
    purge_after: datetime | None

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, value: object) -> list[str]:
        parsed = _as_object(value)
        return cast("list[str]", parsed) if isinstance(parsed, list) else []

    @field_validator("extracted_metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: object) -> dict[str, object]:
        parsed = _as_object(value)
        return cast("dict[str, object]", parsed) if isinstance(parsed, dict) else {}

    @field_validator("meta_locked", mode="before")
    @classmethod
    def _coerce_bool(cls, value: object) -> bool:
        return bool(value)


class ArtifactPreview(DTO):
    """One ``data_graph`` artifact preview (first 200 chars of its value)."""

    key: str
    preview: str


class DocumentDetail(Document):
    """Detail read adding the first N ``data_graph`` artifact previews."""

    artifacts: list[ArtifactPreview]


class ArtifactContent(DTO):
    """One ``data_graph`` artifact with its full value."""

    key: str
    content: str


class DocumentContent(DTO):
    """Full document text reconstructed from ``data_graph`` artifacts."""

    document_id: str
    total_artifacts: int
    artifacts: list[ArtifactContent]


class DuplicateRef(DTO):
    """A prior document matched as a duplicate of a fresh upload."""

    id: str
    original_name: str
    match_type: str
    created_at: datetime | None


class UploadResponse(DTO):
    """Response to a successful upload, with optional duplicate matches."""

    id: str
    original_name: str
    status: str
    file_size: int
    file_hash: str
    duplicates: list[DuplicateRef] | None = None


class UploadRequest(DTO):
    """Multipart upload body: a single file field.

    Extension/size/empty validation stays in the handler (filesystem-dependent),
    not on a ``Field`` constraint.
    """

    file: File


class ClassifyRequest(DTO):
    """Partial classification update; every field optional."""

    category: str | None = None
    project: str | None = None
    date: str | None = None


class AugmentRequest(DTO):
    """Body for adding user context to a document awaiting confirmation."""

    context: str = Field(..., min_length=1)


class SupersedeRequest(DTO):
    """Body marking the new document as superseding a prior one."""

    old_id: str = Field(..., min_length=1)


class SupersedeResponse(DTO):
    """Response confirming the superseded document id."""

    ok: bool
    supersedes_id: str


class ClassificationGroup(DTO):
    """One bucket of a classification grouping (value + document count)."""

    value: str
    count: int


class SearchResult(DTO):
    """One ``data_graph`` recall row projected onto a document search hit."""

    document_id: str
    key: str
    content: str
    source: str


class SearchResponse(DTO):
    """Ranked recall results for a document search."""

    results: list[SearchResult]
    query: str


class SearchQuery(DTO):
    """Inbound query for document search — non-empty ``q``, capped ``limit``."""

    q: str = Field(..., min_length=1)
    limit: int = Field(default=5, le=20)