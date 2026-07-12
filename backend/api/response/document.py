"""Response DTOs for the documents endpoint group.

Mirrors the field shape of the legacy ``api.dto.document`` module so the wire
format stays identical. Documents are not pure CRUD — file I/O, semantic
search, and lifecycle transitions sit alongside id-addressed reads — so this
file owns the read shapes and per-operation response bodies.

The document store persists ``tags`` and ``extracted_metadata`` as JSON
strings; field validators lift those into native types so the wire shape is
always a real list/dict, never a string.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import cast

from pydantic import field_validator

from .response import Response


def _as_object(value: object) -> object:
    """Parse a JSON string into its native value; pass non-strings through."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


class Document(Response):
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


class ArtifactPreview(Response):
    """One ``data_graph`` artifact preview (first 200 chars of its value)."""

    key: str
    preview: str


class DocumentDetail(Document):
    """Detail read adding the first N ``data_graph`` artifact previews."""

    artifacts: list[ArtifactPreview]


class ArtifactContent(Response):
    """One ``data_graph`` artifact with its full value."""

    key: str
    content: str


class DocumentContent(Response):
    """Full document text reconstructed from ``data_graph`` artifacts."""

    document_id: str
    total_artifacts: int
    artifacts: list[ArtifactContent]


class DuplicateRef(Response):
    """A prior document matched as a duplicate of a fresh upload."""

    id: str
    original_name: str
    match_type: str
    created_at: datetime | None


class UploadResponse(Response):
    """Response to a successful upload, with optional duplicate matches."""

    id: str
    original_name: str
    status: str
    file_size: int
    file_hash: str
    duplicates: list[DuplicateRef] | None = None


class SupersedeResponse(Response):
    """Response confirming the superseded document id."""

    ok: bool
    supersedes_id: str


class ClassificationGroup(Response):
    """One bucket of a classification grouping (value + document count)."""

    value: str
    count: int


class SearchResult(Response):
    """One ``data_graph`` recall row projected onto a document search hit."""

    document_id: str
    key: str
    content: str
    source: str


class SearchResponse(Response):
    """Ranked recall results for a document search."""

    results: list[SearchResult]
    query: str
