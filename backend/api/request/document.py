"""Inbound DTOs for the documents endpoint group.

Covers partial classification updates, augmentation of pending documents,
supersede marking, and search queries. Each class subclasses
:class:`backend.api.request.Request` so ``extra="forbid"`` and ISO-8601
datetime serialization carry over from
:class:`backend.api.dto.base.DTO`.

``UploadRequest`` is intentionally absent — the upload action reads
``request.files`` directly (multipart is not a JSON body), so no DTO is
needed there.
"""

from __future__ import annotations

from pydantic import Field

from .request import Request


class ClassifyRequest(Request):
    """Partial classification update; every field optional."""

    category: str | None = None
    project: str | None = None
    date: str | None = None


class AugmentRequest(Request):
    """Body for adding user context to a document awaiting confirmation."""

    context: str = Field(..., min_length=1)


class SupersedeRequest(Request):
    """Body marking the new document as superseding a prior one."""

    old_id: str = Field(..., min_length=1)


class SearchQuery(Request):
    """Inbound query for document search — non-empty ``q``, capped ``limit``."""

    q: str = Field(..., min_length=1)
    limit: int = Field(default=5, le=20)
