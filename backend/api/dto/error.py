"""Universal error DTO for non-2xx bodies and 422 validation payloads."""

from __future__ import annotations

from .base import DTO


class Error(DTO):
    """Universal non-2xx body and the 422 validation payload emitted by :func:`expects`."""

    error: str
    details: list[dict[str, object]] | None = None
