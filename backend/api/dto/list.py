"""DTOs for the lists resource — the read/create/update HTTP contract.

One class per file is the namespace convention; this file owns the list-level
shapes. Validation (length, non-empty) lives on the ``pydantic.Field``
constraints, so handlers never hand-validate. Datetimes serialize as ISO-8601
UTC via :class:`backend.api.dto.base.DTO`.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .base import DTO


class List(DTO):
    """Read shape for a list. Items are a separate sub-resource, not embedded."""

    id: str
    name: str
    list_type: str
    item_count: int
    checked_count: int
    created_at: datetime
    updated_at: datetime


class ListCreate(DTO):
    """Inbound body to create a list."""

    name: str = Field(..., min_length=1, max_length=200)
    list_type: str = "checklist"


class ListUpdate(DTO):
    """Partial update of a list by id; every field optional."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    list_type: str | None = None
