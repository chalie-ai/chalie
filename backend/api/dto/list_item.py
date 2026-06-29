"""DTOs for the list-items sub-resource — the items HTTP contract.

``list_id`` is never on an item DTO: it lives in the URL (the parent segment of
``/api/lists/<list_id>/items/...``). One class per file, per the namespace
convention.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .base import DTO


class ListItem(DTO):
    """Read shape for one list item."""

    id: str
    content: str
    checked: bool
    position: int
    added_at: datetime
    updated_at: datetime


class ItemCreate(DTO):
    """Inbound body to add one item."""

    content: str = Field(..., min_length=1, max_length=500)


class ItemUpdate(DTO):
    """Partial update of an item by id; every field optional."""

    content: str | None = Field(default=None, min_length=1, max_length=500)
    checked: bool | None = None
    position: int | None = None
