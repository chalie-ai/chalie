"""Response DTOs for the lists module.

Mirrors the field shape of the legacy DTOs so the wire format stays identical.
"""

from __future__ import annotations

from datetime import datetime

from .response import Response


class ListResponse(Response):
    """Read shape for a list — matches the legacy List DTO fields."""

    id: str
    name: str
    list_type: str
    item_count: int
    checked_count: int
    created_at: datetime
    updated_at: datetime


class ListItemResponse(Response):
    """Read shape for one list item — matches the legacy ListItem DTO fields."""

    id: str
    content: str
    checked: bool
    position: int
    added_at: datetime
    updated_at: datetime
