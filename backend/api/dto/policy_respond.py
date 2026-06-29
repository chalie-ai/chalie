"""DTO for the permission-gate respond body."""

from __future__ import annotations

from pydantic import Field

from .base import DTO


class PolicyRespond(DTO):
    """Inbound body resolving an in-process permission gate with the user's decision."""

    request_id: str = Field(..., min_length=1)
    approved: bool