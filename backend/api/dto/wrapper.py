"""DTOs for the wrappers namespace — external bearer-token management.

``token`` is a write-only secret: returned exactly once on create via
:class:`WrapperCreated` and never present on the read shape :class:`Wrapper`.
Datetimes serialize as ISO-8601 UTC through :class:`backend.api.dto.base.DTO`.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .base import DTO


class Wrapper(DTO):
    """Read shape for an external wrapper token. The raw token is never returned."""

    id: str
    wrapper_id: str
    name: str
    metadata: dict[str, object]
    last_seen_at: datetime | None
    created_at: datetime


class WrapperCreate(DTO):
    """Inbound body to mint a wrapper token."""

    name: str = Field(..., min_length=1, max_length=200)
    metadata: dict[str, object] = {}


class WrapperCreated(DTO):
    """Create-only response carrying the one-time raw token (shown once)."""

    wrapper_id: str
    token: str
