"""Outbound DTOs for the wrappers endpoint group.

``token`` is a write-only secret: returned exactly once on create via
:class:`WrapperCreated` and never present on the read shape :class:`Wrapper`.
"""

from __future__ import annotations

from datetime import datetime

from .response import Response


class Wrapper(Response):
    """Read shape for an external wrapper token. The raw token is never returned."""

    id: str
    wrapper_id: str
    name: str
    metadata: dict[str, object]
    last_seen_at: datetime | None
    created_at: datetime


class WrapperCreated(Response):
    """Create-only response carrying the one-time raw token (shown once)."""

    wrapper_id: str
    token: str
