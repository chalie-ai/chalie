"""Response DTO for the provider-role actions (selected, vision, delegate)."""

from __future__ import annotations

from .provider import Provider
from .response import Response


class ProviderRole(Response):
    """Response shape for the selected/vision/delegate role actions.

    ``source`` is ``'explicit'`` (pinned), ``'auto'`` (resolved via fallback —
    vision/delegate only), or ``'none'`` (unset).
    """

    provider: Provider | None
    source: str
