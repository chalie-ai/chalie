"""Response DTO for the curated provider catalog action."""

from __future__ import annotations

from .response import Response


class CatalogEntry(Response):
    """One platform offered by the setup wizard."""

    id: str
    name: str
    platform: str
    host: str
    needs_key: bool
    #: Whether the user must supply the host themselves. Sent so the wizard
    #: shows or hides the host field from the platform's own declaration rather
    #: than from a list of platform names it would have to keep in step.
    needs_host: bool
