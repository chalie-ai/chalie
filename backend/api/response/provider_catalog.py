"""Response DTO for the curated provider catalog action."""

from __future__ import annotations

from .response import Response


class CatalogEntry(Response):
    """One curated provider preset for the setup wizard."""

    id: str
    name: str
    platform: str
    host: str
    needs_key: bool
