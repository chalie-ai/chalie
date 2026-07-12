"""Inbound DTOs for the watched-folders endpoint group.

One request DTO covers both create and update: every field is optional —
``None`` means "leave unchanged" on update, while the endpoint's ``post()``
enforces ``folder_path`` is required on create. ``folder_path`` is immutable
on update — the endpoint never forwards it to the update path.
"""

from __future__ import annotations

from pydantic import Field

from .request import Request


class WatchedFolderRequest(Request):
    """Inbound body for watched-folder create/update."""

    folder_path: str | None = Field(default=None, min_length=1)
    label: str | None = None
    file_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None
    recursive: bool | None = None
    scan_interval: int | None = None
    enabled: bool | None = None


class WatchedFolderBrowseRequest(Request):
    """Inbound body for a host-directory browse; ``path`` optional (defaults to home)."""

    path: str | None = None
