"""Response DTOs for the watched-folders endpoint group.

Mirrors the field shape of the DTOs this endpoint group replaced, so the
wire format stays identical. Watched folders are stored as a flat ``SELECT *``
row; the store persists ``enabled``/``recursive`` as 0/1 integers, and the
field validator here lifts them into real booleans so the wire shape is native.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import field_validator

from .response import Response


class WatchedFolder(Response):
    """Read shape for a watched folder (``SELECT *`` on ``watched_folders``)."""

    id: str
    folder_path: str
    label: str | None
    source_type: str
    enabled: bool
    file_patterns: list[str]
    ignore_patterns: list[str]
    recursive: bool
    scan_interval: int
    source_config: dict[str, object]
    created_at: datetime
    updated_at: datetime

    @field_validator("enabled", "recursive", mode="before")
    @classmethod
    def _bool_field(cls, value: object) -> bool:
        return bool(value)


class WatchedFolderBrowse(Response):
    """Result of browsing a host directory: current path, parent, and sub-dirs."""

    current: str
    parent: str | None
    directories: list[str]
