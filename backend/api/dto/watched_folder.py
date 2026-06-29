"""DTOs for the watched-folders sub-resource of the documents namespace.

Watched folders are stored as a flat ``SELECT *`` row, so the read shape mirrors
the table columns. The store persists ``file_patterns``/``ignore_patterns``/
``source_config`` as JSON strings and ``enabled``/``recursive`` as 0/1 integers;
the service parses the JSON fields, and field validators here lift the 0/1 flags
into real booleans so the wire shape is native. Datetimes serialize as
ISO-8601 UTC via :class:`backend.api.dto.base.DTO`.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator

from .base import DTO


def _coerce_bool(value: object) -> bool:
    """Lift a stored 0/1 integer (or bool) into a real bool."""
    return bool(value)


class WatchedFolder(DTO):
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
        return _coerce_bool(value)


class WatchedFolderCreate(DTO):
    """Inbound body to register a new watched folder."""

    folder_path: str = Field(..., min_length=1)
    label: str | None = None
    file_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None
    recursive: bool = True
    scan_interval: int = 300


class WatchedFolderUpdate(DTO):
    """Partial update of a watched folder; every field optional.

    Only the fields ``update_folder`` persists appear here — no unbounded
    ``**data`` splat.
    """

    label: str | None = None
    file_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None
    recursive: bool | None = None
    scan_interval: int | None = None
    enabled: bool | None = None


class BrowseRequest(DTO):
    """Inbound body for a host-directory browse; ``path`` optional (defaults to home)."""

    path: str | None = None


class BrowseResponse(DTO):
    """Result of browsing a host directory: current path, parent, and sub-dirs."""

    current: str
    parent: str | None
    directories: list[str]