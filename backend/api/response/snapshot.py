"""Response DTOs for the snapshot import endpoint.

Mirrors the field shape used by the snapshot import action. ``restarting`` is
the sole output: ``True`` means the instance is restarting to apply the staged
restore.
"""

from __future__ import annotations

from .response import Response


class SnapshotImportResult(Response):
    """Result of a snapshot import; signals whether a restart is in progress."""

    restarting: bool
