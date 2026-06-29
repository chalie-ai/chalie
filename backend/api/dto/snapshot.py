"""DTOs for the snapshot namespace — whole-instance Time-Machine export/import.

Both endpoints are non-CRUD actions (no resource rows): ``export`` streams a
binary ZIP, ``import`` stages an uploaded clone and triggers a restart. The
``password`` fields are the AES-256 snapshot passphrase — **write-only**: they
appear ONLY on the inbound request DTOs and MUST NOT surface on any response DTO
(never returned, never logged).
"""

from __future__ import annotations

from .base import DTO
from .boundary import File


class SnapshotExportRequest(DTO):
    """Inbound body for snapshot export — an optional AES-256 passphrase."""

    password: str | None = None


class SnapshotImportRequest(DTO):
    """Inbound multipart body for snapshot import — the uploaded ZIP + optional passphrase."""

    file: File
    password: str | None = None


class SnapshotImportResult(DTO):
    """Read shape for the import response — the instance is restarting to apply the restore."""

    restarting: bool