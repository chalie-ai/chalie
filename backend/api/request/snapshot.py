"""Inbound DTO for the snapshot export endpoint.

A single optional field: ``password`` is an optional passphrase; when set the
export zip is AES-256 encrypted before being streamed back.
"""

from __future__ import annotations

from .request import Request


class SnapshotExportRequest(Request):
    """Snapshot export request — optional passphrase for AES-256 encryption."""

    password: str | None = None
