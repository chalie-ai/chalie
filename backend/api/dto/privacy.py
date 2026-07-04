"""DTO for the privacy namespace — the nuclear delete-result contract.

The privacy namespace is not CRUD: a summary read (raw passthrough), a streaming
export download (raw Flask Response), and a single irreversible delete whose
result is :class:`DeleteAllResult`. ``timestamp`` serializes as ISO-8601 UTC via
the foundation serializer; ``warnings`` is a joined string (not a list), omitted
(``None``) when there are no truncate failures.
"""

from __future__ import annotations

from datetime import datetime

from .base import DTO


class DeleteAllResult(DTO):
    """Result of the irreversible delete-all: confirmation, time, and any failures."""

    deleted: bool
    timestamp: datetime
    warnings: str | None = None