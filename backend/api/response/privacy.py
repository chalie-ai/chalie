"""Response DTOs for the privacy vertical — data summary and nuclear delete.

``DataSummaryResult`` fixes the table-count map the legacy route returned as a
raw dict: one integer field per user-data table plus the oldest/newest episode
dates (``None`` while there are no episodes). ``DeleteAllResult`` is the
irreversible delete-all confirmation; ``timestamp`` serializes as ISO-8601 UTC
via the base, ``warnings`` is a joined string, ``None`` when every truncate
succeeded.
"""

from __future__ import annotations

from datetime import datetime

from .response import Response


class DataSummaryResult(Response):
    """Row counts per user-data table plus oldest/newest episode dates."""

    episodes: int
    transcript: int
    scheduled_items: int
    lists: int
    list_items: int
    documents: int
    data_graph: int
    oldest_memory: str | None = None
    newest_memory: str | None = None


class DeleteAllResult(Response):
    """Result of the irreversible delete-all: confirmation, time, and any failures."""

    deleted: bool
    timestamp: datetime
    warnings: str | None = None
