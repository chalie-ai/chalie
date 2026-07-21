"""Privacy data-summary action — row counts per user-data table plus oldest/newest memory dates.

Covers:
- GET /api/privacy/data-summary  → get
"""

from __future__ import annotations

import logging

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.response.privacy import DataSummaryResult
from services.database import Database

logger = logging.getLogger(__name__)

_TABLES = (
    "episodes",
    "transcript",
    "scheduled_items",
    "lists",
    "list_items",
    "documents",
    "data_graph",
)


class PrivacyDataSummaryAction(Action):
    """Action returning per-table row counts plus oldest/newest episode dates."""

    response_dto = {
        "get": DocumentedResponse(DataSummaryResult, not_found=False),
    }

    def slug(self) -> str:
        return "privacy"

    def verb(self) -> str:
        return "data-summary"

    def get(self, id: int | str) -> ResponseReturnValue:
        """Per-table row counts plus oldest/newest memory dates."""
        conn = Database.conn()
        counts: dict[str, int] = {}
        for table in _TABLES:
            try:
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row = cursor.fetchone()
                counts[table] = row[0] if row else 0
            except Exception as exc:
                logger.warning(
                    "[REST API] privacy/data-summary count failed for %s: %s", table, exc
                )
                counts[table] = 0

        oldest: str | None = None
        newest: str | None = None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM episodes")
            row = cursor.fetchone()
            if row and row[0]:
                oldest = str(row[0].date()) if hasattr(row[0], "date") else str(row[0])
                newest = str(row[1].date()) if hasattr(row[1], "date") else str(row[1])
        except Exception as exc:
            logger.warning(
                "[REST API] privacy/data-summary timestamp query failed: %s", exc
            )

        return DataSummaryResult(
            episodes=counts["episodes"],
            transcript=counts["transcript"],
            scheduled_items=counts["scheduled_items"],
            lists=counts["lists"],
            list_items=counts["list_items"],
            documents=counts["documents"],
            data_graph=counts["data_graph"],
            oldest_memory=oldest,
            newest_memory=newest,
        ).single()
