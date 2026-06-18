# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ActTrail — read/write/render the ``tool_calls`` trail for one ACT loop.

Lifted from the ``Ability`` god-class (spec §4.3 / the _base.py
elimination). Consumed by ``services.message_processor`` (trail assembly,
compaction, narration, thinking persistence) and the tool-dispatch
chokepoint (``ToolDispatcher.dispatch``), which records every tool outcome.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from services.time_utils import utc_now

if TYPE_CHECKING:
    from services.database_service import DatabaseService

logger = logging.getLogger(__name__)


class ActTrail:
    """Read/write/render the ``tool_calls`` trail for one ACT loop."""

    def __init__(self, db: "DatabaseService | None" = None) -> None:
        # Default to the shared DB service so the common call site is
        # ``ActTrail().record(...)``; injectable for callers that already hold one.
        if db is None:
            from services.database_service import get_shared_db_service  # noqa: PLC0415
            db = get_shared_db_service()
        self._db = db

    def record(
        self,
        *,
        tool_name: str,
        params: dict[str, object],
        result: str,
        transcript_id: "int | None",
    ) -> None:
        """Skips silently when transcript_id is None (delegates with
        skip_transcript have no anchor row; the FK is NOT NULL). A write
        failure logs and is non-fatal — the turn continues. Retention
        janitor in DecayEngineService removes rows older than 7 days."""
        if transcript_id is None:
            logger.debug(
                "[ActTrail.record] skipping record (no transcript_id): tool=%s",
                tool_name,
            )
            return
        try:
            params_json = json.dumps(params)
            now = utc_now().isoformat()
            with self._db.connection() as conn:
                conn.execute(
                    "INSERT INTO tool_calls "
                    "(transcript_id, tool_name, params, result, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (transcript_id, tool_name, params_json, result, now),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[ActTrail.record] trail write failed (non-fatal): tool=%s transcript=%s: %s",
                tool_name, transcript_id, exc,
            )

    def fetch_by_transcript_id(self, transcript_id: int) -> "list[dict[str, object]]":
        """Ordered by autoincrement id (not created_at — one-second
        granularity makes created_at ambiguous when several rows land in
        the same second)."""
        try:
            return self._db.fetch_all(
                "SELECT id, tool_name, params, result, created_at "
                "FROM tool_calls WHERE transcript_id = ? ORDER BY id",
                (transcript_id,),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[ActTrail.fetch_by_transcript_id] query failed for transcript_id=%s: %s",
                transcript_id, exc,
            )
            return []

    @staticmethod
    def render(row: dict[str, object]) -> str:
        """Invariant shape: '[tool_name] params → result'. Same function for
        the LLM prompt, the UI card, and the audit view."""
        tool_name = row.get("tool_name", "unknown")
        params_raw = row.get("params") or "{}"
        result = row.get("result") or ""
        return f"[{tool_name}] {params_raw} → {result}"
