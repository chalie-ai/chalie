# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ActTrail — the repository for the ACT-loop tool-call trail (``tool_calls``).

record / fetch_by_transcript_id / render are raw SQL over the ``tool_calls``
table: writing a row per tool call, reading a transcript's rows oldest→newest,
and rendering one row to its invariant ``[tool_name] params → result`` form.
This is a repository, not an Ability concern — it was lifted out of the
``Ability`` god-class (spec §4.3 / the _base.py elimination).

Consumers: ``services.message_processor`` (trail assembly, compaction,
narration, thinking persistence) and the tool-dispatch chokepoint
(``ToolDispatcher.dispatch``), which records every tool
outcome so the rendered trail tells the model what happened.

It holds a real dependency — the shared DB service — so it is a constructed
object, not a static dumping ground.
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
        params: dict,
        result: str,
        transcript_id: "int | None",
    ) -> None:
        """The ONLY write path. One INSERT, raw params/result, no render.

        Skips silently when transcript_id is None (delegates with skip_transcript
        have no anchor row; the FK column is NOT NULL so we never attempt the
        insert). A write failure logs and is non-fatal — the turn continues.

        All rows are durable; the retention janitor in DecayEngineService removes
        rows older than 7 days.
        """
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

    def fetch_by_transcript_id(self, transcript_id: int) -> "list[dict]":
        """Every ``tool_calls`` row for an ACT loop, oldest→newest.

        Ordered by autoincrement id (not created_at — one-second granularity
        makes created_at ambiguous when several rows land in the same second).

        Spec §4c / F3.
        """
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
    def render(row: dict) -> str:
        """The ONLY render path. One row → its representation.

        Invariant shape: '[tool_name] params → result'. Same function for the
        LLM prompt, the UI card, and the audit view.

        Spec §4c / F4.
        """
        tool_name = row.get("tool_name", "unknown")
        params_raw = row.get("params") or "{}"
        result = row.get("result") or ""
        return f"[{tool_name}] {params_raw} → {result}"
