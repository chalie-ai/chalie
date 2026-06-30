# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ActTrail — read/write/render the ``tool_calls`` trail for one ACT loop.

A tool call anchors ONLY to the transcript input row that drove it
(``transcript_id``); its turn is derived by joining transcript on
(channel, turn_id), so ``tool_calls`` carries no turn_id / channel column of
its own. Consumed by ``services.message_processor`` (trail assembly
and compaction) and the tool-dispatch chokepoint (``ToolDispatcher.dispatch``),
which records every tool outcome.
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

    def start(
        self,
        *,
        tool_name: str,
        params: dict[str, object],
        transcript_id: "int | None",
        summary: "str | None" = None,
    ) -> "int | None":
        """Open a ``tool_calls`` row the instant a call begins executing and return
        its id, so the live ``tool_called`` signal can name the row whose timer it
        starts; :meth:`finish` writes the result when the call returns. The call
        anchors to its input row alone (``transcript_id``) — its turn is derived
        later by joining transcript on (channel, turn_id). ``summary`` is the
        ability's act_summary (the live blue box), persisted so a chat refresh can
        re-render it. Skips (returns None) when transcript_id is None — a delegate
        with skip_transcript has no anchor row and the FK is NOT NULL. A write
        failure logs and yields None; the turn continues unrecorded."""
        if transcript_id is None:
            logger.debug("[ActTrail.start] skipping (no transcript_id): tool=%s", tool_name)
            return None
        try:
            from services.transcript_service import _non_settling_tools  # noqa: PLC0415
            with self._db.connection() as conn:
                cur = conn.execute(
                    "INSERT INTO tool_calls "
                    "(transcript_id, tool_name, params, result, summary, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (transcript_id, tool_name, json.dumps(params), "", summary or "", utc_now().isoformat()),
                )
                call_id = cur.lastrowid
                if tool_name not in _non_settling_tools():
                    conn.execute(
                        "UPDATE transcript SET settled = 0 WHERE id = ?",
                        (transcript_id,),
                    )
                return call_id
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[ActTrail.start] trail open failed (non-fatal): tool=%s transcript=%s: %s",
                tool_name, transcript_id, exc,
            )
            return None

    def finish(self, *, call_id: "int | None", result: str) -> None:
        """Write the result onto a row opened by :meth:`start`. No-op when call_id
        is None (the call was never recorded — no anchor). A write failure logs and
        is non-fatal. Retention janitor in DecayEngineService removes rows older
        than 7 days."""
        if call_id is None:
            return
        try:
            with self._db.connection() as conn:
                conn.execute("UPDATE tool_calls SET result = ? WHERE id = ?", (result, call_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[ActTrail.finish] trail result write failed (non-fatal): call=%s: %s",
                call_id, exc,
            )

    def record(
        self,
        *,
        tool_name: str,
        params: dict[str, object],
        result: str,
        transcript_id: "int | None",
        summary: "str | None" = None,
    ) -> None:
        """One-shot open+result for outcomes that never enter execution (an unknown
        tool, a denied or pre-validation-failed call): no live timer brackets them,
        so the row is written whole after the fact. Skips silently when
        transcript_id is None."""
        self.finish(
            call_id=self.start(
                tool_name=tool_name, params=params, transcript_id=transcript_id, summary=summary,
            ),
            result=result,
        )

    def fetch_by_turn(self, channel: str, turn_id: int) -> "list[dict[str, object]]":
        """Every tool call of one logical turn, ordered by autoincrement id.

        tool_calls carries no turn column; a turn's calls are derived by joining
        transcript on (channel, turn_id). Each turn has exactly one input row —
        the one that opened it; an async result or delegate re-entry is a NEW turn
        with its own turn_id, never a continuation of this one, so the join
        gathers only this turn's calls. Ordered by tc.id, not created_at —
        one-second granularity makes created_at ambiguous when rows land in the
        same second."""
        try:
            return self._db.fetch_all(
                "SELECT tc.id, tc.tool_name, tc.params, tc.result, tc.summary, tc.created_at "
                "FROM tool_calls tc JOIN transcript t ON tc.transcript_id = t.id "
                "WHERE t.channel = ? AND t.turn_id = ? ORDER BY tc.id",
                (channel, turn_id),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[ActTrail.fetch_by_turn] query failed for channel=%s turn_id=%s: %s",
                channel, turn_id, exc,
            )
            return []

    def fetch_by_transcript_id(self, transcript_id: int) -> "list[dict[str, object]]":
        """All tool calls anchored to one input row, ordered by autoincrement id.

        The narrow single-anchor read — every tool call the dispatcher recorded
        against a given input row (``transcript_id``). Equivalent to
        :meth:`fetch_by_turn` for a turn with a single input row; the turn-keyed
        read is preferred in the loop because it also spans async / delegate
        re-entries."""
        try:
            return self._db.fetch_all(
                "SELECT id, tool_name, params, result, summary, created_at "
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
