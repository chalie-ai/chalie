"""``ReviewWindowAbility`` — shared retrieval/formatting base for the two review tools.

``review_tool_calls`` and ``review_transcript`` are near-twins: both anchor on an
ISO ``date_time``, open a ±N-minute window around it, run ONE windowed query
against the shared SQLite database, and hand the model back the rows that fell in
that window (raw tool-call records for one, conversation transcript rows for the
other). They had duplicated the entire flow — the residue guard, the anchor
parse, the window arithmetic, the empty-window handling, and the row formatting —
once each, diverging in subtle ways (one swallowed DB errors into a silent empty
window, one crashed on a non-int buffer, both emitted the banned ``code="error"``
and prose bodies without a ``count`` meta).

This base owns the WHOLE flow. A concrete review tool is a thin subclass that
declares only what differs:

* :meth:`_buffer` — the half-window in minutes (``review_tool_calls`` keeps a
  frozen ±5; ``review_transcript`` parses a clamped ``buffer_minutes`` param).
* :meth:`_fetch` — the ONE windowed SELECT over ``get_shared_db_service()`` that
  returns the rows for this tool (wrapped here in a NARROW ``sqlite3.Error`` so a
  corrupt read is a LOUD ``query-failed``, never a silent empty window).
* :meth:`_row` — turn one fetched record into the structured row dict the model
  sees (compact JSON, rendered by the dispatcher).

Everything else — the structured success/empty/error envelopes, the ``count`` /
``anchor`` / ``buffer`` meta, the ``ACTION_REQUIRED`` pre-gate declaration — is
shared and lives here exactly once. ``code="error"`` appears nowhere.

Listing ``ABC`` in the bases makes ``Ability.__init_subclass__`` skip this base in
the metadata probe (precedent: ``abilities/_budget.py``); concrete subclasses are
probed normally.
"""

from __future__ import annotations

import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import ClassVar
from typing import cast

from abilities._ability import Ability
from abilities._result import ToolResult
from services.time_formatter_service import TimeFormatterService
from services.time_utils import parse_utc

logger = logging.getLogger(__name__)

# ``parse_utc`` never raises — it returns this ``datetime.min`` sentinel for an
# unparseable / missing value (timer + calendar precedent). The base treats the
# sentinel as an invalid anchor and refuses to open a year-0001 window.
_PARSE_UTC_SENTINEL = datetime.min.replace(tzinfo=timezone.utc)

# The one ISO example surfaced in the invalid-time hint so a weak model can
# self-correct without re-reading the schema (calendar precedent code).
_ISO_HINT = "use an ISO timestamp, e.g. 2026-04-07T14:30:00+00:00"


class ReviewWindowAbility(Ability, ABC):
    """Base for a review tool that returns DB rows within ±N minutes of an anchor.

    Owns the entire ``run()`` flow; subclasses fill in only :meth:`_buffer`,
    :meth:`_fetch`, and :meth:`_row`.
    """

    # Both review tools take a single required ``date_time`` and no action; the
    # action-less map key ``""`` (vision / code_eval precedent) makes the
    # dispatcher pre-gate reject a missing/whitespace anchor as ``missing-params``
    # BEFORE the policy gate. Inherited unchanged by both subclasses.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": ("date_time",)}

    def run(self, params: "dict[str, object]") -> ToolResult:
        # 1. Residue guard — defends the direct-caller path. The dispatch path is
        #    already caught by the ACTION_REQUIRED pre-gate, but a direct call
        #    (or a non-dispatch caller) must not reach parse_utc with nothing.
        date_time = cast(str, params.get("date_time") or "").strip()
        if not date_time:
            return ToolResult.err(
                "A date_time anchor is required.",
                code="missing-params",
                hint=_ISO_HINT,
            )

        # 2. Resolve the half-window — a subclass hook (review_transcript parses a
        #    clamped buffer_minutes; review_tool_calls returns a frozen 5). A
        #    buffer error (bad buffer_minutes) is returned as-is.
        buffer = self._buffer(params)
        if isinstance(buffer, ToolResult):
            return buffer

        # 3. Parse the anchor. parse_utc returns the datetime.min sentinel on
        #    garbage rather than raising, so we check the sentinel (no broad
        #    except — there is no exception to catch).
        anchor = parse_utc(date_time)
        if anchor == _PARSE_UTC_SENTINEL:
            return ToolResult.err(
                f"Could not understand the timestamp {date_time!r}.",
                code="invalid-time",
                hint=_ISO_HINT,
            )

        # 4. Window: anchor ± buffer minutes, as ISO strings for the SQL bind.
        lo = (anchor - timedelta(minutes=buffer)).isoformat()
        hi = (anchor + timedelta(minutes=buffer)).isoformat()

        # 5. Fetch — the ONE windowed query, wrapped in a NARROW sqlite3.Error so
        #    a corrupt / dropped table is a LOUD query-failed, never a silent
        #    empty window (the bug the deleted get_by_timerange swallowed).
        try:
            records = self._fetch(lo, hi, params)
        except sqlite3.Error as exc:
            logger.error(
                "[%s] window query failed for date_time=%r: %s",
                self.get_name(), date_time, exc, exc_info=True,
            )
            return ToolResult.err(
                f"query-failed: {exc}",
                code="query-failed",
                hint="the conversation store could not be read; try again.",
            )

        # 6. Empty window — a SUCCESS (nothing failed), never an error. The body
        #    TELLS the model what to try next; count=0 is the branchable signal.
        if not records:
            return ToolResult.ok(
                self._empty_hint(date_time, buffer),
                count=0,
                anchor=date_time,
                buffer=buffer,
            )

        # 7. Rows → structured list, one dict per record via the subclass hook;
        #    the dispatcher renders the list as compact JSON.
        rows: "list[object]" = [self._row(rec, i) for i, rec in enumerate(records, start=1)]
        return ToolResult.ok(rows, count=len(rows), anchor=date_time, buffer=buffer)

    # ── Subclass hooks ────────────────────────────────────────────────────────

    def _buffer(self, params: "dict[str, object]") -> "int | ToolResult":
        """Return an int, or an error ``ToolResult`` when the supplied buffer is
        invalid (``review_transcript`` overrides to parse a clamped param)."""
        return 5

    @abstractmethod
    def _fetch(self, lo: str, hi: str, params: "dict[str, object]") -> "list[dict[str, object]]":
        """Run the ONE windowed SELECT over ``get_shared_db_service()`` and return
        the rows whose ``created_at`` falls in ``[lo, hi]``. Raise only
        ``sqlite3.Error`` (the base catches it loudly); no other exception."""
        ...

    @abstractmethod
    def _row(self, rec: "dict[str, object]", ordinal: int) -> "dict[str, object]":
        """Format one fetched record into the structured row the model sees.

        ``ordinal`` is the 1-based position of the row within the returned window
        (there is no iteration column on the tables — the ordinal IS ``iter``)."""
        ...

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _empty_hint(self, date_time: str, buffer: int) -> str:
        return (
            f"No records found within ±{buffer} minutes of {date_time}. "
            "Try a different timestamp."
        )

    @staticmethod
    def _ts(created_at: object) -> str:
        """Falls back to the raw leading characters when the value is missing / unparseable."""
        return TimeFormatterService.local(cast("str | None", created_at), fmt="%Y-%m-%d %H:%M:%S") \
            or str(created_at or "")[:19]
