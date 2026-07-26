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

* its ``TBag`` — the :class:`ReviewWindowParamsBag` (or subclass) the dispatch
  seam builds for it; validation and clamping live in the bag's ``from_params``.
* :meth:`_buffer` — the half-window in minutes (``review_tool_calls`` keeps a
  frozen ±5; ``review_transcript`` reads its bag's clamped ``buffer_minutes``).
* :meth:`_fetch` — the ONE windowed SELECT over ``Database.conn()`` that
  returns the rows for this tool (wrapped here in a NARROW ``sqlite3.Error`` so a
  corrupt read is a LOUD ``query-failed``, never a silent empty window).
* :meth:`_row` — turn one fetched record into the structured row dict the model
  sees (compact JSON, rendered by the dispatcher).

Everything else — the structured success/empty/error envelopes, the ``count`` /
``anchor`` / ``buffer`` meta, the ``ACTION_REQUIRED`` pre-gate declaration — is
shared and lives here exactly once. ``code="error"`` appears nowhere.

Listing ``ABC`` in the bases marks this as an abstract mix-in, never a tool in its
own right (precedent: ``abilities/_delegate.py``); only concrete subclasses reach
the registry.
"""

from __future__ import annotations

import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import ClassVar, TypeVar
from typing import cast

from abilities._ability import Ability
from abilities._result import ToolResult
from contracts.params.review_window_params_bag import ReviewWindowParamsBag
from services.time_formatter_service import TimeFormatterService
from services.time_utils import parse_utc

logger = logging.getLogger(__name__)

TBag = TypeVar("TBag", bound=ReviewWindowParamsBag)

# ``parse_utc`` never raises — it returns this ``datetime.min`` sentinel for an
# unparseable / missing value (timer + calendar precedent). The base treats the
# sentinel as an invalid anchor and refuses to open a year-0001 window.
_PARSE_UTC_SENTINEL = datetime.min.replace(tzinfo=timezone.utc)

# The one ISO example surfaced in the invalid-time hint so a weak model can
# self-correct without re-reading the schema (calendar precedent code).
_ISO_HINT = "use an ISO timestamp, e.g. 2026-04-07T14:30:00+00:00"


class ReviewWindowAbility(Ability[TBag], ABC):
    """Base for a review tool that returns DB rows within ±N minutes of an anchor.

    Owns the entire ``run()`` flow; subclasses fill in only :meth:`_buffer`,
    :meth:`_fetch`, and :meth:`_row`.
    """

    # Both review tools take a single required ``date_time`` and no action; the
    # action-less map key ``""`` (vision / code_agent precedent) makes the
    # dispatcher pre-gate reject a missing/whitespace anchor as ``missing-params``
    # BEFORE the policy gate. Inherited unchanged by both subclasses.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": ("date_time",)}

    def run(self, params: TBag) -> ToolResult:
        # 1. Residue guard — defends the direct-caller path. The dispatch path
        #    cannot get here blank (the bag's require_str rejects it), but a
        #    directly constructed bag must not reach parse_utc with nothing.
        date_time = params.date_time.strip()
        if not date_time:
            return ToolResult.err(
                "A date_time anchor is required.",
                code="missing-params",
                hint=_ISO_HINT,
            )

        # 2. Resolve the half-window — a subclass hook (review_transcript reads
        #    its bag's clamped buffer_minutes; review_tool_calls returns a
        #    frozen 5). Validation lives in the bag, so this is always an int.
        buffer = self._buffer(params)

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

        # 4. Window: anchor ± buffer minutes, rendered in the subclass's own
        #    column format (see _bound — a mismatch matches NOTHING, silently).
        lo = self._bound(anchor - timedelta(minutes=buffer))
        hi = self._bound(anchor + timedelta(minutes=buffer))

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

    def _buffer(self, params: TBag) -> int:
        """The half-window in minutes. ``review_transcript`` overrides to read
        its bag's clamped ``buffer_minutes``; validation happened in the bag."""
        return 5

    def _bound(self, moment: datetime) -> str:
        """One window bound, in the SAME text format the subclass's table stores
        ``created_at`` in. ``_fetch`` compares strings, so a bound in a different
        format matches NOTHING and returns an empty window with no error:
        ``' '`` (0x20) sorts below ``'T'`` (0x54), so an ISO-T bound excludes
        every space-separated row. The default matches ``tool_calls``, written by
        ``utc_now().isoformat()``; ``review_transcript`` overrides it."""
        return moment.isoformat()

    @abstractmethod
    def _fetch(self, lo: str, hi: str, params: TBag) -> "list[dict[str, object]]":
        """Run the ONE windowed SELECT over ``Database.conn()`` and return
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
