"""Outbound DTOs for the threads endpoint group.

A turn (thread) is many transcript rows sharing one ``turn_id``. The feed
lists collapsed :class:`ThreadSummary` rows; :class:`TurnBlock` is one turn's
full rows plus the collapsed metadata the pill renders from, so a single fetch
fully determines a turn's render. Internal recency/row-count keys from the
query layer are never exposed.

Both shapes carry ``type`` — the ConfigType identity
(``user``/``scheduled``/``discovery``) they were resolved under. ``turn_id`` is
only unique PER TYPE, so a client that learns of a thread from the feed must
carry this forward rather than fall back to the ``user`` default and resolve
to the wrong channel.
"""

from __future__ import annotations

from datetime import datetime

from api.dto.message import Message

from .response import Response


class ThreadSummary(Response):
    """One collapsed feed row: the thread's id, last-activity timestamp, opener
    preview, row count, one-sentence gist, and whether the turn is still
    working. The internal recency sort key (the latest transcript row id)
    drives ordering server-side and is not exposed."""

    turn_id: int | None = None
    last_activity_at: str | None = None
    preview: str
    row_count: int
    gist: str | None = None
    working: bool = False
    type: str


class TurnBlock(Response):
    """One turn's full block — the single read, every batch entry and the WS
    refetch all return this shape. Rows past the turn's settle0 carry
    ``thread_message`` on the message."""

    turn_id: int
    gist: str | None = None
    preview: str
    last_activity_at: str | None = None
    working: bool
    duration_ms: int
    messages: list[Message]
    type: str
    #: True when the turn's most recent execution ended CRASHED (unhandled step
    #: exception or a swept process death). Drives the FE "ended unexpectedly"
    #: note so a reply-less crash isn't mistaken for a normal empty turn.
    crashed: bool = False


class ThreadBatch(Response):
    """``GET /api/threads/batch`` body — many turn blocks in one round-trip."""

    blocks: list[TurnBlock]


class ThinkingLevel(Response):
    """``GET /api/threads/thinking-level/<turn_id>`` body — the literal value
    persisted on the thread's input row (``auto``/``medium``/``high``)."""

    level: str


class TurnExecution(Response):
    """One turn's DB-backed lifecycle row, returned by ``POST /api/threads/<turn_id>``.

    The send opens this row synchronously — the request thread writes it before
    responding — so the surface holds the turn handle inline with no websocket
    round-trip. It is exactly the payload the turn's TurnExecutionService
    broadcasts on every state flip, so no surface ever reconciles two versions
    of what a turn execution looks like. ``started_at``/``ended_at`` are
    persisted as ISO-8601 strings; pydantic parses them and the base re-emits
    canonical ISO-8601 UTC, so handlers never call ``.isoformat()``.
    """

    id: int
    channel: str
    type: str | None
    turn_id: int
    started_at: datetime
    ended_at: datetime | None
    cancel_requested: bool
    state: str
    stop_reason: str | None


class TurnStopResult(Response):
    """200 ack for ``DELETE /api/threads/<turn_id>`` (turn interrupt).

    ``cancelled`` is True when a live turn was stopped, else ``reason`` is
    ``"no_active_turn"`` — an idle or finished turn_id is a harmless ack, never
    a 404, since a turn ends on its own and the stop button would otherwise
    error on a benign race. The cancelled row itself reaches the surface on the
    lifecycle websocket channel every other state flip uses.
    """

    cancelled: bool | None = None
    reason: str | None = None
