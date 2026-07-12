# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""TurnExecutionService — the DB-backed lifecycle of one MessageProcessor turn.

A ``turn_executions`` row is the single source of truth for "is this turn
still running, and did someone ask it to stop" — every read derives from the
DB (§6.6: no fabricated in-memory row). Absorbs the old ``ExecutionTracker``
handle: ``self.mp`` IS the per-turn handle, so there is nothing left to hold
separately. Cross-instance cancel (§6.7) derives its broadcast gate from the
target turn's own row, never from the canceller's ``self.mp.config``, since
the canceller may be an inert MP built for a channel other than the one that
opened the turn.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from configs.enums.config_type import ConfigTypeEnum
from models.turn_execution import TurnExecution
from services.time_utils import utc_now
from services.websocket import Websocket

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

logger = logging.getLogger(__name__)

#: Stamped as ``stop_reason`` when the boot sweep finds a row a killed process
#: never got to close.
STOP_REASON_PROCESS_DEATH = "process death"


class TurnExecutionService:
    """CRUD + lifecycle for ``turn_executions``, keyed off ``self.mp``."""

    def __init__(self, mp: "MessageProcessor") -> None:
        self.mp = mp

    def open(self) -> TurnExecution | None:
        """Insert this turn's row (state=working), stash it on
        ``self.mp.execution``, and emit the lifecycle frame. ``None`` on a
        write failure (logged) — ``should_stop`` fails open when there is no
        row to poll."""
        try:
            execution = TurnExecution(
                channel=self.mp.channel,
                type=self.mp.config.type_value(),
                turn_id=self.mp.turn_id,
                started_at=utc_now().isoformat(),
                ended_at=None,
                cancel_requested=False,
                state=TurnExecution.WORKING,
                stop_reason=None,
            ).save()
        except Exception as exc:
            logger.warning("[TurnExecutionService] open failed for turn_id=%s: %s", self.mp.turn_id, exc)
            return None
        self.mp.execution = execution
        self.mp.push_websocket(execution)
        return execution

    def should_stop(self) -> bool:
        """Single-row PK read of ``cancel_requested`` — the cooperative
        checkpoint the step loop polls, always against the DB (no delegating
        in-memory property). Fails open (never stops the turn) on a missing
        execution or a read error, so a DB hiccup can't silently strand a
        turn mid-cancel."""
        execution: TurnExecution | None = self.mp.execution
        if execution is None or execution.id is None:
            return False
        try:
            row = TurnExecution.filter("id", execution.id).first()
        except Exception as exc:
            logger.warning("[TurnExecutionService] should_stop failed for id=%s: %s", execution.id, exc)
            return False
        return bool(row.cancel_requested) if row is not None else False

    def cancel(self) -> TurnExecution | None:
        """The single cancel-request chokepoint (§2.7): both DELETE
        /api/thread/<turn_id> (an inert MP built for the target turn) and a
        turn's own self-cancelling ability route through here — and, per
        Dylan's ruling, the single AUTHORITY for the turn's terminal state.
        Every provider client is one blocking, non-streaming call with no
        mid-flight abort hook (§ llm_clients/*), so the running turn cannot
        physically be stopped the instant this is called — the in-flight call
        is left to finish server-side, and :meth:`MessageProcessor._step`'s
        cancel checkpoint discards its response instead of storing it. Ending
        the turn's LIFECYCLE does not have to wait for that: this stamps the
        live (``ended_at IS NULL``) row for this (channel, turn_id) CANCELLED
        with ``ended_at`` set right here, synchronously, so the WS frame and
        the DB state both flip the instant cancel is requested rather than
        after the full generation wall-clock. The row IS the cross-request
        handle, since the turn is (almost always) running in another
        thread/instance; ``turn_id`` is a per-channel monotonic counter, not
        globally unique, so the match is channel-scoped. The broadcast
        gate/type derives from the target row, NOT ``self.mp.config`` (§6.7)
        — the canceller may hold a config for a different channel than the
        one that opened the turn. :meth:`finish` becomes a no-op once it
        observes this row already terminal — the row closed here always
        wins, never resurrected or overwritten by the doomed turn's own
        eventual finish()."""
        try:
            execution = TurnExecution.open_turn(self.mp.channel, self.mp.turn_id)
            if execution is None:
                return None
            execution.cancel_requested = True
            execution.state = TurnExecution.CANCELLED
            execution.ended_at = utc_now().isoformat()
            execution.save()
        except Exception as exc:
            logger.warning(
                "[TurnExecutionService] cancel failed for channel=%s turn_id=%s: %s",
                self.mp.channel, self.mp.turn_id, exc,
            )
            return None
        self.mp.execution = execution

        # Gate on the target ROW's type (see docstring): no type → no
        # addressable config → no broadcast; a stale/unknown type must not
        # raise into the caller, so it too falls through to no broadcast.
        if execution.type is not None:
            try:
                if ConfigTypeEnum.get_by_type(execution.type).BROADCASTS_STATE:
                    Websocket.broadcast(execution)
            except ValueError as exc:
                logger.warning(
                    "[TurnExecutionService] unknown config type=%r for turn_id=%s — no broadcast: %s",
                    execution.type, execution.turn_id, exc,
                )
        return execution

    def finish(self, state: str, stop_reason: str | None = None) -> TurnExecution | None:
        """Stamp the terminal state ONCE — ``cancel_requested`` is never
        touched here, so a crash after a cancel request preserves both facts
        with zero special-casing. The terminal state — CRASHED included —
        travels on the lifecycle frame alone; the surface derives the 'turn
        failed' toast from ``state == 'crashed'``, so there is no separate
        error frame to emit.

        Race-safe against :meth:`cancel`, which can land on a DIFFERENT
        connection/thread (the Flask request thread vs. this turn's own
        drive loop, services/database.py) at any point, including between
        two separate statements here — a read-then-write is NOT enough to
        close that window (a plain re-read-then-``save()`` narrows it but
        does not close it: cancel's UPDATE can still land in the gap between
        the read and this write and get silently overwritten). So the close
        is ONE atomic conditional UPDATE — ``WHERE id = ? AND ended_at IS
        NULL`` — SQLite executes a single UPDATE atomically, no explicit lock
        needed. ``rowcount == 0`` means the row was already closed (cancel()
        — or a racing finish() — won outright): that row is authoritative,
        re-read and returned as-is, no further write, no re-broadcast, never
        resurrecting or overwriting a cancelled turn with a synthesized
        'completed'/'crashed' state."""
        current: TurnExecution | None = self.mp.execution
        if current is None or current.id is None:
            return None
        ended_at = utc_now().isoformat()
        try:
            closed = (
                TurnExecution.filter("id", current.id)
                .filter("ended_at", None, "IS")
                .update(ended_at=ended_at, state=state, stop_reason=stop_reason)
            )
        except Exception as exc:
            logger.warning("[TurnExecutionService] finish failed for id=%s: %s", current.id, exc)
            return None
        if closed == 0:
            try:
                row = TurnExecution.get(current.id)
            except Exception as exc:
                logger.warning("[TurnExecutionService] finish re-read failed for id=%s: %s", current.id, exc)
                return None
            if row is not None:
                self.mp.execution = row
            return row
        execution = TurnExecution(
            id=current.id,
            channel=current.channel,
            type=current.type,
            turn_id=current.turn_id,
            started_at=current.started_at,
            ended_at=ended_at,
            cancel_requested=current.cancel_requested,
            state=state,
            stop_reason=stop_reason,
        )
        self.mp.execution = execution
        self.mp.push_websocket(execution)
        return execution

    def sweep_orphaned(self) -> int:
        """Boot-time reconciliation: every row a killed process left open
        (``ended_at IS NULL``) becomes ``crashed`` — the process that owned
        it is gone, so "working" surviving from a previous run is always
        stale. The whole sweep runs inside one ``self.mp.db`` transaction
        (I6) so a mid-loop write failure rolls back every row it already
        touched rather than stranding a partially-swept boot; returns the
        number of rows closed, or 0 if the sweep did not commit."""
        count = 0
        try:
            with self.mp.db.transaction():
                for execution in TurnExecution.open_rows():
                    execution.state = TurnExecution.CRASHED
                    execution.stop_reason = STOP_REASON_PROCESS_DEATH
                    execution.ended_at = utc_now().isoformat()
                    execution.save()
                    count += 1
        except Exception as exc:
            logger.warning("[TurnExecutionService] sweep_orphaned failed: %s", exc)
            return 0
        return count

    def latest_for_turn(self) -> TurnExecution | None:
        """The newest row for this turn (channel-scoped, since ``turn_id`` is
        a per-channel counter) — for a caller that holds no live execution
        for it, e.g. the scheduler deciding whether a first-fire's turn
        survived cancellation before persisting its ``turn_id`` for series
        continuity (mirrors ``cancel``'s pattern: an inert MP built for the
        target (channel, turn_id) reads it via ``self.mp``)."""
        try:
            return (
                TurnExecution.filter("channel", self.mp.channel)
                .filter("turn_id", self.mp.turn_id)
                .order_by("id DESC")
                .limit(1)
                .first()
            )
        except Exception as exc:
            logger.warning(
                "[TurnExecutionService] latest_for_turn failed for channel=%s turn_id=%s: %s",
                self.mp.channel, self.mp.turn_id, exc,
            )
            return None
