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
separately. Cross-instance cancel (§6.7) is addressed by (channel, turn_id)
alone and derives its broadcast from the target turn's own row, since the
canceller holds no MP for the turn it stops.

A subagent's turn also closes its ``agents`` row, in the transaction that
closes its execution row, on every way it ends; the row's frame goes out once
that commits (:class:`~services.agent_service.AgentService`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from configs.channels import config_for
from configs.enums.channels import Channel
from models.queued_message import QueuedMessage
from models.tool_call import ToolCall
from models.transcript import Transcript
from models.turn_execution import TurnExecution
from services.agent_service import AgentService
from services.database import Database
from services.time_utils import utc_now
from services.websocket import Websocket

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor
    from models.agent import Agent
    from services.processor_config import ProcessorConfig

logger = logging.getLogger(__name__)

#: Stamped as ``stop_reason`` when the boot sweep finds a row a killed process
#: never got to close.
STOP_REASON_PROCESS_DEATH = "process death"
#: Stamped as ``stop_reason`` on a delegate's turn stopped by its caller's own
#: stop, or by the chat's ``stop_subagent`` call naming it.
STOP_REASON_PARENT_STOPPED = "parent stopped"


@dataclass(frozen=True)
class CancelledTurn:
    """A turn :meth:`TurnExecutionService.stamp_cancelled` closed: its
    execution row, the agent that turn ended (``None`` when it ended none),
    and the turn's own tool calls, whose delegate turns are stopped with it."""

    execution: TurnExecution
    agent: Agent | None
    calls: list[ToolCall]


class TurnExecutionService:
    """CRUD + lifecycle for ``turn_executions``, keyed off ``self.mp``."""

    def __init__(self, mp: "MessageProcessor") -> None:
        self.mp = mp

    def open(self) -> TurnExecution | None:
        """Insert this turn's row (state=working), stash it on
        ``self.mp.execution``, and emit the lifecycle frame. ``None`` on a
        write failure (logged) — ``should_stop`` fails open when there is no
        row to poll. A turn opened on a subagent — its spawn, or a later turn
        a message resumed it on — first announces its ``agents`` row,
        committed with its input row."""
        if self.mp.agent is not None:
            self.mp.announce(AgentService.frame(self.mp.agent))
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
                input_id=self.mp.uid,
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
        """Cancel this MP's own turn — :meth:`cancel_turn` addressed by
        ``self.mp``'s (channel, turn_id), for a turn's own self-cancelling
        ability. The closed row becomes ``self.mp.execution``."""
        execution = self.cancel_turn(self.mp.channel, self.mp.turn_id)
        if execution is not None:
            self.mp.execution = execution
        return execution

    @classmethod
    def cancel_turn(cls, channel: str, turn_id: int, stop_reason: str | None = None) -> TurnExecution | None:
        """The single cancel-request chokepoint (§2.7): both DELETE
        /api/threads/<turn_id> (addressed by the turn's type or, for a
        watchable delegate turn, its channel) and a turn's own self-cancelling
        ability (:meth:`cancel`) route through here — and the single AUTHORITY
        for the turn's terminal state. This does not reach into the running
        thread: the provider call in flight is aborted by its own watcher once
        that reads the flag (:meth:`ProviderService._abort_on_stop`, about one
        poll later), the retry loop then ends the turn instead of resending,
        and a response that came back before the abort landed is discarded by
        :meth:`MessageProcessor._step`'s cancel checkpoint instead of being
        stored. Ending the turn's LIFECYCLE does not have to wait for that: this stamps the live
        (``ended_at IS NULL``) row for this (channel, turn_id) CANCELLED with
        ``ended_at`` set right here, synchronously, so the WS frame and the DB
        state both flip the instant cancel is requested rather than after the
        full generation wall-clock. The row IS the cross-request handle, since
        the turn is (almost always) running in another thread/instance;
        ``turn_id`` is a per-channel monotonic counter, not globally unique, so
        the match is channel-scoped. The broadcast derives from the target row
        alone (§6.7) — its channel and the config its type resolves to — and
        goes out exactly as that turn's own frames do
        (:meth:`MessageProcessor.broadcast_as`). :meth:`finish` becomes a
        no-op once it observes this row already terminal — the row closed here
        always wins, never resurrected or overwritten by the doomed turn's own
        eventual finish().

        The cancel is two parts. :meth:`stamp_cancelled` is the read and every
        write, run here in one ``BEGIN IMMEDIATE`` transaction, so a turn
        settling or a message joining or queuing on another thread lands
        wholly before or wholly after it; :meth:`propagate_cancel` sends the
        frames and stops the children only once that commits, so no surface
        is told of a state it cannot yet read. A caller that must decide inside
        a transaction of its own whether there is a turn to cancel —
        ``stop_subagent``, which reads its subagent's status there — runs the
        stamp inside that transaction and the propagation once it commits,
        and gets the same ordering. Here, a stamp that fails is rolled back
        and logged, and ``None`` comes back, as when no turn is open.

        Stopping a caller's turn also stops every delegate turn its own tool
        calls spawned — a subtask running in the background as much as a
        child the caller is blocked waiting on, or a subagent's turn a message
        it sent opened, however much later — each cancelled in turn by
        :meth:`propagate_cancel` with :data:`STOP_REASON_PARENT_STOPPED`, the
        same reason ``stop_subagent`` stamps when the chat stops one child by
        name; an already-ended child simply returns ``None``. A message one of
        its calls sent to a subagent, still waiting in that subagent's queue
        (:class:`QueuedMessage`), is deleted in the stamp's transaction, so it
        never opens a turn; its text stays in that call's own row. A message
        waiting on the chat is never deleted, even when one of the stopped
        turn's calls queued it — a subagent's reply, keyed by the call that
        spawned it, or a subagent's message to the chat, keyed by its own call
        — so it still opens its chat turn. A subagent's turn closes its
        ``agents`` row ``cancelled`` in the stamp's transaction, keeping the
        turn's last row."""
        try:
            with Database.transaction():
                cancelled = cls.stamp_cancelled(channel, turn_id, stop_reason)
        except Exception as exc:
            logger.warning(
                "[TurnExecutionService] cancel failed for channel=%s turn_id=%s: %s",
                channel, turn_id, exc,
            )
            return None
        if cancelled is None:
            return None
        cls.propagate_cancel(cancelled)
        return cancelled.execution

    @staticmethod
    def stamp_cancelled(channel: str, turn_id: int, stop_reason: str | None) -> CancelledTurn | None:
        """The writes of :meth:`cancel_turn`, inside the caller's transaction,
        sending nothing: the live row for (channel, turn_id) stamped
        CANCELLED with ``stop_reason`` and ``ended_at``, a subagent's
        ``agents`` row on that turn ended ``cancelled`` with the turn's last
        row, and every message the turn's calls left queued off the chat
        deleted. ``None`` when no row for the turn is open. What comes back is
        what :meth:`propagate_cancel` needs once the transaction commits."""
        execution = TurnExecution.open_turn(channel, turn_id)
        if execution is None:
            return None
        execution.cancel_requested = True
        execution.state = TurnExecution.CANCELLED
        execution.ended_at = utc_now().isoformat()
        execution.stop_reason = stop_reason
        execution.save()
        agent = AgentService.end(
            execution.channel, execution.turn_id, execution.state,
            Transcript.last_id(execution.channel, execution.turn_id),
        )
        calls = ToolCall.by_turn(channel, turn_id)
        QueuedMessage.filter_in("tool_call_id", [c.id for c in calls]).filter(
            "channel", Channel.USER.value, "!=",
        ).delete()
        return CancelledTurn(execution=execution, agent=agent, calls=calls)

    @classmethod
    def propagate_cancel(cls, cancelled: CancelledTurn) -> None:
        """What follows :meth:`stamp_cancelled` once its transaction commits:
        the closed row's frame, sent as that turn's own frames are, the frame
        of the agent it ended, and the cancel of every delegate turn the
        stopped turn's calls spawned (:meth:`cancel_turn`, each in its own
        transaction)."""
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415 — the MP imports this module

        execution = cancelled.execution
        MessageProcessor.broadcast_as(execution, execution.channel, cls._row_config(execution))
        if cancelled.agent is not None:
            Websocket.broadcast(AgentService.frame(cancelled.agent))
        # Cascade to every delegate child the caller's tool calls spawned: a
        # child only reads its own row's flag, so it would otherwise run on
        # against a stopped conversation. An already-ended child returns None.
        # Open-ended: a message a call queued may have opened its turn long
        # after the call ended.
        children = Transcript.delegate_turns({cast("int", c.id): (c.created_at, None) for c in cancelled.calls})
        for entry in children.values():
            cls.cancel_turn(
                cast("str", entry["channel"]), cast("int", entry["turn_id"]),
                stop_reason=STOP_REASON_PARENT_STOPPED,
            )

    @staticmethod
    def _row_config(execution: TurnExecution) -> ProcessorConfig | None:
        """The config a row's type resolves to — ``None`` for a row with no
        type (a delegate or internal turn). A stale/unknown type must not raise
        into the caller, so it is logged and treated as no config."""
        if execution.type is None:
            return None
        try:
            return config_for(execution.type)
        except ValueError as exc:
            logger.warning(
                "[TurnExecutionService] unknown config type=%r for turn_id=%s — no broadcast: %s",
                execution.type, execution.turn_id, exc,
            )
            return None

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
        'completed'/'crashed' state.

        A subagent's turn that this closes closes its ``agents`` row in the
        same transaction — ``done``, or ``waiting`` when it ended on a message
        to its caller, with its final reply row, ``cancelled`` or ``failed``
        with the turn's last row (:meth:`_response_id`) — and that
        row's frame follows the lifecycle frame. A turn whose row :meth:`open`
        failed to write has no row to close, but its agent still ends here, so
        it is never left running."""
        current: TurnExecution | None = self.mp.execution
        if current is None or current.id is None:
            self._end_agent_without_row(state)
            return None
        execution = TurnExecution(
            id=current.id,
            channel=current.channel,
            type=current.type,
            turn_id=current.turn_id,
            started_at=current.started_at,
            ended_at=utc_now().isoformat(),
            cancel_requested=current.cancel_requested,
            state=state,
            stop_reason=stop_reason,
            input_id=current.input_id,
        )
        try:
            with self.mp.db.transaction():
                closed = (
                    TurnExecution.filter("id", current.id)
                    .filter("ended_at", None, "IS")
                    .update(ended_at=execution.ended_at, state=state, stop_reason=stop_reason)
                )
                agent = (
                    AgentService.end(execution.channel, execution.turn_id, state, self._response_id(state))
                    if closed else None
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
        self.mp.execution = execution
        self.mp.push_websocket(execution)
        if agent is not None:
            self.mp.announce(AgentService.frame(agent))
        return execution

    def _response_id(self, state: str) -> int | None:
        """The row this turn ends on, which its agent keeps: the final reply
        when it completed, else the turn's last row."""
        if state == TurnExecution.COMPLETED:
            return self.mp.current_transcript_id
        return Transcript.last_id(self.mp.channel, self.mp.turn_id)

    def _end_agent_without_row(self, state: str) -> None:
        """End the agent this turn opened on when the turn has no execution
        row to close it with — logged, never raised, as for the row itself."""
        if self.mp.agent is None:
            return
        try:
            with self.mp.db.transaction():
                agent = AgentService.end(self.mp.channel, self.mp.turn_id, state, self._response_id(state))
        except Exception as exc:
            logger.warning(
                "[TurnExecutionService] ending agent #%s failed: %s", self.mp.agent.agent_id, exc,
            )
            return
        if agent is not None:
            self.mp.announce(AgentService.frame(agent))

    def sweep_orphaned(self) -> list[TurnExecution]:
        """Boot-time reconciliation: every row a killed process left open
        (``ended_at IS NULL``) becomes ``crashed`` — the process that owned
        it is gone, so "working" surviving from a previous run is always
        stale. The whole sweep runs inside one ``self.mp.db`` transaction
        (I6) so a mid-loop write failure rolls back every row it already
        touched rather than stranding a partially-swept boot; returns the
        rows it closed — so boot can reply to each swept delegate's caller — or
        an empty list if the sweep did not commit. Every ``agents`` row still
        running ends ``failed`` in that transaction — a swept subagent's turn,
        and one whose turn never got a row to sweep — and each row's frame goes
        out once it commits."""
        swept: list[TurnExecution] = []
        agents: list[Agent] = []
        try:
            with self.mp.db.transaction():
                for execution in TurnExecution.open_rows():
                    execution.state = TurnExecution.CRASHED
                    execution.stop_reason = STOP_REASON_PROCESS_DEATH
                    execution.ended_at = utc_now().isoformat()
                    execution.save()
                    swept.append(execution)
                agents = AgentService.fail_running()
        except Exception as exc:
            logger.warning("[TurnExecutionService] sweep_orphaned failed: %s", exc)
            return []
        for agent in agents:
            Websocket.broadcast(AgentService.frame(agent))
        return swept

    def latest_for_turn(self) -> TurnExecution | None:
        """The newest row for this turn (channel-scoped, since ``turn_id`` is
        a per-channel counter) — for a caller that holds no live execution
        for it, e.g. the scheduler deciding whether a first-fire's turn
        survived cancellation before persisting its ``turn_id`` for series
        continuity (an inert MP built for the target (channel, turn_id) reads
        it via ``self.mp``)."""
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
