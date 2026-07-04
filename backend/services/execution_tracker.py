# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""TurnExecutionService + ExecutionTracker — the DB-backed lifecycle of one
MessageProcessor turn.

A ``turn_executions`` row is the single source of truth for "is this turn
still running, and did someone ask it to stop" — it replaces a process-local
registry, which lives only in one worker's memory and is lost on restart.
``TurnExecutionService`` is the plain CRUD layer (ActTrail-shaped: injectable
db, every write try/except-wrapped and logged, never raising into a turn);
``ExecutionTracker`` is the handle a MessageProcessor holds for its whole
life — it opens the row at construction, answers ``should_stop()`` with a
single-row read, and stamps the terminal state exactly once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from services.time_utils import utc_now

if TYPE_CHECKING:
    import sqlite3

    from services.database_service import DatabaseService
    from services.processor_config import ProcessorConfig

logger = logging.getLogger(__name__)


class TurnExecutionState:
    """The ``turn_executions.state`` vocabulary — one non-terminal, three terminal."""

    WORKING = "working"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    CRASHED = "crashed"


#: Stamped as ``stop_reason`` when the boot sweep finds a row a killed process
#: never got to close — see ``TurnExecutionService.sweep_orphaned`` and run.py.
STOP_REASON_PROCESS_DEATH = "process death"


@dataclass(frozen=True, slots=True)
class TurnExecution:
    """One ``turn_executions`` row. Immutable: every state change is a new
    instance the service reads back after writing, never a mutation."""

    id: int
    channel: str
    type: str | None
    turn_id: int
    started_at: str
    ended_at: str | None
    cancel_requested: bool
    state: str
    stop_reason: str | None

    def to_json(self) -> dict[str, object]:
        """The WS lifecycle frame and the REST DTO share this exact shape —
        one serialization, both surfaces."""
        return {
            "id": self.id,
            "channel": self.channel,
            "type": self.type,
            "turn_id": self.turn_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "cancel_requested": self.cancel_requested,
            "state": self.state,
            "stop_reason": self.stop_reason,
        }

    @staticmethod
    def from_row(row: "sqlite3.Row") -> "TurnExecution":
        return TurnExecution(
            id=row["id"],
            channel=row["channel"],
            type=row["type"],
            turn_id=row["turn_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            cancel_requested=bool(row["cancel_requested"]),
            state=row["state"],
            stop_reason=row["stop_reason"],
        )


def _type_value(config: "ProcessorConfig") -> "str | None":
    cfg_type = config.type()
    return cfg_type.value if cfg_type is not None else None


def _broadcasts_state(execution: "TurnExecution") -> bool:
    """Reconstructs BROADCASTS_STATE from a row alone (no live config object) —
    needed by request_cancel_by_turn, which acts on a turn it holds no ProcessorConfig
    for. A row with no ``type`` names no addressable config; it never broadcasts.
    An unrecognised ``type`` (stale/foreign value) must not raise into the caller —
    this class never raises into a turn — so it defaults to not-broadcasting."""
    if execution.type is None:
        return False
    from services.config_type import ConfigTypeEnum  # noqa: PLC0415
    try:
        return ConfigTypeEnum.get_by_type(execution.type).BROADCASTS_STATE
    except ValueError as exc:
        logger.warning(
            "[ExecutionTracker] unknown config type=%r for turn_id=%s — defaulting to no broadcast: %s",
            execution.type, execution.turn_id, exc,
        )
        return False


def _broadcast_lifecycle(execution: "TurnExecution") -> None:
    """The one WS emitter both ExecutionTracker (live, in-process) and
    TurnExecutionService.request_cancel_by_turn (DB-only, cross-request) call.

    Carries the row's fields verbatim — a deliberate, narrow exception to the
    lean "ids only" WS doctrine (see MessageProcessor.broadcast): the frame IS
    the turn's complete execution record, so a surface never pairs it with a
    second REST call to learn why a turn stopped. ``execution.type`` is the
    ConfigType identity (the same routing field MessageProcessor.broadcast's
    signals carry), so there is no separate frame-kind tag to collide with it —
    a surface recognises this frame by the presence of ``state``, exactly as
    it recognises a broadcast() signal by the presence of ``status``."""
    from services.websocket_broker import WebSocketBroker  # noqa: PLC0415
    try:
        WebSocketBroker().broadcast(execution.to_json())
    except Exception as exc:  # noqa: BLE001 — a dead surface must not break the caller
        logger.debug("[ExecutionTracker] broadcast failed: %s", exc)


class TurnExecutionService:
    """CRUD for ``turn_executions``. Every read returns a real
    :class:`TurnExecution` or ``None`` — never a bare dict."""

    def __init__(self, db: "DatabaseService | None" = None) -> None:
        if db is None:
            from services.database_service import get_shared_db_service  # noqa: PLC0415
            db = get_shared_db_service()
        self._db = db

    def open(self, config: "ProcessorConfig", turn_id: int) -> "TurnExecution | None":
        """Insert this turn's row (state=working) and read it back. None on a
        write failure (logged) — callers that require the row for their own
        contract (e.g. the thread API's send response) must check for it;
        ExecutionTracker itself degrades gracefully (should_stop fails open)."""
        try:
            with self._db.connection() as conn:
                cursor = conn.execute(
                    "INSERT INTO turn_executions (channel, type, turn_id, started_at, state) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (config.channel, _type_value(config), turn_id,
                     utc_now().isoformat(), TurnExecutionState.WORKING),
                )
                row = conn.execute(
                    "SELECT * FROM turn_executions WHERE id = ?", (cursor.lastrowid,)
                ).fetchone()
        except Exception as exc:
            logger.warning("[TurnExecutionService] open failed for turn_id=%s: %s", turn_id, exc)
            return None
        return TurnExecution.from_row(row) if row is not None else None

    def should_stop(self, execution_id: int) -> bool:
        """Single-row PK read of ``cancel_requested`` — the cooperative
        checkpoint a turn's step loop polls. Fails open (never stops the turn)
        on a read error, so a DB hiccup can't silently strand a turn mid-cancel;
        it just misses that one poll and tries again next checkpoint."""
        try:
            with self._db.connection() as conn:
                row = conn.execute(
                    "SELECT cancel_requested FROM turn_executions WHERE id = ?", (execution_id,)
                ).fetchone()
        except Exception as exc:
            logger.warning("[TurnExecutionService] should_stop failed for id=%s: %s", execution_id, exc)
            return False
        return bool(row[0]) if row is not None else False

    def request_cancel_by_turn(self, channel: str, turn_id: int) -> "TurnExecution | None":
        """The single cancel-request chokepoint: DELETE /api/thread/<turn_id>
        (no live tracker reference — the row IS the cross-request handle) and
        a tracker's own self-cancel both flow through here. turn_id is a
        per-channel monotonic counter, not globally unique, so every lookup is
        scoped to ``channel`` too — an unscoped match could flip
        cancel_requested on a different channel's row that happens to share the
        same turn_id. Flips cancel_requested on the live (ended_at IS NULL) row
        for this (channel, turn_id) and returns it; None when no such row
        exists (the idle ack)."""
        try:
            with self._db.connection() as conn:
                conn.execute(
                    "UPDATE turn_executions SET cancel_requested = 1 "
                    "WHERE channel = ? AND turn_id = ? AND ended_at IS NULL",
                    (channel, turn_id),
                )
                row = conn.execute(
                    "SELECT * FROM turn_executions WHERE channel = ? AND turn_id = ? AND ended_at IS NULL",
                    (channel, turn_id),
                ).fetchone()
        except Exception as exc:
            logger.warning(
                "[TurnExecutionService] request_cancel_by_turn failed for channel=%s turn_id=%s: %s",
                channel, turn_id, exc,
            )
            return None
        if row is None:
            return None
        execution = TurnExecution.from_row(row)
        if _broadcasts_state(execution):
            _broadcast_lifecycle(execution)
        return execution

    def latest_for_turn(self, channel: str, turn_id: int) -> "TurnExecution | None":
        """The newest row for (channel, turn_id) — used where a caller needs a
        turn's terminal state but holds no live tracker for it (e.g. the
        scheduler deciding whether a first fire's turn survived cancellation
        before persisting its turn_id for series continuity). Channel-scoped
        for the same reason as request_cancel_by_turn: turn_id alone is not
        unique across channels."""
        try:
            with self._db.connection() as conn:
                row = conn.execute(
                    "SELECT * FROM turn_executions WHERE channel = ? AND turn_id = ? "
                    "ORDER BY id DESC LIMIT 1",
                    (channel, turn_id),
                ).fetchone()
        except Exception as exc:
            logger.warning(
                "[TurnExecutionService] latest_for_turn failed for channel=%s turn_id=%s: %s",
                channel, turn_id, exc,
            )
            return None
        return TurnExecution.from_row(row) if row is not None else None

    def finish(self, execution_id: int, state: str, stop_reason: "str | None" = None) -> "TurnExecution | None":
        """Stamp the terminal state ONCE — ``ended_at``/``state``/``stop_reason``
        only; ``cancel_requested`` is never touched here, so a crash after a
        cancel request preserves both facts with zero special-casing."""
        try:
            with self._db.connection() as conn:
                conn.execute(
                    "UPDATE turn_executions SET state = ?, stop_reason = ?, ended_at = ? WHERE id = ?",
                    (state, stop_reason, utc_now().isoformat(), execution_id),
                )
                row = conn.execute(
                    "SELECT * FROM turn_executions WHERE id = ?", (execution_id,)
                ).fetchone()
        except Exception as exc:
            logger.warning("[TurnExecutionService] finish failed for id=%s: %s", execution_id, exc)
            return None
        return TurnExecution.from_row(row) if row is not None else None

    def sweep_orphaned(self) -> int:
        """Boot-time reconciliation: every row a killed process left open
        (``ended_at IS NULL``) becomes ``crashed`` — the process that owned it
        is gone, so "working" surviving from a previous run is always stale.
        Returns the number of rows closed."""
        try:
            with self._db.connection() as conn:
                cursor = conn.execute(
                    "UPDATE turn_executions SET state = ?, stop_reason = ?, ended_at = ? "
                    "WHERE ended_at IS NULL",
                    (TurnExecutionState.CRASHED, STOP_REASON_PROCESS_DEATH, utc_now().isoformat()),
                )
                count = cursor.rowcount
        except Exception as exc:
            logger.warning("[TurnExecutionService] sweep_orphaned failed: %s", exc)
            return 0
        return count


class ExecutionTracker:
    """The per-turn lifecycle handle a MessageProcessor holds for its whole
    life. Opens the ``turn_executions`` row at construction (before any LLM
    call), answers ``should_stop()`` with a single-row read, and stamps the
    terminal state exactly once via :meth:`set_done` / :meth:`set_cancelled` /
    :meth:`set_crashed` — each broadcasts the fresh row to any listening
    surface."""

    def __init__(self, config: "ProcessorConfig", turn_id: int,
                 service: "TurnExecutionService | None" = None) -> None:
        self._service = service or TurnExecutionService()
        self._broadcasts = config.BROADCASTS_STATE
        self._execution = self._service.open(config, turn_id)
        self._emit()

    @property
    def execution(self) -> "TurnExecution | None":
        """The current row snapshot — read by the thread API's synchronous
        send response (no WS round-trip needed to know the turn's identity)."""
        return self._execution

    def should_stop(self) -> bool:
        """Has a cancel been requested for this execution? No row to poll
        (see :meth:`TurnExecutionService.open`) fails open — never stops a
        turn that has no DB-backed lifecycle."""
        if self._execution is None:
            return False
        return self._service.should_stop(self._execution.id)

    def request_cancel(self) -> None:
        """Ask this turn to stop — used by a turn's own self-cancelling
        ability. Routes through the same (channel, turn_id)-keyed chokepoint
        DELETE /api/thread/<turn_id> uses, so both paths behave identically."""
        if self._execution is None:
            return
        execution = self._service.request_cancel_by_turn(
            self._execution.channel, self._execution.turn_id,
        )
        if execution is not None:
            self._execution = execution

    def set_done(self) -> None:
        self._finish(TurnExecutionState.COMPLETED)

    def set_cancelled(self) -> None:
        self._finish(TurnExecutionState.CANCELLED)

    def set_crashed(self, reason: str) -> None:
        self._finish(TurnExecutionState.CRASHED, reason)

    def _finish(self, state: str, stop_reason: "str | None" = None) -> None:
        if self._execution is None:
            return
        execution = self._service.finish(self._execution.id, state, stop_reason)
        if execution is None:
            # The persisted UPDATE failed (already logged loudly inside
            # TurnExecutionService.finish) — the row may still read "working"
            # in the DB, but this turn's process has genuinely stopped. A
            # surface must never hang on a spinner because a DB write failed:
            # reflect the terminal state in-memory and emit best-effort
            # anyway. The next boot sweep reconciles the stale row.
            execution = replace(
                self._execution, state=state, stop_reason=stop_reason,
                ended_at=utc_now().isoformat(),
            )
        self._execution = execution
        self._emit()

    def _emit(self) -> None:
        if not self._broadcasts or self._execution is None:
            return
        _broadcast_lifecycle(self._execution)
