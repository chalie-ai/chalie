# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — the turn_executions DB-backed execution lifecycle.

A ``turn_executions`` row is the single source of truth for "is this turn
still running, and did someone ask it to stop": ``ExecutionTracker`` opens it
in ``MessageProcessor.__init__`` (before any LLM call), every cooperative
checkpoint reads it through ``should_stop()``, and ``MessageProcessor._run()``
stamps its terminal state exactly once — cancelled on ``_TurnCancelled``,
crashed on any other exception, completed at the end of ``_end_turn``. These
tests drive the REAL production classes (``ExecutionTracker``,
``TurnExecutionService``, ``MessageProcessor``) against the real test db and
the real ``WebSocketBroker`` — zero mocks.

Two tests avoid ``MessageProcessor._setup()`` on the ``user`` channel (it runs
the deliberation gate + turn-zero memory flashback — real, offline, but
orthogonal to this file's contract) by using the same delegate-config pattern
``test_turn_signal_lifecycle.py`` already established: a non-``user`` channel
shares the identical ``_run()``/``_step()``/``_TurnCancelled`` machinery this
file is actually pinning.
"""

import json
import sqlite3
import threading
from collections.abc import Generator
from typing import cast

import pytest
from flask.testing import FlaskClient

from configs.channels.user import UserConfig
from configs.channels.web_search import WebSearchConfig
from services.act_trail import ActTrail
from services.execution_tracker import (
    STOP_REASON_PROCESS_DEATH,
    ExecutionTracker,
    TurnExecutionService,
    TurnExecutionState,
)
from services.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig
from services.websocket_broker import WebSocketBroker

pytestmark = pytest.mark.unit

_CHAT = ProcessorConfig.PolicyChannel.CHAT


class _Surface:
    """A real consumer at the WS boundary — one open device/tab. The broker
    pushes JSON strings to it exactly as it does to a browser connection."""

    def __init__(self) -> None:
        self.received: list[dict[str, object]] = []

    def send(self, raw: str) -> None:
        self.received.append(json.loads(raw))


@pytest.fixture
def broker() -> Generator[WebSocketBroker, None, None]:
    """The real process-wide broker singleton, emptied before and after so
    leaked connections from other tests cannot perturb the frame assertions."""
    b = WebSocketBroker()
    with b._lock:
        b._connections.clear()
    try:
        yield b
    finally:
        with b._lock:
            b._connections.clear()


def test_hot_path_turn_opens_working_and_completes_with_full_lifecycle_frames(
    db: sqlite3.Connection, broker: WebSocketBroker,
) -> None:
    """A user turn's real construction opens its turn_executions row
    (state=working) and broadcasts it; running the real persistence chokepoints
    (_store_row, _end_turn) drives it to state=completed with ended_at stamped.
    Both lifecycle frames on the wire carry the row's FULL object (not just an
    id) — a surface never needs a second REST call to learn why a turn
    stopped."""
    surface = _Surface()
    broker.connect(surface)

    mp = MessageProcessor(UserConfig(), raw_input="tell me a fact")
    opened = mp.execution
    assert opened is not None
    assert opened.state == "working"
    assert opened.ended_at is None
    assert opened.cancel_requested is False

    mp._store_row("Here is your answer.")
    mp._end_turn("Here is your answer.")

    completed = mp.execution
    assert completed is not None
    assert completed.state == "completed"
    assert completed.ended_at is not None
    assert completed.id == opened.id

    lifecycle_frames = [f for f in surface.received if "state" in f]
    assert [f["state"] for f in lifecycle_frames] == ["working", "completed"], (
        f"expected exactly a working then a completed lifecycle frame; got {surface.received}"
    )
    assert lifecycle_frames[0] == opened.to_json(), "the open frame must carry the row verbatim"
    assert lifecycle_frames[1] == completed.to_json(), "the completed frame must carry the fresh row verbatim"

    # Cross-step proof: the DB row a fresh read sees matches what the surface
    # was told, not just what the in-memory dataclass claims.
    row = db.execute("SELECT state, ended_at FROM turn_executions WHERE id = ?", (opened.id,)).fetchone()
    assert row["state"] == "completed" and row["ended_at"] is not None


def test_cancellation_checkpoint_purges_the_chains_own_rows_and_resolves_cancelled(
    db: sqlite3.Connection,
) -> None:
    """A cancel requested before the step loop's first checkpoint stops the
    turn without ever reaching the provider — the SAME _TurnCancelled path
    every checkpoint raises into, caught in exactly one place (_run), so it
    resolves to state=cancelled, never crashed. This chain owns the WHOLE
    turn (no fork — self.uid is the turn's own anchor, so the id floor is the
    turn's first row), so _cleanup_cancelled purges every row it wrote: both
    the pre-cancel step's transcript row and its tool call are gone.
    request_cancel_by_turn is channel-scoped — the same (channel, turn_id)
    chokepoint DELETE /api/thread/<turn_id> uses."""
    mp = MessageProcessor(WebSearchConfig(_CHAT), raw_input="research something")
    mp._store_row("partial progress recorded before the cancel arrived")
    ActTrail().record(tool_name="memory", params={"action": "recall"}, result="ok", transcript_id=cast(int, mp.anchor))

    execution = TurnExecutionService().request_cancel_by_turn(mp.config.channel, cast(int, mp.turn_id))
    assert execution is not None and execution.cancel_requested is True

    result = mp._run()

    assert result == ""
    final = mp.execution
    assert final is not None
    assert final.state == "cancelled", "a should_stop()==True checkpoint must resolve to cancelled, not crashed"
    assert final.ended_at is not None
    assert final.cancel_requested is True
    assert final.stop_reason is None, "set_cancelled() carries no stop_reason"

    # This chain owns the turn outright (no fork) — the purge clears it entirely.
    row_count = db.execute(
        "SELECT COUNT(*) FROM transcript WHERE channel = ? AND turn_id = ?",
        (mp.config.channel, mp.turn_id),
    ).fetchone()[0]
    assert row_count == 0, "a non-forked cancel purges the whole turn — anchor and pre-cancel step both gone"
    tool_rows = ActTrail().fetch_by_transcript_id(cast(int, mp.anchor))
    assert tool_rows == [], "the pre-cancel tool call is purged along with its parent transcript row"


def test_crash_preserves_cancel_requested_in_both_directions(db: sqlite3.Connection) -> None:
    """finish() never touches cancel_requested — a crash landing after a cancel
    was already requested preserves BOTH facts at once (state=crashed AND
    cancel_requested=1); a crash with no prior cancel request leaves
    cancel_requested=0. The two facts are independent axes that must never be
    conflated in either direction."""
    cancelled_then_crashed = ExecutionTracker(WebSearchConfig(_CHAT), turn_id=501)
    cancelled_then_crashed.request_cancel()
    cancelled_then_crashed.set_crashed("provider retries exhausted")

    execution = cancelled_then_crashed.execution
    assert execution is not None
    assert execution.state == "crashed"
    assert execution.cancel_requested is True
    assert execution.stop_reason == "provider retries exhausted"
    assert execution.ended_at is not None

    crashed_without_cancel = ExecutionTracker(WebSearchConfig(_CHAT), turn_id=502)
    crashed_without_cancel.set_crashed("provider retries exhausted")

    other_execution = crashed_without_cancel.execution
    assert other_execution is not None
    assert other_execution.state == "crashed"
    assert other_execution.cancel_requested is False, "a crash must not fabricate a cancel request that was never made"


def test_boot_sweep_crashes_orphaned_rows_and_leaves_ended_rows_untouched(db: sqlite3.Connection) -> None:
    """The startup sweep (run.py, between _init_database and
    _run_startup_migrations) closes every row a killed process left open as
    crashed/'process death' — but a row that already ended before the sweep
    ran is untouched, not just "still completed" by coincidence: its ended_at
    timestamp is provably the SAME value finish() originally stamped."""
    svc = TurnExecutionService()
    orphan = svc.open(UserConfig(), 601)
    ended = svc.open(UserConfig(), 602)
    assert orphan is not None and ended is not None
    finished = svc.finish(ended.id, TurnExecutionState.COMPLETED)
    assert finished is not None

    closed = svc.sweep_orphaned()

    assert closed == 1
    swept = db.execute(
        "SELECT state, stop_reason, ended_at FROM turn_executions WHERE id = ?", (orphan.id,)
    ).fetchone()
    assert swept["state"] == "crashed"
    assert swept["stop_reason"] == STOP_REASON_PROCESS_DEATH
    assert swept["ended_at"] is not None

    untouched = db.execute(
        "SELECT state, ended_at FROM turn_executions WHERE id = ?", (ended.id,)
    ).fetchone()
    assert untouched["state"] == "completed"
    assert untouched["ended_at"] == finished.ended_at, "the sweep must not restamp a row that already ended"


def test_post_send_returns_full_turn_execution_dto_with_live_turn_id(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """POST /api/thread/-1 returns the turn's freshly-opened turn_execution row
    inline — not just a bare turn_id — so the FE holds the whole lifecycle
    handle with no WS round-trip. The id it names is real: a fresh read of the
    same row from the db matches the response body field for field."""
    client, db_conn, _store = authed_client

    resp = client.post('/api/thread/-1', data={'text': 'hello there'})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body['state'] == 'working'
    assert body['cancel_requested'] is False
    assert body['channel'] == 'user'
    assert body['ended_at'] is None
    assert isinstance(body['id'], int)
    turn_id = cast(int, body['turn_id'])

    row = db_conn.execute(
        "SELECT state, cancel_requested, ended_at, turn_id FROM turn_executions WHERE id = ?", (body['id'],)
    ).fetchone()
    assert row is not None
    assert row["state"] == "working"
    assert bool(row["cancel_requested"]) is False
    assert row["ended_at"] is None
    assert row["turn_id"] == turn_id

    # Stop the background turn the dispatch spawned (no provider in test env) and
    # WAIT for it to actually exit — MessageProcessor.run() names its daemon
    # thread f"turn-{turn_id}" precisely so a caller can find and join it. A
    # fire-and-forget DELETE only requests cooperative cancellation; the thread's
    # real stop point is asynchronous. Joining it here (a real synchronization
    # primitive, not a sleep) keeps this test's db teardown from racing a write
    # still in flight, which would otherwise leak into whichever test runs next.
    client.delete(f'/api/thread/{turn_id}')
    turn_thread = next((t for t in threading.enumerate() if t.name == f"turn-{turn_id}"), None)
    if turn_thread is not None:
        turn_thread.join(timeout=10)
        assert not turn_thread.is_alive(), "the background turn must exit before the test's db is torn down"
