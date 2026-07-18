"""Feature test: an async delegate finishing after its originating turn
crashed must not respawn the doomed work.

Nightly run 1006 evidence: RunAwayLoop killed a browser-looping turn, then six
[MessageProcessor] turn-crashed cycles at ~20-40s spacing as each surviving
async browser delegate delivered and respawned the loop; work still running
3 minutes after the scenario ended; the next five scenarios' sends never
opened a turn.

The fix lives in ``AsyncDelegateRunner._deliver``: before calling
``MessageProcessor.process`` (which forks a new synthesis turn onto the
originating (channel, turn_id)), we probe the originating turn's latest
``turn_executions`` row via an inert ``MessageProcessor``. If the row exists
and its state is ``CRASHED``, we suppress the delivery with a loud warning —
respawning it would resume the exact doomed work the guard just killed.

This test drives that guard against the real SQLite database (via the
``db`` fixture from ``conftest.py``) with the real ``TurnExecutionService``
and ``TranscriptService`` doing every read/write. The only substitution is
``MessageProcessor.process`` itself — we patch it to a ``MagicMock`` to
verify the guard suppresses or allows delivery as expected.
"""

import sqlite3
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.turn_execution import TurnExecution

pytestmark = pytest.mark.unit

# The fully-qualified path ``_deliver`` ultimately calls — patch this, not the
# instance method on the imported module, to verify the guard's decision point.
_PROCESS = "controllers.message_processor.MessageProcessor.process"


def _seed_turn_execution(
    db: sqlite3.Connection,
    channel: str,
    turn_id: int,
    state: str,
    stop_reason: str | None = None,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
) -> TurnExecution:
    """Insert one ``turn_executions`` row directly into the test DB.

    Mirrors what ``TurnExecutionService.open()`` / ``finish()`` do — the
    guard reads via the real ``TurnExecutionService.latest_for_turn()``, so
    the row must be a real DB row, not an in-memory fake.
    """
    now = started_at or datetime.utcnow()
    if ended_at is None:
        ended_at = now + timedelta(seconds=30)
    if stop_reason is None:
        stop_reason = "test_stop" if state == TurnExecution.CRASHED else None

    row = TurnExecution(
        channel=channel,
        type="user",
        turn_id=turn_id,
        started_at=now.isoformat(),
        ended_at=ended_at.isoformat(),
        cancel_requested=False,
        state=state,
        stop_reason=stop_reason,
    )
    row.save()
    return row


def test_crashed_origin_turn_suppresses_delivery(db: sqlite3.Connection) -> None:
    """A delegate finishing after its originating turn was killed by the
    RunAwayLoop guard must not respawn the doomed work — ``process()`` must
    NOT be called.

    Seeds a CRASHED ``turn_executions`` row for the user channel + a chosen
    turn_id, then calls ``_deliver`` with an inert origin mp. The guard
    probes the row via an inert ``MessageProcessor``, sees ``CRASHED``, and
    returns without calling ``process()``.
    """
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    config = UserConfig()
    channel = config.channel
    turn_id = 9001

    # Seed a CRASHED turn (mimics what RunAwayLoop left behind).
    _seed_turn_execution(
        db,
        channel,
        turn_id,
        state=TurnExecution.CRASHED,
        stop_reason="browser_loop_detected",
        started_at=datetime.utcnow() - timedelta(minutes=2),
        ended_at=datetime.utcnow() - timedelta(minutes=1),
    )

    # Build an inert origin mp (I2: no begin(), no DB/WS side effects).
    origin_mp = MessageProcessor(config, turn_id)

    with patch(_PROCESS, new_callable=MagicMock) as mock_process:
        from services.async_delegate_runner import async_delegate_runner
        async_delegate_runner._deliver(origin_mp, "late browser result")

        # The guard must suppress delivery — process() was NOT called.
        mock_process.assert_not_called()


def test_completed_origin_turn_allows_delivery(db: sqlite3.Connection) -> None:
    """A delegate finishing after its originating turn completed normally must
    deliver the result — ``process()`` must be called exactly once, with the
    originating turn_id.

    Seeds a COMPLETED ``turn_executions`` row, then calls ``_deliver``. The
    guard probes the row, sees ``COMPLETED`` (not ``CRASHED``), and allows
    delivery to proceed.
    """
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    config = UserConfig()
    channel = config.channel
    turn_id = 9002

    # Seed a COMPLETED turn.
    _seed_turn_execution(
        db,
        channel,
        turn_id,
        state=TurnExecution.COMPLETED,
        started_at=datetime.utcnow() - timedelta(minutes=5),
        ended_at=datetime.utcnow() - timedelta(minutes=4),
    )

    origin_mp = MessageProcessor(config, turn_id)

    with patch(_PROCESS, new_callable=MagicMock) as mock_process:
        from services.async_delegate_runner import async_delegate_runner
        async_delegate_runner._deliver(origin_mp, "late result")

        # The guard must allow delivery — process() was called exactly once.
        mock_process.assert_called_once()
        # The turn_id argument passed to process() must equal the originating turn_id.
        _call_args = mock_process.call_args
        called_turn_id = _call_args[0][3]  # 4th positional argument
        assert called_turn_id == turn_id


def test_no_execution_row_allows_delivery(db: sqlite3.Connection) -> None:
    """A delegate finishing when no ``turn_executions`` row exists for the
    originating turn must deliver the result — fail-open: absence of evidence
    never suppresses delivery.

    Seeds no row at all for the turn, then calls ``_deliver``. The guard
    probes the row, sees ``None`` (no row), and allows delivery to proceed.
    """
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    config = UserConfig()
    channel = config.channel
    turn_id = 9003

    # No row seeded — the turn has no execution record at all.

    origin_mp = MessageProcessor(config, turn_id)

    with patch(_PROCESS, new_callable=MagicMock) as mock_process:
        from services.async_delegate_runner import async_delegate_runner
        async_delegate_runner._deliver(origin_mp, "late result")

        # The guard must allow delivery — process() was called.
        mock_process.assert_called_once()
        _call_args = mock_process.call_args
        called_turn_id = _call_args[0][3]  # 4th positional argument
        assert called_turn_id == turn_id


def test_cancelled_origin_turn_allows_delivery(db: sqlite3.Connection) -> None:
    """A delegate finishing after its originating turn was cancelled normally
    must deliver the result — ``process()`` must be called.

    Seeds a CANCELLED ``turn_executions`` row, then calls ``_deliver``. The
    guard probes the row, sees ``CANCELLED`` (not ``CRASHED``), and allows
    delivery to proceed. This is important: the cancelled-DELEGATE notice path
    in ``_run``'s ``cancel_event`` branch must keep delivering.
    """
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    config = UserConfig()
    channel = config.channel
    turn_id = 9004

    # Seed a CANCELLED turn.
    _seed_turn_execution(
        db,
        channel,
        turn_id,
        state=TurnExecution.CANCELLED,
        started_at=datetime.utcnow() - timedelta(minutes=3),
        ended_at=datetime.utcnow() - timedelta(minutes=2),
    )

    origin_mp = MessageProcessor(config, turn_id)

    with patch(_PROCESS, new_callable=MagicMock) as mock_process:
        from services.async_delegate_runner import async_delegate_runner
        async_delegate_runner._deliver(origin_mp, "late result")

        # The guard must allow delivery — process() was called.
        mock_process.assert_called_once()
        _call_args = mock_process.call_args
        called_turn_id = _call_args[0][3]  # 4th positional argument
        assert called_turn_id == turn_id


def test_working_origin_turn_allows_delivery(db: sqlite3.Connection) -> None:
    """A delegate finishing while its originating turn is still WORKING must
    deliver the result — ``process()`` must be called.

    Seeds a WORKING ``turn_executions`` row (no ``ended_at``), then calls
    ``_deliver``. The guard probes the row, sees ``WORKING`` (not ``CRASHED``),
    and allows delivery to proceed. This handles the edge case where a
    backgrounded tool finishes before the foreground turn completes — the
    synthesis reply must still land.
    """
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    config = UserConfig()
    channel = config.channel
    turn_id = 9005

    # Seed a WORKING turn (no ended_at).
    _seed_turn_execution(
        db,
        channel,
        turn_id,
        state=TurnExecution.WORKING,
        started_at=datetime.utcnow(),
        ended_at=None,
    )

    origin_mp = MessageProcessor(config, turn_id)

    with patch(_PROCESS, new_callable=MagicMock) as mock_process:
        from services.async_delegate_runner import async_delegate_runner
        async_delegate_runner._deliver(origin_mp, "late result")

        # The guard must allow delivery — process() was called.
        mock_process.assert_called_once()
        _call_args = mock_process.call_args
        called_turn_id = _call_args[0][3]  # 4th positional argument
        assert called_turn_id == turn_id
