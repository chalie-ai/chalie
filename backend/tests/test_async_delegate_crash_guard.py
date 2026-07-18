"""Feature test: an async delegate finishing after its originating turn
crashed must not respawn the doomed work.

Observed live: RunAwayLoop killed a browser-looping turn, then each surviving
async delegate delivered its late result and respawned the identical loop — a
cascade of [MessageProcessor] turn-crashed cycles at ~20-40s spacing that kept
the channel busy for minutes while no user message could open a turn.

The fix lives in ``AsyncDelegateRunner._deliver``: before calling
``MessageProcessor.process`` (which forks a new synthesis turn onto the
originating (channel, turn_id)), it probes the originating turn's latest
``turn_executions`` row via an inert ``MessageProcessor``. A ``CRASHED`` row
suppresses the delivery with a loud warning; any other state — or no row at
all — delivers exactly as before (fail-open).

Drives the guard against the real SQLite database (``db`` fixture) with the
real ``TurnExecutionService`` doing every read. The only substitution is
``MessageProcessor.process`` itself, patched to a ``MagicMock`` to observe the
guard's decision.
"""

import sqlite3
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.turn_execution import TurnExecution
from services.async_delegate_runner import async_delegate_runner

pytestmark = pytest.mark.unit

# The fully-qualified path ``_deliver`` ultimately calls — patch this, not the
# instance method on the imported module, to verify the guard's decision point.
_PROCESS = "controllers.message_processor.MessageProcessor.process"

_TURN_ID = 9001


def _seed_turn_execution(channel: str, turn_id: int, state: str) -> None:
    """Insert one real ``turn_executions`` row — the guard reads it via the
    real ``TurnExecutionService.latest_for_turn()``. A ``WORKING`` row has no
    ``ended_at`` (the turn is still running); every settled state does."""
    now = datetime.utcnow()
    TurnExecution(
        channel=channel,
        type="user",
        turn_id=turn_id,
        started_at=now.isoformat(),
        ended_at=None if state == TurnExecution.WORKING else (now + timedelta(seconds=30)).isoformat(),
        cancel_requested=False,
        state=state,
        stop_reason="browser_loop_detected" if state == TurnExecution.CRASHED else None,
    ).save()


def test_crashed_origin_turn_suppresses_delivery(db: sqlite3.Connection) -> None:
    """A delegate finishing after its originating turn was killed by the
    RunAwayLoop guard must not respawn the doomed work — ``process()`` is
    never called."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    config = UserConfig()
    _seed_turn_execution(config.channel, _TURN_ID, TurnExecution.CRASHED)
    origin_mp = MessageProcessor(config, _TURN_ID)  # inert (I2)

    with patch(_PROCESS, new_callable=MagicMock) as mock_process:
        async_delegate_runner._deliver(origin_mp, "late browser result")

    mock_process.assert_not_called()


@pytest.mark.parametrize(
    "state",
    [None, TurnExecution.COMPLETED, TurnExecution.CANCELLED, TurnExecution.WORKING],
    ids=["no-row", "completed", "cancelled", "working"],
)
def test_non_crashed_origin_allows_delivery(db: sqlite3.Connection, state: str | None) -> None:
    """Every non-crashed origin delivers exactly as before — COMPLETED and
    CANCELLED settled turns (the cancelled-DELEGATE notice path must keep
    delivering), a still-WORKING turn (a backgrounded tool may finish before
    the foreground turn ends), and no execution row at all (fail-open:
    absence of evidence never suppresses delivery). ``process()`` is called
    exactly once with the originating turn_id."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    config = UserConfig()
    if state is not None:
        _seed_turn_execution(config.channel, _TURN_ID, state)
    origin_mp = MessageProcessor(config, _TURN_ID)  # inert (I2)

    with patch(_PROCESS, new_callable=MagicMock) as mock_process:
        async_delegate_runner._deliver(origin_mp, "late result")

    mock_process.assert_called_once()
    assert mock_process.call_args[0][3] == _TURN_ID  # originating turn_id, 4th positional
