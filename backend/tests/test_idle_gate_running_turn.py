"""Feature tests: an idle-gated job must not start while a user turn is running.

The idle gate used to measure only from the user's last message, so a turn
that ran longer than the 30-minute idle window let discovery start under it.
The two then shared the model, and the user's turn slowed down. These tests
seed real ``transcript`` and ``turn_executions`` rows and run the real
discovery job's gate.
"""

import sqlite3
from datetime import timedelta

import pytest

from configs.enums.channels import Channel
from cron.jobs.discovery import DiscoveryJob
from models.turn_execution import TurnExecution
from services.time_utils import utc_now

pytestmark = pytest.mark.unit


def _user_turn_started(db: sqlite3.Connection, minutes_ago: int) -> TurnExecution:
    """Write what a user turn leaves behind when it starts: the user's input
    row and a still-open execution row, both ``minutes_ago`` in the past."""
    started = utc_now() - timedelta(minutes=minutes_ago)
    cursor = db.execute(
        "INSERT INTO transcript (channel, role, content, created_at) "
        "VALUES (?, 'user', 'research this for me', ?)",
        (Channel.USER.value, started.strftime("%Y-%m-%d %H:%M:%S")),
    )
    return TurnExecution(
        channel=Channel.USER.value,
        type="user",
        turn_id=1,
        started_at=started.isoformat(),
        ended_at=None,
        cancel_requested=False,
        state=TurnExecution.WORKING,
        stop_reason=None,
        input_id=cursor.lastrowid,
    ).save()


def test_gate_stays_closed_while_a_user_turn_runs_past_the_idle_window(
    db: sqlite3.Connection,
) -> None:
    """The user's message is 45 minutes old, but its turn is still running."""
    _user_turn_started(db, minutes_ago=45)

    assert DiscoveryJob()._is_idle() is False, (
        "discovery would start under a running user turn and share the model with it"
    )


def test_gate_opens_once_that_turn_has_ended(db: sqlite3.Connection) -> None:
    """The same turn, now finished: the user is idle and the gate opens."""
    execution = _user_turn_started(db, minutes_ago=45)
    execution.ended_at = utc_now().isoformat()
    execution.state = TurnExecution.COMPLETED
    execution.save()

    assert DiscoveryJob()._is_idle() is True, (
        "the gate stayed closed after the user's turn ended"
    )
