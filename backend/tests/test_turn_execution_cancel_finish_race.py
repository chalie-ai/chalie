# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test — ``TurnExecutionService.cancel()`` vs ``finish()`` no-clobber
race, the exact defect that let a cancelled turn resurface with
``cancel_requested=0``.

Every provider client is one blocking, non-streaming call with no mid-flight
abort hook, so a cancel can land (a separate connection/thread stamping the
row CANCELLED, ``ended_at`` set) WHILE the turn's own drive thread is still
holding a stale, pre-cancel in-memory copy of the execution row and is about
to call its own ``finish()`` on a clean return. Before the fix, ``finish()``
built a fresh row from that stale in-memory copy (``cancel_requested=False``,
``state=working``) and saved it unconditionally — silently overwriting the
already-cancelled DB row back to a fabricated "completed" state with
``cancel_requested`` reverted to ``False``. The fix makes the close a single
atomic conditional ``UPDATE ... WHERE id = ? AND ended_at IS NULL``: when a
cancel already closed the row, that update matches zero rows and ``finish()``
re-reads and returns the cancelled row as-is, never touching it.

Drives two independent, inertly-constructed ``MessageProcessor`` instances —
one standing in for the turn's own drive thread, one for the DELETE request's
handler, exactly as production builds them — against the real, fully-migrated
SQLite database. No mocks.
"""

import sqlite3

import pytest

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.turn_execution import TurnExecution

pytestmark = pytest.mark.unit

_CHANNEL = "user"


def test_finish_does_not_clobber_a_cancel_that_lands_first(db: sqlite3.Connection) -> None:
    """The race: cancel() closes the row CANCELLED first; the doomed turn's
    own finish(COMPLETED) call — still holding the stale pre-cancel in-memory
    row — must observe zero rows to close and return the cancelled row
    untouched, not resurrect it as completed or revert cancel_requested."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    turn_id = 8501
    running = MessageProcessor(UserConfig(), turn_id, "Do the long thing.")  # the doomed turn's own mp, inert (I2)
    running.uid = running.transcript_service.append_input(running.raw_input)
    running.current_transcript_id = running.uid
    opened = running.turn_execution_service.open()
    assert opened is not None and opened.ended_at is None  # sanity: real open, still live

    # A DIFFERENT inert mp for the same (channel, turn_id) — exactly how the
    # real DELETE handler builds its instance (api.threads.ThreadItemResource.delete).
    canceller = MessageProcessor(UserConfig(), turn_id)
    cancelled = canceller.turn_execution_service.cancel()
    assert cancelled is not None
    assert cancelled.state == TurnExecution.CANCELLED
    assert bool(cancelled.cancel_requested) is True

    # ``running.execution`` is still the STALE pre-cancel row (state=working,
    # cancel_requested=False) — exactly what the doomed turn's own ``_drive()``
    # thread would be holding when it reaches its own ``finish()`` call.
    result = running.turn_execution_service.finish(TurnExecution.COMPLETED)

    assert result is not None
    assert result.state == TurnExecution.CANCELLED  # never resurrected as completed
    assert bool(result.cancel_requested) is True  # never reverted by the stale in-memory copy
    assert result.ended_at == cancelled.ended_at  # the cancel's own timestamp survives untouched


def test_finish_closes_normally_when_no_concurrent_cancel(db: sqlite3.Connection) -> None:
    """The should-fire side of the same branch: with no race, finish()'s
    atomic UPDATE finds the still-live row and closes it for real — proving
    the race guard above is a genuine conditional, not a permanent no-op."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    turn_id = 8502
    mp = MessageProcessor(UserConfig(), turn_id, "Do the quick thing.")  # inert (I2)
    mp.uid = mp.transcript_service.append_input(mp.raw_input)
    mp.current_transcript_id = mp.uid
    opened = mp.turn_execution_service.open()
    assert opened is not None and opened.ended_at is None  # sanity: real open, still live

    result = mp.turn_execution_service.finish(TurnExecution.COMPLETED)

    assert result is not None
    assert result.state == TurnExecution.COMPLETED
    assert result.ended_at is not None
    assert bool(result.cancel_requested) is False
