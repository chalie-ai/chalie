# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Regression guard for ``api.threads.serialize_turn``'s ``working`` field on
the ``schedule`` channel (``external_turn_id=True`` — the schedule's own
integer id IS the turn_id, and every tick reuses it).

``working`` used to derive from ``Transcript.settle0`` (a "has this turn ever
settled" history check), which is wrong in both directions for a growing
schedule thread: a schedule that has never fired has zero transcript rows, so
``settle0`` returns ``None`` and ``working`` was pinned ``True`` forever (a
perpetual spinner with no content); once the first tick settles, ``settle0``
is non-``None`` forever, so ``working`` was pinned ``False`` even while a
later tick is actively running. The fix ties ``working`` to the real
execution-lifecycle table instead: ``TurnExecution.open_turn(channel,
turn_id) is not None``.

Drives the real ``TurnExecutionService``/``TranscriptService`` through a real
inert ``MessageProcessor`` under ``ScheduledConfig`` (construction is
side-effect-free, mirrors ``DELETE /api/thread/<turn_id>``'s own cancel path
in ``api/threads.py``) against the real, fully-migrated SQLite database — the
same collaborators production uses to open/finish a turn's execution row and
append its transcript rows. No mocks, no hand-rolled DDL.
"""

import sqlite3
from typing import cast

import pytest

from api.threads import serialize_turn
from configs.channels.scheduled import ScheduledConfig
from controllers.message_processor import MessageProcessor
from models.turn_execution import TurnExecution

pytestmark = pytest.mark.unit

_CHANNEL = "schedule"


def test_never_fired_schedule_is_not_working(db: sqlite3.Connection) -> None:
    """A schedule that has never ticked has zero transcript rows AND zero
    turn_executions rows for its turn_id. Before the fix, ``settle0`` on an
    empty turn returns ``None`` and ``working`` was pinned ``True`` forever —
    a perpetual "thinking" spinner with no content. The fix must report
    ``working=False`` here since there is no live execution row."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    turn_id = 9001

    result = serialize_turn(_CHANNEL, turn_id)

    assert result["working"] is False
    assert result["messages"] == []


def test_reopened_turn_with_prior_settle_is_working(db: sqlite3.Connection) -> None:
    """The second regression direction: a schedule's FIRST tick settles (so
    ``Transcript.settle0`` is permanently non-``None`` for this turn_id from
    here on — schedules reuse one turn_id forever), then a LATER tick reopens
    the same turn_id and is still running. Before the fix, ``working`` read
    ``settle is None`` — permanently ``False`` once any tick had ever
    settled, even while a later tick is actively in flight. The fix must
    report ``working=True`` here because a live (``ended_at IS NULL``)
    turn_executions row exists, regardless of transcript history."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    turn_id = 9002

    first_tick = MessageProcessor(ScheduledConfig(), turn_id)  # inert (I2)
    assert first_tick.turn_execution_service.open() is not None
    first_tick.transcript_service.append_assistant("First tick settled.")
    assert first_tick.turn_execution_service.finish(TurnExecution.COMPLETED) is not None

    later_tick = MessageProcessor(ScheduledConfig(), turn_id)  # inert (I2)
    execution = later_tick.turn_execution_service.open()
    assert execution is not None  # sanity: the real open-row write succeeded

    result = serialize_turn(_CHANNEL, turn_id)

    assert result["working"] is True


def test_finished_execution_with_settled_reply_is_not_working(db: sqlite3.Connection) -> None:
    """A tick that ran to completion has a turn_executions row with
    ``ended_at`` stamped, plus a settled assistant transcript row. Before the
    fix, ``settle0`` on a settled turn is non-``None`` forever, so ``working``
    happened to read ``False`` here too — but for the wrong reason (history,
    not liveness), which breaks the moment a LATER tick reopens the same
    turn_id (covered by the two tests above). This asserts the correct-for-
    the-right-reason outcome: no OPEN execution row → ``working=False``, and
    the settled content is visible in ``messages``."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    turn_id = 9003
    mp = MessageProcessor(ScheduledConfig(), turn_id)  # inert (I2): 0 db, 0 ws at construction

    execution = mp.turn_execution_service.open()
    assert execution is not None
    mp.transcript_service.append_assistant("Standup reminder: 9am daily sync.")
    finished = mp.turn_execution_service.finish(TurnExecution.COMPLETED)
    assert finished is not None and finished.ended_at is not None  # sanity: real close

    result = serialize_turn(_CHANNEL, turn_id)

    assert result["working"] is False
    messages = cast("list[dict[str, object]]", result["messages"])
    contents = [m["content"] for m in messages]
    assert "Standup reminder: 9am daily sync." in contents
