# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the silent context hole a crashed turn used to open.

A turn that dies mid-flight leaves its input row behind: the user still sees
that message in the thread, but the model's history read used to drop it. The
MAIN view floored every turn at its ``settle0`` and skipped any turn that had
none — a condition that is true of a turn still running, true of the memory
step, and ALSO true of a turn that crashed. The first two belong out of the
history; the third is a real message the user is looking at, so dropping it
made the model blind to something on the user's screen with nothing to show
that it had happened.

The rule now is that the two views agree: whatever the rendered thread shows,
the model reads. So each test below asserts BOTH sides of the same turn —
``Transcript.recent_threads`` (the collapsed feed the FE renders) and
``TranscriptService.read()`` (the history ``PromptService`` hands the model) —
and the point is never one view's answer on its own, it is that they match.

Four turn shapes cover the fork:

1. crashed, reply-less — shown by the feed, so it must reach the model;
2. cancelled, reply-less — hidden by the feed, so it must NOT reach the model
   (the mirror-image hole: making crashes visible must not make cancels leak);
3. the memory step's own row — never conversation on any view;
4. the turn in flight — its own input row is the prompt's user body already,
   and must not also arrive as history.

Drives the real ``TranscriptService``/``TurnExecutionService`` through real,
inertly-constructed ``MessageProcessor`` instances (the pattern in
``test_thread_cancel_orphan_exclusion.py``) and the real ``Transcript`` model
classmethods for the feed — all against the real, fully-migrated SQLite
database. No mocks.
"""

import sqlite3

import pytest

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.transcript import Transcript
from models.turn_execution import TurnExecution
from services.memory_step_service import memory_step_config

pytestmark = pytest.mark.unit

_CHANNEL = "user"


def _start_turn(mp: MessageProcessor) -> MessageProcessor:
    """Open a real MAIN turn through the production writes ``begin()`` makes —
    allocate the per-channel turn id, write the anchoring input row, open the
    execution row — minus the drive thread. Returns *mp* for chaining."""
    mp.turn_id = mp.transcript_service.allocate_turn()
    mp.uid = mp.transcript_service.append_input(mp.raw_input)
    mp.current_transcript_id = mp.uid
    assert mp.turn_execution_service.open() is not None  # sanity: real open-row write
    return mp


def _settled_exchange(question: str, answer: str) -> MessageProcessor:
    """A real, completed MAIN turn — the ordinary history every test needs
    around the turn under inspection, so an empty result can never pass by
    accident."""
    mp = _start_turn(MessageProcessor(UserConfig(), raw_input=question))  # inert (I2)
    mp.transcript_service.append_assistant(answer)
    mp.turn_execution_service.finish(TurnExecution.COMPLETED)
    return mp


def _feed_turn_ids() -> list[object]:
    """The turn ids the collapsed thread feed renders — the FE's view."""
    return [t.get("turn_id") for t in Transcript.recent_threads(_CHANNEL)]


def _model_history() -> tuple[MessageProcessor, list[int]]:
    """A fresh MAIN turn's history read — the model's view. Returns the reader
    (so its own in-flight row can be asserted on) and the row ids it saw."""
    reader = _start_turn(MessageProcessor(UserConfig(), raw_input="So where does that leave us?"))  # inert (I2)
    return reader, [int(r.id) for r in reader.transcript_service.read() if r.id is not None]


def test_crashed_turn_reaches_the_model_exactly_as_the_feed_shows_it(
    db: sqlite3.Connection,
) -> None:
    """The reported bug. A turn dies mid-flight — its row survives in the DB
    and renders in the feed, so the model must read it too. It never settled,
    so there is no settle0 to floor on and the turn reads whole."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    _settled_exchange("What's on my calendar today?", "You have no events today.")

    crashed = _start_turn(MessageProcessor(UserConfig(), raw_input="What's wrong in this screenshot?"))  # inert (I2)
    finished = crashed.turn_execution_service.finish(TurnExecution.CRASHED, "disk I/O error")
    assert finished is not None and finished.state == TurnExecution.CRASHED  # sanity: real crash stamp
    assert Transcript.settle0(_CHANNEL, crashed.turn_id) is None  # sanity: nothing ever settled

    _reader, history_ids = _model_history()

    assert crashed.turn_id in _feed_turn_ids()  # the user can see it …
    assert crashed.uid in history_ids  # … so the model must too


def test_cancelled_orphan_is_hidden_from_the_model_exactly_as_the_feed_hides_it(
    db: sqlite3.Connection,
) -> None:
    """The mirror image. A turn stopped before any reply is stripped from the
    rendered thread — the user was shown nothing for it — so it must stay out
    of the model's history as well. Both turns here are unsettled and
    reply-less; only the execution state tells them apart, which is exactly
    what the shared cancel rule keys on."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    crashed = _start_turn(MessageProcessor(UserConfig(), raw_input="Analyse the attached image."))  # inert (I2)
    crashed.turn_execution_service.finish(TurnExecution.CRASHED, "disk I/O error")

    cancelled = _start_turn(MessageProcessor(UserConfig(), raw_input="Actually, forget it."))  # inert (I2)
    stopped = cancelled.turn_execution_service.cancel()
    assert stopped is not None and stopped.state == TurnExecution.CANCELLED  # sanity: real cancel

    _reader, history_ids = _model_history()
    feed_ids = _feed_turn_ids()

    assert cancelled.turn_id not in feed_ids
    assert cancelled.uid not in history_ids
    # …and the crash beside it is still carried, so the exclusion is keyed on
    # the cancel and not on "never settled" all over again.
    assert crashed.turn_id in feed_ids
    assert crashed.uid in history_ids


def test_memory_step_row_never_enters_the_model_history(
    db: sqlite3.Connection,
) -> None:
    """The memory step writes a ``role='memory'`` input row on the channel it
    reads and settles nothing, so it too has no settle0. It is turn plumbing,
    not conversation, and is excluded by its role — the guarantee the old
    settle0 skip was providing by side effect."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    settled = _settled_exchange("Remind me to call the plumber.", "Noted.")

    step = MessageProcessor(  # inert (I2)
        memory_step_config(UserConfig(), [settled.uid] if settled.uid is not None else []),
        raw_input="<transcript window>",
    )
    step.turn_id = step.transcript_service.allocate_turn()
    step.uid = step.transcript_service.append_input(step.raw_input)
    assert step.uid is not None  # sanity: the step really does write an input row
    assert Transcript.settle0(_CHANNEL, step.turn_id) is None  # sanity: the step settles nothing

    _reader, history_ids = _model_history()

    assert step.uid not in history_ids
    assert settled.uid in history_ids  # the conversation it distilled is still there


def test_the_turn_in_flight_does_not_read_its_own_input_row_as_history(
    db: sqlite3.Connection,
) -> None:
    """The reader's own anchoring row is the prompt's user body — it must not
    arrive a second time as history. That turn is unsettled too, and is now
    skipped by its own turn id rather than by the missing settle0 that used to
    stand in for it."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    earlier = _settled_exchange("Book me a table for two.", "Booked for 8pm.")

    reader, history_ids = _model_history()

    assert reader.uid not in history_ids
    assert earlier.uid in history_ids
