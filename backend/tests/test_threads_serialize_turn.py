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
from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.tool_call import ToolCall
from models.transcript import Transcript
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


# ── ``thread_message`` boundary (§ same function, different field) ────────────
#
# The companion regression: ``thread_message`` used to derive from
# ``Transcript.settle0`` (mutable — a reply's own tool call can retroactively
# demote it, erasing every row's tag on re-fetch), then from "first assistant
# row" (wrong the moment an interim tool-using step precedes a turn's own
# final answer). The fix keys the boundary on the id of the turn's SECOND
# ``role='user'`` row — structural and immutable once written — tagging every
# row from that id onward. No second user row → nothing is tagged.

_USER_CHANNEL = "user"


def test_single_exchange_with_interim_tool_step_has_no_thread_replies(db: sqlite3.Connection) -> None:
    """A single exchange with an interim tool step, never replied: one user
    row, an INTERIM assistant row ("Let me check the docs…") whose own tool
    call fires and settles, then the FINAL answer. There is no second
    ``role='user'`` row anywhere in this turn, so the fix's boundary tags
    NOTHING here. The rejected "first assistant row and everything after it
    is a reply" heuristic would have tagged BOTH the interim row and the
    final answer, since the interim row is the first assistant row in turn
    order — wrongly turning this single, never-replied exchange into a
    thread the moment it has a tool step in the middle, which is the common
    case, not an edge case."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    turn_id = 7001
    mp = MessageProcessor(UserConfig(), turn_id, "What tools do I have connected?")  # inert (I2)

    uid = mp.transcript_service.append_input(mp.raw_input)
    mp.uid = uid
    mp.current_transcript_id = uid

    interim_id = mp.transcript_service.append_assistant("Let me check the docs for that.")
    mp.current_transcript_id = interim_id  # mirrors MessageProcessor._store's real wiring
    call_id = mp.tool_call_service.start("web_search", {"query": "connected tools"})
    assert call_id is not None  # sanity: the real tool-call write succeeded
    mp.tool_call_service.finish(call_id, "no direct hit", ToolCall.DONE)

    final_id = mp.transcript_service.append_assistant("Here's what I found connected.")

    result = serialize_turn(_USER_CHANNEL, turn_id)
    messages = cast("list[dict[str, object]]", result["messages"])

    assert [m["id"] for m in messages] == [str(uid), str(interim_id), str(final_id)]
    assert all("thread_message" not in m for m in messages)


def test_reply_with_settling_tool_call_tags_only_the_reply_rows(db: sqlite3.Connection) -> None:
    """Opener plus a reply whose tool call settles: an opener (user row +
    settled answer) that later gets a REPLY whose own tool call is a
    SETTLING ability — when a reply's tool call is a settling ability, it
    unsettles the opener's row: ``ToolCallService.start`` calls
    ``TranscriptService.unsettle()`` on the OPENER's settle0 row the instant
    the reply's tool fires. This is the exact cross-table mutation that broke
    the old ``settle0``-derived
    boundary: settle0 moves off the opener and onto the reply's own answer,
    so re-deriving the boundary from settle0 AFTER the tool call tags
    nothing at all (opener and reply both read as "not thread"). The fix's
    boundary — the second user row's id, written once and never mutated by
    anything downstream — is unaffected: the opener stays untagged and BOTH
    reply rows are tagged, tool chip included."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    turn_id = 7002
    opener = MessageProcessor(UserConfig(), turn_id, "Can you check my calendar for today?")  # inert (I2)
    opener_uid = opener.transcript_service.append_input(opener.raw_input)
    opener.uid = opener_uid
    opener.current_transcript_id = opener_uid
    opener_answer_id = opener.transcript_service.append_assistant("You have no events today.")
    assert Transcript.settle0(_USER_CHANNEL, turn_id) == opener_answer_id  # sanity: opener settled

    reply = MessageProcessor(UserConfig(), turn_id, "Actually, add a 3pm meeting.")  # forked reply, inert (I2)
    reply_uid = reply.transcript_service.append_input(reply.raw_input)
    reply.uid = reply_uid
    reply.current_transcript_id = reply_uid

    call_id = reply.tool_call_service.start(
        "calendar", {"action": "create_event", "summary": "Meeting", "dtstart": "15:00"},
    )
    assert call_id is not None  # sanity: the real tool-call write succeeded
    # sanity: the settling tool call demoted the OPENER's settle0 row (a
    # reply's own tool activity un-settling the original exchange's row) —
    # the exact cross-table mutation the old settle0-derived boundary broke on.
    opener_row = Transcript.filter("id", opener_answer_id).first()
    assert opener_row is not None
    assert opener_row.settled == 0
    assert Transcript.settle0(_USER_CHANNEL, turn_id) is None  # nothing settled mid-tool-call

    reply.tool_call_service.finish(call_id, "created", ToolCall.DONE)
    reply_answer_id = reply.transcript_service.append_assistant("Added a 3pm meeting to your calendar.")

    result = serialize_turn(_USER_CHANNEL, turn_id)
    messages = cast("list[dict[str, object]]", result["messages"])
    by_id = {cast("str", m["id"]): m for m in messages}

    assert "thread_message" not in by_id[str(opener_uid)]
    assert "thread_message" not in by_id[str(opener_answer_id)]
    assert by_id[str(reply_uid)]["thread_message"] is True
    assert by_id[str(reply_answer_id)]["thread_message"] is True

    reply_input_msg = by_id[str(reply_uid)]
    tool_calls = cast("list[dict[str, object]]", reply_input_msg.get("tool_calls", []))
    assert any(c["tool_name"] == "calendar" for c in tool_calls)
