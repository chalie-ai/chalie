# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Regression guard for ``TurnSerializerService.serialize``'s ``working`` field on
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
side-effect-free, mirrors ``DELETE /api/threads/<turn_id>``'s own cancel path
in ``api/threads.py``) against the real, fully-migrated SQLite database — the
same collaborators production uses to open/finish a turn's execution row and
append its transcript rows. No mocks, no hand-rolled DDL.
"""

import json
import sqlite3
from typing import cast

import pytest
from flask.testing import FlaskClient

from services.turn_serializer_service import get_service as _serializer
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

    result = _serializer().serialize(_CHANNEL, turn_id)

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

    result = _serializer().serialize(_CHANNEL, turn_id)

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

    result = _serializer().serialize(_CHANNEL, turn_id)

    assert result["working"] is False
    messages = cast("list[dict[str, object]]", result["messages"])
    contents = [m["content"] for m in messages]
    assert "Standup reminder: 9am daily sync." in contents


# ── ``crashed`` — the terminal-CRASHED signal (§ same function) ───────────────
#
# A turn whose most recent execution ended CRASHED (an unhandled step exception
# or a swept process death) settles to ``working=False`` with, in the worst
# case, only a tool-trace and no assistant reply — data-indistinguishable from a
# normal completed-but-empty turn. ``serialize_turn`` exposes ``crashed`` off the
# terminal execution's ``state`` so the FE can name the failure ("ended
# unexpectedly") instead of rendering a bare, blank footer. Keyed on
# ``TurnExecution.latest`` (most recent row, open or closed), so a later
# COMPLETED retry of the same turn_id correctly clears the flag.


def test_crashed_execution_marks_the_block_crashed(db: sqlite3.Connection) -> None:
    """A turn that ran a tool then died mid-step — an OPEN execution stamped
    terminal CRASHED, a tool call, and no assistant reply row. This is the
    exact reply-less crash that renders as a bare "1 tool used" footer; the
    serializer must report ``crashed=True`` so the FE can name the failure."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    turn_id = 9201
    mp = MessageProcessor(UserConfig(), turn_id, "Do the thing that crashes.")  # inert (I2)
    uid = mp.transcript_service.append_input(mp.raw_input)
    mp.uid = uid
    mp.current_transcript_id = uid

    assert mp.turn_execution_service.open() is not None
    call_id = mp.tool_call_service.start("web_search", {"query": "boom"})
    assert call_id is not None  # sanity: real tool-call write
    mp.tool_call_service.finish(call_id, "partial", ToolCall.DONE)
    crashed = mp.turn_execution_service.finish(TurnExecution.CRASHED, "step raised")
    assert crashed is not None and crashed.ended_at is not None  # sanity: real terminal close

    result = _serializer().serialize(_USER_CHANNEL, turn_id)

    assert result["crashed"] is True
    assert result["working"] is False


def test_completed_execution_is_not_crashed(db: sqlite3.Connection) -> None:
    """The other direction: a turn that ran to COMPLETED must report
    ``crashed=False`` so a normal turn never shows the failure note."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    turn_id = 9202
    mp = MessageProcessor(UserConfig(), turn_id, "Do the thing that works.")  # inert (I2)
    uid = mp.transcript_service.append_input(mp.raw_input)
    mp.uid = uid
    mp.current_transcript_id = uid

    assert mp.turn_execution_service.open() is not None
    mp.transcript_service.append_assistant("Done.")
    assert mp.turn_execution_service.finish(TurnExecution.COMPLETED) is not None

    result = _serializer().serialize(_USER_CHANNEL, turn_id)

    assert result["crashed"] is False


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

    result = _serializer().serialize(_USER_CHANNEL, turn_id)
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

    result = _serializer().serialize(_USER_CHANNEL, turn_id)
    messages = cast("list[dict[str, object]]", result["messages"])
    by_id = {cast("str", m["id"]): m for m in messages}

    assert "thread_message" not in by_id[str(opener_uid)]
    assert "thread_message" not in by_id[str(opener_answer_id)]
    assert by_id[str(reply_uid)]["thread_message"] is True
    assert by_id[str(reply_answer_id)]["thread_message"] is True

    reply_input_msg = by_id[str(reply_uid)]
    tool_calls = cast("list[dict[str, object]]", reply_input_msg.get("tool_calls", []))
    assert any(c["tool_name"] == "calendar" for c in tool_calls)


# ── Rich-media spans resolve PER ACT CYCLE, not per turn (§ same function) ─────
#
# A turn_id spans the WHOLE thread — many user requests — and each request runs
# its own ACT loop with a fresh ``DispatchService`` whose rich-media ordinal
# counter restarts at 1. So two ``image_preview`` calls in one turn BOTH carry
# ``image_preview_1``. ``serialize_turn`` used to resolve every assistant row's
# span against the FLAT turn scope, so ``rich_media_parser._find_payload``
# returned the turn's FIRST such call for EVERY card — every image card rendered
# the first image (the reported "keeps rendering the same thing" bug).
# The fix scopes each assistant row's span to its own cycle's tool calls.


def _image_preview_rich_result(subtitle: str, image_url: str, ordinal: int) -> str:
    """The on-wire result an ``image_preview`` rich card stores: the sealed
    envelope ``[image_preview(status=success)]\\n<payload>\\n\\n<instruction>\\n
    [end:image_preview]`` whose instruction carries the ordinal-keyed span the
    parser anchors on (see ``DispatchService._render_rich``)."""
    payload = json.dumps(
        {"url": image_url, "subtitle": subtitle, "alt": subtitle},
        separators=(",", ":"),
    )
    instruction = (
        "You MUST present this result by wrapping your synthesis in "
        f"<span id='image_preview_{ordinal}'>your synthesis here</span>. "
        "The span renders as a card; without it the user sees only plain text."
    )
    return f"[image_preview(status=success)]\n{payload}\n\n{instruction}\n[end:image_preview]"


def test_same_tool_rich_cards_in_one_turn_resolve_per_cycle(db: sqlite3.Connection) -> None:
    """Two ``image_preview`` calls across two cycles of one turn, both carrying
    ``image_preview_1`` (fresh MP per request → ordinal restarts at 1). Before
    the fix, both cards resolved to the FIRST call's payload; the fix must pair
    each assistant row's span with ITS OWN cycle's tool call."""
    assert db is not None  # fixture taken for its binding side effect (real DB gateway)
    turn_id = 7003

    # Cycle 1 — "northern lights" → image_preview #1 (ordinal 1).
    c1 = MessageProcessor(UserConfig(), turn_id, "Show me the northern lights")  # inert (I2)
    u1 = c1.transcript_service.append_input(c1.raw_input)
    c1.uid = u1
    c1.current_transcript_id = u1
    call1 = c1.tool_call_service.start("image_preview", {"file_path": "https://north.example/a.jpg"})
    assert call1 is not None  # sanity: real tool-call write
    c1.tool_call_service.finish(
        call1,
        _image_preview_rich_result("northern lights", "https://north.example/a.jpg", 1),
        ToolCall.DONE,
    )
    a1 = c1.transcript_service.append_assistant(
        "Here you go: <span id='image_preview_1'>Northern lights.</span>"
    )

    # Cycle 2 (same turn) — "southern lights" → image_preview #1 AGAIN: the fresh
    # MP's ordinal counter restarts at 1, so the tag collides with cycle 1's.
    c2 = MessageProcessor(UserConfig(), turn_id, "Now the southern lights")  # inert (I2)
    u2 = c2.transcript_service.append_input(c2.raw_input)
    c2.uid = u2
    c2.current_transcript_id = u2
    call2 = c2.tool_call_service.start("image_preview", {"file_path": "https://south.example/b.jpg"})
    assert call2 is not None  # sanity: real tool-call write
    c2.tool_call_service.finish(
        call2,
        _image_preview_rich_result("southern lights", "https://south.example/b.jpg", 1),
        ToolCall.DONE,
    )
    a2 = c2.transcript_service.append_assistant(
        "And these: <span id='image_preview_1'>Southern lights.</span>"
    )

    result = _serializer().serialize(_USER_CHANNEL, turn_id)
    messages = cast("list[dict[str, object]]", result["messages"])
    by_id = {cast("str", m["id"]): m for m in messages}

    def rich_subtitle(msg: dict[str, object]) -> str:
        segs = cast("list[dict[str, object]]", msg["segments"])
        rich = [s for s in segs if s.get("type") == "rich"]
        assert len(rich) == 1, f"expected exactly one rich segment, got {segs}"
        payload = cast("dict[str, object]", rich[0]["payload"])
        return cast("str", payload["subtitle"])

    # Each card resolves to ITS OWN cycle's image — not the turn's first call.
    assert rich_subtitle(by_id[str(a1)]) == "northern lights"
    assert rich_subtitle(by_id[str(a2)]) == "southern lights"


# ── ``type`` stamping — the ConfigType identity carried on every read ──
#
# ``ThreadSummary`` and ``TurnBlock`` both carry a required ``type: str`` — the
# ConfigType identity string (``user``/``scheduled``/``discovery``) the caller
# resolved ``channel`` from. A client that discovers a thread via the feed or a
# WS ``updated`` signal has no other way to recover which channel a turn_id
# lives on, so it must carry ``type`` forward on any refetch. Before the fix,
# neither DTO exposed it, so a client that dropped ``type`` on refetch silently
# fell back to the ``user`` default, resolved to the WRONG channel, and a real
# scheduled-thread turn rendered as an empty block — the reported
# "disappearing answer" defect. These tests pin the contract at both the
# service level (``serialize_turn``/``_thread_summaries``) and the REST
# boundary (``authed_client``, real Flask routes, real DB) so a regression in
# either the stamping or the channel resolution it depends on is caught.

_SCHEDULED_TYPE = "scheduled"
_USER_TYPE = "user"


def test_serialize_turn_stamps_the_config_type_it_was_called_with(db: sqlite3.Connection) -> None:
    """A scheduled-channel turn fetched with ``config_type="scheduled"`` carries
    ``type == "scheduled"`` on its block; a user-channel turn fetched with the
    default arg carries ``type == "user"``. Each is the one signal a
    refetching client needs to keep addressing the right channel instead of
    silently resolving to the ``user`` default."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    scheduled_turn_id = 9101
    scheduled_mp = MessageProcessor(ScheduledConfig(), scheduled_turn_id)  # inert (I2)
    assert scheduled_mp.turn_execution_service.open() is not None
    scheduled_mp.transcript_service.append_assistant("Standup reminder: 9am daily sync.")
    assert scheduled_mp.turn_execution_service.finish(TurnExecution.COMPLETED) is not None

    scheduled_result = _serializer().serialize(_CHANNEL, scheduled_turn_id, config_type=_SCHEDULED_TYPE)
    assert scheduled_result["type"] == _SCHEDULED_TYPE

    user_turn_id = 7101
    user_mp = MessageProcessor(UserConfig(), user_turn_id, "What's the weather?")  # inert (I2)
    uid = user_mp.transcript_service.append_input(user_mp.raw_input)
    user_mp.uid = uid
    user_mp.current_transcript_id = uid
    user_mp.transcript_service.append_assistant("Sunny, 22C.")

    user_result = _serializer().serialize(_USER_CHANNEL, user_turn_id)  # default config_type

    assert user_result["type"] == _USER_TYPE


def test_thread_feed_stamps_every_summary_with_the_requested_type(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """``GET /api/threads/all?type=scheduled`` must resolve to the ``schedule``
    channel and stamp EVERY summary it returns ``type == "scheduled"`` — the
    identity a client needs to correctly refetch any thread it discovers via
    this feed. A sibling thread on the ``user`` channel must not leak into
    the scheduled feed at all, proving the stamped type actually matches the
    channel the summaries were resolved from, not just a hardcoded label."""
    client, _db, _store = authed_client
    scheduled_turn_id = 9104
    scheduled_mp = MessageProcessor(ScheduledConfig(), scheduled_turn_id)  # inert (I2)
    assert scheduled_mp.turn_execution_service.open() is not None
    scheduled_mp.transcript_service.append_assistant("Weekly digest is ready.")
    assert scheduled_mp.turn_execution_service.finish(TurnExecution.COMPLETED) is not None

    user_turn_id = 7104
    user_mp = MessageProcessor(UserConfig(), user_turn_id, "What's on my plate today?")  # inert (I2)
    uid = user_mp.transcript_service.append_input(user_mp.raw_input)
    user_mp.uid = uid
    user_mp.current_transcript_id = uid
    user_mp.transcript_service.append_assistant("Nothing scheduled.")

    resp = client.get(f"/api/threads/all?type={_SCHEDULED_TYPE}")
    assert resp.status_code == 200
    threads = resp.get_json()["result"]

    assert threads  # sanity: the seeded scheduled thread is actually returned
    assert all(t["type"] == _SCHEDULED_TYPE for t in threads)
    assert scheduled_turn_id in [t["turn_id"] for t in threads]
    assert user_turn_id not in [t["turn_id"] for t in threads]


def test_scheduled_turn_fetched_with_wrong_type_renders_empty_but_right_type_renders_full_thread(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The regression pinned at the REST boundary: a real
    scheduled-channel turn refetched with the WRONG (default) ``type=user``
    resolves to the ``user`` channel, where this turn_id names no rows —
    the empty-turn shape a client would render after a WS refetch that
    dropped ``type`` (the reported "disappearing answer"). The SAME turn_id
    fetched with the correct ``type=scheduled`` must return its full, real
    message list, stamped ``type == "scheduled"``."""
    client, _db, _store = authed_client
    turn_id = 9103
    mp = MessageProcessor(ScheduledConfig(), turn_id)  # inert (I2)
    assert mp.turn_execution_service.open() is not None
    mp.transcript_service.append_assistant("Standup reminder: 9am daily sync.")
    assert mp.turn_execution_service.finish(TurnExecution.COMPLETED) is not None

    wrong_type_resp = client.get(f"/api/threads/{turn_id}")  # default type=user
    assert wrong_type_resp.status_code == 200
    wrong_body = wrong_type_resp.get_json()["result"]
    assert wrong_body["messages"] == []

    right_type_resp = client.get(f"/api/threads/{turn_id}?type={_SCHEDULED_TYPE}")
    assert right_type_resp.status_code == 200
    right_body = right_type_resp.get_json()["result"]
    assert right_body["type"] == _SCHEDULED_TYPE
    contents = [m["content"] for m in right_body["messages"]]
    assert "Standup reminder: 9am daily sync." in contents
