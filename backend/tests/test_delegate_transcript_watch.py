"""Feature test: a delegate's child turn can be followed from the parent's tool pill.

A delegate ability (``web_search`` here) runs a child turn on its own
``delegate:<name>`` channel. Called from the person's chat, the call returns at
once and the child's answer comes back later as a message of its own on the
chat, linked to the same call. The surface needs three things to let a person
watch that child:

* a durable link from the parent's ``tool_calls`` row to the child's input row;
* live frames addressed by channel (the child's numeric turn id collides with the
  user spine's, so a frame carrying ``type: "user"`` would land in the wrong
  thread);
* a REST read of the child turn, plus a ``delegate`` reference on the parent's
  chips, that both degrade quietly once retention has purged the child.

Drives the real production entry point (construct inertly, ``begin()``,
``result()`` — what ``MessageProcessor.process()`` does) against the real,
fully-migrated SQLite DB: the real ``DelegateAbility``, ``DispatchService``,
``ToolCallService``, ``TurnSerializerService``, Flask routes and
``GarbageCollectorService`` all run. Frames are observed through the REAL socket
registry via an in-process client implementing the broker's ``send(str)``
Protocol. The only substitution is the LLM network boundary
(``services.provider_service.build_client``, the seam
``test_message_processor_runaway_loop.py`` uses). The child is held inside its
first model call until the caller's turn has ended, so its answer always comes
back as a turn of its own. The two failure tests inject their fault into the
real database (a trigger, or a row write) at the moment the caller's own
thread reaches it — nothing in-process is replaced.
"""

import logging
import sqlite3
import threading
from dataclasses import dataclass
from typing import cast

import pytest
from flask.testing import FlaskClient

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.turn_execution import TurnExecution
from services.database import Database
from services.garbage_collector_service import GarbageCollectorService
from tests._delegates import (
    USER,
    ChannelProvider,
    Held,
    assistant_texts,
    block,
    calls,
    calls_of,
    chip,
    child_turn,
    execution,
    messages,
    replies,
    run_turn,
    text,
    tool,
    watching_warnings,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

_CHILD_CHANNEL = "delegate:web_search"
_QUESTION = "what is the capital of France?"
_INSTRUCTIONS = "find the capital of France"
_CHILD_ANSWER = "Paris is the capital of France."
_STARTED_ANSWER = "I have started a search."
_REPLY_ANSWER = "It is Paris."


@dataclass(frozen=True)
class _DelegateRun:
    """One finished user turn that called ``web_search`` (a real delegate) and
    ``find_tools`` (an ordinary tool), with the facts every test reads off it."""

    parent: MessageProcessor
    frames: list[dict[str, object]]
    web_search_call_id: int
    child_turn_id: int

    @property
    def ref(self) -> dict[str, object]:
        """The ``delegate`` reference the parent's pill must carry."""
        return {"channel": _CHILD_CHANNEL, "turn_id": self.child_turn_id}


@pytest.fixture
def delegate_run(db: sqlite3.Connection) -> _DelegateRun:
    """A user turn that runs an ordinary tool and a ``web_search`` delegate whose
    child turn makes a tool call of its own and answers once the caller's turn
    has ended; the reply that brings the answer back is answered in turn."""
    held = Held(calls(tool("recall", query="capital of France")))
    provider = ChannelProvider(
        chat=[
            calls(tool("find_tools", query=["weather"]), tool("web_search", instructions=_INSTRUCTIONS)),
            text(_STARTED_ANSWER),
            text(_REPLY_ANSWER),
        ],
        web_search=[held, text(_CHILD_ANSWER)],
    )
    parent, frames = run_turn(provider, UserConfig(), _QUESTION)
    assert held.opened and provider.unused() == {}
    call_id = cast("int", calls_of(db, "web_search")[0]["id"])
    return _DelegateRun(parent, frames, call_id, child_turn(db, _CHILD_CHANNEL, call_id))


def test_delegate_input_row_is_stamped_with_the_parent_tool_call(
    db: sqlite3.Connection, delegate_run: _DelegateRun,
) -> None:
    """The child's input row and the reply that carries its answer back to the
    chat — and only they — carry the parent's ``tool_calls`` id, so the pill can
    be followed to the child's transcript and the answer traced to its call."""
    stamped = db.execute(
        "SELECT channel, role, content, tool_call_id FROM transcript WHERE tool_call_id IS NOT NULL ORDER BY id"
    ).fetchall()

    call_id = str(delegate_run.web_search_call_id)
    assert [(r["channel"], r["role"], r["tool_call_id"]) for r in stamped] == [
        (_CHILD_CHANNEL, "web_search", call_id),
        (USER, "user", call_id),
    ]
    assert stamped[0]["content"] == _INSTRUCTIONS
    assert cast("str", stamped[1]["content"]).startswith(_CHILD_ANSWER)


def test_every_delegate_frame_is_addressed_by_channel_never_by_type(delegate_run: _DelegateRun) -> None:
    """The child's frames — its execution frames and its own tool frames alike —
    reach the wire carrying the full channel and the child's turn id and NO
    ``type``: a ``type`` would route them into the user thread whose numeric turn
    id they share."""
    delegate_frames = [f for f in delegate_run.frames if f.get("channel") == _CHILD_CHANNEL]

    assert delegate_frames, "the child turn's frames must reach the wire"
    assert all("type" not in f for f in delegate_frames)
    assert all(f["turn_id"] == delegate_run.child_turn_id for f in delegate_frames)
    assert any(f.get("tool_name") == "recall" for f in delegate_frames), (
        "the child's own tool frames must be addressed too"
    )


def test_parent_pill_is_re_announced_once_its_child_exists(delegate_run: _DelegateRun) -> None:
    """The ``started`` frame leaves before any child turn exists, so it carries no
    reference; the frame sent again once the child's input row exists, and the
    terminal ``done`` frame, both point at the child. Every parent frame stays
    typed ``user`` with no channel, and an ordinary tool's frames never gain a
    reference."""
    web_search = [
        f for f in delegate_run.frames if f.get("tool_name") == "web_search" and "channel" not in f
    ]

    assert len(web_search) >= 3
    assert web_search[0]["state"] == "started" and web_search[0]["delegate"] is None
    assert any(f["state"] == "started" and f["delegate"] == delegate_run.ref for f in web_search[1:])
    assert web_search[-1]["state"] == "done" and web_search[-1]["delegate"] == delegate_run.ref
    assert all(f["type"] == "user" for f in web_search)
    find_tools = [f for f in delegate_run.frames if f.get("tool_name") == "find_tools"]
    assert find_tools and all(f["delegate"] is None for f in find_tools)


def test_user_turn_without_a_delegate_stays_user_addressed(db: sqlite3.Connection) -> None:
    """Nothing about a turn that calls no delegate changes shape: every frame is
    typed ``user`` and none is addressed to a delegate channel, tool frames gain
    only an explicit ``delegate: null``, and no delegate transcript exists."""
    provider = ChannelProvider(chat=[calls(tool("find_tools", query=["weather"])), text("All done.")])

    _mp, frames = run_turn(provider, UserConfig(), "check the weather tools")

    tool_frames = [f for f in frames if "tool_name" in f]
    assert tool_frames
    assert all(f["type"] == "user" for f in frames)
    assert not any(str(f.get("channel")).startswith("delegate:") for f in frames)
    assert all("delegate" in f and f["delegate"] is None for f in tool_frames)
    assert db.execute("SELECT COUNT(*) FROM transcript WHERE channel LIKE 'delegate:%'").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM transcript WHERE tool_call_id IS NOT NULL").fetchone()[0] == 0


def test_parent_thread_read_links_the_delegate_chip_to_its_child_turn(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], delegate_run: _DelegateRun,
) -> None:
    """Reading the parent thread back gives the ``web_search`` chip the same
    ``delegate`` reference the live frame carried, plus whether that child still
    runs, leaves an ordinary tool's chip
    unlinked, and keeps the block itself a plain user thread."""
    client, _db, _store = authed_client

    thread = block(client, f"/api/threads/{delegate_run.parent.turn_id}")

    # The read also says the child is no longer running — it has replied.
    assert chip(thread, "web_search")["delegate"] == {**delegate_run.ref, "working": False}
    assert chip(thread, "find_tools")["delegate"] is None
    assert thread["type"] == "user" and thread["channel"] is None


def test_child_turn_reads_by_channel_with_its_instruction_as_a_user_message(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], delegate_run: _DelegateRun,
) -> None:
    """The child turn is read by channel: its block is addressed by that channel
    with no type, its stamped input row (role ``web_search`` in storage) reads as
    the user's message, and the child's calls follow as assistant messages — its
    tool-only step as an empty one, then its answer; a child that ran to its
    answer never reads as stopped.
    The user thread sharing the same numeric turn id is a different block."""
    client, _db, _store = authed_client

    thread = block(client, f"/api/threads/{delegate_run.child_turn_id}?channel={_CHILD_CHANNEL}")

    assert thread["channel"] == _CHILD_CHANNEL and thread["type"] is None
    assert thread["cancelled"] is False
    rows = messages(thread)
    assert (rows[0]["role"], rows[0]["content"]) == ("user", _INSTRUCTIONS)
    assert [m["content"] for m in rows if m["role"] == "assistant"] == ["", _CHILD_ANSWER]
    user_thread = block(client, f"/api/threads/{delegate_run.child_turn_id}")
    assert user_thread["type"] == "user" and user_thread["channel"] is None
    assert _INSTRUCTIONS not in [m["content"] for m in messages(user_thread)]


def test_purged_delegate_turn_unlinks_the_chip_and_reads_as_an_unknown_turn(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], delegate_run: _DelegateRun,
) -> None:
    """Once retention has purged the child (the real garbage collector, run over
    rows aged past any retention window) the parent's chip falls back to no
    ``delegate`` and the child's channel reads exactly like a turn that never
    existed — a quiet empty block, not an error."""
    client, db, _store = authed_client
    parent_path = f"/api/threads/{delegate_run.parent.turn_id}"
    child_path = f"/api/threads/{delegate_run.child_turn_id}?channel={_CHILD_CHANNEL}"
    assert chip(block(client, parent_path), "web_search")["delegate"] == {**delegate_run.ref, "working": False}

    db.execute(
        "UPDATE transcript SET created_at = datetime('now', '-200 days') WHERE channel = ?", (_CHILD_CHANNEL,),
    )
    db.commit()
    GarbageCollectorService().run()

    assert db.execute("SELECT COUNT(*) FROM transcript WHERE channel = ?", (_CHILD_CHANNEL,)).fetchone()[0] == 0
    assert chip(block(client, parent_path), "web_search")["delegate"] is None
    purged = block(client, child_path)
    never_existed = block(client, f"/api/threads/987654?channel={_CHILD_CHANNEL}")
    assert purged["messages"] == []
    assert {k: v for k, v in purged.items() if k != "turn_id"} == {
        k: v for k, v in never_existed.items() if k != "turn_id"
    }


def _write_on_this_thread(*statements: str) -> None:
    """Run *statements* through the calling thread's real database connection."""
    conn = Database.conn()
    for sql in statements:
        conn.execute(sql)
    conn.commit()


def _web_search_call(db: sqlite3.Connection) -> sqlite3.Row:
    [call] = calls_of(db, "web_search")
    return call


def _late_web_search_provider() -> tuple[ChannelProvider, Held]:
    """A chat turn that calls ``web_search`` and answers its started result,
    then answers the reply; the child, held until the chat turn has ended,
    answers at once."""
    held = Held(text(_CHILD_ANSWER))
    provider = ChannelProvider(
        chat=[calls(tool("web_search", instructions=_INSTRUCTIONS)), text(_STARTED_ANSWER), text(_REPLY_ANSWER)],
        web_search=[held],
    )
    return provider, held


def _assert_call_returned_at_once_and_its_child_replied(
    db: sqlite3.Connection, call: sqlite3.Row, parent: MessageProcessor,
) -> None:
    """The call finished without waiting for its child — its result carries no
    answer — while the child ran to its end on its own: its answer is stored, its
    execution is COMPLETED, and the answer came back to the chat as one reply
    linked to the call, opening a turn of its own that completed with the
    model's answer to it."""
    assert call["state"] == "done" and _CHILD_ANSWER not in call["result"]
    child = child_turn(db, _CHILD_CHANNEL, call["id"])
    assert assistant_texts(db, _CHILD_CHANNEL, child) == [_CHILD_ANSWER]
    assert execution(_CHILD_CHANNEL, child).state == TurnExecution.COMPLETED
    [reply] = replies(db, call["id"])
    assert cast("str", reply["content"]).startswith(_CHILD_ANSWER)
    assert reply["turn_id"] != parent.turn_id
    assert execution(USER, reply["turn_id"]).state == TurnExecution.COMPLETED
    assert assistant_texts(db, USER, reply["turn_id"]) == [_REPLY_ANSWER]


def _assert_parent_turn_completed_with_its_answer(db: sqlite3.Connection, parent: MessageProcessor) -> None:
    assert execution(USER, parent.turn_id).state == TurnExecution.COMPLETED
    # The call that ran the delegate stored an empty row; the answer is the next.
    assert assistant_texts(db, USER, parent.turn_id) == ["", _STARTED_ANSWER]


def test_failed_announcement_does_not_fail_the_delegate_call(db: sqlite3.Connection) -> None:
    """The parent's pill is announced again once the child exists — and that
    announcement can fail (here: its ``tool_calls`` row is gone at that moment, a
    purge or a rewrite racing the delegate). It is reported with a warning but
    never fails the work: the call still finishes normally, the child runs to
    its end, its answer comes back to the chat and the turns complete.

    The fault is a database trigger that moves the parent's row out of reach the
    instant the child's input row is written; the warning puts it back (and
    drops the trigger) on the caller's own thread before the warning call
    returns — so the row is missing for exactly the announcement and present
    again for the finish."""
    db.execute(
        "CREATE TRIGGER vanish_caller_row AFTER INSERT ON transcript WHEN NEW.tool_call_id IS NOT NULL "
        "BEGIN UPDATE tool_calls SET id = -id WHERE id = CAST(NEW.tool_call_id AS INTEGER); END"
    )
    db.commit()
    provider, held = _late_web_search_provider()

    def put_the_row_back() -> None:
        _write_on_this_thread("UPDATE tool_calls SET id = -id WHERE id < 0", "DROP TRIGGER IF EXISTS vanish_caller_row")

    with watching_warnings(put_the_row_back) as watch:
        parent, frames = run_turn(provider, UserConfig(), _QUESTION)

    assert watch.records, "the failed announcement must be reported with a warning"
    assert held.opened and provider.unused() == {}
    call = _web_search_call(db)
    _assert_call_returned_at_once_and_its_child_replied(db, call, parent)
    _assert_parent_turn_completed_with_its_answer(db, parent)
    web_search = [f for f in frames if f.get("tool_name") == "web_search" and "channel" not in f]
    assert not any(f["state"] == "started" and f["delegate"] is not None for f in web_search), (
        "the announcement must not have gone out"
    )
    assert web_search[-1]["state"] == "done"
    assert web_search[-1]["delegate"] == {
        "channel": _CHILD_CHANNEL, "turn_id": child_turn(db, _CHILD_CHANNEL, call["id"]),
    }


@dataclass(frozen=True)
class _SpoiledStampRun:
    """One finished user turn whose ``web_search`` call's own timestamp became
    unreadable once its child was running."""

    frames: list[dict[str, object]]
    warnings: list[logging.LogRecord]

    @property
    def pill_frames(self) -> list[dict[str, object]]:
        """Every live frame of the caller's ``web_search`` pill, in order."""
        return [f for f in self.frames if f.get("tool_name") == "web_search" and "channel" not in f]


@pytest.fixture
def spoiled_stamp_run(db: sqlite3.Connection) -> _SpoiledStampRun:
    """A ``web_search`` turn whose call row has its ``created_at`` overwritten with
    garbage after the parent's pill was announced and before the call finishes —
    so only the lookup of "which child did this call spawn" at finish can trip.
    The overwrite runs on the caller's own thread, as the announcement is
    broadcast, so it always lands between the two."""
    spoiled = threading.Event()

    def spoil_on_announcement(frame: dict[str, object]) -> None:
        if (
            not spoiled.is_set()
            and frame.get("tool_name") == "web_search" and "channel" not in frame
            and frame["state"] == "started" and frame["delegate"] is not None
        ):
            _write_on_this_thread("UPDATE tool_calls SET created_at = 'not-a-timestamp' WHERE tool_name = 'web_search'")
            spoiled.set()

    provider, held = _late_web_search_provider()
    with watching_warnings() as watch:
        _, frames = run_turn(provider, UserConfig(), _QUESTION, on_frame=spoil_on_announcement)
    assert spoiled.is_set(), "the call must have been announced before the stamp was spoiled"
    assert held.opened
    return _SpoiledStampRun(frames, watch.records)


def test_live_pill_still_finishes_when_its_child_cannot_be_looked_up(
    db: sqlite3.Connection, spoiled_stamp_run: _SpoiledStampRun,
) -> None:
    """The row says ``done``, so the live pill must say so too — with no link to
    the child, which is all that could not be worked out — and the lookup that
    failed is reported with exactly one warning, not swallowed and not repeated."""
    last = spoiled_stamp_run.pill_frames[-1]

    assert _web_search_call(db)["state"] == "done"
    assert last["state"] == "done" and last["delegate"] is None
    assert len(spoiled_stamp_run.warnings) == 1
