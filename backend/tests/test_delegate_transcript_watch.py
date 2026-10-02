"""Feature test: a delegate's child turn can be followed from the parent's tool pill.

A delegate ability (``web_search`` here) runs a child turn on its own
``delegate:<name>`` channel while the caller's tool call is still open. The
surface needs three things to let a person watch that child:

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
``test_message_processor_runaway_loop.py`` uses). The two failure tests inject
their fault into the real database (a trigger, or a row write from the child's
own provider call) — nothing in-process is replaced.
"""

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

import pytest
from flask.testing import FlaskClient

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.provider_response import ProviderResponse
from models.turn_execution import TurnExecution
from services.database import Database
from services.garbage_collector_service import GarbageCollectorService
from services.tool_call_service import ToolCallService
from services.websocket import Websocket

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

# ProviderService builds its thin transport client via this factory call — the
# real network boundary (see test_message_processor_runaway_loop.py).
_BUILD_CLIENT = "services.provider_service.build_client"

_CHILD_CHANNEL = "delegate:web_search"
_INSTRUCTIONS = "find the capital of France"
_CHILD_ANSWER = "Paris is the capital of France."


def _tool(name: str, **params: object) -> dict[str, object]:
    """A provider-shaped tool call: ``{"name": ..., "input": {...}}``."""
    return {"name": name, "input": params}


class _ScriptedProvider:
    """Replays one scripted ``ProviderResponse`` per ``send()`` call, in order.

    One provider serves the parent turn AND its child turn (they run one after the
    other, the parent blocked on the child), so the script interleaves both.
    Past the end of the script it returns a benign terminal response: a completed
    ``user`` turn spawns fire-and-forget post-turn turns that make their own
    provider calls on daemon threads this test does not script for — hence the
    lock. A scripted response consumed by the wrong turn would break the asserted
    turn's outcome, so the tolerance cannot mask a failure."""

    _TERMINAL = ProviderResponse(text="", model="scripted-overflow", tool_calls=None)

    def __init__(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
        self._sends = 0
        self._lock = threading.Lock()

    def get_context_limit(self) -> int:
        return 200000

    def send(self, _dto: object) -> ProviderResponse:
        with self._lock:
            if self._sends >= len(self._responses):
                return self._TERMINAL
            response = self._responses[self._sends]
            self._sends += 1
            return response


class _HeldChildProvider(_ScriptedProvider):
    """A scripted provider whose SECOND ``send()`` — the child turn's first model
    call, the parent being blocked on its delegate — waits for *gate* before
    answering, then runs *then* on the child's own thread.

    The child is otherwise free to finish before the parent has even tried to
    announce it, which would make a fault injected "between the announcement and
    the finish" land in a different place on every run. Holding the child on a
    ``threading.Event`` pins the order. ``entered`` is set the moment the child's
    call arrives, so a test can act on the running child; ``gate_opened`` records
    whether the wait ended because the gate opened (not because it timed out) so
    a test can fail on a gate that never opened instead of on its downstream
    symptom."""

    _GATE_TIMEOUT_S = 10.0

    def __init__(self, gate: threading.Event, then: Callable[[], None], *responses: ProviderResponse) -> None:
        super().__init__(*responses)
        self._gate = gate
        self._then = then
        self._calls = 0
        self.entered = threading.Event()
        self.gate_opened = False

    def send(self, dto: object) -> ProviderResponse:
        with self._lock:
            self._calls += 1
            is_child_first_call = self._calls == 2
        if is_child_first_call:
            self.entered.set()
            self.gate_opened = self._gate.wait(timeout=self._GATE_TIMEOUT_S)
            self._then()
        return super().send(dto)


class _WarningWatch(logging.Handler):
    """A real logging handler that notes every WARNING-or-worse record it is given
    and, when given a *gate*, opens it on the first. The contract under test is
    that a failed bookkeeping step is reported loudly yet non-fatally, so its
    warning is both the signal that the failure happened and — for the held child
    — the cue to let it run on."""

    def __init__(self, gate: threading.Event | None = None) -> None:
        super().__init__(level=logging.WARNING)
        self._gate = gate
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        if self._gate is not None:
            self._gate.set()


@contextmanager
def _watching_warnings(
    gate: threading.Event | None = None, source_name: str = ToolCallService.__module__,
) -> Iterator[_WarningWatch]:
    """Watch the warnings of one service — by default the one that owns the
    call's frames — and only them. Unrelated services warn on their own (the
    on-device model loaders do whenever their files are absent, as on a fresh
    checkout), and such a warning must neither release the held child early nor
    count as a report of the failure under test."""
    watch = _WarningWatch(gate)
    source = logging.getLogger(source_name)
    source.addHandler(watch)
    try:
        yield watch
    finally:
        source.removeHandler(watch)


class _RecordingClient:
    """A real socket-registry subscriber: implements the registry's ``send(str)``
    Protocol and captures every broadcast frame."""

    def __init__(self, on_frame: Callable[[dict[str, object]], None] | None = None) -> None:
        self.frames: list[dict[str, object]] = []
        self._on_frame = on_frame

    def send(self, data: str) -> None:
        frame = json.loads(data)
        self.frames.append(frame)
        if self._on_frame is not None:
            self._on_frame(frame)


def _drain_background_turns(timeout_s: float = 15.0) -> None:
    """Join the fire-and-forget post-turn daemon turns a completed ``user`` turn
    spawns (skill suggestion, thread gist) so they finish inside THIS test's
    provider+DB patch and never leak into the next — the same cross-test
    corruption guard as ``test_message_processor_runaway_loop.py``'s helper."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pending = [
            t for t in threading.enumerate()
            if t.name in ("skill-suggest", "thread-gist") or t.name.startswith("turn-")
        ]
        if not pending:
            return
        for t in pending:
            t.join(timeout=max(0.0, deadline - time.monotonic()))


@contextmanager
def _listening(on_frame: Callable[[dict[str, object]], None] | None = None) -> Iterator[_RecordingClient]:
    """A client connected to the real socket registry for the duration of the
    block, recording every frame broadcast meanwhile."""
    listener = _RecordingClient(on_frame)
    Websocket._connect(listener)
    try:
        yield listener
    finally:
        Websocket._disconnect(listener)


def _run(
    provider: _ScriptedProvider, raw_input: str, on_frame: Callable[[dict[str, object]], None] | None = None,
) -> tuple[MessageProcessor, list[dict[str, object]]]:
    """Drive a real user turn to termination against *provider*, returning the
    processor and every WS frame the registry saw while it ran (each also handed
    to *on_frame* as it arrives, on the thread that broadcast it)."""
    with _listening(on_frame) as listener:
        mp = MessageProcessor(UserConfig(), raw_input=raw_input)  # inert (I2)
        with patch(_BUILD_CLIENT, return_value=provider):
            mp.begin()
            mp.result()
            _drain_background_turns()
    return mp, listener.frames


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
    child turn makes a tool call of its own and answers."""
    provider = _ScriptedProvider(
        ProviderResponse(text="", model="m", tool_calls=[
            _tool("find_tools", query=["weather"]),
            _tool("web_search", instructions=_INSTRUCTIONS),
        ]),
        ProviderResponse(text="", model="m", tool_calls=[_tool("recall", query="capital of France")]),
        ProviderResponse(text=_CHILD_ANSWER, model="m", tool_calls=None),
        ProviderResponse(text="It is Paris.", model="m", tool_calls=None),
    )
    parent, frames = _run(provider, "what is the capital of France?")
    call = db.execute("SELECT id FROM tool_calls WHERE tool_name = 'web_search'").fetchone()
    child = db.execute(
        "SELECT turn_id FROM transcript WHERE channel = ? AND role = 'web_search'", (_CHILD_CHANNEL,),
    ).fetchone()
    return _DelegateRun(parent, frames, call["id"], child["turn_id"])


def _block(client: FlaskClient, path: str) -> dict[str, object]:
    """GET *path* and return the thread block under ``result``."""
    response = client.get(path)
    assert response.status_code == 200, response.data
    return cast("dict[str, object]", cast("dict[str, object]", response.get_json())["result"])


def _chips(block: dict[str, object]) -> list[dict[str, object]]:
    messages = cast("list[dict[str, object]]", block["messages"])
    return [
        chip for message in messages
        for chip in cast("list[dict[str, object]]", message.get("tool_calls") or [])
    ]


def _chip(block: dict[str, object], tool_name: str) -> dict[str, object]:
    return next(c for c in _chips(block) if c["tool_name"] == tool_name)


def test_delegate_input_row_is_stamped_with_the_parent_tool_call(
    db: sqlite3.Connection, delegate_run: _DelegateRun,
) -> None:
    """The child's input row — and only it — carries the parent's ``tool_calls``
    id, so the pill can be followed to the child's transcript."""
    stamped = db.execute(
        "SELECT channel, role, content, tool_call_id FROM transcript WHERE tool_call_id IS NOT NULL"
    ).fetchall()

    assert [tuple(r) for r in stamped] == [
        (_CHILD_CHANNEL, "web_search", _INSTRUCTIONS, str(delegate_run.web_search_call_id)),
    ]


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
    provider = _ScriptedProvider(
        ProviderResponse(text="", model="m", tool_calls=[_tool("find_tools", query=["weather"])]),
        ProviderResponse(text="All done.", model="m", tool_calls=None),
    )

    _mp, frames = _run(provider, "check the weather tools")

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
    ``delegate`` reference the live frame carried, leaves an ordinary tool's chip
    unlinked, and keeps the block itself a plain user thread."""
    client, _db, _store = authed_client

    block = _block(client, f"/api/threads/{delegate_run.parent.turn_id}")

    assert _chip(block, "web_search")["delegate"] == delegate_run.ref
    assert _chip(block, "find_tools")["delegate"] is None
    assert block["type"] == "user" and block["channel"] is None


def test_child_turn_reads_by_channel_with_its_instruction_as_a_user_message(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], delegate_run: _DelegateRun,
) -> None:
    """The child turn is read by channel: its block is addressed by that channel
    with no type, its stamped input row (role ``web_search`` in storage) reads as
    the user's message, and the child's answer follows as an assistant message.
    The user thread sharing the same numeric turn id is a different block."""
    client, _db, _store = authed_client

    block = _block(client, f"/api/threads/{delegate_run.child_turn_id}?channel={_CHILD_CHANNEL}")

    assert block["channel"] == _CHILD_CHANNEL and block["type"] is None
    messages = cast("list[dict[str, object]]", block["messages"])
    assert (messages[0]["role"], messages[0]["content"]) == ("user", _INSTRUCTIONS)
    assert [m["content"] for m in messages if m["role"] == "assistant"] == [_CHILD_ANSWER]
    user_block = _block(client, f"/api/threads/{delegate_run.child_turn_id}")
    assert user_block["type"] == "user" and user_block["channel"] is None
    assert _INSTRUCTIONS not in [m["content"] for m in cast("list[dict[str, object]]", user_block["messages"])]


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
    assert _chip(_block(client, parent_path), "web_search")["delegate"] == delegate_run.ref

    db.execute(
        "UPDATE transcript SET created_at = datetime('now', '-200 days') WHERE channel = ?", (_CHILD_CHANNEL,),
    )
    db.commit()
    GarbageCollectorService().run()

    assert db.execute("SELECT COUNT(*) FROM transcript WHERE channel = ?", (_CHILD_CHANNEL,)).fetchone()[0] == 0
    assert _chip(_block(client, parent_path), "web_search")["delegate"] is None
    purged = _block(client, child_path)
    never_existed = _block(client, f"/api/threads/987654?channel={_CHILD_CHANNEL}")
    assert purged["messages"] == []
    assert {k: v for k, v in purged.items() if k != "turn_id"} == {
        k: v for k, v in never_existed.items() if k != "turn_id"
    }


def _write_on_this_thread(sql: str) -> None:
    """Run *sql* through the calling thread's real database connection."""
    conn = Database.conn()
    conn.execute(sql)
    conn.commit()


def _web_search_call(db: sqlite3.Connection) -> sqlite3.Row:
    return cast("sqlite3.Row", db.execute(
        "SELECT state, result, ended_at, created_at FROM tool_calls WHERE tool_name = 'web_search'"
    ).fetchone())


def _assert_child_answered_and_closed_before_the_call_ended(db: sqlite3.Connection, call: sqlite3.Row) -> None:
    """The child turn ran to the end — its answer is stored and its execution is
    COMPLETED — and it finished BEFORE the parent's call did: a delegate that
    outlived its caller would be a turn running detached."""
    child_turn = db.execute(
        "SELECT turn_id FROM transcript WHERE channel = ? AND role = 'web_search'", (_CHILD_CHANNEL,),
    ).fetchone()["turn_id"]
    answers = db.execute(
        "SELECT content FROM transcript WHERE channel = ? AND turn_id = ? AND role = 'assistant'",
        (_CHILD_CHANNEL, child_turn),
    ).fetchall()
    assert [r["content"] for r in answers] == [_CHILD_ANSWER]
    execution = TurnExecution.latest(_CHILD_CHANNEL, child_turn)
    assert execution is not None and execution.state == TurnExecution.COMPLETED
    assert execution.ended_at is not None and execution.ended_at <= call["ended_at"]


def _assert_parent_turn_completed_with_its_answer(
    db: sqlite3.Connection, parent: MessageProcessor, reply: str = "It is Paris.",
) -> None:
    execution = TurnExecution.latest(UserConfig().channel, parent.turn_id)
    assert execution is not None and execution.state == TurnExecution.COMPLETED
    replies = db.execute(
        "SELECT content FROM transcript WHERE channel = ? AND turn_id = ? AND role = 'assistant'",
        (UserConfig().channel, parent.turn_id),
    ).fetchall()
    assert [r["content"] for r in replies] == [reply]


def test_failed_announcement_does_not_fail_the_delegate_call(db: sqlite3.Connection) -> None:
    """The parent's pill is announced again once the child exists — and that
    announcement can fail (here: its ``tool_calls`` row is gone at that moment, a
    purge or a rewrite racing the delegate). It is reported with a warning but
    never fails the work: the call still finishes normally with the child's
    answer, the child runs to its end first, the model gets the answer and the
    turn completes.

    The fault is a database trigger that moves the parent's row out of reach the
    instant the child's input row is written, and the child's own model call puts
    it back once the warning has been logged — so the row is missing for exactly
    the announcement and present again for the finish."""
    gate = threading.Event()
    provider = _HeldChildProvider(
        gate,
        lambda: _write_on_this_thread("UPDATE tool_calls SET id = -id WHERE id < 0"),
        ProviderResponse(text="", model="m", tool_calls=[_tool("web_search", instructions=_INSTRUCTIONS)]),
        ProviderResponse(text=_CHILD_ANSWER, model="m", tool_calls=None),
        ProviderResponse(text="It is Paris.", model="m", tool_calls=None),
    )
    db.execute(
        "CREATE TRIGGER vanish_caller_row AFTER INSERT ON transcript WHEN NEW.tool_call_id IS NOT NULL "
        "BEGIN UPDATE tool_calls SET id = -id WHERE id = CAST(NEW.tool_call_id AS INTEGER); END"
    )
    db.commit()

    with _watching_warnings(gate) as watch:
        parent, frames = _run(provider, "what is the capital of France?")

    assert provider.gate_opened, "the failed announcement must be reported with a warning"
    assert watch.records
    call = _web_search_call(db)
    assert call["state"] == "done" and _CHILD_ANSWER in call["result"]
    _assert_child_answered_and_closed_before_the_call_ended(db, call)
    _assert_parent_turn_completed_with_its_answer(db, parent)
    web_search = [f for f in frames if f.get("tool_name") == "web_search" and "channel" not in f]
    assert not any(f["state"] == "started" and f["delegate"] is not None for f in web_search), (
        "the announcement must not have gone out"
    )
    child_turn = db.execute(
        "SELECT turn_id FROM transcript WHERE channel = ? AND role = 'web_search'", (_CHILD_CHANNEL,),
    ).fetchone()["turn_id"]
    assert web_search[-1]["state"] == "done"
    assert web_search[-1]["delegate"] == {"channel": _CHILD_CHANNEL, "turn_id": child_turn}


@dataclass(frozen=True)
class _SpoiledStampRun:
    """One finished user turn whose ``web_search`` call's own timestamp became
    unreadable once its child was running."""

    parent: MessageProcessor
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
    so only the lookup of "which child did this call spawn" at finish can trip."""
    announced = threading.Event()

    def watch_for_announcement(frame: dict[str, object]) -> None:
        if (
            frame.get("tool_name") == "web_search" and "channel" not in frame
            and frame["state"] == "started" and frame["delegate"] is not None
        ):
            announced.set()

    provider = _HeldChildProvider(
        announced,
        lambda: _write_on_this_thread("UPDATE tool_calls SET created_at = 'not-a-timestamp' WHERE tool_name = 'web_search'"),
        ProviderResponse(text="", model="m", tool_calls=[_tool("web_search", instructions=_INSTRUCTIONS)]),
        ProviderResponse(text=_CHILD_ANSWER, model="m", tool_calls=None),
        ProviderResponse(text="It is Paris.", model="m", tool_calls=None),
    )
    with _watching_warnings() as watch:
        parent, frames = _run(provider, "what is the capital of France?", on_frame=watch_for_announcement)
    assert provider.gate_opened, "the call must have been announced before the stamp was spoiled"
    return _SpoiledStampRun(parent, frames, watch.records)


def test_unreadable_call_stamp_does_not_fail_the_delegate_call_or_its_turn(
    db: sqlite3.Connection, spoiled_stamp_run: _SpoiledStampRun,
) -> None:
    """Finding the child a call spawned is bookkeeping for the pill. When the
    call's own timestamp cannot be read at the moment it finishes, the call still
    finishes with the child's answer, the child runs to its end first, and the
    parent's turn completes with its reply."""
    call = _web_search_call(db)

    assert call["created_at"] == "not-a-timestamp"
    assert call["state"] == "done" and _CHILD_ANSWER in call["result"]
    _assert_child_answered_and_closed_before_the_call_ended(db, call)
    _assert_parent_turn_completed_with_its_answer(db, spoiled_stamp_run.parent)


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
