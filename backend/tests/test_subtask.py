"""Feature test: a subtask runs beside its caller and always answers it.

A ``subtask`` call opens a background turn on the ``delegate:subtask`` channel
and returns at once, so the caller keeps working. Whatever way the subtask
ends, its outcome comes back on the caller's channel as an input row linked to
the call that asked for it ("Reply from subtask #N"):

* it joins the caller's turn while that turn still works, and opens a new user
  turn once the caller has finished — or, for a scheduled task, continues the
  schedule's own thread;
* a subtask cannot start one of its own, and only the conversation that owns a
  subtask can message it;
* a subtask stopped by its caller's stop sends nothing back — no one is
  waiting — while one stopped on its own, one that crashed, and one a restart
  cut off each send a notice saying so;
* the memory step runs on the subtask channel too, in a turn of its own, and
  that turn has no caller: it sends nothing back, finished or cut off.

Drives the real production entry point (construct inertly, ``begin()``,
``result()``) against the real, fully-migrated SQLite DB: the real
``SubtaskAbility``, ``DispatchService``, ``TurnExecutionService``, boot-time
sweep and reply, and Flask thread reads all run. The only substitution is the
LLM network boundary (``services.provider_service.build_client``). A subtask
and its caller run on separate threads at once, so the fake provider routes
each request by what it contains, and a turn whose order matters is pinned by
holding its provider call on a ``threading.Event``.
"""

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import NoReturn, TypeVar, cast
from unittest.mock import patch

import pytest
from flask.testing import FlaskClient

from configs.channels import config_for
from configs.channels.scheduled import ScheduledConfig
from configs.channels.subtask import UserSubtaskConfig
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from configs.enums.config_type import ConfigTypeEnum
from controllers.message_processor import _MAX_PROVIDER_ATTEMPTS, MessageProcessor
from models.provider_request import ProviderRequest
from models.provider_response import ProviderResponse
from models.tool_call import ToolCall
from models.transcript import Transcript
from models.turn_execution import TurnExecution
from run import _reply_to_swept_subtasks
from services.memory_step_service import memory_step_config
from services.turn_execution_service import TurnExecutionService
from tests.helpers import NO_PLAN, is_planning_request

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

# ProviderService builds its thin transport client via this factory call — the
# real network boundary (see test_message_processor_runaway_loop.py).
_BUILD_CLIENT = "services.provider_service.build_client"

_SUBTASK = Channel.DELEGATE_SUBTASK.value
_USER = Channel.USER.value
_SCHEDULE = Channel.SCHEDULE.value
_TASK = "list the primes below 6"
_ANSWER = "2, 3, 5"
_EXHAUSTED = "The AI provider failed to respond after several attempts. Please try again in a moment."
_CUT_OFF = "This subtask was cut off by a restart before it finished."
_WAIT_S = 15.0

_T = TypeVar("_T")


def _tool(name: str, **params: object) -> dict[str, object]:
    """A provider-shaped tool call: ``{"name": ..., "input": {...}}``."""
    return {"name": name, "input": params}


def _calls(*tools: dict[str, object]) -> ProviderResponse:
    return ProviderResponse(text="", model="m", tool_calls=list(tools))


def _text(text: str) -> ProviderResponse:
    return ProviderResponse(text=text, model="m", tool_calls=None)


_START = _calls(_tool("subtask", action="start", instructions=_TASK))


class _Held:
    """A script entry that holds its turn's provider call until the test
    releases it, then answers with *outcome* — or raises it, for a failing
    call. ``entered`` is set the moment the call arrives, so the test can act
    on the turn while it is mid-call; ``opened`` records whether the wait
    ended because the gate opened rather than because it timed out, so a test
    fails on a gate that never opened instead of on its downstream symptom."""

    _GATE_TIMEOUT_S = 10.0

    def __init__(self, outcome: ProviderResponse | Exception) -> None:
        self._outcome = outcome
        self.entered = threading.Event()
        self.gate = threading.Event()
        self.opened = False

    def __call__(self) -> ProviderResponse:
        self.entered.set()
        self.opened = self.gate.wait(timeout=self._GATE_TIMEOUT_S)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def _raises(exc: Exception) -> Callable[[], ProviderResponse]:
    """A script entry whose provider call fails with *exc*."""
    def entry() -> ProviderResponse:
        raise exc
    return entry


class _ProcessDeath(BaseException):
    """The process dying mid-call. Raised from a provider call it is no
    ``Exception``, so nothing in the turn catches it: the turn's thread ends
    with its execution row still open and nothing reported — what a killed
    process leaves behind for the next boot to sweep."""


def _dies() -> NoReturn:
    """A script entry whose provider call is the last thing its process does."""
    raise _ProcessDeath


@contextmanager
def _process_deaths_expected() -> Iterator[None]:
    """Let turn threads die of :class:`_ProcessDeath` inside the block without
    it counting as an unhandled thread exception; any other exception a
    thread dies of still reaches the hook that was there before."""
    previous = threading.excepthook

    def hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is not _ProcessDeath:
            previous(args)

    threading.excepthook = hook
    try:
        yield
    finally:
        threading.excepthook = previous


_Entry = ProviderResponse | Callable[[], ProviderResponse]


class _RoutedProvider:
    """Answers the caller's turns and the subtask's turns from separate scripts.

    A subtask runs on its own thread while its caller keeps working, so one
    ordered script cannot say whose call comes next. Each request is routed
    by what it carries instead: a request that offers no ``find_tools`` is
    background work a finished turn spawns (the memory step among it, which
    reads the subtask's thread too, so this is asked first); the subtask's own
    conversation opens with its "Task from Chalie" row; anything else is a
    chat turn. Each script is consumed in order under a lock — past its end
    the terminal response is returned, and a test asserts its scripts were
    used up, so a response consumed by the wrong turn fails it. Background
    work past its script answers "nothing new", which is how the memory step
    ends a pass with nothing to save. A callable entry runs outside the lock,
    so a held call never blocks the other route. Every request is recorded as
    (route, text, offered tool names)."""

    _TERMINAL = ProviderResponse(text="", model="overflow", tool_calls=None)
    _NOTHING_NEW = ProviderResponse(text="nothing new", model="overflow", tool_calls=None)

    def __init__(
        self, *, chat: list[_Entry], subtask: list[_Entry] | None = None, background: list[_Entry] | None = None,
    ) -> None:
        self._scripts: dict[str, list[_Entry]] = {"chat": list(chat), "subtask": list(subtask or [])}
        if background is not None:
            self._scripts["background"] = list(background)
        self._held = [e for script in self._scripts.values() for e in script if isinstance(e, _Held)]
        self._lock = threading.Lock()
        self.requests: list[tuple[str, str, frozenset[str]]] = []

    def get_context_limit(self) -> int:
        return 200000

    def abort(self) -> None:
        """A stopped turn's watcher aborts its call in flight; a held call
        still waits for its own gate, which the test opens."""

    def send(self, dto: ProviderRequest) -> ProviderResponse:
        if is_planning_request(dto):
            return NO_PLAN
        text = json.dumps(dto.messages, default=str)
        names = frozenset(str(tool["name"]) for tool in dto.tools or [])
        if "find_tools" not in names:
            route = "background"
        elif "Task from Chalie" in text:
            route = "subtask"
        else:
            route = "chat"
        with self._lock:
            self.requests.append((route, text, names))
            script = self._scripts.get(route)
            if not script:
                return self._NOTHING_NEW if route == "background" else self._TERMINAL
            entry = script.pop(0)
        return entry() if callable(entry) else entry

    def texts(self, route: str) -> list[str]:
        """The text of every request routed to *route*, in arrival order."""
        with self._lock:
            return [text for r, text, _names in self.requests if r == route]

    def unused(self) -> dict[str, int]:
        """How many entries each script still holds (background work only
        when the test scripted it)."""
        with self._lock:
            return {route: len(script) for route, script in self._scripts.items()}

    def release_all(self) -> None:
        """Open every held entry's gate, consumed or not, so no turn waits on a
        test that has already failed."""
        for held in self._held:
            held.gate.set()


def _turn_threads() -> list[threading.Thread]:
    """The live threads turns run on, and the background work a finished turn
    spawns (post-turn work, the memory step)."""
    return [
        t for t in threading.enumerate()
        if t.name in ("skill-suggest", "thread-gist") or t.name.startswith(("turn-", "memory-step-"))
    ]


def _drain_background_turns(timeout_s: float = _WAIT_S) -> None:
    """Join every turn and the background work turns spawn, so they finish
    inside THIS test's provider+DB patch and never leak into the next — and
    fail loudly if any is still running when the deadline passes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pending = _turn_threads()
        if not pending:
            return
        for t in pending:
            t.join(timeout=max(0.0, deadline - time.monotonic()))
    leftover = [t.name for t in _turn_threads()]
    assert not leftover, f"still running after {timeout_s:.0f}s: {leftover}"


@contextmanager
def _patched(provider: _RoutedProvider) -> Iterator[None]:
    """Serve every provider call from *provider* for the block, and before
    leaving it open every held call and wait for every turn to end."""
    with patch(_BUILD_CLIENT, return_value=provider):
        try:
            yield
        finally:
            provider.release_all()
            _drain_background_turns()


def _poll(probe: Callable[[], _T], what: str, timeout_s: float = _WAIT_S) -> _T:
    """Return *probe*'s first truthy value, failing the test if none comes
    within *timeout_s*."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = probe()
        if value:
            return value
        time.sleep(0.02)
    pytest.fail(f"timed out after {timeout_s:.0f}s waiting for {what}")


def _wait(event: threading.Event, what: str) -> None:
    assert event.wait(timeout=_WAIT_S), f"timed out after {_WAIT_S:.0f}s waiting for {what}"


def _begin(raw_input: str) -> MessageProcessor:
    """Start a real user turn on its own thread and return its processor."""
    mp = MessageProcessor(UserConfig(), raw_input=raw_input)  # inert (I2)
    mp.begin()
    return mp


def _subtask_calls(db: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every ``subtask`` tool call, oldest first."""
    return db.execute(
        "SELECT id, state, result FROM tool_calls WHERE tool_name = 'subtask' ORDER BY id"
    ).fetchall()


def _subtask_turn(db: sqlite3.Connection, call_id: int) -> int:
    """The subtask turn the call *call_id* started — read off the subtask's
    first input row, which carries the call's id."""
    row = db.execute(
        "SELECT turn_id FROM transcript WHERE channel = ? AND tool_call_id = ? ORDER BY id LIMIT 1",
        (_SUBTASK, str(call_id)),
    ).fetchone()
    assert row is not None, f"no subtask turn carries call {call_id}"
    return cast("int", row["turn_id"])


def _replies(db: sqlite3.Connection, call_id: int, channel: str = _USER) -> list[sqlite3.Row]:
    """Every reply row a subtask sent back for the call *call_id*: an input
    row on the caller's *channel* linked to that call."""
    return db.execute(
        "SELECT content, turn_id, joined FROM transcript "
        "WHERE channel = ? AND role = 'user' AND tool_call_id = ? ORDER BY id",
        (channel, str(call_id)),
    ).fetchall()


def _all_replies(db: sqlite3.Connection) -> list[str]:
    """The content of every reply row any subtask sent to the user's
    conversation, oldest first."""
    return [
        cast("str", r["content"]) for r in db.execute(
            "SELECT content FROM transcript WHERE channel = ? AND role = 'user' AND tool_call_id IS NOT NULL "
            "ORDER BY id",
            (_USER,),
        ).fetchall()
    ]


def _memory_turns(db: sqlite3.Connection) -> list[int]:
    """Every memory-step turn on the subtask channel — the turns whose
    opening row carries the ``memory`` role."""
    return [
        cast("int", r["turn_id"]) for r in db.execute(
            "SELECT DISTINCT turn_id FROM transcript WHERE channel = ? AND role = 'memory' ORDER BY turn_id",
            (_SUBTASK,),
        ).fetchall()
    ]


def _errors(caplog: pytest.LogCaptureFixture, logger: str) -> list[str]:
    """Every ERROR record *logger* emitted during the test."""
    return [r.getMessage() for r in caplog.records if r.name == logger and r.levelno >= logging.ERROR]


def _await_reply(db: sqlite3.Connection, call_id: int, channel: str = _USER) -> sqlite3.Row:
    """The one reply row for *call_id*, waiting for it to land."""
    rows = _poll(lambda: _replies(db, call_id, channel), f"the reply to call {call_id}")
    assert len(rows) == 1, [tuple(r) for r in rows]
    return rows[0]


def _execution(channel: str, turn_id: int) -> TurnExecution:
    execution = TurnExecution.latest(channel, turn_id)
    assert execution is not None, f"no turn execution for {channel} turn {turn_id}"
    return execution


def _block(client: FlaskClient, path: str) -> dict[str, object]:
    """GET *path* and return the thread block under ``result``."""
    response = client.get(path)
    assert response.status_code == 200, response.data
    return cast("dict[str, object]", cast("dict[str, object]", response.get_json())["result"])


def _messages(block: dict[str, object]) -> list[dict[str, object]]:
    return cast("list[dict[str, object]]", block["messages"])


def _chips(block: dict[str, object]) -> list[dict[str, object]]:
    return [
        chip for message in _messages(block)
        for chip in cast("list[dict[str, object]]", message.get("tool_calls") or [])
    ]


def _chip(block: dict[str, object], tool_name: str) -> dict[str, object]:
    return next(c for c in _chips(block) if c["tool_name"] == tool_name)


def test_reply_after_the_caller_finished_opens_a_new_user_turn(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], db: sqlite3.Connection,
) -> None:
    """The caller answers and finishes while its subtask still works; the
    subtask's answer then arrives as the opening message of a new user turn,
    labelled with the subtask's number, and that turn answers it. The caller's
    pill points at the subtask, which no longer works; the subtask's thread
    opens with the task, labelled as Chalie's."""
    client, _db, _store = authed_client
    held = _Held(_text(_ANSWER))
    provider = _RoutedProvider(
        chat=[_START, _text("Started it."), _text("Got the reply.")], subtask=[held],
    )

    with _patched(provider):
        parent = _begin("find the primes below 6 in the background")
        parent.result()
        _wait(held.entered, "the subtask's provider call")
        call_id = cast("int", _subtask_calls(db)[0]["id"])
        held.gate.set()
        reply = _await_reply(db, call_id)

    assert held.opened
    assert provider.unused() == {"chat": 0, "subtask": 0}
    n = _subtask_turn(db, call_id)
    assert (reply["content"], reply["joined"]) == (_ANSWER, 0)
    assert reply["turn_id"] != parent.turn_id
    assert _execution(_SUBTASK, n).state == TurnExecution.COMPLETED
    assert _execution(_USER, reply["turn_id"]).state == TurnExecution.COMPLETED
    assert f"Reply from subtask #{n}" in provider.texts("chat")[-1]

    opener = _messages(_block(client, f"/api/threads/{reply['turn_id']}"))[0]
    assert (opener["role"], opener["content"], opener["speaker"]) == ("user", _ANSWER, f"Reply from subtask #{n}")
    parent_block = _block(client, f"/api/threads/{parent.turn_id}")
    assert _chip(parent_block, "subtask")["delegate"] == {"channel": _SUBTASK, "turn_id": n, "working": False}
    task = _messages(_block(client, f"/api/threads/{n}?channel={_SUBTASK}"))[0]
    assert (task["content"], task["speaker"]) == (_TASK, "Task from Chalie")


def test_reply_joins_the_caller_turn_while_it_still_works(db: sqlite3.Connection) -> None:
    """A subtask that finishes while its caller is mid-call lands its answer on
    the caller's own turn as a joined message; the caller does not end on the
    call it was making but steps again with the reply in view."""
    caller_held = _Held(_text("Done for now."))
    subtask_held = _Held(_text(_ANSWER))
    provider = _RoutedProvider(
        chat=[_START, caller_held, _text("Got the reply.")], subtask=[subtask_held],
    )

    with _patched(provider):
        parent = _begin("find the primes below 6 in the background")
        _wait(caller_held.entered, "the caller's second provider call")
        _wait(subtask_held.entered, "the subtask's provider call")
        subtask_held.gate.set()
        call_id = cast("int", _subtask_calls(db)[0]["id"])
        reply = _await_reply(db, call_id)
        caller_held.gate.set()
        parent.result()

    assert caller_held.opened and subtask_held.opened
    assert provider.unused() == {"chat": 0, "subtask": 0}
    n = _subtask_turn(db, call_id)
    assert (reply["content"], reply["turn_id"], reply["joined"]) == (_ANSWER, parent.turn_id, 1)
    assert f"Reply from subtask #{n}" in provider.texts("chat")[-1]
    assert _execution(_USER, parent.turn_id).state == TurnExecution.COMPLETED
    count = db.execute(
        "SELECT COUNT(*) FROM turn_executions WHERE channel = ? AND turn_id = ?", (_USER, parent.turn_id),
    ).fetchone()[0]
    assert count == 1
    final = db.execute(
        "SELECT content FROM transcript WHERE channel = ? AND turn_id = ? AND role = 'assistant' "
        "ORDER BY id DESC LIMIT 1",
        (_USER, parent.turn_id),
    ).fetchone()
    assert final["content"] == "Got the reply."


def test_a_subtask_cannot_start_a_subtask(db: sqlite3.Connection) -> None:
    """A subtask's own ``subtask`` call is refused: no second subtask opens,
    the refusal is what the subtask sees next, and it still answers its
    caller."""
    nested = _Held(_calls(_tool("subtask", action="start", instructions="count to three")))
    provider = _RoutedProvider(
        chat=[_START, _text("Started it."), _text("Got the reply.")],
        subtask=[nested, _text("I did it myself.")],
    )

    with _patched(provider):
        parent = _begin("find the primes below 6 in the background")
        parent.result()
        _wait(nested.entered, "the subtask's provider call")
        nested.gate.set()
        start_id = cast("int", _subtask_calls(db)[0]["id"])
        reply = _await_reply(db, start_id)

    assert nested.opened
    assert provider.unused() == {"chat": 0, "subtask": 0}
    calls = _subtask_calls(db)
    assert len(calls) == 2
    refused = calls[1]
    assert refused["state"] == ToolCall.ERROR
    assert "code=subtask-nested" in refused["result"]
    linked = db.execute(
        "SELECT COUNT(*) FROM transcript WHERE channel = ? AND tool_call_id IS NOT NULL", (_SUBTASK,),
    ).fetchone()[0]
    assert linked == 1
    assert "subtask-nested" in provider.texts("subtask")[-1]
    assert reply["content"] == "I did it myself."


def test_message_reaches_a_working_subtask_and_an_unknown_one_is_refused(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], db: sqlite3.Connection,
) -> None:
    """The caller messages its working subtask and a number no subtask of its
    has: the first joins the subtask's turn, labelled as Chalie's, and the
    subtask answers with it in view — the reply then goes back to the call
    that sent the message; the second is refused as not found."""
    client, _db, _store = authed_client
    subtask_held = _Held(_text(_ANSWER))

    def message_both() -> ProviderResponse:
        _wait(subtask_held.entered, "the subtask's provider call")
        started = TurnExecution.filter("channel", _SUBTASK).first()
        assert started is not None
        return _calls(
            _tool("subtask", action="message", subtask=started.turn_id, message="also add 7"),
            _tool("subtask", action="message", subtask=9999, message="hello"),
        )

    provider = _RoutedProvider(
        chat=[_START, message_both, _text("Told it."), _text("Got the reply.")],
        subtask=[subtask_held, _text("2, 3, 5, 7")],
    )

    with _patched(provider):
        parent = _begin("find the primes below 6 in the background")
        parent.result()
        subtask_held.gate.set()
        start, sent, refused = _subtask_calls(db)
        reply = _await_reply(db, cast("int", sent["id"]))

    assert subtask_held.opened
    assert provider.unused() == {"chat": 0, "subtask": 0}
    n = _subtask_turn(db, cast("int", start["id"]))
    assert sent["state"] == ToolCall.DONE
    assert f"Message sent to subtask #{n}." in sent["result"]
    assert refused["state"] == ToolCall.ERROR
    assert "code=subtask-not-found" in refused["result"]
    assert "No subtask #9999 belongs to this conversation." in refused["result"]

    message = db.execute(
        "SELECT turn_id, joined FROM transcript WHERE channel = ? AND tool_call_id = ?",
        (_SUBTASK, str(sent["id"])),
    ).fetchone()
    assert (message["turn_id"], message["joined"]) == (n, 1)
    assert "also add 7" in provider.texts("subtask")[-1]
    assert reply["content"] == "2, 3, 5, 7"
    assert _replies(db, cast("int", start["id"])) == []

    thread = _messages(_block(client, f"/api/threads/{n}?channel={_SUBTASK}"))
    assert [(m["content"], m["speaker"]) for m in thread if m["role"] == "user"] == [
        (_TASK, "Task from Chalie"), ("also add 7", "Message from Chalie"),
    ]


def test_stopping_the_caller_stops_its_subtask_without_a_reply(db: sqlite3.Connection) -> None:
    """Stopping a caller mid-turn stops the subtask it started as well, marked
    as stopped by its caller, and the subtask sends nothing back: no one is
    waiting for its answer."""
    caller_held = _Held(_text("Done for now."))
    subtask_held = _Held(_text(_ANSWER))
    provider = _RoutedProvider(chat=[_START, caller_held], subtask=[subtask_held])

    with _patched(provider):
        parent = _begin("find the primes below 6 in the background")
        _wait(caller_held.entered, "the caller's second provider call")
        _wait(subtask_held.entered, "the subtask's provider call")
        TurnExecutionService.cancel_turn(_USER, parent.turn_id)
        caller_held.gate.set()
        subtask_held.gate.set()
        parent.result()

    assert caller_held.opened and subtask_held.opened
    call_id = cast("int", _subtask_calls(db)[0]["id"])
    n = _subtask_turn(db, call_id)
    caller = _execution(_USER, parent.turn_id)
    assert (caller.state, caller.stop_reason) == (TurnExecution.CANCELLED, None)
    subtask = _execution(_SUBTASK, n)
    assert (subtask.state, subtask.stop_reason) == (TurnExecution.CANCELLED, "parent stopped")
    assert _replies(db, call_id) == []


def test_a_stopped_subtask_tells_its_caller(db: sqlite3.Connection) -> None:
    """A subtask stopped on its own — its caller still fine — sends a notice
    that it was stopped before it finished."""
    held = _Held(_text(_ANSWER))
    provider = _RoutedProvider(
        chat=[_START, _text("Started it."), _text("Noted.")], subtask=[held],
    )

    with _patched(provider):
        parent = _begin("find the primes below 6 in the background")
        parent.result()
        _wait(held.entered, "the subtask's provider call")
        call_id = cast("int", _subtask_calls(db)[0]["id"])
        n = _subtask_turn(db, call_id)
        TurnExecutionService.cancel_turn(_SUBTASK, n)
        held.gate.set()
        reply = _await_reply(db, call_id)

    assert held.opened
    assert provider.unused() == {"chat": 0, "subtask": 0}
    subtask = _execution(_SUBTASK, n)
    assert (subtask.state, subtask.stop_reason) == (TurnExecution.CANCELLED, None)
    assert reply["content"] == "This subtask was stopped before it finished."
    assert reply["turn_id"] != parent.turn_id


def test_a_crashed_subtask_tells_its_caller_why(db: sqlite3.Connection) -> None:
    """A subtask whose provider fails on every attempt crashes, and its caller
    is told so with the reason the turn ended on."""
    boom = RuntimeError("boom")
    first = _Held(boom)
    provider = _RoutedProvider(
        chat=[_START, _text("Started it."), _text("Noted.")],
        subtask=[first, *[_raises(boom) for _ in range(_MAX_PROVIDER_ATTEMPTS - 1)]],
    )

    with _patched(provider):
        parent = _begin("find the primes below 6 in the background")
        parent.result()
        _wait(first.entered, "the subtask's provider call")
        first.gate.set()
        call_id = cast("int", _subtask_calls(db)[0]["id"])
        reply = _await_reply(db, call_id)

    assert first.opened
    assert provider.unused() == {"chat": 0, "subtask": 0}
    assert len(provider.texts("subtask")) == _MAX_PROVIDER_ATTEMPTS
    subtask = _execution(_SUBTASK, _subtask_turn(db, call_id))
    assert (subtask.state, subtask.stop_reason) == (TurnExecution.CRASHED, _EXHAUSTED)
    assert reply["content"] == f"This subtask crashed: {_EXHAUSTED}"


def test_the_memory_step_after_a_subtask_answers_no_one(
    real_memory_step: None, db: sqlite3.Connection, caplog: pytest.LogCaptureFixture,
) -> None:
    """With the real memory step armed, a finished subtask's settle runs a
    memory step on the subtask channel, in a turn of its own. Only the
    subtask's own turn answers the caller: exactly one reply reaches it, and
    nothing fails trying to report the memory step's turn to a caller it does
    not have."""
    held = _Held(_text(_ANSWER))
    provider = _RoutedProvider(
        chat=[_START, _text("Started it."), _text("Got the reply.")], subtask=[held],
    )

    with _patched(provider):
        parent = _begin("find the primes below 6 in the background")
        parent.result()
        _wait(held.entered, "the subtask's provider call")
        call_id = cast("int", _subtask_calls(db)[0]["id"])
        held.gate.set()
        _await_reply(db, call_id)

    assert held.opened
    assert provider.unused() == {"chat": 0, "subtask": 0}
    n = _subtask_turn(db, call_id)
    [memory] = _memory_turns(db)
    assert memory != n
    assert _execution(_SUBTASK, memory).state == TurnExecution.COMPLETED
    assert _all_replies(db) == [_ANSWER]
    assert _errors(caplog, "controllers.message_processor") == []


def test_boot_tells_a_cut_off_subtask_s_caller_once_and_a_cut_off_memory_step_no_one(
    db: sqlite3.Connection, caplog: pytest.LogCaptureFixture,
) -> None:
    """The process dies with two turns open on the subtask channel: a subtask
    mid-call, and the memory step of an earlier subtask that had already
    answered. Boot closes both as cut off by the restart and tells the cut-off
    subtask's caller exactly once, in a new user turn that answers it; the
    memory step's turn has no caller, so it gets no notice and nothing fails
    trying to send one."""
    first = _Held(_text(_ANSWER))
    provider = _RoutedProvider(
        chat=[
            _START, _text("Started it."), _text("Got the reply."),
            _START, _text("Started another."), _text("Noted."),
        ],
        subtask=[first, _dies],
        background=[_dies],
    )

    with _patched(provider), _process_deaths_expected():
        _begin("find the primes below 6 in the background").result()
        _wait(first.entered, "the first subtask's provider call")
        first.gate.set()
        answered_id = cast("int", _subtask_calls(db)[0]["id"])
        _await_reply(db, answered_id)
        _poll(lambda: not _turn_threads(), "the reply turn to end")
        ids = Transcript.ids_for_turn(_SUBTASK, _subtask_turn(db, answered_id))
        MessageProcessor.process(memory_step_config(UserSubtaskConfig(), ids)).result()
        caller = _begin("now find the primes below 10 in the background")
        caller.result()
        _poll(lambda: not _turn_threads(), "every turn the dying process left to end")
        cut_id = cast("int", _subtask_calls(db)[1]["id"])
        cut = _subtask_turn(db, cut_id)
        [memory] = _memory_turns(db)
        assert sorted((r.channel, r.turn_id) for r in TurnExecution.open_rows()) == sorted([
            (_SUBTASK, memory), (_SUBTASK, cut),
        ])

        swept = MessageProcessor(config_for(ConfigTypeEnum.USER)).turn_execution_service.sweep_orphaned()
        _reply_to_swept_subtasks(swept)
        notice = _await_reply(db, cut_id)

    assert first.opened
    assert provider.unused() == {"chat": 0, "subtask": 0, "background": 0}
    assert sorted((r.channel, r.turn_id, r.state, r.stop_reason) for r in swept) == sorted([
        (_SUBTASK, memory, TurnExecution.CRASHED, "process death"),
        (_SUBTASK, cut, TurnExecution.CRASHED, "process death"),
    ])
    assert _all_replies(db) == [_ANSWER, _CUT_OFF]
    assert notice["turn_id"] != caller.turn_id
    assert notice["joined"] == 0
    assert _execution(_USER, cast("int", notice["turn_id"])).state == TurnExecution.COMPLETED
    assert _errors(caplog, "run") == []
    assert list(TurnExecution.open_rows()) == []


def test_a_scheduled_task_hears_from_its_subtask_on_its_own_thread(db: sqlite3.Connection) -> None:
    """A scheduled task can start a subtask, and the subtask's answer
    continues the schedule's own thread: an input row on the schedule
    channel, on that schedule's turn, linked to the call that started it,
    which the schedule then answers. The user's conversation does not own
    that subtask, so its message to it is refused as not found and never
    reaches it."""
    held = _Held(_text(_ANSWER))

    def message_it() -> ProviderResponse:
        started = TurnExecution.filter("channel", _SUBTASK).first()
        assert started is not None
        return _calls(_tool("subtask", action="message", subtask=started.turn_id, message="also add 7"))

    provider = _RoutedProvider(
        chat=[_START, _text("Started it."), message_it, _text("Told it."), _text("Got the reply.")],
        subtask=[held],
    )

    with _patched(provider):
        schedule = MessageProcessor.process(ScheduledConfig(), "write a status update")
        schedule.result()
        start = _subtask_calls(db)[0]
        assert start["state"] == ToolCall.DONE, start["result"]
        _wait(held.entered, "the subtask's provider call")
        _begin("tell the schedule's subtask to also add 7").result()
        refused = _subtask_calls(db)[1]
        assert refused["state"] == ToolCall.ERROR, refused["result"]
        assert "code=subtask-not-found" in refused["result"]
        held.gate.set()
        call_id = cast("int", start["id"])
        reply = _await_reply(db, call_id, _SCHEDULE)

    assert held.opened
    assert provider.unused() == {"chat": 0, "subtask": 0}
    n = _subtask_turn(db, call_id)
    assert (reply["content"], reply["turn_id"], reply["joined"]) == (_ANSWER, schedule.turn_id, 0)
    assert _replies(db, call_id) == []
    assert f"Reply from subtask #{n}" in provider.texts("chat")[-1]
    assert _execution(_SCHEDULE, schedule.turn_id).state == TurnExecution.COMPLETED
    reached = db.execute(
        "SELECT COUNT(*) FROM transcript WHERE channel = ? AND tool_call_id = ?", (_SUBTASK, str(refused["id"])),
    ).fetchone()[0]
    assert reached == 0
