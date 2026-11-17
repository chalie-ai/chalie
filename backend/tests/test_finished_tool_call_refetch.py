"""Feature test: a finished tool call makes its surface refetch the turn.

A turn's block carries every tool call from the moment it starts, so a refetch
taken while a call runs shows it in flight. The frontend only refetches on an
``updated`` signal; a tool frame alone just moves the live pill. Without a
signal after the call ends, a call that was running at the last refetch keeps
showing in flight under the reply until the next assistant row lands — minutes,
for a long delegate. Each finished call (done, failed or rejected before it
ran) must therefore be followed on the same address by ``updated``, and the
block that refetch reads must already show the call finished. A call that is
still running sends no ``updated``: the live pill shows it. A subagent the chat
calls returns at once and its child works on in its own turn, so the subagent's
call and its child's own calls finish on different surfaces at different
times; each still sends its own ``updated``.

Drives the real production entry point (construct inertly, ``begin()``,
``result()``) against the real, fully-migrated SQLite DB, observing frames
through the REAL socket registry and reading the turn back through the real
Flask route from another thread — its own database connection, as a browser's
refetch request gets — the moment each ``updated`` is broadcast. The only
substitution is the LLM network boundary (``services.provider_service.build_client``).
"""

import sqlite3
import threading
from dataclasses import dataclass
from typing import cast

import pytest
from flask.testing import FlaskClient
from werkzeug.test import TestResponse

from abilities.chalie_docs import ChalieDocsAbility
from abilities.find_tools import FindToolsAbility
from abilities.message_subagent import MessageSubagentAbility
from abilities.recall import Recall
from abilities.web_search import WebSearchAbility
from configs.channels.user import UserConfig
from models.provider_response import ProviderResponse
from tests._delegates import ChannelProvider, Held, calls, chip, delegate_channel, run_turn, text, tool

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

_READ_TIMEOUT_S = 30.0
_STEP_WAIT_S = 10.0
_CHILD_CHANNEL = delegate_channel(WebSearchAbility.NAME)

Frame = dict[str, object]


def _address(frame: Frame) -> tuple[object, object]:
    """The surface a frame is for: a delegate turn by its channel, any other by its type."""
    return frame.get("channel") or frame.get("type"), frame.get("turn_id")


def _thread_path(frame: Frame) -> str:
    """The REST read of the turn a frame addresses."""
    if "channel" in frame:
        return f"/api/threads/{frame['turn_id']}?channel={frame['channel']}"
    return f"/api/threads/{frame['turn_id']}?type={frame['type']}"


def _next_on_address(frames: list[Frame], frame: Frame) -> Frame | None:
    """The first frame after *frame* that addresses the same surface."""
    at = next(i for i, f in enumerate(frames) if f is frame)
    return next((f for f in frames[at + 1:] if _address(f) == _address(frame)), None)


class _AfterRefetch:
    """A model step that waits until its surface has refetched the call that
    finished just before it. Without its own ``updated``, a finished call would
    only be refetched once this step's row lands — so ``in_time`` records
    whether the call's own poke released the step rather than its wait running
    out."""

    def __init__(self, outcome: ProviderResponse) -> None:
        self._outcome = outcome
        self.released = threading.Event()
        self.in_time = False

    def __call__(self) -> ProviderResponse:
        self.in_time = self.released.wait(timeout=_STEP_WAIT_S)
        return self._outcome


@dataclass(frozen=True)
class _RefetchRun:
    """A finished user turn whose tool calls ended every way a call can end, the
    frames it broadcast, and the turn read back the moment each ``updated`` arrived."""

    frames: list[Frame]
    reads: list[tuple[Frame, TestResponse | None]]
    next_steps: list[_AfterRefetch]

    @property
    def finished(self) -> list[Frame]:
        return [f for f in self.frames if "tool_name" in f and f["state"] != "started"]

    def read_at(self, poke: Frame) -> TestResponse | None:
        return next(response for frame, response in self.reads if frame is poke)


@pytest.fixture
def refetch_run(authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> _RefetchRun:
    """A user turn that runs an ordinary tool to ``done``, sends a call its tool
    rejects before running it, calls ``web_search`` — which returns at once,
    ``done``, while its child turn runs ``recall`` to ``done`` after the user
    turn has ended — then messages a subagent this conversation never started,
    which fails while running, and replies; the child's reply is answered in a
    turn of its own. Each call runs in its own model step, so no two calls on
    one surface are ever in flight together, and the step after each finished
    call waits for that surface's refetch, as a slow model would."""
    client, _db, _store = authed_client
    reads: list[tuple[Frame, TestResponse | None]] = []
    lock = threading.Lock()
    waiting = {
        "user": [
            _AfterRefetch(calls(tool(ChalieDocsAbility.NAME))),
            _AfterRefetch(calls(tool(WebSearchAbility.NAME, instructions="find the capital of France"))),
            _AfterRefetch(calls(tool(MessageSubagentAbility.NAME, agent_id=9999, message="any news?"))),
            _AfterRefetch(text("I could not find it.")),
        ],
        _CHILD_CHANNEL: [_AfterRefetch(text(""))],
    }
    next_steps = [step for steps in waiting.values() for step in steps]
    just_finished: set[str] = set()

    def refetch_on_poke(frame: Frame) -> None:
        surface = str(_address(frame)[0])
        if surface not in waiting:
            return
        if "tool_name" in frame and frame["state"] != "started":
            just_finished.add(surface)
            return
        if frame.get("status") != "updated":
            return
        got: list[TestResponse] = []
        reader = threading.Thread(
            target=lambda: got.append(client.application.test_client().get(_thread_path(frame))),
        )
        with lock:
            reader.start()
            reader.join(timeout=_READ_TIMEOUT_S)
            reads.append((frame, got[0] if got else None))
            if surface in just_finished and waiting[surface]:
                just_finished.discard(surface)
                waiting[surface].pop(0).released.set()

    provider = ChannelProvider(
        chat=[
            calls(tool(FindToolsAbility.NAME, query=[MessageSubagentAbility.NAME])),
            *waiting["user"],
            text("The search found nothing either."),
        ],
        web_search=[Held(calls(tool(Recall.NAME, query="capital of France"))), *waiting[_CHILD_CHANNEL]],
    )
    _chat, frames = run_turn(provider, UserConfig(), "what is the capital of France?", refetch_on_poke)
    assert provider.unused() == {}
    return _RefetchRun(frames, reads, next_steps)


def test_every_finished_call_is_followed_by_updated_on_its_own_surface(refetch_run: _RefetchRun) -> None:
    """A call that ran to ``done``, one that failed while running, one rejected
    before it ran, a subagent's call that returned at once, and that subagent's
    child's own call each send ``updated`` to the surface showing them, as the
    very next frame that surface receives — before the model's next step on
    that surface has answered, not only once its row lands."""
    finished = refetch_run.finished

    assert sorted((_address(f)[0], f["tool_name"], f["state"]) for f in finished) == [
        (_CHILD_CHANNEL, Recall.NAME, "done"),
        ("user", ChalieDocsAbility.NAME, "error"),
        ("user", FindToolsAbility.NAME, "done"),
        ("user", MessageSubagentAbility.NAME, "error"),
        ("user", WebSearchAbility.NAME, "done"),
    ]
    for call in finished:
        poke = _next_on_address(refetch_run.frames, call)
        assert poke is not None and poke.get("status") == "updated", (call["tool_name"], poke)
    assert [step.in_time for step in refetch_run.next_steps] == [True] * 5


def test_the_refetch_updated_triggers_reads_the_call_finished(refetch_run: _RefetchRun) -> None:
    """The turn read back the moment ``updated`` is broadcast — on another
    connection — already shows that call in the state its frame announced."""
    for call in refetch_run.finished:
        poke = cast("Frame", _next_on_address(refetch_run.frames, call))
        response = refetch_run.read_at(poke)
        assert response is not None, f"the read for {call['tool_name']} never returned"
        assert response.status_code == 200, response.data
        block = cast("dict[str, object]", cast("dict[str, object]", response.get_json())["result"])
        assert chip(block, cast("str", call["tool_name"]))["state"] == call["state"]


def test_a_running_call_sends_no_updated(refetch_run: _RefetchRun) -> None:
    """From a call's ``started`` frame to its terminal one — the delegate's second
    ``started``, announcing its child, included — its surface receives no
    ``updated``: a running call is the live pill's to show."""
    windows: dict[object, list[Frame]] = {}
    for call in refetch_run.finished:
        on_address = [f for f in refetch_run.frames if _address(f) == _address(call)]
        start = next((i for i, f in enumerate(on_address) if f.get("tool_name") == call["tool_name"]), None)
        if start is not None and on_address[start]["state"] == "started":
            windows[call["tool_name"]] = on_address[start:on_address.index(call)]

    assert sorted(cast("list[str]", list(windows))) == sorted(
        [FindToolsAbility.NAME, MessageSubagentAbility.NAME, Recall.NAME, WebSearchAbility.NAME],
    )
    assert [f["state"] for f in windows[WebSearchAbility.NAME] if f.get("tool_name") == WebSearchAbility.NAME] == [
        "started", "started",
    ]
    for tool_name, window in windows.items():
        assert not [f for f in window if f.get("status") == "updated"], tool_name
