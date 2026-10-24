"""Feature test: a finished tool call makes its surface refetch the turn.

A turn's block carries every tool call from the moment it starts, so a refetch
taken while a call runs shows it in flight. The frontend only refetches on an
``updated`` signal; a tool frame alone just moves the live pill. Without a
signal after the call ends, a call that was running at the last refetch keeps
showing in flight under the reply until the next assistant row lands — minutes,
for a long delegate. Each finished call (done, failed or rejected before it
ran) must therefore be followed on the same address by ``updated``, and the
block that refetch reads must already show the call finished. A call that is
still running sends no ``updated``: the live pill shows it.

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
from abilities.recall import Recall
from abilities.web_search import WebSearchAbility
from models.provider_response import ProviderResponse
from tests.test_delegate_transcript_watch import (
    _CHILD_CHANNEL,
    _INSTRUCTIONS,
    _chip,
    _run,
    _ScriptedProvider,
    _tool,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

_READ_TIMEOUT_S = 30.0

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


@dataclass(frozen=True)
class _RefetchRun:
    """A finished user turn whose tool calls ended every way a call can end, the
    frames it broadcast, and the turn read back the moment each ``updated`` arrived."""

    frames: list[Frame]
    reads: list[tuple[Frame, TestResponse | None]]

    @property
    def finished(self) -> list[Frame]:
        return [f for f in self.frames if "tool_name" in f and f["state"] != "started"]

    def read_at(self, poke: Frame) -> TestResponse | None:
        return next(response for frame, response in self.reads if frame is poke)


@pytest.fixture
def refetch_run(authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> _RefetchRun:
    """A user turn that runs an ordinary tool to ``done``, sends a call its tool
    rejects before running it, then calls ``web_search`` — whose child turn runs
    ``recall`` to ``done`` but answers nothing, so the delegate call ends in
    ``error`` — and finally replies. Each call runs in its own model step, so no
    two calls on one surface are ever in flight together."""
    client, _db, _store = authed_client
    reads: list[tuple[Frame, TestResponse | None]] = []
    lock = threading.Lock()

    def refetch_on_poke(frame: Frame) -> None:
        if frame.get("status") != "updated" or _address(frame)[0] not in ("user", _CHILD_CHANNEL):
            return
        got: list[TestResponse] = []
        reader = threading.Thread(
            target=lambda: got.append(client.application.test_client().get(_thread_path(frame))),
        )
        with lock:
            reader.start()
            reader.join(timeout=_READ_TIMEOUT_S)
            reads.append((frame, got[0] if got else None))

    provider = _ScriptedProvider(
        ProviderResponse(text="", model="m", tool_calls=[_tool(FindToolsAbility.NAME, query=["weather"])]),
        ProviderResponse(text="", model="m", tool_calls=[_tool(ChalieDocsAbility.NAME)]),
        ProviderResponse(text="", model="m", tool_calls=[_tool(WebSearchAbility.NAME, instructions=_INSTRUCTIONS)]),
        ProviderResponse(text="", model="m", tool_calls=[_tool(Recall.NAME, query="capital of France")]),
        ProviderResponse(text="", model="m", tool_calls=None),
        ProviderResponse(text="I could not find it.", model="m", tool_calls=None),
    )
    _parent, frames = _run(provider, "what is the capital of France?", refetch_on_poke)
    return _RefetchRun(frames, reads)


def test_every_finished_call_is_followed_by_updated_on_its_own_surface(refetch_run: _RefetchRun) -> None:
    """A call that ran to ``done``, one that failed while running, one rejected
    before it ran, and a delegate child's own call each send ``updated`` to the
    surface showing them, as the very next frame that surface receives."""
    finished = refetch_run.finished

    assert sorted((_address(f)[0], f["tool_name"], f["state"]) for f in finished) == [
        (_CHILD_CHANNEL, Recall.NAME, "done"),
        ("user", ChalieDocsAbility.NAME, "error"),
        ("user", FindToolsAbility.NAME, "done"),
        ("user", WebSearchAbility.NAME, "error"),
    ]
    for call in finished:
        poke = _next_on_address(refetch_run.frames, call)
        assert poke is not None and poke.get("status") == "updated", (call["tool_name"], poke)


def test_the_refetch_updated_triggers_reads_the_call_finished(refetch_run: _RefetchRun) -> None:
    """The turn read back the moment ``updated`` is broadcast — on another
    connection — already shows that call in the state its frame announced."""
    for call in refetch_run.finished:
        poke = cast("Frame", _next_on_address(refetch_run.frames, call))
        response = refetch_run.read_at(poke)
        assert response is not None, f"the read for {call['tool_name']} never returned"
        assert response.status_code == 200, response.data
        block = cast("dict[str, object]", cast("dict[str, object]", response.get_json())["result"])
        assert _chip(block, cast("str", call["tool_name"]))["state"] == call["state"]


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

    assert sorted(cast("list[str]", list(windows))) == sorted([FindToolsAbility.NAME, Recall.NAME, WebSearchAbility.NAME])
    assert [f["state"] for f in windows[WebSearchAbility.NAME] if f.get("tool_name") == WebSearchAbility.NAME] == [
        "started", "started",
    ]
    for tool_name, window in windows.items():
        assert not [f for f in window if f.get("status") == "updated"], tool_name
