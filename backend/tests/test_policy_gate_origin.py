"""Feature test: a gated ``ask`` asks the user, carrying the turn it belongs to.

Real path, zero mocks of the units under test. A scripted provider asks for a
real, registered probe ability whose CHAT-channel policy row is ``ask``; the real
``DispatchService`` → ``PolicyManager`` gate opens a permission ask through the
real ``AskUser`` service, which broadcasts an ``ask_opened`` frame through the
real socket registry to a recording subscriber — exactly what the interface
receives. The test answers the ask through ``AskUser.reply`` — the call the
reply route makes — and proves:

  * a user turn's ask carries that turn as its ``origin`` and the call's
    ``act_summary`` as its ``summary`` — the wire contract — and an
    ``ask_closed`` frame follows once it is answered;
  * a turn with no origin (discovery) still asks — the ask's origin is null —
    and a denial comes back AS the tool result, audited as ``user_denied``;
  * a nested delegate turn handed ``metadata={"origin": …}`` asks on the
    CALLER's origin, not on its own turn.
"""
from __future__ import annotations

import json
import queue
import sqlite3
import threading
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import cast
from unittest.mock import patch

import pytest

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.channels.discovery import DiscoveryConfig
from configs.channels.user import UserConfig
from configs.enums.ability_category import AbilityCategory
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from models.policy import Policy
from models.provider_response import ProviderResponse
from models.turn_execution import TurnExecution
from services.ask_user import AskUser
from services.websocket import Websocket
from tests.helpers import NO_PLAN, drain_background_turns, is_planning_request

pytestmark = pytest.mark.unit

_BUILD_CLIENT = "services.provider_service.build_client"
_SUMMARY = "Probe the gate"


# ── Socket subscriber ──────────────────────────────────────────────────────────


class _RecordingClient:
    """A real socket-registry subscriber: implements the registry's ``send(str)``
    protocol and keeps every frame, so the test asserts on what the interface
    would have been handed."""

    def __init__(self) -> None:
        self.frames: list[dict[str, object]] = []
        self.opened: queue.Queue[str] = queue.Queue()

    def close(self) -> None:
        """Nothing to release: the captured frames stay readable after the registry lets go."""

    def send(self, data: str) -> None:
        frame = json.loads(data)
        self.frames.append(frame)
        if frame["type"] == "ask_opened":
            self.opened.put(cast(str, frame["ask_id"]))

    def of_type(self, kind: str) -> list[dict[str, object]]:
        return [f for f in self.frames if f.get("type") == kind]


@pytest.fixture()
def recorder() -> Iterator[_RecordingClient]:
    client = _RecordingClient()
    Websocket._connect(client)
    try:
        yield client
    finally:
        Websocket._disconnect(client)


# ── Provider + turn helpers ────────────────────────────────────────────────────


def _tool(name: str, **params: object) -> dict[str, object]:
    return {"name": name, "input": params}


class _ScriptedProvider:
    """Replays one ``ProviderResponse`` per ``send()``; past the script it answers
    a benign terminal so the post-turn daemon turns (skill suggestion, thread
    gist) this test does not script for still terminate."""

    _TERMINAL = ProviderResponse(text="", model="scripted-gate-origin", tool_calls=None)

    def __init__(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
        self.sends = 0

    def get_context_limit(self) -> int:
        return 200000

    def send(self, dto: object) -> ProviderResponse:
        if is_planning_request(dto):
            return NO_PLAN
        if self.sends >= len(self._responses):
            return self._TERMINAL
        response = self._responses[self.sends]
        self.sends += 1
        return response


def _probe_call() -> _ScriptedProvider:
    return _ScriptedProvider(
        ProviderResponse(text="", model="scripted", tool_calls=[_tool("gate_probe", q="x", act_summary=_SUMMARY)]),
        ProviderResponse(text="All done.", model="scripted", tool_calls=None),
    )


def _run(mp: MessageProcessor, provider: _ScriptedProvider) -> None:
    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()
        drain_background_turns()


class _Answerer:
    """Answers the turn's ask through ``AskUser.reply`` — what the reply route
    calls — once its ``ask_opened`` frame lands. If no frame ever arrives it
    answers whatever ask is open, so a broken broadcast fails the assertions
    instead of leaving the turn waiting forever."""

    def __init__(self, recorder: _RecordingClient, allow: bool, timeout_s: float = 10.0) -> None:
        self._recorder = recorder
        self._allow = allow
        self._timeout_s = timeout_s
        self._thread = threading.Thread(target=self._answer, name="ask-answerer", daemon=True)
        self._thread.start()

    def _answer(self) -> None:
        try:
            ask_id = self._recorder.opened.get(timeout=self._timeout_s)
        except queue.Empty:
            still_open = AskUser.open_asks()
            if not still_open:
                return                               # nothing asked: the turn ran on, and the assertions fail
            ask_id = still_open[0].ask_id
        AskUser.reply(ask_id, self._allow)

    def join(self) -> None:
        self._thread.join(timeout=10.0)


# ── The registered ability ─────────────────────────────────────────────────────


class gate_probe(Ability):
    """A real, DISCOVERABLE ability for the gate-origin tests: dispatched by name
    through the real binder, so the real policy gate runs in front of it."""

    NAME = "gate_probe"
    CATEGORY = AbilityCategory.SYSTEM

    def get_search_tooltip(self) -> str:
        return "permission-gate origin test fixture"

    def get_summary(self) -> str:
        return "Permission-gate origin test fixture — returns a fixed success."

    def get_examples(self) -> list[str]:
        return [
            "probe the gate",
            "test the permission gate",
            "fire the gate probe",
            "run the gate probe",
            "hit the gate probe",
            "trigger the gate probe",
            "use the gate probe",
        ]

    def get_parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": [],
        }

    def run(self, params: dict[str, object]) -> ToolResult:
        return ToolResult.ok("gate probe done")


def _gate_probe_asks() -> None:
    Policy.upsert(PolicyChannel.CHAT.value, "gate_probe", "ask")


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_user_turn_ask_prompts_on_that_turn_with_summary(
    chat_provider: sqlite3.Connection, recorder: _RecordingClient,
) -> None:
    """A main-spine user turn: the ask's origin IS this turn, its summary IS
    the call's act_summary; allowing it runs the tool; the ask is closed."""
    _gate_probe_asks()
    mp = MessageProcessor(UserConfig(), raw_input="probe the gate")  # inert until begin()
    answerer = _Answerer(recorder, allow=True)
    _run(mp, _probe_call())
    answerer.join()

    asks = recorder.of_type("ask_opened")
    assert len(asks) == 1, recorder.frames
    ask_id = asks[0]["ask_id"]
    # The ask instant rides on the frame as ISO-8601 UTC — it must parse back
    # to an aware UTC datetime so a reloaded client can age the live line.
    asked_at = datetime.fromisoformat(cast(str, asks[0]["asked_at"]))
    assert asked_at.tzinfo is not None and asked_at.utcoffset() == timedelta(0)
    assert asks[0] == {
        "type": "ask_opened",
        "ask_id": ask_id,
        "kind": "permission",
        "data": {
            "action_id": "gate_probe",
            "summary": _SUMMARY,
            "origin": {"type": "user", "turn_id": mp.turn_id, "forked": False},
        },
        "asked_at": asked_at.isoformat(timespec="seconds"),
    }
    assert recorder.of_type("ask_closed") == [{"type": "ask_closed", "ask_id": ask_id}]
    assert AskUser.open_asks() == []

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED
    rows = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "gate_probe"]
    assert len(rows) == 1
    assert "gate probe done" in rows[0].result
    assert chat_provider.execute("SELECT count(*) FROM policy_blocked_log").fetchone()[0] == 0


def test_turn_with_no_origin_still_asks_and_a_denial_blocks_the_tool(
    chat_provider: sqlite3.Connection, recorder: _RecordingClient,
) -> None:
    """A discovery turn belongs to no conversation, yet the user is still asked —
    the ask's origin is null. Denying it makes the block message the tool's
    result, audits it as user_denied, and the turn still completes."""
    _gate_probe_asks()
    mp = MessageProcessor(DiscoveryConfig(), raw_input="probe the gate")
    answerer = _Answerer(recorder, allow=False)
    _run(mp, _probe_call())
    answerer.join()

    asks = recorder.of_type("ask_opened")
    assert len(asks) == 1, recorder.frames
    assert asks[0]["data"] == {"action_id": "gate_probe", "summary": _SUMMARY, "origin": None}
    assert recorder.of_type("ask_closed") == [{"type": "ask_closed", "ask_id": asks[0]["ask_id"]}]
    rows = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "gate_probe"]
    assert len(rows) == 1
    assert rows[0].result == "The gate_probe action is not allowed. Do NOT retry."
    blocked = chat_provider.execute("SELECT action_id, context, reason FROM policy_blocked_log").fetchall()
    assert [tuple(r) for r in blocked] == [("gate_probe", "chat", "user_denied")]
    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED


def test_nested_turn_inherits_callers_origin(
    chat_provider: sqlite3.Connection, recorder: _RecordingClient,
) -> None:
    """A delegate turn spawned with the caller's origin in its metadata asks on
    the CALLER's behalf — the ask carries that origin, not the delegate's own
    turn."""
    _gate_probe_asks()
    callers_origin: dict[str, object] = {"type": "scheduled", "turn_id": 42, "forked": True}
    mp = MessageProcessor(DiscoveryConfig(), raw_input="probe the gate", metadata={"origin": callers_origin})
    answerer = _Answerer(recorder, allow=True)
    _run(mp, _probe_call())
    answerer.join()

    asks = recorder.of_type("ask_opened")
    assert len(asks) == 1, recorder.frames
    data = cast(dict[str, object], asks[0]["data"])
    assert data["origin"] == callers_origin
    assert data["summary"] == _SUMMARY
    assert recorder.of_type("ask_closed") == [{"type": "ask_closed", "ask_id": asks[0]["ask_id"]}]
    rows = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "gate_probe"]
    assert len(rows) == 1
    assert "gate probe done" in rows[0].result
