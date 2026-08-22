"""Feature test: a gated ``ask`` prompts on the turn's surface — or denies outright.

Real path, zero mocks of the units under test. A scripted provider asks for a
real, registered probe ability whose CHAT-channel policy row is ``ask``; the real
``DispatchService`` → ``PolicyManager`` gate parks the turn and broadcasts a
``permission_request`` frame through the real socket registry to a recording
subscriber — exactly what the interface receives. The test answers the gate the
way ``POST /api/policies/respond`` does (result + event) and proves:

  * a user turn's frame carries that turn as its ``origin`` and the call's
    ``act_summary`` as its ``summary`` — the frozen wire contract — and a
    ``permission_resolved`` frame follows once the gate is answered;
  * a turn with no surface (discovery: ``BROADCASTS_STATE`` False) gets the
    steer sentence back AS the tool result, nothing is broadcast, and the block
    is audited as ``user_unavailable``;
  * a nested delegate turn handed ``metadata={"origin": …}`` prompts on the
    CALLER's origin, not on its own surfaceless turn.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
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
from services.policy_manager import _ASK_NO_SURFACE, _permission_gates
from services.websocket import Websocket

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

    def send(self, data: str) -> None:
        self.frames.append(json.loads(data))

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

    def send(self, _dto: object) -> ProviderResponse:
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


def _drain_background_turns(timeout_s: float = 10.0) -> None:
    """Join the fire-and-forget post-turn daemon turns so they finish inside THIS
    test's provider + DB patch and never leak into the next test."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pending = [
            t for t in threading.enumerate()
            if t.name in ("skill-suggest", "thread-gist") or t.name.startswith("turn-")
        ]
        if not pending:
            return
        for t in pending:
            t.join(timeout=deadline - time.monotonic())


def _run(mp: MessageProcessor, provider: _ScriptedProvider) -> None:
    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()
        _drain_background_turns()


class _Answerer:
    """Answers the parked gate the way ``POST /api/policies/respond`` does — sets
    the verdict and the event — once the ``permission_request`` frame lands.
    Falls back to whatever gate is parked if no frame ever arrives, so a broken
    broadcast fails the assertions instead of hanging the turn forever."""

    def __init__(self, recorder: _RecordingClient, verdict: str, timeout_s: float = 10.0) -> None:
        self._recorder = recorder
        self._verdict = verdict
        self._deadline = time.monotonic() + timeout_s
        self.pending_while_parked: list[dict[str, object]] | None = None
        self._thread = threading.Thread(target=self._answer, name="gate-answerer", daemon=True)
        self._thread.start()

    def _answer(self) -> None:
        while time.monotonic() < self._deadline:
            frames = self._recorder.of_type("permission_request")
            rid = cast(str, frames[0]["request_id"]) if frames else next(iter(_permission_gates), None)
            if rid is not None and rid in _permission_gates:
                gate = _permission_gates[rid]
                gate["result"] = self._verdict
                cast(threading.Event, gate["event"]).set()
                return
            time.sleep(0.02)

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
    """A main-spine user turn: the frame's origin IS this turn, its summary IS
    the call's act_summary; approval runs the tool; the card is resolved."""
    _gate_probe_asks()
    mp = MessageProcessor(UserConfig(), raw_input="probe the gate")  # inert until begin()
    answerer = _Answerer(recorder, "approved")
    _run(mp, _probe_call())
    answerer.join()

    requests = recorder.of_type("permission_request")
    assert len(requests) == 1, recorder.frames
    rid = requests[0]["request_id"]
    assert requests[0] == {
        "type": "permission_request",
        "request_id": rid,
        "action_id": "gate_probe",
        "summary": _SUMMARY,
        "origin": {"type": "user", "turn_id": mp.turn_id, "forked": False},
    }
    assert recorder.of_type("permission_resolved") == [{"type": "permission_resolved", "request_id": rid}]
    assert rid not in _permission_gates

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED
    rows = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "gate_probe"]
    assert len(rows) == 1
    assert "gate probe done" in rows[0].result
    assert chat_provider.execute("SELECT count(*) FROM policy_blocked_log").fetchone()[0] == 0


def test_surfaceless_turn_ask_is_denied_with_steer_and_never_broadcasts(
    chat_provider: sqlite3.Connection, recorder: _RecordingClient,
) -> None:
    """A discovery turn has no surface a human could answer from: the tool result
    IS the steer sentence, nothing goes over the socket, the block is audited
    as user_unavailable, and the turn still completes."""
    _gate_probe_asks()
    mp = MessageProcessor(DiscoveryConfig(), raw_input="probe the gate")
    _run(mp, _probe_call())

    assert recorder.of_type("permission_request") == []
    assert recorder.of_type("permission_resolved") == []
    rows = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "gate_probe"]
    assert len(rows) == 1
    assert rows[0].result == _ASK_NO_SURFACE.format(permission="gate_probe", channel="chat")
    blocked = chat_provider.execute("SELECT action_id, context, reason FROM policy_blocked_log").fetchall()
    assert [tuple(r) for r in blocked] == [("gate_probe", "chat", "user_unavailable")]
    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED


def test_nested_turn_inherits_callers_origin(
    chat_provider: sqlite3.Connection, recorder: _RecordingClient,
) -> None:
    """A delegate turn spawned with the caller's origin in its metadata prompts
    on the CALLER's surface — the frame carries that origin, not the delegate's
    own (surfaceless) turn."""
    _gate_probe_asks()
    callers_origin: dict[str, object] = {"type": "scheduled", "turn_id": 42, "forked": True}
    mp = MessageProcessor(DiscoveryConfig(), raw_input="probe the gate", metadata={"origin": callers_origin})
    answerer = _Answerer(recorder, "approved")
    _run(mp, _probe_call())
    answerer.join()

    requests = recorder.of_type("permission_request")
    assert len(requests) == 1, recorder.frames
    assert requests[0]["origin"] == callers_origin
    assert requests[0]["summary"] == _SUMMARY
    assert recorder.of_type("permission_resolved") == [
        {"type": "permission_resolved", "request_id": requests[0]["request_id"]},
    ]
    rows = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "gate_probe"]
    assert len(rows) == 1
    assert "gate probe done" in rows[0].result
