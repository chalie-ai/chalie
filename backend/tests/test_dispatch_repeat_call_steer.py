# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test: the repeat-call steer.

The runaway guard kills a turn at ``_RUNAWAY_TOOL_CALL_LIMIT`` (5) identical
calls — but the model never sees the steer that would break the loop before
that. This test verifies the new behaviour:

* The second identical call gets a steering suffix appended to the persisted
  ``tool_calls.result`` row, so the model re-reads it on its next step.
* The steer persists into the row (not just the return value), fixing the
  dead-letter defect where the ACT loop discarded the steer.
* The steer covers BOTH paths: bound-tool success/error AND unknown-tool.
* The runaway kill still works — five identical calls in one batch crash the
  turn before any dispatch, zero steer_probe rows.

Drives the real production entry point (construct inertly, ``begin()``,
``result()``) against the real SQLite DB, with the real ``DispatchService`` and
``ToolCallService``. The only substitution is the LLM network boundary:
``services.provider_service.build_client``. The fake client replays a scripted
sequence of provider responses, one per step. Each ``steer_probe`` test seeds a
CHAT-channel ``allow`` policy row first (see ``_allow_steer_probe``) — the
production grant path through the real gate.

The registered ability ``steer_probe`` is a real ``Ability`` subclass at module
level — ``DISCOVERABLE`` is its default (``True``), so it is permanently visible
to ``Ability.__subclasses__()`` for the rest of the process once collected. No
custom ``__init__``: ``DispatchService._bind()`` always builds a FRESH instance
via ``type(template)(mp=...)`` for the real dispatch-by-name seam. The ability
returns a deterministic ``ToolResult.ok("steer probe done")`` so the persisted
row carries the success envelope and the steer can be asserted on it."""

import sqlite3
import threading
import time
from unittest.mock import patch

import pytest

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.channels.user import UserConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import (
    _RUNAWAY_TOOL_CALL_LIMIT,
    MessageProcessor,
)
from models.policy import Policy
from models.provider_response import ProviderResponse
from models.turn_execution import TurnExecution
from services.dispatch_service import _REPEAT_CALL_STEER

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

# ProviderService builds its thin transport client via this factory call — the
# real network boundary (see test_message_processor_runaway_loop.py).
_BUILD_CLIENT = "services.provider_service.build_client"


def _tool(name: str, **params: object) -> dict[str, object]:
    """A provider-shaped tool call: ``{"name": ..., "input": {...}}``."""
    return {"name": name, "input": params}


class _ScriptedProvider:
    """Replays one scripted ``ProviderResponse`` per ``send()`` call, in order.

    Past the end of the script it returns a benign terminal (no-tool, empty)
    response rather than raising: a completed ``user`` turn spawns a
    fire-and-forget skill-suggestion turn (``_proactive_suggestion``) that makes
    its own provider call on a daemon thread this test does not script for. This
    tolerance cannot mask a steer failure — a steer that failed to append would
    instead leave the row bare, which every steer test asserts against."""

    _TERMINAL = ProviderResponse(text="", model="scripted-repeat-call", tool_calls=None)

    def __init__(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
        self.sends = 0

    def get_context_limit(self) -> int:
        return 200000

    def estimate_request_tokens(self, _dto: object) -> int:
        return 1

    def send(self, _dto: object) -> ProviderResponse:
        if self.sends >= len(self._responses):
            return self._TERMINAL
        response = self._responses[self.sends]
        self.sends += 1
        return response


def _drain_background_turns(timeout_s: float = 10.0) -> None:
    """Join the fire-and-forget post-turn daemon turns a completed ``user`` turn
    spawns (skill suggestion, thread gist) so they run to completion inside THIS
    test's provider+DB patch and never leak into the next test.

    ``_end`` starts them unjoined (``services.skill_suggestion_message_processor``
    / ``thread_gist_message_processor``): each is a whole ``MessageProcessor``
    turn on its own daemon (named ``skill-suggest``/``thread-gist`` and,
    transitively, a ``turn-*`` drive thread). Left running, such a turn would call
    ``build_client`` while the *next* test holds the patch, hitting that test's
    scripted provider and repatched DB — the exact cross-test corruption this
    guards against."""
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


def _run(provider: _ScriptedProvider, raw_input: str) -> MessageProcessor:
    """Drive a real user turn to termination against *provider*, then quiesce any
    post-turn daemon turns inside the patch so the test stays hermetic."""
    mp = MessageProcessor(UserConfig(), raw_input=raw_input)  # inert (I2)
    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()
        _drain_background_turns()
    return mp


def _allow_steer_probe() -> None:
    """Seed the CHAT-channel ``allow`` policy row for ``steer_probe``.

    A registered, non-INTERNAL ability dispatched on the CHAT channel otherwise
    lazily seeds an ``ask`` row (``Policy.get_or_create_default``) and parks the
    turn on the interactive permission gate (``PolicyManager._ask_user``),
    waiting on a human approval pytest never supplies — the turn thread never
    terminates and ``mp.result()`` joins it forever. Seeding ``allow`` is the
    production grant path, not a bypass: the gate still runs on every dispatch
    and reads this row. (``noop_probe`` needs no row — unknown tools never
    reach the gate.)"""
    Policy.upsert(PolicyChannel.CHAT.value, "steer_probe", "allow")


# ── The registered ability ──────────────────────────────────────────────────────


class steer_probe(Ability):
    """A real, DISCOVERABLE ability for the repeat-call steer tests.

    Returns ``ToolResult.ok("steer probe done")`` deterministically. No custom
    ``__init__``: ``DispatchService._bind()`` always builds a FRESH instance via
    ``type(template)(mp=...)`` for the real dispatch-by-name seam, so any state
    lives on the class (not needed here) rather than on a per-instance
    constructor the binder can no longer supply."""

    def get_name(self) -> str:
        return "steer_probe"

    def get_search_tooltip(self) -> str:
        return "repeat-call steer test fixture"

    def get_summary(self) -> str:
        return "Repeat-call steer test fixture — returns a fixed success."

    def get_examples(self) -> list[str]:
        return [
            "probe the steer",
            "test the repeat guard",
            "fire the steer probe",
            "run the steer probe",
            "hit the steer probe",
            "trigger the steer probe",
            "use the steer probe",
        ]

    def get_parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": [],
        }

    def run(self, params: dict[str, object]) -> ToolResult:
        return ToolResult.ok("steer probe done")


# ── Test: steer fires on the second identical call ─────────────────────────────


def test_second_identical_call_gets_steer(db: sqlite3.Connection) -> None:
    """Two sequential steps each calling ``steer_probe(q="x")`` once, then a
    terminal no-tool response: the first row's result does NOT contain
    ``[loop-guard]``; the second row's result ENDS WITH ``_REPEAT_CALL_STEER``;
    the turn completes normally (COMPLETED state).

    Proves the steer is persisted into the row (not just returned), that it
    fires on the SECOND identical call (threshold=2), and that steering below
    the kill threshold never crashes."""
    assert db is not None
    _allow_steer_probe()
    provider = _ScriptedProvider(
        ProviderResponse(text="", model="scripted", tool_calls=[_tool("steer_probe", q="x")]),
        ProviderResponse(text="", model="scripted", tool_calls=[_tool("steer_probe", q="x")]),
        ProviderResponse(text="All done.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "test the steer")

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED

    rows = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "steer_probe"]
    assert len(rows) == 2
    assert "[loop-guard]" not in rows[0].result
    assert rows[1].result.endswith(_REPEAT_CALL_STEER)


def test_four_identical_calls_across_steps_all_steer_below_kill(db: sqlite3.Connection) -> None:
    """Four identical calls across four steps then terminal: the first row is
    bare and every later row carries the steer, turn COMPLETED.

    Proves the steer fires on every call at or above the threshold (not just
    once), and that steering below the kill threshold never crashes — the turn
    completes normally with all four calls dispatched."""
    assert db is not None
    _allow_steer_probe()
    steps = [
        ProviderResponse(text="", model="scripted", tool_calls=[_tool("steer_probe", q="x")])
        for _ in range(4)
    ]
    steps.append(ProviderResponse(text="All done.", model="scripted", tool_calls=None))
    provider = _ScriptedProvider(*steps)

    mp = _run(provider, "test the steer")

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED

    rows = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "steer_probe"]
    assert len(rows) == 4
    # First call: tally=1, below threshold, no steer.
    assert "[loop-guard]" not in rows[0].result
    # Calls 2-4: tally>=2, steer appended.
    assert rows[1].result.endswith(_REPEAT_CALL_STEER)
    assert rows[2].result.endswith(_REPEAT_CALL_STEER)
    assert rows[3].result.endswith(_REPEAT_CALL_STEER)


# ── Test: five identical calls in one batch still crashes ──────────────────────


def test_five_identical_calls_in_one_batch_crashes(db: sqlite3.Connection) -> None:
    """Five identical calls in one batch: ``RunAwayLoop`` escalation intact —
    turn CRASHED, zero ``steer_probe`` rows (guard fires BEFORE dispatch).

    Same shape as ``test_identical_tool_call_at_limit_crashes_the_turn`` in
    ``test_message_processor_runaway_loop.py``. Proves the new steer does NOT
    interfere with the hard kill: five identical calls in one batch crash the
    turn before any of them dispatch, so zero steer_probe rows exist."""
    assert db is not None
    # Seeded even though the kill should fire BEFORE any dispatch: if the kill
    # ever regresses, the calls dispatch and the row assert fails loudly instead
    # of the turn parking forever on the ask gate.
    _allow_steer_probe()
    batch = [_tool("steer_probe", q="x") for _ in range(_RUNAWAY_TOOL_CALL_LIMIT)]
    provider = _ScriptedProvider(ProviderResponse(text="", model="scripted", tool_calls=batch))

    mp = _run(provider, "test the steer")

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.CRASHED
    assert execution.stop_reason is not None
    assert "identical parameters" in execution.stop_reason
    assert "steer_probe" in execution.stop_reason
    # Guard trips before dispatch: no steer_probe call executed.
    assert [c for c in mp.tool_call_service.by_turn() if c.tool_name == "steer_probe"] == []


# ── Test: different params do NOT steer ────────────────────────────────────────


def test_different_params_do_not_steer(db: sqlite3.Connection) -> None:
    """Two steps with DIFFERENT params (``q="a"`` then ``q="b"``) then terminal:
    NO row contains ``[loop-guard]``.

    Proves the steer keys on the FULL canonical params (not just the tool
    name), so a model varying a parameter is not steered away."""
    assert db is not None
    _allow_steer_probe()
    provider = _ScriptedProvider(
        ProviderResponse(text="", model="scripted", tool_calls=[_tool("steer_probe", q="a")]),
        ProviderResponse(text="", model="scripted", tool_calls=[_tool("steer_probe", q="b")]),
        ProviderResponse(text="All done.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "test the steer")

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED

    rows = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "steer_probe"]
    assert len(rows) == 2
    for row in rows:
        assert "[loop-guard]" not in row.result


# ── Test: unknown-tool path gets the call-steer ────────────────────────────────


def test_unknown_tool_gets_call_steer(db: sqlite3.Connection) -> None:
    """Two sequential steps calling an UNREGISTERED tool (``noop_probe``): the
    second row's persisted result ends with ``_REPEAT_CALL_STEER``.

    Proves the call-steer covers the unknown-tool path, which the error-steer's
    envelope check misses (unknown-tool returns plain text ``"Unknown tool: X"``,
    not the ``[<tool>(status=error`` envelope the error-steer requires). The
    runaway guard tallies unknown tools, so the call-steer fires on the second
    identical unknown call."""
    assert db is not None
    provider = _ScriptedProvider(
        ProviderResponse(text="", model="scripted", tool_calls=[_tool("noop_probe", q="loop")]),
        ProviderResponse(text="", model="scripted", tool_calls=[_tool("noop_probe", q="loop")]),
        ProviderResponse(text="All done.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "test the steer")

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED

    rows = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "noop_probe"]
    assert len(rows) == 2
    # First call: tally=1, below threshold, no steer.
    assert "[loop-guard]" not in rows[0].result
    # Second call: tally=2, steer appended.
    assert rows[1].result.endswith(_REPEAT_CALL_STEER)
