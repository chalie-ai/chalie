# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test: the empty-completion guard.

A completion with no tool calls and no text, on a turn that has run no tools at
all, means the model did nothing whatsoever. Settling it renders silence as an
answered turn (an empty assistant row stored, turn COMPLETED, the only tool row
the pre-turn auto recall seed). The guard steers a bounded retry —
the steer text enters the NEXT request body, since without it the re-send would
be byte-identical — and past ``_EMPTY_COMPLETION_STEER_LIMIT`` steers trips a
loud ``EmptyCompletionLoop`` (CRASHED turn, same path as ``RunAwayLoop``). A
turn that DID run tools may still finish silently — background channels end
that way by design.

Same harness as ``test_message_processor_runaway_loop.py``: the real production
entry point (inert construct, ``begin()``, ``result()``) against the real DB and
services; the only substitution is the LLM network boundary via
``services.provider_service.build_client``. The scripted client here also
records every request so steer injection is asserted on the actual bodies sent.
"""

import sqlite3
import threading
import time
from typing import cast
from unittest.mock import patch

import pytest

from configs.channels.user import UserConfig
from controllers.message_processor import (
    _EMPTY_COMPLETION_STEER,
    _EMPTY_COMPLETION_STEER_LIMIT,
    MessageProcessor,
)
from models.provider_response import ProviderResponse
from models.transcript import Transcript
from models.turn_execution import TurnExecution

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

_BUILD_CLIENT = "services.provider_service.Factory.build_client"


class _ScriptedProvider:
    """Replays one scripted ``ProviderResponse`` per ``send()``, in order, and
    records every request DTO. Past the end of the script it returns an empty
    terminal — under this guard a zero-tool turn hitting it repeatedly CRASHES
    (never hangs), so the overflow tolerance for unscripted post-turn daemon
    turns still terminates them."""

    _TERMINAL = ProviderResponse(text="", model="scripted-overflow", tool_calls=None)

    def __init__(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
        self.requests: list[object] = []

    def get_context_limit(self) -> int:
        return 200000

    def send(self, dto: object) -> ProviderResponse:
        self.requests.append(dto)
        if len(self.requests) > len(self._responses):
            return self._TERMINAL
        return self._responses[len(self.requests) - 1]


def _drain_background_turns(timeout_s: float = 10.0) -> None:
    """Join the fire-and-forget post-turn daemon turns a completed ``user`` turn
    spawns, so they run to completion inside THIS test's provider+DB patch (see
    ``test_message_processor_runaway_loop.py`` for the full rationale)."""
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
    mp = MessageProcessor(UserConfig(), raw_input=raw_input)  # inert (I2)
    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()
        _drain_background_turns()
    return mp


def _assistant_rows(mp: MessageProcessor) -> list[dict[str, object]]:
    return [r for r in Transcript.by_turn(mp.channel, mp.turn_id) if r["role"] == "assistant"]


def _body(provider: _ScriptedProvider, i: int) -> str:
    """The user-message content of the i-th request actually sent. Requests are
    appended in send order, and the asserted turn's sends all precede any
    post-turn daemon turn's (those spawn inside the terminal step)."""
    request = provider.requests[i]
    messages = cast("list[dict[str, object]]", getattr(request, "messages"))
    return cast("str", messages[0]["content"])


def _tool(name: str, **params: object) -> dict[str, object]:
    return {"name": name, "input": params}


# ── Steered recovery ──────────────────────────────────────────────────────────


def test_empty_completion_is_steered_then_settles(db: sqlite3.Connection) -> None:
    """The observed defect shape, recovered: the first completion is completely empty
    (no tool calls, no text), the guard steers instead of settling, and the
    model's second response answers. The turn COMPLETES with the real answer;
    the steer text entered exactly the second request body."""
    assert db is not None
    provider = _ScriptedProvider(
        ProviderResponse(text="", model="scripted", tool_calls=None),
        ProviderResponse(text="Here is my answer.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "tell me about the docs")

    assert mp.result() == "Here is my answer."
    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED
    # Exactly one assistant row — the answer; the steered empty step stored nothing.
    rows = _assistant_rows(mp)
    assert len(rows) == 1
    assert "Here is my answer." in cast("str", rows[0]["content"])
    # The steer entered the retry request and ONLY the retry request.
    assert _EMPTY_COMPLETION_STEER not in _body(provider, 0)
    assert _EMPTY_COMPLETION_STEER in _body(provider, 1)


def test_whitespace_only_completion_counts_as_empty(db: sqlite3.Connection) -> None:
    """Whitespace-only text is no answer: it steers exactly like the empty
    completion instead of being stored as the turn's prose."""
    assert db is not None
    provider = _ScriptedProvider(
        ProviderResponse(text="  \n\t ", model="scripted", tool_calls=None),
        ProviderResponse(text="Real answer.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "hello?")

    assert mp.result() == "Real answer."
    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED
    assert _EMPTY_COMPLETION_STEER in _body(provider, 1)


# ── Bounded: persistent emptiness crashes ─────────────────────────────────────


def test_persistent_empty_completions_crash_the_turn(db: sqlite3.Connection) -> None:
    """A model that stays empty past the steer limit trips ``EmptyCompletionLoop``:
    the turn ends CRASHED with the reason naming the empty completions, no
    assistant row is ever written, and the send count is exactly
    ``1 + _EMPTY_COMPLETION_STEER_LIMIT`` — the guard is bounded, never an
    infinite re-send loop."""
    assert db is not None
    provider = _ScriptedProvider()  # every send returns the empty terminal

    mp = _run(provider, "say something")

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.CRASHED
    assert execution.stop_reason is not None
    assert "empty completions" in execution.stop_reason
    assert _assistant_rows(mp) == []
    # First send unsteered, every steered retry after it, then the raise — a
    # crashed turn spawns no post-turn daemons, so these are ALL the sends.
    assert len(provider.requests) == 1 + _EMPTY_COMPLETION_STEER_LIMIT
    assert _EMPTY_COMPLETION_STEER not in _body(provider, 0)
    for i in range(1, len(provider.requests)):
        assert _EMPTY_COMPLETION_STEER in _body(provider, i)


# ── Silent finish after real tool work stays legitimate ───────────────────────


def test_silent_finish_after_tool_work_settles(db: sqlite3.Connection) -> None:
    """An empty terminal completion AFTER a tool-bearing step settles exactly as
    before — background channels end silently by design once their work ran. The
    guard keys on the turn-wide tool tally, not on the terminal step alone."""
    assert db is not None
    provider = _ScriptedProvider(
        ProviderResponse(text="", model="scripted", tool_calls=[_tool("noop_probe", q="x")]),
        ProviderResponse(text="", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "run the probe")

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED
    # The silent settle stored the (empty) assistant row, as today.
    assert len(_assistant_rows(mp)) == 1
    # The main turn was never steered: neither of its two requests carries the
    # steer text (post-turn daemon turns may be steered — different turns).
    assert _EMPTY_COMPLETION_STEER not in _body(provider, 0)
    assert _EMPTY_COMPLETION_STEER not in _body(provider, 1)
