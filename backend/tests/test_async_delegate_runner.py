# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for AsyncDelegateRunner — the Processes-panel delegate lifecycle.

Drives the runner's real daemon-thread path with no mocks: spawn returns the
placeholder immediately (non-blocking), registers a rich snapshot (tool name,
the model's act-summary, start time, cancel event), runs the real ability
through the shared sync-run primitive, and deregisters on completion. Both
lifecycle events are observed through the REAL WebSocketBroker fan-out via an
in-process client implementing the broker's send Protocol — the same seam the
WS route uses — so a dropped event or a missing payload key fails the test loud.

cancel() sets the cooperative cancel event, which also makes ``_run`` skip the
captured-mp delivery — so the happy lifecycle is provable without firing the
real-LLM synthesis turn (that is QA-env / end-to-end territory).
"""

import json
import threading
import time
from typing import cast

import pytest

from abilities._ability import Ability
from abilities._result import ToolResult
from services.async_delegate_runner import AsyncDelegateRunner
from services.time_utils import parse_utc
from services.websocket_broker import WebSocketBroker

pytestmark = pytest.mark.unit


class _RecordingClient:
    """A real WebSocketBroker subscriber: implements the broker's send(str)
    Protocol and captures every broadcast frame as the inspectable surface the
    Processes panel renders from."""

    def __init__(self) -> None:
        self.frames: list[dict[str, object]] = []

    def send(self, data: str) -> None:
        self.frames.append(json.loads(data))

    def frames_for(self, event_type: str, sub_id: str) -> list[dict[str, object]]:
        return [f for f in self.frames if f.get("type") == event_type and f.get("sub_id") == sub_id]


class _GatedAbility(Ability):
    _SYNTHETIC = True

    def get_name(self) -> str: return "test_runner_gated"
    def get_search_tooltip(self) -> str: return "block until released — runner feature-test fixture"
    def get_summary(self) -> str: return "Block until released — feature-test fixture only."
    def get_examples(self) -> list[str]: return [
        "block until released",
        "wait for the gate",
        "hold until signalled",
        "pause for the test",
        "stay blocked",
        "wait then return",
    ]
    def get_parameters(self) -> dict[str, object]: return {"type": "object", "properties": {}, "required": []}

    def __init__(self, release: threading.Event, started: threading.Event, done: list[bool]) -> None:
        self._release = release
        self._started = started
        self._done = done

    def run(self, params: dict[str, object]) -> ToolResult:
        self._started.set()
        self._release.wait(timeout=5)
        self._done.append(True)
        return cast(ToolResult, "gated done")


def test_spawn_pushes_rich_snapshot_then_emits_end_on_deregister() -> None:
    release = threading.Event()
    started = threading.Event()
    done: list[bool] = []
    ability = _GatedAbility(release, started, done)
    ability.mp = object()  # delivery is skipped (cancelled below) — never reaches a real LLM.

    client = _RecordingClient()
    WebSocketBroker().connect(client)
    runner = AsyncDelegateRunner()
    try:
        placeholder = runner.spawn(ability, {}, ability.mp, "Drafting the weekly digest")

        # Non-blocking: the placeholder returns while run() is still gated.
        assert placeholder.startswith("test_runner_gated dispatched (id: ")
        assert started.wait(timeout=2), "background run() never started"
        assert not done, "spawn blocked until run() finished — not backgrounded"

        # The panel's snapshot carries everything a row renders.
        rows = runner.active()
        assert len(rows) == 1
        row = rows[0]
        delegate_id = cast(str, row["sub_id"])
        assert delegate_id.startswith("test_runner_gated_")
        assert row["tool_name"] == "test_runner_gated"
        assert row["summary"] == "Drafting the weekly digest"
        assert parse_utc(cast(str, row["started_at"])).year >= 2025  # real timestamp, not datetime.min

        # The same data was pushed live to the connected client as subagent_start.
        starts = client.frames_for("subagent_start", delegate_id)
        assert len(starts) == 1
        assert starts[0]["tool_name"] == "test_runner_gated"
        assert starts[0]["summary"] == "Drafting the weekly digest"
        assert starts[0]["started_at"] == row["started_at"]  # snapshot == pushed frame

        # Stop control: flip the cancel event → _run skips delivery (no real LLM),
        # finishes run(), and deregisters + emits subagent_end in `finally`.
        assert runner.cancel(delegate_id) is True
        release.set()
        for _ in range(100):
            if not runner.active():
                break
            time.sleep(0.05)
        assert runner.active() == []
        assert done == [True]

        ends = client.frames_for("subagent_end", delegate_id)
        assert len(ends) == 1
    finally:
        WebSocketBroker().disconnect(client)


def test_cancel_unknown_id_returns_false() -> None:
    runner = AsyncDelegateRunner()
    assert runner.cancel("nope_12345678") is False
    assert runner.active() == []
