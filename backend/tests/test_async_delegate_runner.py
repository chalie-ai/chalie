# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for AsyncDelegateRunner (spec §4.4 / §4.0) — P6.

Exercises the runner's real daemon-thread lifecycle with no mocks: spawn returns
the placeholder immediately (non-blocking), registers the delegate, runs the
real ability through the shared sync-run primitive, and deregisters on
completion. cancel() sets the cooperative cancel event, which also makes the
``_run`` body skip the captured-mp delivery — so the happy lifecycle is provable
without firing the real-LLM synthesis turn (that is QA-env / nightly territory).
"""

import threading
import time

import pytest

from abilities._ability import Ability
from services.async_delegate_runner import AsyncDelegateRunner

pytestmark = pytest.mark.unit


class _GatedAbility(Ability):
    """Real Ability whose run() blocks on a caller-controlled event. _SYNTHETIC
    keeps it out of the static registry (matches dynamic/MCP abilities)."""

    _SYNTHETIC = True
    NAME = "test_runner_gated"
    SEARCH_TOOLTIP = "block until released — runner feature-test fixture"
    SUMMARY = "Block until released — feature-test fixture only."
    EXAMPLES = [
        "block until released",
        "wait for the gate",
        "hold until signalled",
        "pause for the test",
        "stay blocked",
        "wait then return",
    ]
    INPUT_SCHEMA = {"type": "object", "properties": {}, "required": []}

    def __init__(self, release: threading.Event, started: threading.Event, done: list):
        self._release = release
        self._started = started
        self._done = done

    def run(self, params: dict) -> str:
        self._started.set()
        self._release.wait(timeout=5)
        self._done.append(True)
        return "gated done"


def test_spawn_is_non_blocking_registers_and_deregisters():
    release = threading.Event()
    started = threading.Event()
    done = []
    ability = _GatedAbility(release, started, done)
    ability.MessageProcessor = object()  # delivery is skipped (cancelled below)

    runner = AsyncDelegateRunner()
    placeholder = runner.spawn(ability, {}, mp=ability.MessageProcessor)

    # Returns immediately with the placeholder while run() is still blocked.
    assert placeholder.startswith("test_runner_gated dispatched (id: ")
    assert started.wait(timeout=2), "background run() never started"
    assert not done, "spawn blocked until run() finished — not backgrounded"

    ids = runner.active_ids()
    assert len(ids) == 1
    delegate_id = ids[0]
    assert delegate_id.startswith("test_runner_gated_")

    # Cancel before releasing → _run skips delivery (no real LLM), then the
    # daemon completes run() and deregisters in `finally`.
    assert runner.cancel(delegate_id) is True
    release.set()

    for _ in range(100):
        if not runner.active_ids():
            break
        time.sleep(0.05)
    assert runner.active_ids() == []
    assert done == [True]


def test_cancel_unknown_id_returns_false():
    runner = AsyncDelegateRunner()
    assert runner.cancel("nope_12345678") is False
    assert runner.active_ids() == []
