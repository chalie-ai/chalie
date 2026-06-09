# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for per-call async (spec §4.0 / §4.1 / §4.8d — P4).

Real hot path, zero mocks:

  * The framework ``async`` boolean is exposed on a tool's schema ONLY on a
    ``SUPPORTS_ASYNC`` channel (UserConfig) — never on a delegate/system
    channel (DmnConfig) and never when there is no live processor.  This is the
    schema-exposure gate, not a routing gate (§4.8d).
  * A real ability dispatched WITHOUT ``async`` runs inline through the real
    ``ToolDispatcher._execute`` → ``_run`` → ``run`` → ``_normalise`` chain
    and returns its actual result.
  * The same ability dispatched WITH ``async: true`` returns the
    ``dispatched (id: …)`` placeholder immediately (non-blocking — the real
    ``run`` is still in flight) and registers exactly one active delegate.

The async LLM follow-up turn delivery (the captured-mp synthesis on the user
channel) is exercised end-to-end in the QA-env / nightly run, not here — it
requires a real model.  This test proves the *decision* and the *non-blocking*
contract without firing the LLM by cancelling the delegate before releasing it,
which makes ``AsyncDelegateRunner._run`` skip delivery (§4.4).
"""

import threading
import time

import pytest

from abilities._ability import Ability
from abilities._dispatcher import ToolDispatcher
from abilities._registry import AbilityRegistry
from configs.channels import DmnConfig, UserConfig
from services.async_delegate_runner import async_delegate_runner

pytestmark = pytest.mark.unit


class _Ctx:
    """Minimal real MP-shaped context — exactly what execute / get_input_schema
    read off the live processor: ``config`` (for the SUPPORTS_ASYNC gate + the
    emitter). The MessageProcessor is constructor-injected onto the ability as
    ``self.mp``."""

    def __init__(self, config):
        self.config = config


def test_async_property_exposed_only_on_supports_async_channel():
    """§4.8d — the schema-exposure gate keys off config.SUPPORTS_ASYNC
    alone, applied at the single ``get_input_schema()`` assembler. The processor
    is constructor-injected, so a fresh mp-bound instance is built per channel."""
    weather_cls = type(AbilityRegistry.get("weather"))

    user_schema = weather_cls(mp=_Ctx(UserConfig({}))).get_input_schema()["input_schema"]
    assert "async" in user_schema["properties"]
    assert user_schema["properties"]["async"]["type"] == "boolean"
    assert user_schema["properties"]["async"]["default"] is False
    # The declared params (get_parameters) are never mutated — exposure deepcopies.
    assert "async" not in weather_cls().get_parameters()["properties"]

    # A non-async channel (delegate/system) never sees the flag.
    dmn_schema = weather_cls(mp=_Ctx(DmnConfig())).get_input_schema()["input_schema"]
    assert "async" not in dmn_schema.get("properties", {})

    # No live processor (mp=None) → synchronous default, no flag.
    assert "async" not in weather_cls().get_input_schema()["input_schema"].get("properties", {})


def test_build_tools_exposes_async_on_user_channel():
    """The real tool-presentation path (build_tools → mp-bound get_input_schema)
    surfaces the async flag alongside act_summary."""

    class _Proc:
        config = UserConfig({})
        active_tools = ["weather"]

    tools = AbilityRegistry.build_tools(_Proc())
    weather_tool = next(t for t in tools if t["name"] == "weather")
    props = weather_tool["input_schema"]["properties"]
    assert "async" in props
    assert "act_summary" in props


class _EchoAbility(Ability):
    """A real concrete Ability used to drive the actual execute() chain.

    ``_SYNTHETIC`` is the production escape hatch (used by dynamic/MCP abilities)
    that keeps a concrete Ability out of the static file registry — here it
    stops this fixture leaking into ``AbilityRegistry`` and skewing other tests.
    It does not alter the execute/run/get_input_schema paths under test.
    """

    _SYNTHETIC = True

    def __init__(self, mp=None):
        super().__init__(mp)
        self.ran_with = None

    def get_name(self) -> str:
        return "test_per_call_echo"

    def get_search_tooltip(self) -> str:
        return "echo back the input for the per-call-async feature test"

    def get_summary(self) -> str:
        return "Echo the given text straight back — feature-test fixture only."

    def get_examples(self) -> list[str]:
        return [
            "echo hello",
            "echo the word banana",
            "repeat this phrase back to me",
            "say exactly what I say",
            "mirror my input",
            "give me back my text",
        ]

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    def run(self, params: dict) -> str:
        self.ran_with = dict(params)
        return f"echoed: {params.get('text', '')}"


def test_dispatch_without_async_runs_inline():
    """async absent → default False → execute runs run() inline and returns it.
    DmnConfig keeps broadcast_to=None so the emitter is a no-op (no WS side
    effects); the async DECISION is independent of channel."""
    ctx = _Ctx(DmnConfig())
    ability = _EchoAbility(mp=ctx)

    result = ToolDispatcher(ctx)._execute(ability, {"text": "hi"}, None)

    assert result == "echoed: hi"
    # The framework popped the async flag before handing params to run().
    assert ability.ran_with == {"text": "hi"}


def test_dispatch_with_async_returns_placeholder_and_registers_delegate():
    """async: true → execute spawns run() on a daemon thread, returns the
    placeholder immediately (proving non-blocking), and registers one delegate.
    The delegate is cancelled before release so the post-run delivery is
    skipped — no LLM synthesis turn fires."""
    release = threading.Event()
    started = threading.Event()
    finished = []

    class _BlockingAbility(Ability):
        _SYNTHETIC = True  # keep out of the static registry (see _EchoAbility)

        def get_name(self) -> str:
            return "test_per_call_blocking"

        def get_search_tooltip(self) -> str:
            return "block until released — async feature-test fixture"

        def get_summary(self) -> str:
            return "Block until a test releases it — feature-test fixture only."

        def get_examples(self) -> list[str]:
            return [
                "block until I say go",
                "wait for my signal",
                "hold here until released",
                "pause until the gate opens",
                "stay blocked for the test",
                "wait then finish",
            ]

        def get_parameters(self) -> dict:
            return {"type": "object", "properties": {}, "required": []}

        def run(self, params: dict) -> str:
            started.set()
            release.wait(timeout=5)
            finished.append(True)
            return "done"

    ctx = _Ctx(DmnConfig())
    ability = _BlockingAbility(mp=ctx)

    before = set(async_delegate_runner.active_ids())
    result = ToolDispatcher(ctx)._execute(ability, {"async": True}, None)

    # Non-blocking: the placeholder came back while run() is still blocked.
    assert "dispatched (id:" in result
    assert started.wait(timeout=2), "background run() never started"
    assert not finished, "run() returned before release — was not backgrounded"

    new = set(async_delegate_runner.active_ids()) - before
    assert len(new) == 1
    delegate_id = new.pop()
    assert delegate_id.startswith("test_per_call_blocking_")

    # Cancel BEFORE releasing → AsyncDelegateRunner._run skips delivery
    # (cancel_event.is_set()), so no real-LLM synthesis turn is spawned.
    async_delegate_runner.cancel(delegate_id)
    release.set()

    # The daemon finishes run(), skips delivery, and deregisters in `finally`.
    for _ in range(100):
        if delegate_id not in async_delegate_runner.active_ids():
            break
        time.sleep(0.05)
    assert delegate_id not in async_delegate_runner.active_ids()
    assert finished == [True]
