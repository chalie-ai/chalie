# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for per-call async (spec §4.0 / §4.1 / §4.8d — P4).

The framework ``async`` boolean is exposed on a tool's schema ONLY on a
``SUPPORTS_ASYNC`` channel (UserConfig) — never on a delegate/system
channel (DmnConfig) and never when there is no live processor.  This is the
schema-exposure gate, not a routing gate (§4.8d).

The async LLM follow-up turn delivery — and the on-cancel "cancelled by the
user" notice turn — is exercised end-to-end where a real model is available,
not here.  This test proves the *decision* and the
*non-blocking* contract without firing the LLM: the captured mp carries no
config, so the post-run delivery bails at ``deliver_async_result``'s no-config
guard instead of synthesising a turn (§4.4).
"""

import threading
import time
from typing import cast

import pytest

from abilities._ability import Ability
from abilities._dispatcher import ToolDispatcher
from abilities._registry import AbilityRegistry
from abilities._result import ToolResult
from configs.channels import DmnConfig, UserConfig
from services.async_delegate_runner import async_delegate_runner
from services.processor_config import ProcessorConfig

pytestmark = pytest.mark.unit


class _Ctx:
    def __init__(self, config: "ProcessorConfig | None") -> None:
        self.config = config


def test_async_property_exposed_only_on_supports_async_channel() -> None:
    weather_cls = type(AbilityRegistry.get("weather"))

    user_schema = cast("dict[str, object]", weather_cls(mp=_Ctx(UserConfig({}))).get_input_schema()["input_schema"])
    assert "async" in cast("dict[str, object]", user_schema["properties"])
    assert cast("dict[str, object]", cast("dict[str, object]", user_schema["properties"])["async"])["type"] == "boolean"
    assert cast("dict[str, object]", cast("dict[str, object]", user_schema["properties"])["async"])["default"] is False
    # The declared params (get_parameters) are never mutated — exposure deepcopies.
    assert "async" not in cast("dict[str, object]", weather_cls().get_parameters()["properties"])

    # A non-async channel (delegate/system) never sees the flag.
    dmn_schema = cast("dict[str, object]", weather_cls(mp=_Ctx(DmnConfig())).get_input_schema()["input_schema"])
    assert "async" not in cast("dict[str, object]", dmn_schema.get("properties", {}))

    # No live processor (mp=None) → synchronous default, no flag.
    assert "async" not in cast("dict[str, object]", cast("dict[str, object]", weather_cls().get_input_schema()["input_schema"]).get("properties", {}))


def test_build_tools_exposes_async_on_user_channel() -> None:

    class _Proc:
        config = UserConfig({})
        active_tools = ["weather"]

    tools = AbilityRegistry.build_tools(_Proc())
    weather_tool = next(t for t in tools if t["name"] == "weather")
    props = cast("dict[str, object]", cast("dict[str, object]", weather_tool["input_schema"])["properties"])
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

    def __init__(self, mp: object = None) -> None:
        super().__init__(mp)
        self.ran_with: dict[str, object] | None = None

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

    def get_parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    def run(self, params: dict[str, object]) -> ToolResult:
        self.ran_with = dict(params)
        return ToolResult.ok(f"echoed: {params.get('text', '')}")


def test_dispatch_without_async_runs_inline() -> None:
    ctx = _Ctx(DmnConfig())
    ability = _EchoAbility(mp=ctx)

    result = ToolDispatcher(ctx)._execute(ability, {"text": "hi"}, None)

    # _execute renders the ToolResult through the single envelope formatter.
    assert result == (
        "[test_per_call_echo(status=success)]\n"
        "echoed: hi\n"
        "[end:test_per_call_echo]"
    )
    # The framework popped the async flag before handing params to run().
    assert ability.ran_with == {"text": "hi"}


def test_dispatch_with_async_returns_placeholder_and_registers_delegate() -> None:
    release = threading.Event()
    started = threading.Event()
    finished: list[bool] = []

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

        def get_parameters(self) -> dict[str, object]:
            return {"type": "object", "properties": {}, "required": []}

        def run(self, params: dict[str, object]) -> ToolResult:
            started.set()
            release.wait(timeout=5)
            finished.append(True)
            return ToolResult.ok("done")

    # Config-less captured mp: the dispatcher's async branch needs no config
    # (the emitter no-ops on None), and on cancel the notice delivery bails at
    # deliver_async_result's no-config guard — keeping this test LLM-free.
    ctx = _Ctx(None)
    ability = _BlockingAbility(mp=ctx)

    before = {cast(str, d["sub_id"]) for d in async_delegate_runner.active()}
    result = ToolDispatcher(ctx)._execute(ability, {"async": True}, None)

    # Non-blocking: the placeholder came back while run() is still blocked.
    assert "dispatched (id:" in result
    assert started.wait(timeout=2), "background run() never started"
    assert not finished, "run() returned before release — was not backgrounded"

    new = {cast(str, d["sub_id"]) for d in async_delegate_runner.active()} - before
    assert len(new) == 1
    delegate_id = new.pop()
    assert delegate_id.startswith("test_per_call_blocking_")

    # Cancel BEFORE releasing → on completion _run builds the "cancelled by the
    # user" notice and routes it to delivery; the captured mp has no config, so
    # deliver_async_result bails at its no-config guard — no real-LLM turn.
    async_delegate_runner.cancel(delegate_id)
    release.set()

    # The daemon finishes run(), the notice delivery bails, and it deregisters in `finally`.
    for _ in range(100):
        if delegate_id not in {cast(str, d["sub_id"]) for d in async_delegate_runner.active()}:
            break
        time.sleep(0.05)
    assert delegate_id not in {cast(str, d["sub_id"]) for d in async_delegate_runner.active()}
    assert finished == [True]
