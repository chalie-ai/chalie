# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the dispatch replacement: Ability.use / match / execute.

use() is the single ACT-loop chokepoint: match → resolve permission →
PolicyManager.wrap(execute) → record → return STRING.  These tests pin the
contract that every boundary returns a string, that deny/unknown still get
recorded (so the model sees the outcome and stops retrying the blocked tool),
and that MCP tools now flow through the gate via the synthetic _MCPAbility.
"""
from unittest.mock import MagicMock

import pytest

from abilities import _base
from abilities._base import Ability, _MCPAbility
from services.processor_config import ProcessorConfig

pytestmark = pytest.mark.unit

CH = ProcessorConfig.POLICY_CHANNEL


class _Cfg:
    channel = "user"
    policy_channel = CH.CHAT
    broadcast_to = None          # _emit no-ops → no WS stub needed


class _MP:
    config = _Cfg()
    uid = None                   # record() skips the INSERT when uid is None
    active_tools: list = []
    cancel_event = None


@pytest.fixture()
def captured(monkeypatch):
    """Capture every Ability.record call without touching the DB."""
    calls = []
    monkeypatch.setattr(Ability, "record", staticmethod(lambda **kw: calls.append(kw)))
    return calls


# 1. match() routes native via the registry, _mcp_* via the synthetic proxy, unknown → None
def test_match_routes_native_mcp_and_unknown():
    real = Ability.match("memory")
    assert real is not None and real.NAME == "memory"

    proxy = Ability.match("_mcp_taskie_create_document")
    assert isinstance(proxy, _MCPAbility)
    assert proxy.NAME == "_mcp_taskie_create_document"

    assert Ability.match("definitely_not_a_real_tool") is None


# 2. allow path: wrap runs the callback; use returns execute's STRING + records it
def test_use_allow_runs_records_and_returns_string(monkeypatch, captured):
    ability = MagicMock()
    ability.execute.return_value = "OK"                     # execute returns a STRING
    monkeypatch.setattr(Ability, "match", staticmethod(lambda name: ability))
    monkeypatch.setattr(_base.PolicyManager, "wrap",
                        staticmethod(lambda channel, permission, callback, error=None: callback()))

    out = Ability.use(_MP(), "email", {"action": "search", "act_summary": "Searching email"})

    assert out == "OK"
    # act_summary is popped before execute/record and threaded to execute as a kwarg
    ability.execute.assert_called_once()
    _mp_arg, exec_params, exec_summary = ability.execute.call_args.args
    assert "act_summary" not in exec_params and exec_summary == "Searching email"
    assert captured[-1]["tool_name"] == "email"
    assert captured[-1]["result"] == "OK" and "act_summary" not in captured[-1]["params"]


# 3. permission key resolved inline from the action sub-key
@pytest.mark.parametrize("params,expected", [
    ({"action": "search"}, "email.search"),
    ({}, "email"),
])
def test_use_resolves_permission_key(monkeypatch, captured, params, expected):
    seen = {}
    ability = MagicMock()
    ability.execute.return_value = "x"
    monkeypatch.setattr(Ability, "match", staticmethod(lambda name: ability))

    def fake_wrap(channel, permission, callback, error=None):
        seen["permission"] = permission
        seen["channel"] = channel
        return callback()

    monkeypatch.setattr(_base.PolicyManager, "wrap", staticmethod(fake_wrap))
    Ability.use(_MP(), "email", dict(params))
    assert seen["permission"] == expected and seen["channel"] == CH.CHAT


# 4. deny path: wrap returns the block STRING; use returns it AND records it; callback never ran
def test_use_deny_returns_block_string_and_records(monkeypatch, captured):
    ability = MagicMock()
    monkeypatch.setattr(Ability, "match", staticmethod(lambda name: ability))
    monkeypatch.setattr(_base.PolicyManager, "wrap",
                        staticmethod(lambda channel, permission, callback, error=None:
                                     "The bash.execute action is not allowed. Do NOT retry."))

    out = Ability.use(_MP(), "bash", {"action": "execute"})

    assert out == "The bash.execute action is not allowed. Do NOT retry."
    ability.execute.assert_not_called()
    assert captured[-1]["result"] == "The bash.execute action is not allowed. Do NOT retry."


# 5. unknown tool: returns a string, records it, NEVER calls wrap (nothing to gate)
def test_use_unknown_returns_string_records_without_gating(monkeypatch, captured):
    monkeypatch.setattr(Ability, "match", staticmethod(lambda name: None))
    called = {"wrap": False}
    monkeypatch.setattr(_base.PolicyManager, "wrap",
                        staticmethod(lambda *a, **k: called.__setitem__("wrap", True)))

    out = Ability.use(_MP(), "no_such_tool", {"action": "x"})

    assert out == "Unknown tool: no_such_tool"
    assert called["wrap"] is False
    assert captured[-1]["tool_name"] == "no_such_tool"
    assert captured[-1]["result"] == "Unknown tool: no_such_tool"


# 6. MCP tools are gated through wrap (closes the previously un-gated gap)
def test_use_gates_mcp_through_wrap(monkeypatch, captured):
    monkeypatch.setattr(_base.PolicyManager, "wrap",
                        staticmethod(lambda channel, permission, callback, error=None:
                                     "The _mcp_taskie_create_document action is not allowed. Do NOT retry."))
    out = Ability.use(_MP(), "_mcp_taskie_create_document", {})
    assert out == "The _mcp_taskie_create_document action is not allowed. Do NOT retry."
    assert captured[-1]["tool_name"] == "_mcp_taskie_create_document"


# 7. execute(): runs via run() in the timeout thread and returns the result STRING
def test_execute_runs_and_returns_string():
    class _Echo(Ability):
        _SYNTHETIC = True                       # skip __init_subclass__ + registry walk
        NAME = "echo"
        def run(self, channel, params, telemetry=None):
            return {"status": "success", "result": "ECHO"}

    out = _Echo().execute(_MP(), {"x": 1}, act_summary="echoing")
    assert out == "ECHO"
