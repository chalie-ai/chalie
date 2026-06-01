# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
§9a I1–I14, N4 — Ability.dispatch chokepoint contract.

Tests authored BLIND from spec §1–8 only (T3 slice).  Per the GOVERNING TEST
PRINCIPLE:
  - Encode behaviour exactly as §9a states.
  - A failing test means the code is wrong, not the test.
  - Never weaken, delete, xfail, or "correct" a §9a test.

Spec references: §4, §5, §7b, §2.
"""

import threading
import time
from unittest.mock import MagicMock, patch, call

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Minimal stub helpers
# ---------------------------------------------------------------------------

def _make_mp(channel="user", broadcast_to=None, uid=None):
    """Build a minimal object that looks like a MessageProcessor for dispatch."""
    mp = MagicMock()
    mp.config = MagicMock()
    mp.config.channel = channel
    mp.config.broadcast_to = broadcast_to
    mp.uid = uid
    mp.cancel_event = threading.Event()
    mp.discovered_tools = []
    return mp


def _make_ability(name="test_tool", timeout=10, run_return=None, async_capable=False):
    """Build a minimal Ability-like stub."""
    ability = MagicMock()
    ability.NAME = name
    ability.TIMEOUT = timeout
    ability.ASYNC_CAPABLE = async_capable
    if run_return is None:
        run_return = {"status": "success", "result": "ok"}
    ability.run.return_value = run_return
    return ability


# ---------------------------------------------------------------------------
# I1. Every ACT-loop tool call routes through Ability.dispatch              (§4/§5/§7b)
# ---------------------------------------------------------------------------

class TestI1AllToolCallsRouteThrough:
    """I1: Ability.dispatch is the single chokepoint — all tool calls enter here."""

    def test_dispatch_is_callable_static_on_ability(self):
        from abilities._base import Ability
        assert callable(Ability.dispatch)

    def test_dispatch_returns_string(self):
        """Ability.dispatch must return a str result."""
        from abilities._base import Ability

        mp = _make_mp()
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "success", "result": "recalled data"}

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"):
            result = Ability.dispatch(mp, "memory", {"action": "recall"})

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# I2. dispatch sanitizes params and pops act_summary                        (§5)
# ---------------------------------------------------------------------------

class TestI2SanitizeAndPopActSummary:
    """I2: params are sanitized and act_summary is removed before tool sees them."""

    def test_act_summary_not_passed_to_tool(self):
        from abilities._base import Ability

        mp = _make_mp()
        captured_params = {}
        stub = _make_ability("memory")

        def capture_run(channel, params, telemetry):
            captured_params.update(params)
            return {"status": "success", "result": "ok"}

        stub.run.side_effect = capture_run

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"):
            Ability.dispatch(mp, "memory", {"action": "recall", "act_summary": "thinking about memory"})

        assert "act_summary" not in captured_params

    def test_act_summary_not_stored(self):
        """act_summary must not appear in the recorded trail row."""
        from abilities._base import Ability

        mp = _make_mp()
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "success", "result": "ok"}
        recorded_params = {}

        def capture_record(**kwargs):
            recorded_params.update(kwargs.get("params", {}))

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record", side_effect=capture_record):
            Ability.dispatch(mp, "memory", {"action": "recall", "act_summary": "summary text"})

        assert "act_summary" not in recorded_params

    def test_params_are_sanitized_before_dispatch(self):
        """LLM sentinel tokens (<|...|>) in string params are stripped before execution."""
        from abilities._base import Ability

        mp = _make_mp()
        captured_params = {}
        stub = _make_ability("memory")

        def capture_run(channel, params, telemetry):
            captured_params.update(params)
            return {"status": "success", "result": "ok"}

        stub.run.side_effect = capture_run

        # Inject a known Llama-style LLM sentinel — the sanitizer strips <|...|>
        dirty_params = {"query": "<|INST|>hello<|/INST|>"}

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"):
            Ability.dispatch(mp, "memory", dirty_params)

        # The sentinel tokens should be stripped; the human text remains
        assert "query" in captured_params
        assert "<|INST|>" not in captured_params["query"]
        assert "<|/INST|>" not in captured_params["query"]


# ---------------------------------------------------------------------------
# I3. dispatch generates a call_id when none provided                       (§5)
# ---------------------------------------------------------------------------

class TestI3CallIdGenerated:
    """I3: A call_id is minted when the caller passes None."""

    def test_call_id_generated_when_none(self):
        """dispatch must not raise when call_id=None; it mints its own."""
        from abilities._base import Ability

        mp = _make_mp()
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "success", "result": "ok"}

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"):
            result = Ability.dispatch(mp, "memory", {}, call_id=None)

        assert isinstance(result, str)

    def test_provided_call_id_accepted(self):
        """A non-None call_id is accepted without error."""
        from abilities._base import Ability

        mp = _make_mp()
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "success", "result": "ok"}

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"):
            result = Ability.dispatch(mp, "memory", {}, call_id="abc123")

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# I4. Start/end events gated on broadcast_to                                (§5/§2)
# ---------------------------------------------------------------------------

class TestI4BroadcastGated:
    """I4: WS events emitted when broadcast_to is set; suppressed when None."""

    def test_no_events_when_broadcast_to_none(self):
        """No WS broadcast at all when config.broadcast_to is None."""
        from abilities._base import Ability

        mp = _make_mp(broadcast_to=None)
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "success", "result": "ok"}

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base.WebSocketBroker") as mock_ws:
            Ability.dispatch(mp, "memory", {})

        mock_ws.return_value.broadcast.assert_not_called()

    def test_start_and_end_events_when_broadcast_to_set(self):
        """Both act_tool_start and act_tool_end are broadcast when broadcast_to='user'."""
        from abilities._base import Ability

        mp = _make_mp(broadcast_to="user")
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "success", "result": "ok"}

        broadcast_calls = []

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base.WebSocketBroker") as mock_ws_cls:
            mock_ws_cls.return_value.broadcast.side_effect = lambda d: broadcast_calls.append(d)
            Ability.dispatch(mp, "memory", {})

        types = [c["type"] for c in broadcast_calls]
        assert "act_tool_start" in types
        assert "act_tool_end" in types


# ---------------------------------------------------------------------------
# I5. End event reports ok=false on error                                   (§5)
# ---------------------------------------------------------------------------

class TestI5EndEventOkFalseOnError:
    """I5: act_tool_end carries ok=False when the tool returns an error status."""

    def test_end_event_ok_false_on_tool_error(self):
        from abilities._base import Ability

        mp = _make_mp(broadcast_to="user")
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "error", "result": "something broke"}

        end_events = []

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base.WebSocketBroker") as mock_ws_cls:
            mock_ws_cls.return_value.broadcast.side_effect = lambda d: end_events.append(d) if d.get("type") == "act_tool_end" else None
            Ability.dispatch(mp, "memory", {})

        assert len(end_events) == 1
        assert end_events[0]["ok"] is False

    def test_end_event_ok_true_on_success(self):
        from abilities._base import Ability

        mp = _make_mp(broadcast_to="user")
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "success", "result": "good"}

        end_events = []

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base.WebSocketBroker") as mock_ws_cls:
            mock_ws_cls.return_value.broadcast.side_effect = lambda d: end_events.append(d) if d.get("type") == "act_tool_end" else None
            Ability.dispatch(mp, "memory", {})

        assert len(end_events) == 1
        assert end_events[0]["ok"] is True


# ---------------------------------------------------------------------------
# I6. Unknown tool → error result, still recorded                           (§5)
# ---------------------------------------------------------------------------

class TestI6UnknownToolRecorded:
    """I6: Unknown tool_name yields an error result AND is still recorded."""

    def test_unknown_tool_returns_error_string(self):
        from abilities._base import Ability

        mp = _make_mp()

        with patch("abilities._base.AbilityRegistry.get", return_value=None), \
             patch("abilities._base.Ability.record"):
            result = Ability.dispatch(mp, "nonexistent_xyz", {})

        assert "nonexistent_xyz" in result or "Unknown" in result

    def test_unknown_tool_still_recorded(self):
        from abilities._base import Ability

        mp = _make_mp()
        recorded = []

        with patch("abilities._base.AbilityRegistry.get", return_value=None), \
             patch("abilities._base.Ability.record", side_effect=lambda **kw: recorded.append(kw)):
            Ability.dispatch(mp, "nonexistent_xyz", {})

        assert len(recorded) == 1
        assert recorded[0]["tool_name"] == "nonexistent_xyz"


# ---------------------------------------------------------------------------
# I7. Policy gate blocks before execution — run() not called                (§5)
# ---------------------------------------------------------------------------

class TestI7PolicyGateBlocks:
    """I7: When policy returns a block result, ability.run() is never called."""

    def test_run_not_called_when_policy_blocks(self):
        from abilities._base import Ability

        mp = _make_mp()
        stub = _make_ability("email")
        block_result = {"status": "policy_denied", "result": "Blocked by policy"}

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=block_result), \
             patch("abilities._base.Ability.record"):
            Ability.dispatch(mp, "email", {"action": "send"})

        stub.run.assert_not_called()

    def test_policy_block_result_is_recorded(self):
        from abilities._base import Ability

        mp = _make_mp()
        stub = _make_ability("email")
        block_result = {"status": "policy_denied", "result": "Blocked"}
        recorded = []

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=block_result), \
             patch("abilities._base.Ability.record", side_effect=lambda **kw: recorded.append(kw)):
            Ability.dispatch(mp, "email", {"action": "send"})

        assert len(recorded) == 1
        assert "Blocked" in recorded[0]["result"]


# ---------------------------------------------------------------------------
# I8. Policy enforced with running config's channel                         (§5/§6)
# ---------------------------------------------------------------------------

class TestI8PolicyUsesConfigChannel:
    """I8: PolicyService.enforce is called with the config's channel string."""

    def test_policy_receives_config_channel(self):
        from abilities._base import Ability

        mp = _make_mp(channel="dmn")
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "success", "result": "ok"}

        enforce_calls = []

        def fake_enforce(tool_name, params, channel):
            enforce_calls.append(channel)
            return None  # allow

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", side_effect=fake_enforce), \
             patch("abilities._base.Ability.record"):
            Ability.dispatch(mp, "memory", {})

        assert enforce_calls == ["dmn"]


# ---------------------------------------------------------------------------
# I9. _mcp_ tools route through the MCP dispatch path                       (§5)
# ---------------------------------------------------------------------------

class TestI9McpRouting:
    """I9: tool_name starting with '_mcp_' goes through the MCP path."""

    def test_mcp_tool_does_not_use_ability_registry(self):
        from abilities._base import Ability

        mp = _make_mp()

        with patch("abilities._base.AbilityRegistry.get") as mock_get, \
             patch("abilities._base._dispatch_mcp", return_value={"status": "success", "result": "mcp result"}), \
             patch("abilities._base.Ability.record"):
            Ability.dispatch(mp, "_mcp_server_tool", {"arg": "val"})

        mock_get.assert_not_called()

    def test_mcp_tool_dispatches_via_mcp_path(self):
        from abilities._base import Ability

        mp = _make_mp()
        mcp_calls = []

        def fake_mcp(mp_, tool_name, params):
            mcp_calls.append((tool_name, params))
            return {"status": "success", "result": "mcp ok"}

        with patch("abilities._base._dispatch_mcp", side_effect=fake_mcp), \
             patch("abilities._base.Ability.record"):
            Ability.dispatch(mp, "_mcp_myserver_mytool", {"key": "val"})

        assert len(mcp_calls) == 1
        assert mcp_calls[0][0] == "_mcp_myserver_mytool"

    def test_mcp_tool_result_is_recorded(self):
        from abilities._base import Ability

        mp = _make_mp()
        recorded = []

        with patch("abilities._base._dispatch_mcp", return_value={"status": "success", "result": "mcp data"}), \
             patch("abilities._base.Ability.record", side_effect=lambda **kw: recorded.append(kw)):
            Ability.dispatch(mp, "_mcp_server_tool", {})

        assert len(recorded) == 1
        assert "mcp data" in recorded[0]["result"]


# ---------------------------------------------------------------------------
# I10. Exactly one trail row per call                                        (§5)
# ---------------------------------------------------------------------------

class TestI10ExactlyOneTrailRow:
    """I10: Ability.record is called exactly once per dispatch, regardless of outcome."""

    def test_one_record_on_success(self):
        from abilities._base import Ability

        mp = _make_mp()
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "success", "result": "ok"}
        recorded = []

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record", side_effect=lambda **kw: recorded.append(kw)):
            Ability.dispatch(mp, "memory", {})

        assert len(recorded) == 1

    def test_one_record_on_error(self):
        from abilities._base import Ability

        mp = _make_mp()
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "error", "result": "broken"}
        recorded = []

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record", side_effect=lambda **kw: recorded.append(kw)):
            Ability.dispatch(mp, "memory", {})

        assert len(recorded) == 1

    def test_one_record_on_policy_block(self):
        from abilities._base import Ability

        mp = _make_mp()
        stub = _make_ability("email")
        block = {"status": "policy_denied", "result": "blocked"}
        recorded = []

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=block), \
             patch("abilities._base.Ability.record", side_effect=lambda **kw: recorded.append(kw)):
            Ability.dispatch(mp, "email", {})

        assert len(recorded) == 1

    def test_one_record_on_unknown_tool(self):
        from abilities._base import Ability

        mp = _make_mp()
        recorded = []

        with patch("abilities._base.AbilityRegistry.get", return_value=None), \
             patch("abilities._base.Ability.record", side_effect=lambda **kw: recorded.append(kw)):
            Ability.dispatch(mp, "ghost_tool", {})

        assert len(recorded) == 1


# ---------------------------------------------------------------------------
# I11. dispatch returns result text string to the loop                       (§5)
# ---------------------------------------------------------------------------

class TestI11ReturnsResultText:
    """I11: dispatch() returns the str result of the tool call."""

    def test_returns_string_from_successful_run(self):
        from abilities._base import Ability

        mp = _make_mp()
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "success", "result": "recalled memories"}

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"):
            result = Ability.dispatch(mp, "memory", {})

        assert result == "recalled memories"

    def test_returns_string_from_error_run(self):
        from abilities._base import Ability

        mp = _make_mp()
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "error", "result": "something went wrong"}

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"):
            result = Ability.dispatch(mp, "memory", {})

        assert result == "something went wrong"

    def test_returns_string_from_policy_block(self):
        from abilities._base import Ability

        mp = _make_mp()
        stub = _make_ability("email")
        block = {"status": "policy_denied", "result": "access denied"}

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=block), \
             patch("abilities._base.Ability.record"):
            result = Ability.dispatch(mp, "email", {})

        assert result == "access denied"


# ---------------------------------------------------------------------------
# I12. find_tools discovery side-effect on success only                     (§5)
# ---------------------------------------------------------------------------

class TestI12FindToolsSideEffect:
    """I12: find_tools result's _discovered_tools are added to mp.discovered_tools
    on success; skipped on error."""

    def test_discovered_tools_added_on_success(self):
        from abilities._base import Ability

        mp = _make_mp()
        mp.discovered_tools = []
        stub = _make_ability("find_tools")
        stub.run.return_value = {
            "status": "success",
            "result": "found tools",
            "_discovered_tools": ["search", "read"],
        }

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base._process_discovered_tools") as mock_process:
            Ability.dispatch(mp, "find_tools", {"query": "search tools"})

        mock_process.assert_called_once()

    def test_discovered_tools_skipped_on_error(self):
        from abilities._base import Ability

        mp = _make_mp()
        mp.discovered_tools = []
        stub = _make_ability("find_tools")
        stub.run.return_value = {
            "status": "error",
            "result": "search failed",
            "_discovered_tools": ["search"],
        }

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base._process_discovered_tools") as mock_process:
            Ability.dispatch(mp, "find_tools", {"query": "search tools"})

        mock_process.assert_not_called()


# ---------------------------------------------------------------------------
# I13. Sync tools execute in a timeout thread bounded by TIMEOUT             (§5)
# ---------------------------------------------------------------------------

class TestI13TimeoutThread:
    """I13: Sync ability.run() executes in a timeout thread; slow tools get cut off."""

    def test_slow_sync_tool_times_out(self):
        from abilities._base import Ability

        mp = _make_mp()
        stub = _make_ability("memory", timeout=1)

        def slow_run(channel, params, telemetry):
            time.sleep(10)
            return {"status": "success", "result": "too late"}

        stub.run.side_effect = slow_run

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"):
            t0 = time.monotonic()
            result = Ability.dispatch(mp, "memory", {})
            elapsed = time.monotonic() - t0

        # Should return well before the tool's 10-second sleep
        assert elapsed < 5.0
        assert isinstance(result, str)

    def test_fast_sync_tool_returns_result(self):
        from abilities._base import Ability

        mp = _make_mp()
        stub = _make_ability("memory", timeout=10)
        stub.run.return_value = {"status": "success", "result": "fast result"}

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"):
            result = Ability.dispatch(mp, "memory", {})

        assert result == "fast result"


# ---------------------------------------------------------------------------
# I14. Cancel mid-tool-loop stops further dispatches                         (§4)
# ---------------------------------------------------------------------------

class TestI14CancelStopsFurtherDispatches:
    """I14: When cancel_event is set, the ACT loop stops dispatching more tools."""

    def test_dispatch_after_cancel_is_not_called(self):
        """If cancel_event is set before dispatch is called, tool must not run."""
        from abilities._base import Ability

        mp = _make_mp()
        mp.cancel_event.set()  # pre-set the cancel flag
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "success", "result": "ok"}

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"):
            result = Ability.dispatch(mp, "memory", {})

        # When cancelled before dispatch, no tool execution; result is empty or cancel signal
        stub.run.assert_not_called()

    def test_dispatch_with_cancel_not_set_runs_normally(self):
        """Without cancel, dispatch works as normal."""
        from abilities._base import Ability

        mp = _make_mp()
        # cancel_event not set
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "success", "result": "normal"}

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"):
            result = Ability.dispatch(mp, "memory", {})

        stub.run.assert_called_once()
        assert result == "normal"


# ---------------------------------------------------------------------------
# N4. Tool start/end payload shapes                                          (§5)
# ---------------------------------------------------------------------------

class TestN4ToolEventPayloadShapes:
    """N4: act_tool_start includes summary; act_tool_end includes ok."""

    def test_start_event_includes_name_id_summary(self):
        from abilities._base import Ability

        mp = _make_mp(broadcast_to="user")
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "success", "result": "ok"}

        start_events = []

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base.WebSocketBroker") as mock_ws_cls:
            mock_ws_cls.return_value.broadcast.side_effect = lambda d: start_events.append(d) if d.get("type") == "act_tool_start" else None
            Ability.dispatch(mp, "memory", {"act_summary": "recalling facts"})

        assert len(start_events) == 1
        ev = start_events[0]
        assert ev["type"] == "act_tool_start"
        assert ev["name"] == "memory"
        assert "id" in ev
        assert ev.get("summary") == "recalling facts"

    def test_start_event_no_summary_when_not_provided(self):
        from abilities._base import Ability

        mp = _make_mp(broadcast_to="user")
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "success", "result": "ok"}

        start_events = []

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base.WebSocketBroker") as mock_ws_cls:
            mock_ws_cls.return_value.broadcast.side_effect = lambda d: start_events.append(d) if d.get("type") == "act_tool_start" else None
            Ability.dispatch(mp, "memory", {})

        assert len(start_events) == 1
        ev = start_events[0]
        # summary should be None or absent when not provided
        assert ev.get("summary") is None

    def test_end_event_includes_name_id_ok(self):
        from abilities._base import Ability

        mp = _make_mp(broadcast_to="user")
        stub = _make_ability("memory")
        stub.run.return_value = {"status": "success", "result": "ok"}

        end_events = []

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base.WebSocketBroker") as mock_ws_cls:
            mock_ws_cls.return_value.broadcast.side_effect = lambda d: end_events.append(d) if d.get("type") == "act_tool_end" else None
            Ability.dispatch(mp, "memory", {})

        assert len(end_events) == 1
        ev = end_events[0]
        assert ev["type"] == "act_tool_end"
        assert ev["name"] == "memory"
        assert "id" in ev
        assert "ok" in ev

    def test_end_event_ok_false_on_unknown_tool(self):
        from abilities._base import Ability

        mp = _make_mp(broadcast_to="user")
        end_events = []

        with patch("abilities._base.AbilityRegistry.get", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base.WebSocketBroker") as mock_ws_cls:
            mock_ws_cls.return_value.broadcast.side_effect = lambda d: end_events.append(d) if d.get("type") == "act_tool_end" else None
            Ability.dispatch(mp, "ghost_tool", {})

        assert len(end_events) == 1
        assert end_events[0]["ok"] is False
