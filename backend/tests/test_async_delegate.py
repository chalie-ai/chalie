# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""§9a area J (J1–J10) — Async delegate / ASYNC_CAPABLE primitive, spec §5/§5b.

Tests authored per the GOVERNING TEST PRINCIPLE:
  - Encode behaviour exactly as §9a states.
  - A failing test means the CODE is wrong, not the test.
  - Never weaken, delete, xfail, or "correct" a §9a test.

Spec references: §5, §5b.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mp(channel="user", broadcast_to=None, uid=99):
    """Build a minimal object that resembles a MessageProcessor for dispatch."""
    mp = MagicMock(spec_set=["config", "uid", "cancel_event", "discovered_tools"])
    mp.config = MagicMock(spec_set=["channel", "broadcast_to"])
    mp.config.channel = channel
    mp.config.broadcast_to = broadcast_to
    mp.uid = uid
    mp.cancel_event = threading.Event()
    mp.discovered_tools = []
    return mp


def _make_ability(
    name: str = "test_tool",
    timeout: int = 10,
    run_return=None,
    async_capable: bool = False,
):
    """Build a minimal Ability-like stub (spec_set to catch phantom attrs)."""
    ability = MagicMock(spec_set=["NAME", "TIMEOUT", "ASYNC_CAPABLE", "run"])
    ability.NAME = name
    ability.TIMEOUT = timeout
    ability.ASYNC_CAPABLE = async_capable
    if run_return is None:
        run_return = {"status": "success", "result": "tool output"}
    ability.run.return_value = run_return
    return ability


# ---------------------------------------------------------------------------
# J2. Async path only when ASYNC_CAPABLE AND channel supports async           (§5/§5b)
# ---------------------------------------------------------------------------

class TestJ2AsyncPathConditions:
    """J2: Async spawn happens ONLY when both conditions are true:
    tool.ASYNC_CAPABLE == True AND _supports_async_delivery(channel) == True.
    """

    def test_async_capable_true_on_user_channel_spawns_daemon(self):
        """ASYNC_CAPABLE=True + channel='user' must spawn a daemon thread and
        return the ack immediately (run() is NOT called synchronously)."""
        from abilities._base import Ability

        mp = _make_mp(channel="user")
        stub = _make_ability("delegate_tool", async_capable=True)

        dispatched_delegate_ids = []

        def fake_run_async(ability, channel, params, delegate_id, cancel_event):
            dispatched_delegate_ids.append(delegate_id)

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base._run_async_delegate", side_effect=fake_run_async) as mock_async:
            result = Ability.dispatch(mp, "delegate_tool", {"goal": "do something"})

        # The async helper was invoked (daemon thread target)
        mock_async.assert_called_once()
        # Result is the ack string, not the real tool result
        assert "dispatched" in result.lower()

    def test_async_capable_true_on_non_user_channel_is_sync(self):
        """ASYNC_CAPABLE=True but channel='dmn' (not async-capable) must NOT
        spawn a daemon — fall through to synchronous _run_with_timeout."""
        from abilities._base import Ability

        mp = _make_mp(channel="dmn")
        stub = _make_ability("delegate_tool", async_capable=True)
        stub.run.return_value = {"status": "success", "result": "sync result"}

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base._run_async_delegate") as mock_async:
            result = Ability.dispatch(mp, "delegate_tool", {"goal": "do something"})

        # Async helper must NOT have been called
        mock_async.assert_not_called()
        # Result is the tool's actual output
        assert result == "sync result"

    def test_async_capable_false_on_user_channel_is_sync(self):
        """ASYNC_CAPABLE=False on user channel must NOT spawn a daemon — sync path."""
        from abilities._base import Ability

        mp = _make_mp(channel="user")
        stub = _make_ability("native_tool", async_capable=False)
        stub.run.return_value = {"status": "success", "result": "native result"}

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base._run_async_delegate") as mock_async:
            result = Ability.dispatch(mp, "native_tool", {})

        mock_async.assert_not_called()
        assert result == "native result"


# ---------------------------------------------------------------------------
# J3. Async ack tells the model work was dispatched (id + notify promise)    (§5/§5b)
# ---------------------------------------------------------------------------

class TestJ3AsyncAckContent:
    """J3: The ack string returned by dispatch() for an async call must include
    the delegate_id and a promise that the model will be notified."""

    def test_ack_contains_delegate_id(self):
        """The ack result string must include the delegate_id."""
        from abilities._base import Ability

        mp = _make_mp(channel="user")
        stub = _make_ability("research", async_capable=True)

        captured_delegate_id = {}

        def fake_run_async(ability, channel, params, delegate_id, cancel_event):
            captured_delegate_id["id"] = delegate_id

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base._run_async_delegate", side_effect=fake_run_async):
            result = Ability.dispatch(mp, "research", {"goal": "test"})

        assert captured_delegate_id["id"] in result, (
            f"Ack must contain the delegate_id '{captured_delegate_id['id']}' "
            f"but got: {result!r}"
        )


# ---------------------------------------------------------------------------
# J4. Async ack recorded as a trail row; parent loop continues               (§5b/§6)
# ---------------------------------------------------------------------------

class TestJ4AckRecordedAsTrailRow:
    """J4: dispatch() records exactly one trail row for an async call — the ack
    text becomes that row's result. The parent loop sees the ack as the tool
    result and continues without blocking."""

    def test_async_dispatch_records_exactly_one_row(self):
        """Async dispatch must call Ability.record() exactly once."""
        from abilities._base import Ability

        mp = _make_mp(channel="user", uid=42)
        stub = _make_ability("research", async_capable=True)

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record") as mock_record, \
             patch("abilities._base._run_async_delegate"):
            Ability.dispatch(mp, "research", {"goal": "test"})

        mock_record.assert_called_once()

    def test_async_dispatch_returns_immediately(self):
        """Async dispatch must not block — it returns before the delegate finishes."""
        from abilities._base import Ability

        mp = _make_mp(channel="user")
        stub = _make_ability("slow_tool", timeout=600, async_capable=True)
        # _run_async_delegate sleeps to simulate a slow tool
        sleep_started = threading.Event()

        def slow_async(ability, channel, params, delegate_id, cancel_event):
            sleep_started.set()
            time.sleep(10)  # would block for 10s if called synchronously

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base._run_async_delegate", side_effect=slow_async):
            t0 = time.monotonic()
            Ability.dispatch(mp, "slow_tool", {})
            elapsed = time.monotonic() - t0

        # The call must return in well under 1s (certainly not 10s).
        assert elapsed < 2.0, (
            f"Async dispatch blocked for {elapsed:.3f}s — must return immediately"
        )


# ---------------------------------------------------------------------------
# J5. Async result delivered via dispatch_message(hidden_input=True)         (§5b/§6)
# ---------------------------------------------------------------------------

class TestJ5AsyncResultDelivery:
    """J5: When the delegate finishes, _run_async_delegate delivers the result
    via dispatch_message(channel, hidden_input=True) — triggering a fresh
    parent turn with the result text."""

    def test_run_async_delegate_calls_dispatch_message_on_success(self):
        """_run_async_delegate must call dispatch_message with hidden_input=True
        after the tool completes successfully."""
        from abilities._base import _run_async_delegate, _active_delegates

        stub = _make_ability("research", async_capable=True)
        stub.run.return_value = {"status": "success", "result": "research done"}

        delegate_id = "research_test123"
        cancel_event = threading.Event()
        _active_delegates[delegate_id] = cancel_event

        with patch("abilities._base._run_with_timeout", return_value={"status": "success", "result": "research done"}), \
             patch("api.chat.dispatch_message") as mock_dispatch:
            _run_async_delegate(stub, "user", {}, delegate_id, cancel_event)

        mock_dispatch.assert_called_once()
        _, call_kwargs = mock_dispatch.call_args
        assert call_kwargs.get("hidden_input") is True, (
            "dispatch_message must be called with hidden_input=True"
        )
        assert call_kwargs.get("channel") == "user", (
            "dispatch_message must be called with channel='user'"
        )

    def test_run_async_delegate_does_not_deliver_when_cancelled(self):
        """If the cancel_event is set before delivery, dispatch_message is NOT called."""
        from abilities._base import _run_async_delegate, _active_delegates

        stub = _make_ability("research", async_capable=True)
        delegate_id = "research_cancelled"
        cancel_event = threading.Event()
        cancel_event.set()  # pre-cancelled
        _active_delegates[delegate_id] = cancel_event

        with patch("abilities._base._run_with_timeout", return_value={"status": "success", "result": "done"}), \
             patch("api.chat.dispatch_message") as mock_dispatch:
            _run_async_delegate(stub, "user", {}, delegate_id, cancel_event)

        mock_dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# J7. Active delegate registry tracks running async delegates                 (§5)
# ---------------------------------------------------------------------------

class TestJ7ActiveDelegateRegistry:
    """J7: get_active_delegates() returns IDs of currently-running async delegates."""

    def test_dispatch_registers_before_thread_starts(self):
        """The delegate_id must be in _active_delegates immediately after
        dispatch() returns — before the daemon thread has a chance to complete."""
        from abilities._base import Ability, _active_delegates

        mp = _make_mp(channel="user")
        stub = _make_ability("research", async_capable=True)

        # Block the async thread so it can't deregister before we check.
        barrier = threading.Event()

        def blocking_run_async(ability, channel, params, delegate_id, cancel_event):
            barrier.wait(timeout=5)

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base._run_async_delegate", side_effect=blocking_run_async):
            result = Ability.dispatch(mp, "research", {"goal": "check"})

        # At this point the thread is blocking on barrier.wait().
        # The delegate_id embedded in the ack must be in _active_delegates.
        # Extract the id from the ack string: "research dispatched (id: <id>). ..."
        import re
        match = re.search(r"id: (research_[0-9a-f]+)", result)
        assert match is not None, f"Could not find delegate_id in ack: {result!r}"
        delegate_id = match.group(1)
        assert delegate_id in _active_delegates, (
            f"Delegate id {delegate_id!r} not found in _active_delegates"
        )
        # Cleanup
        barrier.set()


# ---------------------------------------------------------------------------
# J8. cancel_delegate cancels a known delegate (True), False for unknown     (§5)
# ---------------------------------------------------------------------------

class TestJ8CancelDelegate:
    """J8: cancel_delegate() signals cooperative stop for known delegates;
    returns False for unknown ids."""

    def test_cancel_known_delegate_returns_true(self):
        """cancel_delegate must return True and set the cancel_event when the
        delegate_id is in the registry."""
        from abilities._base import Ability, _active_delegates

        cancel_event = threading.Event()
        _active_delegates["known_delegate_j8"] = cancel_event
        try:
            result = Ability.cancel_delegate("known_delegate_j8")
            assert result is True
            assert cancel_event.is_set(), "cancel_event must be set after cancel_delegate"
        finally:
            _active_delegates.pop("known_delegate_j8", None)

    def test_cancel_unknown_delegate_returns_false(self):
        """cancel_delegate must return False for an id not in the registry."""
        from abilities._base import Ability
        result = Ability.cancel_delegate("nonexistent_delegate_j8_abc")
        assert result is False


# ---------------------------------------------------------------------------
# J9. Completed async delegate is deregistered                                (§5b)
# ---------------------------------------------------------------------------

class TestJ9CompletedDelegateDeregistered:
    """J9: After _run_async_delegate completes (success or failure), the
    delegate_id must be removed from _active_delegates."""

    def test_delegate_deregistered_on_success(self):
        """After the async thread body completes successfully, the delegate_id
        must no longer appear in _active_delegates."""
        from abilities._base import _run_async_delegate, _active_delegates

        stub = _make_ability("research", async_capable=True)
        delegate_id = "j9_success_test"
        cancel_event = threading.Event()
        _active_delegates[delegate_id] = cancel_event

        with patch("abilities._base._run_with_timeout",
                   return_value={"status": "success", "result": "done"}), \
             patch("api.chat.dispatch_message"):
            _run_async_delegate(stub, "user", {}, delegate_id, cancel_event)

        assert delegate_id not in _active_delegates, (
            f"Delegate {delegate_id!r} must be deregistered after completion"
        )

    def test_delegate_deregistered_on_exception(self):
        """Even when _run_with_timeout raises, the delegate_id must be removed."""
        from abilities._base import _run_async_delegate, _active_delegates

        stub = _make_ability("research", async_capable=True)
        delegate_id = "j9_exception_test"
        cancel_event = threading.Event()
        _active_delegates[delegate_id] = cancel_event

        with patch("abilities._base._run_with_timeout", side_effect=RuntimeError("boom")):
            _run_async_delegate(stub, "user", {}, delegate_id, cancel_event)

        assert delegate_id not in _active_delegates, (
            f"Delegate {delegate_id!r} must be deregistered even after exception"
        )


# ---------------------------------------------------------------------------
# J10. A native tool can opt into async with zero framework change           (§5b)
# ---------------------------------------------------------------------------

class TestJ10NativeToolOptIntoAsync:
    """J10: Any Ability subclass can set ASYNC_CAPABLE = True and get async
    dispatch without any framework changes — the framework reads the class attr."""

    def test_setting_async_capable_requires_no_framework_change(self):
        """The framework checks ability.ASYNC_CAPABLE — the tool author writes
        ASYNC_CAPABLE = True and nothing else changes."""
        from abilities._base import Ability

        # Verify the check is purely attribute-based (duck typing, no isinstance)
        mp = _make_mp(channel="user")

        class _MinimalNativeAsync:
            NAME = "minimal_async_native"
            TIMEOUT = 60
            ASYNC_CAPABLE = True

            def run(self, channel, params, telemetry):
                return {"status": "success", "result": "native async done"}

        stub = _MinimalNativeAsync()

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base._run_async_delegate") as mock_async:
            result = Ability.dispatch(mp, "minimal_async_native", {})

        mock_async.assert_called_once()
        assert "dispatched" in result.lower()
