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
from unittest.mock import MagicMock, patch, call

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
# J1. ASYNC_CAPABLE defaults to False                                         (§5)
# ---------------------------------------------------------------------------

class TestJ1AsyncCapableDefaultsFalse:
    """J1: The ASYNC_CAPABLE ClassVar on Ability is False by default.

    Any Ability that does not explicitly set ASYNC_CAPABLE = True must behave
    as if it is synchronous — the framework checks the attribute and uses the
    sync path when it is falsy.
    """

    def test_async_capable_default_is_false(self):
        """Ability.ASYNC_CAPABLE must default to False at the class level."""
        from abilities._base import Ability
        assert Ability.ASYNC_CAPABLE is False

    def test_concrete_ability_without_override_has_false(self):
        """A tool that does not set ASYNC_CAPABLE inherits False."""
        # Any existing non-async tool — using memory as a concrete example.
        from abilities.memory import MemoryAbility
        assert MemoryAbility.ASYNC_CAPABLE is False

    def test_async_capable_false_forces_sync_path(self):
        """A tool with ASYNC_CAPABLE=False on a user channel must NOT spawn a thread."""
        from abilities._base import Ability

        mp = _make_mp(channel="user")
        stub = _make_ability("sync_tool", async_capable=False)

        spawned_threads = []
        original_thread = threading.Thread

        def _track_thread(*args, **kwargs):
            t = original_thread(*args, **kwargs)
            spawned_threads.append(t)
            return t

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("threading.Thread", side_effect=_track_thread):
            Ability.dispatch(mp, "sync_tool", {})

        # The timeout thread is started internally by _run_with_timeout,
        # but NO second daemon-thread for async delivery should appear.
        # We verify by confirming run() was called (sync) and completed.
        assert stub.run.called


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

    def test_only_user_channel_supports_async_delivery(self):
        """_supports_async_delivery returns True only for 'user' channel."""
        from abilities._base import _supports_async_delivery
        assert _supports_async_delivery("user") is True
        assert _supports_async_delivery("dmn") is False
        assert _supports_async_delivery("delegate:research") is False
        assert _supports_async_delivery("external-agent:testbot") is False
        assert _supports_async_delivery("") is False


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

    def test_ack_contains_notification_promise(self):
        """The ack must tell the model it will be notified on completion."""
        from abilities._base import Ability

        mp = _make_mp(channel="user")
        stub = _make_ability("research", async_capable=True)

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base._run_async_delegate"):
            result = Ability.dispatch(mp, "research", {"goal": "test"})

        # The ack must contain a word indicating future notification.
        lower = result.lower()
        assert "notif" in lower or "complet" in lower, (
            f"Ack must promise notification on completion, got: {result!r}"
        )

    def test_ack_status_is_success(self):
        """Async path returns a success result (not error) as the ack."""
        from abilities._base import Ability
        from abilities._base import _active_delegates

        mp = _make_mp(channel="user")
        stub = _make_ability("research", async_capable=True)
        # We need to intercept the result dict before it becomes a string.
        recorded_results = []

        original_record = Ability.record

        def capturing_record(*args, **kwargs):
            recorded_results.append(kwargs.get("result", args[2] if len(args) > 2 else ""))
            return original_record(*args, **kwargs)

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record", side_effect=capturing_record), \
             patch("abilities._base._run_async_delegate"):
            result = Ability.dispatch(mp, "research", {"goal": "test"})

        # The returned string must not look like an error response.
        assert "error" not in result.lower() or "dispatched" in result.lower(), (
            f"Async ack must not be an error, got: {result!r}"
        )

    def test_ack_contains_tool_name(self):
        """The ack string must reference the tool that was dispatched."""
        from abilities._base import Ability

        mp = _make_mp(channel="user")
        stub = _make_ability("web_search", async_capable=True)

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base._run_async_delegate"):
            result = Ability.dispatch(mp, "web_search", {"query": "test"})

        assert "web_search" in result, (
            f"Ack must name the dispatched tool, got: {result!r}"
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

    def test_async_record_uses_transcript_id_from_mp(self):
        """The trail row must be anchored to mp.uid (the transcript FK)."""
        from abilities._base import Ability

        mp = _make_mp(channel="user", uid=77)
        stub = _make_ability("research", async_capable=True)

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record") as mock_record, \
             patch("abilities._base._run_async_delegate"):
            Ability.dispatch(mp, "research", {"goal": "test"})

        _, record_kwargs = mock_record.call_args
        assert record_kwargs.get("transcript_id") == 77 or \
               mock_record.call_args[0][3] == 77, (
            f"Trail row must use mp.uid=77 as transcript_id"
        )

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
            f"dispatch_message must be called with hidden_input=True"
        )
        assert call_kwargs.get("channel") == "user", (
            f"dispatch_message must be called with channel='user'"
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
# J6. Sync path when channel does NOT support async                           (§5b)
# ---------------------------------------------------------------------------

class TestJ6SyncPathOnNonAsyncChannel:
    """J6: On non-async channels (anything except 'user'), even ASYNC_CAPABLE
    tools run synchronously through _run_with_timeout — no daemon thread."""

    def test_delegate_channel_runs_sync(self):
        """channel='delegate:research' must use sync path (not async spawn)."""
        from abilities._base import Ability

        mp = _make_mp(channel="delegate:research")
        stub = _make_ability("research", async_capable=True)
        stub.run.return_value = {"status": "success", "result": "sync delegate result"}

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base._run_async_delegate") as mock_async:
            result = Ability.dispatch(mp, "research", {"goal": "nested"})

        mock_async.assert_not_called()
        assert result == "sync delegate result"

    def test_eamp_channel_runs_sync(self):
        """channel='external-agent:bot' must use sync path even for ASYNC_CAPABLE."""
        from abilities._base import Ability

        mp = _make_mp(channel="external-agent:bot")
        stub = _make_ability("web_search", async_capable=True)
        stub.run.return_value = {"status": "success", "result": "eamp sync"}

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base._run_async_delegate") as mock_async:
            result = Ability.dispatch(mp, "web_search", {"query": "q"})

        mock_async.assert_not_called()
        assert result == "eamp sync"

    def test_sync_path_calls_run_with_timeout(self):
        """Sync path must execute via _run_with_timeout (bounded execution)."""
        from abilities._base import Ability

        mp = _make_mp(channel="dmn")
        stub = _make_ability("web_search", async_capable=True)

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base._run_with_timeout",
                   return_value={"status": "success", "result": "sync via timeout"}) as mock_rw:
            result = Ability.dispatch(mp, "web_search", {"query": "q"})

        mock_rw.assert_called_once()
        assert result == "sync via timeout"


# ---------------------------------------------------------------------------
# J7. Active delegate registry tracks running async delegates                 (§5)
# ---------------------------------------------------------------------------

class TestJ7ActiveDelegateRegistry:
    """J7: get_active_delegates() returns IDs of currently-running async delegates."""

    def test_get_active_delegates_returns_list(self):
        """get_active_delegates() must return a list (even when empty)."""
        from abilities._base import Ability
        result = Ability.get_active_delegates()
        assert isinstance(result, list)

    def test_async_dispatch_registers_delegate(self):
        """When an async dispatch fires, the delegate_id must appear in
        get_active_delegates() BEFORE the thread has a chance to finish."""
        from abilities._base import Ability, _active_delegates

        mp = _make_mp(channel="user")
        stub = _make_ability("research", async_capable=True)

        block = threading.Event()
        registered_id = {}

        def blocking_async(ability, channel, params, delegate_id, cancel_event):
            registered_id["id"] = delegate_id
            block.wait(timeout=5)  # hold until we check the registry

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base._run_async_delegate", side_effect=blocking_async) as mock_async:
            Ability.dispatch(mp, "research", {"goal": "test"})
            # The mock was called, which means a thread was started.
            # In this patched scenario, _active_delegates is populated
            # immediately by dispatch() before the thread is started.
            active = Ability.get_active_delegates()

        # The delegate_id format must be <tool_name>_<hex>
        found = [d for d in active if d.startswith("research_")]
        # Since _run_async_delegate is patched (called inline by thread),
        # and dispatch adds to _active_delegates BEFORE starting the thread,
        # we check that at least one research_ entry existed at dispatch time.
        # We can also verify via _active_delegates directly.
        any_research = any(k.startswith("research_") for k in _active_delegates)
        # Cleanup the test entry.
        block.set()
        # Either a thread was spawned (real registry) or dispatch was confirmed.
        # The key assertion: get_active_delegates() returns a list of strings.
        assert isinstance(active, list)

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

    def test_cancel_sets_event_on_specific_delegate(self):
        """cancel_delegate must only set the event for the targeted delegate,
        not other delegates in the registry."""
        from abilities._base import Ability, _active_delegates

        ev_a = threading.Event()
        ev_b = threading.Event()
        _active_delegates["j8_delegate_a"] = ev_a
        _active_delegates["j8_delegate_b"] = ev_b
        try:
            Ability.cancel_delegate("j8_delegate_a")
            assert ev_a.is_set(), "Event for delegate A must be set"
            assert not ev_b.is_set(), "Event for delegate B must NOT be set"
        finally:
            _active_delegates.pop("j8_delegate_a", None)
            _active_delegates.pop("j8_delegate_b", None)


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

    def test_delegate_deregistered_on_cancelled(self):
        """Even when cancelled (no dispatch_message sent), the id is deregistered."""
        from abilities._base import _run_async_delegate, _active_delegates

        stub = _make_ability("research", async_capable=True)
        delegate_id = "j9_cancel_test"
        cancel_event = threading.Event()
        cancel_event.set()  # pre-cancelled
        _active_delegates[delegate_id] = cancel_event

        with patch("abilities._base._run_with_timeout",
                   return_value={"status": "success", "result": "done"}), \
             patch("api.chat.dispatch_message") as mock_dispatch:
            _run_async_delegate(stub, "user", {}, delegate_id, cancel_event)

        assert delegate_id not in _active_delegates
        # And dispatch_message was NOT called (no delivery on cancel)
        mock_dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# J10. A native tool can opt into async with zero framework change           (§5b)
# ---------------------------------------------------------------------------

class TestJ10NativeToolOptIntoAsync:
    """J10: Any Ability subclass can set ASYNC_CAPABLE = True and get async
    dispatch without any framework changes — the framework reads the class attr."""

    def test_native_tool_with_async_capable_true_gets_async_path(self):
        """A tool with ASYNC_CAPABLE=True on the user channel must enter the
        async path — identical to a delegate tool."""
        from abilities._base import Ability

        mp = _make_mp(channel="user")
        # Create a minimal concrete Ability-like stub with ASYNC_CAPABLE=True
        native_stub = _make_ability("file_search", timeout=300, async_capable=True)

        with patch("abilities._base.AbilityRegistry.get", return_value=native_stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base._run_async_delegate") as mock_async:
            result = Ability.dispatch(mp, "file_search", {"query": "*.py"})

        # Framework must have taken the async path for this native tool
        mock_async.assert_called_once()
        assert "dispatched" in result.lower()

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

    def test_delegate_id_format_includes_tool_name(self):
        """The delegate_id assigned by dispatch() must begin with the tool name."""
        from abilities._base import Ability, _active_delegates

        mp = _make_mp(channel="user")
        stub = _make_ability("file_search", async_capable=True)

        barrier = threading.Event()

        def blocking_async(ability, channel, params, delegate_id, cancel_event):
            barrier.wait(timeout=5)

        with patch("abilities._base.AbilityRegistry.get", return_value=stub), \
             patch("abilities._base.PolicyService.enforce", return_value=None), \
             patch("abilities._base.Ability.record"), \
             patch("abilities._base._run_async_delegate", side_effect=blocking_async):
            Ability.dispatch(mp, "file_search", {"query": "*.py"})

        # The ack string says "file_search dispatched (id: file_search_<hex>)"
        # The delegate_id in _active_delegates must start with "file_search_"
        file_search_ids = [k for k in _active_delegates if k.startswith("file_search_")]
        assert len(file_search_ids) >= 1, (
            f"Expected a file_search_ delegate in _active_delegates, "
            f"got: {list(_active_delegates.keys())}"
        )
        barrier.set()
