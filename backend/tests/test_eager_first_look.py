"""Tests for eager hook dispatch on first monitor cycle.

When a capability's ``monitor()`` runs for the first time (initial sync
after connect), hooks fire *before* ``_do_monitor()`` so time-sensitive
hooks like ``first_look`` can use data from other capabilities that
already synced — without waiting for this capability's slow initial fetch.
"""

import pytest
from unittest.mock import patch

from capabilities.base import AbstractCapability


class _StubCapability(AbstractCapability):
    """Minimal concrete capability for testing base-class behaviour."""

    def __init__(self):
        super().__init__()
        self.do_monitor_calls = 0

    def get_id(self):
        return "stub"

    def get_manifest(self):
        return {"id": "stub", "name": "Stub"}

    def configure(self, credentials):
        pass

    def connect(self):
        self._connected = True
        return True

    def disconnect(self):
        self._connected = False

    def ingest(self, since=None):
        return []

    def understand(self, items):
        return []

    def _do_monitor(self):
        self.do_monitor_calls += 1

    def act(self, action, params):
        return {"success": True}

    def get_tools(self):
        return []


@pytest.mark.unit
class TestEagerFirstMonitorHooks:
    """Verify hooks fire before _do_monitor on the first monitor call."""

    def test_hooks_fire_before_do_monitor_on_first_call(self):
        """On the first monitor(), _dispatch_hooks runs before _do_monitor."""
        cap = _StubCapability()
        call_order = []

        original_dispatch = cap._dispatch_hooks

        def track_dispatch():
            call_order.append(("hooks", cap.do_monitor_calls))
            original_dispatch()

        original_do_monitor = cap._do_monitor

        def track_do_monitor():
            call_order.append(("monitor", cap.do_monitor_calls))
            original_do_monitor()

        with patch.object(cap, "_dispatch_hooks", side_effect=track_dispatch), \
             patch.object(cap, "_do_monitor", side_effect=track_do_monitor):
            cap.monitor()

        # First call: hooks fire at monitor_calls=0 (before _do_monitor),
        # then _do_monitor at 0, then hooks again at 1 (after _do_monitor)
        assert len(call_order) == 3
        assert call_order[0][0] == "hooks"     # pre-hooks
        assert call_order[1][0] == "monitor"   # _do_monitor
        assert call_order[2][0] == "hooks"     # post-hooks

    def test_second_monitor_call_no_pre_hooks(self):
        """On subsequent monitor() calls, hooks only fire after _do_monitor."""
        cap = _StubCapability()
        call_order = []

        original_dispatch = cap._dispatch_hooks

        def track_dispatch():
            call_order.append("hooks")
            original_dispatch()

        original_do_monitor = cap._do_monitor

        def track_do_monitor():
            call_order.append("monitor")
            original_do_monitor()

        # First call (clears the first_monitor flag)
        with patch.object(cap, "_dispatch_hooks", side_effect=track_dispatch), \
             patch.object(cap, "_do_monitor", side_effect=track_do_monitor):
            cap.monitor()

        call_order.clear()

        # Second call — no pre-hooks
        with patch.object(cap, "_dispatch_hooks", side_effect=track_dispatch), \
             patch.object(cap, "_do_monitor", side_effect=track_do_monitor):
            cap.monitor()

        assert call_order == ["monitor", "hooks"]

    def test_first_monitor_flag_initialized_true(self):
        """New capability instances have _first_monitor = True."""
        cap = _StubCapability()
        assert cap._first_monitor is True

    def test_first_monitor_flag_cleared_after_first_call(self):
        """After the first monitor() call, _first_monitor is False."""
        cap = _StubCapability()
        with patch.object(cap, "_dispatch_hooks"), \
             patch.object(cap, "_do_monitor"):
            cap.monitor()
        assert cap._first_monitor is False
