"""Tests for the proactive hook registry in capabilities.base."""

import pytest
from unittest.mock import MagicMock, patch

from capabilities.base import (
    _hook_registry,
    register_hook,
    AbstractCapability,
)


class _TestCap(AbstractCapability):
    """Minimal concrete capability for testing dispatch."""

    def get_id(self):
        return "test"

    def get_manifest(self):
        return {"id": "test"}

    def configure(self, credentials):
        pass

    def connect(self):
        return True

    def disconnect(self):
        pass

    def ingest(self):
        return []

    def understand(self, items):
        return []

    def _do_monitor(self):
        pass

    def act(self, action, params):
        return {}

    def get_tools(self):
        return []


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore the hook registry around each test."""
    original = _hook_registry.copy()
    _hook_registry.clear()
    yield
    _hook_registry.clear()
    _hook_registry.extend(original)


@pytest.mark.unit
class TestRegisterHook:
    """Tests for register_hook()."""

    def test_adds_function_to_registry(self):
        fn = MagicMock(__name__="hook_a")
        register_hook(fn)
        assert fn in _hook_registry

    def test_deduplicates(self):
        fn = MagicMock(__name__="hook_a")
        register_hook(fn)
        register_hook(fn)
        assert _hook_registry.count(fn) == 1

    def test_multiple_hooks_ordered(self):
        a = MagicMock(__name__="a")
        b = MagicMock(__name__="b")
        register_hook(a)
        register_hook(b)
        assert _hook_registry == [a, b]


@pytest.mark.unit
class TestDispatchHooks:
    """Tests for AbstractCapability._dispatch_hooks()."""

    def test_calls_all_registered_hooks(self):
        a = MagicMock(__name__="a")
        b = MagicMock(__name__="b")
        register_hook(a)
        register_hook(b)

        cap = _TestCap()
        with patch("capabilities.base._discover_hooks"):
            cap._dispatch_hooks()

        a.assert_called_once()
        b.assert_called_once()

    def test_failing_hook_does_not_block_others(self):
        failing = MagicMock(__name__="failing", side_effect=RuntimeError)
        passing = MagicMock(__name__="passing")
        register_hook(failing)
        register_hook(passing)

        cap = _TestCap()
        with patch("capabilities.base._discover_hooks"):
            cap._dispatch_hooks()

        failing.assert_called_once()
        passing.assert_called_once()

    def test_empty_registry_is_noop(self):
        cap = _TestCap()
        with patch("capabilities.base._discover_hooks"):
            cap._dispatch_hooks()  # should not raise

    def test_dispatch_triggers_discovery(self):
        cap = _TestCap()
        with patch("capabilities.base._discover_hooks") as mock_disc:
            cap._dispatch_hooks()
            mock_disc.assert_called_once()
