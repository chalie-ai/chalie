"""
Unit tests for CapabilitySchedulerService.

Covers:
- Construction with default and custom intervals.
- ``_run_cycle()`` skipping disconnected capabilities.
- ``_run_cycle()`` calling ``ingest()`` on connected capabilities.
- Exception isolation: a failing ``ingest()`` never propagates out of the cycle.
- Backoff counter increments on consecutive failures.
- Backoff counter resets on success.
- Module-level ``capability_scheduler_worker`` callable contract.

All tests are marked ``@pytest.mark.unit`` and avoid real network, database,
or filesystem I/O.  ``time.sleep`` is patched where needed to avoid delays.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_cap(*, connected: bool, ingest_side_effect=None) -> MagicMock:
    """Create a mock capability object with a controllable ``is_connected`` return value.

    Args:
        connected: Value that ``is_connected()`` should return.
        ingest_side_effect: Optional side-effect (e.g. an exception class or instance)
            to assign to the ``ingest`` mock method.  When ``None`` the mock returns
            an empty list by default.

    Returns:
        MagicMock: A pre-configured mock capability.
    """
    cap = MagicMock()
    cap.is_connected.return_value = connected
    if ingest_side_effect is not None:
        cap.ingest.side_effect = ingest_side_effect
    else:
        cap.ingest.return_value = []
    return cap


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCapabilitySchedulerServiceInstantiation:
    """Construction tests for :class:`CapabilitySchedulerService`."""

    def test_instantiation(self):
        """CapabilitySchedulerService() should instantiate with a default interval of 15.

        Verifies that constructing without arguments results in
        ``interval_minutes == 15``.
        """
        from services.capability_scheduler_service import CapabilitySchedulerService

        svc = CapabilitySchedulerService()
        assert svc.interval_minutes == 15

    def test_instantiation_with_custom_interval(self):
        """CapabilitySchedulerService(interval_minutes=30) should store 30.

        Verifies that a custom interval is persisted on the instance so that
        the sleep between cycles honours the caller's preference.
        """
        from services.capability_scheduler_service import CapabilitySchedulerService

        svc = CapabilitySchedulerService(interval_minutes=30)
        assert svc.interval_minutes == 30


class TestRunCycleConnectivity:
    """Tests for how ``_run_cycle`` handles connected vs disconnected capabilities."""

    def test_skips_disconnected_capabilities(self):
        """``_run_cycle`` must never call ``ingest()`` on a disconnected capability.

        Patches ``load_capabilities`` to return a single disconnected mock and
        asserts that ``cap.ingest`` is never invoked.  The ``capabilities``
        module is injected via ``sys.modules`` so the ``from capabilities import
        load_capabilities`` statement inside ``_run_cycle`` resolves to the mock.
        """
        from services.capability_scheduler_service import CapabilitySchedulerService

        cap = _make_mock_cap(connected=False)
        svc = CapabilitySchedulerService()

        mock_caps_module = MagicMock()
        mock_caps_module.load_capabilities.return_value = {"stub": cap}

        with patch.dict("sys.modules", {"capabilities": mock_caps_module}):
            svc._run_cycle()

        cap.ingest.assert_not_called()

    def test_calls_ingest_on_connected_capability(self):
        """``_run_cycle`` must call ``ingest()`` exactly once on a connected capability.

        Patches ``load_capabilities`` to return a single connected mock, stubs
        ``_store_last_sync`` to be a no-op, and verifies ``ingest`` is called.
        """
        from services.capability_scheduler_service import CapabilitySchedulerService

        cap = _make_mock_cap(connected=True)
        svc = CapabilitySchedulerService()

        mock_caps_module = MagicMock()
        mock_caps_module.load_capabilities.return_value = {"caldav": cap}

        with patch.dict("sys.modules", {"capabilities": mock_caps_module}), \
             patch.object(svc, "_store_last_sync"):
            svc._run_cycle()

        cap.ingest.assert_called_once()


class TestRunCycleExceptionIsolation:
    """Tests that exceptions in individual capabilities are contained."""

    def test_exception_in_ingest_does_not_propagate(self):
        """A ``RuntimeError`` raised by ``ingest()`` must not propagate out of ``_run_cycle``.

        Verifies that the scheduler is resilient to individual capability
        failures: one bad capability should never crash the worker loop.
        """
        from services.capability_scheduler_service import CapabilitySchedulerService

        cap = _make_mock_cap(connected=True, ingest_side_effect=RuntimeError("boom"))
        svc = CapabilitySchedulerService()

        mock_caps_module = MagicMock()
        mock_caps_module.load_capabilities.return_value = {"failing_cap": cap}

        # Should not raise — the scheduler must swallow the exception
        with patch.dict("sys.modules", {"capabilities": mock_caps_module}):
            svc._run_cycle()  # must not raise


class TestBackoffBehaviour:
    """Tests for per-capability exponential back-off state management."""

    def test_backoff_increments_on_failure(self):
        """``_failure_counts[cap_id]`` must increment after each call to ``_record_failure``.

        Calls ``_record_failure`` twice for the same capability and asserts that
        the counter reaches 2, confirming it accumulates consecutive failures.
        """
        from services.capability_scheduler_service import CapabilitySchedulerService

        svc = CapabilitySchedulerService()
        cap_id = "test_cap"

        svc._record_failure(cap_id)
        assert svc._failure_counts[cap_id] == 1

        svc._record_failure(cap_id)
        assert svc._failure_counts[cap_id] == 2

    def test_backoff_resets_on_success(self):
        """``_failure_counts[cap_id]`` must be cleared after ``_record_success``.

        Pre-seeds the failure counter to 3 to simulate prior failures, then
        calls ``_record_success`` and asserts that the counter is removed
        (i.e., the capability is no longer in back-off).
        """
        from services.capability_scheduler_service import CapabilitySchedulerService

        svc = CapabilitySchedulerService()
        cap_id = "recovering_cap"

        # Simulate prior failures
        svc._failure_counts[cap_id] = 3
        svc._record_success(cap_id)

        assert cap_id not in svc._failure_counts


class TestWorkerEntryPoint:
    """Tests for the module-level worker function."""

    def test_capability_scheduler_worker_is_callable(self):
        """``capability_scheduler_worker`` must be callable and accept ``shared_state={}``.

        Patches ``CapabilitySchedulerService.run`` to return immediately so the
        test completes without blocking in the infinite loop.  Verifies that
        the function can be invoked without raising.
        """
        from services.capability_scheduler_service import capability_scheduler_worker

        assert callable(capability_scheduler_worker)

        with patch(
            "services.capability_scheduler_service.CapabilitySchedulerService.run",
            return_value=None,
        ):
            # Must not raise
            capability_scheduler_worker(shared_state={})
