"""Tests for ActDispatcherService — dispatch, timeout, confidence estimation."""

import time
import pytest

from services.act_dispatcher_service import ActDispatcherService, _estimate_confidence


pytestmark = pytest.mark.unit


@pytest.fixture
def service(db):
    """Create an ActDispatcherService with real innate-skill registration.

    Uses the ``db`` fixture so that ToolRegistryService._load_tools() queries
    the local test DB instead of the real chalie.db on the network mount.
    SQLite on SMB is unreliable for file locking; without this the fixture hangs
    indefinitely on the first SELECT to tool_configs.
    """
    svc = ActDispatcherService(timeout=2.0)
    yield svc


# ── Unknown / Missing Handler ─────────────────────────────────


class TestUnknownHandler:

    def test_unknown_handler_returns_error(self, service):
        """Dispatching an action with no registered handler returns status=error."""
        result = service.dispatch_action('topic', {'type': 'nonexistent_action'})

        assert result['status'] == 'error'
        assert result['confidence'] == pytest.approx(0.0, abs=1e-9)
        assert 'Unknown action type' in result['result']
        assert result['action_type'] == 'nonexistent_action'


# ── Successful Dispatch ────────────────────────────────────────


class TestSuccessfulDispatch:

    def test_successful_handler_returns_success(self, service):
        """A handler that returns a value produces status=success with the result."""
        # Use a safe action type to bypass the execution gate
        service.handlers['recall'] = lambda topic, action: {'output': 'ok'}

        result = service.dispatch_action('topic', {'type': 'recall'})

        assert result['status'] == 'success'
        assert result['result'] == {'output': 'ok'}
        assert result['action_type'] == 'recall'


# ── Handler Exception ──────────────────────────────────────────


class TestHandlerException:

    def test_handler_exception_returns_error(self, service):
        """When the handler raises, dispatch catches it and returns status=error."""
        def exploding_handler(topic, action):
            raise ValueError("something broke")

        service.handlers['recall'] = exploding_handler

        result = service.dispatch_action('topic', {'type': 'recall'})

        assert result['status'] == 'error'
        assert result['confidence'] == pytest.approx(0.0, abs=1e-9)
        assert 'something broke' in result['result']


# ── Timeout ────────────────────────────────────────────────────


class TestTimeout:

    def test_slow_handler_returns_timeout(self):
        """A handler that exceeds the timeout produces status=timeout."""
        svc = ActDispatcherService(timeout=0.1)

        def slow_handler(topic, action):
            time.sleep(5)
            return 'too late'

        svc.handlers['recall'] = slow_handler

        result = svc.dispatch_action('topic', {'type': 'recall'})

        assert result['status'] == 'timeout'
        assert result['confidence'] == pytest.approx(0.0, abs=1e-9)


# ── Confidence Estimation ──────────────────────────────────────


class TestConfidenceEstimation:

    def test_memory_confidence_is_deterministic(self, service):
        """Deterministic actions like 'memory' get 0.92 confidence."""
        service.handlers['memory'] = lambda topic, action: 'stored'

        result = service.dispatch_action('topic', {'type': 'memory'})

        assert result['confidence'] == pytest.approx(0.92)

    def test_find_tools_long_result_confidence(self, service):
        """A read action with a result longer than 100 chars gets 0.75 confidence."""
        service.handlers['find_tools'] = lambda topic, action: 'x' * 101

        result = service.dispatch_action('topic', {'type': 'find_tools'})

        assert result['confidence'] == pytest.approx(0.75)

    def test_find_tools_short_result_confidence(self, service):
        """A read action with a result of 20 chars or fewer gets 0.40 confidence."""
        service.handlers['find_tools'] = lambda topic, action: 'short'

        result = service.dispatch_action('topic', {'type': 'find_tools'})

        assert result['confidence'] == pytest.approx(0.40)

    def test_default_confidence_for_unknown_action_type(self, service):
        """An action type not in deterministic or read sets gets 0.50 confidence."""
        service.handlers['custom_thing'] = lambda topic, action: 'result'

        result = service.dispatch_action('topic', {'type': 'custom_thing'})

        assert result['confidence'] == pytest.approx(0.50)


# ── _estimate_confidence unit tests (direct) ──────────────────



