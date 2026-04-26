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
        assert result['confidence'] == 0.0
        assert 'unknown-action-type' in result['result']
        assert result['action_type'] == 'nonexistent_action'

    def test_missing_type_defaults_to_unknown(self, service):
        """Action dict without a 'type' key falls back to 'unknown'."""
        result = service.dispatch_action('topic', {})

        assert result['status'] == 'error'
        assert result['action_type'] == 'unknown'


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

    def test_execution_time_is_tracked(self, service):
        """Result includes a positive execution_time."""
        service.handlers['recall'] = lambda topic, action: 'done'

        result = service.dispatch_action('topic', {'type': 'recall'})

        assert result['status'] == 'success'
        assert 'execution_time' in result
        assert result['execution_time'] > 0


# ── Handler Exception ──────────────────────────────────────────


class TestHandlerException:

    def test_handler_exception_returns_error(self, service):
        """When the handler raises, dispatch catches it and returns status=error."""
        def exploding_handler(topic, action):
            raise ValueError("something broke")

        service.handlers['recall'] = exploding_handler

        result = service.dispatch_action('topic', {'type': 'recall'})

        assert result['status'] == 'error'
        assert result['confidence'] == 0.0
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
        assert result['confidence'] == 0.0


# ── Confidence Estimation ──────────────────────────────────────


class TestConfidenceEstimation:

    def test_memorize_confidence_is_deterministic(self, service):
        """Deterministic actions like 'memorize' get 0.92 confidence."""
        # Use the real memorize handler if available, or a lambda
        # Since we want to pressure test the dispatcher's confidence logic:
        service.handlers['memorize'] = lambda topic, action: 'stored'

        result = service.dispatch_action('topic', {'type': 'memorize'})

        assert result['confidence'] == pytest.approx(0.92)

    def test_recall_long_result_confidence(self, service):
        """Recall with a result longer than 100 chars gets 0.75 confidence."""
        service.handlers['recall'] = lambda topic, action: 'x' * 101

        result = service.dispatch_action('topic', {'type': 'recall'})

        assert result['confidence'] == pytest.approx(0.75)

    def test_recall_medium_result_confidence(self, service):
        """Recall with a result between 21 and 100 chars gets 0.60 confidence."""
        service.handlers['recall'] = lambda topic, action: 'x' * 50

        result = service.dispatch_action('topic', {'type': 'recall'})

        assert result['confidence'] == pytest.approx(0.60)

    def test_recall_short_result_confidence(self, service):
        """Recall with a result of 20 chars or fewer gets 0.40 confidence."""
        service.handlers['recall'] = lambda topic, action: 'short'

        result = service.dispatch_action('topic', {'type': 'recall'})

        assert result['confidence'] == pytest.approx(0.40)

    def test_default_confidence_for_unknown_action_type(self, service):
        """An action type not in deterministic or read sets gets 0.50 confidence."""
        service.handlers['custom_thing'] = lambda topic, action: 'result'

        result = service.dispatch_action('topic', {'type': 'custom_thing'})

        assert result['confidence'] == pytest.approx(0.50)


# ── _estimate_confidence unit tests (direct) ──────────────────


class TestEstimateConfidenceDirectly:

    def test_deterministic_ignores_result_content(self):
        """Deterministic confidence is fixed regardless of result."""
        assert _estimate_confidence('memorize', '') == pytest.approx(0.92)
        assert _estimate_confidence('memorize', None) == pytest.approx(0.92)

    def test_read_with_none_result(self):
        """Read action with None result gets the lowest read confidence."""
        assert _estimate_confidence('recall', None) == pytest.approx(0.40)


# ── Untagged result paths are now canonical blocks ─────────────


class TestUntaggedPathsAreTagged:

    def test_unknown_action_returns_tagged_error(self, service):
        """Unknown action type result is wrapped in canonical [<name>(...)] block."""
        result = service.dispatch_action('user', {'type': 'no_such_action'})

        assert result['result'].startswith('[no_such_action(')
        assert result['result'].endswith('[end:no_such_action]')
        assert 'unknown-action-type' in result['result']

    def test_handler_exception_returns_tagged_error(self, service):
        """Handler that raises is wrapped in canonical [<name>(...)] block."""
        service.handlers['boom'] = lambda c, a: (_ for _ in ()).throw(RuntimeError('kaboom'))

        result = service.dispatch_action('user', {'type': 'boom'})

        assert result['result'].startswith('[boom(')
        assert result['result'].endswith('[end:boom]')
        assert 'handler-exception' in result['result']
        assert 'kaboom' in result['result']

    def test_timeout_returns_tagged_error(self):
        """Handler that exceeds timeout is wrapped in canonical [<name>(...)] block."""
        svc = ActDispatcherService(timeout=0.05)

        def slow_handler(c, a):
            time.sleep(0.5)
            return 'too late'

        svc.handlers['recall'] = slow_handler

        result = svc.dispatch_action('user', {'type': 'recall'})

        assert result['result'].startswith('[recall(')
        assert result['result'].endswith('[end:recall]')
        assert 'timeout' in result['result']
