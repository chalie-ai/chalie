"""Tests for ActDispatcherService — dispatch, timeout, confidence estimation."""

import time
import pytest
from unittest.mock import patch, MagicMock

from services.act_dispatcher_service import ActDispatcherService, _estimate_confidence


pytestmark = pytest.mark.unit


@pytest.fixture
def service():
    """Create an ActDispatcherService with innate-skill registration mocked out."""
    with patch('services.innate_skills.register_innate_skills'):
        svc = ActDispatcherService(timeout=2.0)
    return svc


# ── Unknown / Missing Handler ─────────────────────────────────


class TestUnknownHandler:

    def test_unknown_handler_returns_error(self, service):
        """Dispatching an action with no registered handler returns status=error."""
        result = service.dispatch_action('topic', {'type': 'nonexistent_action'})

        assert result['status'] == 'error'
        assert result['confidence'] == 0.0
        assert 'Unknown action type' in result['result']
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
        service.handlers['test_action'] = lambda topic, action: {'output': 'ok'}

        result = service.dispatch_action('topic', {'type': 'test_action'})

        assert result['status'] == 'success'
        assert result['result'] == {'output': 'ok'}
        assert result['action_type'] == 'test_action'

    def test_execution_time_is_tracked(self, service):
        """Result includes a positive execution_time."""
        service.handlers['test_action'] = lambda topic, action: 'done'

        result = service.dispatch_action('topic', {'type': 'test_action'})

        assert result['status'] == 'success'
        assert 'execution_time' in result
        assert result['execution_time'] > 0


# ── Handler Exception ──────────────────────────────────────────


class TestHandlerException:

    def test_handler_exception_returns_error(self, service):
        """When the handler raises, dispatch catches it and returns status=error."""
        def exploding_handler(topic, action):
            raise ValueError("something broke")

        service.handlers['boom'] = exploding_handler

        result = service.dispatch_action('topic', {'type': 'boom'})

        assert result['status'] == 'error'
        assert result['confidence'] == 0.0
        assert 'something broke' in result['result']


# ── Timeout ────────────────────────────────────────────────────


class TestTimeout:

    def test_slow_handler_returns_timeout(self):
        """A handler that exceeds the timeout produces status=timeout."""
        with patch('services.innate_skills.register_innate_skills'):
            svc = ActDispatcherService(timeout=0.1)

        def slow_handler(topic, action):
            time.sleep(5)
            return 'too late'

        svc.handlers['slow'] = slow_handler

        result = svc.dispatch_action('topic', {'type': 'slow'})

        assert result['status'] == 'timeout'
        assert result['confidence'] == 0.0


# ── Confidence Estimation ──────────────────────────────────────


class TestConfidenceEstimation:

    def test_memorize_confidence_is_deterministic(self, service):
        """Deterministic actions like 'memorize' get 0.92 confidence."""
        service.handlers['memorize'] = lambda topic, action: 'stored'

        result = service.dispatch_action('topic', {'type': 'memorize'})

        assert result['confidence'] == pytest.approx(0.92)

    def test_introspect_confidence_is_deterministic(self, service):
        """Deterministic actions like 'introspect' get 0.92 confidence."""
        service.handlers['introspect'] = lambda topic, action: 'reflected'

        result = service.dispatch_action('topic', {'type': 'introspect'})

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
        assert _estimate_confidence('memorize', '') == 0.92
        assert _estimate_confidence('introspect', None) == 0.92

    def test_read_with_none_result(self):
        """Read action with None result gets the lowest read confidence."""
        assert _estimate_confidence('recall', None) == 0.40

    def test_associate_follows_read_rules(self):
        """'associate' is a read action and follows length-based confidence."""
        assert _estimate_confidence('associate', 'y' * 101) == 0.75
        assert _estimate_confidence('associate', 'y' * 50) == 0.60
        assert _estimate_confidence('associate', 'y') == 0.40

    def test_autobiography_follows_read_rules(self):
        """'autobiography' is a read action and follows length-based confidence."""
        assert _estimate_confidence('autobiography', 'z' * 200) == 0.75
