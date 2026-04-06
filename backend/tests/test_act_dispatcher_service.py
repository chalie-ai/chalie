"""Tests for ActDispatcherService — dispatch, timeout, confidence estimation."""

import time
import pytest
from unittest.mock import patch, MagicMock

from services.act_dispatcher_service import ActDispatcherService, _estimate_confidence


pytestmark = pytest.mark.unit


@pytest.fixture
def _auto_execute_gate():
    """Mock the execution gate to always allow auto-execution in tests."""
    mock_gate = MagicMock()
    mock_gate.evaluate.return_value = {
        'auto_execute': True,
        'consequence_tier': 0,
        'consequence_name': 'observe',
        'domain': 'test',
        'domain_confidence': 1.0,
        'threshold': 0.0,
        'reasoning': 'test override',
    }
    with patch('services.autonomous_execution_gate.get_autonomous_execution_gate',
               return_value=mock_gate):
        yield mock_gate


@pytest.fixture
def service(_auto_execute_gate):
    """Create an ActDispatcherService with innate-skill registration mocked out."""
    with patch('services.innate_skills.register_innate_skills'):
        svc = ActDispatcherService(timeout=2.0)
        yield svc


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

    def test_slow_handler_returns_timeout(self, _auto_execute_gate):
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


# ── Execution Gate ────────────────────────────────────────────


class TestExecutionGate:

    def test_gate_blocks_tier3_action(self):
        """Tier 3 (COMMIT) actions return requires_confirmation, not success."""
        mock_gate = MagicMock()
        mock_gate.evaluate.return_value = {
            'auto_execute': False,
            'consequence_tier': 3,
            'consequence_name': 'commit',
            'domain': 'general',
            'domain_confidence': 0.0,
            'threshold': float('inf'),
            'reasoning': 'Tier 3 (commit) — irreversible; always requires explicit user approval.',
        }
        with patch('services.autonomous_execution_gate.get_autonomous_execution_gate',
                   return_value=mock_gate), \
             patch('services.innate_skills.register_innate_skills'):
            svc = ActDispatcherService(timeout=2.0)
            svc.handlers['document'] = lambda topic, action: 'deleted'

            result = svc.dispatch_action('topic', {
                'type': 'document',
                'description': 'delete all my documents permanently',
            })

        assert result['status'] == 'requires_confirmation'
        assert result['confidence'] == 0.0
        assert 'gate_decision' in result
        assert result['gate_decision']['consequence_tier'] == 3

    def test_safe_actions_bypass_gate(self):
        """Safe innate skills (recall, memorize, etc.) bypass the gate entirely."""
        mock_gate = MagicMock()
        mock_gate.evaluate.return_value = {
            'auto_execute': False,  # Would block if called
            'consequence_tier': 2,
            'consequence_name': 'act',
            'domain': 'general',
            'domain_confidence': 0.0,
            'threshold': 0.75,
            'reasoning': 'Would block.',
        }
        with patch('services.autonomous_execution_gate.get_autonomous_execution_gate',
                   return_value=mock_gate), \
             patch('services.innate_skills.register_innate_skills'):
            svc = ActDispatcherService(timeout=2.0)
            svc.handlers['recall'] = lambda topic, action: 'memory result'

            result = svc.dispatch_action('topic', {'type': 'recall'})

        assert result['status'] == 'success'
        mock_gate.evaluate.assert_not_called()


class TestEstimateConfidenceDirectly:

    def test_deterministic_ignores_result_content(self):
        """Deterministic confidence is fixed regardless of result."""
        assert _estimate_confidence('memorize', '') == 0.92
        assert _estimate_confidence('introspect', None) == 0.92

    def test_read_with_none_result(self):
        """Read action with None result gets the lowest read confidence."""
        assert _estimate_confidence('recall', None) == 0.40

    def test_autobiography_follows_read_rules(self):
        """'autobiography' is a read action and follows length-based confidence."""
        assert _estimate_confidence('autobiography', 'z' * 200) == 0.75
