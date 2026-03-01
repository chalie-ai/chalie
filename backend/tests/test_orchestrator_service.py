"""Tests for OrchestratorService — path routing, validation, and execution."""

import pytest
from unittest.mock import patch, MagicMock

from services.orchestrator_service import OrchestratorService
from services.orchestrator.path_schemas import ORCHESTRATOR_PATHS


pytestmark = pytest.mark.unit


# ── Valid context fixtures for each mode ──────────────────────


def _respond_context(**overrides):
    ctx = {
        'response': 'Here is the answer.',
        'confidence': 0.8,
        'topic': 'test-topic',
        'destination': 'user',
    }
    ctx.update(overrides)
    return ctx


def _act_context(**overrides):
    ctx = {
        'actions': [{'type': 'recall', 'query': 'weather'}],
        'topic': 'test-topic',
    }
    ctx.update(overrides)
    return ctx


def _clarify_context(**overrides):
    ctx = {
        'clarification_question': 'Could you clarify what you mean?',
        'topic': 'test-topic',
        'destination': 'user',
    }
    ctx.update(overrides)
    return ctx


def _acknowledge_context(**overrides):
    ctx = {
        'topic': 'test-topic',
        'destination': 'user',
    }
    ctx.update(overrides)
    return ctx


def _ignore_context(**overrides):
    ctx = {
        'topic': 'test-topic',
    }
    ctx.update(overrides)
    return ctx


@pytest.fixture
def service():
    """Create an OrchestratorService with external dependencies mocked."""
    with patch('services.orchestrator_service.ActDispatcherService'), \
         patch('services.orchestrator_service.OutputService') as mock_output_cls:
        # OutputService.enqueue_text must return an output_id string
        mock_output = mock_output_cls.return_value
        mock_output.enqueue_text.return_value = 'output-id-123'
        svc = OrchestratorService()
    return svc


# ── Available Paths ────────────────────────────────────────────


class TestAvailablePaths:

    def test_returns_all_five_paths(self, service):
        """get_available_paths returns exactly the 5 defined paths."""
        paths = service.get_available_paths()
        names = [p['name'] for p in paths]

        assert len(paths) == 5
        assert set(names) == {'RESPOND', 'ACT', 'CLARIFY', 'ACKNOWLEDGE', 'IGNORE'}

    def test_each_path_has_required_keys(self, service):
        """Every path definition contains name, type, required_fields, and description."""
        paths = service.get_available_paths()

        for path in paths:
            assert 'name' in path, f"Path missing 'name': {path}"
            assert 'type' in path, f"Path missing 'type': {path}"
            assert 'required_fields' in path, f"Path missing 'required_fields': {path}"
            assert 'description' in path, f"Path missing 'description': {path}"
            assert isinstance(path['required_fields'], list)


# ── RESPOND Path ───────────────────────────────────────────────


class TestRespondPath:

    def test_respond_with_valid_context_succeeds(self, service):
        """RESPOND with all required fields returns status=success."""
        result = service.route_path('RESPOND', _respond_context())

        assert result['status'] == 'success'
        assert result['mode'] == 'RESPOND'
        assert 'result' in result

    def test_respond_missing_fields_fails(self, service):
        """RESPOND without required fields returns status=error."""
        result = service.route_path('RESPOND', {'topic': 'test'})

        assert result['status'] == 'error'
        assert 'Validation failed' in result['message']

    def test_respond_requires_confidence_between_0_and_1(self, service):
        """RESPOND with confidence outside [0,1] fails validation."""
        result = service.route_path('RESPOND', _respond_context(confidence=1.5))

        assert result['status'] == 'error'
        assert 'Validation failed' in result['message']

    def test_respond_with_empty_response_fails(self, service):
        """RESPOND with an empty response string fails validation."""
        result = service.route_path('RESPOND', _respond_context(response=''))

        assert result['status'] == 'error'


# ── ACT Path ──────────────────────────────────────────────────


class TestActPath:

    def test_act_with_valid_context_succeeds(self, service):
        """ACT with a non-empty actions list returns status=success."""
        result = service.route_path('ACT', _act_context())

        assert result['status'] == 'success'
        assert result['mode'] == 'ACT'

    def test_act_with_empty_actions_fails(self, service):
        """ACT with an empty actions list fails validation."""
        result = service.route_path('ACT', _act_context(actions=[]))

        assert result['status'] == 'error'
        assert 'Validation failed' in result['message']

    def test_act_missing_actions_field_fails(self, service):
        """ACT without the actions field at all fails validation."""
        result = service.route_path('ACT', {'topic': 'test-topic'})

        assert result['status'] == 'error'


# ── CLARIFY Path ──────────────────────────────────────────────


class TestClarifyPath:

    def test_clarify_with_valid_context_succeeds(self, service):
        """CLARIFY with a clarification question returns status=success."""
        result = service.route_path('CLARIFY', _clarify_context())

        assert result['status'] == 'success'
        assert result['mode'] == 'CLARIFY'


# ── ACKNOWLEDGE Path ──────────────────────────────────────────


class TestAcknowledgePath:

    def test_acknowledge_with_valid_context_succeeds(self, service):
        """ACKNOWLEDGE with topic and destination returns status=success."""
        result = service.route_path('ACKNOWLEDGE', _acknowledge_context())

        assert result['status'] == 'success'
        assert result['mode'] == 'ACKNOWLEDGE'


# ── IGNORE Path ───────────────────────────────────────────────


class TestIgnorePath:

    def test_ignore_with_valid_context_succeeds(self, service):
        """IGNORE with a topic returns status=success with ignored=True."""
        result = service.route_path('IGNORE', _ignore_context())

        assert result['status'] == 'success'
        assert result['mode'] == 'IGNORE'
        assert result['result']['ignored'] is True


# ── Unknown Mode ──────────────────────────────────────────────


class TestUnknownMode:

    def test_unknown_mode_returns_error(self, service):
        """Routing to a mode that does not exist returns status=error."""
        result = service.route_path('UNKNOWN_MODE', {'topic': 'test'})

        assert result['status'] == 'error'
        assert 'Unknown path' in result['message']


# ── Handler Exception ─────────────────────────────────────────


class TestHandlerException:

    def test_handler_exception_returns_error(self, service):
        """When a handler raises, route_path catches it and returns status=error."""
        # Replace the RESPOND handler with one that explodes
        mock_handler = MagicMock()
        mock_handler.execute.side_effect = RuntimeError("handler crashed")
        service.handlers['RESPOND'] = mock_handler

        # Also update the path definition so it uses the broken handler
        ORCHESTRATOR_PATHS['RESPOND'].handler = mock_handler

        result = service.route_path('RESPOND', _respond_context())

        assert result['status'] == 'error'
        assert 'Execution failed' in result['message']
