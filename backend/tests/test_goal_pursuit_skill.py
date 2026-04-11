"""Tests for the goal_pursuit innate skill.

Verifies schema contract, input validation, thread spawning, and OutputService
surface behaviour on both success and failure.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from services.innate_skills.goal_pursuit_skill import (
    handle_goal_pursuit,
    TOOL_SCHEMA,
)

pytestmark = pytest.mark.unit


# ── Schema contract ────────────────────────────────────────────────────────────

class TestToolSchema:
    def test_schema_name_is_goal_pursuit(self):
        """TOOL_SCHEMA must advertise the canonical skill name."""
        assert TOOL_SCHEMA['name'] == 'goal_pursuit'

    def test_schema_has_required_goal_parameter(self):
        """'goal' must be a required parameter."""
        required = TOOL_SCHEMA['input_schema'].get('required', [])
        assert 'goal' in required

    def test_schema_goal_property_is_string(self):
        """The 'goal' property type must be 'string'."""
        props = TOOL_SCHEMA['input_schema']['properties']
        assert props['goal']['type'] == 'string'

    def test_schema_input_schema_type_is_object(self):
        """The top-level input schema must be type 'object'."""
        assert TOOL_SCHEMA['input_schema']['type'] == 'object'


# ── Input validation ───────────────────────────────────────────────────────────

class TestHandleGoalPursuitValidation:
    def test_empty_goal_returns_error_json(self):
        """Empty string goal returns a JSON error without spawning a thread."""
        result = json.loads(handle_goal_pursuit('user', {'goal': ''}))
        assert result['success'] is False
        assert 'error' in result

    def test_whitespace_only_goal_returns_error_json(self):
        """Whitespace-only goal is treated as empty and returns an error."""
        result = json.loads(handle_goal_pursuit('user', {'goal': '   \t\n  '}))
        assert result['success'] is False
        assert 'error' in result

    def test_missing_goal_key_returns_error_json(self):
        """Missing 'goal' key in params returns a JSON error."""
        result = json.loads(handle_goal_pursuit('user', {}))
        assert result['success'] is False
        assert 'error' in result

    def test_empty_goal_does_not_start_thread(self):
        """No thread must be spawned when the goal is invalid."""
        with patch('threading.Thread') as mock_thread_cls:
            handle_goal_pursuit('user', {'goal': ''})
            mock_thread_cls.assert_not_called()


# ── Happy-path return contract ─────────────────────────────────────────────────

class TestHandleGoalPursuitSuccess:
    def _call_with_patched_thread(self, goal='Summarise the latest news'):
        """Call handle_goal_pursuit with threading.Thread patched to a no-op."""
        with patch('threading.Thread') as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            raw = handle_goal_pursuit('user', {'goal': goal})
            parsed = json.loads(raw)
            return parsed, mock_thread_cls, mock_thread

    def test_valid_goal_returns_success_true(self):
        """A well-formed goal must return success=True."""
        result, _, _ = self._call_with_patched_thread()
        assert result['success'] is True

    def test_valid_goal_returns_pursuit_id(self):
        """Return JSON must include a non-empty 'pursuit_id'."""
        result, _, _ = self._call_with_patched_thread()
        assert 'pursuit_id' in result
        assert result['pursuit_id']  # non-empty string

    def test_valid_goal_response_contains_working_on_it(self):
        """The 'response' field must communicate that work is in progress."""
        result, _, _ = self._call_with_patched_thread()
        assert 'response' in result
        assert 'Working on it' in result['response']

    def test_valid_goal_returns_json_string(self):
        """handle_goal_pursuit must return a JSON-parseable string."""
        with patch('threading.Thread') as mock_thread_cls:
            mock_thread_cls.return_value = MagicMock()
            raw = handle_goal_pursuit('user', {'goal': 'do something'})
        assert isinstance(raw, str)
        # Must not raise
        json.loads(raw)

    def test_pursuit_id_is_hex_string(self):
        """pursuit_id must be the hex representation of a UUID (32 hex chars)."""
        result, _, _ = self._call_with_patched_thread()
        pid = result['pursuit_id']
        assert len(pid) == 32
        int(pid, 16)  # raises ValueError if not hex


# ── Thread spawning ────────────────────────────────────────────────────────────

class TestHandleGoalPursuitThread:
    def test_daemon_thread_is_started(self):
        """A daemon thread must be started for a valid goal."""
        with patch('threading.Thread') as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            handle_goal_pursuit('user', {'goal': 'do some research'})

        mock_thread_cls.assert_called_once()
        mock_thread.start.assert_called_once()

    def test_thread_is_marked_as_daemon(self):
        """Thread must be created with daemon=True for clean process exit."""
        with patch('threading.Thread') as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            handle_goal_pursuit('user', {'goal': 'do some research'})

        _, kwargs = mock_thread_cls.call_args
        assert kwargs.get('daemon') is True

    def test_thread_name_contains_pursuit_id_prefix(self):
        """Thread name must be prefixed with 'goal-pursuit-' for observability."""
        with patch('threading.Thread') as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            handle_goal_pursuit('user', {'goal': 'do something'})

        _, kwargs = mock_thread_cls.call_args
        assert kwargs.get('name', '').startswith('goal-pursuit-')

    def test_thread_returns_before_completion(self):
        """handle_goal_pursuit must return immediately — it does not join the thread."""
        # We verify by NOT calling join() on the mock thread
        with patch('threading.Thread') as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            handle_goal_pursuit('user', {'goal': 'do something'})

        mock_thread.join.assert_not_called()


# ── Background thread behaviour: success path ─────────────────────────────────

class TestGoalPursuitThreadSuccess:
    def _run_background_target(self, goal='Finish the task', response_text='Task complete.'):
        """Capture and immediately invoke the thread target function.

        The production code constructs GoalPursuitProcessor(raw_input=..., metadata=...)
        and calls .send() on the result (v2 API). The mock wires .send.return_value so
        the response text propagates correctly into OutputService.enqueue_proactive.
        """
        captured_target = {}

        def capture_thread(**kwargs):
            captured_target['fn'] = kwargs['target']
            m = MagicMock()
            return m

        mock_processor_instance = MagicMock()
        mock_processor_instance.send.return_value = response_text

        mock_output_instance = MagicMock()

        with patch('threading.Thread', side_effect=capture_thread), \
             patch('services.goal_pursuit_processor.GoalPursuitProcessor',
                   return_value=mock_processor_instance), \
             patch('services.output_service.OutputService',
                   return_value=mock_output_instance):
            handle_goal_pursuit('user', {'goal': goal})
            # Invoke the background function synchronously
            captured_target['fn']()

        return mock_processor_instance, mock_output_instance

    def test_success_calls_enqueue_proactive(self):
        """On success, OutputService.enqueue_proactive must be called once."""
        _, mock_output = self._run_background_target()
        mock_output.enqueue_proactive.assert_called_once()

    def test_success_source_is_goal_pursuit(self):
        """Proactive message source must be 'goal_pursuit'."""
        _, mock_output = self._run_background_target()
        _, kwargs = mock_output.enqueue_proactive.call_args
        assert kwargs.get('source') == 'goal_pursuit'

    def test_success_topic_is_user(self):
        """Proactive message topic must be 'user'."""
        _, mock_output = self._run_background_target()
        _, kwargs = mock_output.enqueue_proactive.call_args
        assert kwargs.get('topic') == 'user'

    def test_success_response_text_propagated(self):
        """The response text from the processor must appear in the proactive message."""
        _, mock_output = self._run_background_target(response_text='I found what you needed.')
        _, kwargs = mock_output.enqueue_proactive.call_args
        assert kwargs.get('response') == 'I found what you needed.'

    def test_empty_processor_response_uses_fallback_message(self):
        """When processor returns empty text, a fallback message is used instead."""
        mock_processor_instance = MagicMock()
        # .send() returns an empty string (v2 API returns str, not dict)
        mock_processor_instance.send.return_value = ''

        mock_output_instance = MagicMock()
        captured_target = {}

        def capture_thread(**kwargs):
            captured_target['fn'] = kwargs['target']
            return MagicMock()

        with patch('threading.Thread', side_effect=capture_thread), \
             patch('services.goal_pursuit_processor.GoalPursuitProcessor',
                   return_value=mock_processor_instance), \
             patch('services.output_service.OutputService',
                   return_value=mock_output_instance):
            handle_goal_pursuit('user', {'goal': 'a task'})
            captured_target['fn']()

        _, kwargs = mock_output_instance.enqueue_proactive.call_args
        # Must not be empty
        assert kwargs.get('response')
        assert kwargs['response'] != ''


# ── Background thread behaviour: failure path ─────────────────────────────────

class TestGoalPursuitThreadFailure:
    def _run_failing_target(self, error_message='Something went wrong'):
        """Run the thread target where GoalPursuitProcessor.send() raises (v2 API)."""
        captured_target = {}

        def capture_thread(**kwargs):
            captured_target['fn'] = kwargs['target']
            return MagicMock()

        mock_processor_instance = MagicMock()
        mock_processor_instance.send.side_effect = RuntimeError(error_message)

        mock_output_instance = MagicMock()

        with patch('threading.Thread', side_effect=capture_thread), \
             patch('services.goal_pursuit_processor.GoalPursuitProcessor',
                   return_value=mock_processor_instance), \
             patch('services.output_service.OutputService',
                   return_value=mock_output_instance):
            handle_goal_pursuit('user', {'goal': 'a task'})
            captured_target['fn']()  # must not raise

        return mock_output_instance

    def test_processor_failure_calls_enqueue_proactive(self):
        """On processor failure, an error surface via enqueue_proactive must still happen."""
        mock_output = self._run_failing_target()
        mock_output.enqueue_proactive.assert_called_once()

    def test_processor_failure_source_is_goal_pursuit(self):
        """Error proactive message source must still be 'goal_pursuit'."""
        mock_output = self._run_failing_target()
        _, kwargs = mock_output.enqueue_proactive.call_args
        assert kwargs.get('source') == 'goal_pursuit'

    def test_processor_failure_response_mentions_failure(self):
        """Error message surfaced to user must reference the failure."""
        mock_output = self._run_failing_target(error_message='network timeout')
        _, kwargs = mock_output.enqueue_proactive.call_args
        response = kwargs.get('response', '')
        assert 'failed' in response.lower() or 'error' in response.lower() or 'network timeout' in response

    def test_processor_failure_does_not_raise(self):
        """The background thread target must never propagate exceptions to the caller."""
        # If _run_failing_target didn't raise, the test passes
        self._run_failing_target()
