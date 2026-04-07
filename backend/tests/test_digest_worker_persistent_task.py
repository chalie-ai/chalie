"""Tests for digest_worker._handle_persistent_task_result() and routing logic.

Focused on the persistent_task_result shortcut path in digest_worker:
  - source='persistent_task_result' triggers _handle_persistent_task_result
  - correct parameters passed to _run_response_pipeline
  - missing thread_id falls back to resolve_thread
  - graceful error handling
"""

import pytest
from unittest.mock import patch, MagicMock, call


pytestmark = pytest.mark.unit


# ── Routing: source='persistent_task_result' takes the shortcut ───────

class TestDigestWorkerRouting:

    @patch('workers.digest_worker._handle_persistent_task_result')
    def test_persistent_task_result_source_triggers_handler(self, mock_handler):
        """digest_worker routes to _handle_persistent_task_result when source matches."""
        from workers.digest_worker import digest_worker

        mock_handler.return_value = "Task 5 | Surfaced"

        result = digest_worker(
            text="The research is complete.",
            metadata={
                'source': 'persistent_task_result',
                'task_id': 5,
                'thread_id': 'thread-abc',
                'goal': 'Research LLMs',
            }
        )

        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        assert call_args[0][0] == "The research is complete."
        assert call_args[0][1]['source'] == 'persistent_task_result'
        assert call_args[0][1]['task_id'] == 5

    @patch('workers.digest_worker._handle_persistent_task_result')
    @patch('workers.digest_worker.load_configs')
    def test_non_persistent_source_does_not_trigger_handler(self, mock_load_configs, mock_handler):
        """digest_worker does NOT call _handle_persistent_task_result for other sources."""
        from workers.digest_worker import digest_worker

        # Simulate load_configs raising so we don't run the full pipeline
        mock_load_configs.side_effect = Exception("stop pipeline")

        try:
            digest_worker(
                text="Hello there",
                metadata={'source': 'user'}
            )
        except Exception:
            pass

        mock_handler.assert_not_called()

    @patch('workers.digest_worker._handle_cron_tool_result')
    @patch('workers.digest_worker._handle_persistent_task_result')
    def test_cron_tool_source_does_not_trigger_persistent_handler(
        self, mock_pt_handler, mock_cron_handler
    ):
        """cron_tool: source takes its own path — persistent_task handler is not invoked."""
        from workers.digest_worker import digest_worker

        mock_cron_handler.return_value = "cron done"

        digest_worker(
            text="Cron result",
            metadata={'source': 'cron_tool:my_tool'}
        )

        mock_pt_handler.assert_not_called()
        mock_cron_handler.assert_called_once()


# ── _handle_persistent_task_result — pipeline parameters ─────────────

class TestHandlePersistentTaskResult:
    """Tests for _handle_persistent_task_result.

    ConfigService is lazy-imported inside the function so must be patched at
    services.config_service.ConfigService, not workers.digest_worker.ConfigService.
    """

    def _call_handler(self, text, metadata, pipeline_return=None, load_configs_return=None,
                      resolve_thread_id='thread-1', load_configs_side_effect=None):
        """Helper: invoke _handle_persistent_task_result with all its dependencies mocked."""
        from workers.digest_worker import _handle_persistent_task_result

        if pipeline_return is None:
            pipeline_return = ({'generation_time': 1.0}, False)

        if load_configs_return is None:
            load_configs_return = {
                'cortex': {'config': {'max_working_memory_turns': 5}, 'prompt_map': {}}
            }

        mock_pipeline = MagicMock(return_value=pipeline_return)
        mock_wm = MagicMock()

        load_mock = MagicMock(
            side_effect=load_configs_side_effect,
            return_value=load_configs_return if load_configs_side_effect is None else None,
        )

        with patch('workers.digest_worker._run_response_pipeline', mock_pipeline), \
             patch('workers.digest_worker.WorkingMemoryService', mock_wm), \
             patch('workers.digest_worker.load_configs', load_mock), \
             patch('services.config_service.ConfigService.resolve_agent_config', return_value={}), \
             patch('services.config_service.ConfigService.get_agent_prompt', return_value="Summarise."):
            result = _handle_persistent_task_result(text, metadata)

        return result, mock_pipeline, None

    def test_run_response_pipeline_called_with_injected_text(self):
        """_run_response_pipeline receives text that includes the goal and result."""
        result, mock_pipeline, _ = self._call_handler(
            "Found 5 papers on quantum computing.",
            {
                'source': 'persistent_task_result',
                'task_id': 42,
                'thread_id': 'thread-99',
                'goal': 'Research quantum computing',
            },
            resolve_thread_id='thread-99',
        )

        mock_pipeline.assert_called_once()
        pipeline_kwargs = mock_pipeline.call_args.kwargs
        assert 'Research quantum computing' in pipeline_kwargs['text']
        assert 'Found 5 papers on quantum computing.' in pipeline_kwargs['text']

    def test_run_response_pipeline_receives_thread_id(self):
        """thread_id from metadata is passed straight to _run_response_pipeline."""
        _, mock_pipeline, _ = self._call_handler(
            "Result here",
            {'task_id': 7, 'thread_id': 'thread-explicit', 'goal': 'Do something'},
            resolve_thread_id='thread-explicit',
        )

        pipeline_kwargs = mock_pipeline.call_args.kwargs
        assert pipeline_kwargs['thread_id'] == 'thread-explicit'

    def test_missing_thread_id_falls_back_to_default_channel(self):
        """When thread_id is None, defaults to 'persistent_task' channel."""
        _, mock_pipeline, _ = self._call_handler(
            "Result here",
            {'task_id': 8, 'thread_id': None, 'goal': 'Background work'},
        )

        pipeline_kwargs = mock_pipeline.call_args.kwargs
        assert pipeline_kwargs['thread_id'] == 'persistent_task'

    def test_return_value_includes_task_id_and_time(self):
        """Return string includes task_id and surfacing time."""
        result, _, _ = self._call_handler(
            "Done.",
            {'task_id': 99, 'thread_id': 'thread-1', 'goal': 'Some goal'},
            pipeline_return=({'generation_time': 2.75}, False),
        )

        assert '99' in result
        assert '2.75' in result or 'Surfaced' in result

    def test_empty_pipeline_response_returns_empty_marker(self):
        """When pipeline returns is_empty=True, return string signals empty response."""
        result, _, _ = self._call_handler(
            "Done.",
            {'task_id': 12, 'thread_id': 'thread-1', 'goal': 'Goal'},
            pipeline_return=({}, True),
        )

        assert '12' in result
        assert 'empty' in result.lower() or 'Empty' in result

    def test_pipeline_exception_returns_error_string(self):
        """Exception inside the handler is caught and returns an error string — does not raise."""
        result, _, _ = self._call_handler(
            "Done.",
            {'task_id': 20, 'thread_id': None, 'goal': 'Goal'},
            load_configs_side_effect=RuntimeError("configs exploded"),
        )

        assert '20' in result
        assert 'error' in result.lower() or 'ERROR' in result

    def test_log_event_type_is_persistent_task_completed(self):
        """_run_response_pipeline is called with log_event_type='persistent_task_completed'."""
        _, mock_pipeline, _ = self._call_handler(
            "Done.",
            {'task_id': 30, 'thread_id': 'thread-1', 'goal': 'Audit logs'},
        )

        pipeline_kwargs = mock_pipeline.call_args.kwargs
        assert pipeline_kwargs.get('log_event_type') == 'persistent_task_completed'

    def test_topic_includes_task_id(self):
        """The topic passed to _run_response_pipeline includes the task_id."""
        _, mock_pipeline, _ = self._call_handler(
            "Done.",
            {'task_id': 77, 'thread_id': 'thread-1', 'goal': 'Run audit'},
        )

        pipeline_kwargs = mock_pipeline.call_args.kwargs
        assert '77' in pipeline_kwargs['channel']
