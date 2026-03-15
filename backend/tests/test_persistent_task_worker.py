"""Tests for persistent_task_worker and persistent_task_skill — event-driven constants, surfacing logic, and skill actions."""

import pytest
from unittest.mock import patch, MagicMock

from workers.persistent_task_worker import (
    EXECUTE_QUEUE_KEY,
    BLPOP_TIMEOUT,
)


pytestmark = pytest.mark.unit


# ── Constants ────────────────────────────────────────────────────────

class TestConstants:

    def test_execute_queue_key(self):
        assert EXECUTE_QUEUE_KEY == "persistent_task:execute"

    def test_blpop_timeout(self):
        """Heartbeat fallback fires every 5 minutes."""
        assert BLPOP_TIMEOUT == 300


# ── Surfacing logic ──────────────────────────────────────────────────

class TestSurfacingLogic:
    """
    Surfacing conditions (inlined in _process_task):
      should_surface = (cycles_completed == 2) or (coverage_jump > 0.15)
    """

    @staticmethod
    def _should_surface(cycles_completed, coverage_jump):
        return (
            (cycles_completed == 2) or
            (coverage_jump > 0.15)
        )

    def test_surfaces_at_cycle_2(self):
        assert self._should_surface(cycles_completed=2, coverage_jump=0.0) is True

    def test_surfaces_when_coverage_jump_exceeds_threshold(self):
        assert self._should_surface(cycles_completed=5, coverage_jump=0.20) is True

    def test_no_surface_when_neither_condition_met(self):
        assert self._should_surface(cycles_completed=3, coverage_jump=0.05) is False

    def test_no_surface_on_first_cycle(self):
        assert self._should_surface(cycles_completed=1, coverage_jump=0.0) is False

    def test_exact_threshold_does_not_surface(self):
        """coverage_jump must be strictly greater than 0.15."""
        assert self._should_surface(cycles_completed=5, coverage_jump=0.15) is False


# ── Chat history builder ──────────────────────────────────────────────

class TestTaskChatHistory:

    def test_empty_task_returns_empty_history(self):
        from workers.persistent_task_worker import _build_task_chat_history
        task = {'progress': {}, 'result': ''}
        assert _build_task_chat_history(task) == []

    def test_includes_last_summary(self):
        from workers.persistent_task_worker import _build_task_chat_history
        task = {'progress': {'last_summary': 'Found 3 bugs', 'cycles_completed': 2}, 'result': ''}
        history = _build_task_chat_history(task)
        assert len(history) == 1
        assert 'Found 3 bugs' in history[0]['content']
        assert 'cycle 2' in history[0]['content'].lower()

    def test_includes_intermediate_results(self):
        from workers.persistent_task_worker import _build_task_chat_history
        task = {'progress': {}, 'result': 'Some intermediate findings here'}
        history = _build_task_chat_history(task)
        assert len(history) == 1
        assert 'intermediate' in history[0]['content'].lower()

    def test_truncates_long_results(self):
        from workers.persistent_task_worker import _build_task_chat_history
        task = {'progress': {}, 'result': 'x' * 5000}
        history = _build_task_chat_history(task)
        assert len(history[0]['content']) < 4000

    def test_both_summary_and_results(self):
        from workers.persistent_task_worker import _build_task_chat_history
        task = {'progress': {'last_summary': 'Progress made', 'cycles_completed': 1}, 'result': 'Details here'}
        history = _build_task_chat_history(task)
        assert len(history) == 2


# ── Skill: complete action ──────────────────────────────────────────

class TestSkillCompleteAction:

    @patch('services.innate_skills.persistent_task_skill._surface_completion_from_skill')
    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_complete_happy_path(self, mock_account, mock_get_svc, mock_surface):
        from services.innate_skills.persistent_task_skill import _complete

        mock_service = MagicMock()
        mock_service.get_task.return_value = {
            'id': 42, 'goal': 'Research quantum computing', 'thread_id': 't1',
            'progress': {'coverage_estimate': 0.8},
        }
        mock_service.get_active_tasks.return_value = []
        mock_get_svc.return_value = mock_service

        result = _complete('topic', {'task_id': '42', 'result': 'Found 5 papers.'})

        mock_service.complete_task.assert_called_once_with(42, 'Found 5 papers.', None)
        mock_surface.assert_called_once()
        assert '42' in result
        assert 'completed' in result.lower()

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_complete_task_not_found(self, mock_account, mock_get_svc):
        from services.innate_skills.persistent_task_skill import _complete

        mock_service = MagicMock()
        mock_service.get_task.return_value = None
        mock_service.get_active_tasks.return_value = []
        mock_get_svc.return_value = mock_service

        result = _complete('topic', {'task_id': '99'})
        assert 'not found' in result.lower()

    @patch('services.innate_skills.persistent_task_skill._surface_completion_from_skill')
    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_complete_uses_context_extras_fallback(self, mock_account, mock_get_svc, mock_surface):
        """When no task_id in params, falls back to persistent_task_id from context_extras."""
        from services.innate_skills.persistent_task_skill import _complete

        mock_service = MagicMock()
        mock_service.get_task.return_value = {
            'id': 7, 'goal': 'Analyze logs', 'thread_id': 't2',
            'progress': {},
        }
        mock_service.get_active_tasks.return_value = []
        mock_get_svc.return_value = mock_service

        result = _complete('topic', {'persistent_task_id': 7, 'result': 'Done.'})

        mock_service.complete_task.assert_called_once_with(7, 'Done.', None)
        assert 'completed' in result.lower()

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_complete_no_task_id_at_all(self, mock_account, mock_get_svc):
        from services.innate_skills.persistent_task_skill import _complete

        mock_service = MagicMock()
        mock_service.get_active_tasks.return_value = []
        mock_get_svc.return_value = mock_service

        result = _complete('topic', {})
        assert 'could not identify' in result.lower()


# ── Skill: progress action ──────────────────────────────────────────

class TestSkillProgressAction:

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_progress_updates_coverage(self, mock_account, mock_get_svc):
        from services.innate_skills.persistent_task_skill import _progress_update

        mock_service = MagicMock()
        mock_service.get_task.return_value = {
            'id': 10, 'goal': 'Build widget', 'progress': {'coverage_estimate': 0.2},
        }
        mock_service.get_active_tasks.return_value = []
        mock_get_svc.return_value = mock_service

        result = _progress_update('topic', {'task_id': '10', 'coverage': 0.6, 'summary': 'Halfway there'})

        mock_service.checkpoint.assert_called_once()
        call_kwargs = mock_service.checkpoint.call_args
        progress_arg = call_kwargs[1]['progress'] if 'progress' in call_kwargs[1] else call_kwargs[0][1]
        assert progress_arg['coverage_estimate'] == 0.6
        assert progress_arg['last_summary'] == 'Halfway there'
        assert '60%' in result

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_progress_task_not_found(self, mock_account, mock_get_svc):
        from services.innate_skills.persistent_task_skill import _progress_update

        mock_service = MagicMock()
        mock_service.get_task.return_value = None
        mock_service.get_active_tasks.return_value = []
        mock_get_svc.return_value = mock_service

        result = _progress_update('topic', {'task_id': '99', 'coverage': 0.5})
        assert 'not found' in result.lower()

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_progress_clamps_coverage(self, mock_account, mock_get_svc):
        """Coverage values outside 0.0-1.0 are clamped."""
        from services.innate_skills.persistent_task_skill import _progress_update

        mock_service = MagicMock()
        mock_service.get_task.return_value = {
            'id': 10, 'goal': 'Build widget', 'progress': {},
        }
        mock_service.get_active_tasks.return_value = []
        mock_get_svc.return_value = mock_service

        _progress_update('topic', {'task_id': '10', 'coverage': 1.5})
        call_kwargs = mock_service.checkpoint.call_args
        progress_arg = call_kwargs[1]['progress'] if 'progress' in call_kwargs[1] else call_kwargs[0][1]
        assert progress_arg['coverage_estimate'] == 1.0

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_progress_uses_context_extras_fallback(self, mock_account, mock_get_svc):
        from services.innate_skills.persistent_task_skill import _progress_update

        mock_service = MagicMock()
        mock_service.get_task.return_value = {
            'id': 5, 'goal': 'Scan docs', 'progress': {'coverage_estimate': 0.1},
        }
        mock_service.get_active_tasks.return_value = []
        mock_get_svc.return_value = mock_service

        result = _progress_update('topic', {'persistent_task_id': 5, 'coverage': 0.4})
        mock_service.checkpoint.assert_called_once()
        assert 'updated' in result.lower()
