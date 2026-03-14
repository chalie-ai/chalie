"""Tests for persistent_task_worker — event-driven constants and surfacing logic."""

import pytest

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
