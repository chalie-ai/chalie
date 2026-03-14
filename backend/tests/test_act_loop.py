"""Tests for ActLoopService — iteration manager, execution, telemetry."""

import time
import pytest
from unittest.mock import patch, MagicMock
from services.act_loop_service import ActLoopService


pytestmark = pytest.mark.unit


def _make_config(**overrides):
    config = {
        'cost_base': 1.0,
    }
    config.update(overrides)
    return config


def _make_result(action_type='recall', status='success', result='found 3 items', exec_time=0.1):
    return {
        'action_type': action_type,
        'status': status,
        'result': result,
        'execution_time': exec_time,
    }


# ── Loop Termination Conditions ──────────────────────────────


class TestCanContinue:

    def test_can_continue_within_timeout(self):
        """Under timeout → can_continue returns True."""
        svc = ActLoopService(_make_config(), cumulative_timeout=60.0)
        svc.start_time = time.time()
        can, reason = svc.can_continue(mode='ACT')
        assert can is True
        assert reason is None

    def test_stops_on_timeout(self):
        """Over timeout → can_continue returns False with 'timeout'."""
        svc = ActLoopService(_make_config(), cumulative_timeout=60.0)
        svc.start_time = time.time() - 61
        can, reason = svc.can_continue(mode='ACT')
        assert can is False
        assert reason == 'timeout'

    def test_stops_on_max_iterations(self):
        """At max_iterations → can_continue returns False."""
        svc = ActLoopService(_make_config(), cumulative_timeout=60.0, max_iterations=3)
        svc.start_time = time.time()
        svc.iteration_number = 3
        can, reason = svc.can_continue(mode='ACT')
        assert can is False
        assert reason == 'max_iterations'

    def test_terminal_mode_stops(self):
        """Non-ACT mode → always stops."""
        svc = ActLoopService(_make_config(), cumulative_timeout=60.0)
        svc.start_time = time.time()
        can, reason = svc.can_continue(mode='RESPOND')
        assert can is False
        assert reason == 'terminal_mode_respond'

    def test_default_max_iterations_is_30(self):
        """Default hard cap is 30."""
        svc = ActLoopService(_make_config())
        assert svc.max_iterations == 30

    def test_continues_below_max_iterations(self):
        """Below max_iterations → continues."""
        svc = ActLoopService(_make_config(), max_iterations=30)
        svc.start_time = time.time()
        svc.iteration_number = 29
        can, reason = svc.can_continue(mode='ACT')
        assert can is True
        assert reason is None


# ── History Token Budget ─────────────────────────────────────


class TestHistoryTokenBudget:

    def test_can_continue_respects_max_history_tokens(self):
        """can_continue returns False when history exceeds token budget."""
        svc = ActLoopService(_make_config())
        svc.start_time = time.time()
        # Add many large results with realistic multi-word text to exceed token budget
        large_text = ' '.join(['word'] * 200)  # 200 words per result
        for i in range(10):
            svc.append_results([_make_result('recall', result=large_text)])
        can, reason = svc.can_continue(max_history_tokens=100)
        assert can is False
        assert reason == 'history_token_budget'

    def test_can_continue_passes_within_token_budget(self):
        """can_continue returns True when history is within token budget."""
        svc = ActLoopService(_make_config())
        svc.start_time = time.time()
        svc.append_results([_make_result('recall', result='short result')])
        can, reason = svc.can_continue(max_history_tokens=4000)
        assert can is True
        assert reason is None

    def test_get_history_context_truncates_when_over_budget(self):
        """get_history_context moves older entries to notes, keeps most recent 3."""
        svc = ActLoopService(_make_config())
        # Add 6 results with realistic multi-word text
        for i in range(6):
            svc.append_results([_make_result('recall', result=f'Result {i}: ' + ' '.join(['data'] * 100))])

        # Very small budget forces pruning
        context = svc.get_history_context(max_history_tokens=50)
        assert "moved to notes" in context
        # Should still have the last 3 results
        assert "Result 3" in context
        assert "Result 4" in context
        assert "Result 5" in context
        # First 3 should be pruned
        assert "Result 0:" not in context

    def test_get_history_context_no_truncation_within_budget(self):
        """get_history_context returns all entries when within budget."""
        svc = ActLoopService(_make_config())
        svc.append_results([_make_result('recall', result='short')])
        context = svc.get_history_context(max_history_tokens=4000)
        assert "truncated" not in context
        assert "[recall]" in context


# ── Concurrent Execution ─────────────────────────────────────


class TestConcurrentExecution:

    @patch('services.act_dispatcher_service.ActDispatcherService')
    def test_single_action_runs_directly(self, mock_dispatcher_cls):
        """Single action dispatches via the dispatcher."""
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch_action.return_value = _make_result('recall')
        mock_dispatcher_cls.return_value = mock_dispatcher

        svc = ActLoopService(_make_config())
        results = svc.execute_actions('test-topic', [{'type': 'recall', 'query': 'test'}])

        assert len(results) == 1
        assert results[0]['action_type'] == 'recall'
        mock_dispatcher.dispatch_action.assert_called_once()

    @patch('services.act_dispatcher_service.ActDispatcherService')
    def test_multiple_actions_return_in_order(self, mock_dispatcher_cls):
        """Multiple actions return results preserving original order."""
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch_action.side_effect = [
            _make_result('recall', result='recall result'),
            _make_result('associate', result='assoc result'),
        ]
        mock_dispatcher_cls.return_value = mock_dispatcher

        svc = ActLoopService(_make_config())
        results = svc.execute_actions('test-topic', [
            {'type': 'recall', 'query': 'test'},
            {'type': 'associate', 'seeds': ['news']},
        ])

        assert len(results) == 2
        assert results[0]['action_type'] == 'recall'
        assert results[1]['action_type'] == 'associate'

    @patch('services.act_dispatcher_service.ActDispatcherService')
    def test_sequential_error_propagates(self, mock_dispatcher_cls):
        """If a sequential action throws, the error propagates up."""
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch_action.side_effect = [
            _make_result('recall'),
            Exception("connection failed"),
        ]
        mock_dispatcher_cls.return_value = mock_dispatcher

        svc = ActLoopService(_make_config())
        with pytest.raises(Exception, match="connection failed"):
            svc.execute_actions('test-topic', [
                {'type': 'recall', 'query': 'test'},
                {'type': 'associate', 'seeds': ['news']},
            ])


# ── History and Append ───────────────────────────────────────


class TestHistoryManagement:

    def test_append_results_updates_history(self):
        """Results added to act_history."""
        svc = ActLoopService(_make_config())
        assert len(svc.act_history) == 0
        results = [_make_result(), _make_result('memorize', result='stored')]
        svc.append_results(results)
        assert len(svc.act_history) == 2

    def test_history_context_formatting(self):
        """get_history_context returns readable string."""
        svc = ActLoopService(_make_config())
        assert svc.get_history_context() == "(none)"
        svc.append_results([_make_result('recall', result='found data')])
        context = svc.get_history_context()
        assert "## Internal Cognitive Actions" in context
        assert "[recall]" in context
        assert "SUCCESS" in context
        assert "found data" in context


# ── Loop Telemetry ───────────────────────────────────────────


class TestLoopTelemetry:

    def test_telemetry_structure(self):
        """get_loop_telemetry returns expected keys and values."""
        svc = ActLoopService(_make_config())
        svc.iteration_number = 3
        svc.act_history = [_make_result(), _make_result(), _make_result()]

        telemetry = svc.get_loop_telemetry()

        assert telemetry['iterations_used'] == 3
        assert telemetry['max_iterations'] == 30
        assert telemetry['actions_total'] == 3
        assert 'elapsed_seconds' in telemetry

    def test_telemetry_no_fatigue_keys(self):
        """Loop telemetry contains no fatigue-related keys."""
        svc = ActLoopService(_make_config())
        telemetry = svc.get_loop_telemetry()
        for key in telemetry:
            assert 'fatigue' not in key.lower(), f"Unexpected fatigue key: {key}"


# ── Soft Nudge Flag ──────────────────────────────────────────


class TestSoftNudge:

    def test_soft_nudge_injected_flag_default_false(self):
        """soft_nudge_injected starts False."""
        svc = ActLoopService(_make_config())
        assert svc.soft_nudge_injected is False

    def test_soft_nudge_flag_can_be_set(self):
        """Orchestrator sets soft_nudge_injected when nudge is emitted."""
        svc = ActLoopService(_make_config())
        svc.soft_nudge_injected = True
        assert svc.soft_nudge_injected is True
