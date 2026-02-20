"""Tests for ActLoopService — fatigue model, concurrent execution, net value, telemetry."""

import time
import pytest
from unittest.mock import patch, MagicMock
from services.act_loop_service import ActLoopService, ACTION_FATIGUE_COSTS


pytestmark = pytest.mark.unit


def _make_config(**overrides):
    config = {
        'cost_base': 1.0,
        'cost_growth_factor': 1.5,
        'fatigue_budget': 10.0,
        'fatigue_growth_rate': 0.3,
        'fatigue_costs': {},
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


# ── Fatigue Accumulation ────────────────────────────────────


class TestFatigueAccumulation:

    def test_basic_fatigue_accumulation(self):
        """Single action at iteration 0 costs its base cost."""
        svc = ActLoopService(_make_config())
        actions = [_make_result('introspect')]
        added = svc.accumulate_fatigue(actions, iteration_number=0)
        assert added == pytest.approx(0.5)
        assert svc.fatigue == pytest.approx(0.5)

    def test_fatigue_grows_with_iteration(self):
        """Same action costs more at higher iterations (non-linear growth)."""
        svc = ActLoopService(_make_config())
        # Iteration 0: base_cost * (1 + 0.3*0) = 1.0
        added_0 = svc.accumulate_fatigue([_make_result('recall')], iteration_number=0)
        assert added_0 == pytest.approx(1.0)
        # Iteration 2: base_cost * (1 + 0.3*2) = 1.6
        added_2 = svc.accumulate_fatigue([_make_result('recall')], iteration_number=2)
        assert added_2 == pytest.approx(1.6)
        assert svc.fatigue == pytest.approx(2.6)

    def test_multiple_actions_accumulate(self):
        """Multiple actions in one iteration sum their costs."""
        svc = ActLoopService(_make_config())
        actions = [_make_result('introspect'), _make_result('recall')]
        added = svc.accumulate_fatigue(actions, iteration_number=0)
        assert added == pytest.approx(1.5)  # 0.5 + 1.0

    def test_expensive_actions_cost_more(self):
        """delegate costs significantly more than introspect."""
        svc = ActLoopService(_make_config())
        svc.accumulate_fatigue([_make_result('delegate')], iteration_number=0)
        web_fatigue = svc.fatigue
        svc2 = ActLoopService(_make_config())
        svc2.accumulate_fatigue([_make_result('introspect')], iteration_number=0)
        assert web_fatigue > svc2.fatigue * 3

    def test_custom_fatigue_costs_override(self):
        """Config-level fatigue_costs override defaults."""
        config = _make_config(fatigue_costs={'recall': 5.0})
        svc = ActLoopService(config)
        added = svc.accumulate_fatigue([_make_result('recall')], iteration_number=0)
        assert added == pytest.approx(5.0)


# ── Fatigue Stops Loop ──────────────────────────────────────


class TestFatigueStopsLoop:

    def test_fatigue_exhausted_stops_loop(self):
        """When fatigue >= budget, can_continue returns False."""
        svc = ActLoopService(_make_config(fatigue_budget=2.0))
        svc.start_time = time.time()
        svc.fatigue = 2.0
        can, reason = svc.can_continue()
        assert can is False
        assert reason == 'fatigue_exhausted'

    def test_fatigue_below_budget_continues(self):
        """When fatigue < budget, can_continue returns True."""
        svc = ActLoopService(_make_config(fatigue_budget=10.0))
        svc.start_time = time.time()
        svc.fatigue = 5.0
        can, reason = svc.can_continue()
        assert can is True
        assert reason is None

    def test_fatigue_checked_before_timeout(self):
        """Fatigue is checked first — even if timeout not reached."""
        svc = ActLoopService(_make_config(fatigue_budget=1.0))
        svc.start_time = time.time()  # Just started
        svc.fatigue = 1.5  # Over budget
        can, reason = svc.can_continue()
        assert can is False
        assert reason == 'fatigue_exhausted'


# ── Cheap Actions = More Iterations ─────────────────────────


class TestCheapActionsMoreIterations:

    def test_introspect_loop_more_iterations_than_delegate(self):
        """Pure introspect loop should get more iterations than delegate loop before fatigue."""
        budget = 10.0
        config = _make_config(fatigue_budget=budget)

        # Introspect-only loop
        svc1 = ActLoopService(config)
        svc1.start_time = time.time()
        introspect_iters = 0
        while svc1.fatigue < budget:
            svc1.accumulate_fatigue([_make_result('introspect')], iteration_number=introspect_iters)
            introspect_iters += 1

        # Delegate-only loop
        svc2 = ActLoopService(config)
        svc2.start_time = time.time()
        delegate_iters = 0
        while svc2.fatigue < budget:
            svc2.accumulate_fatigue([_make_result('delegate')], iteration_number=delegate_iters)
            delegate_iters += 1

        assert introspect_iters > delegate_iters
        assert introspect_iters >= 7  # ~8-9 expected
        assert delegate_iters <= 5  # ~3-4 expected


# ── Concurrent Execution ────────────────────────────────────


class TestConcurrentExecution:

    @patch('services.act_dispatcher_service.ActDispatcherService')
    def test_single_action_runs_directly(self, mock_dispatcher_cls):
        """Single action doesn't use ThreadPoolExecutor."""
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
    def test_concurrent_error_handled(self, mock_dispatcher_cls):
        """If one concurrent action throws, it's caught as an error result."""
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch_action.side_effect = [
            _make_result('recall'),
            Exception("connection failed"),
        ]
        mock_dispatcher_cls.return_value = mock_dispatcher

        svc = ActLoopService(_make_config())
        results = svc.execute_actions('test-topic', [
            {'type': 'recall', 'query': 'test'},
            {'type': 'associate', 'seeds': ['news']},
        ])

        assert len(results) == 2
        assert results[0]['action_type'] == 'recall'
        assert results[1]['status'] == 'error'
        assert 'connection failed' in results[1]['result']


# ── Net Value Estimator ─────────────────────────────────────


class TestNetValueEstimator:

    def test_success_with_substantial_result(self):
        """Success with >50 chars = 1.0 value."""
        actions = [_make_result('recall', result='A' * 51)]
        value = ActLoopService.estimate_net_value(actions, iteration_number=0)
        assert value == pytest.approx(1.0)

    def test_success_with_minimal_result(self):
        """Success with <=50 chars = 0.3 value."""
        actions = [_make_result('recall', result='short')]
        value = ActLoopService.estimate_net_value(actions, iteration_number=0)
        assert value == pytest.approx(0.3)

    def test_timeout_penalized(self):
        """Timeout = -0.5 value."""
        actions = [_make_result('recall', status='timeout', result='')]
        value = ActLoopService.estimate_net_value(actions, iteration_number=0)
        assert value == pytest.approx(-0.5)

    def test_diminishing_returns(self):
        """Same action at higher iteration yields less value."""
        actions = [_make_result('recall', result='A' * 51)]
        v0 = ActLoopService.estimate_net_value(actions, iteration_number=0)
        v3 = ActLoopService.estimate_net_value(actions, iteration_number=3)
        assert v0 > v3


# ── Fatigue Telemetry ───────────────────────────────────────


class TestFatigueTelemetry:

    def test_telemetry_structure(self):
        """get_fatigue_telemetry returns expected keys and values."""
        svc = ActLoopService(_make_config(fatigue_budget=10.0))
        svc.fatigue = 5.0
        svc.iteration_number = 3
        svc.act_history = [_make_result(), _make_result(), _make_result()]

        telemetry = svc.get_fatigue_telemetry()

        assert telemetry['fatigue_total'] == 5.0
        assert telemetry['fatigue_budget'] == 10.0
        assert telemetry['fatigue_utilization'] == pytest.approx(0.5)
        assert telemetry['iterations_used'] == 3
        assert telemetry['actions_total'] == 3
        assert telemetry['budget_headroom'] == 5.0


# ── Existing Behavior (preserved) ──────────────────────────


class TestExistingBehavior:

    def test_can_continue_within_timeout(self):
        """Under 60s → can_continue returns True."""
        svc = ActLoopService(_make_config(), cumulative_timeout=60.0)
        svc.start_time = time.time()
        can, reason = svc.can_continue(mode='ACT')
        assert can is True
        assert reason is None

    def test_stops_on_timeout(self):
        """Over 60s → can_continue returns False with 'timeout'."""
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
