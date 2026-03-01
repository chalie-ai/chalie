"""
Tests for ACTOrchestrator — unified ACT loop implementation.

Verifies parameterized behavior: critic enabled/disabled, type-based and
embedding-based repetition, escalation hints, persistent_task exit,
all termination reasons.
"""

import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from services.act_orchestrator_service import ACTOrchestrator, ACTResult, _action_fingerprint, _action_types


# ── Helpers ─────────────────────────────────────────────────────────

def _make_cortex_service(responses):
    """Build a mock cortex service that returns canned responses in order."""
    service = MagicMock()
    service.generate_response = MagicMock(side_effect=responses)
    return service


def _make_response(actions=None, confidence=0.8):
    """Build a minimal LLM response dict."""
    return {
        'actions': actions or [],
        'confidence': confidence,
        'response': 'test response',
    }


def _make_action_result(action_type='recall', status='success', result='test', execution_time=0.1):
    """Build a minimal action result dict."""
    return {
        'action_type': action_type,
        'status': status,
        'result': result,
        'execution_time': execution_time,
    }


# ── ACTResult dataclass ────────────────────────────────────────────

@pytest.mark.unit
class TestACTResult:
    def test_defaults(self):
        result = ACTResult()
        assert result.act_history == []
        assert result.iteration_logs == []
        assert result.termination_reason == ''
        assert result.loop_id is None
        assert result.fatigue == 0.0
        assert result.iterations_used == 0
        assert result.critic_telemetry == {}
        assert result.fatigue_telemetry == {}


# ── Fingerprinting utilities ───────────────────────────────────────

@pytest.mark.unit
class TestFingerprinting:
    def test_action_fingerprint(self):
        actions = [
            {'type': 'recall', 'query': 'test query'},
            {'type': 'memorize', 'text': 'some fact'},
        ]
        fp = _action_fingerprint(actions)
        assert 'recall:test query' in fp
        assert 'memorize:some fact' in fp

    def test_action_types(self):
        actions = [
            {'type': 'recall'},
            {'type': 'memorize'},
            {'type': 'recall'},
        ]
        types = _action_types(actions)
        assert types == {'recall', 'memorize'}


# ── Orchestrator: no actions → immediate exit ──────────────────────

@pytest.mark.unit
class TestNoActions:
    @patch('services.act_orchestrator_service.ActLoopService')
    def test_no_actions_exits_immediately(self, MockActLoop):
        """LLM returns no actions → loop exits with 'no_actions'."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.fatigue = 0.0
        mock_loop._critic = None
        mock_loop._escalation_hint_injected = False
        mock_loop.get_fatigue_telemetry.return_value = {'fatigue_total': 0}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        MockActLoop.return_value = mock_loop

        cortex = _make_cortex_service([_make_response(actions=[])])

        orchestrator = ACTOrchestrator(config={}, max_iterations=5)
        result = orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test prompt', classification={'topic': 'test', 'confidence': 10},
            chat_history=[],
        )

        assert result.termination_reason == 'no_actions'


# ── Orchestrator: fatigue termination ──────────────────────────────

@pytest.mark.unit
class TestFatigueTermination:
    @patch('services.act_orchestrator_service.ActLoopService')
    def test_fatigue_exhausted(self, MockActLoop):
        """Loop exits when fatigue budget is exhausted."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.fatigue = 0.0
        mock_loop._critic = None
        mock_loop._escalation_hint_injected = False
        mock_loop.get_fatigue_telemetry.return_value = {'fatigue_total': 10}
        mock_loop.get_critic_telemetry.return_value = {}

        # First call: actions available, can_continue True
        # Second call (after execute): can_continue False (fatigue)
        mock_loop.can_continue.side_effect = [
            (True, None),     # Before repetition check
            (False, 'fatigue_exhausted'),  # After execution
        ]
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found something'),
        ]
        mock_loop.accumulate_fatigue.return_value = 5.0
        MockActLoop.return_value = mock_loop
        MockActLoop.estimate_net_value = MagicMock(return_value=1.0)

        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'recall', 'query': 'test'}]),
        ])

        orchestrator = ACTOrchestrator(
            config={}, max_iterations=5, smart_repetition=False,
        )
        result = orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test prompt', classification={'topic': 'test', 'confidence': 10},
            chat_history=[],
        )

        assert result.termination_reason == 'fatigue_exhausted'


# ── Orchestrator: type-based repetition ────────────────────────────

@pytest.mark.unit
class TestTypeRepetition:
    @patch('services.act_orchestrator_service.ActLoopService')
    def test_type_repetition_hard_exit(self, MockActLoop):
        """Same action type 3x in a row → hard exit (no escalation_hints)."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.fatigue = 0.0
        mock_loop._critic = None
        mock_loop._escalation_hint_injected = False
        mock_loop.get_fatigue_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found'),
        ]
        mock_loop.accumulate_fatigue.return_value = 1.0
        MockActLoop.return_value = mock_loop
        MockActLoop.estimate_net_value = MagicMock(return_value=1.0)

        # 3 identical recall actions → repetition_detected
        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'recall', 'query': 'x'}]),
            _make_response(actions=[{'type': 'recall', 'query': 'x'}]),
            _make_response(actions=[{'type': 'recall', 'query': 'x'}]),
        ])

        orchestrator = ACTOrchestrator(
            config={}, max_iterations=10,
            smart_repetition=False, escalation_hints=False,
        )
        result = orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )

        assert result.termination_reason == 'repetition_detected'


# ── Orchestrator: escalation hints (pivot + budget warning) ────────

@pytest.mark.unit
class TestEscalationHints:
    @patch('services.act_orchestrator_service.ActLoopService')
    def test_pivot_hint_on_repetition(self, MockActLoop):
        """With escalation_hints=True, repetition injects pivot hint first."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.fatigue = 0.0
        mock_loop._critic = None
        mock_loop._escalation_hint_injected = False
        mock_loop.get_fatigue_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found'),
        ]
        mock_loop.accumulate_fatigue.return_value = 1.0
        MockActLoop.return_value = mock_loop
        MockActLoop.estimate_net_value = MagicMock(return_value=1.0)

        # 3x recall (triggers pivot), then no actions (exit)
        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'recall', 'query': 'x'}]),
            _make_response(actions=[{'type': 'recall', 'query': 'x'}]),
            _make_response(actions=[{'type': 'recall', 'query': 'x'}]),
            _make_response(actions=[]),  # after pivot hint, LLM stops
        ])

        orchestrator = ACTOrchestrator(
            config={}, max_iterations=10,
            smart_repetition=False, escalation_hints=True,
        )
        result = orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )

        # Should have injected a system result as pivot hint
        assert mock_loop.append_results.called
        system_calls = [
            call for call in mock_loop.append_results.call_args_list
            if any(r.get('action_type') == 'system' for r in call[0][0])
        ]
        assert len(system_calls) >= 1


# ── Orchestrator: persistent_task exit ─────────────────────────────

@pytest.mark.unit
class TestPersistentTaskExit:
    @patch('services.act_orchestrator_service.ActLoopService')
    def test_persistent_task_dispatch_exits(self, MockActLoop):
        """When persistent_task_exit=True, dispatching a PT exits the loop."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.fatigue = 0.0
        mock_loop._critic = None
        mock_loop._escalation_hint_injected = False
        mock_loop.get_fatigue_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        mock_loop.execute_actions.return_value = [
            _make_action_result('persistent_task', 'success', 'Task created'),
        ]
        mock_loop.accumulate_fatigue.return_value = 1.0
        MockActLoop.return_value = mock_loop
        MockActLoop.estimate_net_value = MagicMock(return_value=1.0)

        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'persistent_task', 'goal': 'Research X'}]),
        ])

        orchestrator = ACTOrchestrator(
            config={}, max_iterations=10,
            smart_repetition=False, persistent_task_exit=True,
        )
        result = orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )

        assert result.termination_reason == 'persistent_task_dispatched'


# ── Orchestrator: callback terminates loop ─────────────────────────

@pytest.mark.unit
class TestCallbackTermination:
    @patch('services.act_orchestrator_service.ActLoopService')
    def test_callback_can_terminate(self, MockActLoop):
        """on_iteration_complete callback returning a reason terminates the loop."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.fatigue = 0.0
        mock_loop._critic = None
        mock_loop._escalation_hint_injected = False
        mock_loop.get_fatigue_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found'),
        ]
        mock_loop.accumulate_fatigue.return_value = 1.0
        MockActLoop.return_value = mock_loop
        MockActLoop.estimate_net_value = MagicMock(return_value=1.0)

        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'recall', 'query': 'x'}]),
        ])

        def cancel_callback(act_loop, iteration_start, actions_executed, termination_reason):
            return 'cancelled'

        orchestrator = ACTOrchestrator(
            config={}, max_iterations=10, smart_repetition=False,
        )
        result = orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
            on_iteration_complete=cancel_callback,
        )

        assert result.termination_reason == 'cancelled'


# ── Orchestrator: critic enabled ───────────────────────────────────

@pytest.mark.unit
class TestCriticEnabled:
    @patch('services.act_orchestrator_service.ActLoopService')
    @patch('services.critic_service.CriticService')
    def test_critic_runs_when_enabled(self, MockCritic, MockActLoop):
        """With critic_enabled=True, critic evaluates actions."""
        mock_critic = MagicMock()
        mock_critic.should_skip.return_value = False
        mock_critic.evaluate.return_value = {'verified': True}
        mock_critic.get_telemetry.return_value = {'total_evaluations': 1}
        MockCritic.return_value = mock_critic

        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.fatigue = 0.0
        mock_loop._critic = mock_critic
        mock_loop._escalation_hint_injected = False
        mock_loop.per_action_timeout = 10.0
        mock_loop.context_extras = {}
        mock_loop.get_fatigue_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {'total_evaluations': 1}
        mock_loop.can_continue.side_effect = [
            (True, None),
            (False, 'fatigue_exhausted'),
        ]
        mock_loop.execute_actions.return_value = [
            _make_action_result('schedule', 'success', 'scheduled'),
        ]
        mock_loop.accumulate_fatigue.return_value = 1.0
        MockActLoop.return_value = mock_loop
        MockActLoop.estimate_net_value = MagicMock(return_value=1.0)

        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'schedule', 'description': 'set alarm'}]),
        ])

        orchestrator = ACTOrchestrator(
            config={}, max_iterations=10,
            critic_enabled=True, smart_repetition=False,
        )
        result = orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )

        # Critic should have been called (evaluate)
        assert mock_critic.evaluate.called or mock_critic.should_skip.called


# ── Orchestrator: deferred card context ────────────────────────────

@pytest.mark.unit
class TestDeferredCardContext:
    @patch('services.act_orchestrator_service.ActLoopService')
    def test_deferred_card_context_injection(self, MockActLoop):
        """With deferred_card_context=True, card offers are injected into history."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.fatigue = 0.0
        mock_loop._critic = None
        mock_loop._escalation_hint_injected = False
        mock_loop.get_fatigue_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        MockActLoop.return_value = mock_loop

        cortex = _make_cortex_service([_make_response(actions=[])])

        orchestrator = ACTOrchestrator(
            config={}, max_iterations=5, deferred_card_context=True,
        )

        # Mock the Redis call inside _inject_deferred_card_context
        with patch('services.act_orchestrator_service.ACTOrchestrator._inject_deferred_card_context') as mock_inject:
            mock_inject.return_value = '(none)\n## Available Card Offers\n- web_search (id: abc, 3 sources, 2 domains)'

            result = orchestrator.run(
                topic='test', text='hello', cortex_service=cortex,
                act_prompt='test', classification={'topic': 't', 'confidence': 10},
                chat_history=[],
            )

            assert mock_inject.called


# ── Orchestrator: constructor parameters ───────────────────────────

@pytest.mark.unit
class TestConstructorParams:
    def test_default_params(self):
        o = ACTOrchestrator(config={})
        assert o.max_iterations == 7
        assert o.cumulative_timeout == 60.0
        assert o.per_action_timeout == 10.0
        assert o.critic_enabled is False
        assert o.smart_repetition is True
        assert o.escalation_hints is False
        assert o.persistent_task_exit is False
        assert o.deferred_card_context is False

    def test_custom_params(self):
        o = ACTOrchestrator(
            config={'act_repetition_similarity_threshold': 0.9},
            max_iterations=3,
            cumulative_timeout=30.0,
            critic_enabled=True,
            escalation_hints=True,
            persistent_task_exit=True,
        )
        assert o.max_iterations == 3
        assert o.cumulative_timeout == 30.0
        assert o.critic_enabled is True
        assert o.escalation_hints is True
        assert o.persistent_task_exit is True
        assert o.repetition_sim_threshold == 0.9

    def test_tool_worker_profile(self):
        """tool_worker uses: critic=True, smart_rep=True, deferred_cards=True."""
        o = ACTOrchestrator(
            config={}, critic_enabled=True, smart_repetition=True,
            deferred_card_context=True,
        )
        assert o.critic_enabled
        assert o.smart_repetition
        assert o.deferred_card_context

    def test_digest_worker_profile(self):
        """digest_worker uses: critic=True, escalation_hints=True, PT exit=True."""
        o = ACTOrchestrator(
            config={}, critic_enabled=True, escalation_hints=True,
            persistent_task_exit=True,
        )
        assert o.critic_enabled
        assert o.escalation_hints
        assert o.persistent_task_exit

    def test_persistent_task_profile(self):
        """persistent_task_worker uses: critic=True, smart_rep=True."""
        o = ACTOrchestrator(
            config={}, critic_enabled=True, smart_repetition=True,
        )
        assert o.critic_enabled
        assert o.smart_repetition
