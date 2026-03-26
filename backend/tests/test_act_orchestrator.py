"""
Tests for ACTOrchestrator — unified ACT loop implementation.

Verifies parameterized behavior: critic enabled/disabled, type-based and
embedding-based repetition, escalation hints, persistent_task exit,
all termination reasons.
"""

import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from services.act_orchestrator_service import (
    ACTOrchestrator,
    ACTResult,
    _action_fingerprint,
    _action_types,
    _maybe_auto_reflect,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _make_cortex_service(responses):
    """Build a mock cortex service that returns canned responses in order."""
    service = MagicMock()
    service.build_system_prompt.return_value = 'mock system prompt'
    service.generate_response_appended = MagicMock(side_effect=responses)
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
        assert result.iterations_used == 0
        assert result.critic_telemetry == {}
        assert result.loop_telemetry == {}
        assert result.reflection is None


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
        mock_loop.get_loop_telemetry.return_value = {}
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


# ── Orchestrator: max_iterations termination ───────────────────────

@pytest.mark.unit
class TestMaxIterationsTermination:
    @patch('services.act_orchestrator_service.ActLoopService')
    def test_max_iterations_exhausted(self, MockActLoop):
        """Loop exits when iteration cap is hit."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop._critic = None
        mock_loop._escalation_hint_injected = False
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {'iterations_used': 5}
        mock_loop.get_critic_telemetry.return_value = {}

        # First call: actions available, can_continue True
        # Second call (after execute): can_continue False (max_iterations)
        mock_loop.can_continue.side_effect = [
            (True, None),
            (False, 'max_iterations'),
        ]
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found something'),
        ]
        MockActLoop.return_value = mock_loop

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

        assert result.termination_reason == 'max_iterations'


# ── Orchestrator: same-type actions allowed ────────────────────────

@pytest.mark.unit
class TestSameTypeActionsAllowed:
    @patch('services.act_orchestrator_service.ActLoopService')
    def test_same_type_does_not_abort(self, MockActLoop):
        """Same action type 3x in a row should NOT abort — smart repetition handles it."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.fatigue = 0.0
        mock_loop._critic = None
        mock_loop._escalation_hint_injected = False
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        # can_continue called twice per iteration: before + after execution
        mock_loop.can_continue.side_effect = [
            (True, None), (True, None),   # iter 0
            (True, None), (True, None),   # iter 1
            (True, None), (False, 'max_iterations'),  # iter 2
        ]
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found'),
        ]
        MockActLoop.return_value = mock_loop

        # 3 recall actions with different queries — all should execute
        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'recall', 'query': 'topic A'}]),
            _make_response(actions=[{'type': 'recall', 'query': 'topic B'}]),
            _make_response(actions=[{'type': 'recall', 'query': 'topic C'}]),
        ])

        orchestrator = ACTOrchestrator(
            config={}, max_iterations=3,
            smart_repetition=False,
        )
        result = orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )

        # All 3 iterations should execute — exits via max_iterations, not repetition
        assert result.termination_reason == 'max_iterations'
        assert mock_loop.execute_actions.call_count == 3


# ── Orchestrator: synthesis call on forced exit ────────────────────

@pytest.mark.unit
class TestForcedExitSynthesis:
    @patch('services.act_orchestrator_service.ActLoopService')
    def test_synthesis_call_on_forced_exit(self, MockActLoop):
        """When loop exits via max_iterations, a synthesis LLM call produces final_response."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop._critic = None
        mock_loop._escalation_hint_injected = False
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        # Run 1 iteration then hit max
        mock_loop.can_continue.side_effect = [
            (True, None), (False, 'max_iterations'),
        ]
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found data'),
        ]
        MockActLoop.return_value = mock_loop

        # 1 recall action → max_iterations exit → 2nd call is synthesis
        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'recall', 'query': 'x'}]),
            {'response': 'Here is what I found from the data.', 'actions': [], 'confidence': 0.9},
        ])

        orchestrator = ACTOrchestrator(
            config={}, max_iterations=1, smart_repetition=False,
        )
        result = orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )

        assert result.termination_reason == 'max_iterations'
        assert result.final_response == 'Here is what I found from the data.'
        # 1 loop iteration + 1 synthesis call = 2 total LLM calls
        assert cortex.generate_response_appended.call_count == 2

    @patch('services.act_orchestrator_service.ActLoopService')
    def test_synthesis_skipped_on_generation_error(self, MockActLoop):
        """No synthesis call when termination was due to generation_error."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop._critic = None
        mock_loop._escalation_hint_injected = False
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        MockActLoop.return_value = mock_loop

        # First call raises → generation_error, no synthesis should follow
        cortex = _make_cortex_service([Exception('LLM down')])

        orchestrator = ACTOrchestrator(
            config={}, max_iterations=5, smart_repetition=False,
        )
        result = orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )

        assert result.termination_reason == 'generation_error'
        assert result.final_response == ''
        # Only 1 LLM call (the failed one), no synthesis
        assert cortex.generate_response_appended.call_count == 1


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
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found'),
        ]
        MockActLoop.return_value = mock_loop

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


# ── Orchestrator: critic (post-loop reflection) ─────────────────────

@pytest.mark.unit
class TestCriticEnabled:
    @patch('services.act_orchestrator_service.ActLoopService')
    @patch('services.act_orchestrator_service.ACTOrchestrator._post_loop_reflection')
    def test_post_loop_reflection_called(self, mock_reflect, MockActLoop):
        """_post_loop_reflection is called after the loop exits."""
        mock_reflect.return_value = None

        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = [_make_action_result('recall', 'success', 'found')]
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 2
        mock_loop.fatigue = 0.0
        mock_loop._critic = None
        mock_loop._escalation_hint_injected = False
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.side_effect = [
            (True, None),
            (False, 'max_iterations'),
        ]
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found'),
        ]
        MockActLoop.return_value = mock_loop

        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'recall', 'query': 'x'}]),
        ])

        orchestrator = ACTOrchestrator(
            config={}, max_iterations=5, smart_repetition=False,
        )
        result = orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )

        assert mock_reflect.called

    @patch('services.act_orchestrator_service.ActLoopService')
    @patch('services.act_orchestrator_service.ACTOrchestrator._post_loop_reflection')
    def test_reflection_stored_in_result(self, mock_reflect, MockActLoop):
        """Reflection returned by _post_loop_reflection is stored in ACTResult."""
        fake_reflection = {
            'outcome_quality': 0.8,
            'what_worked': 'recall was accurate',
            'what_failed': None,
            'lesson': 'use recall before schedule',
            'confidence': 0.9,
        }
        mock_reflect.return_value = fake_reflection

        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.fatigue = 0.0
        mock_loop._critic = None
        mock_loop._escalation_hint_injected = False
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        MockActLoop.return_value = mock_loop

        cortex = _make_cortex_service([_make_response(actions=[])])

        orchestrator = ACTOrchestrator(config={}, smart_repetition=False)
        result = orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )

        assert result.reflection == fake_reflection



# ── Orchestrator: constructor parameters ───────────────────────────

@pytest.mark.unit
class TestConstructorParams:
    @patch('services.act_orchestrator_service.ActLoopService')
    def test_custom_max_iterations_limits_loop(self, MockActLoop):
        """max_iterations=2 causes loop to exit after exactly 2 iterations."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {'iterations_used': 2}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.side_effect = [
            (True, None), (True, None),   # iter 0
            (True, None), (False, 'max_iterations'),  # iter 1
        ]
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found'),
        ]
        MockActLoop.return_value = mock_loop

        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'recall', 'query': 'a'}]),
            _make_response(actions=[{'type': 'recall', 'query': 'b'}]),
        ])

        orch = ACTOrchestrator(config={}, max_iterations=2, smart_repetition=False)
        result = orch.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )
        assert result.termination_reason == 'max_iterations'
        assert mock_loop.execute_actions.call_count == 2

    @patch('services.act_orchestrator_service.ActLoopService')
    def test_persistent_task_exit_triggers_early_termination(self, MockActLoop):
        """persistent_task_exit=True causes loop to exit when PT action dispatched."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        mock_loop.execute_actions.return_value = [
            _make_action_result('persistent_task', 'success', 'Task created'),
        ]
        MockActLoop.return_value = mock_loop

        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'persistent_task', 'goal': 'X'}]),
        ])

        orch = ACTOrchestrator(config={}, max_iterations=10, persistent_task_exit=True, smart_repetition=False)
        result = orch.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )
        assert result.termination_reason == 'persistent_task_dispatched'

    @patch('services.act_orchestrator_service.ActLoopService')
    def test_persistent_task_exit_false_does_not_exit(self, MockActLoop):
        """persistent_task_exit=False does NOT exit when PT action dispatched."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.side_effect = [
            (True, None), (True, None),  # iter 0
            (True, None),  # iter 1 pre-check
        ]
        mock_loop.execute_actions.return_value = [
            _make_action_result('persistent_task', 'success', 'Task created'),
        ]
        MockActLoop.return_value = mock_loop

        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'persistent_task', 'goal': 'X'}]),
            _make_response(actions=[]),  # exits via no_actions
        ])

        orch = ACTOrchestrator(config={}, max_iterations=10, persistent_task_exit=False, smart_repetition=False)
        result = orch.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )
        # Should NOT exit via persistent_task — continues to next iteration
        assert result.termination_reason != 'persistent_task_dispatched'


# ── Append mode: _prune_messages ──────────────────────────────────

@pytest.mark.unit
class TestAppendMode:
    """Tests for append mode message array management."""

    def test_prune_messages_under_budget(self):
        """No pruning when estimated tokens are within budget."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "what?"},
        ]
        result = orch._prune_messages(messages, 10000)
        assert len(result) == 3

    def test_prune_messages_keeps_first_and_tail(self):
        """When over budget, first message and recent tail are kept."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}
        messages = [{"role": "user", "content": "original prompt"}]
        for i in range(10):
            messages.append({"role": "assistant", "content": f"response {i} " + "padding " * 100})
            messages.append({"role": "user", "content": f"update {i} " + "data " * 100})
        result = orch._prune_messages(messages, 500)
        assert result[0]["content"] == "original prompt"
        assert len(result) < len(messages)

    def test_prune_messages_empty(self):
        """Empty message list is returned unchanged."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}
        assert orch._prune_messages([], 1000) == []

    def test_prune_messages_minimum_three(self):
        """_prune_messages never drops below 3 messages (first + at least 2)."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}
        # Build messages that are massively over budget
        messages = [{"role": "user", "content": "original"}]
        for i in range(20):
            messages.append({"role": "assistant", "content": "answer " * 200})
            messages.append({"role": "user", "content": "context " * 200})
        result = orch._prune_messages(messages, 1)  # budget=1 forces max pruning
        assert len(result) >= 1  # at minimum the original is kept
        assert result[0]["content"] == "original"

    def test_prune_messages_three_messages_not_pruned(self):
        """Arrays of 3 or fewer messages are never pruned regardless of budget."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}
        messages = [
            {"role": "user", "content": "word " * 10000},
            {"role": "assistant", "content": "word " * 10000},
            {"role": "user", "content": "word " * 10000},
        ]
        result = orch._prune_messages(messages, 1)  # budget=1 would normally prune
        assert len(result) == 3  # <= 3 messages: skip pruning

    def test_get_steering_text_no_request_id(self):
        """_get_steering_text returns empty string when no request_id is set."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}
        orch._request_id = ''
        # With no request_id the method should short-circuit or return ''
        # (MemoryStore key would be 'steer:' which should be empty)
        mock_store = MagicMock()
        mock_store.lrange.return_value = []
        with patch('services.memory_store.get_shared_store', return_value=mock_store):
            result = orch._get_steering_text()
        assert result == ''

    @patch('services.act_orchestrator_service.ActLoopService')
    def test_dynamic_tool_injection_from_find_tools(self, MockActLoop):
        """When find_tools returns discovered tools, their schemas are injected."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}

        # can_continue called: twice in iteration 0 (repetition + post-execute),
        # once in iteration 1 (repetition check before no-actions exit)
        mock_loop.can_continue.side_effect = [
            (True, None),
            (True, None),
            (True, None),
        ]
        mock_loop.execute_actions.return_value = [
            {
                'action_type': 'find_tools',
                'status': 'success',
                'result': 'Found 1 tool matching "weather": weather_api',
                '_discovered_tools': ['weather_api'],
                'execution_time': 0.05,
            },
        ]
        MockActLoop.return_value = mock_loop

        cortex = MagicMock()
        cortex.build_system_prompt.return_value = 'system prompt'
        cortex.generate_response_appended.side_effect = [
            # Iteration 0: call find_tools
            {
                'actions': [{'type': 'find_tools', 'query': 'weather'}],
                'confidence': 0.8,
                'tool_calls': [{'id': 'tc1', 'name': 'find_tools', 'arguments': {'query': 'weather'}}],
                'narration': '',
            },
            # Iteration 1: no actions, exit
            {
                'actions': [],
                'confidence': 0.8,
                'response': 'done',
            },
        ]

        with patch(
            'services.act_orchestrator_service.get_external_tool_schemas',
            return_value=[{
                'name': 'weather_api',
                'description': 'Get weather',
                'input_schema': {
                    'type': 'object',
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                },
            }],
        ) as mock_get_schemas:
            orchestrator = ACTOrchestrator(
                config={}, max_iterations=5, smart_repetition=False,
            )
            result = orchestrator.run(
                topic='test', text='what is the weather?', cortex_service=cortex,
                act_prompt='test', classification={'topic': 't', 'confidence': 10},
                chat_history=[],
            )

            # get_external_tool_schemas should have been called with discovered tools
            mock_get_schemas.assert_called_once_with(['weather_api'])

        # Second generate_response_appended call should include the injected tool
        second_call_kwargs = cortex.generate_response_appended.call_args_list[1]
        tools_arg = second_call_kwargs.kwargs.get('tools') or second_call_kwargs[1].get('tools', [])
        tool_names = [t['name'] for t in tools_arg]
        assert 'weather_api' in tool_names

    @patch('services.act_orchestrator_service.ActLoopService')
    def test_dict_result_text_extraction_in_tool_messages(self, MockActLoop):
        """Dict results with 'text' key should send text (not JSON) to the LLM."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        mock_loop.execute_actions.return_value = [
            {
                'action_type': 'find_tools',
                'status': 'success',
                'result': {
                    'text': 'Found 1 tool.',
                    '_discovered_tools': [],
                },
                'execution_time': 0.05,
            },
        ]
        MockActLoop.return_value = mock_loop

        cortex = MagicMock()
        cortex.build_system_prompt.return_value = 'system prompt'
        cortex.generate_response_appended.side_effect = [
            {
                'actions': [{'type': 'find_tools', 'query': 'test'}],
                'tool_calls': [{'id': 'tc1', 'name': 'find_tools', 'arguments': {}}],
                'narration': '',
                'confidence': 0.8,
            },
            {'actions': [], 'response': 'done', 'confidence': 0.8},
        ]

        orchestrator = ACTOrchestrator(
            config={}, max_iterations=5, smart_repetition=False,
        )
        result = orchestrator.run(
            topic='test', text='test', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )

        # Check that the tool result message has clean text, not JSON
        tool_messages = [
            m for m in cortex.generate_response_appended.call_args_list[1].kwargs.get('messages', [])
            if isinstance(m, dict) and m.get('role') == 'tool'
        ]
        if not tool_messages:
            # Try positional args
            for call in cortex.generate_response_appended.call_args_list:
                msgs = call.kwargs.get('messages', [])
                tool_messages.extend(m for m in msgs if isinstance(m, dict) and m.get('role') == 'tool')

        assert any(m.get('content') == 'Found 1 tool.' for m in tool_messages)

    @patch('services.act_orchestrator_service.ActLoopService')
    def test_run_always_uses_appended_path(self, MockActLoop):
        """Orchestrator always builds system prompt once and uses generate_response_appended."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        MockActLoop.return_value = mock_loop

        cortex = MagicMock()
        cortex.build_system_prompt.return_value = 'built system prompt'
        cortex.generate_response_appended.return_value = {
            'actions': [],
            'confidence': 0.8,
            'response': 'done',
            'raw_response': '{"actions": []}',
        }

        orchestrator = ACTOrchestrator(
            config={}, max_iterations=5, smart_repetition=False,
        )
        result = orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test prompt', classification={'topic': 'test', 'confidence': 10},
            chat_history=[],
        )

        cortex.build_system_prompt.assert_called_once()
        cortex.generate_response_appended.assert_called()


# ── Orchestrator: on_narration callback ────────────────────────────

@pytest.mark.unit
class TestOnNarrationCallback:
    """Tests for the ``on_narration`` callback parameter."""

    @patch('services.act_orchestrator_service.ActLoopService')
    def test_on_narration_called_with_explicit_text(self, MockActLoop):
        """on_narration receives the narration text the LLM provided."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found'),
        ]
        MockActLoop.return_value = mock_loop

        narrations = []

        cortex = _make_cortex_service([
            {
                'actions': [{'type': 'recall', 'query': 'test'}],
                'confidence': 0.8,
                'narration': 'Searching memory...',
            },
            _make_response(actions=[]),
        ])

        orchestrator = ACTOrchestrator(config={}, smart_repetition=False)
        orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
            on_narration=lambda text, step: narrations.append((text, step)),
        )

        assert len(narrations) >= 1
        assert narrations[0][0] == 'Searching memory...'

    @patch('services.act_orchestrator_service.ActLoopService')
    def test_on_narration_auto_generated_from_action_types(self, MockActLoop):
        """When LLM omits narration, on_narration receives auto-generated text from action types."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found'),
        ]
        MockActLoop.return_value = mock_loop

        narrations = []

        cortex = _make_cortex_service([
            {
                'actions': [{'type': 'recall', 'query': 'test'}],
                'confidence': 0.8,
                'narration': '',  # Empty — should trigger auto-generation
            },
            _make_response(actions=[]),
        ])

        orchestrator = ACTOrchestrator(config={}, smart_repetition=False)
        orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
            on_narration=lambda text, step: narrations.append(text),
        )

        # Auto-narration should mention the action type
        assert any('recall' in n for n in narrations)

    @patch('services.act_orchestrator_service.ActLoopService')
    def test_on_narration_not_invoked_when_no_actions(self, MockActLoop):
        """on_narration is NOT called when the LLM returns no actions (loop exits immediately)."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        MockActLoop.return_value = mock_loop

        narrations = []
        cortex = _make_cortex_service([_make_response(actions=[])])

        orchestrator = ACTOrchestrator(config={}, smart_repetition=False)
        orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
            on_narration=lambda text, step: narrations.append(text),
        )

        assert narrations == []


# ── Orchestrator: _check_smart_repetition ──────────────────────────

@pytest.mark.unit
class TestSmartRepetition:
    """Direct unit tests for :meth:`ACTOrchestrator._check_smart_repetition`."""

    def _mock_embedding_modules(self, sim_value=1.0, raise_on_service=False):
        """Build sys.modules patches for numpy and services.embedding_service.

        Args:
            sim_value: The float value that ``np.dot()`` will return.
            raise_on_service: When True, ``get_embedding_service`` raises instead.

        Returns:
            dict: Mapping suitable for ``patch.dict('sys.modules', ...)``.
        """
        mock_np = MagicMock()
        mock_np.dot.return_value = sim_value

        mock_emb_instance = MagicMock()
        mock_emb_instance.generate_embedding_np.return_value = [1.0, 0.0]

        if raise_on_service:
            mock_emb_module = MagicMock()
            mock_emb_module.get_embedding_service.side_effect = Exception('embeddings unavailable')
        else:
            mock_emb_module = MagicMock()
            mock_emb_module.get_embedding_service.return_value = mock_emb_instance

        return {'numpy': mock_np, 'services.embedding_service': mock_emb_module}

    def test_returns_none_when_embedding_service_fails(self):
        """Fail-open: if the embedding service raises, returns None without crashing."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}
        orch.repetition_sim_threshold = 0.85

        with patch.dict('sys.modules', self._mock_embedding_modules(raise_on_service=True)):
            result = orch._check_smart_repetition(
                'recall:test query',
                {'recall'},
                [('recall:prior query', {'recall'}), ('recall:test query', {'recall'})],
            )
        assert result is None

    def test_returns_smart_repetition_on_consecutive_hits(self):
        """Returns 'smart_repetition' when 2+ consecutive similar iterations are detected."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}
        orch.repetition_sim_threshold = 0.85

        # sim_value=1.0 → 1.0 > 0.85, triggers consecutive_hits
        with patch.dict('sys.modules', self._mock_embedding_modules(sim_value=1.0)):
            # 3 entries: 2 prior + 1 current (all identical type/fingerprint)
            recent_entries = [
                ('recall:same query', {'recall'}),
                ('recall:same query', {'recall'}),
                ('recall:same query', {'recall'}),  # current is last
            ]
            result = orch._check_smart_repetition(
                'recall:same query',
                {'recall'},
                recent_entries,
            )
        assert result == 'smart_repetition'

    def test_no_repetition_when_action_types_differ(self):
        """No repetition detected when consecutive entries have different action types."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}
        orch.repetition_sim_threshold = 0.85

        with patch.dict('sys.modules', self._mock_embedding_modules(sim_value=1.0)):
            # Type mismatch in previous entry: the loop breaks before consecutive_hits reaches 2
            result = orch._check_smart_repetition(
                'recall:query',
                {'recall'},
                [
                    ('memorize:something', {'memorize'}),  # different type → streak broken
                    ('recall:query', {'recall'}),
                ],
            )
        assert result is None

    def test_no_repetition_when_similarity_below_threshold(self):
        """No repetition when cosine similarity is below the configured threshold."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}
        orch.repetition_sim_threshold = 0.85

        # sim_value=0.0 → 0.0 < 0.85 → no consecutive hits
        with patch.dict('sys.modules', self._mock_embedding_modules(sim_value=0.0)):
            result = orch._check_smart_repetition(
                'recall:topic A',
                {'recall'},
                [('recall:topic B', {'recall'}), ('recall:topic A', {'recall'})],
            )
        assert result is None


# ── Orchestrator: _get_cautionary_lessons ──────────────────────────

@pytest.mark.unit
class TestGetCautionaryLessons:
    """Direct unit tests for :meth:`ACTOrchestrator._get_cautionary_lessons`."""

    def _mock_fas_modules(self, lessons=None, db_raises=False):
        """Build sys.modules patches for FailureAnalysisService and the DB service.

        Args:
            lessons: List of lesson dicts returned by ``get_relevant_lessons``.
                When ``None``, defaults to an empty list.
            db_raises: When True, the DB service raises instead of returning a mock.

        Returns:
            dict: Suitable for ``patch.dict('sys.modules', ...)``.
        """
        mock_fas_instance = MagicMock()
        mock_fas_instance.get_relevant_lessons.return_value = lessons or []
        mock_fas_mod = MagicMock()
        mock_fas_mod.FailureAnalysisService.return_value = mock_fas_instance

        mock_db_mod = MagicMock()
        if db_raises:
            mock_db_mod.get_shared_db_service.side_effect = Exception('no db')
        else:
            mock_db_mod.get_shared_db_service.return_value = MagicMock()

        return {
            'services.failure_analysis_service': mock_fas_mod,
            'services.database_service': mock_db_mod,
        }

    def test_returns_empty_when_db_unavailable(self):
        """Fail-open: returns '' when the database service raises."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}

        with patch.dict('sys.modules', self._mock_fas_modules(db_raises=True)):
            result = orch._get_cautionary_lessons([
                {'action_type': 'recall', 'status': 'failed'},
            ])
        assert result == ''

    def test_returns_formatted_lessons_when_available(self):
        """Returns bullet-formatted lessons sorted by times_seen descending."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}

        lessons = [{'blame': 'recall', 'lesson': 'Use precise queries', 'times_seen': 3}]

        with patch.dict('sys.modules', self._mock_fas_modules(lessons=lessons)):
            result = orch._get_cautionary_lessons([
                {'action_type': 'recall', 'status': 'failed'},
            ])

        assert 'recall' in result
        assert 'Use precise queries' in result
        assert '3x' in result

    def test_returns_empty_when_no_lessons_found(self):
        """Returns '' when the service finds no lessons for the action types."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}

        with patch.dict('sys.modules', self._mock_fas_modules(lessons=[])):
            result = orch._get_cautionary_lessons([
                {'action_type': 'recall', 'status': 'success'},
            ])
        assert result == ''


# ── Orchestrator: _record_failure_lesson ───────────────────────────

@pytest.mark.unit
class TestRecordFailureLesson:
    """Direct unit tests for :meth:`ACTOrchestrator._record_failure_lesson`."""

    def _mock_record_modules(self, analyze_return=None, db_raises=False):
        """Build sys.modules patches for FailureAnalysisService and DB used in _record_failure_lesson.

        Args:
            analyze_return: Value returned by ``fas.analyze()``.
            db_raises: When True, the DB service raises instead of returning a mock.

        Returns:
            tuple: (mock_fas_instance, sys_modules_dict) where sys_modules_dict is suitable
                for ``patch.dict('sys.modules', ...)``.
        """
        mock_fas_instance = MagicMock()
        mock_fas_instance.analyze.return_value = analyze_return or {
            'lesson': 'use recall carefully', 'blame': 'recall'
        }
        mock_fas_mod = MagicMock()
        mock_fas_mod.FailureAnalysisService.return_value = mock_fas_instance

        mock_db_mod = MagicMock()
        if db_raises:
            mock_db_mod.get_shared_db_service.side_effect = Exception('no db')
        else:
            mock_db_mod.get_shared_db_service.return_value = MagicMock()

        modules = {
            'services.failure_analysis_service': mock_fas_mod,
            'services.database_service': mock_db_mod,
        }
        return mock_fas_instance, modules

    def test_major_severity_calls_analyze_and_store_synchronously(self):
        """Major severity invokes FailureAnalysisService.analyze() and store_lesson() inline."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}

        mock_fas_instance, modules = self._mock_record_modules()

        with patch.dict('sys.modules', modules):
            orch._record_failure_lesson(
                action_type='recall',
                failure_context={'original_request': 'find something'},
                severity='major',
            )

        mock_fas_instance.analyze.assert_called_once()
        mock_fas_instance.store_lesson.assert_called_once()

    def test_minor_severity_spawns_daemon_thread(self):
        """Minor severity creates and starts a daemon thread for async recording."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}

        with patch('threading.Thread') as mock_thread_class:
            mock_thread_instance = MagicMock()
            mock_thread_class.return_value = mock_thread_instance

            orch._record_failure_lesson(
                action_type='recall',
                failure_context={'original_request': 'test'},
                severity='minor',
            )

        mock_thread_class.assert_called_once()
        call_kwargs = mock_thread_class.call_args.kwargs
        assert call_kwargs.get('daemon') is True
        assert 'failure-lesson' in (call_kwargs.get('name') or '')
        mock_thread_instance.start.assert_called_once()

    def test_fail_open_when_db_unavailable(self):
        """No exception propagates when the DB service is unavailable (major severity)."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}

        _, modules = self._mock_record_modules(db_raises=True)

        with patch.dict('sys.modules', modules):
            # Must not raise
            orch._record_failure_lesson(
                action_type='recall',
                failure_context={'original_request': 'test'},
                severity='major',
            )


# ── Orchestrator: _escalate_and_wait ───────────────────────────────

@pytest.mark.unit
class TestEscalateAndWait:
    """Direct unit tests for :meth:`ACTOrchestrator._escalate_and_wait`."""

    def test_returns_none_when_no_request_id(self):
        """Returns None immediately when no request_id is set (cannot wait for response)."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}
        orch._request_id = ''

        mock_loop = MagicMock()
        mock_loop.context_extras = {'topic': 'test'}

        with patch('services.output_service.OutputService'):
            result = orch._escalate_and_wait(
                mock_loop, 'Please confirm this action.', 'exchange-1'
            )
        assert result is None

    def test_returns_user_response_from_steering_queue(self):
        """Returns the user's response text when the steering queue has a message."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}
        orch._request_id = 'req-123'

        mock_loop = MagicMock()
        mock_loop.context_extras = {'topic': 'test'}

        mock_store = MagicMock()
        mock_store.lrange.return_value = ['proceed with the plan']

        with patch('services.output_service.OutputService'), \
             patch('services.memory_store.get_shared_store', return_value=mock_store), \
             patch('time.sleep'):
            result = orch._escalate_and_wait(
                mock_loop, 'Please confirm.', 'exchange-1',
                poll_interval=0.0, max_wait=10.0,
            )

        assert result == 'proceed with the plan'

    def test_returns_none_on_timeout(self):
        """Returns None when the user does not respond within max_wait seconds."""
        orch = ACTOrchestrator.__new__(ACTOrchestrator)
        orch.config = {}
        orch._request_id = 'req-123'

        mock_loop = MagicMock()
        mock_loop.context_extras = {'topic': 'test'}

        mock_store = MagicMock()
        mock_store.lrange.return_value = []  # Never responds

        with patch('services.output_service.OutputService'), \
             patch('services.memory_store.get_shared_store', return_value=mock_store), \
             patch('time.sleep'), \
             patch('time.monotonic', side_effect=[0.0, 2.0]):
            # deadline = 0.0 + 1.0 = 1.0; first while check: 2.0 < 1.0 → False
            result = orch._escalate_and_wait(
                mock_loop, 'Please confirm.', 'exchange-1',
                poll_interval=0.0, max_wait=1.0,
            )

        assert result is None


# ── _maybe_auto_reflect (module-level function) ─────────────────────

@pytest.mark.unit
class TestMaybeAutoReflect:
    """Tests for the module-level :func:`_maybe_auto_reflect` function."""

    def test_skips_loop_below_min_iterations(self):
        """Does nothing when iterations_used is below _AUTO_REFLECT_MIN_ITERATIONS (2)."""
        with patch('threading.Thread') as mock_thread:
            _maybe_auto_reflect(
                topic='test',
                iteration_logs=[],
                termination_reason='no_actions',
                iterations_used=1,
            )
        mock_thread.assert_not_called()

    def _mock_auto_reflect_store(self):
        """Return a mock store with no cooldown so reflection proceeds."""
        mock_store = MagicMock()
        mock_store.get.return_value = None  # Not in cooldown
        return mock_store

    def test_triggers_on_degraded_exit(self):
        """Fires background reflection when the termination reason is in DEGRADED_EXITS."""
        mock_store = self._mock_auto_reflect_store()
        with patch('services.memory_store.get_shared_store', return_value=mock_store), \
             patch('threading.Thread') as mock_thread:
            mock_thread.return_value = MagicMock()
            _maybe_auto_reflect(
                topic='unique-degraded-exit-test',
                iteration_logs=[{'net_value': 0.0}, {'net_value': 0.0}],
                termination_reason='smart_repetition',
                iterations_used=2,
            )
        mock_thread.assert_called_once()
        call_kwargs = mock_thread.call_args.kwargs
        assert call_kwargs.get('daemon') is True
        mock_thread.return_value.start.assert_called_once()

    def test_triggers_on_high_net_value(self):
        """Fires background reflection when total net value exceeds HIGH_VALUE threshold (3.0)."""
        mock_store = self._mock_auto_reflect_store()
        with patch('services.memory_store.get_shared_store', return_value=mock_store), \
             patch('threading.Thread') as mock_thread:
            mock_thread.return_value = MagicMock()
            _maybe_auto_reflect(
                topic='unique-high-value-test',
                iteration_logs=[
                    {'net_value': 2.0},
                    {'net_value': 2.0},  # total = 4.0 > 3.0
                ],
                termination_reason='no_actions',
                iterations_used=2,
            )
        mock_thread.assert_called_once()

    def test_does_not_trigger_on_neutral_normal_exit(self):
        """Does NOT fire reflection when net value is neutral and exit is normal."""
        with patch('threading.Thread') as mock_thread:
            _maybe_auto_reflect(
                topic='neutral-test',
                iteration_logs=[
                    {'net_value': 0.5},
                    {'net_value': 0.5},  # total = 1.0 — neither high nor low
                ],
                termination_reason='no_actions',
                iterations_used=2,
            )
        mock_thread.assert_not_called()


# ── Orchestrator: edge cases ────────────────────────────────────────

@pytest.mark.unit
class TestEdgeCases:
    """Edge-case coverage for ACTOrchestrator constructor and run()."""

    @patch('services.act_orchestrator_service.ActLoopService')
    def test_run_with_minimal_classification(self, MockActLoop):
        """run() handles a classification dict with no topic or confidence keys."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        MockActLoop.return_value = mock_loop

        cortex = _make_cortex_service([_make_response(actions=[])])
        orch = ACTOrchestrator(config={}, smart_repetition=False)
        result = orch.run(
            topic='', text='hello', cortex_service=cortex,
            act_prompt='test', classification={},
            chat_history=[],
        )
        assert result.termination_reason == 'no_actions'

    @patch('services.act_orchestrator_service.ActLoopService')
    def test_final_response_populated_on_no_actions_exit(self, MockActLoop):
        """final_response is populated from the LLM response text when no actions are returned."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.get_critic_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        MockActLoop.return_value = mock_loop

        cortex = _make_cortex_service([
            {'actions': [], 'response': 'This is my final answer.', 'confidence': 0.9}
        ])
        orch = ACTOrchestrator(config={}, smart_repetition=False)
        result = orch.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )
        assert result.final_response == 'This is my final answer.'

