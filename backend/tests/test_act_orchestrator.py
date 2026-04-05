"""
Tests for ACTOrchestrator — unified ACT loop implementation.

Verifies parameterized behavior: critic enabled/disabled, type-based and
embedding-based repetition, escalation hints, all termination reasons.
"""

import pytest
from unittest.mock import MagicMock, patch

from services.memory_store import MemoryStore
from services.act_orchestrator_service import (
    ACTOrchestrator,
    ACTResult,
    _action_fingerprint,
    _action_types,
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

        # can_continue: iter 0 pre-check, iter 0 post-execute, tool-free iter pre-check
        mock_loop.can_continue.side_effect = [
            (True, None),           # iter 0 pre-check
            (False, 'max_iterations'),  # iter 0 post-execute → sets termination_reason
            (True, None),           # tool-free iteration pre-check
        ]
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found something'),
        ]
        MockActLoop.return_value = mock_loop

        # 1 action iteration + 1 tool-free iteration (LLM responds with text)
        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'recall', 'query': 'test'}]),
            _make_response(actions=[]),  # tool-free iteration: LLM responds naturally
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
        # 2 LLM calls: 1 action iteration + 1 tool-free response iteration
        assert cortex.generate_response_appended.call_count == 2


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
        # can_continue: 2 per action iteration + 1 pre-check for tool-free iteration
        mock_loop.can_continue.side_effect = [
            (True, None), (True, None),          # iter 0
            (True, None), (True, None),          # iter 1
            (True, None), (False, 'max_iterations'),  # iter 2 → sets termination_reason
            (True, None),                        # tool-free iteration pre-check
        ]
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found'),
        ]
        MockActLoop.return_value = mock_loop

        # 3 action iterations + 1 tool-free iteration (LLM responds naturally)
        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'recall', 'query': 'topic A'}]),
            _make_response(actions=[{'type': 'recall', 'query': 'topic B'}]),
            _make_response(actions=[{'type': 'recall', 'query': 'topic C'}]),
            _make_response(actions=[]),  # tool-free iteration
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
    def test_tool_free_iteration_on_forced_exit(self, MockActLoop):
        """When loop exits via max_iterations, a tool-free iteration lets LLM respond naturally."""
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
        # can_continue: iter 0 pre-check, iter 0 post-execute, tool-free iter pre-check
        mock_loop.can_continue.side_effect = [
            (True, None),               # iter 0 pre-check
            (False, 'max_iterations'),  # iter 0 post-execute → sets termination_reason
            (True, None),               # tool-free iteration pre-check
        ]
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found data'),
        ]
        MockActLoop.return_value = mock_loop

        # 1 action iteration → max_iterations exit → tool-free iteration (LLM responds)
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
        # 1 action iteration + 1 tool-free response iteration = 2 total LLM calls
        assert cortex.generate_response_appended.call_count == 2
        # Second call must have empty tools (tool-free iteration)
        second_call_kwargs = cortex.generate_response_appended.call_args_list[1][1]
        assert second_call_kwargs.get('tools') == []

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

        # 1 action iteration → callback cancels → tool-free iteration (LLM responds)
        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'recall', 'query': 'x'}]),
            _make_response(actions=[]),  # tool-free iteration
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


# ── Orchestrator: methodology learning ─────────────────────────────

@pytest.mark.unit
class TestMethodologyLearning:
    @patch('services.act_orchestrator_service.threading')
    @patch('services.act_orchestrator_service.ActLoopService')
    def test_methodology_thread_started_after_multi_iteration_loop(self, MockActLoop, mock_threading):
        """Methodology learning daemon thread starts when loop runs >= 2 iterations."""
        mock_thread = MagicMock()
        mock_threading.Thread.return_value = mock_thread

        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = 'recall -> success: found data'
        mock_loop.act_history = [_make_action_result('recall', 'success', 'found')]
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 2
        mock_loop.loop_id = 'test-loop-id'
        mock_loop._escalation_hint_injected = False
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {}
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

        orchestrator = ACTOrchestrator(config={}, max_iterations=5, smart_repetition=False)
        orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )

        mock_threading.Thread.assert_called_once()
        call_kwargs = mock_threading.Thread.call_args[1]
        assert call_kwargs.get('daemon') is True
        mock_thread.start.assert_called_once()

    @patch('services.act_orchestrator_service.ActLoopService')
    def test_methodology_thread_not_started_for_single_iteration(self, MockActLoop):
        """Methodology learning thread is NOT started for loops with < 2 iterations.

        The mock loop starts at iteration_number=0. The orchestrator increments it
        once via += 1 when no_actions exits. Final value = 1, which is < 2 so no
        methodology thread is started.
        """
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop.loop_id = 'test-id'
        mock_loop._escalation_hint_injected = False
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        MockActLoop.return_value = mock_loop

        cortex = _make_cortex_service([_make_response(actions=[])])

        with patch('services.act_orchestrator_service.threading') as mock_threading:
            orchestrator = ACTOrchestrator(config={}, smart_repetition=False)
            orchestrator.run(
                topic='test', text='hello', cortex_service=cortex,
                act_prompt='test', classification={'topic': 't', 'confidence': 10},
                chat_history=[],
            )
            mock_threading.Thread.assert_not_called()

    @patch('services.act_orchestrator_service.ActLoopService')
    def test_reflection_is_none_in_result(self, MockActLoop):
        """ACTResult.reflection is always None (methodology learning is async)."""
        mock_loop = MagicMock()
        mock_loop.get_history_context.return_value = '(none)'
        mock_loop.act_history = []
        mock_loop.iteration_logs = []
        mock_loop.iteration_number = 0
        mock_loop._escalation_hint_injected = False
        mock_loop.soft_nudge_injected = False
        mock_loop.get_loop_telemetry.return_value = {}
        mock_loop.can_continue.return_value = (True, None)
        MockActLoop.return_value = mock_loop

        cortex = _make_cortex_service([_make_response(actions=[])])
        orchestrator = ACTOrchestrator(config={}, smart_repetition=False)
        result = orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )
        assert result.reflection is None


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
            (True, None), (True, None),          # iter 0
            (True, None), (False, 'max_iterations'),  # iter 1 → sets termination_reason
            (True, None),                        # tool-free iteration pre-check
        ]
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found'),
        ]
        MockActLoop.return_value = mock_loop

        # 2 action iterations + 1 tool-free iteration (LLM responds naturally)
        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'recall', 'query': 'a'}]),
            _make_response(actions=[{'type': 'recall', 'query': 'b'}]),
            _make_response(actions=[]),  # tool-free iteration
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
    def test_persistent_task_loop_continues_after_create(self, MockActLoop):
        """After persistent_task create, loop continues and LLM produces a text response."""
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
            (True, None),  # iter 1 pre-check (soft_nudge path)
        ]
        mock_loop.execute_actions.return_value = [
            _make_action_result('persistent_task', 'success', '{"success": true, "response": "Working on it."}'),
        ]
        MockActLoop.return_value = mock_loop

        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'persistent_task', 'goal': 'X'}]),
            _make_response(actions=[]),  # LLM reads tool result and responds
        ])

        orch = ACTOrchestrator(config={}, max_iterations=10, smart_repetition=False)
        result = orch.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )
        # Loop continues naturally — exits via no_actions, not persistent_task_dispatched
        assert result.termination_reason == 'no_actions'
        assert cortex.generate_response_appended.call_count == 2


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
        store = MemoryStore()
        with patch('services.memory_client.MemoryClientService.create_connection') as mock_conn:
            mock_conn.return_value = store
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
            orchestrator.run(
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
        orchestrator.run(
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
        orchestrator.run(
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


# ── Orchestrator: forced-exit static fallback ──────────────────────

@pytest.mark.unit
class TestForcedExitFallback:
    @patch('services.act_orchestrator_service.ActLoopService')
    def test_static_fallback_when_tool_free_iteration_produces_no_text(self, MockActLoop):
        """When tool-free iteration returns no text, static fallback string is used."""
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
        mock_loop.can_continue.side_effect = [
            (True, None),               # iter 0 pre-check
            (False, 'max_iterations'),  # iter 0 post-execute
            (True, None),               # tool-free iteration pre-check
        ]
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found'),
        ]
        MockActLoop.return_value = mock_loop

        # Tool-free iteration returns empty response text
        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'recall', 'query': 'x'}]),
            {'response': '', 'actions': [], 'confidence': 0.9},
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
        assert result.final_response == "I wasn't able to complete this request."

    @patch('services.act_orchestrator_service.ActLoopService')
    def test_proactive_mode_no_fallback_on_forced_exit(self, MockActLoop):
        """Proactive mode does not inject fallback text when forced exit has no text response."""
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
        mock_loop.can_continue.side_effect = [
            (True, None),               # iter 0 pre-check
            (False, 'max_iterations'),  # iter 0 post-execute
            (True, None),               # tool-free iteration pre-check
        ]
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found'),
        ]
        MockActLoop.return_value = mock_loop

        # Tool-free iteration returns empty response
        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'recall', 'query': 'x'}]),
            {'response': '', 'actions': [], 'confidence': 0.9},
        ])

        orchestrator = ACTOrchestrator(
            config={}, max_iterations=1, smart_repetition=False, proactive=True,
        )
        result = orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )

        assert result.termination_reason == 'max_iterations'
        # Proactive mode: no static fallback injected
        assert result.final_response == ''

    @patch('services.act_orchestrator_service.ActLoopService')
    def test_double_forced_exit_breaks_loop(self, MockActLoop):
        """If the tool-free iteration itself hits a termination, the loop exits without looping again."""
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
        mock_loop.can_continue.side_effect = [
            (True, None),               # iter 0 pre-check
            (False, 'max_iterations'),  # iter 0 post-execute → first forced exit
            (True, None),               # tool-free iter pre-check
            (False, 'timeout'),         # tool-free iter post-execute → second forced exit
        ]
        mock_loop.execute_actions.return_value = [
            _make_action_result('recall', 'success', 'found'),
        ]
        MockActLoop.return_value = mock_loop

        # Tool-free iteration returns another action (triggering a second termination)
        cortex = _make_cortex_service([
            _make_response(actions=[{'type': 'recall', 'query': 'x'}]),
            _make_response(actions=[{'type': 'recall', 'query': 'y'}]),
        ])

        orchestrator = ACTOrchestrator(
            config={}, max_iterations=1, smart_repetition=False,
        )
        result = orchestrator.run(
            topic='test', text='hello', cortex_service=cortex,
            act_prompt='test', classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )

        # Loop must have broken — only 2 LLM calls, not infinite
        assert cortex.generate_response_appended.call_count == 2
        # Termination reason is the original forced-exit cause, not the second
        assert result.termination_reason == 'max_iterations'

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

        store = MemoryStore()
        store.rpush('steer:req-123', 'proceed with the plan')

        with patch('services.output_service.OutputService'), \
             patch('services.memory_client.MemoryClientService') as mock_mem, \
             patch('time.sleep'):
            mock_mem.create_connection.return_value = store
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

        store = MemoryStore()  # Empty store — no steering response ever arrives

        with patch('services.output_service.OutputService'), \
             patch('services.memory_client.MemoryClientService') as mock_mem, \
             patch('time.sleep'), \
             patch('time.monotonic', side_effect=[0.0, 2.0]):
            # deadline = 0.0 + 1.0 = 1.0; first while check: 2.0 < 1.0 → False
            mock_mem.create_connection.return_value = store
            result = orch._escalate_and_wait(
                mock_loop, 'Please confirm.', 'exchange-1',
                poll_interval=0.0, max_wait=1.0,
            )

        assert result is None


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

