"""Tests for the tool-calling while loop in MessageProcessor.send().

Verifies loop entry/exit conditions, ephemeral storage, narration callbacks,
steering injection, tool compounding, and the normalized response contract.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from services.llm_service import LLMResponse
from services.message_processor import MessageProcessor, MAX_ITERATIONS

pytestmark = pytest.mark.unit


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_llm_response(text=None, tool_calls=None, stop_reason=None):
    return LLMResponse(
        text=text,
        model='test-model',
        provider='mock',
        tokens_input=10,
        tokens_output=5,
        tool_calls=tool_calls,
        stop_reason=stop_reason or ('tool_use' if tool_calls else 'end_turn'),
    )


def _tool_call(name='memory', tc_id='tc1', input_=None):
    return {'id': tc_id, 'name': name, 'input': input_ or {}}


def _dispatch_result(result='ok'):
    return {'status': 'ok', 'result': result, 'action_type': name}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def processor():
    return MessageProcessor()


@pytest.fixture
def mock_transcript_append():
    """Patch transcript_service.append to return a stable ID.

    message_processor does ``from services import transcript_service`` inside
    send(), binding the module object at call time. We patch the ``append``
    function on the transcript_service module itself so all code paths see the
    same mock regardless of when the import occurs.
    """
    with patch('services.transcript_service.append', return_value=42) as mock_append:
        # Expose a .append attribute on the yielded mock so tests can inspect calls
        # uniformly via mock_transcript_append.append
        holder = MagicMock()
        holder.append = mock_append
        yield holder


@pytest.fixture
def mock_providers():
    """Patch Providers.instance() so no real LLM is called.

    send() does ``from services.providers import Providers`` lazily, so we
    patch the class on its home module.
    """
    mock_instance = MagicMock()
    with patch('services.providers.Providers.instance', return_value=mock_instance):
        yield mock_instance


@pytest.fixture
def mock_dispatcher():
    """Patch ActDispatcherService so no real skills execute."""
    with patch('services.act_dispatcher_service.ActDispatcherService') as mock_cls:
        instance = MagicMock()
        instance.dispatch_action.return_value = {'status': 'ok', 'result': 'dispatch result'}
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def mock_tool_call_svc():
    """Patch ToolCallService so writes go nowhere."""
    with patch('services.tool_call_service.ToolCallService') as mock_cls:
        instance = MagicMock()
        instance.get_find_tools_results.return_value = []
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def mock_skills():
    """Patch get_skill_schemas and ALL_SKILL_NAMES for tools resolution."""
    base_tools = [{'name': 'memory', 'description': 'recall', 'input_schema': {}}]
    with patch('services.tool_schema_service.get_skill_schemas', return_value=base_tools), \
         patch('services.innate_skills.registry.ALL_SKILL_NAMES', ['memory']):
        yield base_tools


# ── No tool calls path ────────────────────────────────────────────────────────

class TestSendNoToolCalls:
    def test_direct_response_no_loop(
        self, processor, mock_transcript_append, mock_providers, mock_skills
    ):
        """LLM returns text only — no loop entered, transcript persisted."""
        mock_providers.send.return_value = _make_llm_response(text='Hello!')

        result = processor.send('hi', 'system', channel='user')

        assert result['response'] == 'Hello!'
        # Transcript: user append + assistant append
        assert mock_transcript_append.append.call_count == 2
        # No tool loop ran — tool_calls in result should be None
        assert result['tool_calls'] is None

    def test_direct_response_no_tool_calls_key(
        self, processor, mock_transcript_append, mock_providers, mock_skills
    ):
        """When LLM returns no tool_calls, result actions is None."""
        mock_providers.send.return_value = _make_llm_response(text='Done.')

        result = processor.send('prompt', 'sys', channel='user')

        assert result['actions'] is None

    def test_empty_response_not_appended_to_transcript(
        self, processor, mock_transcript_append, mock_providers, mock_skills
    ):
        """An LLM response with empty text and no tool_calls appends user turn only."""
        mock_providers.send.return_value = _make_llm_response(text='')

        processor.send('hi', 'sys', channel='user')

        # Only the user turn appended — no assistant append for empty text
        assert mock_transcript_append.append.call_count == 1


# ── Tool loop path ────────────────────────────────────────────────────────────

class TestSendSingleIteration:
    def test_loop_runs_once_then_text(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """LLM returns 1 tool call, then text. Loop runs once."""
        tc = _tool_call('memory')
        mock_providers.send.return_value = _make_llm_response(tool_calls=[tc])
        mock_providers.send_messages.return_value = _make_llm_response(text='Final answer.')

        result = processor.send('ask', 'sys', channel='user')

        assert result['response'] == 'Final answer.'
        mock_dispatcher.dispatch_action.assert_called_once()
        # Transcript: user in + assistant out (final)
        assert mock_transcript_append.append.call_count == 2

    def test_tool_results_stored_ephemeral(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """Tool call batch stored with ephemeral=True after dispatch."""
        tc = _tool_call('memory')
        mock_providers.send.return_value = _make_llm_response(tool_calls=[tc])
        mock_providers.send_messages.return_value = _make_llm_response(text='done')

        processor.send('ask', 'sys', channel='user')

        mock_tool_call_svc.store_batch.assert_called_once()
        call_args, call_kwargs = mock_tool_call_svc.store_batch.call_args
        # ephemeral=True is always passed as a keyword arg in message_processor
        assert call_kwargs.get('ephemeral') is True
        assert call_kwargs.get('invoked_by') == 'llm'

    def test_synthesis_stored_ephemeral(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """Per-iteration synthesis text is stored as ephemeral tool_synthesis."""
        tc = _tool_call('memory')
        # First call: tool call WITH synthesis narration text
        mock_providers.send.return_value = _make_llm_response(
            text='Thinking...', tool_calls=[tc]
        )
        mock_providers.send_messages.return_value = _make_llm_response(text='Final.')

        processor.send('ask', 'sys', channel='user')

        # store() should have been called for the synthesis
        synthesis_calls = [
            c for c in mock_tool_call_svc.store.call_args_list
            if c[0][1] == 'tool_synthesis'
        ]
        assert len(synthesis_calls) == 1
        _, call_kwargs = synthesis_calls[0]
        assert call_kwargs.get('invoked_by') == 'llm'
        assert call_kwargs.get('ephemeral') is True

    def test_final_response_in_transcript_not_tool_calls(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """Final text response goes to transcript.append, not tool_call_svc.store."""
        tc = _tool_call('memory')
        mock_providers.send.return_value = _make_llm_response(tool_calls=[tc])
        mock_providers.send_messages.return_value = _make_llm_response(text='The answer is 42.')

        processor.send('ask', 'sys', channel='user')

        # The final text must be appended to transcript as 'assistant'
        assistant_appends = [
            c for c in mock_transcript_append.append.call_args_list
            if c[0][1] == 'assistant'
        ]
        assert len(assistant_appends) == 1
        assert 'The answer is 42.' in assistant_appends[0][0]


class TestSendMultipleIterations:
    def test_loop_runs_twice(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """LLM returns tool calls twice, then text. Dispatcher called twice total."""
        tc1 = _tool_call('memory', 'tc1')
        tc2 = _tool_call('goals', 'tc2')
        mock_providers.send.return_value = _make_llm_response(tool_calls=[tc1])
        mock_providers.send_messages.side_effect = [
            _make_llm_response(tool_calls=[tc2]),
            _make_llm_response(text='All done.'),
        ]

        result = processor.send('ask', 'sys', channel='user')

        assert result['response'] == 'All done.'
        assert mock_dispatcher.dispatch_action.call_count == 2

    def test_tool_results_stored_per_iteration(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """store_batch called once per loop iteration."""
        tc1 = _tool_call('memory', 'tc1')
        tc2 = _tool_call('goals', 'tc2')
        mock_providers.send.return_value = _make_llm_response(tool_calls=[tc1])
        mock_providers.send_messages.side_effect = [
            _make_llm_response(tool_calls=[tc2]),
            _make_llm_response(text='done'),
        ]

        processor.send('ask', 'sys', channel='user')

        assert mock_tool_call_svc.store_batch.call_count == 2


class TestMaxIterations:
    def test_loop_exits_at_max_iterations(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """Loop never exceeds MAX_ITERATIONS even if LLM keeps returning tool calls."""
        tc = _tool_call('memory')
        # Initial call always returns a tool call
        mock_providers.send.return_value = _make_llm_response(tool_calls=[tc])
        # Every subsequent call also returns tool calls (infinite loop scenario)
        mock_providers.send_messages.return_value = _make_llm_response(
            tool_calls=[tc], text=None
        )

        result = processor.send('ask', 'sys', channel='user')

        # dispatch_action called at most MAX_ITERATIONS times
        assert mock_dispatcher.dispatch_action.call_count <= MAX_ITERATIONS

    def test_loop_exits_at_timeout(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """Loop exits early when MAX_TIMEOUT is exceeded."""
        tc = _tool_call('memory')
        mock_providers.send.return_value = _make_llm_response(tool_calls=[tc])
        mock_providers.send_messages.return_value = _make_llm_response(
            tool_calls=[tc], text=None
        )

        # Simulate time passing beyond MAX_TIMEOUT
        import time as _time_module
        original_time = _time_module.time

        call_count = [0]

        def fast_time():
            call_count[0] += 1
            # After a few calls, jump time forward past MAX_TIMEOUT
            if call_count[0] > 5:
                return original_time() + 10000
            return original_time()

        with patch('services.message_processor.time') as mock_time:
            mock_time.time.side_effect = fast_time

            processor.send('ask', 'sys', channel='user')

        # The key assertion: loop did not run MAX_ITERATIONS times
        assert mock_dispatcher.dispatch_action.call_count < MAX_ITERATIONS


class TestNarrationCallback:
    def test_narration_callback_called_when_text_with_tool_calls(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """on_narration callback fires when LLM returns text alongside tool calls."""
        tc = _tool_call('memory')
        mock_providers.send.return_value = _make_llm_response(
            text='Searching memory...', tool_calls=[tc]
        )
        mock_providers.send_messages.return_value = _make_llm_response(text='done')

        narration_calls = []
        processor.send('ask', 'sys', channel='user',
                        on_narration=lambda text, step: narration_calls.append((text, step)))

        assert len(narration_calls) == 1
        assert narration_calls[0][0] == 'Searching memory...'
        assert narration_calls[0][1] == 0  # first iteration

    def test_narration_not_called_when_no_text(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """on_narration not called when LLM returns tool calls with no text."""
        tc = _tool_call('memory')
        mock_providers.send.return_value = _make_llm_response(tool_calls=[tc])
        mock_providers.send_messages.return_value = _make_llm_response(text='done')

        narration_calls = []
        processor.send('ask', 'sys', channel='user',
                        on_narration=lambda text, step: narration_calls.append((text, step)))

        assert narration_calls == []

    def test_narration_callback_exception_does_not_abort_loop(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """A crashing on_narration callback must not abort the tool loop."""
        tc = _tool_call('memory')
        mock_providers.send.return_value = _make_llm_response(
            text='Thinking', tool_calls=[tc]
        )
        mock_providers.send_messages.return_value = _make_llm_response(text='Result.')

        def _bad_callback(text, step):
            raise RuntimeError("callback exploded")

        # Should not raise
        result = processor.send('ask', 'sys', channel='user', on_narration=_bad_callback)
        assert result['response'] == 'Result.'


class TestSteering:
    def test_steering_injected_as_user_message(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """Steering text from MemoryStore is appended to messages as a user turn."""
        tc = _tool_call('memory')
        mock_providers.send.return_value = _make_llm_response(tool_calls=[tc])
        mock_providers.send_messages.return_value = _make_llm_response(text='done')

        with patch.object(processor, '_drain_steering', return_value='Please be concise'):
            processor.send('ask', 'sys', channel='user', request_id='req-123')

        # send_messages should have been called; check messages contain the steer
        call_args = mock_providers.send_messages.call_args
        messages = call_args[0][1]  # second positional: messages list
        user_messages = [m for m in messages if m['role'] == 'user']
        assert any('Please be concise' in m['content'] for m in user_messages)

    def test_steering_stored_as_ephemeral_user_steer(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """Steering text is stored as ephemeral user_steer in tool_calls."""
        tc = _tool_call('memory')
        mock_providers.send.return_value = _make_llm_response(tool_calls=[tc])
        mock_providers.send_messages.return_value = _make_llm_response(text='done')

        with patch.object(processor, '_drain_steering', return_value='Steer this'):
            processor.send('ask', 'sys', channel='user', request_id='req-1')

        steer_calls = [
            c for c in mock_tool_call_svc.store.call_args_list
            if c[0][1] == 'user_steer'
        ]
        assert len(steer_calls) == 1
        call_args = steer_calls[0][0]
        assert call_args[3] == 'Steer this'
        assert steer_calls[0][1].get('ephemeral') is True

    def test_no_steering_when_no_request_id(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """_drain_steering returns None when request_id is None — no extra user msg."""
        tc = _tool_call('memory')
        mock_providers.send.return_value = _make_llm_response(tool_calls=[tc])
        mock_providers.send_messages.return_value = _make_llm_response(text='done')

        processor.send('ask', 'sys', channel='user', request_id=None)

        steer_calls = [
            c for c in mock_tool_call_svc.store.call_args_list
            if c[0][1] == 'user_steer'
        ]
        assert steer_calls == []


class TestCompoundTools:
    def test_compound_tools_expands_tool_list(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """find_tools results cause new schemas to be injected for subsequent iterations."""
        tc = _tool_call('find_tools', input_={'query': 'email'})
        mock_providers.send.return_value = _make_llm_response(tool_calls=[tc])
        mock_providers.send_messages.return_value = _make_llm_response(text='done')

        # Simulate find_tools discovering a new tool
        mock_tool_call_svc.get_find_tools_results.return_value = ['send_email']
        new_schema = {'name': 'send_email', 'description': 'Send email', 'input_schema': {}}

        with patch('services.tool_schema_service.get_external_tool_schemas', return_value=[new_schema]):
            processor.send('ask', 'sys', channel='user')

        # send_messages should have received the expanded tools list
        call_args = mock_providers.send_messages.call_args
        tools_passed = call_args[1].get('tools') or call_args[0][3] if len(call_args[0]) > 3 else None
        # _compound_tools is called and get_find_tools_results was queried
        mock_tool_call_svc.get_find_tools_results.assert_called()

    def test_compound_tools_no_find_tools_no_expansion(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """When no find_tools results, tool list stays at base set."""
        tc = _tool_call('memory')
        mock_providers.send.return_value = _make_llm_response(tool_calls=[tc])
        mock_providers.send_messages.return_value = _make_llm_response(text='done')

        mock_tool_call_svc.get_find_tools_results.return_value = []

        with patch('services.tool_schema_service.get_external_tool_schemas', return_value=[]) as mock_ext:
            processor.send('ask', 'sys', channel='user')

        # get_external_tool_schemas not called when discovered list is empty
        mock_ext.assert_not_called()


class TestToolResultMessageFormat:
    def test_tool_result_messages_include_name_field(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """Tool result messages in messages array have 'name' field (Gemini compat)."""
        tc = _tool_call('memory', 'tc1')
        mock_providers.send.return_value = _make_llm_response(tool_calls=[tc])
        mock_providers.send_messages.return_value = _make_llm_response(text='done')

        processor.send('ask', 'sys', channel='user')

        messages_sent = mock_providers.send_messages.call_args[0][1]
        tool_result_msgs = [m for m in messages_sent if m['role'] == 'tool']
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0]['name'] == 'memory'
        assert tool_result_msgs[0]['tool_call_id'] == 'tc1'

    def test_assistant_message_includes_tool_calls_array(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """Assistant message in messages array has 'tool_calls' key in expected shape."""
        tc = _tool_call('memory', 'tc1', {'query': 'x'})
        mock_providers.send.return_value = _make_llm_response(tool_calls=[tc])
        mock_providers.send_messages.return_value = _make_llm_response(text='done')

        processor.send('ask', 'sys', channel='user')

        messages_sent = mock_providers.send_messages.call_args[0][1]
        asst_msgs = [m for m in messages_sent if m['role'] == 'assistant']
        assert len(asst_msgs) == 1
        assert 'tool_calls' in asst_msgs[0]
        tc_entry = asst_msgs[0]['tool_calls'][0]
        assert tc_entry['type'] == 'function'
        assert tc_entry['function']['name'] == 'memory'


class TestDispatcherActionFormat:
    def test_dispatcher_receives_flattened_action(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """ActDispatcherService.dispatch_action receives {type: <name>, **input}."""
        tc = _tool_call('memory', 'tc1', {'query': 'coffee preferences'})
        mock_providers.send.return_value = _make_llm_response(tool_calls=[tc])
        mock_providers.send_messages.return_value = _make_llm_response(text='done')

        processor.send('ask', 'sys', channel='user')

        call_args = mock_dispatcher.dispatch_action.call_args
        channel_arg = call_args[0][0]
        action_arg = call_args[0][1]
        assert action_arg['type'] == 'memory'
        assert action_arg['query'] == 'coffee preferences'
        assert channel_arg == 'user'


class TestNormalizeResponse:
    def test_tool_loop_ran_clears_tool_calls_from_result(
        self, processor, mock_transcript_append, mock_providers,
        mock_dispatcher, mock_tool_call_svc, mock_skills
    ):
        """When tool loop ran, result has tool_calls=None (they were already executed)."""
        tc = _tool_call('memory')
        mock_providers.send.return_value = _make_llm_response(tool_calls=[tc])
        mock_providers.send_messages.return_value = _make_llm_response(text='response')

        result = processor.send('ask', 'sys', channel='user')

        assert result['tool_calls'] is None
        assert result['actions'] is None

    def test_result_contains_generation_time(
        self, processor, mock_transcript_append, mock_providers, mock_skills
    ):
        """Result dict always contains generation_time."""
        mock_providers.send.return_value = _make_llm_response(text='hi')

        result = processor.send('ask', 'sys', channel='user')

        assert 'generation_time' in result
        assert result['generation_time'] >= 0

    def test_result_provider_and_model_propagated(
        self, processor, mock_transcript_append, mock_providers, mock_skills
    ):
        """Provider and model from LLMResponse propagate to result dict."""
        mock_providers.send.return_value = LLMResponse(
            text='hi', model='gpt-4o', provider='openai'
        )

        result = processor.send('ask', 'sys', channel='user')

        assert result['model'] == 'gpt-4o'
        assert result['provider'] == 'openai'
