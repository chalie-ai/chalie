"""Tests for FrontalCortexService — facade delegation and re-export smoke tests."""

import pytest
from unittest.mock import MagicMock, patch

from services.frontal_cortex_service import (
    FrontalCortexService,
    ChatHistoryProcessor,
    _ONBOARDING_SCHEDULE,
)


pytestmark = pytest.mark.unit


class TestChatHistoryProcessor:
    """Tests for ChatHistoryProcessor.process() — deterministic formatting."""

    def test_empty_history_returns_no_conversation(self):
        """Empty chat history should produce the sentinel string."""
        processor = ChatHistoryProcessor()
        result = processor.process([])
        assert result == "No previous conversation"

    def test_none_history_returns_no_conversation(self):
        """None-ish (empty list) input should return sentinel."""
        processor = ChatHistoryProcessor()
        result = processor.process([])
        assert result == "No previous conversation"

    def test_single_exchange_formatting(self):
        """A single user+assistant exchange should produce 2 lines."""
        processor = ChatHistoryProcessor()
        history = [
            {
                'prompt': {'message': 'What is Python?'},
                'response': {'message': 'A programming language.'},
            }
        ]
        result = processor.process(history)
        assert 'User: What is Python?' in result
        assert 'Assistant: A programming language.' in result

    def test_multi_exchange_formatting(self):
        """Multiple exchanges should all appear in output."""
        processor = ChatHistoryProcessor()
        history = [
            {
                'prompt': {'message': 'Hello'},
                'response': {'message': 'Hi there'},
            },
            {
                'prompt': {'message': 'How are you?'},
                'response': {'message': 'Good, thanks.'},
            },
        ]
        result = processor.process(history)
        assert 'User: Hello' in result
        assert 'Assistant: Hi there' in result
        assert 'User: How are you?' in result
        assert 'Assistant: Good, thanks.' in result

    def test_max_exchanges_limit_trims_oldest(self):
        """Setting max_exchanges should keep only the most recent N exchanges."""
        processor = ChatHistoryProcessor(max_exchanges=2)
        history = [
            {'prompt': {'message': f'q{i}'}, 'response': {'message': f'a{i}'}}
            for i in range(5)
        ]
        result = processor.process(history)
        # Only the last 2 exchanges should be present
        assert 'User: q0' not in result
        assert 'User: q1' not in result
        assert 'User: q2' not in result
        assert 'User: q3' in result
        assert 'User: q4' in result

    def test_response_as_plain_string(self):
        """When response is a plain string (not dict), it should still format."""
        processor = ChatHistoryProcessor()
        history = [
            {
                'prompt': {'message': 'Tell me a joke'},
                'response': 'Why did the chicken cross the road?',
            },
        ]
        result = processor.process(history)
        assert 'Assistant: Why did the chicken cross the road?' in result

    def test_response_with_error_key(self):
        """Error responses should be formatted with [Error: ...] prefix."""
        processor = ChatHistoryProcessor()
        history = [
            {
                'prompt': {'message': 'Do something'},
                'response': {'error': 'Service unavailable'},
            },
        ]
        result = processor.process(history)
        assert 'Assistant: [Error: Service unavailable]' in result

    def test_exchange_without_prompt_key_skips_user_line(self):
        """Exchanges missing 'prompt' key should not produce a User line."""
        processor = ChatHistoryProcessor()
        history = [
            {
                'response': {'message': 'Proactive message'},
            },
        ]
        result = processor.process(history)
        assert 'User:' not in result
        assert 'Assistant: Proactive message' in result


class TestOnboardingSchedule:
    """Tests for the _ONBOARDING_SCHEDULE structure."""

    def test_schedule_is_non_empty_list(self):
        """The onboarding schedule should have at least one entry."""
        assert isinstance(_ONBOARDING_SCHEDULE, list)
        assert len(_ONBOARDING_SCHEDULE) > 0

    def test_each_entry_has_required_keys(self):
        """Every schedule entry must have trait, min_turn, cooldown_turns, max_attempts, hint."""
        required_keys = {'trait', 'min_turn', 'cooldown_turns', 'max_attempts', 'hint'}
        for entry in _ONBOARDING_SCHEDULE:
            missing = required_keys - set(entry.keys())
            assert not missing, f"Entry for '{entry.get('trait', '?')}' missing keys: {missing}"

    def test_min_turn_values_are_ascending(self):
        """Traits should be elicited in order of increasing min_turn."""
        min_turns = [entry['min_turn'] for entry in _ONBOARDING_SCHEDULE]
        assert min_turns == sorted(min_turns), (
            f"min_turn values should be ascending: {min_turns}"
        )

    def test_name_is_first_trait(self):
        """The 'name' trait should be the first one elicited."""
        assert _ONBOARDING_SCHEDULE[0]['trait'] == 'name'


# ── FrontalCortexService facade delegation ─────────────────────────


class TestFrontalCortexServiceFacade:
    """Tests verifying that :class:`FrontalCortexService` correctly delegates to sub-services.

    All external sub-services are replaced with :class:`~unittest.mock.MagicMock`
    so no real LLM calls or DB queries occur.
    """

    def _make_service(self):
        """Create a :class:`FrontalCortexService` with both sub-services fully mocked.

        Returns:
            FrontalCortexService: Instance with ``_prompt_assembly`` and
            ``_response_gen`` replaced by MagicMock objects.
        """
        with patch('services.frontal_cortex_service.PromptAssemblyService') as mock_pa, \
             patch('services.frontal_cortex_service.ResponseGenerationService') as mock_rg:
            svc = FrontalCortexService(config={'platform': 'test'})
        # mock_pa.return_value and mock_rg.return_value persist after the with block
        return svc

    def test_init_stores_config(self):
        """__init__ stores the config dict on ``self.config``."""
        with patch('services.frontal_cortex_service.PromptAssemblyService'), \
             patch('services.frontal_cortex_service.ResponseGenerationService'):
            svc = FrontalCortexService(config={'platform': 'test', 'key': 'val'})
        assert svc.config == {'platform': 'test', 'key': 'val'}

    def test_init_creates_prompt_assembly_service_with_config(self):
        """__init__ instantiates PromptAssemblyService with the given config."""
        config = {'platform': 'test'}
        with patch('services.frontal_cortex_service.PromptAssemblyService') as mock_pa, \
             patch('services.frontal_cortex_service.ResponseGenerationService'):
            FrontalCortexService(config=config)
        mock_pa.assert_called_once_with(config)

    def test_init_creates_response_generation_service_with_config(self):
        """__init__ instantiates ResponseGenerationService with the given config."""
        config = {'platform': 'test'}
        with patch('services.frontal_cortex_service.PromptAssemblyService'), \
             patch('services.frontal_cortex_service.ResponseGenerationService') as mock_rg:
            FrontalCortexService(config=config)
        mock_rg.assert_called_once_with(config)

    def test_get_context_limit_delegates_to_response_gen(self):
        """get_context_limit() returns the value from ResponseGenerationService."""
        svc = self._make_service()
        svc._response_gen.get_context_limit.return_value = 128_000
        assert svc.get_context_limit() == 128_000
        svc._response_gen.get_context_limit.assert_called_once()

    def test_count_tokens_delegates_with_all_args(self):
        """count_tokens() forwards messages, system_prompt, and tools to ResponseGenerationService."""
        svc = self._make_service()
        svc._response_gen.count_tokens.return_value = 500
        messages = [{'role': 'user', 'content': 'hello'}]
        result = svc.count_tokens(messages, system_prompt='sys', tools=None)
        assert result == 500
        svc._response_gen.count_tokens.assert_called_once_with(messages, 'sys', None)

    def test_count_tokens_passes_optional_tools(self):
        """count_tokens() forwards an optional tools list to ResponseGenerationService."""
        svc = self._make_service()
        svc._response_gen.count_tokens.return_value = 750
        tools = [{'name': 'recall', 'description': 'Search memory'}]
        svc.count_tokens([], system_prompt='', tools=tools)
        svc._response_gen.count_tokens.assert_called_once_with([], '', tools)

    def test_build_system_prompt_delegates_to_prompt_assembly(self):
        """build_system_prompt() delegates to PromptAssemblyService and returns its result."""
        svc = self._make_service()
        svc._prompt_assembly.build_system_prompt.return_value = 'assembled system prompt'
        result = svc.build_system_prompt(
            system_prompt_template='template {{topic}}',
            original_prompt='hello',
            classification={'topic': 'coding', 'confidence': 10},
            chat_history=[],
        )
        assert result == 'assembled system prompt'
        svc._prompt_assembly.build_system_prompt.assert_called_once()

    def test_generate_response_appended_delegates_to_response_gen(self):
        """generate_response_appended() delegates fully to ResponseGenerationService."""
        svc = self._make_service()
        expected = {'mode': 'UNIFIED', 'response': 'hello', 'actions': []}
        svc._response_gen.generate_response_appended.return_value = expected
        messages = [{'role': 'user', 'content': 'test'}]
        result = svc.generate_response_appended('sys', messages, cache_prefix=True)
        assert result is expected
        svc._response_gen.generate_response_appended.assert_called_once_with(
            'sys', messages, True, None
        )

    def test_generate_response_appended_passes_tools(self):
        """generate_response_appended() forwards an optional tools list unchanged."""
        svc = self._make_service()
        svc._response_gen.generate_response_appended.return_value = {}
        tools = [{'name': 'recall'}]
        svc.generate_response_appended('sys', [], tools=tools)
        svc._response_gen.generate_response_appended.assert_called_once_with(
            'sys', [], False, tools
        )

    def test_parse_response_text_delegates_to_response_gen(self):
        """_parse_response_text() forwards raw text and generation_time to ResponseGenerationService."""
        svc = self._make_service()
        expected = {'mode': 'UNIFIED', 'response': 'test result', 'confidence': 0.9}
        svc._response_gen._parse_response_text.return_value = expected
        result = svc._parse_response_text('{"mode": "UNIFIED"}', 0.42)
        assert result is expected
        svc._response_gen._parse_response_text.assert_called_once_with(
            '{"mode": "UNIFIED"}', 0.42
        )

    def test_generate_response_calls_inject_parameters_then_generate(self):
        """generate_response() injects parameters via PromptAssemblyService then calls ResponseGenerationService."""
        svc = self._make_service()
        svc._prompt_assembly._inject_parameters.return_value = 'injected system prompt'
        svc._response_gen.generate_response.return_value = {
            'mode': 'UNIFIED', 'response': 'final answer',
        }
        result = svc.generate_response(
            system_prompt_template='template',
            original_prompt='user question',
            classification={'topic': 't', 'confidence': 10},
            chat_history=[],
        )
        svc._prompt_assembly._inject_parameters.assert_called_once()
        svc._response_gen.generate_response.assert_called_once_with(
            'injected system prompt', 'user question', 'template'
        )
        assert result['mode'] == 'UNIFIED'

    def test_generate_response_returns_response_gen_result(self):
        """generate_response() returns exactly what ResponseGenerationService returns."""
        svc = self._make_service()
        svc._prompt_assembly._inject_parameters.return_value = 'prompt'
        expected = {
            'mode': 'ACT',
            'actions': [{'type': 'recall'}],
            'confidence': 0.8,
            'response': '',
        }
        svc._response_gen.generate_response.return_value = expected
        result = svc.generate_response(
            system_prompt_template='t',
            original_prompt='q',
            classification={'topic': 'test'},
            chat_history=[],
        )
        assert result is expected


# ── FrontalCortexService re-export smoke tests ──────────────────────


class TestFrontalCortexReExports:
    """Smoke tests verifying that :mod:`services.frontal_cortex_service` re-exports
    ``ChatHistoryProcessor`` and ``_ONBOARDING_SCHEDULE`` from their canonical sources.
    """

    def test_chat_history_processor_importable_from_facade(self):
        """ChatHistoryProcessor is importable from frontal_cortex_service."""
        from services.frontal_cortex_service import ChatHistoryProcessor as FacadeCHP
        assert FacadeCHP is not None

    def test_onboarding_schedule_importable_from_facade(self):
        """_ONBOARDING_SCHEDULE is importable from frontal_cortex_service."""
        from services.frontal_cortex_service import _ONBOARDING_SCHEDULE as FacadeSchedule
        assert FacadeSchedule is not None

    def test_chat_history_processor_is_same_class_as_canonical(self):
        """The re-exported ChatHistoryProcessor is the identical class object from response_generation_service."""
        from services.frontal_cortex_service import ChatHistoryProcessor as FacadeCHP
        from services.response_generation_service import ChatHistoryProcessor as CanonicalCHP
        assert FacadeCHP is CanonicalCHP

    def test_onboarding_schedule_is_same_object_as_canonical(self):
        """The re-exported _ONBOARDING_SCHEDULE is the identical list object from prompt_assembly_service."""
        from services.frontal_cortex_service import _ONBOARDING_SCHEDULE as FacadeSchedule
        from services.prompt_assembly_service import _ONBOARDING_SCHEDULE as CanonicalSchedule
        assert FacadeSchedule is CanonicalSchedule

    def test_all_exports_contains_expected_names(self):
        """__all__ declares FrontalCortexService, ChatHistoryProcessor, and _ONBOARDING_SCHEDULE."""
        import services.frontal_cortex_service as mod
        assert 'FrontalCortexService' in mod.__all__
        assert 'ChatHistoryProcessor' in mod.__all__
        assert '_ONBOARDING_SCHEDULE' in mod.__all__

    def test_chat_history_processor_from_facade_is_functional(self):
        """ChatHistoryProcessor imported from the facade correctly processes an empty history."""
        from services.frontal_cortex_service import ChatHistoryProcessor
        proc = ChatHistoryProcessor()
        assert proc.process([]) == 'No previous conversation'

    def test_onboarding_schedule_from_facade_has_name_as_first_trait(self):
        """_ONBOARDING_SCHEDULE from the facade schedules 'name' elicitation first."""
        from services.frontal_cortex_service import _ONBOARDING_SCHEDULE
        assert _ONBOARDING_SCHEDULE[0]['trait'] == 'name'
