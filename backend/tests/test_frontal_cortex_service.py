"""Tests for FrontalCortexService — pure function tests (no LLM calls)."""

import pytest

from services.frontal_cortex_service import ChatHistoryProcessor, _ONBOARDING_SCHEDULE


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
