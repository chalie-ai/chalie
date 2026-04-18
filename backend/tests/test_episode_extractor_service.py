"""Tests for EpisodeExtractorService — LLM mocked, pure unit tests."""

import json
import pytest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


def _make_entry(entry_id: int, role: str = 'user', content: str = 'hello') -> dict:
    return {
        'id': entry_id,
        'role': role,
        'content': content,
        'created_at': '2026-04-06T10:00:00+00:00',
        'tool_name': None,
    }


def _make_valid_episode(entry_range: list) -> dict:
    return {
        'intent': {'type': 'exploration', 'direction': 'understand topic'},
        'context': 'User is learning about Python',
        'action': 'User asked a question, assistant explained',
        'emotion': {'valence': 'positive', 'intensity': 'low'},
        'outcome': 'User understood the concept',
        'gist': 'User asked about Python decorators. Assistant explained with examples. User confirmed understanding.',
        'salience_factors': {
            'novelty': 2,
            'emotional_weight': 1,
            'goal_relevance': 2,
            'decision_made': False,
            'open_loop_created': False,
        },
        'open_loops': [],
        'entry_range': entry_range,
        'entities': ['Python'],
        'goal_tags': ['learn python'],
        'emotional_valence': 0.5,
        'emotional_arousal': 0.3,
        'traits': [
            {'key': 'language_preference', 'value': 'Python', 'kind': 'preference', 'decay_class': 'slow'}
        ],
    }


def _make_svc(llm_response_text: str):
    """Build an EpisodeExtractorService with a mocked LLM and a mocked prompt."""
    from services.llm_service import LLMResponse  # noqa: F401 — direct module import

    mock_llm = MagicMock()
    mock_llm.send_message.return_value = LLMResponse(
        text=llm_response_text,
        model='test-model',
        provider='mock',
    )

    with patch('services.episode_extractor_service.ConfigService.resolve_agent_config',
               return_value={}), \
         patch('services.episode_extractor_service._PROMPT_PATH') as mock_path, \
         patch('services.episode_extractor_service.create_llm_service',
               return_value=mock_llm):
        mock_path.read_text.return_value = 'Prompt: {{transcript_window}} Topic: {{topic}}'
        from services.episode_extractor_service import EpisodeExtractorService
        svc = EpisodeExtractorService()

    svc._llm = mock_llm
    return svc


class TestExtractReturnsCorrectStructure:

    def test_returns_list(self):
        # entry_range values are transcript IDs (what the LLM sees in [id] brackets),
        # not 0-based positions.
        episode = _make_valid_episode([10, 30])
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(i) for i in [10, 20, 30]]

        result = svc.extract(entries, 'programming')

        assert isinstance(result, list)

    def test_single_episode_has_required_keys(self):
        episode = _make_valid_episode([10, 30])
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(i) for i in [10, 20, 30]]

        result = svc.extract(entries, 'programming')

        assert len(result) == 1
        ep = result[0]
        for key in ('intent', 'context', 'action', 'emotion', 'outcome', 'gist',
                    'salience_factors', 'transcript_ids', 'entities', 'goal_tags',
                    'emotional_valence', 'emotional_arousal', 'traits', 'open_loops'):
            assert key in ep, f"Missing key: {key}"

    def test_multiple_episodes_returned(self):
        ep1 = _make_valid_episode([10, 20])
        ep2 = _make_valid_episode([30, 40])
        svc = _make_svc(json.dumps([ep1, ep2]))
        entries = [_make_entry(i) for i in [10, 20, 30, 40]]

        result = svc.extract(entries, 'general')

        assert len(result) == 2


class TestExtractEmptyAndTrivialWindows:

    def test_empty_entries_returns_empty_list(self):
        svc = _make_svc('[]')

        result = svc.extract([], 'any-topic')

        assert result == []

    def test_llm_returns_empty_array_returns_empty(self):
        entries = [_make_entry(1, content='ok')]
        svc = _make_svc('[]')

        result = svc.extract(entries, 'general')

        assert result == []

    def test_llm_returns_invalid_json_returns_empty(self):
        entries = [_make_entry(1, content='hello')]
        svc = _make_svc('this is not json at all')

        result = svc.extract(entries, 'general')

        assert result == []

    def test_llm_returns_non_list_json_returns_empty(self):
        entries = [_make_entry(1, content='hello')]
        svc = _make_svc('{"not": "a list"}')

        result = svc.extract(entries, 'general')

        assert result == []


class TestTranscriptIdComputation:

    def test_entry_range_maps_to_transcript_ids(self):
        # entries have ids 101, 102, 103; entry_range [101, 103] (inclusive ID
        # range as returned by the LLM) covers all three.
        episode = _make_valid_episode([101, 103])
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(101), _make_entry(102), _make_entry(103)]

        result = svc.extract(entries, 'test')

        assert result[0]['transcript_ids'] == [101, 102, 103]

    def test_partial_entry_range_maps_correctly(self):
        episode = _make_valid_episode([20, 30])
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(10), _make_entry(20), _make_entry(30)]

        result = svc.extract(entries, 'test')

        assert result[0]['transcript_ids'] == [20, 30]

    def test_transcript_id_start_end_derived_from_entry_range(self):
        # range [3, 10] spans all three entries regardless of iteration order
        episode = _make_valid_episode([3, 10])
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(5), _make_entry(10), _make_entry(3)]

        result = svc.extract(entries, 'test')

        ep = result[0]
        # transcript_ids = window_ids filtered to [3..10] = [5, 10, 3]
        assert ep['transcript_id_start'] == 3
        assert ep['transcript_id_end'] == 10

    def test_episode_with_invalid_entry_range_is_skipped(self):
        # ids 500-999 don't exist in the 3-entry window; no overlap → skip
        episode = _make_valid_episode([500, 999])
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(1), _make_entry(2), _make_entry(3)]

        result = svc.extract(entries, 'test')

        assert result == []

    def test_episode_with_non_list_entry_range_is_skipped(self):
        episode = _make_valid_episode([1, 3])
        episode['entry_range'] = 'invalid'
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(1), _make_entry(2), _make_entry(3)]

        result = svc.extract(entries, 'test')

        assert result == []

    def test_single_entry_range_is_skipped(self):
        episode = _make_valid_episode([1, 3])
        episode['entry_range'] = [1]  # must be exactly 2 elements
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(1), _make_entry(2), _make_entry(3)]

        result = svc.extract(entries, 'test')

        assert result == []

    def test_single_entry_range_point(self):
        # [42, 42] is a valid range covering one entry with id 42
        episode = _make_valid_episode([42, 42])
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(42)]

        result = svc.extract(entries, 'test')

        assert result[0]['transcript_ids'] == [42]
        assert result[0]['transcript_id_start'] == 42
        assert result[0]['transcript_id_end'] == 42


class TestTraitStructure:

    def test_traits_have_required_keys(self):
        episode = _make_valid_episode([1, 1])
        episode['traits'] = [
            {'key': 'name', 'value': 'Dylan', 'kind': 'trait', 'decay_class': 'permanent'}
        ]
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'intro')

        traits = result[0]['traits']
        assert len(traits) == 1
        trait = traits[0]
        assert trait['key'] == 'name'
        assert trait['value'] == 'Dylan'
        assert trait['kind'] == 'trait'
        assert trait['decay_class'] == 'permanent'

    def test_missing_traits_field_defaults_to_empty_list(self):
        episode = _make_valid_episode([1, 1])
        del episode['traits']
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['traits'] == []

    def test_non_list_traits_replaced_with_empty_list(self):
        episode = _make_valid_episode([1, 1])
        episode['traits'] = 'not a list'
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['traits'] == []


class TestEmotionalRanges:

    def test_emotional_valence_clamped_to_minus_one_to_one(self):
        episode = _make_valid_episode([1, 1])
        episode['emotional_valence'] = 5.0  # out of range
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_valence'] == pytest.approx(1.0)

    def test_emotional_valence_negative_clamped(self):
        episode = _make_valid_episode([1, 1])
        episode['emotional_valence'] = -3.0
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_valence'] == pytest.approx(-1.0)

    def test_emotional_arousal_clamped_to_zero_to_one(self):
        episode = _make_valid_episode([1, 1])
        episode['emotional_arousal'] = -0.5  # below zero
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_arousal'] == pytest.approx(0.0)

    def test_emotional_arousal_above_one_clamped(self):
        episode = _make_valid_episode([1, 1])
        episode['emotional_arousal'] = 2.0
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_arousal'] == pytest.approx(1.0)

    def test_none_emotional_values_stay_none(self):
        episode = _make_valid_episode([1, 1])
        episode['emotional_valence'] = None
        episode['emotional_arousal'] = None
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_valence'] is None
        assert result[0]['emotional_arousal'] is None

    def test_valid_range_values_preserved(self):
        episode = _make_valid_episode([1, 1])
        episode['emotional_valence'] = -0.3
        episode['emotional_arousal'] = 0.7
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_valence'] == pytest.approx(-0.3)
        assert result[0]['emotional_arousal'] == pytest.approx(0.7)


class TestEpisodesSkippedForMissingFields:

    def test_episode_missing_required_field_is_skipped(self):
        episode = _make_valid_episode([1, 1])
        del episode['gist']  # remove required field
        svc = _make_svc(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result == []

    def test_valid_episode_mixed_with_invalid_only_returns_valid(self):
        valid = _make_valid_episode([1, 1])
        invalid = {'intent': {'type': 'exploration'}}  # missing most fields
        svc = _make_svc(json.dumps([valid, invalid]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert len(result) == 1
        assert result[0]['gist'] == valid['gist']


class TestMissingPromptTemplate:

    def test_missing_prompt_returns_empty(self):
        from services.llm_service import LLMResponse

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = LLMResponse(
            text='[]', model='test', provider='mock'
        )

        with patch('services.episode_extractor_service.ConfigService.resolve_agent_config',
                   return_value={}), \
             patch('services.episode_extractor_service._PROMPT_PATH') as mock_path, \
             patch('services.episode_extractor_service.create_llm_service',
                   return_value=mock_llm):
            mock_path.read_text.side_effect = FileNotFoundError("no prompt")
            from services.episode_extractor_service import EpisodeExtractorService
            svc = EpisodeExtractorService()

        entries = [_make_entry(1, content='hello')]
        result = svc.extract(entries, 'test')

        assert result == []
        mock_llm.send_message.assert_not_called()
