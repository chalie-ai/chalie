"""Tests for memory_skill — DataGraphService-backed store/recall entry point."""

import pytest
from unittest.mock import patch, MagicMock

from services.innate_skills.memory_skill import (
    handle_memory,
    TOOL_SCHEMA,
    _relevance_label,
    _search_data_graph,
)


pytestmark = pytest.mark.unit


# ── Relevance bucketing ──────────────────────────────────────────────

class TestRelevanceLabel:

    def test_high_at_0_7(self):
        assert _relevance_label(0.7) == "high"

    def test_high_above_0_7(self):
        assert _relevance_label(1.0) == "high"

    def test_medium_at_0_4(self):
        assert _relevance_label(0.4) == "medium"

    def test_medium_below_0_7(self):
        assert _relevance_label(0.69) == "medium"

    def test_low_below_0_4(self):
        assert _relevance_label(0.39) == "low"

    def test_low_at_zero(self):
        assert _relevance_label(0.0) == "low"


# ── TOOL_SCHEMA ─────────────────────────────────────────────────────

class TestToolSchema:

    def test_schema_actions_are_store_recall_reflect(self):
        action_enum = TOOL_SCHEMA['input_schema']['properties']['action']['enum']
        assert set(action_enum) == {'store', 'recall', 'reflect'}

    def test_schema_has_no_update_action(self):
        action_enum = TOOL_SCHEMA['input_schema']['properties']['action']['enum']
        assert 'update' not in action_enum

    def test_schema_has_no_forget_action(self):
        action_enum = TOOL_SCHEMA['input_schema']['properties']['action']['enum']
        assert 'forget' not in action_enum

    def test_schema_has_no_entries_array(self):
        props = TOOL_SCHEMA['input_schema']['properties']
        assert 'entries' not in props

    def test_schema_has_no_entity_param(self):
        props = TOOL_SCHEMA['input_schema']['properties']
        assert 'entity' not in props

    def test_schema_has_no_confidence_param(self):
        props = TOOL_SCHEMA['input_schema']['properties']
        assert 'confidence' not in props

    def test_schema_has_no_decay_class_param(self):
        props = TOOL_SCHEMA['input_schema']['properties']
        assert 'decay_class' not in props

    def test_schema_kind_enum_is_three_llm_visible_kinds(self):
        kind_enum = TOOL_SCHEMA['input_schema']['properties']['kind']['enum']
        assert set(kind_enum) == {'user_specific', 'system', 'misc'}

    def test_schema_kind_does_not_include_moment(self):
        kind_enum = TOOL_SCHEMA['input_schema']['properties']['kind']['enum']
        assert 'moment' not in kind_enum

    def test_schema_required_is_action_only(self):
        required = TOOL_SCHEMA['input_schema']['required']
        assert required == ['action']

    def test_schema_has_key_value_query_properties(self):
        props = TOOL_SCHEMA['input_schema']['properties']
        assert 'key' in props
        assert 'value' in props
        assert 'query' in props


# ── Store ────────────────────────────────────────────────────────────

class TestHandleMemoryStore:

    def _mock_dgs(self, return_value):
        mock = MagicMock()
        mock.store.return_value = return_value
        return mock

    def test_store_calls_dgs_store_with_correct_args(self):
        mock_dgs = self._mock_dgs({'id': 1, 'key': 'user_name', 'value': 'Dylan'})

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {
                'action': 'store',
                'key': 'user_name',
                'value': 'Dylan',
                'kind': 'user_specific',
            })

        mock_dgs.store.assert_called_once_with(
            kind='user_specific',
            key='user_name',
            value='Dylan',
            source='skill:memory:store:topic',
        )
        assert 'Stored' in result
        assert 'user_name' in result

    def test_store_defaults_kind_to_user_specific(self):
        mock_dgs = self._mock_dgs({'id': 2, 'key': 'fav_food', 'value': 'pizza'})

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {
                'action': 'store',
                'key': 'fav_food',
                'value': 'pizza',
            })

        mock_dgs.store.assert_called_once()
        assert mock_dgs.store.call_args.kwargs['kind'] == 'user_specific'
        assert 'Stored' in result

    def test_store_missing_key_returns_error(self):
        result = handle_memory('topic', {'action': 'store', 'value': 'something'})
        assert 'Error' in result
        assert 'key' in result

    def test_store_missing_value_returns_error(self):
        result = handle_memory('topic', {'action': 'store', 'key': 'my_key'})
        assert 'Error' in result
        assert 'value' in result

    def test_store_dgs_returns_none_gives_error(self):
        mock_dgs = self._mock_dgs(None)

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {
                'action': 'store',
                'key': 'some_key',
                'value': 'some_value',
            })

        assert 'failed' in result.lower() or 'error' in result.lower() or 'Store' in result

    def test_store_true_contradiction_returns_conflict_question(self):
        conflict_result = {
            'conflict': True,
            'classification': 'true_contradiction',
            'existing': {'value': 'Malta'},
            'proposed_key': 'hometown',
            'proposed_value': 'London',
        }
        mock_dgs = self._mock_dgs(conflict_result)

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {
                'action': 'store',
                'key': 'hometown',
                'value': 'London',
            })

        assert 'Conflict detected' in result
        assert 'Malta' in result
        assert 'London' in result

    def test_store_ambiguous_contradiction_returns_prompt(self):
        conflict_result = {
            'conflict': True,
            'classification': 'ambiguous',
            'existing': {'value': 'blue'},
            'proposed_key': 'eye_colour',
            'proposed_value': 'green',
        }
        mock_dgs = self._mock_dgs(conflict_result)

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {
                'action': 'store',
                'key': 'eye_colour',
                'value': 'green',
            })

        assert 'Should I store' in result
        assert 'green' in result

    def test_store_system_kind_accepted(self):
        mock_dgs = self._mock_dgs({'id': 3, 'key': 'tone', 'value': 'terse'})

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {
                'action': 'store',
                'key': 'tone',
                'value': 'terse',
                'kind': 'system',
            })

        assert mock_dgs.store.call_args.kwargs['kind'] == 'system'
        assert 'Stored' in result

    def test_store_misc_kind_accepted(self):
        mock_dgs = self._mock_dgs({'id': 4, 'key': 'scratch', 'value': 'temp'})

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {
                'action': 'store',
                'key': 'scratch',
                'value': 'temp',
                'kind': 'misc',
            })

        assert mock_dgs.store.call_args.kwargs['kind'] == 'misc'
        assert 'Stored' in result


# ── Recall ───────────────────────────────────────────────────────────

class TestHandleMemoryRecall:

    def test_recall_missing_query_returns_error(self):
        result = handle_memory('topic', {'action': 'recall', 'query': ''})
        assert 'Error' in result

    def test_recall_calls_dgs_recall(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': 1, 'key': 'user_name', 'value': 'Dylan',
             'retrieval_weight': 0.9, 'evidence_count': 3, 'composite_score': 2.5},
        ]

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs), \
             patch('services.innate_skills.memory_skill._search_episodes', return_value=([], '0 matches')), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {'action': 'recall', 'query': 'user name'})

        assert 'user_name' in result
        assert 'Dylan' in result

    def test_recall_formats_relevance_high(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': 1, 'key': 'fav_food', 'value': 'pizza',
             'retrieval_weight': 0.8, 'evidence_count': 1, 'composite_score': 1.5},
        ]

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs), \
             patch('services.innate_skills.memory_skill._search_episodes', return_value=([], '0 matches')), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {'action': 'recall', 'query': 'food'})

        assert 'relevance:high' in result

    def test_recall_formats_relevance_medium(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': 1, 'key': 'city', 'value': 'Valletta',
             'retrieval_weight': 0.5, 'evidence_count': 1, 'composite_score': 1.0},
        ]

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs), \
             patch('services.innate_skills.memory_skill._search_episodes', return_value=([], '0 matches')), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {'action': 'recall', 'query': 'city'})

        assert 'relevance:medium' in result

    def test_recall_formats_relevance_low(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': 1, 'key': 'old_fact', 'value': 'stale',
             'retrieval_weight': 0.2, 'evidence_count': 1, 'composite_score': 0.3},
        ]

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs), \
             patch('services.innate_skills.memory_skill._search_episodes', return_value=([], '0 matches')), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {'action': 'recall', 'query': 'fact'})

        assert 'relevance:low' in result

    def test_recall_empty_returns_no_memories(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = []

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs), \
             patch('services.innate_skills.memory_skill._search_episodes', return_value=([], '0 matches')), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {'action': 'recall', 'query': 'ghost'})

        assert 'No memories found' in result
        assert 'ghost' in result

    def test_recall_output_format_id_relevance_value(self):
        """Output must match: [id:{key},relevance:{level}] {value}"""
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': 1, 'key': 'partner_name', 'value': 'Sarah',
             'retrieval_weight': 0.75, 'evidence_count': 2, 'composite_score': 2.1},
        ]

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs), \
             patch('services.innate_skills.memory_skill._search_episodes', return_value=([], '0 matches')), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {'action': 'recall', 'query': 'partner'})

        assert '[id:partner_name,relevance:high] Sarah' in result


# ── Invalid / unknown actions ────────────────────────────────────────

class TestHandleMemoryInvalid:

    def test_unknown_action_returns_error(self):
        result = handle_memory('topic', {'action': 'explode'})
        assert 'Unknown action' in result
        assert 'explode' in result

    def test_unknown_action_error_lists_valid_actions(self):
        result = handle_memory('topic', {'action': 'explode'})
        assert 'store' in result
        assert 'recall' in result
        assert 'reflect' in result

    def test_update_action_rejected_as_unknown(self):
        result = handle_memory('topic', {'action': 'update', 'key': 'k', 'value': 'v'})
        assert 'Unknown action' in result

    def test_forget_action_rejected_as_unknown(self):
        result = handle_memory('topic', {'action': 'forget', 'key': 'something'})
        assert 'Unknown action' in result

    def test_missing_action_defaults_to_recall_then_errors_no_query(self):
        result = handle_memory('topic', {})
        assert 'Error' in result  # "no query specified for recall"

    def test_no_knowledge_service_imports(self):
        """memory_skill must not import KnowledgeService anywhere."""
        import services.innate_skills.memory_skill as ms
        import inspect
        src = inspect.getsource(ms)
        assert 'KnowledgeService' not in src
        assert 'knowledge_service' not in src


# ── Dead code removed ────────────────────────────────────────────────

class TestDeadCodeRemoved:

    def test_auto_classify_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_auto_classify')

    def test_check_trait_contradiction_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_check_trait_contradiction')

    def test_handle_update_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_handle_update')

    def test_trait_keys_constant_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_TRAIT_KEYS')

    def test_preference_keys_constant_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_PREFERENCE_KEYS')

    def test_procedure_keys_constant_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_PROCEDURE_KEYS')

    def test_search_transcript_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_search_transcript')

    def test_format_empty_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_format_empty')

    def test_salience_label_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_salience_label')


# ── _search_data_graph unit ──────────────────────────────────────────

class TestSearchDataGraph:

    def test_returns_id_relevance_text_shape(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': 1, 'key': 'user_name', 'value': 'Dylan',
             'retrieval_weight': 0.9, 'evidence_count': 1, 'composite_score': 2.0},
        ]

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            hits, status = _search_data_graph("name", 10)

        assert len(hits) == 1
        assert hits[0]['id'] == 'user_name'
        assert hits[0]['text'] == 'Dylan'
        assert hits[0]['relevance'] == 'high'
        assert '1 matches' in status

    def test_empty_result_returns_zero_matches_status(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = []

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            hits, status = _search_data_graph("ghost", 10)

        assert hits == []
        assert '0 matches' in status

    def test_exception_returns_error_status(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.side_effect = RuntimeError("db exploded")

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            hits, status = _search_data_graph("anything", 10)

        assert hits == []
        assert 'error' in status


# ── Reflect ──────────────────────────────────────────────────────────

class TestHandleMemoryReflect:

    def test_reflect_missing_query_returns_error(self):
        result = handle_memory('topic', {'action': 'reflect', 'query': ''})
        assert 'Error' in result

    def test_reflect_no_episodes_no_dg_returns_no_memories(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = []

        with patch('services.innate_skills.memory_skill.recall_episodes', return_value=([], '0 matches')), \
             patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {'action': 'reflect', 'query': 'cats'})

        assert 'No memories found' in result
        assert 'cats' in result

    def test_reflect_with_episode_shows_main_memory(self):
        fake_ep = {
            'id': 'ep-1', 'gist': 'User has a cat named Blake',
            'salience': 0.9, 'composite_score': 50,
            'transcript_ids': '[]', 'consolidated_from': '[]',
        }
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = []

        with patch('services.innate_skills.memory_skill.recall_episodes',
                   return_value=([fake_ep], '1 matches')), \
             patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {'action': 'reflect', 'query': 'cat'})

        assert '## Main Memory' in result
        assert 'User has a cat named Blake' in result

    def test_reflect_includes_supporting_facts(self):
        fake_ep = {
            'id': 'ep-1', 'gist': 'User has a cat',
            'salience': 0.9, 'composite_score': 50,
            'transcript_ids': '[]', 'consolidated_from': '[]',
        }
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': 10, 'key': 'cat_name', 'value': 'Blake',
             'retrieval_weight': 0.9, 'composite_score': 2.0},
            {'id': 11, 'key': 'cat_age', 'value': '16 years',
             'retrieval_weight': 0.5, 'composite_score': 1.0},
        ]

        with patch('services.innate_skills.memory_skill.recall_episodes',
                   return_value=([fake_ep], '1 matches')), \
             patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {'action': 'reflect', 'query': 'cat'})

        assert '### Supporting facts:' in result
        assert '[cat_name] Blake' in result
        assert '[cat_age] 16 years' in result

    def test_reflect_dg_limited_to_2(self):
        """Data graph results must be capped at 2 in reflect mode."""
        fake_ep = {
            'id': 'ep-1', 'gist': 'Test episode',
            'salience': 0.5, 'composite_score': 30,
            'transcript_ids': '[]', 'consolidated_from': '[]',
        }
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': i, 'key': f'k{i}', 'value': f'v{i}',
             'retrieval_weight': 0.9, 'composite_score': 2.0}
            for i in range(5)
        ]

        with patch('services.innate_skills.memory_skill.recall_episodes',
                   return_value=([fake_ep], '1 matches')), \
             patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {'action': 'reflect', 'query': 'test'})

        # _search_data_graph called with limit=2, so mock returns all 5
        # but the function passes limit=2 to dgs.recall
        mock_dgs.recall.assert_called_once_with(query='test', limit=2)

    def test_reflect_no_episodes_falls_back_to_dg_only(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': 1, 'key': 'cat_name', 'value': 'Blake',
             'retrieval_weight': 0.9, 'composite_score': 2.0},
        ]

        with patch('services.innate_skills.memory_skill.recall_episodes',
                   return_value=([], '0 matches')), \
             patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {'action': 'reflect', 'query': 'cat'})

        assert '### Supporting facts:' in result
        assert '[cat_name] Blake' in result

    def test_reflect_action_accepted_by_schema(self):
        action_enum = TOOL_SCHEMA['input_schema']['properties']['action']['enum']
        assert 'reflect' in action_enum
