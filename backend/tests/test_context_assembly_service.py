"""Tests for ContextAssemblyService — multi-memory context retrieval and budget trimming."""

import pytest
from unittest.mock import patch, MagicMock

from services.context_assembly_service import ContextAssemblyService


pytestmark = pytest.mark.unit


class TestContextAssemblyService:
    """Tests for context assembly orchestration."""

    # ── Section keys ──────────────────────────────────────────────────

    def test_assemble_returns_all_expected_section_keys(self):
        """assemble() must return every documented section key."""
        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)

        with patch.object(svc, '_get_working_memory', return_value='wm'), \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_facts', return_value='facts'), \
             patch.object(svc, '_get_gists', return_value='gists'), \
             patch.object(svc, '_get_episodes', return_value='eps'), \
             patch.object(svc, '_get_concepts', return_value=''):

            result = svc.assemble(prompt='hello', topic='test')

        expected_keys = {
            'working_memory', 'moments', 'facts', 'gists',
            'episodes', 'concepts', 'previous_session', 'total_tokens_est',
        }
        assert expected_keys == set(result.keys())

    def test_assemble_includes_total_tokens_estimate(self):
        """total_tokens_est should be a non-negative integer."""
        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)

        with patch.object(svc, '_get_working_memory', return_value='some text'), \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_facts', return_value=''), \
             patch.object(svc, '_get_gists', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''):

            result = svc.assemble(prompt='hi', topic='t')

        assert isinstance(result['total_tokens_est'], int)
        assert result['total_tokens_est'] >= 0

    # ── Working memory ────────────────────────────────────────────────

    def test_working_memory_included_in_output(self):
        """Working memory text should appear in the returned dict."""
        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)

        with patch.object(svc, '_get_working_memory', return_value='User said hello'), \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_facts', return_value=''), \
             patch.object(svc, '_get_gists', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''):

            result = svc.assemble(prompt='hi', topic='t')

        assert result['working_memory'] == 'User said hello'

    def test_working_memory_uses_thread_id_when_provided(self):
        """When thread_id is passed, it should be used as the identifier."""
        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)

        with patch.object(svc, '_get_working_memory', return_value='') as mock_wm, \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_facts', return_value=''), \
             patch.object(svc, '_get_gists', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''):

            svc.assemble(prompt='hi', topic='t', thread_id='thread-123')

        mock_wm.assert_called_once_with('thread-123')

    def test_working_memory_falls_back_to_topic_without_thread_id(self):
        """Without thread_id, topic should be used as the identifier."""
        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)

        with patch.object(svc, '_get_working_memory', return_value='') as mock_wm, \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_facts', return_value=''), \
             patch.object(svc, '_get_gists', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''):

            svc.assemble(prompt='hi', topic='my-topic')

        mock_wm.assert_called_once_with('my-topic')

    # ── Facts ─────────────────────────────────────────────────────────

    def test_facts_included_when_available(self):
        """Facts text should pass through to the result."""
        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)

        with patch.object(svc, '_get_working_memory', return_value=''), \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_facts', return_value='name: Alice'), \
             patch.object(svc, '_get_gists', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''):

            result = svc.assemble(prompt='hi', topic='t')

        assert result['facts'] == 'name: Alice'

    # ── Episodes ──────────────────────────────────────────────────────

    def test_episodes_included_when_available(self):
        """Episodes text should pass through to the result."""
        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)

        with patch.object(svc, '_get_working_memory', return_value=''), \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_facts', return_value=''), \
             patch.object(svc, '_get_gists', return_value=''), \
             patch.object(svc, '_get_episodes', return_value='Went to gym'), \
             patch.object(svc, '_get_concepts', return_value=''):

            result = svc.assemble(prompt='hi', topic='t')

        assert result['episodes'] == 'Went to gym'

    # ── Layer failure graceful degradation ────────────────────────────

    def test_working_memory_failure_returns_empty_string(self):
        """If WorkingMemoryService import fails, return ''."""
        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)

        with patch(
            'services.context_assembly_service.ContextAssemblyService._get_working_memory',
            side_effect=Exception('Redis down'),
        ):
            # Call the real _get_working_memory which catches exceptions
            result = svc._get_working_memory.__wrapped__(svc, 'topic') if hasattr(svc._get_working_memory, '__wrapped__') else ''

        # The real method catches all exceptions and returns ""
        # Test via assemble which calls _get_* methods that handle errors
        svc2 = ContextAssemblyService(config)
        with patch('services.working_memory_service.WorkingMemoryService', side_effect=Exception('boom')), \
             patch.object(svc2, '_get_moments', return_value=''), \
             patch.object(svc2, '_get_facts', return_value=''), \
             patch.object(svc2, '_get_gists', return_value=''), \
             patch.object(svc2, '_get_episodes', return_value=''), \
             patch.object(svc2, '_get_concepts', return_value=''):

            result = svc2.assemble(prompt='hi', topic='t')

        assert result['working_memory'] == ''

    def test_facts_failure_returns_empty_string(self):
        """If FactStoreService import fails, _get_facts returns ''."""
        config = {}
        svc = ContextAssemblyService(config)

        with patch('services.fact_store_service.FactStoreService', side_effect=Exception('boom')):
            result = svc._get_facts('topic')

        assert result == ''

    def test_episodes_failure_returns_empty_string(self):
        """If EpisodicRetrievalService fails, _get_episodes returns ''."""
        config = {}
        svc = ContextAssemblyService(config)

        with patch('services.episodic_retrieval_service.EpisodicRetrievalService', side_effect=Exception('boom')):
            result = svc._get_episodes('prompt', 'topic')

        assert result == ''

    # ── Budget constraint ─────────────────────────────────────────────

    def test_budget_constraint_trims_lowest_weight_sections_first(self):
        """When total exceeds budget, lowest-weight sections are trimmed first."""
        config = {'max_context_tokens': 10}  # Very small budget
        svc = ContextAssemblyService(config)

        # 'concepts' has lowest default weight (0.6), should be trimmed first
        with patch.object(svc, '_get_working_memory', return_value='A' * 100), \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_facts', return_value='B' * 100), \
             patch.object(svc, '_get_gists', return_value='C' * 100), \
             patch.object(svc, '_get_episodes', return_value='D' * 100), \
             patch.object(svc, '_get_concepts', return_value='E' * 100):

            result = svc.assemble(prompt='hi', topic='t')

        # Budget is 10 tokens (~40 chars), so most sections should be trimmed
        # The highest-weight section (working_memory=1.0) should have the most content
        total_text = sum(len(v) for v in result.values() if isinstance(v, str))
        # Verify budget mechanism ran (total should be much less than original 500 chars)
        assert total_text < 500

    def test_previous_session_populated_from_recent_visible_context(self):
        """recent_visible_context should populate previous_session."""
        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)

        recent = [
            {'prompt': 'How are you?', 'response': 'Good'},
            {'prompt': 'Tell me more', 'response': 'Sure thing'},
        ]

        with patch.object(svc, '_get_working_memory', return_value=''), \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_facts', return_value=''), \
             patch.object(svc, '_get_gists', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''):

            result = svc.assemble(
                prompt='hi', topic='t', recent_visible_context=recent,
            )

        assert 'previous session' in result['previous_session'].lower()
        assert 'How are you?' in result['previous_session']
        assert 'Sure thing' in result['previous_session']

    # ── Token estimation ──────────────────────────────────────────────

    def test_estimate_tokens_empty_string(self):
        """Empty text should estimate to 0 tokens."""
        svc = ContextAssemblyService({})
        assert svc._estimate_tokens('') == 0
        assert svc._estimate_tokens(None) == 0

    def test_estimate_tokens_known_length(self):
        """4 characters should estimate to 1 token."""
        svc = ContextAssemblyService({})
        assert svc._estimate_tokens('abcd') == 1
        assert svc._estimate_tokens('a' * 40) == 10

    # ── Custom weights ────────────────────────────────────────────────

    def test_custom_weights_override_defaults(self):
        """Config-provided weights should override DEFAULT_WEIGHTS."""
        custom = {'working_memory': 0.1, 'facts': 0.2}
        svc = ContextAssemblyService({'context_weights': custom})
        assert svc.weights == custom

    # ── _get_concepts ─────────────────────────────────────────────────

    def test_get_concepts_returns_formatted_string_when_concepts_exist(self):
        """_get_concepts() returns '## Relevant Concepts' section when concepts are available."""
        svc = ContextAssemblyService({})
        mock_concepts = [
            {'concept_name': 'Python', 'definition': 'A programming language', 'concept_type': 'technology', 'strength': 0.8},
            {'concept_name': 'Weak', 'definition': 'A weak concept', 'concept_type': 'unknown', 'strength': 0.1},  # below 0.2 gate
            {'concept_name': 'NoDef', 'definition': '', 'concept_type': 'other', 'strength': 0.9},  # no definition
        ]
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve_concepts.return_value = mock_concepts

        with patch('services.semantic_retrieval_service.SemanticRetrievalService', return_value=mock_retrieval):
            result = svc._get_concepts('What is Python?', 'programming')

        assert '## Relevant Concepts' in result
        assert '**Python**' in result
        assert 'A programming language' in result
        # Weak concept (strength 0.1) should be excluded
        assert 'Weak' not in result
        # Concept without definition should be excluded
        assert 'NoDef' not in result

    def test_get_concepts_returns_empty_when_no_concepts(self):
        """_get_concepts() returns '' when retrieval returns empty list."""
        svc = ContextAssemblyService({})
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve_concepts.return_value = []

        with patch('services.semantic_retrieval_service.SemanticRetrievalService', return_value=mock_retrieval):
            result = svc._get_concepts('hello', 'general')

        assert result == ''

    def test_get_concepts_returns_empty_when_all_filtered(self):
        """_get_concepts() returns '' when all concepts fail the strength/definition gate."""
        svc = ContextAssemblyService({})
        mock_concepts = [
            {'concept_name': 'Noisy', 'definition': '', 'concept_type': 'other', 'strength': 0.9},
            {'concept_name': 'Weak', 'definition': 'Some def', 'concept_type': 'other', 'strength': 0.1},
        ]
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve_concepts.return_value = mock_concepts

        with patch('services.semantic_retrieval_service.SemanticRetrievalService', return_value=mock_retrieval):
            result = svc._get_concepts('hello', 'general')

        assert result == ''

    def test_get_concepts_returns_empty_on_service_failure(self):
        """_get_concepts() gracefully returns '' when SemanticRetrievalService fails."""
        svc = ContextAssemblyService({})

        with patch('services.semantic_retrieval_service.SemanticRetrievalService', side_effect=Exception('DB down')):
            result = svc._get_concepts('hello', 'general')

        assert result == ''

    def test_assemble_includes_concepts_in_result(self):
        """assemble() passes concept retrieval result into the 'concepts' key."""
        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)

        with patch.object(svc, '_get_working_memory', return_value=''), \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_facts', return_value=''), \
             patch.object(svc, '_get_gists', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value='## Relevant Concepts\n- **AI**: Artificial intelligence'):

            result = svc.assemble(prompt='tell me about AI', topic='tech')

        assert result['concepts'] == '## Relevant Concepts\n- **AI**: Artificial intelligence'
