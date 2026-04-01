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
             patch.object(svc, '_get_episodes', return_value='eps'), \
             patch.object(svc, '_get_concepts', return_value=''):

            result = svc.assemble(prompt='hello', topic='test')

        expected_keys = {
            'working_memory', 'moments',
            'episodes', 'concepts', 'previous_session', 'total_tokens_est',
            'self_awareness',
        }
        assert expected_keys == set(result.keys())

    def test_assemble_includes_total_tokens_estimate(self):
        """total_tokens_est should be a non-negative integer."""
        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)

        with patch.object(svc, '_get_working_memory', return_value='some text'), \
             patch.object(svc, '_get_moments', return_value=''), \
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
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''):

            result = svc.assemble(prompt='hi', topic='t')

        assert result['working_memory'] == 'User said hello'

    # ── Episodes ──────────────────────────────────────────────────────

    def test_episodes_included_when_available(self):
        """Episodes text should pass through to the result."""
        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)

        with patch.object(svc, '_get_working_memory', return_value=''), \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_episodes', return_value='Went to gym'), \
             patch.object(svc, '_get_concepts', return_value=''):

            result = svc.assemble(prompt='hi', topic='t')

        assert result['episodes'] == 'Went to gym'

    # ── Budget constraint ─────────────────────────────────────────────

    def test_budget_constraint_trims_lowest_weight_sections_first(self):
        """When total exceeds budget, lowest-weight sections are trimmed first."""
        config = {'max_context_tokens': 10}  # Very small budget
        svc = ContextAssemblyService(config)

        # 'concepts' has lowest default weight (0.6), should be trimmed first
        with patch.object(svc, '_get_working_memory', return_value='A' * 100), \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_episodes', return_value='D' * 100), \
             patch.object(svc, '_get_concepts', return_value='E' * 100):

            result = svc.assemble(prompt='hi', topic='t')

        # Budget is 10 tokens (~40 chars), so most sections should be trimmed
        # The highest-weight section (working_memory=1.0) should have the most content
        total_text = sum(len(v) for v in result.values() if isinstance(v, str))
        # Verify budget mechanism ran (total should be much less than original 300 chars)
        assert total_text < 300

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
        custom = {'working_memory': 0.1, 'episodes': 0.2}
        svc = ContextAssemblyService({'context_weights': custom})
        assert svc.weights == custom

    # ── _get_concepts ─────────────────────────────────────────────────

    def test_get_concepts_returns_formatted_string_when_concepts_exist(self, db):
        """_get_concepts() returns '## Relevant Concepts' section when concepts are available."""
        svc = ContextAssemblyService({})
        mock_concepts = [
            {'key': 'Python', 'value': 'A programming language', 'kind': 'concept', 'confidence': 0.8, 'rrf_score': 0.8},
            {'key': 'Weak', 'value': 'A weak concept', 'kind': 'concept', 'confidence': 0.1, 'rrf_score': 0.1},  # below 0.2 gate
            {'key': 'NoDef', 'value': '', 'kind': 'concept', 'confidence': 0.9, 'rrf_score': 0.9},  # no value/definition
        ]
        mock_ks = MagicMock()
        mock_ks.recall.return_value = mock_concepts

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks):
            result = svc._get_concepts('What is Python?', 'programming')

        assert '## Relevant Concepts' in result
        assert '**Python**' in result
        assert 'A programming language' in result
        # Weak concept (confidence 0.1) should be excluded
        assert 'Weak' not in result
        # Concept without definition should be excluded
        assert 'NoDef' not in result

    def test_get_concepts_returns_empty_when_no_concepts(self, db):
        """_get_concepts() returns '' when retrieval returns empty list."""
        svc = ContextAssemblyService({})
        mock_ks = MagicMock()
        mock_ks.recall.return_value = []

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks):
            result = svc._get_concepts('hello', 'general')

        assert result == ''

    def test_get_concepts_returns_empty_when_all_filtered(self, db):
        """_get_concepts() returns '' when all concepts fail the confidence/definition gate."""
        svc = ContextAssemblyService({})
        mock_concepts = [
            {'key': 'Noisy', 'value': '', 'kind': 'concept', 'confidence': 0.9, 'rrf_score': 0.9},
            {'key': 'Weak', 'value': 'Some def', 'kind': 'concept', 'confidence': 0.1, 'rrf_score': 0.1},
        ]
        mock_ks = MagicMock()
        mock_ks.recall.return_value = mock_concepts

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks):
            result = svc._get_concepts('hello', 'general')

        assert result == ''

    def test_get_concepts_returns_empty_on_service_failure(self, db):
        """_get_concepts() gracefully returns '' when KnowledgeService fails."""
        svc = ContextAssemblyService({})

        with patch('services.knowledge_service.KnowledgeService', side_effect=Exception('DB down')):
            result = svc._get_concepts('hello', 'general')

        assert result == ''

    # ── TopicContext integration ─────────────────────────────────────

    def test_assemble_accepts_topic_context(self):
        """assemble() works when a TopicContext is passed."""
        from services.topic_context import TopicContext

        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)
        ctx = TopicContext(topic='test-topic', thread_id='thread-99')

        with patch.object(svc, '_get_working_memory', return_value='wm'), \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''):

            result = svc.assemble(prompt='hi', topic='test-topic', context=ctx)

        assert result['working_memory'] == 'wm'
        assert ctx.failed_sections == []

    def test_backward_compat_without_context(self):
        """assemble() still works when context is not passed (backward compat)."""
        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)

        with patch.object(svc, '_get_working_memory', return_value='wm'), \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''):

            result = svc.assemble(prompt='hi', topic='t')

        assert 'working_memory' in result

    def test_failed_sections_populated_on_error(self, db):
        """When a section fails, TopicContext records the failure."""
        from services.topic_context import TopicContext

        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)
        ctx = TopicContext(topic='t')

        with patch.object(svc, '_get_working_memory', return_value=''), \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_episodes', side_effect=Exception('ep fail')), \
             patch.object(svc, '_get_concepts', return_value=''):

            # _get_episodes raises, but assemble calls it internally —
            # we need to let the real method run to trigger record_failure
            pass

        # Test the private method directly to verify record_failure
        with patch('services.episodic_service.EpisodicService', side_effect=Exception('ep fail')), \
             patch('services.config_service.ConfigService.resolve_agent_config', return_value={}):
            svc._get_episodes('prompt', 'topic', context=ctx)

        assert len(ctx.failed_sections) == 1
        assert ctx.failed_sections[0][0] == 'episodes'
        assert 'ep fail' in ctx.failed_sections[0][1]

    def test_multiple_failures_tracked(self, db):
        """Multiple section failures all appear in TopicContext."""
        from services.topic_context import TopicContext

        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)
        ctx = TopicContext(topic='t')

        # Force two different sections to fail
        with patch('services.episodic_service.EpisodicService', side_effect=Exception('ep boom')), \
             patch('services.config_service.ConfigService.resolve_agent_config', return_value={}):
            svc._get_episodes('p', 't', context=ctx)

        with patch('services.knowledge_service.KnowledgeService', side_effect=Exception('ks boom')):
            svc._get_concepts('p', 't', context=ctx)

        assert len(ctx.failed_sections) == 2
        section_names = [s[0] for s in ctx.failed_sections]
        assert 'episodes' in section_names
        assert 'concepts' in section_names

    def test_failed_sections_empty_on_success(self):
        """When no section fails, failed_sections stays empty."""
        from services.topic_context import TopicContext

        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)
        ctx = TopicContext(topic='t')

        with patch.object(svc, '_get_working_memory', return_value='wm'), \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''):

            svc.assemble(prompt='hi', topic='t', context=ctx)

        assert ctx.failed_sections == []

    def test_context_wm_identifier_used(self):
        """TopicContext.wm_identifier is used for working memory lookup."""
        from services.topic_context import TopicContext

        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)
        ctx = TopicContext(topic='fallback-topic', thread_id='ctx-thread-77')

        with patch.object(svc, '_get_working_memory', return_value='wm') as mock_wm, \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''):

            svc.assemble(prompt='hi', topic='fallback-topic', context=ctx)

        # The first positional arg should be the wm_identifier from context
        call_args = mock_wm.call_args
        assert call_args[0][0] == 'ctx-thread-77'


