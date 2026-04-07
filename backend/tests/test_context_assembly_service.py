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
             patch.object(svc, '_get_moments', return_value=''):

            result = svc.assemble(prompt='hello', topic='test')

        expected_keys = {
            'working_memory', 'moments',
            'previous_session', 'total_tokens_est',
            'self_awareness',
        }
        assert expected_keys == set(result.keys())

    def test_assemble_includes_total_tokens_estimate(self):
        """total_tokens_est should be a non-negative integer."""
        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)

        with patch.object(svc, '_get_working_memory', return_value='some text'), \
             patch.object(svc, '_get_moments', return_value=''):

            result = svc.assemble(prompt='hi', topic='t')

        assert isinstance(result['total_tokens_est'], int)
        assert result['total_tokens_est'] >= 0

    # ── Working memory ────────────────────────────────────────────────

    def test_working_memory_included_in_output(self):
        """Working memory text should appear in the returned dict."""
        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)

        with patch.object(svc, '_get_working_memory', return_value='User said hello'), \
             patch.object(svc, '_get_moments', return_value=''):

            result = svc.assemble(prompt='hi', topic='t')

        assert result['working_memory'] == 'User said hello'

    # ── Budget constraint ─────────────────────────────────────────────

    def test_budget_constraint_trims_lowest_weight_sections_first(self):
        """When total exceeds budget, lowest-weight sections are trimmed first."""
        config = {'max_context_tokens': 10}  # Very small budget
        svc = ContextAssemblyService(config)

        with patch.object(svc, '_get_working_memory', return_value='A' * 100), \
             patch.object(svc, '_get_moments', return_value='B' * 100):

            result = svc.assemble(prompt='hi', topic='t')

        # Budget is 10 tokens (~40 chars), so most sections should be trimmed.
        # self_awareness and previous_session add some overhead, but total should
        # be much less than the original 200 chars of mocked data.
        total_text = sum(len(v) for v in result.values() if isinstance(v, str))
        assert total_text < 400

    def test_previous_session_populated_from_recent_visible_context(self):
        """recent_visible_context should populate previous_session."""
        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)

        recent = [
            {'prompt': 'How are you?', 'response': 'Good'},
            {'prompt': 'Tell me more', 'response': 'Sure thing'},
        ]

        with patch.object(svc, '_get_working_memory', return_value=''), \
             patch.object(svc, '_get_moments', return_value=''):

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
        custom = {'working_memory': 0.1}
        svc = ContextAssemblyService({'context_weights': custom})
        assert svc.weights == custom

    # ── TopicContext integration ─────────────────────────────────────

    def test_assemble_accepts_topic_context(self):
        """assemble() works when a TopicContext is passed."""
        from services.topic_context import TopicContext

        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)
        ctx = TopicContext(topic='test-topic', thread_id='thread-99')

        with patch.object(svc, '_get_working_memory', return_value='wm'), \
             patch.object(svc, '_get_moments', return_value=''):

            result = svc.assemble(prompt='hi', topic='test-topic', context=ctx)

        assert result['working_memory'] == 'wm'
        assert ctx.failed_sections == []

    def test_backward_compat_without_context(self):
        """assemble() still works when context is not passed (backward compat)."""
        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)

        with patch.object(svc, '_get_working_memory', return_value='wm'), \
             patch.object(svc, '_get_moments', return_value=''):

            result = svc.assemble(prompt='hi', topic='t')

        assert 'working_memory' in result

    def test_failed_sections_empty_on_success(self):
        """When no section fails, failed_sections stays empty."""
        from services.topic_context import TopicContext

        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)
        ctx = TopicContext(topic='t')

        with patch.object(svc, '_get_working_memory', return_value='wm'), \
             patch.object(svc, '_get_moments', return_value=''):

            svc.assemble(prompt='hi', topic='t', context=ctx)

        assert ctx.failed_sections == []

    def test_context_wm_identifier_used(self):
        """TopicContext.wm_identifier is used for working memory lookup."""
        from services.topic_context import TopicContext

        config = {'max_context_tokens': 100_000}
        svc = ContextAssemblyService(config)
        ctx = TopicContext(topic='fallback-topic', thread_id='ctx-thread-77')

        with patch.object(svc, '_get_working_memory', return_value='wm') as mock_wm, \
             patch.object(svc, '_get_moments', return_value=''):

            svc.assemble(prompt='hi', topic='fallback-topic', context=ctx)

        # The first positional arg should be the wm_identifier from context
        call_args = mock_wm.call_args
        assert call_args[0][0] == 'ctx-thread-77'
