"""Tests for Phase 4E: Budget-aware working memory from transcript + compaction.

Tests cover:
- Compaction text included in working memory
- Recent transcript entries included
- Token budget constrains entries
- Falls back to MemoryStore when no transcript data
- Tool name formatting
"""

import pytest
from unittest.mock import patch, MagicMock
from services.context_assembly_service import ContextAssemblyService


def _make_service(**overrides):
    config = {'max_context_tokens': 4000}
    config.update(overrides)
    return ContextAssemblyService(config)


def _make_entries(count, content_template='Message {}', start_id=1):
    entries = []
    for i in range(count):
        entries.append({
            'id': start_id + i,
            'role': 'user' if i % 2 == 0 else 'assistant',
            'content': content_template.format(i),
            'tool_name': None,
        })
    return entries


class TestTranscriptBasedWorkingMemory:
    def test_compaction_text_included(self):
        svc = _make_service()
        compaction = {
            'compacted_text': 'User prefers dark mode. Lives in Malta.',
            'compacted_up_to_id': 10,
            'token_count': 20,
        }
        with patch('services.compaction_service.get_compaction', return_value=compaction), \
             patch('services.transcript_service.get_recent', return_value=[]):
            result = svc._get_working_memory('thread-1', 'test-topic')

        assert '## Conversation History Summary' in result
        assert 'User prefers dark mode' in result

    def test_recent_entries_included(self):
        svc = _make_service()
        entries = _make_entries(4)

        with patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.transcript_service.get_recent', return_value=entries):
            result = svc._get_working_memory('thread-1', 'test-topic')

        assert '## Recent Conversation' in result
        assert 'Message 0' in result
        assert 'Message 3' in result

    def test_compaction_plus_entries(self):
        svc = _make_service()
        compaction = {
            'compacted_text': 'Summary of older conversation',
            'compacted_up_to_id': 5,
            'token_count': 10,
        }
        entries = _make_entries(3, start_id=6)

        with patch('services.compaction_service.get_compaction', return_value=compaction), \
             patch('services.transcript_service.get_recent', return_value=entries):
            result = svc._get_working_memory('thread-1', 'test-topic')

        assert '## Conversation History Summary' in result
        assert 'Summary of older conversation' in result
        assert '## Recent Conversation' in result
        assert 'Message 0' in result

    def test_token_budget_constrains_entries(self):
        svc = _make_service(max_context_tokens=100)
        # Each entry ~6.5 tokens ('Message X' = 2 words * 1.3).
        # Budget = 100 // 2 = 50 tokens for turns.
        # Create many entries to exceed budget.
        long_entries = []
        for i in range(20):
            long_entries.append({
                'id': i + 1,
                'role': 'user',
                'content': ' '.join([f'word{j}' for j in range(30)]),
                'tool_name': None,
            })

        with patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.transcript_service.get_recent', return_value=long_entries):
            result = svc._get_working_memory('thread-1', 'test-topic')

        # Should NOT include all 20 entries — budget should cap it
        lines = [l for l in result.split('\n') if l.startswith('User:')]
        assert len(lines) < 20
        assert len(lines) > 0

    def test_most_recent_entries_prioritized(self):
        svc = _make_service(max_context_tokens=40)
        # Very tight budget — should select only the most recent entries
        entries = _make_entries(10, content_template='Turn number {}')

        with patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.transcript_service.get_recent', return_value=entries):
            result = svc._get_working_memory('thread-1', 'test-topic')

        # Most recent entries should be present, oldest may be dropped
        assert 'Turn number 9' in result

    def test_tool_name_formatting(self):
        svc = _make_service()
        entries = [{
            'id': 1,
            'role': 'tool',
            'content': 'Search results for Malta weather',
            'tool_name': 'web_search',
        }]

        with patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.transcript_service.get_recent', return_value=entries):
            result = svc._get_working_memory('thread-1', 'test-topic')

        assert 'Tool (web_search):' in result


class TestLegacyFallback:
    def test_falls_back_when_no_transcript_data(self):
        svc = _make_service()

        with patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.transcript_service.get_recent', return_value=[]), \
             patch.object(svc, '_get_working_memory_legacy', return_value='Legacy WM') as mock_legacy:
            result = svc._get_working_memory('thread-1', 'test-topic')

        mock_legacy.assert_called_once_with('thread-1')
        assert result == 'Legacy WM'

    def test_falls_back_on_import_error(self):
        svc = _make_service()

        with patch('services.compaction_service.get_compaction', side_effect=ImportError('no module')), \
             patch.object(svc, '_get_working_memory_legacy', return_value='Fallback') as mock_legacy:
            result = svc._get_working_memory('thread-1', 'test-topic')

        mock_legacy.assert_called_once()
        assert result == 'Fallback'

    def test_uses_topic_when_no_thread_id(self):
        svc = _make_service()
        entries = _make_entries(2)

        with patch('services.compaction_service.get_compaction', return_value=None) as mock_comp, \
             patch('services.transcript_service.get_recent', return_value=entries) as mock_recent:
            svc._get_working_memory('my-topic', None)

        # Should use identifier ('my-topic') as effective_topic since topic=None
        mock_comp.assert_called_once_with('my-topic')


class TestAssembleIntegration:
    def test_assemble_passes_topic_to_working_memory(self):
        svc = _make_service()

        with patch.object(svc, '_get_working_memory', return_value='wm content') as mock_wm, \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_procedural_hints', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''), \
             patch('services.self_model_service.SelfModelService') as mock_sm:
            mock_sm.return_value.has_noteworthy_state.return_value = False
            result = svc.assemble(
                prompt='hello',
                topic='my-topic',
                thread_id='thread-123',
            )

        # Should pass both thread_id and topic
        mock_wm.assert_called_once_with('thread-123', 'my-topic')
        assert result['working_memory'] == 'wm content'
