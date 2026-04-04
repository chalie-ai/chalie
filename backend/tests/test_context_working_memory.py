"""Tests for Phase 4E: Budget-aware working memory from transcript + compaction.

Tests cover:
- Compaction text included in working memory
- Recent transcript entries included
- Token budget constrains entries
- Falls back to MemoryStore when no transcript data
- Tool name formatting
"""

from unittest.mock import patch
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
        lines = [line for line in result.split('\n') if line.startswith('User:')]
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
            'tool_name': 'search',
        }]

        with patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.transcript_service.get_recent', return_value=entries):
            result = svc._get_working_memory('thread-1', 'test-topic')

        assert 'Tool (search):' in result


class TestTopicContextWorkingMemory:
    def test_records_failure_on_compaction_error(self):
        """When compaction raises, failure is recorded on TopicContext."""
        from services.topic_context import TopicContext

        svc = _make_service()
        ctx = TopicContext(topic='test-topic', thread_id='thread-1')

        with patch('services.compaction_service.get_compaction', side_effect=Exception('db locked')), \
             patch.object(svc, '_get_working_memory_legacy', return_value=''):
            svc._get_working_memory('thread-1', 'test-topic', context=ctx)

        assert len(ctx.failed_sections) == 1
        assert ctx.failed_sections[0][0] == 'working_memory'
        assert 'db locked' in ctx.failed_sections[0][1]

    def test_wm_identifier_from_topic_context(self):
        """TopicContext.wm_identifier is used when context is passed to assemble."""
        from services.topic_context import TopicContext

        svc = _make_service()
        ctx = TopicContext(topic='general', thread_id='thread-42')

        with patch.object(svc, '_get_working_memory', return_value='wm') as mock_wm, \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''):

            svc.assemble(prompt='hello', topic='general', context=ctx)

        # First positional arg should be wm_identifier = thread_id
        assert mock_wm.call_args[0][0] == 'thread-42'
