"""Tests for Phase 4D: Compaction Service.

Tests cover:
- check_and_compact() trigger logic
- get_compaction() retrieval
- get_entries_since() delegation
- _run_compaction() LLM call and storage
"""

import pytest
from unittest.mock import patch, MagicMock


def _insert_entries(conn, topic, count, content_template='Message {}'):
    """Insert transcript entries and return their IDs."""
    ids = []
    for i in range(count):
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO topic_transcript (topic, role, content) VALUES (?, ?, ?)",
            (topic, 'user' if i % 2 == 0 else 'assistant', content_template.format(i)),
        )
        ids.append(cursor.lastrowid)
    conn.commit()
    return ids


class TestGetCompaction:
    def test_returns_none_when_no_compaction(self, db):
        from services.compaction_service import get_compaction
        assert get_compaction('nonexistent-topic') is None

    def test_returns_stored_compaction(self, db):
        from services.compaction_service import get_compaction

        # Seed transcript entries so FK constraint is satisfied
        ids = _insert_entries(db, 'test-topic', 10)
        watermark = ids[-1]

        db.execute(
            "INSERT INTO topic_compactions (topic, compacted_text, compacted_up_to_id, token_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ('test-topic', 'Summary of conversation', watermark, 50, '2026-03-20T10:00:00+00:00'),
        )
        db.commit()

        result = get_compaction('test-topic')
        assert result is not None
        assert result['compacted_text'] == 'Summary of conversation'
        assert result['compacted_up_to_id'] == watermark
        assert result['token_count'] == 50


class TestCheckAndCompact:
    def test_returns_false_for_empty_topic(self, db):
        from services.compaction_service import check_and_compact
        assert check_and_compact('', 32000) is False

    def test_returns_false_when_too_few_entries(self, db):
        from services.compaction_service import check_and_compact

        # Only 2 entries — below _MIN_ENTRIES_TO_COMPACT (4)
        with patch('services.compaction_service.get_entries_since', return_value=[
            {'id': 1, 'content': 'Hello'},
            {'id': 2, 'content': 'Hi there'},
        ]):
            assert check_and_compact('test', 32000) is False

    def test_returns_false_when_below_threshold(self, db):
        from services.compaction_service import check_and_compact

        entries = [{'id': i, 'content': f'Short msg {i}'} for i in range(5)]

        with patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.compaction_service.get_entries_since', return_value=entries):
            # 5 short entries, well below 85% of 32000
            assert check_and_compact('test', 32000) is False

    def test_absolute_threshold_fires_on_large_context_budget(self, db):
        """Large model context budget should still trigger compaction at absolute ceiling."""
        from services.compaction_service import check_and_compact

        # ~30K tokens of content — above _MAX_UNCOMPACTED_TOKENS (24K) but
        # well below 85% of 200K budget (170K)
        long_content = 'word ' * 6000  # ~7500 tokens each
        entries = [{'id': i, 'content': long_content} for i in range(5)]

        with patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.compaction_service.get_entries_since', return_value=entries), \
             patch('services.compaction_service._run_compaction', return_value=True) as mock_run:
            result = check_and_compact('test', 200_000)

        assert result is True
        mock_run.assert_called_once()

    def test_small_budget_uses_fraction_threshold(self, db):
        """Small context budget should use fraction-based threshold (below absolute ceiling)."""
        from services.compaction_service import check_and_compact

        # Budget=10000, fraction threshold=8500, absolute=24000 → min=8500
        # ~10K tokens of content — above 8500 fraction threshold
        long_content = 'word ' * 2000  # ~2500 tokens each
        entries = [{'id': i, 'content': long_content} for i in range(5)]

        with patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.compaction_service.get_entries_since', return_value=entries), \
             patch('services.compaction_service._run_compaction', return_value=True) as mock_run:
            result = check_and_compact('test', 10_000)

        assert result is True
        mock_run.assert_called_once()

class TestRunCompaction:
    def test_stores_compaction_result(self, db):
        from services.compaction_service import _run_compaction

        # Seed transcript entries so FK constraint is satisfied
        ids = _insert_entries(db, 'test-topic', 4, 'Weather msg {}')
        entries = [
            {'id': ids[0], 'role': 'user', 'content': 'What is the weather?'},
            {'id': ids[1], 'role': 'assistant', 'content': 'It is sunny in Malta today.'},
            {'id': ids[2], 'role': 'user', 'content': 'Thanks!'},
            {'id': ids[3], 'role': 'assistant', 'content': 'You are welcome.'},
        ]

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='- Weather in Malta: sunny')

        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            result = _run_compaction('test-topic', '', entries)

        assert result is True

        # Verify stored in database
        cursor = db.cursor()
        cursor.execute("SELECT compacted_text, compacted_up_to_id FROM topic_compactions WHERE topic = ?",
                        ('test-topic',))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == '- Weather in Malta: sunny'
        assert row[1] == ids[3]  # highest entry id

    def test_updates_existing_compaction(self, db):
        from services.compaction_service import _run_compaction

        # Seed transcript entries — old batch + new batch
        old_ids = _insert_entries(db, 'test-topic', 5, 'Old msg {}')
        new_ids = _insert_entries(db, 'test-topic', 2, 'New msg {}')

        # Insert initial compaction covering old entries
        db.execute(
            "INSERT INTO topic_compactions (topic, compacted_text, compacted_up_to_id, token_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ('test-topic', 'Old summary', old_ids[-1], 20, '2026-03-20T09:00:00'),
        )
        db.commit()

        entries = [
            {'id': new_ids[0], 'role': 'user', 'content': 'New information here'},
            {'id': new_ids[1], 'role': 'assistant', 'content': 'Noted and processed'},
        ]

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='Updated summary with new info')

        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            result = _run_compaction('test-topic', 'Old summary', entries)

        assert result is True

        cursor = db.cursor()
        cursor.execute("SELECT compacted_text, compacted_up_to_id FROM topic_compactions WHERE topic = ?",
                        ('test-topic',))
        row = cursor.fetchone()
        assert row[0] == 'Updated summary with new info'
        assert row[1] == new_ids[1]

    def test_includes_previous_text_in_prompt(self, db):
        from services.compaction_service import _run_compaction

        ids = _insert_entries(db, 'test', 1)
        entries = [{'id': ids[0], 'role': 'user', 'content': 'Hello'}]

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='Compacted')

        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            _run_compaction('test', 'Previous context here', entries)

        # Verify the user message includes the previous summary
        call_args = mock_llm.send_message.call_args
        user_message = call_args[0][1]
        assert '## Previous Summary' in user_message
        assert 'Previous context here' in user_message
        assert '## New Conversation Turns' in user_message

    def test_includes_tool_name_in_formatted_entries(self, db):
        from services.compaction_service import _run_compaction

        ids = _insert_entries(db, 'test', 1)
        entries = [
            {'id': ids[0], 'role': 'tool', 'content': 'Search results', 'tool_name': 'web_search'},
        ]

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='Compacted')

        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            _run_compaction('test', '', entries)

        user_message = mock_llm.send_message.call_args[0][1]
        assert '[tool — web_search]' in user_message

    def test_returns_false_on_empty_llm_response(self, db):
        from services.compaction_service import _run_compaction

        ids = _insert_entries(db, 'test', 1)
        entries = [{'id': ids[0], 'role': 'user', 'content': 'Hello'}]

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='   ')

        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            result = _run_compaction('test', '', entries)

        assert result is False

    def test_returns_false_on_llm_error(self, db):
        from services.compaction_service import _run_compaction

        ids = _insert_entries(db, 'test', 1)
        entries = [{'id': ids[0], 'role': 'user', 'content': 'Hello'}]

        with patch('services.llm_service.create_refreshable_llm_service', side_effect=Exception('LLM down')):
            result = _run_compaction('test', '', entries)

        assert result is False


class TestCompactionTopicContext:
    def test_get_compaction_accepts_topic_context(self, db):
        """get_compaction works when a TopicContext is passed."""
        from services.compaction_service import get_compaction
        from services.topic_context import TopicContext

        ctx = TopicContext(topic='test-topic')
        result = get_compaction('nonexistent', _context=ctx)
        assert result is None
        assert ctx.failed_sections == []

    def test_get_compaction_records_failure_to_context(self):
        """When DB fails, the failure is recorded on TopicContext."""
        from services.compaction_service import get_compaction
        from services.topic_context import TopicContext

        ctx = TopicContext(topic='test')
        with patch('services.database_service.get_shared_db_service', side_effect=Exception('db locked')):
            result = get_compaction('test', _context=ctx)

        assert result is None
        assert len(ctx.failed_sections) == 1
        assert ctx.failed_sections[0][0] == 'compaction_read'
        assert 'db locked' in ctx.failed_sections[0][1]

    def test_backward_compat_without_context(self, db):
        """get_compaction still works without _context (backward compat)."""
        from services.compaction_service import get_compaction
        result = get_compaction('nonexistent')
        assert result is None
