"""Tests for Phase 4D: Compaction Service.

Tests cover:
- check_and_compact() trigger logic
- get_compaction() retrieval
- get_entries_since() delegation
- _run_compaction() LLM call and storage
"""

import pytest
import sqlite3
from unittest.mock import patch, MagicMock


@pytest.fixture
def compaction_db():
    """Create an in-memory SQLite database with transcript + compaction schemas."""
    conn = sqlite3.connect(':memory:')
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE topic_transcript (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_call_id TEXT,
            tool_name TEXT,
            internal INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE topic_compactions (
            topic TEXT PRIMARY KEY,
            compacted_text TEXT NOT NULL,
            compacted_up_to_id INTEGER NOT NULL,
            token_count INTEGER DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (compacted_up_to_id) REFERENCES topic_transcript(id)
        )
    """)
    conn.commit()

    mock_db = MagicMock()
    mock_db.connection.return_value.__enter__ = lambda s: conn
    mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)

    with patch('services.database_service.get_shared_db_service', return_value=mock_db):
        yield conn

    conn.close()


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
    def test_returns_none_when_no_compaction(self, compaction_db):
        from services.compaction_service import get_compaction
        assert get_compaction('nonexistent-topic') is None

    def test_returns_stored_compaction(self, compaction_db):
        from services.compaction_service import get_compaction

        compaction_db.execute(
            "INSERT INTO topic_compactions (topic, compacted_text, compacted_up_to_id, token_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ('test-topic', 'Summary of conversation', 10, 50, '2026-03-20T10:00:00+00:00'),
        )
        compaction_db.commit()

        result = get_compaction('test-topic')
        assert result is not None
        assert result['compacted_text'] == 'Summary of conversation'
        assert result['compacted_up_to_id'] == 10
        assert result['token_count'] == 50


class TestGetEntriesSince:
    def test_delegates_to_transcript_service(self, compaction_db):
        from services.compaction_service import get_entries_since

        _insert_entries(compaction_db, 'test', 5)

        with patch('services.transcript_service.get_recent') as mock_recent:
            mock_recent.return_value = [{'id': 3, 'content': 'msg'}]
            result = get_entries_since('test', watermark=2)

        mock_recent.assert_called_once_with('test', limit=500, since_id=2)
        assert len(result) == 1


class TestCheckAndCompact:
    def test_returns_false_for_empty_topic(self, compaction_db):
        from services.compaction_service import check_and_compact
        assert check_and_compact('', 32000) is False

    def test_returns_false_when_too_few_entries(self, compaction_db):
        from services.compaction_service import check_and_compact

        # Only 2 entries — below _MIN_ENTRIES_TO_COMPACT (4)
        with patch('services.compaction_service.get_entries_since', return_value=[
            {'id': 1, 'content': 'Hello'},
            {'id': 2, 'content': 'Hi there'},
        ]):
            assert check_and_compact('test', 32000) is False

    def test_returns_false_when_below_threshold(self, compaction_db):
        from services.compaction_service import check_and_compact

        entries = [{'id': i, 'content': f'Short msg {i}'} for i in range(5)]

        with patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.compaction_service.get_entries_since', return_value=entries):
            # 5 short entries, well below 85% of 32000
            assert check_and_compact('test', 32000) is False

    def test_fires_compaction_when_above_threshold(self, compaction_db):
        from services.compaction_service import check_and_compact

        # Create entries with enough content to exceed threshold
        long_content = 'word ' * 5000  # ~6500 tokens
        entries = [{'id': i, 'content': long_content} for i in range(5)]

        with patch('services.compaction_service.get_compaction', return_value=None), \
             patch('services.compaction_service.get_entries_since', return_value=entries), \
             patch('services.compaction_service._run_compaction', return_value=True) as mock_run:
            # Budget = 1000 tokens, 5 entries * ~6500 tokens each >> threshold
            result = check_and_compact('test', 1000)

        assert result is True
        mock_run.assert_called_once()

    def test_absolute_threshold_fires_on_large_context_budget(self, compaction_db):
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

    def test_small_budget_uses_fraction_threshold(self, compaction_db):
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

    def test_includes_existing_compaction_tokens(self, compaction_db):
        from services.compaction_service import check_and_compact

        existing = {
            'compacted_text': 'Previous summary',
            'compacted_up_to_id': 5,
            'token_count': 500,
            'updated_at': '2026-03-20T10:00:00',
        }
        entries = [{'id': i, 'content': 'word ' * 100} for i in range(6, 12)]

        with patch('services.compaction_service.get_compaction', return_value=existing), \
             patch('services.compaction_service.get_entries_since', return_value=entries), \
             patch('services.compaction_service._run_compaction', return_value=True) as mock_run:
            # Budget = 1000, existing 500 + new entries should exceed 850
            result = check_and_compact('test', 1000)

        assert result is True
        mock_run.assert_called_once_with('test', 'Previous summary', entries)


class TestRunCompaction:
    def test_stores_compaction_result(self, compaction_db):
        from services.compaction_service import _run_compaction

        entries = [
            {'id': 10, 'role': 'user', 'content': 'What is the weather?'},
            {'id': 11, 'role': 'assistant', 'content': 'It is sunny in Malta today.'},
            {'id': 12, 'role': 'user', 'content': 'Thanks!'},
            {'id': 13, 'role': 'assistant', 'content': 'You are welcome.'},
        ]

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='- Weather in Malta: sunny')

        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            result = _run_compaction('test-topic', '', entries)

        assert result is True

        # Verify stored in database
        cursor = compaction_db.cursor()
        cursor.execute("SELECT compacted_text, compacted_up_to_id FROM topic_compactions WHERE topic = ?",
                        ('test-topic',))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == '- Weather in Malta: sunny'
        assert row[1] == 13  # highest entry id

    def test_updates_existing_compaction(self, compaction_db):
        from services.compaction_service import _run_compaction

        # Insert initial compaction
        compaction_db.execute(
            "INSERT INTO topic_compactions (topic, compacted_text, compacted_up_to_id, token_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ('test-topic', 'Old summary', 5, 20, '2026-03-20T09:00:00'),
        )
        compaction_db.commit()

        entries = [
            {'id': 14, 'role': 'user', 'content': 'New information here'},
            {'id': 15, 'role': 'assistant', 'content': 'Noted and processed'},
        ]

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='Updated summary with new info')

        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            result = _run_compaction('test-topic', 'Old summary', entries)

        assert result is True

        cursor = compaction_db.cursor()
        cursor.execute("SELECT compacted_text, compacted_up_to_id FROM topic_compactions WHERE topic = ?",
                        ('test-topic',))
        row = cursor.fetchone()
        assert row[0] == 'Updated summary with new info'
        assert row[1] == 15

    def test_includes_previous_text_in_prompt(self, compaction_db):
        from services.compaction_service import _run_compaction

        entries = [{'id': 1, 'role': 'user', 'content': 'Hello'}]

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

    def test_includes_tool_name_in_formatted_entries(self, compaction_db):
        from services.compaction_service import _run_compaction

        entries = [
            {'id': 1, 'role': 'tool', 'content': 'Search results', 'tool_name': 'web_search'},
        ]

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='Compacted')

        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            _run_compaction('test', '', entries)

        user_message = mock_llm.send_message.call_args[0][1]
        assert '[tool — web_search]' in user_message

    def test_returns_false_on_empty_llm_response(self, compaction_db):
        from services.compaction_service import _run_compaction

        entries = [{'id': 1, 'role': 'user', 'content': 'Hello'}]

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='   ')

        with patch('services.llm_service.create_refreshable_llm_service', return_value=mock_llm):
            result = _run_compaction('test', '', entries)

        assert result is False

    def test_returns_false_on_llm_error(self, compaction_db):
        from services.compaction_service import _run_compaction

        entries = [{'id': 1, 'role': 'user', 'content': 'Hello'}]

        with patch('services.llm_service.create_refreshable_llm_service', side_effect=Exception('LLM down')):
            result = _run_compaction('test', '', entries)

        assert result is False
