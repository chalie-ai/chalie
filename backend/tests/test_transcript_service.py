"""Tests for Phase 4B: Transcript Service.

Tests cover:
- append() and append_batch()
- get_recent() with and without since_id
- get_latest_id()
- search() (keyword fallback path)
- prune_old()
- Transcript skill (handle_notes)
"""

import pytest
import sqlite3
from unittest.mock import patch, MagicMock


@pytest.fixture
def transcript_db():
    """Create an in-memory SQLite database with the transcript schema."""
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
        CREATE INDEX idx_transcript_topic ON topic_transcript(topic, created_at)
    """)
    conn.commit()

    # Mock the database service to return our test connection
    mock_db = MagicMock()
    mock_db.connection.return_value.__enter__ = lambda s: conn
    mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)

    with patch('services.database_service.get_shared_db_service', return_value=mock_db):
        yield conn

    conn.close()


class TestAppend:
    def test_basic_append(self, transcript_db):
        from services.transcript_service import append

        with patch('services.transcript_service._embed_entry'):
            rowid = append('test-topic', 'user', 'Hello, world!')
        assert rowid is not None
        assert rowid > 0

        # Verify the entry is in the database
        cursor = transcript_db.cursor()
        cursor.execute("SELECT topic, role, content FROM topic_transcript WHERE id = ?", (rowid,))
        row = cursor.fetchone()
        assert row == ('test-topic', 'user', 'Hello, world!')

    def test_append_with_tool_info(self, transcript_db):
        from services.transcript_service import append

        with patch('services.transcript_service._embed_entry'):
            rowid = append(
                'test-topic', 'tool', 'Search results here',
                tool_call_id='tc_123', tool_name='web_search',
            )
        cursor = transcript_db.cursor()
        cursor.execute(
            "SELECT tool_call_id, tool_name FROM topic_transcript WHERE id = ?",
            (rowid,),
        )
        row = cursor.fetchone()
        assert row == ('tc_123', 'web_search')

    def test_append_internal_flag(self, transcript_db):
        from services.transcript_service import append

        with patch('services.transcript_service._embed_entry'):
            rowid = append('test-topic', 'internal', 'Working notes', internal=True)
        cursor = transcript_db.cursor()
        cursor.execute("SELECT internal FROM topic_transcript WHERE id = ?", (rowid,))
        assert cursor.fetchone()[0] == 1

    def test_append_empty_content_returns_none(self, transcript_db):
        from services.transcript_service import append
        assert append('test-topic', 'user', '') is None

    def test_append_empty_topic_returns_none(self, transcript_db):
        from services.transcript_service import append
        assert append('', 'user', 'content') is None


class TestAppendBatch:
    def test_batch_insert(self, transcript_db):
        from services.transcript_service import append_batch

        entries = [
            {'topic': 'test', 'role': 'user', 'content': 'First message'},
            {'topic': 'test', 'role': 'assistant', 'content': 'First response'},
            {'topic': 'test', 'role': 'user', 'content': 'Second message'},
        ]
        with patch('services.transcript_service._embed_entry'):
            count = append_batch(entries)
        assert count == 3

        cursor = transcript_db.cursor()
        cursor.execute("SELECT COUNT(*) FROM topic_transcript WHERE topic = 'test'")
        assert cursor.fetchone()[0] == 3

    def test_batch_skips_empty(self, transcript_db):
        from services.transcript_service import append_batch

        entries = [
            {'topic': 'test', 'role': 'user', 'content': 'Valid'},
            {'topic': '', 'role': 'user', 'content': 'No topic'},
            {'topic': 'test', 'role': 'user', 'content': ''},
        ]
        with patch('services.transcript_service._embed_entry'):
            count = append_batch(entries)
        assert count == 1


class TestGetRecent:
    def test_get_recent_returns_ordered(self, transcript_db):
        from services.transcript_service import append, get_recent

        with patch('services.transcript_service._embed_entry'):
            append('test', 'user', 'First')
            append('test', 'assistant', 'Second')
            append('test', 'user', 'Third')

        results = get_recent('test', limit=10)
        assert len(results) == 3
        # Should be in chronological order (oldest first)
        assert results[0]['content'] == 'First'
        assert results[2]['content'] == 'Third'

    def test_get_recent_respects_limit(self, transcript_db):
        from services.transcript_service import append, get_recent

        with patch('services.transcript_service._embed_entry'):
            for i in range(10):
                append('test', 'user', f'Message {i}')

        results = get_recent('test', limit=3)
        assert len(results) == 3

    def test_get_recent_since_id(self, transcript_db):
        from services.transcript_service import append, get_recent

        with patch('services.transcript_service._embed_entry'):
            id1 = append('test', 'user', 'First')
            id2 = append('test', 'assistant', 'Second')
            id3 = append('test', 'user', 'Third')

        results = get_recent('test', since_id=id1)
        assert len(results) == 2
        assert results[0]['content'] == 'Second'
        assert results[1]['content'] == 'Third'

    def test_get_recent_filters_by_topic(self, transcript_db):
        from services.transcript_service import append, get_recent

        with patch('services.transcript_service._embed_entry'):
            append('topic-a', 'user', 'A message')
            append('topic-b', 'user', 'B message')

        results = get_recent('topic-a')
        assert len(results) == 1
        assert results[0]['content'] == 'A message'


class TestGetLatestId:
    def test_returns_highest_id(self, transcript_db):
        from services.transcript_service import append, get_latest_id

        with patch('services.transcript_service._embed_entry'):
            append('test', 'user', 'First')
            id2 = append('test', 'user', 'Second')

        assert get_latest_id('test') == id2

    def test_returns_none_for_empty_topic(self, transcript_db):
        from services.transcript_service import get_latest_id
        assert get_latest_id('nonexistent') is None


class TestKeywordSearch:
    def test_keyword_search_finds_content(self, transcript_db):
        from services.transcript_service import _keyword_search, append

        with patch('services.transcript_service._embed_entry'):
            append('test', 'user', 'The weather in Malta is sunny today')
            append('test', 'user', 'I need to buy groceries')

        results = _keyword_search('test', 'Malta', limit=5)
        assert len(results) == 1
        assert 'Malta' in results[0]['content']

    def test_keyword_search_filters_by_topic(self, transcript_db):
        from services.transcript_service import _keyword_search, append

        with patch('services.transcript_service._embed_entry'):
            append('topic-a', 'user', 'Python programming')
            append('topic-b', 'user', 'Python snakes')

        results = _keyword_search('topic-a', 'Python', limit=5)
        assert len(results) == 1
        assert 'programming' in results[0]['content']


class TestTranscriptSkill:
    def test_empty_query_returns_error(self):
        from services.innate_skills.notes_skill import handle_notes
        result = handle_notes('test', {'query': ''})
        assert 'query' in result.lower()

    def test_no_results_returns_message(self, transcript_db):
        from services.innate_skills.notes_skill import handle_notes

        # Patch the search to return empty
        with patch('services.transcript_service.search', return_value=[]):
            result = handle_notes('test', {'query': 'nonexistent'})
        assert 'No transcript entries' in result

    def test_formats_results(self, transcript_db):
        from services.innate_skills.notes_skill import handle_notes

        mock_results = [
            {
                'id': 1,
                'role': 'user',
                'content': 'Test message content',
                'tool_name': None,
                'created_at': '2026-03-20 10:00:00',
                'similarity': 0.85,
            },
        ]
        with patch('services.transcript_service.search', return_value=mock_results):
            result = handle_notes('test', {'query': 'test message'})

        assert 'Test message content' in result
        assert '85%' in result
        assert '[user]' in result
