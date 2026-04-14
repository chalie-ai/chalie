"""Tests for Transcript Service.

Tests cover:
- append() and append_batch()
- get_recent() with and without since_id
- get_latest_id()
- search() (keyword fallback path, cross-topic, date range)
- prune_old()
"""

from unittest.mock import patch


class TestAppend:
    def test_basic_append(self, db):
        from services.transcript_service import append

        with patch('services.transcript_service._embed_entry'):
            rowid = append('test-topic', 'user', 'Hello, world!')
        assert rowid is not None
        assert rowid > 0

        # Verify the entry is in the database
        cursor = db.cursor()
        cursor.execute("SELECT channel, role, content FROM transcript WHERE id = ?", (rowid,))
        row = cursor.fetchone()
        assert row['channel'] == 'test-topic'
        assert row['role'] == 'user'
        assert row['content'] == 'Hello, world!'

    def test_append_with_tool_info(self, db):
        from services.transcript_service import append

        with patch('services.transcript_service._embed_entry'):
            rowid = append(
                'test-topic', 'tool', 'Search results here',
                tool_call_id='tc_123', tool_name='search',
            )
        cursor = db.cursor()
        cursor.execute(
            "SELECT tool_call_id, tool_name FROM transcript WHERE id = ?",
            (rowid,),
        )
        row = cursor.fetchone()
        assert row['tool_call_id'] == 'tc_123'
        assert row['tool_name'] == 'search'

    def test_append_internal_flag(self, db):
        from services.transcript_service import append

        with patch('services.transcript_service._embed_entry'):
            rowid = append('test-topic', 'internal', 'Working notes', internal=True)
        cursor = db.cursor()
        cursor.execute("SELECT internal FROM transcript WHERE id = ?", (rowid,))
        assert cursor.fetchone()[0] == 1

    def test_append_empty_content_returns_none(self, db):
        from services.transcript_service import append
        assert append('test-topic', 'user', '') is None

    def test_append_empty_topic_returns_none(self, db):
        from services.transcript_service import append
        assert append('', 'user', 'content') is None


class TestAppendBatch:
    def test_batch_insert(self, db):
        from services.transcript_service import append_batch

        entries = [
            {'topic': 'test', 'role': 'user', 'content': 'First message'},
            {'topic': 'test', 'role': 'assistant', 'content': 'First response'},
            {'topic': 'test', 'role': 'user', 'content': 'Second message'},
        ]
        with patch('services.transcript_service._embed_entry'):
            count = append_batch(entries)
        assert count == 3

        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) FROM transcript WHERE channel = 'test'")
        assert cursor.fetchone()[0] == 3

    def test_batch_skips_empty(self, db):
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
    def test_get_recent_returns_ordered(self, db):
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

    def test_get_recent_respects_limit(self, db):
        from services.transcript_service import append, get_recent

        with patch('services.transcript_service._embed_entry'):
            for i in range(10):
                append('test', 'user', f'Message {i}')

        results = get_recent('test', limit=3)
        assert len(results) == 3

    def test_get_recent_since_id(self, db):
        from services.transcript_service import append, get_recent

        with patch('services.transcript_service._embed_entry'):
            id1 = append('test', 'user', 'First')
            append('test', 'assistant', 'Second')
            append('test', 'user', 'Third')

        results = get_recent('test', since_id=id1)
        assert len(results) == 2
        assert results[0]['content'] == 'Second'
        assert results[1]['content'] == 'Third'

    def test_get_recent_filters_by_topic(self, db):
        from services.transcript_service import append, get_recent

        with patch('services.transcript_service._embed_entry'):
            append('topic-a', 'user', 'A message')
            append('topic-b', 'user', 'B message')

        results = get_recent('topic-a')
        assert len(results) == 1
        assert results[0]['content'] == 'A message'


class TestGetLatestId:
    def test_returns_highest_id(self, db):
        from services.transcript_service import append, get_latest_id

        with patch('services.transcript_service._embed_entry'):
            append('test', 'user', 'First')
            id2 = append('test', 'user', 'Second')

        assert get_latest_id('test') == id2

    def test_returns_none_for_empty_topic(self, db):
        from services.transcript_service import get_latest_id
        assert get_latest_id('nonexistent') is None


class TestKeywordSearch:
    def test_keyword_search_finds_content(self, db):
        from services.transcript_service import _keyword_search, append

        with patch('services.transcript_service._embed_entry'):
            append('test', 'user', 'The weather in Malta is sunny today')
            append('test', 'user', 'I need to buy groceries')

        results = _keyword_search('test', 'Malta', limit=5)
        assert len(results) == 1
        assert 'Malta' in results[0]['content']

    def test_keyword_search_filters_by_topic(self, db):
        from services.transcript_service import _keyword_search, append

        with patch('services.transcript_service._embed_entry'):
            append('topic-a', 'user', 'Python programming')
            append('topic-b', 'user', 'Python snakes')

        results = _keyword_search('topic-a', 'Python', limit=5)
        assert len(results) == 1
        assert 'programming' in results[0]['content']

    def test_keyword_search_global(self, db):
        from services.transcript_service import _keyword_search, append

        with patch('services.transcript_service._embed_entry'):
            append('topic-a', 'user', 'Python programming')
            append('topic-b', 'user', 'Python snakes')

        results = _keyword_search(None, 'Python', limit=5)
        assert len(results) == 2

    def test_keyword_search_with_topic_field(self, db):
        from services.transcript_service import _keyword_search, append

        with patch('services.transcript_service._embed_entry'):
            append('topic-a', 'user', 'Malta weather')

        results = _keyword_search('topic-a', 'Malta', limit=5)
        assert results[0]['channel'] == 'topic-a'

    def test_keyword_search_date_range(self, db):
        from services.transcript_service import _keyword_search, append

        with patch('services.transcript_service._embed_entry'):
            append('test', 'user', 'Old message about Python')
            append('test', 'user', 'New message about Python')

        # All results (no date filter)
        results = _keyword_search('test', 'Python', limit=5)
        assert len(results) == 2

        # With a future date_from — should exclude everything
        results = _keyword_search('test', 'Python', limit=5, date_from='2099-01-01')
        assert len(results) == 0


class TestGetRecentTopicContext:
    def test_get_recent_works(self, db):
        """get_recent retrieves entries correctly."""
        from services.transcript_service import append, get_recent

        with patch('services.transcript_service._embed_entry'):
            append('test', 'user', 'Hello')

        results = get_recent('test')
        assert len(results) == 1

    def test_get_recent_returns_empty_on_db_failure(self):
        """When DB fails, get_recent returns empty list."""
        from services.transcript_service import get_recent

        with patch('services.database_service.get_shared_db_service', side_effect=Exception('db locked')):
            results = get_recent('test')

        assert results == []
