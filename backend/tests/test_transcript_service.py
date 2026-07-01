"""Tests for Transcript Service.

Covers ``Transcript.get_recent()`` — chronological ordering, ``limit``,
``since_id``, and the per-channel topic filter — against the real production
stack on the shared ``db`` fixture (in-memory SQLite built from schema.sql).
"""

import sqlite3

import pytest

pytestmark = pytest.mark.unit


class TestGetRecent:
    def test_get_recent_returns_ordered(self, db: sqlite3.Connection) -> None:
        from services.transcript_service import Transcript

        Transcript.write_input_row('test', 'user', 'First')
        Transcript.write_assistant_row('test', 'Second')
        Transcript.write_input_row('test', 'user', 'Third')

        results = Transcript.get_recent('test', limit=10)
        assert len(results) == 3
        # Should be in chronological order (oldest first)
        assert results[0]['content'] == 'First'
        assert results[2]['content'] == 'Third'

    def test_get_recent_respects_limit(self, db: sqlite3.Connection) -> None:
        from services.transcript_service import Transcript

        for i in range(10):
            Transcript.write_input_row('test', 'user', f'Msg{i}')

        results = Transcript.get_recent('test', limit=3)
        assert len(results) == 3

    def test_get_recent_since_id(self, db: sqlite3.Connection) -> None:
        from services.transcript_service import Transcript

        id1 = Transcript.write_input_row('test', 'user', 'First')
        Transcript.write_assistant_row('test', 'Second')
        Transcript.write_input_row('test', 'user', 'Third')

        results = Transcript.get_recent('test', since_id=id1)
        assert len(results) == 2
        assert results[0]['content'] == 'Second'
        assert results[1]['content'] == 'Third'

    def test_get_recent_filters_by_topic(self, db: sqlite3.Connection) -> None:
        from services.transcript_service import Transcript

        Transcript.write_input_row('topic-a', 'user', 'A message')
        Transcript.write_input_row('topic-b', 'user', 'B message')

        results = Transcript.get_recent('topic-a')
        assert len(results) == 1
        assert results[0]['content'] == 'A message'
