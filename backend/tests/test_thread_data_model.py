"""Feature tests for the thread data model (workstream B).

Drives the real production writers (Transcript.write_input_row /
write_assistant_row) against the real test DB — zero mocks — and asserts the
cross-step invariants that make the thread model work:

1. A thread-starter (turn_id=None) allocates a fresh turn_id.
2. A reply (turn_id supplied) appends rows carrying the EXISTING turn_id — no
   new allocation.
3. Transcript.recent_threads returns collapsed metadata ordered by recency.
4. Transcript.thread_rows returns the full row set of one thread.
5. The conversation API endpoints (/conversation/threads, /conversation/thread/<id>)
   project the real DB rows through the same pipeline as /conversation/recent.
"""
import sqlite3
from typing import cast

import pytest

from api.conversation import _rows_to_messages
from services.transcript_service import Transcript


def _seed_thread(question: str, *answers: str) -> int:
    """Open a new thread (turn_id=None → fresh allocation) and append its rows."""
    input_id = Transcript.write_input_row('user', 'user', question)
    turn_id = Transcript.turn_id_of_row(input_id)
    for ans in answers:
        Transcript.write_assistant_row('user', ans, turn_id=turn_id)
    return turn_id


@pytest.mark.unit
class TestThreadDataModel:

    def test_thread_starter_allocates_fresh_turn_id(self, db: sqlite3.Connection) -> None:
        """A thread-starter (no turn_id supplied) gets the next monotonic turn_id
        for the channel — the same allocation as before the thread redefinition."""
        t1 = _seed_thread('Hello', 'Hi there!')
        t2 = _seed_thread('Second thread', 'Reply')
        assert t2 == t1 + 1, "turn_id must be monotonically increasing"

    def test_reply_appends_to_existing_thread(self, db: sqlite3.Connection) -> None:
        """A reply (turn_id supplied) appends rows with the EXISTING turn_id — no
        new allocation. This is the core invariant: replies stay in their thread."""
        turn_id = _seed_thread('What is 2+2?', 'It is 4.')

        reply_input_id = Transcript.write_input_row('user', 'user', 'Thanks!', turn_id=turn_id)
        reply_turn_id = Transcript.turn_id_of_row(reply_input_id)
        assert reply_turn_id == turn_id, "reply must carry the existing thread's turn_id"

        Transcript.write_assistant_row('user', 'You are welcome.', turn_id=turn_id)

        rows = Transcript.by_turn('user', turn_id)
        contents = [r['content'] for r in rows]
        assert contents == ['What is 2+2?', 'It is 4.', 'Thanks!', 'You are welcome.']
        assert all(r['turn_id'] == turn_id for r in rows), "every row must share the thread turn_id"

    def test_reply_does_not_advance_turn_counter(self, db: sqlite3.Connection) -> None:
        """A reply does NOT consume a turn_id — the next thread-starter gets the
        same next value it would have without the reply."""
        t1 = _seed_thread('Thread 1', 'A1')
        reply_id = Transcript.write_input_row('user', 'user', 'Reply to thread 1', turn_id=t1)
        assert Transcript.turn_id_of_row(reply_id) == t1

        t2 = _seed_thread('Thread 2', 'A2')
        assert t2 == t1 + 1, "reply must not consume a turn slot"

    def test_recent_threads_returns_collapsed_metadata(self, db: sqlite3.Connection) -> None:
        """recent_threads returns one summary row per thread, ordered by last
        activity (MAX(id)). Each thread carries a preview, row count and turn_id."""
        t1 = _seed_thread('First question', 'First answer')
        t2 = _seed_thread('Second question', 'Second answer', 'Second follow-up')

        threads, has_more, returned = Transcript.recent_threads('user', limit=50)
        assert returned == 2
        assert has_more is False
        assert threads[0]['turn_id'] == t2, "newest thread first"
        assert threads[1]['turn_id'] == t1
        assert threads[0]['row_count'] == 3, "t2 has 1 input + 2 assistant = 3 rows"
        assert threads[1]['row_count'] == 2, "t1 has 1 input + 1 assistant = 2 rows"
        assert 'Second question' in cast("str", threads[0]['preview'])

    def test_thread_rows_returns_full_thread(self, db: sqlite3.Connection) -> None:
        """thread_rows returns every row of one thread, oldest-first — the
        expand-on-click payload."""
        t1 = _seed_thread('Q1', 'A1a', 'A1b')
        _seed_thread('Q2', 'A2')  # different thread, must not appear

        rows = Transcript.thread_rows('user', t1)
        assert len(rows) == 3
        assert [r['content'] for r in rows] == ['Q1', 'A1a', 'A1b']
        assert all(r['turn_id'] == t1 for r in rows)

    def test_thread_rows_excludes_compaction_via_api(self, db: sqlite3.Connection) -> None:
        """The conversation API endpoint excludes compaction rows — the feed view
        stays clean. thread_rows itself takes an exclude_roles tuple."""
        t1 = _seed_thread('Q1', 'A1')
        Transcript.write_input_row('user', 'compaction', 'compacted text', turn_id=t1)

        rows = Transcript.thread_rows('user', t1, exclude_roles=('compaction',))
        assert all(r['role'] != 'compaction' for r in rows), "compaction excluded when requested"
        full = Transcript.by_turn('user', t1)
        assert any(r['role'] == 'compaction' for r in full), "by_turn includes compaction"

    def test_rows_to_messages_projects_thread_rows(self, db: sqlite3.Connection) -> None:
        """_rows_to_messages (the shared projection) correctly maps thread_rows
        into the conversation message shape — the same path the API endpoints use."""
        t1 = _seed_thread('Hello', 'Hi there!')
        rows = Transcript.thread_rows('user', t1)
        messages = _rows_to_messages(rows)
        assert len(messages) == 2
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == 'Hello'
        assert messages[0]['turn_id'] == t1
        assert messages[1]['role'] == 'assistant'
        assert messages[1]['turn_id'] == t1

    def test_recent_threads_pagination(self, db: sqlite3.Connection) -> None:
        """recent_threads paginates by thread, same as recent_turns."""
        for i in range(5):
            _seed_thread(f'Q{i}', f'A{i}')

        page1, has_more, returned = Transcript.recent_threads('user', limit=2, offset=0)
        assert returned == 2
        assert has_more is True

        page2, has_more, returned = Transcript.recent_threads('user', limit=2, offset=2)
        assert returned == 2
        assert has_more is True

        page3, has_more, returned = Transcript.recent_threads('user', limit=2, offset=4)
        assert returned == 1
        assert has_more is False

        all_turn_ids = {t['turn_id'] for t in page1 + page2 + page3}
        assert len(all_turn_ids) == 5, "no thread appears twice across pages"