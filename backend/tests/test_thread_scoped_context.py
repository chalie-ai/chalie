"""Feature tests for thread-scoped context + compaction (workstream C).

Drives the real production paths — ``_previous_rows``, ``get_compaction``,
``_wrap_with_checkpoint``, and the compactor's thread-scoped write — against
the real test DB with zero mocks. Pins the invariants that make thread-scoped
context work:

1. ``_previous_rows`` returns ONLY the active thread's rows — no cross-thread
   leakage (the core context-scoping invariant).
2. ``get_compaction(channel, turn_id)`` is thread-scoped: a compaction in
   thread A is invisible to thread B.
3. ``_wrap_with_checkpoint`` wraps with the THREAD's compaction, not another
   thread's.
4. The compactor writes its checkpoint row INTO the thread (turn_id preserved)
   so the thread-scoped watermark re-keys from ``(channel, turn_id)``.
5. The compactor's prior-checkpoint carry-forward is thread-scoped.
"""
import sqlite3
from typing import cast

import pytest

from services import compaction_persistence
from services.transcript_service import Transcript


def _seed_thread(question: str, *answers: str) -> int:
    """Open a new thread (turn_id=None → fresh allocation) and append its rows."""
    input_id = Transcript.write_input_row('user', 'user', question)
    turn_id = Transcript.turn_id_of_row(input_id)
    for ans in answers:
        Transcript.write_assistant_row('user', ans, turn_id=turn_id)
    return turn_id


def _seed_compaction(channel: str, turn_id: int, summary: str) -> int:
    """Write a compaction row INTO a thread — the production compactor's path."""
    return Transcript.write_input_row(channel, 'compaction', summary, turn_id=turn_id)


@pytest.mark.unit
class TestThreadScopedContext:

    def test_previous_rows_returns_only_active_thread(self, db: sqlite3.Connection) -> None:
        """_previous_rows scoped to turn_id returns ONLY that thread's rows.
        Thread B's rows are invisible to thread A — the core context invariant."""
        from services.message_processor import MessageProcessor
        from tests.helpers import make_stub_config

        t1 = _seed_thread('Hello from thread 1', 'Reply 1')
        _seed_thread('Hello from thread 2', 'Reply 2', 'Reply 2b')

        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, 'next message', None)
        mp.config = make_stub_config(channel='user')
        mp.turn_id = t1

        rows = mp._previous_rows()
        contents = [cast("str", r['content']) for r in rows]
        assert 'Hello from thread 1' in contents
        assert 'Reply 1' in contents
        assert 'Hello from thread 2' not in contents, "thread 2 must not leak into thread 1's context"
        assert 'Reply 2b' not in contents

    def test_previous_rows_excludes_compaction_row_via_watermark(self, db: sqlite3.Connection) -> None:
        """A compaction row in the thread acts as the watermark: rows at/below it
        are excluded from _previous_rows, rows after it survive."""
        from services.message_processor import MessageProcessor
        from tests.helpers import make_stub_config

        t1 = _seed_thread('old q', 'old a')
        _seed_compaction('user', t1, 'checkpoint summary')
        Transcript.write_input_row('user', 'user', 'post-compaction reply', turn_id=t1)
        Transcript.write_assistant_row('user', 'post-compaction answer', turn_id=t1)

        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, 'next', None)
        mp.config = make_stub_config(channel='user')
        mp.turn_id = t1

        rows = mp._previous_rows()
        contents = [cast("str", r['content']) for r in rows]
        assert 'post-compaction reply' in contents
        assert 'post-compaction answer' in contents
        assert 'checkpoint summary' not in contents, "compaction row itself excluded"
        assert 'old q' not in contents, "pre-watermark rows excluded"
        assert 'old a' not in contents

    def test_previous_rows_none_turn_id_is_channel_wide(self, db: sqlite3.Connection) -> None:
        """A None turn_id (housekeeping channel) keeps the legacy channel-wide read
        — the fallback path for channels that never allocate a thread."""
        from services.message_processor import MessageProcessor
        from tests.helpers import make_stub_config

        _seed_thread('thread A', 'A reply')
        _seed_thread('thread B', 'B reply')

        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, 'housekeeping', None)
        mp.config = make_stub_config(channel='user')
        mp.turn_id = None

        rows = mp._previous_rows()
        contents = [cast("str", r['content']) for r in rows]
        assert 'thread A' in contents
        assert 'thread B' in contents, "None turn_id reads channel-wide (legacy fallback)"


@pytest.mark.unit
class TestThreadScopedCompaction:

    def test_get_compaction_thread_scoped(self, db: sqlite3.Connection) -> None:
        """get_compaction(channel, turn_id) returns the compaction for THAT thread
        only — a compaction in thread A is invisible to thread B."""
        t1 = _seed_thread('thread 1 q', 'thread 1 a')
        t2 = _seed_thread('thread 2 q', 'thread 2 a')
        _seed_compaction('user', t1, 'thread 1 checkpoint')

        row1 = compaction_persistence.get_compaction('user', t1)
        row2 = compaction_persistence.get_compaction('user', t2)
        assert row1 is not None
        assert cast("str", row1['compacted_text']) == 'thread 1 checkpoint'
        assert row2 is None, "thread 2 has no compaction — thread 1's must not leak"

    def test_get_compaction_none_turn_id_is_channel_wide(self, db: sqlite3.Connection) -> None:
        """get_compaction(channel, None) returns the channel-wide latest compaction
        (legacy/housekeeping path)."""
        t1 = _seed_thread('thread 1 q', 'thread 1 a')
        _seed_compaction('user', t1, 'channel checkpoint')
        Transcript.write_input_row('user', 'compaction', 'legacy checkpoint')

        row = compaction_persistence.get_compaction('user', None)
        assert row is not None
        assert cast("str", row['compacted_text']) == 'legacy checkpoint'

    def test_get_compaction_returns_none_for_unknown_thread(self, db: sqlite3.Connection) -> None:
        """A turn_id with no compaction row returns None — no false watermark."""
        t1 = _seed_thread('thread 1 q', 'thread 1 a')
        _seed_compaction('user', t1, 'thread 1 checkpoint')

        assert compaction_persistence.get_compaction('user', 999) is None

    def test_get_compaction_returns_latest_by_id_in_thread(self, db: sqlite3.Connection) -> None:
        """When a thread has multiple compaction rows, the latest (highest id) is
        the watermark — the same MAX(id) semantics as the legacy channel-wide read."""
        t1 = _seed_thread('thread 1 q', 'thread 1 a')
        _seed_compaction('user', t1, 'old checkpoint')
        newer = _seed_compaction('user', t1, 'new checkpoint')

        row = compaction_persistence.get_compaction('user', t1)
        assert row is not None
        assert cast("int", row['compacted_up_to_id']) == newer
        assert cast("str", row['compacted_text']) == 'new checkpoint'


@pytest.mark.unit
class TestThreadScopedCheckpoint:

    def test_wrap_with_checkpoint_uses_thread_compaction(self, db: sqlite3.Connection) -> None:
        """_wrap_with_checkpoint(channel, body, turn_id) wraps with the THREAD's
        compaction, not another thread's."""
        from services.message_processor import _wrap_with_checkpoint

        t1 = _seed_thread('thread 1 q', 'thread 1 a')
        t2 = _seed_thread('thread 2 q', 'thread 2 a')
        _seed_compaction('user', t1, 'thread 1 checkpoint text')
        _seed_compaction('user', t2, 'thread 2 checkpoint text')

        result1 = _wrap_with_checkpoint('user', 'current body', t1)
        result2 = _wrap_with_checkpoint('user', 'current body', t2)
        assert 'thread 1 checkpoint text' in result1
        assert 'thread 2 checkpoint text' not in result1, "thread 2's checkpoint leaked into thread 1"
        assert 'thread 2 checkpoint text' in result2
        assert 'thread 1 checkpoint text' not in result2

    def test_wrap_with_checkpoint_no_compaction_returns_body(self, db: sqlite3.Connection) -> None:
        """When the thread has no compaction, the body is returned unwrapped."""
        from services.message_processor import _wrap_with_checkpoint

        t1 = _seed_thread('thread 1 q', 'thread 1 a')
        result = _wrap_with_checkpoint('user', 'plain body', t1)
        assert result == 'plain body'

    def test_wrap_with_checkpoint_none_turn_id_is_channel_wide(self, db: sqlite3.Connection) -> None:
        """_wrap_with_checkpoint(channel, body, None) uses the channel-wide
        compaction (legacy fallback)."""
        from services.message_processor import _wrap_with_checkpoint

        Transcript.write_input_row('user', 'compaction', 'channel-wide checkpoint')
        result = _wrap_with_checkpoint('user', 'current body', None)
        assert 'channel-wide checkpoint' in result


@pytest.mark.unit
class TestCompactorWritesIntoThread:

    def test_compaction_row_carries_thread_turn_id(self, db: sqlite3.Connection) -> None:
        """A compaction row written with turn_id lands IN the thread — the
        thread-scoped get_compaction re-keys from (channel, turn_id)."""
        t1 = _seed_thread('thread 1 q', 'thread 1 a')
        cid = _seed_compaction('user', t1, 'thread 1 checkpoint')

        row = Transcript.by_turn('user', t1)
        compaction_rows = [r for r in row if r['role'] == 'compaction']
        assert len(compaction_rows) == 1
        assert cast("int", compaction_rows[0]['id']) == cid
        assert cast("int", compaction_rows[0]['turn_id']) == t1, "compaction row must carry the thread's turn_id"

    def test_compaction_in_one_thread_does_not_affect_another(self, db: sqlite3.Connection) -> None:
        """Compacting thread 1 does not create a watermark for thread 2 — each
        thread carries its own checkpoint independently."""
        t1 = _seed_thread('thread 1 q', 'thread 1 a')
        t2 = _seed_thread('thread 2 q', 'thread 2 a')
        _seed_compaction('user', t1, 'thread 1 checkpoint')

        row1 = compaction_persistence.get_compaction('user', t1)
        row2 = compaction_persistence.get_compaction('user', t2)
        assert row1 is not None
        assert row2 is None, "thread 2 unaffected by thread 1's compaction"