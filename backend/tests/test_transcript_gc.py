"""Feature test for transcript garbage collection.

``Transcript.cleanup_unlinked_entries`` is the decay engine's transcript-GC step:
for every channel that has advanced a compaction watermark, it hard-deletes the
transcript rows BELOW that watermark that no live episode still cites, while
protecting the cited rows and everything at/above the watermark.

The regression this guards: discovery looked for a ``tool_calls`` row with
``tool_name='compaction'`` and ``$.status='success'`` — a shape the redesigned
compactor never writes (it writes a ``role='compaction'`` *transcript* row whose
own id is the watermark). So the all-channels pass found zero channels and
deleted nothing — a permanent silent no-op.

No mocks, no alternative path: real transcript writes via the production writer
the compactor itself calls (``write_input_row``, chat_history_compactor.py:116),
a real ``episodes`` reference row, and the real ``DecayEngineService`` decay step
as the single entry point. This test FAILS against the pre-fix discovery query
(nothing is reaped) and PASSES once discovery reads the watermark source.
"""

import json
import sqlite3

import pytest

from services.database_service import get_shared_db_service
from services.decay_engine_service import DecayEngineService
from services.transcript_service import Transcript


pytestmark = pytest.mark.unit


def _exists(conn: sqlite3.Connection, row_id: int) -> bool:
    return conn.execute(
        "SELECT 1 FROM transcript WHERE id = ?", (row_id,)
    ).fetchone() is not None


class TestTranscriptGC:
    def test_decay_pass_reaps_unlinked_below_watermark_and_protects_the_rest(
        self, db: sqlite3.Connection
    ) -> None:
        """One decay pass deletes only the unreferenced pre-watermark rows.

        Thread-scoped compaction (workstream C): the watermark keys off
        (channel, turn_id), so the compaction row is written INTO the thread
        (turn_id passed, exactly as the thread-scoped compactor does) and GC
        reaps rows below it IN THAT THREAD.

        Pins both sides of the contract in a single run:
          * below watermark + unreferenced  -> deleted
          * below watermark + episode-cited -> survives (evidence never orphaned)
          * the watermark row itself        -> survives (it is AT, not below)
          * above the watermark             -> survives (still live history)
        """
        channel = "user"
        turn_id = 1

        # Older conversation rows in ONE thread — the prod writer that opens
        # every turn. Replies carry the existing turn_id (workstream B).
        r1 = Transcript.write_input_row(channel, "user", "first old message", turn_id=turn_id)
        r2 = Transcript.write_input_row(channel, "user", "old message cited by an episode", turn_id=turn_id)
        r3 = Transcript.write_input_row(channel, "user", "third old message", turn_id=turn_id)

        # A live episode cites r2 — GC must never reap an episode's evidence.
        db.execute(
            "INSERT INTO episodes (id, gist, salience, channel, transcript_ids) "
            "VALUES (?, ?, ?, ?, ?)",
            ("ep-cite", "covers the second old message", 5, channel, json.dumps([r2])),
        )
        db.commit()

        # The compaction watermark row, written INTO the thread exactly as the
        # thread-scoped compactor does (turn_id passed so the checkpoint lands in
        # the thread it compacted); its own id is the watermark the GC must
        # discover for this (channel, turn_id) pair.
        watermark = Transcript.write_input_row(
            channel, "compaction", "## Voice\nliving checkpoint", turn_id=turn_id
        )

        # A newer row in the SAME thread, written after the watermark — must survive.
        r4 = Transcript.write_input_row(channel, "user", "post-watermark message", turn_id=turn_id)

        # Drive the real decay entry point: no channel arg -> the all-channels
        # discovery path that the regression broke.
        deleted = DecayEngineService()._cleanup_transcript()

        conn = get_shared_db_service()._get_connection()
        assert deleted == 2, "exactly r1 + r3 (below watermark, unreferenced) reaped"
        assert not _exists(conn, r1)
        assert not _exists(conn, r3)
        assert _exists(conn, r2), "episode-cited row must survive"
        assert _exists(conn, watermark), "the compaction watermark row itself must survive"
        assert _exists(conn, r4), "post-watermark row must survive"

    def test_channel_without_a_watermark_reaps_nothing(self, db: sqlite3.Connection) -> None:
        """No compaction row -> no watermark -> a pure no-op (rows untouched).

        Guards the other direction: GC must not delete history on a channel that
        has never compacted, even though rows exist.
        """
        channel = "user"
        a = Transcript.write_input_row(channel, "user", "message one")
        b = Transcript.write_input_row(channel, "user", "message two")

        deleted = DecayEngineService()._cleanup_transcript()

        conn = get_shared_db_service()._get_connection()
        assert deleted == 0
        assert _exists(conn, a)
        assert _exists(conn, b)
