"""Feature test for transcript garbage collection (two-axis).

``Transcript.cleanup_unlinked_entries`` is the decay engine's transcript-GC step.
It discovers compacted scopes from the dedicated ``compactions`` table and, per
turn, hard-deletes only the rows a checkpoint has folded — partitioned at the
turn's settle0:

* **main-owned** rows (``id < settle0``) — deletable once main absorbs the turn
  (``turn_id <= main_watermark``, the turn_id axis);
* **fork-owned** rows (``settle0 < id <= fork_watermark``) — deletable once the
  fork's own transcript-id checkpoint passes them.

settle0 itself is never collected (the fork read re-derives it every read), and
episode-cited rows are always preserved.

The regression this guards: discovery once looked for a watermark shape the
redesigned compactor never writes, so the all-channels pass found zero channels
and reaped nothing — a permanent silent no-op. No mocks, no alternative path:
real transcript writes via the production writers, real ``compactions`` rows via
the production ``write_compaction`` the compactor itself calls, a real
``episodes`` reference row, and the real ``DecayEngineService`` decay step as the
single entry point.
"""

import json
import sqlite3

import pytest

from services import compaction_persistence
from services.act_trail import ActTrail
from services.database_service import get_shared_db_service
from services.decay_engine_service import DecayEngineService
from services.transcript_service import Transcript

pytestmark = pytest.mark.unit


def _exists(conn: sqlite3.Connection, row_id: int) -> bool:
    return conn.execute(
        "SELECT 1 FROM transcript WHERE id = ?", (row_id,)
    ).fetchone() is not None


class TestTranscriptGC:
    def test_decay_pass_reaps_main_owned_below_settle0_and_protects_the_rest(
        self, db: sqlite3.Connection
    ) -> None:
        """A MAIN compaction absorbs a turn (turn_id axis); GC reaps the turn's
        main-owned rows below settle0 — except episode-cited ones — and preserves
        settle0 and every later, un-absorbed turn.

        Pins all four sides in one run:
          * below settle0 + un-cited + turn absorbed -> deleted
          * below settle0 + episode-cited            -> survives (evidence kept)
          * settle0 itself                           -> survives (re-derived each read)
          * a later turn above the watermark         -> survives (still live history)
        """
        channel = "user"

        # Turn 1 — an absorbed turn. The input row + an un-cited tool step sit
        # below settle0; a second tool step is episode-cited; the no-tool answer
        # is settle0.
        in1 = Transcript.write_input_row(channel, "user", "first old message")
        t1 = Transcript.turn_id_of_row(in1)
        step_uncited = Transcript.write_assistant_row(channel, "an un-cited working step", turn_id=t1)
        ActTrail().record(tool_name="search_files", params={}, result="ok", transcript_id=step_uncited)
        step_cited = Transcript.write_assistant_row(channel, "a step an episode cites", turn_id=t1)
        ActTrail().record(tool_name="search_files", params={}, result="ok", transcript_id=step_cited)
        settle1 = Transcript.write_assistant_row(channel, "first old answer", turn_id=t1)

        # A live episode cites step_cited — GC must never reap an episode's evidence.
        db.execute(
            "INSERT INTO episodes (id, gist, salience, channel, transcript_ids) "
            "VALUES (?, ?, ?, ?, ?)",
            ("ep-cite", "covers the cited step", 5, channel, json.dumps([step_cited])),
        )
        db.commit()

        # The MAIN checkpoint, on the turn_id axis (for_turn_id None) — exactly the
        # production write_compaction the compactor fires once the LLM returns.
        compaction_persistence.write_compaction(channel, None, t1, "## Voice\nliving checkpoint")

        # Turn 2 — opened after the watermark; turn_id > main_wm, so un-absorbed.
        in2 = Transcript.write_input_row(channel, "user", "post-watermark message")
        t2 = Transcript.turn_id_of_row(in2)
        settle2 = Transcript.write_assistant_row(channel, "post-watermark answer", turn_id=t2)

        deleted = DecayEngineService()._cleanup_transcript()

        conn = get_shared_db_service()._get_connection()
        assert deleted == 2, "exactly in1 + step_uncited (below settle0, absorbed, un-cited) reaped"
        assert not _exists(conn, in1)
        assert not _exists(conn, step_uncited)
        assert _exists(conn, step_cited), "episode-cited row must survive"
        assert _exists(conn, settle1), "settle0 itself must survive — re-derived on every read"
        assert _exists(conn, in2), "later un-absorbed turn must survive"
        assert _exists(conn, settle2)

    def test_decay_pass_reaps_fork_folded_rows_above_settle0_only(
        self, db: sqlite3.Connection
    ) -> None:
        """A FORK compaction (transcript.id axis) folds a thread's continuation up
        to its watermark; GC reaps only those fork-owned rows. With no MAIN
        watermark, the opening pair below settle0 is untouched (intersection, not
        union), and continuation above the fork watermark survives.
        """
        channel = "user"

        in0 = Transcript.write_input_row(channel, "user", "thread opener")
        t = Transcript.turn_id_of_row(in0)
        settle0 = Transcript.write_assistant_row(channel, "thread anchor answer", turn_id=t)
        r1_in = Transcript.write_input_row(channel, "user", "folded follow-up", turn_id=t)
        r1_settle = Transcript.write_assistant_row(channel, "folded follow-up answer", turn_id=t)
        r2_in = Transcript.write_input_row(channel, "user", "kept follow-up", turn_id=t)
        r2_settle = Transcript.write_assistant_row(channel, "kept follow-up answer", turn_id=t)

        # FORK checkpoint on turn t, folding the continuation through r1_settle.
        compaction_persistence.write_compaction(channel, t, r1_settle, "fork checkpoint")

        deleted = DecayEngineService()._cleanup_transcript()

        conn = get_shared_db_service()._get_connection()
        assert deleted == 2, "only r1_in + r1_settle (settle0 < id <= fork_wm) reaped"
        assert _exists(conn, in0), "main-owned opener survives — no MAIN watermark"
        assert _exists(conn, settle0), "settle0 itself is never collected"
        assert not _exists(conn, r1_in)
        assert not _exists(conn, r1_settle)
        assert _exists(conn, r2_in), "continuation above the fork watermark survives"
        assert _exists(conn, r2_settle)
