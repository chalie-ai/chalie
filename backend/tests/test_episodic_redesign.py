"""
Tests for the Episodic Memory Pipeline Redesign — Schema and EpisodicService storage.

Covers the store_episode() behaviour (columns, overlap/super-episode detection,
freshness-optional). EpisodeExtractorService tests were removed in Commit B when
that service was replaced by EpisodeEncoderProcessor.

Database strategy: builds an in-memory SQLite database using the real schema.sql so that
column definitions, constraints, and indexes are never out of sync with production.
"""

import json
import sqlite3
import uuid
from pathlib import Path
import pytest
from contextlib import contextmanager

pytestmark = pytest.mark.unit

_SCHEMA_PATH = Path(__file__).parent.parent / "schema.sql"


def _build_schema(conn: sqlite3.Connection) -> None:
    """Apply the full production schema.sql to an in-memory connection.

    Tries to load sqlite-vec first. If unavailable, skips vec0 statements so
    that the rest of the schema (episodes, FTS5, indexes) still applies.
    """
    sql = _SCHEMA_PATH.read_text()

    vec_available = False
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        vec_available = True
    except Exception:
        pass

    if vec_available:
        conn.executescript(sql)  # NOSONAR S3649 — sql is bundled schema.sql, not user input
    else:
        filtered = ';'.join(
            s for s in (stmt.strip() for stmt in sql.split(';'))
            if s and 'vec0' not in s.lower()
        )
        conn.executescript(filtered)  # NOSONAR S3649 — filtered from bundled schema.sql


# ── Fixture helpers ───────────────────────────────────────────────────────────

class _FakeDB:
    """Thin wrapper that satisfies EpisodicService's db_service.connection() API."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    @contextmanager
    def connection(self):
        yield self._conn
        self._conn.commit()


@pytest.fixture
def mem_db():
    """In-memory SQLite built from the real schema.sql — no DDL duplication."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _build_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def episodic_svc(mem_db):
    """EpisodicService backed by a fresh in-memory episodes table."""
    from services.episodic_service import EpisodicService
    fake_db = _FakeDB(mem_db)
    return EpisodicService(fake_db)


def _ep(**overrides) -> dict:
    """Minimal valid episode_data dict — 3 required fields."""
    base = {
        'gist': 'Test conversation',
        'salience': 5,
        'channel': 'programming',
    }
    base.update(overrides)
    return base


# ── store_episode: all 10 new Phase-0 columns ────────────────────────────────

class TestStoreEpisodeNewColumns:

    def test_transcript_ids_stored_as_json(self, mem_db, episodic_svc):
        """transcript_ids list is persisted as a JSON array."""
        data = _ep(transcript_ids=[10, 20, 30])

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT transcript_ids FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert json.loads(row['transcript_ids']) == [10, 20, 30]

    def test_transcript_id_start_end_stored(self, mem_db, episodic_svc):
        """transcript_id_start and transcript_id_end are stored independently."""
        data = _ep(transcript_id_start=5, transcript_id_end=29)

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute(
            "SELECT transcript_id_start, transcript_id_end FROM episodes WHERE id = ?",
            (episode_id,)
        ).fetchone()
        assert row['transcript_id_start'] == 5
        assert row['transcript_id_end'] == 29

    def test_emotional_valence_stored(self, mem_db, episodic_svc):
        """emotional_valence float is stored and retrievable."""
        data = _ep(emotional_valence=0.75)

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT emotional_valence FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert row['emotional_valence'] == pytest.approx(0.75)

    def test_emotional_arousal_stored(self, mem_db, episodic_svc):
        """emotional_arousal float is stored and retrievable."""
        data = _ep(emotional_arousal=0.4)

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT emotional_arousal FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert row['emotional_arousal'] == pytest.approx(0.4)

    def test_consolidated_from_stored_as_json(self, mem_db, episodic_svc):
        """consolidated_from list is persisted as a JSON array."""
        source_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        data = _ep(consolidated_from=source_ids)

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT consolidated_from FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert json.loads(row['consolidated_from']) == source_ids

    def test_storage_strength_stored(self, mem_db, episodic_svc):
        """storage_strength is stored and retrievable."""
        data = _ep(storage_strength=1.5)

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT storage_strength FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert row['storage_strength'] == pytest.approx(1.5)

    def test_retrieval_weight_stored(self, mem_db, episodic_svc):
        """retrieval_weight is stored and retrievable."""
        data = _ep(retrieval_weight=0.8)

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT retrieval_weight FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert row['retrieval_weight'] == pytest.approx(0.8)

    def test_new_columns_default_when_absent(self, mem_db, episodic_svc):
        """When optional columns are omitted, defaults are applied."""
        data = _ep()

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("""
            SELECT transcript_ids, emotional_valence,
                   emotional_arousal, consolidated_from, storage_strength, retrieval_weight,
                   transcript_id_start, transcript_id_end
            FROM episodes WHERE id = ?
        """, (episode_id,)).fetchone()

        assert json.loads(row['transcript_ids']) == []
        assert row['emotional_valence'] is None
        assert row['emotional_arousal'] is None
        assert json.loads(row['consolidated_from']) == []
        assert row['storage_strength'] == pytest.approx(1.0)
        assert row['retrieval_weight'] == pytest.approx(1.0)
        assert row['transcript_id_start'] is None
        assert row['transcript_id_end'] is None

    def test_new_columns_round_trip(self, mem_db, episodic_svc):
        """All optional columns survive a full store-and-read round trip."""
        src_ids = [str(uuid.uuid4())]
        data = _ep(
            transcript_ids=[1, 2, 3],
            transcript_id_start=1,
            transcript_id_end=3,
            emotional_valence=-0.3,
            emotional_arousal=0.6,
            consolidated_from=src_ids,
            storage_strength=1.2,
            retrieval_weight=0.9,
        )

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("""
            SELECT transcript_ids, transcript_id_start, transcript_id_end,
                   emotional_valence, emotional_arousal,
                   consolidated_from, storage_strength, retrieval_weight
            FROM episodes WHERE id = ?
        """, (episode_id,)).fetchone()

        assert json.loads(row['transcript_ids']) == [1, 2, 3]
        assert row['transcript_id_start'] == 1
        assert row['transcript_id_end'] == 3
        assert row['emotional_valence'] == pytest.approx(-0.3)
        assert row['emotional_arousal'] == pytest.approx(0.6)
        assert json.loads(row['consolidated_from']) == src_ids
        assert row['storage_strength'] == pytest.approx(1.2)
        assert row['retrieval_weight'] == pytest.approx(0.9)


# ── store_episode: overlap / super-episode behaviour ─────────────────────────
#
# Three tests that asserted INLINE super-episode creation inside store_episode()
# were deleted in Commit D (episodic-simplification arbiter pass):
#
#   test_exact_range_overlap_stores_both_and_creates_super
#   test_overlapping_range_stores_new_and_creates_super
#   test_super_episode_has_merged_transcript_range
#
# Rationale: the inline >50% overlap → super-episode path inside store_episode()
# was removed.  Super-episodes are now triggered by SubconsciousWorker step 1,
# which iterates channels with unconsolidated rows and runs
# SuperEpisodeEncoderProcessor.send() per channel (driven by
# find_super_candidates in episodic_service).  Coverage lives in
# test_super_episode_pipeline.py.

class TestStoreEpisodeOverlap:

    def test_adjacent_non_overlapping_range_is_stored_without_super(self, mem_db, episodic_svc):
        """Episodes with non-overlapping ranges are both stored with no super episode."""
        episodic_svc.store_episode(_ep(transcript_id_start=1, transcript_id_end=25))
        episodic_svc.store_episode(_ep(transcript_id_start=26, transcript_id_end=50))

        count = mem_db.execute(
            "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL"
        ).fetchone()[0]
        assert count == 2

        supers = mem_db.execute(
            "SELECT COUNT(*) FROM episodes WHERE consolidated_from != '[]'"
        ).fetchone()[0]
        assert supers == 0

    def test_soft_deleted_episode_does_not_trigger_super(self, mem_db, episodic_svc):
        """A soft-deleted episode's range is excluded from overlap detection."""
        first_id = episodic_svc.store_episode(
            _ep(transcript_id_start=1, transcript_id_end=25)
        )
        mem_db.execute(
            "UPDATE episodes SET deleted_at = datetime('now') WHERE id = ?",
            (first_id,)
        )
        mem_db.commit()

        second_id = episodic_svc.store_episode(
            _ep(transcript_id_start=1, transcript_id_end=25)
        )

        assert second_id != first_id
        active = mem_db.execute(
            "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL"
        ).fetchone()[0]
        # only the second episode is active (no super created for deleted overlap)
        assert active == 1

    def test_no_transcript_range_always_stores_without_super(self, mem_db, episodic_svc):
        """Episodes without transcript_id_start/end skip overlap check entirely."""
        id1 = episodic_svc.store_episode(_ep())
        id2 = episodic_svc.store_episode(_ep())

        assert id1 != id2

        count = mem_db.execute(
            "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL"
        ).fetchone()[0]
        assert count == 2

        supers = mem_db.execute(
            "SELECT COUNT(*) FROM episodes WHERE consolidated_from != '[]'"
        ).fetchone()[0]
        assert supers == 0
