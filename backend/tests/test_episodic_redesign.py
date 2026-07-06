# EpisodicService store_episode() — columns, overlap/super-episode detection, freshness.
# EpisodeExtractorService tests removed in Commit B (replaced by EpisodeEncoderProcessor).
# DB strategy: in-memory SQLite built from real schema.sql so column defs, constraints,
# and indexes stay in sync with production.

import json
import sqlite3
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from services.file_mapper_service import FileMapperService

if TYPE_CHECKING:
    from services.episodic_service import EpisodicService

pytestmark = pytest.mark.unit

_SCHEMA_PATH = FileMapperService.get_schema_path()


def _build_schema(conn: sqlite3.Connection) -> None:
    """Loads sqlite-vec extension if available; otherwise filters out vec0 statements."""
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

@pytest.fixture
def mem_db() -> Generator[sqlite3.Connection, None, None]:
    import services.database as _db_gateway

    conn = sqlite3.connect(":memory:")
    conn.isolation_level = None  # autocommit — matches the Database gateway's connections
    conn.row_factory = sqlite3.Row
    _build_schema(conn)

    # EpisodicService() reaches the DB through the Database gateway
    # (Database.conn()/transaction() → FileMapperService.get_db_path()). An in-memory
    # db is per-connection, so point the gateway at THIS exact handle — the only way
    # the service's writes and the test's reads share one database.
    sentinel = Path(":memory:episodic-test")
    with patch.object(FileMapperService, "get_db_path", return_value=sentinel):
        _db_gateway._local.conns = {str(sentinel): conn}
        _db_gateway._local.depths = {}
        # Bind the Model base's connection getter onto this exact handle — the
        # active-record Episode model derives every write/read connection through
        # Model._bound_connection(), so without this bind the service's INSERTs
        # would land on a stale (or unbound) connection, not this in-memory db.
        _db_gateway.Database().bind()
        try:
            yield conn
        finally:
            _db_gateway._local.conns = {}
            _db_gateway._local.depths = {}
    conn.close()


@pytest.fixture
def episodic_svc(mem_db: sqlite3.Connection) -> "EpisodicService":
    from services.episodic_service import EpisodicService
    return EpisodicService()


# Minimal valid episode_data dict — 3 required fields.
def _ep(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        'gist': 'Test conversation',
        'salience': 5,
        'channel': 'programming',
    }
    base.update(overrides)
    return base


# ── store_episode: column persistence ────────────────────────────────────────

class TestStoreEpisodeNewColumns:

    def test_transcript_ids_stored_as_json(self, mem_db: sqlite3.Connection, episodic_svc: "EpisodicService") -> None:
        data = _ep(transcript_ids=[10, 20, 30])

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT transcript_ids FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert json.loads(row['transcript_ids']) == [10, 20, 30]

    def test_transcript_id_start_end_stored(self, mem_db: sqlite3.Connection, episodic_svc: "EpisodicService") -> None:
        data = _ep(transcript_id_start=5, transcript_id_end=29)

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute(
            "SELECT transcript_id_start, transcript_id_end FROM episodes WHERE id = ?",
            (episode_id,)
        ).fetchone()
        assert row['transcript_id_start'] == 5
        assert row['transcript_id_end'] == 29

    def test_consolidated_from_stored_as_json(self, mem_db: sqlite3.Connection, episodic_svc: "EpisodicService") -> None:
        source_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        data = _ep(consolidated_from=source_ids)

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT consolidated_from FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert json.loads(row['consolidated_from']) == source_ids

    def test_new_columns_default_when_absent(self, mem_db: sqlite3.Connection, episodic_svc: "EpisodicService") -> None:
        data = _ep()

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("""
            SELECT transcript_ids, consolidated_from, retrieval_weight,
                   transcript_id_start, transcript_id_end
            FROM episodes WHERE id = ?
        """, (episode_id,)).fetchone()

        assert json.loads(row['transcript_ids']) == []
        assert json.loads(row['consolidated_from']) == []
        assert row['retrieval_weight'] == pytest.approx(1.0)
        assert row['transcript_id_start'] is None
        assert row['transcript_id_end'] is None

    def test_new_columns_round_trip(self, mem_db: sqlite3.Connection, episodic_svc: "EpisodicService") -> None:
        src_ids = [str(uuid.uuid4())]
        data = _ep(
            transcript_ids=[1, 2, 3],
            transcript_id_start=1,
            transcript_id_end=3,
            consolidated_from=src_ids,
            retrieval_weight=0.9,
        )

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("""
            SELECT transcript_ids, transcript_id_start, transcript_id_end,
                   consolidated_from, retrieval_weight
            FROM episodes WHERE id = ?
        """, (episode_id,)).fetchone()

        assert json.loads(row['transcript_ids']) == [1, 2, 3]
        assert row['transcript_id_start'] == 1
        assert row['transcript_id_end'] == 3
        assert json.loads(row['consolidated_from']) == src_ids
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

    def test_adjacent_non_overlapping_range_is_stored_without_super(self, mem_db: sqlite3.Connection, episodic_svc: "EpisodicService") -> None:
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

    def test_soft_deleted_episode_does_not_trigger_super(self, mem_db: sqlite3.Connection, episodic_svc: "EpisodicService") -> None:
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

    def test_no_transcript_range_always_stores_without_super(self, mem_db: sqlite3.Connection, episodic_svc: "EpisodicService") -> None:
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
