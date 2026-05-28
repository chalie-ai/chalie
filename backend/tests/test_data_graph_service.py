"""Tests for DataGraphService — typed knowledge graph with decay, edges, and LUT canonicalization.

Covers store, recall, fetch, edge operations, reinforce/demote, deletion,
decay_cycle, and LUT-based upsert paths.
"""

import contextlib
import math
import struct
import sqlite3

import pytest
from unittest.mock import MagicMock, patch

from services.data_graph_service import (
    DataGraphService,
    KIND_USER_SPECIFIC,
    KIND_SYSTEM,
    KIND_MISC,
    KIND_DOCUMENT,
    _KIND_POLICY,
)
from services.database_service import DatabaseService
from services.time_utils import utc_now


pytestmark = pytest.mark.unit



# ── Schema ────────────────────────────────────────────────────────

# Standalone (non-content-table) FTS so manual INSERT/DELETE work in tests.
# The production schema uses content='data_graph' with triggers, but for unit
# tests we control FTS sync explicitly.
DATA_GRAPH_DDL = [
    """
    CREATE TABLE IF NOT EXISTS data_graph (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        kind              TEXT NOT NULL,
        key               TEXT NOT NULL,
        value             TEXT,
        storage_strength  REAL NOT NULL DEFAULT 0.5,
        retrieval_weight  REAL NOT NULL DEFAULT 1.0,
        salience_score    REAL NOT NULL DEFAULT 0.0,
        evidence_count    INTEGER NOT NULL DEFAULT 1,
        first_seen_at     TEXT NOT NULL DEFAULT (datetime('now')),
        last_confirmed_at TEXT NOT NULL DEFAULT (datetime('now')),
        last_accessed_at  TEXT,
        source            TEXT,
        deleted_at        TEXT,
        active            INTEGER NOT NULL DEFAULT 1,
        search_queries    TEXT DEFAULT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_data_graph_kind ON data_graph(kind)",
    "CREATE INDEX IF NOT EXISTS idx_data_graph_key ON data_graph(key)",
    "CREATE INDEX IF NOT EXISTS idx_data_graph_retrieval ON data_graph(retrieval_weight DESC)",
    "CREATE INDEX IF NOT EXISTS idx_data_graph_active ON data_graph(kind, active) WHERE deleted_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_data_graph_confirmed ON data_graph(last_confirmed_at)",
    # Standalone FTS — production uses content-table trigger approach, but for
    # unit tests we sync manually (same pattern as knowledge tests).
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS data_graph_fts USING fts5(
        key, value, kind, search_queries,
        tokenize='porter unicode61'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS data_graph_edges (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        from_id          INTEGER NOT NULL REFERENCES data_graph(id) ON DELETE CASCADE,
        to_id            INTEGER NOT NULL REFERENCES data_graph(id) ON DELETE CASCADE,
        edge_type        TEXT NOT NULL DEFAULT 'related',
        strength         REAL NOT NULL DEFAULT 1.0,
        created_at       TEXT NOT NULL DEFAULT (datetime('now')),
        last_accessed_at TEXT,
        UNIQUE (from_id, to_id, edge_type)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_data_graph_edges_from ON data_graph_edges(from_id, edge_type)",
    "CREATE INDEX IF NOT EXISTS idx_data_graph_edges_to ON data_graph_edges(to_id, edge_type)",
    # Stub vec tables — sqlite-vec (vec0) is NOT available in unit test environments.
    # The service calls `DELETE FROM data_graph_key_vec WHERE rowid=?` inside
    # hard_delete_by_id(). We create regular tables that accept those DELETEs so
    # the service's exception handler is not triggered and the operation succeeds.
    """
    CREATE TABLE IF NOT EXISTS data_graph_key_vec (
        rowid INTEGER PRIMARY KEY,
        embedding BLOB
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS data_graph_value_vec (
        rowid INTEGER PRIMARY KEY,
        embedding BLOB
    )
    """,
    # LUT miss tracking — required by the LUT miss path in store()
    """
    CREATE TABLE IF NOT EXISTS concept_lut_misses (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        kind       TEXT NOT NULL,
        key        TEXT NOT NULL,
        value_preview TEXT,
        count      INTEGER NOT NULL DEFAULT 1,
        first_seen TEXT NOT NULL,
        last_seen  TEXT NOT NULL,
        UNIQUE(kind, key)
    )
    """,
]

# sqlite-vec virtual tables (vec0) are not available in bare SQLite unit test
# builds. The service handles their absence gracefully (non-fatal logged debug).
# We intentionally skip creating them so tests stay hermetic.


# ── DB fixture ────────────────────────────────────────────────────

@pytest.fixture
def db_service(tmp_path):
    """Real SQLite DB with data_graph schema, shared connection for nested contexts.

    Uses a single persistent connection to mimic the production thread-local
    pattern, so the service's nested ``with self.db.connection()`` blocks all
    see the same uncommitted data within a transaction.
    """
    db_path = str(tmp_path / "test_data_graph.db")

    shared_conn = sqlite3.connect(db_path)
    shared_conn.row_factory = sqlite3.Row
    shared_conn.execute("PRAGMA foreign_keys = ON")
    for ddl in DATA_GRAPH_DDL:
        shared_conn.execute(ddl)
    shared_conn.commit()

    db = DatabaseService.__new__(DatabaseService)
    db._db_path = db_path
    db._init_complete = True
    db.db_path = db_path

    _depth = [0]

    @contextlib.contextmanager
    def _connection():
        _depth[0] += 1
        try:
            yield shared_conn
            if _depth[0] == 1:
                shared_conn.commit()
        except Exception:
            if _depth[0] == 1:
                shared_conn.rollback()
            raise
        finally:
            _depth[0] -= 1

    db.connection = _connection
    yield db
    shared_conn.close()


@pytest.fixture
def svc(db_service):
    """DataGraphService with real SQLite backend, embedding generation disabled.

    _generate_embedding returns None so vec0 operations are skipped (graceful
    degradation path in the service).
    """
    service = DataGraphService(db_service)
    service._generate_embedding = MagicMock(return_value=None)
    return service


# ── Helpers ───────────────────────────────────────────────────────

def _rid(result: dict) -> int:
    """Extract a stable integer row identifier from a store/fetch result dict.

    SQLite's ``SELECT rowid, *`` on a table with ``id INTEGER PRIMARY KEY``
    makes ``rowid`` and ``id`` point to the same integer. Depending on the
    sqlite3.Row serialisation, ``dict(row)`` may expose the column only as
    ``id``. This helper checks both so test assertions stay stable.
    """
    return result.get('rowid') or result.get('id')


def _raw_row(db_service, rowid: int) -> dict:
    """Fetch a data_graph row by rowid for assertion, bypassing the service."""
    with db_service.connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT rowid, * FROM data_graph WHERE rowid=?", (rowid,))
        row = cursor.fetchone()
        cursor.close()
        return dict(row) if row else None


def _raw_all(db_service) -> list:
    """Fetch all data_graph rows (including deleted/inactive)."""
    with db_service.connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT rowid, * FROM data_graph")
        rows = cursor.fetchall()
        cursor.close()
        return [dict(r) for r in rows]


def _raw_fts(db_service, query: str) -> list:
    """Direct FTS search; returns list of matching rowids."""
    with db_service.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT rowid FROM data_graph_fts WHERE data_graph_fts MATCH ?",
            (query,)
        )
        rows = cursor.fetchall()
        cursor.close()
        return [r[0] for r in rows]


def _insert_row(db_service, *, kind='user_specific', key='test_key',
                value='test_value', active=1, deleted_at=None,
                retrieval_weight=1.0, storage_strength=0.5,
                evidence_count=1, last_confirmed_at=None, source=None) -> int:
    """Insert a raw data_graph row and sync FTS; returns the rowid."""
    now = utc_now().isoformat()
    lc = last_confirmed_at or now
    with db_service.connection() as conn:
        conn.execute("""
            INSERT INTO data_graph
                (kind, key, value, active, deleted_at, retrieval_weight,
                 storage_strength, evidence_count, first_seen_at,
                 last_confirmed_at, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (kind, key, value, active, deleted_at, retrieval_weight,
              storage_strength, evidence_count, now, lc, source))
        rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO data_graph_fts(rowid, key, value, kind, search_queries) VALUES (?, ?, ?, ?, ?)",
            (rowid, key, value or '', kind, '')
        )
        return rowid


def _get_db_id(db_service, rowid: int) -> int:
    """Retrieve the data_graph.id for a given rowid (they are equal for INTEGER PRIMARY KEY)."""
    with db_service.connection() as conn:
        row = conn.execute("SELECT id FROM data_graph WHERE rowid=?", (rowid,)).fetchone()
        return row[0] if row else None


# TestStore
# ══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestStore:

    def test_store_new_row(self, svc, db_service):
        """Basic insert: all required columns populated, FTS synced."""
        result = svc.store(KIND_USER_SPECIFIC, 'user_name', 'Jordan')

        assert result is not None
        assert result['kind'] == KIND_USER_SPECIFIC
        assert result['key'] == 'user_name'
        assert result['value'] == 'Jordan'
        assert result['evidence_count'] == 1
        assert result['retrieval_weight'] == pytest.approx(1.0)
        assert result['storage_strength'] == pytest.approx(0.5)

        # Row persisted in DB
        row_id = _rid(result)
        assert row_id is not None
        raw = _raw_row(db_service, row_id)
        assert raw is not None
        assert raw['active'] == 1
        assert raw['deleted_at'] is None

        # FTS indexed — the value token should be findable
        fts_hits = _raw_fts(db_service, '"Jordan"*')
        assert row_id in fts_hits

    def test_store_reinforce_same_value_bumps_evidence(self, svc, db_service):
        """Same kind+key+value → evidence_count increases, retrieval_weight reset to 1.0."""
        r1 = svc.store(KIND_USER_SPECIFIC, 'favourite_food', 'pizza')
        assert r1 is not None
        assert r1['evidence_count'] == 1
        r1_id = _rid(r1)

        # Decay retrieval_weight before reinforce so we can see it reset
        with db_service.connection() as conn:
            conn.execute(
                "UPDATE data_graph SET retrieval_weight=0.7 WHERE rowid=?", (r1_id,)
            )

        r2 = svc.store(KIND_USER_SPECIFIC, 'favourite_food', 'pizza')
        assert r2 is not None
        assert r2['evidence_count'] == 2
        assert r2['retrieval_weight'] == pytest.approx(1.0)
        assert r2['storage_strength'] > r1['storage_strength']

        # Only one active row for this key — first_seen_at must be unchanged
        all_rows = _raw_all(db_service)
        active = [r for r in all_rows if r['key'] == 'favourite_food' and r['active']]
        assert len(active) == 1
        assert active[0]['first_seen_at'] == r1['first_seen_at']

    _FAKE_EMB = [0.1] * 768  # non-None sentinel; KNN is patched so value doesn't matter

    def _lut_hit(self, canonical_key: str, rule: str, cos: float = 0.95):
        """Build a fake LUT hit dict for patching _lookup_concept_lut."""
        return {'canonical_key': canonical_key, 'rule': rule, 'cos': cos}

    def test_store_user_specific_lut_temporal_canonicalizes_and_supersedes(self, svc, db_service):
        """New key hits LUT → canonical key rewrite → existing canonical row demoted, new inserted, edges created."""
        # Seed a row with the canonical key directly
        seed_rowid = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='residence', value='Valletta')
        seed_id = _get_db_id(db_service, seed_rowid)

        # Store with an alias key — LUT maps it to 'residence' (temporal rule).
        # _generate_embedding must return a non-None value so _lookup_concept_lut is invoked.
        svc._generate_embedding = MagicMock(return_value=self._FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=self._lut_hit('residence', 'temporal')):
            r2 = svc.store(KIND_USER_SPECIFIC, 'residency', 'Swieqi')

        assert r2 is not None
        assert r2['value'] == 'Swieqi'
        assert r2['key'] == 'residence'
        assert r2['id'] != seed_id

        # Old canonical row demoted
        old = _raw_row(db_service, seed_rowid)
        assert old['active'] == 0
        assert old['retrieval_weight'] < 1.0

        # supersedes/superseded_by edges present
        with db_service.connection() as conn:
            edges = conn.execute("SELECT from_id, to_id, edge_type FROM data_graph_edges").fetchall()
        edge_set = {(e[0], e[1], e[2]) for e in edges}
        assert (r2['id'], seed_id, 'supersedes') in edge_set
        assert (seed_id, r2['id'], 'superseded_by') in edge_set

    def test_store_user_specific_lut_coexist_different_value_inserts_new(self, svc, db_service):
        """Coexist rule + different value → both rows active, no supersession."""
        r1 = svc.store(KIND_USER_SPECIFIC, 'favorite_foods', 'pizza')
        assert r1 is not None

        # _generate_embedding must return non-None so LUT lookup is attempted.
        svc._generate_embedding = MagicMock(return_value=self._FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=self._lut_hit('favorite_foods', 'coexist')):
            r2 = svc.store(KIND_USER_SPECIFIC, 'favorite_foods', 'pasta')

        assert r2 is not None
        assert r2['id'] != r1['id']

        # Both rows active
        with db_service.connection() as conn:
            rows = conn.execute(
                "SELECT active FROM data_graph WHERE key='favorite_foods'"
            ).fetchall()
        assert all(r[0] == 1 for r in rows), "Both coexist rows must be active"
        assert len(rows) == 2

        # No supersedes edge
        with db_service.connection() as conn:
            edges = conn.execute(
                "SELECT edge_type FROM data_graph_edges"
            ).fetchall()
        assert not any(e[0] == 'supersedes' for e in edges)

    def test_store_user_specific_lut_immutable_same_value_reinforces(self, svc, db_service):
        """Immutable rule + same value → reinforce, no conflict."""
        r1 = svc.store(KIND_USER_SPECIFIC, 'birth_date', '1990-01-01')
        assert r1 is not None
        r1_id = _rid(r1)

        with patch.object(svc, '_lookup_concept_lut', return_value=self._lut_hit('birth_date', 'immutable')):
            r2 = svc.store(KIND_USER_SPECIFIC, 'birth_date', '1990-01-01')

        assert _rid(r2) == r1_id
        assert r2['evidence_count'] == 2

    def test_store_user_specific_lut_immutable_different_value_returns_conflict(self, svc, db_service):
        """Immutable rule + different value → conflict dict returned, original row preserved."""
        svc.store(KIND_USER_SPECIFIC, 'birth_date', 'March 15')

        svc._generate_embedding = MagicMock(return_value=self._FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=self._lut_hit('birth_date', 'immutable')):
            result = svc.store(KIND_USER_SPECIFIC, 'birth_date', 'March 20')

        assert result is not None
        assert result.get('status') == 'conflict'
        assert result['rule'] == 'immutable'
        assert result.get('old_value') == 'March 15'
        assert result.get('value') == 'March 20'

        # Original row still active
        with db_service.connection() as conn:
            active_rows = conn.execute(
                "SELECT value FROM data_graph WHERE key='birth_date' AND active=1"
            ).fetchall()
        assert len(active_rows) == 1
        assert active_rows[0][0] == 'March 15'

    def test_store_user_specific_lut_miss_inserts_and_records_miss(self, svc, db_service):
        """LUT miss (no hit above threshold) → row inserted with original key, miss recorded."""
        svc._generate_embedding = MagicMock(return_value=self._FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=None), \
             patch.object(svc, '_get_lut_miss_top_cos', return_value=0.42):
            result = svc.store(KIND_USER_SPECIFIC, 'dryer_streak', 'won 3 in a row')

        assert result is not None
        assert result['key'] == 'dryer_streak'
        assert result['value'] == 'won 3 in a row'

        with db_service.connection() as conn:
            miss = conn.execute(
                "SELECT count, key FROM concept_lut_misses WHERE key='dryer_streak'"
            ).fetchone()
        assert miss is not None
        assert miss[0] == 1



# TestRecall
# ══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRecall:

    def test_recall_basic_fts_match(self, svc, db_service):
        """Rows matching FTS query are returned with a composite_score > 0."""
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='hobby', value='photography')

        results = svc.recall("photography")
        assert len(results) == 1
        assert results[0]['key'] == 'hobby'
        assert results[0]['composite_score'] > 0.0

    def test_recall_filters_deleted_rows(self, svc, db_service):
        """Rows with deleted_at set are not returned."""
        now = utc_now().isoformat()
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='secret',
                    value='buried', deleted_at=now)
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='visible', value='present')

        results = svc.recall("present buried")
        keys = [r['key'] for r in results]
        assert 'secret' not in keys

    def test_recall_kinds_filter(self, svc, db_service):
        """kinds= restricts results to those kinds only."""
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='pref', value='dark mode')
        _insert_row(db_service, kind=KIND_SYSTEM, key='rule', value='dark mode response')

        results = svc.recall("dark mode", kinds=[KIND_SYSTEM])
        kinds_returned = {r['kind'] for r in results}
        assert kinds_returned == {KIND_SYSTEM}

    def test_recall_touch_accessed_sets_last_accessed_at(self, svc, db_service):
        """After recall, last_accessed_at is populated on returned rows."""
        rowid = _insert_row(db_service, kind=KIND_USER_SPECIFIC,
                            key='birthplace', value='Malta')

        raw_before = _raw_row(db_service, rowid)
        assert raw_before['last_accessed_at'] is None

        results = svc.recall("Malta birthplace")
        assert len(results) == 1

        raw_after = _raw_row(db_service, rowid)
        assert raw_after['last_accessed_at'] is not None

    def test_recall_graph_expansion_includes_neighbours(self, svc, db_service):
        """Rows connected by edges are included in the result set via 1-hop expansion."""
        rowid_a = _insert_row(db_service, kind=KIND_USER_SPECIFIC,
                              key='partner', value='Alex', retrieval_weight=1.0)
        rowid_b = _insert_row(db_service, kind=KIND_USER_SPECIFIC,
                              key='partner_job', value='doctor')

        id_a = _get_db_id(db_service, rowid_a)
        id_b = _get_db_id(db_service, rowid_b)

        with db_service.connection() as conn:
            conn.execute("""
                INSERT INTO data_graph_edges (from_id, to_id, edge_type, strength, created_at)
                VALUES (?, ?, 'related', 1.0, datetime('now'))
            """, (id_a, id_b))

        results = svc.recall("Alex partner", expand_graph=True, limit=10)
        keys = [r['key'] for r in results]
        assert 'partner' in keys
        assert 'partner_job' in keys



# TestFetch
# ══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFetch:

    def test_fetch_basic_returns_active_non_deleted(self, svc, db_service):
        """Default fetch returns only active, non-deleted rows."""
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='name', value='Jordan')
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='gone', value='x',
                    deleted_at=utc_now().isoformat())
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='inactive', value='y', active=0)

        rows = svc.fetch()
        keys = {r['key'] for r in rows}
        assert 'name' in keys
        assert 'gone' not in keys
        assert 'inactive' not in keys



# TestDeletion
# ══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestDeletion:

    def test_soft_delete_sets_deleted_at(self, svc, db_service):
        """soft_delete_by_id marks the row with a deleted_at timestamp."""
        rowid = _insert_row(db_service, kind=KIND_USER_SPECIFIC,
                            key='del_key', value='del_val')

        result = svc.soft_delete_by_id(rowid)
        assert result is True

        raw = _raw_row(db_service, rowid)
        assert raw['deleted_at'] is not None

    def test_soft_delete_removes_from_fts(self, svc, db_service):
        """After soft delete the row is no longer findable via FTS."""
        rowid = _insert_row(db_service, kind=KIND_USER_SPECIFIC,
                            key='fts_del', value='uniquetoken')

        fts_before = _raw_fts(db_service, '"uniquetoken"*')
        assert rowid in fts_before

        svc.soft_delete_by_id(rowid)

        fts_after = _raw_fts(db_service, '"uniquetoken"*')
        assert rowid not in fts_after


# TestDecayCycle
# ══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestDecayCycle:

    def test_decay_skips_system_kind(self, svc, db_service):
        """system kind has ttl_days=None so decay_cycle must never modify it."""
        assert _KIND_POLICY[KIND_SYSTEM]['ttl_days'] is None

        rowid = _insert_row(db_service, kind=KIND_SYSTEM, key='rule',
                            value='be concise', retrieval_weight=0.9,
                            last_confirmed_at='2020-01-01T00:00:00+00:00')

        svc.decay_cycle()

        raw = _raw_row(db_service, rowid)
        assert raw['retrieval_weight'] == pytest.approx(0.9, abs=0.001)

    def test_decay_reduces_retrieval_weight_for_old_user_specific(self, svc, db_service):
        """Old user_specific rows have retrieval_weight reduced via power-law decay."""
        # Row confirmed in 2020 → age ~years → decayed well below 1.0
        old_ts = '2020-01-01T00:00:00+00:00'
        rowid = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='old_fact',
                            value='some fact', retrieval_weight=1.0,
                            last_confirmed_at=old_ts)

        svc.decay_cycle()

        raw = _raw_row(db_service, rowid)
        # Must be less than 1.0 and at or above salience_floor=0.2
        assert raw['retrieval_weight'] < 1.0
        assert raw['retrieval_weight'] >= _KIND_POLICY[KIND_USER_SPECIFIC]['salience_floor']

    def test_decay_hard_deletes_expired_misc_at_floor(self, svc, db_service):
        """misc rows with retrieval_weight < 0.01 and confirmed > 2 days ago are hard-deleted."""
        old_ts = '2020-01-01T00:00:00+00:00'
        rowid = _insert_row(db_service, kind=KIND_MISC, key='old_scratch',
                            value='irrelevant', retrieval_weight=0.005,
                            last_confirmed_at=old_ts)

        svc.decay_cycle()

        raw = _raw_row(db_service, rowid)
        assert raw is None, "Expired misc row should have been hard-deleted by decay_cycle"




# TestDocumentKind
# ══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestDocumentKind:

    def test_store_document_kind(self, svc, db_service):
        """Store a document artifact and verify kind, key, value, source are persisted."""
        result = svc.store(KIND_DOCUMENT, 'doc:test123:000', 'chunk zero content', source='doc:test123')

        assert result is not None
        assert result['kind'] == KIND_DOCUMENT
        assert result['key'] == 'doc:test123:000'
        assert result['value'] == 'chunk zero content'
        assert result['source'] == 'doc:test123'

        row = _raw_row(db_service, _rid(result))
        assert row is not None
        assert row['kind'] == KIND_DOCUMENT
        assert row['active'] == 1
        assert row['deleted_at'] is None



# TestHardDeleteBySourcePrefix
# ══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestHardDeleteBySourcePrefix:
    """hard_delete_by_source_prefix deletes all rows whose source starts with
    the given prefix, cleans FTS entries, and returns the correct count."""

    def test_deletes_matching_rows_returns_count(self, svc, db_service):
        """All rows with the target source prefix are removed; count matches."""
        _insert_row(db_service, kind=KIND_DOCUMENT, key='doc:abc:000',
                    value='chunk 0', source='document:abc')
        _insert_row(db_service, kind=KIND_DOCUMENT, key='doc:abc:001',
                    value='chunk 1', source='document:abc')
        _insert_row(db_service, kind=KIND_DOCUMENT, key='doc:abc:002',
                    value='chunk 2', source='document:abc')

        deleted = svc.hard_delete_by_source_prefix('document:abc')

        assert deleted == 3
        remaining = _raw_all(db_service)
        assert all(r.get('source') != 'document:abc' for r in remaining)




@pytest.mark.unit
class TestForget:
    """forget() — rule-aware hard-delete for user memory.

    Each test covers one distinct branch of the forget() dispatch.  All tests
    use the real DB fixture with no mocks — embedding generation and LUT lookup
    are bypassed by patching only the two private helpers that touch external
    state (embedding model, sqlite-vec), matching the pattern used in TestStore.

    Mock decision: _generate_embedding and _lookup_concept_lut are patched via
    patch.object because they touch the ONNX model and sqlite-vec extension,
    neither of which is available in the unit test environment.  All DB writes,
    reads, edge deletions, and FTS operations run against the real SQLite
    fixture.  The nightly scenarios 084, 096, 097, 098, 099 provide end-to-end
    coverage for these paths against the production stack.
    """

    _FAKE_EMB = [0.1] * 768

    def _lut_hit(self, canonical_key: str, rule: str, cos: float = 0.95):
        """Return a fake LUT hit dict for patching _lookup_concept_lut."""
        return {'canonical_key': canonical_key, 'rule': rule, 'cos': cos}

    def test_forget_temporal_no_value_deletes_all_versions(self, svc, db_service):
        """Temporal key forget with no value param removes ALL version rows and their edges."""
        # Store two versions: seed old row directly, let service create the current one.
        old_rowid = _insert_row(db_service, kind=KIND_USER_SPECIFIC,
                                key='residence', value='Valletta', active=0)
        new_rowid = _insert_row(db_service, kind=KIND_USER_SPECIFIC,
                                key='residence', value='Swieqi', active=1)
        old_id = _get_db_id(db_service, old_rowid)
        new_id = _get_db_id(db_service, new_rowid)

        # Wire a supersedes edge between the two so we can assert it is also removed.
        with db_service.connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO data_graph_edges (from_id, to_id, edge_type, strength, created_at) "
                "VALUES (?, ?, 'supersedes', 1.0, datetime('now'))",
                (new_id, old_id)
            )

        svc._generate_embedding = MagicMock(return_value=self._FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=self._lut_hit('residence', 'temporal')):
            result = svc.forget(KIND_USER_SPECIFIC, 'residence')

        assert result is not None
        assert result['status'] == 'forgotten_all'
        assert result['versions_removed'] == 2

        # All rows gone
        with db_service.connection() as conn:
            rows = conn.execute(
                "SELECT id FROM data_graph WHERE key='residence'"
            ).fetchall()
        assert rows == [], "All temporal versions must be hard-deleted"

        # Edges cleaned up
        with db_service.connection() as conn:
            edges = conn.execute(
                "SELECT id FROM data_graph_edges "
                "WHERE from_id IN (?,?) OR to_id IN (?,?)",
                (old_id, new_id, old_id, new_id)
            ).fetchall()
        assert edges == [], "Edges for deleted rows must be removed"

    def test_forget_coexist_specific_value_removes_only_that_row(self, svc, db_service):
        """Coexist key forget with explicit value removes that value, leaves others active."""
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='food_and_drink', value='pizza')
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='food_and_drink', value='pasta')

        svc._generate_embedding = MagicMock(return_value=self._FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=self._lut_hit('food_and_drink', 'coexist')):
            result = svc.forget(KIND_USER_SPECIFIC, 'food_and_drink', value='pizza')

        assert result is not None
        assert result['status'] == 'forgotten'

        with db_service.connection() as conn:
            rows = conn.execute(
                "SELECT value FROM data_graph WHERE key='food_and_drink' AND deleted_at IS NULL"
            ).fetchall()
        values = [r[0] for r in rows]
        assert 'pizza' not in values, "Forgotten value must be physically removed"
        assert 'pasta' in values, "Other coexist values must remain untouched"

    def test_forget_immutable_hard_deletes_without_protection(self, svc, db_service):
        """Immutable key forget removes the single row — no protection from deletion."""
        rowid = _insert_row(db_service, kind=KIND_USER_SPECIFIC,
                            key='birth_date', value='1990-03-15')

        svc._generate_embedding = MagicMock(return_value=self._FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=self._lut_hit('birth_date', 'immutable')):
            result = svc.forget(KIND_USER_SPECIFIC, 'birth_date')

        assert result is not None
        assert result['status'] == 'forgotten'
        assert result['old_value'] == '1990-03-15'

        assert _raw_row(db_service, rowid) is None, "Immutable forget must physically remove the row"



# TestBackfillMissingEmbeddings
# ══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestBackfillMissingEmbeddings:
    """Verify that recall() triggers lazy backfill for rows inserted via raw SQL.

    The test environment uses plain tables for data_graph_key_vec and
    data_graph_value_vec (not sqlite-vec virtuals), so INSERT OR REPLACE works
    identically to production — only the MATCH KNN syntax is absent, which is
    why backfill-seeded rows are surfaced via FTS rather than vec search here.
    """

    _FAKE_EMB = [0.1, 0.2, 0.3]

    def _bare_insert(self, db_service, *, kind='user_specific',
                     key='sister_name', value='Sofia') -> int:
        """Insert a data_graph row with NO vec or FTS entries — mirrors raw-SQL seeder."""
        now = utc_now().isoformat()
        with db_service.connection() as conn:
            conn.execute("""
                INSERT INTO data_graph
                    (kind, key, value, active, first_seen_at, last_confirmed_at)
                VALUES (?, ?, ?, 1, ?, ?)
            """, (kind, key, value, now, now))
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def test_backfill_populates_key_vec_on_recall(self, svc, db_service):
        """Recall triggers backfill: key_vec gains a row for a bare-inserted id."""
        row_id = self._bare_insert(db_service, key='sisters_name', value='Sofia')

        # Baseline: key_vec must be empty for this id before recall
        with db_service.connection() as conn:
            count_before = conn.execute(
                "SELECT COUNT(*) FROM data_graph_key_vec WHERE rowid=?", (row_id,)
            ).fetchone()[0]
        assert count_before == 0, "key_vec must have no entry before backfill"

        # Override _generate_embedding so backfill actually stores a blob
        svc._generate_embedding = MagicMock(return_value=self._FAKE_EMB)

        svc.recall("sisters name Sofia")

        # After recall, key_vec must contain the backfilled row
        with db_service.connection() as conn:
            count_after = conn.execute(
                "SELECT COUNT(*) FROM data_graph_key_vec WHERE rowid=?", (row_id,)
            ).fetchone()[0]
        assert count_after == 1, "key_vec must have an entry after backfill"


# ══════════════════════════════════════════════════════════════════
# TestRecallCosScore  (cycle-33: cos_score relevance labelling)
# ══════════════════════════════════════════════════════════════════


def _norm(v):
    """Return unit-normalised copy of float list v."""
    mag = math.sqrt(sum(x * x for x in v))
    if mag == 0:
        return v
    return [x / mag for x in v]


def _pack(v):
    """Pack float list to 32-bit float blob (sqlite-vec format)."""
    return struct.pack(f'{len(v)}f', *v)


def _unpack(blob, n):
    """Unpack n floats from a blob."""
    return list(struct.unpack(f'{n}f', blob[:n * 4]))


def _l2sq(a, b):
    """Squared L2 distance between two equal-length float lists."""
    return sum((x - y) ** 2 for x, y in zip(a, b))


class _VecSimCursor:
    """Wraps a real sqlite3.Cursor.

    Intercepts ``SELECT rowid, distance FROM data_graph_key_vec WHERE embedding
    MATCH ? AND k = ?`` (and the value-vec equivalent), replacing them with a
    real cosine-computation pass over the plain blob rows in the test tables.

    All other queries are forwarded transparently.
    """

    _KEY_VEC_MATCH = "data_graph_key_vec"
    _VAL_VEC_MATCH = "data_graph_value_vec"

    def __init__(self, real_cursor, conn, dim):
        self._cur = real_cursor
        self._conn = conn
        self._dim = dim
        self._rows = None  # populated when a MATCH query is intercepted

    def _intercept_match(self, sql, params):
        """Return True if we should handle this query ourselves."""
        s = sql.lower()
        return (
            "match" in s
            and ("data_graph_key_vec" in s or "data_graph_value_vec" in s)
        )

    def _run_vec_sim(self, sql, params):
        """Compute cosine distances manually from the plain blob table."""
        s = sql.lower()
        table = (
            "data_graph_key_vec"
            if "data_graph_key_vec" in s
            else "data_graph_value_vec"
        )
        query_blob, k = params[0], params[1]
        query_vec = _norm(_unpack(query_blob, self._dim))
        # Use the real underlying connection cursor (not the wrapper) to avoid
        # recursion when fetching the plain blob rows.
        plain_cur = self._conn.cursor()
        plain_cur.execute(f"SELECT rowid, embedding FROM {table}")
        rows = plain_cur.fetchall()
        plain_cur.close()
        results = []
        for rowid, blob in rows:
            if not blob:
                continue
            row_vec = _norm(_unpack(blob, self._dim))
            dist_sq = _l2sq(query_vec, row_vec)
            # sqlite-vec returns sqrt of dist² for L2; production code uses
            # _l2_dist_to_cosine(distance) = max(0, 1 - distance^2/2)
            # which expects the raw L2 distance (not squared).
            dist = math.sqrt(max(0.0, dist_sq))
            results.append((rowid, dist))
        results.sort(key=lambda x: x[1])
        self._rows = results[:k]

    def execute(self, sql, params=()):
        if self._intercept_match(sql, params):
            self._run_vec_sim(sql, params)
        else:
            self._rows = None
            self._cur.execute(sql, params)
        return self

    def fetchall(self):
        if self._rows is not None:
            r = self._rows
            self._rows = None
            return r
        return self._cur.fetchall()

    def fetchone(self):
        if self._rows is not None:
            r = self._rows[0] if self._rows else None
            self._rows = None
            return r
        return self._cur.fetchone()

    def close(self):
        self._cur.close()

    def __iter__(self):
        if self._rows is not None:
            yield from self._rows
            self._rows = None
        else:
            yield from self._cur


class _VecSimConn:
    """Thin proxy around a real sqlite3.Connection that intercepts MATCH queries.

    sqlite3.Connection.cursor is a read-only C-level slot, so we cannot monkey-
    patch it directly.  Instead we expose a full connection-like interface and
    override only cursor() to return a _VecSimCursor.
    """

    def __init__(self, real_conn, dim):
        self._conn = real_conn
        self._dim = dim

    # -- cursor interception ------------------------------------------------

    def cursor(self):
        return _VecSimCursor(self._conn.cursor(), self._conn, self._dim)

    # -- transaction forwarding ---------------------------------------------

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        return self._conn.close()

    # -- convenience pass-throughs (production code calls these directly on conn)

    def execute(self, sql, params=()):
        """Forward direct conn.execute() calls through a VecSimCursor."""
        cur = self.cursor()
        cur.execute(sql, params)
        return cur

    def __getattr__(self, name):
        """Proxy any attribute not explicitly overridden to the real connection."""
        return getattr(self._conn, name)


class _VecSimDb:
    """Wraps a db_service fixture to make data_graph_key/value_vec tables
    behave like sqlite-vec virtual tables for unit tests.

    The embedding dimension is fixed at construction time so the cursor can
    unpack stored blobs consistently.
    """

    def __init__(self, real_db, dim):
        self._db = real_db
        self._dim = dim
        # Expose attributes the service reads directly.
        self.db_path = real_db.db_path
        self._db_path = real_db.db_path
        self._init_complete = real_db._init_complete

    @contextlib.contextmanager
    def connection(self):
        with self._db.connection() as conn:
            yield _VecSimConn(conn, self._dim)


@pytest.fixture
def svc_vec(db_service):
    """DataGraphService with a vec-simulating DB wrapper (dim=4).

    _generate_embedding is replaced with a real Python callable that returns
    controlled 4-dimensional float vectors for specific text tokens.

    Vector assignments (unit-normalised):
        'food recipe'             → [1, 0, 0, 0]   (query direction)
        'apple pie recipe'        → [1, 0, 0, 0]   (aligned with query)
        'dentist appointment'     → [1, 0, 0, 0]   (query direction for test 3)
        'dentist dental appt'     → [0, 1, 0, 0]   (orthogonal row key embedding)
        'dentist appt high'       → [0, 1, 0, 0]   (orthogonal, high-weight row)

    Row key embeddings (stored as blobs) vs query embedding → cosine scores:
        apple pie recipe vs food recipe            → cos ≈ 1.0  (aligned)
        dentist dental appt vs food recipe         → cos ≈ 0.0  (orthogonal)
        dentist appt high vs dentist appointment   → cos ≈ 0.0  (orthogonal)
    """
    DIM = 4

    _VEC_MAP = {
        'food recipe':              _norm([1.0, 0.0, 0.0, 0.0]),
        'apple pie recipe':         _norm([1.0, 0.0, 0.0, 0.0]),
        'dentist appointment':      _norm([1.0, 0.0, 0.0, 0.0]),
        'dentist dental appt':      _norm([0.0, 1.0, 0.0, 0.0]),
        'dentist appt high':        _norm([0.0, 1.0, 0.0, 0.0]),
    }

    def _fake_emb(text):
        return _VEC_MAP.get(text, _norm([0.5, 0.5, 0.0, 0.0]))

    vec_db = _VecSimDb(db_service, DIM)
    service = DataGraphService(vec_db)
    service._generate_embedding = _fake_emb
    # Suppress backfill embedding calls for rows not in our map
    service._backfill_missing_embeddings = lambda: None
    return service, vec_db, db_service


@pytest.mark.unit
class TestRecallCosScore:
    """cos_score is computed from semantic similarity, not retrieval_weight."""

    def _insert_with_key_vec(self, db_service, rowid, key_vec, *, kind='user_specific',
                             key, value, retrieval_weight=1.0):
        """Insert a data_graph row and seed its key embedding blob."""
        _insert_row(db_service, kind=kind, key=key, value=value,
                    retrieval_weight=retrieval_weight)
        with db_service.connection() as conn:
            # Retrieve the actual rowid just inserted (last_insert_rowid is
            # not reliable here since _insert_row commits separately; use key).
            cur = conn.execute(
                "SELECT id FROM data_graph WHERE key=? AND active=1 ORDER BY id DESC LIMIT 1",
                (key,)
            )
            row = cur.fetchone()
            db_id = row[0] if row else None
        if db_id is not None:
            with db_service.connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO data_graph_key_vec(rowid, embedding) VALUES (?, ?)",
                    (db_id, _pack(key_vec))
                )
        return db_id

    def test_high_cos_score_row_carries_cos_score_gte_07(self, svc_vec):
        """Row whose key embedding aligns with the query returns cos_score >= 0.7."""
        service, _, db_service = svc_vec

        # apple pie recipe key embedding ≈ query 'food recipe' (both [1,0,0,0] unit)
        self._insert_with_key_vec(
            db_service, None,
            _norm([1.0, 0.0, 0.0, 0.0]),
            key='apple pie recipe',
            value='mix flour butter sugar',
        )

        results = service.recall('food recipe')
        matching = [r for r in results if r['key'] == 'apple pie recipe']
        assert matching, "Expected 'apple pie recipe' row in recall results"
        assert matching[0]['cos_score'] >= 0.7, (
            f"cos_score should be ≥ 0.7 for a nearly-identical embedding, "
            f"got {matching[0]['cos_score']}"
        )

    def test_low_cos_score_row_carries_cos_score_lt_04(self, svc_vec):
        """Row found via FTS but whose key embedding is orthogonal to the query returns cos_score < 0.4.

        Row key: 'dentist dental appt'  → blob [0,1,0,0]
        Query:   'food recipe'          → embedding [1,0,0,0]

        FTS surfaces the row (the key text contains 'dentist' and 'dental' which
        were added to the FTS index).  The vec cursor computes real cosine between
        the orthogonal vectors → cos = 0.0.  cos_score = max(0.0, 0.0) = 0.0 < 0.4.
        """
        service, _, db_service = svc_vec

        # Row with orthogonal blob to the query direction [1,0,0,0].
        self._insert_with_key_vec(
            db_service, None,
            _norm([0.0, 1.0, 0.0, 0.0]),
            key='dentist dental appt',
            value='appointment next tuesday',
        )

        # Ensure FTS also finds this row when query='food recipe' by injecting
        # 'food' and 'recipe' into its search_queries FTS column, which is
        # included in the FTS5 schema.
        with db_service.connection() as conn:
            row = conn.execute(
                "SELECT id FROM data_graph WHERE key='dentist dental appt' AND active=1"
            ).fetchone()
            if row:
                conn.execute(
                    "INSERT OR REPLACE INTO data_graph_fts(rowid, key, value, kind, search_queries) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (row[0], 'dentist dental appt', 'appointment next tuesday',
                     'user_specific', 'food recipe')
                )

        # query embedding [1,0,0,0] vs stored blob [0,1,0,0] → cosine = 0.0
        results = service.recall('food recipe')
        matching = [r for r in results if r['key'] == 'dentist dental appt']
        assert matching, (
            "Expected 'dentist dental appt' row to surface via FTS (search_queries='food recipe')"
        )
        assert matching[0]['cos_score'] < 0.4, (
            f"cos_score should be < 0.4 for an orthogonal embedding, "
            f"got {matching[0]['cos_score']}"
        )

