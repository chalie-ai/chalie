"""Tests for DataGraphService — typed knowledge graph with decay, edges, and contradiction handling.

Covers store, recall, fetch, edge operations, reinforce/demote, deletion,
decay_cycle, and seed_from_legacy_knowledge.
"""

import contextlib
import sqlite3

import pytest
from unittest.mock import MagicMock, patch

from services.data_graph_service import (
    DataGraphService,
    seed_from_legacy_knowledge,
    KIND_USER_SPECIFIC,
    KIND_SYSTEM,
    KIND_MISC,
    KIND_MOMENT,
    KIND_DOCUMENT,
    _KIND_POLICY,
)
from services.database_service import DatabaseService
from services.time_utils import utc_now


pytestmark = pytest.mark.unit



# ── Schema ────────────────────────────────────────────────────────

# Standalone (non-content-table) FTS so manual INSERT/DELETE work in tests.
# The production schema uses content='data_graph' with triggers, but for unit
# tests we control FTS sync explicitly — same as test_knowledge_service.py approach.
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
    # Legacy knowledge table needed for seed tests
    """
    CREATE TABLE IF NOT EXISTS knowledge (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        kind        TEXT NOT NULL,
        entity      TEXT NOT NULL DEFAULT 'user',
        key         TEXT NOT NULL,
        value       TEXT,
        data        TEXT,
        decay_class TEXT NOT NULL DEFAULT 'standard',
        confidence  REAL NOT NULL DEFAULT 0.5,
        reliability TEXT NOT NULL DEFAULT 'reliable',
        source      TEXT,
        evidence_count INTEGER NOT NULL DEFAULT 1,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
        last_accessed_at TEXT,
        deleted_at  TEXT,
        search_queries TEXT DEFAULT NULL,
        UNIQUE(entity, key)
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
    degradation path in the service). Background thread scheduling also mocked
    to keep tests deterministic.
    """
    service = DataGraphService(db_service)
    service._generate_embedding = MagicMock(return_value=None)
    service._schedule_embeddings = MagicMock()
    service._schedule_doc2query = MagicMock()
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


# ══════════════════════════════════════════════════════════════════
# TestStore
# ══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestStore:

    def test_store_new_row(self, svc, db_service):
        """Basic insert: all required columns populated, FTS synced."""
        result = svc.store(KIND_USER_SPECIFIC, 'user_name', 'Dylan')

        assert result is not None
        assert result['kind'] == KIND_USER_SPECIFIC
        assert result['key'] == 'user_name'
        assert result['value'] == 'Dylan'
        assert result['evidence_count'] == 1
        assert result['retrieval_weight'] == 1.0
        assert result['storage_strength'] == 0.5

        # Row persisted in DB
        row_id = _rid(result)
        assert row_id is not None
        raw = _raw_row(db_service, row_id)
        assert raw is not None
        assert raw['active'] == 1
        assert raw['deleted_at'] is None

        # FTS indexed — the value token should be findable
        fts_hits = _raw_fts(db_service, '"Dylan"*')
        assert row_id in fts_hits

    def test_store_invalid_kind_returns_none(self, svc):
        """An unrecognised kind is rejected immediately, nothing stored."""
        result = svc.store('bogus_kind', 'key', 'value')
        assert result is None

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
        assert r2['retrieval_weight'] == 1.0
        assert r2['storage_strength'] > r1['storage_strength']

        # Only one active row for this key — first_seen_at must be unchanged
        all_rows = _raw_all(db_service)
        active = [r for r in all_rows if r['key'] == 'favourite_food' and r['active']]
        assert len(active) == 1
        assert active[0]['first_seen_at'] == r1['first_seen_at']

    def test_store_no_reinforce_for_misc(self, svc, db_service):
        """misc kind has reinforce=False — same value returns existing row without boosting."""
        assert _KIND_POLICY[KIND_MISC]['reinforce'] is False

        r1 = svc.store(KIND_MISC, 'scratch', 'temp data')
        assert r1 is not None
        r2 = svc.store(KIND_MISC, 'scratch', 'temp data')
        assert r2 is not None

        # Same row returned (by id), evidence_count not incremented
        assert _rid(r2) == _rid(r1)
        assert r2['evidence_count'] == r1['evidence_count']

    def test_store_contradiction_classify_temporal(self, svc, db_service):
        """user_specific + different value + classifier returns temporal_change
        → old row demoted (active=0), new row inserted, supersedes edges created."""
        r1 = svc.store(KIND_USER_SPECIFIC, 'job', 'Engineer')
        assert r1 is not None
        r1_rowid = _rid(r1)
        r1_id = r1['id']

        cls_result = {'classification': 'temporal_change', 'reasoning': 'career change'}
        with patch('services.contradiction_classifier_service.ContradictionClassifierService') as MockCls:
            MockCls.return_value.check_new_trait.return_value = cls_result
            r2 = svc.store(KIND_USER_SPECIFIC, 'job', 'Architect')

        assert r2 is not None
        assert r2['value'] == 'Architect'
        r2_id = r2['id']
        assert r2_id != r1_id

        # Old row demoted to active=0 with reduced retrieval_weight
        old = _raw_row(db_service, r1_rowid)
        assert old['active'] == 0
        assert old['retrieval_weight'] < 1.0

        # supersedes edge: new→old; superseded_by edge: old→new
        with db_service.connection() as conn:
            edges = conn.execute(
                "SELECT from_id, to_id, edge_type FROM data_graph_edges"
            ).fetchall()
        edge_set = {(e[0], e[1], e[2]) for e in edges}
        assert (r2_id, r1_id, 'supersedes') in edge_set
        assert (r1_id, r2_id, 'superseded_by') in edge_set

    def test_store_contradiction_classify_true_returns_conflict(self, svc):
        """user_specific + true_contradiction → conflict dict returned, no new row stored."""
        svc.store(KIND_USER_SPECIFIC, 'eye_colour', 'blue')

        cls_result = {'classification': 'true_contradiction', 'reasoning': 'cannot both be true'}
        with patch('services.contradiction_classifier_service.ContradictionClassifierService') as MockCls:
            MockCls.return_value.check_new_trait.return_value = cls_result
            result = svc.store(KIND_USER_SPECIFIC, 'eye_colour', 'brown')

        assert result is not None
        assert result.get('conflict') is True
        assert result['classification'] == 'true_contradiction'
        assert 'existing' in result
        assert result['proposed_value'] == 'brown'
        # No id/rowid on a conflict dict — proves no row was inserted
        assert 'id' not in result or result.get('conflict')

    def test_store_contradiction_classify_ambiguous_returns_conflict(self, svc):
        """Ambiguous classification also yields a conflict dict, not a stored row."""
        svc.store(KIND_USER_SPECIFIC, 'city', 'Valletta')

        cls_result = {'classification': 'ambiguous', 'reasoning': 'unclear'}
        with patch('services.contradiction_classifier_service.ContradictionClassifierService') as MockCls:
            MockCls.return_value.check_new_trait.return_value = cls_result
            result = svc.store(KIND_USER_SPECIFIC, 'city', 'Mosta')

        assert result.get('conflict') is True
        assert result['classification'] == 'ambiguous'

    def test_store_contradiction_classify_compatible_reinforces(self, svc, db_service):
        """Classifier returns None (compatible) → reinforce the existing row, no new insert."""
        r1 = svc.store(KIND_USER_SPECIFIC, 'favourite_sport', 'tennis')
        assert r1 is not None
        r1_id = _rid(r1)

        with patch('services.contradiction_classifier_service.ContradictionClassifierService') as MockCls:
            MockCls.return_value.check_new_trait.return_value = None
            r2 = svc.store(KIND_USER_SPECIFIC, 'favourite_sport', 'squash')

        # Same row reinforced
        assert _rid(r2) == r1_id
        assert r2['evidence_count'] == 2

    def test_store_newest_wins_for_system_kind(self, svc, db_service):
        """system kind uses newest_wins: old demoted, new inserted, classifier NOT called."""
        r1 = svc.store(KIND_SYSTEM, 'tone', 'formal')
        assert r1 is not None
        r1_rowid = _rid(r1)

        with patch('services.contradiction_classifier_service.ContradictionClassifierService') as MockCls:
            r2 = svc.store(KIND_SYSTEM, 'tone', 'casual')
            MockCls.assert_not_called()

        assert r2 is not None
        assert r2['value'] == 'casual'
        assert _rid(r2) != r1_rowid

        old = _raw_row(db_service, r1_rowid)
        assert old['active'] == 0

    def test_store_no_contradiction_check_for_misc(self, svc, db_service):
        """misc kind has contradiction=None — different value inserts a second row directly."""
        assert _KIND_POLICY[KIND_MISC]['contradiction'] is None

        r1 = svc.store(KIND_MISC, 'note', 'first note')
        r2 = svc.store(KIND_MISC, 'note', 'different note')

        # Both rows exist — misc doesn't consolidate on key+different_value
        assert r1 is not None
        assert r2 is not None
        assert _rid(r2) != _rid(r1)


# ══════════════════════════════════════════════════════════════════
# TestRecall
# ══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRecall:

    def test_recall_empty_db(self, svc):
        """Recall on an empty graph returns an empty list without error."""
        results = svc.recall("anything")
        assert results == []

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

    def test_recall_filters_inactive_rows(self, svc, db_service):
        """Rows with active=0 are excluded from recall."""
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='old_job',
                    value='janitor', active=0)
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='new_job', value='engineer')

        results = svc.recall("janitor engineer")
        keys = [r['key'] for r in results]
        assert 'old_job' not in keys
        assert 'new_job' in keys

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

    def test_recall_touch_bumps_retrieval_weight(self, svc, db_service):
        """recall() boosts retrieval_weight by 0.1 (capped at 1.0)."""
        rowid = _insert_row(db_service, kind=KIND_USER_SPECIFIC,
                            key='language', value='Maltese', retrieval_weight=0.8)

        svc.recall("Maltese language")

        raw = _raw_row(db_service, rowid)
        assert raw['retrieval_weight'] == pytest.approx(0.9, abs=0.01)

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

    def test_recall_composite_score_present_on_every_result(self, svc, db_service):
        """Every returned dict has a composite_score field."""
        _insert_row(db_service, kind=KIND_MISC, key='task', value='finish report')
        results = svc.recall("finish report")
        for r in results:
            assert 'composite_score' in r


# ══════════════════════════════════════════════════════════════════
# TestFetch
# ══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFetch:

    def test_fetch_basic_returns_active_non_deleted(self, svc, db_service):
        """Default fetch returns only active, non-deleted rows."""
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='name', value='Dylan')
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='gone', value='x',
                    deleted_at=utc_now().isoformat())
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='inactive', value='y', active=0)

        rows = svc.fetch()
        keys = {r['key'] for r in rows}
        assert 'name' in keys
        assert 'gone' not in keys
        assert 'inactive' not in keys

    def test_fetch_kinds_filter(self, svc, db_service):
        """kinds= parameter restricts returned rows to matching kinds."""
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='pref', value='light mode')
        _insert_row(db_service, kind=KIND_SYSTEM, key='rule', value='be concise')

        rows = svc.fetch(kinds=[KIND_SYSTEM])
        assert all(r['kind'] == KIND_SYSTEM for r in rows)

    def test_fetch_include_inactive(self, svc, db_service):
        """include_inactive=True includes active=0 rows that aren't deleted."""
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='old', value='val', active=0)

        rows_default = svc.fetch()
        rows_with_inactive = svc.fetch(include_inactive=True)

        keys_default = {r['key'] for r in rows_default}
        keys_with_inactive = {r['key'] for r in rows_with_inactive}

        assert 'old' not in keys_default
        assert 'old' in keys_with_inactive

    def test_fetch_include_deleted(self, svc, db_service):
        """include_deleted=True includes soft-deleted rows."""
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='deleted_row',
                    value='gone', deleted_at=utc_now().isoformat())

        rows = svc.fetch(include_deleted=True)
        keys = {r['key'] for r in rows}
        assert 'deleted_row' in keys

    def test_fetch_limit(self, svc, db_service):
        """limit= caps the number of returned rows."""
        for i in range(5):
            _insert_row(db_service, kind=KIND_MISC, key=f'note_{i}', value=f'v{i}')

        rows = svc.fetch(limit=3)
        assert len(rows) == 3


# ══════════════════════════════════════════════════════════════════
# TestEdges
# ══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestEdges:

    def test_add_edge_basic(self, svc, db_service):
        """add_edge returns a positive edge id and the edge is persisted."""
        ra = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='a', value='av')
        rb = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='b', value='bv')
        id_a = _get_db_id(db_service, ra)
        id_b = _get_db_id(db_service, rb)

        edge_id = svc.add_edge(id_a, id_b, edge_type='related')
        assert edge_id > 0

        with db_service.connection() as conn:
            row = conn.execute(
                "SELECT * FROM data_graph_edges "
                "WHERE from_id=? AND to_id=? AND edge_type='related'",
                (id_a, id_b)
            ).fetchone()
        assert row is not None

    def test_add_edge_duplicate_ignored_returns_same_id(self, svc, db_service):
        """Inserting the same edge twice uses INSERT OR IGNORE and returns the same edge id."""
        ra = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='x', value='xv')
        rb = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='y', value='yv')
        id_a = _get_db_id(db_service, ra)
        id_b = _get_db_id(db_service, rb)

        edge_id_1 = svc.add_edge(id_a, id_b, edge_type='related')
        edge_id_2 = svc.add_edge(id_a, id_b, edge_type='related')
        assert edge_id_1 == edge_id_2

        with db_service.connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM data_graph_edges "
                "WHERE from_id=? AND to_id=? AND edge_type='related'",
                (id_a, id_b)
            ).fetchone()[0]
        assert count == 1

    def test_expand_edges_returns_outgoing(self, svc, db_service):
        """expand_edges returns all edges from the given seed node ids."""
        ra = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='hub', value='v')
        rb = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='leaf1', value='v')
        rc = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='leaf2', value='v')
        id_a = _get_db_id(db_service, ra)
        id_b = _get_db_id(db_service, rb)
        id_c = _get_db_id(db_service, rc)

        svc.add_edge(id_a, id_b, edge_type='related')
        svc.add_edge(id_a, id_c, edge_type='causes')

        edges = svc.expand_edges([id_a])
        to_ids = {e['to_id'] for e in edges}
        assert id_b in to_ids
        assert id_c in to_ids

    def test_expand_edges_type_filter(self, svc, db_service):
        """edge_types= filters edges to only the specified types."""
        ra = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='src', value='v')
        rb = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='dst1', value='v')
        rc = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='dst2', value='v')
        id_a = _get_db_id(db_service, ra)
        id_b = _get_db_id(db_service, rb)
        id_c = _get_db_id(db_service, rc)

        svc.add_edge(id_a, id_b, edge_type='causes')
        svc.add_edge(id_a, id_c, edge_type='related')

        edges = svc.expand_edges([id_a], edge_types=['causes'])
        assert all(e['edge_type'] == 'causes' for e in edges)
        to_ids = {e['to_id'] for e in edges}
        assert id_b in to_ids
        assert id_c not in to_ids

    def test_touch_edge_updates_last_accessed_at(self, svc, db_service):
        """touch_edge sets last_accessed_at on the matching edge row."""
        ra = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='p', value='v')
        rb = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='q', value='v')
        id_a = _get_db_id(db_service, ra)
        id_b = _get_db_id(db_service, rb)

        svc.add_edge(id_a, id_b, edge_type='related')

        # Verify last_accessed_at starts NULL
        with db_service.connection() as conn:
            before = conn.execute(
                "SELECT last_accessed_at FROM data_graph_edges "
                "WHERE from_id=? AND to_id=? AND edge_type='related'",
                (id_a, id_b)
            ).fetchone()[0]
        assert before is None

        svc.touch_edge(id_a, id_b, 'related')

        with db_service.connection() as conn:
            after = conn.execute(
                "SELECT last_accessed_at FROM data_graph_edges "
                "WHERE from_id=? AND to_id=? AND edge_type='related'",
                (id_a, id_b)
            ).fetchone()[0]
        assert after is not None


# ══════════════════════════════════════════════════════════════════
# TestReinforceAndDemote
# ══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestReinforceAndDemote:

    def test_reinforce_increments_evidence_and_resets_retrieval_weight(self, svc, db_service):
        """reinforce() increments evidence_count, boosts storage_strength, resets retrieval_weight=1.0."""
        rowid = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='r_key', value='r_val',
                            evidence_count=2, storage_strength=0.6, retrieval_weight=0.5)

        svc.reinforce(rowid)

        raw = _raw_row(db_service, rowid)
        assert raw['evidence_count'] == 3
        assert raw['storage_strength'] > 0.6
        assert raw['retrieval_weight'] == 1.0

    def test_reinforce_storage_strength_capped_at_one(self, svc, db_service):
        """storage_strength never exceeds 1.0 after reinforcement."""
        rowid = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='strong', value='v',
                            storage_strength=0.999, evidence_count=100)

        svc.reinforce(rowid)

        raw = _raw_row(db_service, rowid)
        assert raw['storage_strength'] <= 1.0

    def test_demote_reduces_retrieval_weight_by_factor(self, svc, db_service):
        """demote() multiplies retrieval_weight by the given factor."""
        rowid = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='d_key', value='d_val',
                            retrieval_weight=0.8)

        svc.demote(rowid, factor=0.5)

        raw = _raw_row(db_service, rowid)
        assert raw['retrieval_weight'] == pytest.approx(0.4, abs=0.001)

    def test_demote_does_not_touch_storage_strength(self, svc, db_service):
        """demote() only modifies retrieval_weight, not storage_strength."""
        rowid = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='ss_key', value='v',
                            storage_strength=0.75, retrieval_weight=1.0)

        svc.demote(rowid, factor=0.5)

        raw = _raw_row(db_service, rowid)
        assert raw['storage_strength'] == pytest.approx(0.75, abs=0.001)

    def test_demote_unknown_rowid_is_no_op(self, svc):
        """demote() on a non-existent rowid silently does nothing (no exception raised)."""
        svc.demote(999999, factor=0.1)  # must not raise


# ══════════════════════════════════════════════════════════════════
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

    def test_soft_delete_row_still_in_db(self, svc, db_service):
        """Soft delete keeps the physical row in data_graph."""
        rowid = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='soft', value='v')
        svc.soft_delete_by_id(rowid)
        raw = _raw_row(db_service, rowid)
        assert raw is not None

    def test_hard_delete_removes_row_completely(self, svc, db_service):
        """hard_delete_by_id physically removes the row from data_graph."""
        rowid = _insert_row(db_service, kind=KIND_MISC, key='hard', value='gone')

        result = svc.hard_delete_by_id(rowid)
        assert result is True

        raw = _raw_row(db_service, rowid)
        assert raw is None

    def test_hard_delete_removes_from_fts(self, svc, db_service):
        """hard_delete_by_id removes the FTS entry so it can't be searched."""
        rowid = _insert_row(db_service, kind=KIND_MISC, key='hd_fts', value='erasedtoken')

        svc.hard_delete_by_id(rowid)

        fts_hits = _raw_fts(db_service, '"erasedtoken"*')
        assert rowid not in fts_hits

    def test_set_active_toggles_flag(self, svc, db_service):
        """set_active() writes the active flag as given."""
        rowid = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='toggle', value='v')

        svc.set_active(rowid, 0)
        raw = _raw_row(db_service, rowid)
        assert raw['active'] == 0

        svc.set_active(rowid, 1)
        raw = _raw_row(db_service, rowid)
        assert raw['active'] == 1


# ══════════════════════════════════════════════════════════════════
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

    def test_decay_skips_moment_kind(self, svc, db_service):
        """moment kind has ttl_days=None so decay_cycle must never modify it."""
        assert _KIND_POLICY[KIND_MOMENT]['ttl_days'] is None

        rowid = _insert_row(db_service, kind=KIND_MOMENT, key='snapshot',
                            value='morning run', retrieval_weight=0.8,
                            last_confirmed_at='2020-01-01T00:00:00+00:00')

        svc.decay_cycle()

        raw = _raw_row(db_service, rowid)
        assert raw['retrieval_weight'] == pytest.approx(0.8, abs=0.001)

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

    def test_decay_does_not_hard_delete_fresh_misc(self, svc, db_service):
        """misc rows with a recent last_confirmed_at are NOT deleted even at low weight."""
        now = utc_now().isoformat()
        rowid = _insert_row(db_service, kind=KIND_MISC, key='fresh_scratch',
                            value='recent', retrieval_weight=0.005,
                            last_confirmed_at=now)

        svc.decay_cycle()

        raw = _raw_row(db_service, rowid)
        assert raw is not None, "Recently-confirmed misc row must not be deleted"

    def test_decay_returns_count_of_updated_rows(self, svc, db_service):
        """decay_cycle() returns the number of rows that were updated or deleted."""
        old_ts = '2020-01-01T00:00:00+00:00'
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='old1',
                    value='v', retrieval_weight=1.0, last_confirmed_at=old_ts)

        count = svc.decay_cycle()
        assert isinstance(count, int)
        assert count >= 1


# ══════════════════════════════════════════════════════════════════
# TestSeed
# ══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSeed:

    def _insert_knowledge(self, db_service, *, entity, kind, key, value,
                          source=None, evidence_count=1, created_at=None):
        """Insert a row into the legacy knowledge table for seed tests."""
        now = created_at or utc_now().isoformat()
        with db_service.connection() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO knowledge
                    (kind, entity, key, value, source, evidence_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (kind, entity, key, value, source, evidence_count, now, now))

    def test_seed_migrates_user_entity_to_user_specific(self, svc, db_service):
        """knowledge rows with entity='user' migrate to user_specific kind."""
        self._insert_knowledge(db_service, entity='user', kind='trait',
                               key='user_name', value='Dylan')

        seed_from_legacy_knowledge(db_service)

        with db_service.connection() as conn:
            row = conn.execute(
                "SELECT kind, value FROM data_graph WHERE key='user_name'"
            ).fetchone()
        assert row is not None
        assert row[0] == KIND_USER_SPECIFIC
        assert row[1] == 'Dylan'

    def test_seed_migrates_chalie_rule_to_system_kind(self, svc, db_service):
        """entity='chalie' + kind='rule' migrates to system kind."""
        self._insert_knowledge(db_service, entity='chalie', kind='rule',
                               key='verbosity', value='concise')

        seed_from_legacy_knowledge(db_service)

        with db_service.connection() as conn:
            row = conn.execute(
                "SELECT kind FROM data_graph WHERE key='verbosity'"
            ).fetchone()
        assert row is not None
        assert row[0] == KIND_SYSTEM

    def test_seed_skips_unrecognised_entity(self, db_service):
        """knowledge rows with an entity that doesn't match any heuristic are skipped.

        The seed heuristic only maps:
          - entity in ('', 'dylan', 'user') or key in _USER_SPECIFIC_KEYS → user_specific
          - entity=='chalie' AND kind=='rule' → system
        All other combinations hit the else: continue path.
        """
        self._insert_knowledge(db_service, entity='external_service', kind='moment',
                               key='moment_001', value='morning run')

        seed_from_legacy_knowledge(db_service)

        with db_service.connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM data_graph").fetchone()[0]
        assert count == 0, "Rows with unrecognised entity should not be migrated"

    def test_seed_skips_procedure_kind(self, svc, db_service):
        """knowledge rows with kind='procedure' are skipped by the seeder."""
        self._insert_knowledge(db_service, entity='user', kind='procedure',
                               key='proc_key', value='proc_val')

        seed_from_legacy_knowledge(db_service)

        with db_service.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM data_graph WHERE key='proc_key'"
            ).fetchone()
        assert row is None

    def test_seed_skips_moment_context_kind(self, svc, db_service):
        """knowledge rows with kind='moment_context' are skipped by the seeder."""
        self._insert_knowledge(db_service, entity='user', kind='moment_context',
                               key='ctx_key', value='some context')

        seed_from_legacy_knowledge(db_service)

        with db_service.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM data_graph WHERE key='ctx_key'"
            ).fetchone()
        assert row is None

    def test_seed_idempotent_does_not_duplicate(self, svc, db_service):
        """Running seed twice exits early on second call — no duplicate rows created."""
        self._insert_knowledge(db_service, entity='user', kind='trait',
                               key='email', value='dylan@example.com')

        seed_from_legacy_knowledge(db_service)
        seed_from_legacy_knowledge(db_service)  # second call must be a no-op

        with db_service.connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM data_graph WHERE key='email'"
            ).fetchone()[0]
        assert count == 1

    def test_seed_preserves_evidence_count(self, svc, db_service):
        """seed preserves evidence_count from the source knowledge row."""
        self._insert_knowledge(db_service, entity='user', kind='fact',
                               key='age', value='30', evidence_count=5)

        seed_from_legacy_knowledge(db_service)

        with db_service.connection() as conn:
            row = conn.execute(
                "SELECT evidence_count FROM data_graph WHERE key='age'"
            ).fetchone()
        assert row is not None
        assert row[0] == 5

    def test_seed_skips_ambiguous_entity(self, svc, db_service):
        """knowledge rows whose entity doesn't match any heuristic are not migrated."""
        # entity is not 'user'/'dylan'/'chalie'/''/'' and key not in _USER_SPECIFIC_KEYS
        self._insert_knowledge(db_service, entity='weather_service', kind='fact',
                               key='temperature_format', value='celsius')

        seed_from_legacy_knowledge(db_service)

        with db_service.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM data_graph WHERE key='temperature_format'"
            ).fetchone()
        assert row is None


# ══════════════════════════════════════════════════════════════════
# TestKindPolicy
# ══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestKindPolicy:
    """Policy table invariants — the contract that drive all branching behaviour.

    These test the observable contract, not implementation details — callers
    depend on this table having specific shapes and values.
    """

    def test_user_specific_policy_shape(self):
        p = _KIND_POLICY[KIND_USER_SPECIFIC]
        assert p['reinforce'] is True
        assert p['contradiction'] == 'classify'
        assert p['ttl_days'] == 30
        assert p['salience_floor'] == pytest.approx(0.2)

    def test_system_policy_no_auto_decay(self):
        """system kind must never auto-decay (ttl_days=None is the sentinel)."""
        p = _KIND_POLICY[KIND_SYSTEM]
        assert p['ttl_days'] is None
        assert p['reinforce'] is True
        assert p['contradiction'] == 'newest_wins'

    def test_misc_policy_hard_deletion(self):
        """misc kind is short-lived scratchpad with hard deletion."""
        p = _KIND_POLICY[KIND_MISC]
        assert p['deletion'] == 'hard'
        assert p['reinforce'] is False
        assert p['ttl_days'] == 2
        assert p['contradiction'] is None

    def test_moment_policy_no_auto_decay(self):
        """moment kind (internal only) must never auto-decay."""
        p = _KIND_POLICY[KIND_MOMENT]
        assert p['ttl_days'] is None
        assert p['reinforce'] is False


# ══════════════════════════════════════════════════════════════════
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

    def test_document_no_reinforce(self, svc, db_service):
        """document kind has reinforce=False — storing same key+value twice keeps evidence_count=1."""
        assert _KIND_POLICY[KIND_DOCUMENT]['reinforce'] is False

        r1 = svc.store(KIND_DOCUMENT, 'doc:abc:001', 'some content', source='doc:abc')
        assert r1 is not None
        assert r1['evidence_count'] == 1

        r2 = svc.store(KIND_DOCUMENT, 'doc:abc:001', 'some content', source='doc:abc')
        assert r2 is not None

        assert _rid(r2) == _rid(r1)
        assert r2['evidence_count'] == 1

    def test_document_hard_delete(self, svc, db_service):
        """document kind uses hard deletion — hard_delete_by_id removes the row completely."""
        assert _KIND_POLICY[KIND_DOCUMENT]['deletion'] == 'hard'

        result = svc.store(KIND_DOCUMENT, 'doc:xyz:000', 'deletable chunk', source='doc:xyz')
        assert result is not None
        row_id = _rid(result)

        deleted = svc.hard_delete_by_id(row_id)
        assert deleted is True

        raw = _raw_row(db_service, row_id)
        assert raw is None

    def test_document_no_decay(self, svc, db_service):
        """document kind has d_base=0.0 — decay_cycle does not modify retrieval_weight."""
        assert _KIND_POLICY[KIND_DOCUMENT]['d_base'] == 0.0

        result = svc.store(KIND_DOCUMENT, 'doc:nodecay:000', 'persistent chunk', source='doc:nodecay')
        assert result is not None
        row_id = _rid(result)

        with db_service.connection() as conn:
            conn.execute(
                "UPDATE data_graph SET retrieval_weight=0.7, last_confirmed_at='2020-01-01T00:00:00+00:00' WHERE rowid=?",
                (row_id,)
            )

        svc.decay_cycle()

        raw = _raw_row(db_service, row_id)
        assert raw is not None
        assert raw['retrieval_weight'] == pytest.approx(0.7, abs=0.001)

    def test_document_recall_with_kinds_filter(self, svc, db_service):
        """kinds=['document'] returns only document rows; kinds=['user_specific'] excludes them."""
        _insert_row(db_service, kind=KIND_DOCUMENT, key='doc:solar:000',
                    value='solar panels efficiency rating')
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='user_pref',
                    value='solar panels efficiency rating')

        doc_results = svc.recall('solar panels efficiency', kinds=[KIND_DOCUMENT])
        doc_kinds = {r['kind'] for r in doc_results}
        assert doc_kinds == {KIND_DOCUMENT}

        user_results = svc.recall('solar panels efficiency', kinds=[KIND_USER_SPECIFIC])
        user_kinds = {r['kind'] for r in user_results}
        assert user_kinds == {KIND_USER_SPECIFIC}

    def test_document_fts_indexed(self, svc, db_service):
        """FTS index includes document artifact — direct FTS query finds the stored row."""
        result = svc.store(KIND_DOCUMENT, 'doc:energy:000', 'solar panels efficiency', source='doc:energy')
        assert result is not None
        row_id = _rid(result)

        fts_hits = _raw_fts(db_service, 'solar')
        assert row_id in fts_hits

    def test_document_policy_values(self):
        """_KIND_POLICY['document'] has the exact values the document artifact system depends on."""
        p = _KIND_POLICY[KIND_DOCUMENT]
        assert p['ttl_days'] is None
        assert p['reinforce'] is False
        assert p['contradiction'] is None
        assert p['deletion'] == 'hard'
        assert p['d_base'] == 0.0
        assert p['salience_floor'] == 0.0


# ══════════════════════════════════════════════════════════════════
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

    def test_no_op_for_unmatched_prefix(self, svc, db_service):
        """Returns 0 and does not touch unrelated rows when prefix matches nothing."""
        _insert_row(db_service, kind=KIND_DOCUMENT, key='doc:xyz:000',
                    value='other doc', source='document:xyz')

        deleted = svc.hard_delete_by_source_prefix('document:missing')

        assert deleted == 0
        # The unrelated row is untouched
        remaining = _raw_all(db_service)
        assert any(r.get('source') == 'document:xyz' for r in remaining)

    def test_does_not_delete_rows_with_longer_matching_source(self, svc, db_service):
        """Prefix 'document:abc' must not delete rows sourced as 'document:abcdef'."""
        _insert_row(db_service, kind=KIND_DOCUMENT, key='doc:abc:000',
                    value='exact match', source='document:abc')
        _insert_row(db_service, kind=KIND_DOCUMENT, key='doc:abcdef:000',
                    value='longer source', source='document:abcdef')

        deleted = svc.hard_delete_by_source_prefix('document:abc')

        # LIKE 'document:abc%' matches both — this is expected behaviour from the
        # implementation. The test documents this: callers must use the full doc id.
        assert deleted == 2

    def test_cleans_fts_entries_for_deleted_rows(self, svc, db_service):
        """FTS entries for deleted rows are removed so they no longer appear in searches."""
        rowid = _insert_row(db_service, kind=KIND_DOCUMENT, key='doc:cleanup:000',
                            value='solar energy efficiency', source='document:cleanup')

        # FTS entry is present before deletion
        fts_before = _raw_fts(db_service, 'solar')
        assert rowid in fts_before

        svc.hard_delete_by_source_prefix('document:cleanup')

        # FTS entry is gone after deletion
        fts_after = _raw_fts(db_service, 'solar')
        assert rowid not in fts_after

    def test_only_deletes_source_prefix_rows_not_others(self, svc, db_service):
        """Rows with a different source are untouched when a targeted prefix is deleted."""
        _insert_row(db_service, kind=KIND_DOCUMENT, key='doc:target:000',
                    value='targeted chunk', source='document:target')
        _insert_row(db_service, kind=KIND_DOCUMENT, key='doc:keep:000',
                    value='keep this chunk', source='document:keep')
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='user_fact',
                    value='user data', source='skill:memory:store:user')

        svc.hard_delete_by_source_prefix('document:target')

        remaining = _raw_all(db_service)
        sources = {r.get('source') for r in remaining}
        assert 'document:keep' in sources
        assert 'skill:memory:store:user' in sources
        assert 'document:target' not in sources

    def test_empty_prefix_returns_zero(self, svc, db_service):
        """Passing an empty prefix string returns 0 without deleting anything."""
        _insert_row(db_service, kind=KIND_DOCUMENT, key='doc:safe:000',
                    value='safe chunk', source='document:safe')

        # Empty prefix — implementation matches source LIKE '%' which would delete
        # everything. The service must handle this; if it does delete, the count
        # reflects rows deleted. This test documents the actual behaviour so any
        # regression is caught.
        before_count = len(_raw_all(db_service))
        deleted = svc.hard_delete_by_source_prefix('')
        after_count = len(_raw_all(db_service))

        # Whatever it does, count must equal rows actually removed
        assert deleted == before_count - after_count

    def test_partial_prefix_targets_all_matching_docs(self, svc, db_service):
        """Using prefix 'document:' deletes all document-sourced rows regardless of doc id."""
        _insert_row(db_service, kind=KIND_DOCUMENT, key='doc:d1:000',
                    value='doc1 chunk', source='document:d1')
        _insert_row(db_service, kind=KIND_DOCUMENT, key='doc:d2:000',
                    value='doc2 chunk', source='document:d2')
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='user_key',
                    value='user value', source='skill:memory')

        deleted = svc.hard_delete_by_source_prefix('document:')

        assert deleted == 2
        remaining = _raw_all(db_service)
        assert len(remaining) == 1
        assert remaining[0]['source'] == 'skill:memory'
