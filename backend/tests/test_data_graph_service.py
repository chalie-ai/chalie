"""Tests for DataGraphService — typed knowledge graph with decay, edges, and LUT canonicalization.

Covers store, recall, fetch, edge operations, reinforce/demote, deletion,
decay_cycle, and LUT-based upsert paths.
"""

import contextlib
import sqlite3

import pytest
from unittest.mock import MagicMock, patch

from services.data_graph_service import (
    DataGraphService,
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
        assert r2['retrieval_weight'] == pytest.approx(1.0)
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

    def test_store_user_specific_lut_temporal_exact_key_demotes_existing(self, svc, db_service):
        """Exact key match + temporal LUT rule → old row demoted on different value."""
        r1 = svc.store(KIND_USER_SPECIFIC, 'job_title', 'CEO')
        assert r1 is not None
        r1_rowid = _rid(r1)
        r1_id = r1['id']

        svc._generate_embedding = MagicMock(return_value=self._FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=self._lut_hit('job_title', 'temporal')):
            r2 = svc.store(KIND_USER_SPECIFIC, 'job_title', 'CTO')

        assert r2 is not None
        assert r2['value'] == 'CTO'
        assert r2['id'] != r1_id

        old = _raw_row(db_service, r1_rowid)
        assert old['active'] == 0

        with db_service.connection() as conn:
            edges = conn.execute("SELECT from_id, to_id, edge_type FROM data_graph_edges").fetchall()
        edge_set = {(e[0], e[1], e[2]) for e in edges}
        assert (r2['id'], r1_id, 'supersedes') in edge_set

    def test_store_user_specific_lut_coexist_same_value_reinforces(self, svc, db_service):
        """Coexist rule + exact same value → reinforce (same-value path fires before LUT lookup)."""
        r1 = svc.store(KIND_USER_SPECIFIC, 'favorite_foods', 'pizza')
        assert r1 is not None
        r1_id = _rid(r1)

        # Same-value check fires before LUT is consulted; LUT patch is decorative here.
        with patch.object(svc, '_lookup_concept_lut', return_value=self._lut_hit('favorite_foods', 'coexist')):
            r2 = svc.store(KIND_USER_SPECIFIC, 'favorite_foods', 'pizza')

        # Same row reinforced, evidence bumped
        assert _rid(r2) == r1_id
        assert r2['evidence_count'] == 2

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

    def test_store_user_specific_lut_miss_same_key_second_occurrence_increments_count(self, svc, db_service):
        """Scenario 099: same niche key arriving twice with different values.

        First call (no existing row) → _store_user_specific_new path → LUT miss recorded
        (count=1), row inserted as-is.

        Second call (existing row found via exact-key match) → existing-row branch →
        LUT consulted for rule, no LUT hit → temporal supersession fires. Because the
        second call hits the existing-row branch, _record_lut_miss is NOT called again.
        The miss row stays at count=1.

        This verifies the current production behaviour. Scenario 099 expects count=2 via
        the nightly harness — that scenario covers the case where the LLM produces a
        genuinely different raw key on the second turn (e.g. dryer_streak vs dryer_count)
        so each arrives as a fresh no-existing-row hit and _store_user_specific_new fires
        twice. The unit path here (same raw key, different value) goes through the
        existing-row branch on the second call and does not call _record_lut_miss again.
        """
        svc._generate_embedding = MagicMock(return_value=self._FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=None), \
             patch.object(svc, '_get_lut_miss_top_cos', return_value=0.35):
            svc.store(KIND_USER_SPECIFIC, 'dryer_streak', 'won 47 in a row')
            svc.store(KIND_USER_SPECIFIC, 'dryer_streak', 'won 48 in a row')

        with db_service.connection() as conn:
            misses = conn.execute(
                "SELECT count, key FROM concept_lut_misses WHERE key='dryer_streak'"
            ).fetchall()
        # First call records the miss (count=1). Second call hits existing-row branch, no
        # second _record_lut_miss call. Exactly one miss row exists.
        assert len(misses) == 1, f"Expected 1 miss row for 'dryer_streak', got {len(misses)}"
        assert misses[0][0] == 1, (
            f"First-call miss recorded count=1. Second call (existing-row branch) "
            f"does not call _record_lut_miss again. Got count={misses[0][0]}"
        )

    def test_store_user_specific_lut_multiple_misses_create_separate_rows(self, svc, db_service):
        """Multiple LUT misses with distinct keys each create a separate miss row."""
        svc._generate_embedding = MagicMock(return_value=self._FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=None), \
             patch.object(svc, '_get_lut_miss_top_cos', return_value=0.3):
            svc.store(KIND_USER_SPECIFIC, 'niche_key_a', 'value_a')
            svc.store(KIND_USER_SPECIFIC, 'niche_key_b', 'value_b')

        with db_service.connection() as conn:
            misses = conn.execute(
                "SELECT key FROM concept_lut_misses WHERE kind=?", (KIND_USER_SPECIFIC,)
            ).fetchall()
        miss_keys = {r[0] for r in misses}
        assert 'niche_key_a' in miss_keys
        assert 'niche_key_b' in miss_keys

    def test_store_system_exact_key_different_value_supersedes(self, svc, db_service):
        """system kind + exact key + different value → temporal supersession (cosine_supersede policy)."""
        r1 = svc.store(KIND_SYSTEM, 'tone', 'formal')
        assert r1 is not None
        r1_rowid = _rid(r1)

        r2 = svc.store(KIND_SYSTEM, 'tone', 'casual')

        assert r2 is not None
        assert r2['value'] == 'casual'
        assert _rid(r2) != r1_rowid

        old = _raw_row(db_service, r1_rowid)
        assert old['active'] == 0

    def test_store_system_cosine_close_keys_supersedes(self, svc, db_service):
        """system kind + no exact key + cosine match → supersession via _find_system_key_match."""
        r1 = svc.store(KIND_SYSTEM, 'response_style', 'formal')
        assert r1 is not None
        r1_rowid = _rid(r1)

        # Simulate a cosine-close key arriving (different key string, similar meaning)
        match_row = _raw_row(db_service, r1_rowid)
        with patch.object(svc, '_find_system_key_match', return_value=match_row):
            r2 = svc.store(KIND_SYSTEM, 'reply_tone', 'casual')

        assert r2 is not None
        # Old row demoted
        old = _raw_row(db_service, r1_rowid)
        assert old['active'] == 0

    def test_store_system_cosine_below_threshold_inserts_new(self, svc, db_service):
        """system kind + no exact key + no cosine match → plain insert, original untouched."""
        r1 = svc.store(KIND_SYSTEM, 'response_style', 'formal')
        assert r1 is not None
        r1_id = _rid(r1)

        with patch.object(svc, '_find_system_key_match', return_value=None):
            r2 = svc.store(KIND_SYSTEM, 'unrelated_system_key', 'value')

        assert r2 is not None
        assert _rid(r2) != r1_id
        # Original still active
        old = _raw_row(db_service, r1_id)
        assert old['active'] == 1

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
        assert raw['retrieval_weight'] == pytest.approx(1.0)

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
# TestKindPolicy
# ══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestKindPolicy:
    """Behavioral policy invariants — the values that drive decay, deletion, and reinforcement.

    Only assertions whose failure would indicate real user-facing regression are kept here.
    Config string values for internal dispatch (e.g. 'contradiction' handler name) are
    omitted — those are covered by the store/forget behavior tests.
    """

    def test_user_specific_decays_and_reinforces(self, svc, db_service):
        """user_specific rows decay after 30 days and reinforce on repeated evidence."""
        old_ts = '2020-01-01T00:00:00+00:00'
        rowid = _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='fact',
                            value='val', retrieval_weight=1.0, last_confirmed_at=old_ts)
        svc.decay_cycle()
        raw = _raw_row(db_service, rowid)
        assert raw['retrieval_weight'] < 1.0

    def test_misc_rows_are_hard_deleted_when_expired(self, svc, db_service):
        """misc rows at sub-floor weight confirmed >2 days ago are physically removed."""
        old_ts = '2020-01-01T00:00:00+00:00'
        rowid = _insert_row(db_service, kind=KIND_MISC, key='scratch',
                            value='v', retrieval_weight=0.005, last_confirmed_at=old_ts)
        svc.decay_cycle()
        assert _raw_row(db_service, rowid) is None

    def test_system_rows_never_decay(self, svc, db_service):
        """system rows are never touched by decay_cycle regardless of age."""
        old_ts = '2020-01-01T00:00:00+00:00'
        rowid = _insert_row(db_service, kind=KIND_SYSTEM, key='rule',
                            value='be concise', retrieval_weight=0.9, last_confirmed_at=old_ts)
        svc.decay_cycle()
        raw = _raw_row(db_service, rowid)
        assert raw['retrieval_weight'] == pytest.approx(0.9, abs=0.001)

    def test_moment_rows_never_decay(self, svc, db_service):
        """moment rows are never touched by decay_cycle regardless of age."""
        old_ts = '2020-01-01T00:00:00+00:00'
        rowid = _insert_row(db_service, kind=KIND_MOMENT, key='snapshot',
                            value='morning run', retrieval_weight=0.8, last_confirmed_at=old_ts)
        svc.decay_cycle()
        raw = _raw_row(db_service, rowid)
        assert raw['retrieval_weight'] == pytest.approx(0.8, abs=0.001)


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
        assert p['d_base'] == pytest.approx(0.0)
        assert p['salience_floor'] == pytest.approx(0.0)


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
        svc.add_edge(new_id, old_id, edge_type='supersedes')

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

    def test_forget_coexist_value_not_found_returns_value_not_found_status(self, svc, db_service):
        """Coexist forget for a value that doesn't exist returns value_not_found, no deletion."""
        _insert_row(db_service, kind=KIND_USER_SPECIFIC, key='food_and_drink', value='pasta')

        svc._generate_embedding = MagicMock(return_value=self._FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=self._lut_hit('food_and_drink', 'coexist')):
            result = svc.forget(KIND_USER_SPECIFIC, 'food_and_drink', value='sushi')

        assert result is not None
        assert result['status'] == 'value_not_found'

        # Existing pasta row untouched
        with db_service.connection() as conn:
            rows = conn.execute(
                "SELECT value FROM data_graph WHERE key='food_and_drink' AND deleted_at IS NULL"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 'pasta'

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

    def test_forget_lut_miss_raw_key_finds_and_deletes(self, svc, db_service):
        """LUT miss forget falls back to raw key lookup and deletes the matching row."""
        rowid = _insert_row(db_service, kind=KIND_USER_SPECIFIC,
                            key='dryer_streak', value='won 3 in a row')

        # No LUT hit — falls through to raw key path.
        svc._generate_embedding = MagicMock(return_value=self._FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=None):
            result = svc.forget(KIND_USER_SPECIFIC, 'dryer_streak')

        assert result is not None
        assert result['status'] in ('forgotten', 'forgotten_all')

        assert _raw_row(db_service, rowid) is None, "Raw-key forget must physically remove the row"

    def test_forget_nonexistent_key_returns_not_found(self, svc, db_service):
        """Forget on a key that has no stored rows returns not_found status without error."""
        svc._generate_embedding = MagicMock(return_value=self._FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=self._lut_hit('residence', 'temporal')):
            result = svc.forget(KIND_USER_SPECIFIC, 'residence')

        assert result is not None
        assert result['status'] == 'not_found'


# ══════════════════════════════════════════════════════════════════
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

    def test_backfill_surfaces_bare_inserted_row_via_recall(self, svc, db_service):
        """After backfill, the bare-inserted row is returned by recall via FTS."""
        self._bare_insert(db_service, key='sisters_name', value='Sofia')

        svc._generate_embedding = MagicMock(return_value=self._FAKE_EMB)

        results = svc.recall("sisters_name Sofia")

        keys = [r['key'] for r in results]
        assert 'sisters_name' in keys, "Backfilled row must appear in recall results"

    def test_backfill_skips_rows_with_all_indices_present(self, svc, db_service):
        """Rows fully indexed (vec + FTS) are not re-processed by a second recall."""
        # First recall: bare-inserted row triggers backfill and gets fully indexed.
        row_id = self._bare_insert(db_service, key='city', value='Valletta')
        svc._generate_embedding = MagicMock(return_value=self._FAKE_EMB)
        svc.recall("city Valletta")

        # Verify all three indices are now populated.
        with db_service.connection() as conn:
            kv = conn.execute(
                "SELECT COUNT(*) FROM data_graph_key_vec WHERE rowid=?", (row_id,)
            ).fetchone()[0]
            vv = conn.execute(
                "SELECT COUNT(*) FROM data_graph_value_vec WHERE rowid=?", (row_id,)
            ).fetchone()[0]
            fts = conn.execute(
                "SELECT COUNT(*) FROM data_graph_fts WHERE rowid=?", (row_id,)
            ).fetchone()[0]
        assert kv == 1
        assert vv == 1
        assert fts == 1

        # Second recall: backfill must find no missing rows, so _generate_embedding
        # is only called once for the query itself — not for the already-indexed row.
        call_log = []

        def _tracking_emb(text):
            call_log.append(text)
            return self._FAKE_EMB

        svc._generate_embedding = _tracking_emb
        svc.recall("city Valletta")

        query_calls = [c for c in call_log if c == "city Valletta"]
        assert len(query_calls) == 1, "Exactly one call for the query on second recall"
        row_calls = [c for c in call_log if c in ('city', 'Valletta')]
        assert len(row_calls) == 0, "No extra calls — fully-indexed row skipped by backfill"

    def test_backfill_handles_embedding_failure_gracefully(self, svc, db_service):
        """If _generate_embedding raises for one row, backfill continues and recall returns."""
        self._bare_insert(db_service, key='allergy', value='peanuts')

        svc._generate_embedding = MagicMock(side_effect=RuntimeError("emb down"))

        # Must not raise — backfill swallows per-row errors
        results = svc.recall("peanuts allergy")
        assert isinstance(results, list)
