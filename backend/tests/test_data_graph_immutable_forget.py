"""Unit tests for the immutable-rule forget guard in DataGraphService.

Scenario 085 residual: the model could bypass the immutable rule by calling
forget() to hard-delete the active row, then store() with a new value on the
now-empty key.  This test suite verifies that _forget_immutable rejects the
operation and that the subsequent store() still hits the conflict path.
"""

import contextlib
import sqlite3

import pytest
from unittest.mock import MagicMock, patch

from services.data_graph_service import (
    DataGraphService,
    KIND_USER_SPECIFIC,
)
from services.database_service import DatabaseService


pytestmark = pytest.mark.unit


# ── Schema ────────────────────────────────────────────────────────

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
    "CREATE INDEX IF NOT EXISTS idx_data_graph_active ON data_graph(kind, active) WHERE deleted_at IS NULL",
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


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def db_service(tmp_path):
    db_path = str(tmp_path / "test_immutable_forget.db")

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
    service = DataGraphService(db_service)
    service._generate_embedding = MagicMock(return_value=None)
    return service


# ── Helpers ───────────────────────────────────────────────────────

_FAKE_EMB = [0.1] * 768


def _lut_hit(canonical_key: str, rule: str):
    return {'canonical_key': canonical_key, 'rule': rule, 'cos': 0.95}


def _active_row(db_service, key: str):
    """Return the active data_graph row for key, or None."""
    with db_service.connection() as conn:
        row = conn.execute(
            "SELECT * FROM data_graph WHERE key=? AND active=1 AND deleted_at IS NULL LIMIT 1",
            (key,),
        ).fetchone()
    return dict(row) if row else None


# ── Tests ─────────────────────────────────────────────────────────

class TestImmutableForgetGuard:
    """_forget_immutable must block deletion of an active immutable row."""

    def test_forget_immutable_returns_immutable_blocked(self, svc, db_service):
        """forget() on an immutable key returns immutable_blocked, not forgotten."""
        svc._generate_embedding = MagicMock(return_value=_FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=_lut_hit('birth_date', 'immutable')):
            svc.store(KIND_USER_SPECIFIC, 'birth_date', 'March 15')

        with patch.object(svc, '_lookup_concept_lut', return_value=_lut_hit('birth_date', 'immutable')):
            result = svc.forget(KIND_USER_SPECIFIC, 'birth_date')

        assert result is not None
        assert result['status'] == 'immutable_blocked'

    def test_forget_immutable_row_stays_active(self, svc, db_service):
        """The active row must still exist after a blocked forget."""
        svc._generate_embedding = MagicMock(return_value=_FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=_lut_hit('birth_date', 'immutable')):
            svc.store(KIND_USER_SPECIFIC, 'birth_date', 'March 15')

        with patch.object(svc, '_lookup_concept_lut', return_value=_lut_hit('birth_date', 'immutable')):
            svc.forget(KIND_USER_SPECIFIC, 'birth_date')

        row = _active_row(db_service, 'birth_date')
        assert row is not None, "Active row must not be deleted"
        assert row['active'] == 1
        assert row['value'] == 'March 15'

    def test_forget_immutable_result_carries_old_value(self, svc, db_service):
        """immutable_blocked result exposes old_value so the formatter can use it."""
        svc._generate_embedding = MagicMock(return_value=_FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=_lut_hit('birth_date', 'immutable')):
            svc.store(KIND_USER_SPECIFIC, 'birth_date', 'March 15')

        with patch.object(svc, '_lookup_concept_lut', return_value=_lut_hit('birth_date', 'immutable')):
            result = svc.forget(KIND_USER_SPECIFIC, 'birth_date')

        assert result.get('old_value') == 'March 15'

    def test_store_after_blocked_forget_still_conflicts(self, svc, db_service):
        """After a blocked forget, a store() with a new value must still return conflict."""
        svc._generate_embedding = MagicMock(return_value=_FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=_lut_hit('birth_date', 'immutable')):
            svc.store(KIND_USER_SPECIFIC, 'birth_date', 'March 15')

        # Attempt forget (blocked)
        with patch.object(svc, '_lookup_concept_lut', return_value=_lut_hit('birth_date', 'immutable')):
            forget_result = svc.forget(KIND_USER_SPECIFIC, 'birth_date')
        assert forget_result['status'] == 'immutable_blocked'

        # Attempt store with new value — must still conflict
        with patch.object(svc, '_lookup_concept_lut', return_value=_lut_hit('birth_date', 'immutable')):
            store_result = svc.store(KIND_USER_SPECIFIC, 'birth_date', 'March 20')

        assert store_result is not None
        assert store_result.get('status') == 'conflict'

        # Original row still active with original value
        row = _active_row(db_service, 'birth_date')
        assert row is not None
        assert row['value'] == 'March 15'

    def test_forget_immutable_not_found_returns_not_found(self, svc, db_service):
        """forget() on a non-existent immutable key returns not_found, not immutable_blocked."""
        svc._generate_embedding = MagicMock(return_value=_FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=_lut_hit('birth_date', 'immutable')):
            result = svc.forget(KIND_USER_SPECIFIC, 'birth_date')

        assert result is not None
        assert result['status'] == 'not_found'

    def test_forget_immutable_logs_conflict(self, svc, db_service):
        """_log_immutable_conflict must be called when a forget is blocked."""
        svc._generate_embedding = MagicMock(return_value=_FAKE_EMB)
        with patch.object(svc, '_lookup_concept_lut', return_value=_lut_hit('birth_date', 'immutable')):
            svc.store(KIND_USER_SPECIFIC, 'birth_date', 'March 15')

        with patch.object(svc, '_log_immutable_conflict') as mock_log, \
             patch.object(svc, '_lookup_concept_lut', return_value=_lut_hit('birth_date', 'immutable')):
            svc.forget(KIND_USER_SPECIFIC, 'birth_date')

        mock_log.assert_called_once_with('birth_date')
