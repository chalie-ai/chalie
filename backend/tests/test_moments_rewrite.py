"""
Tests for moments API — data_graph-backed pin/list/search/forget.

Coverage:
  POST /moments              — happy path, 404 missing, 400 user role, idempotency
  GET  /moments              — list active, exclude deleted
  POST /moments/<id>/forget  — soft-delete
  GET  /moments/search       — semantic search
"""

import contextlib
import sqlite3
import pytest
from unittest.mock import MagicMock, patch
from flask import Flask

from api.moments import moments_bp
from services.data_graph_service import DataGraphService
from services.database_service import DatabaseService
from services.time_utils import utc_now

pytestmark = pytest.mark.unit


# ── DB fixtures ──────────────────────────────────────────────────────────────
#
# Minimal schema: data_graph (standalone FTS, no vec tables) + transcript.
# Standalone FTS5 (no content= clause) avoids the sqlite-vec dependency.

_DATA_GRAPH_DDL = [
    """
    CREATE TABLE IF NOT EXISTS data_graph (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        kind              TEXT NOT NULL CHECK (kind IN (
                              'user_specific', 'system', 'misc', 'moment'
                          )),
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
    "CREATE INDEX IF NOT EXISTS idx_data_graph_kind      ON data_graph(kind)",
    "CREATE INDEX IF NOT EXISTS idx_data_graph_key       ON data_graph(key)",
    "CREATE INDEX IF NOT EXISTS idx_data_graph_active    ON data_graph(kind, active) WHERE deleted_at IS NULL",
    # Standalone FTS5 — no content= so DataGraphService._sync_fts INSERT works directly.
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS data_graph_fts USING fts5(
        key, value, kind, search_queries,
        tokenize='porter unicode61'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS data_graph_edges (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        from_id          INTEGER NOT NULL,
        to_id            INTEGER NOT NULL,
        edge_type        TEXT NOT NULL DEFAULT 'related',
        strength         REAL NOT NULL DEFAULT 1.0,
        created_at       TEXT NOT NULL DEFAULT (datetime('now')),
        last_accessed_at TEXT,
        UNIQUE (from_id, to_id, edge_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS transcript (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        channel     TEXT    NOT NULL,
        role        TEXT    NOT NULL,
        content     TEXT    NOT NULL,
        internal    INTEGER DEFAULT 0,
        created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,
]


@pytest.fixture
def db_service(tmp_path):
    """
    Isolated SQLite with data_graph + transcript schema.

    Uses a single shared connection so nested ``with db.connection()``
    calls inside DataGraphService all share the same connection and can
    see each other's uncommitted rows.
    """
    db_path = str(tmp_path / "moments_test.db")
    shared_conn = sqlite3.connect(db_path)
    shared_conn.row_factory = sqlite3.Row
    for ddl in _DATA_GRAPH_DDL:
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
def dgs(db_service):
    """DataGraphService backed by the isolated SQLite. Embeddings disabled."""
    svc = DataGraphService(db_service)
    svc._generate_embedding = MagicMock(return_value=None)
    return svc


def _insert_transcript(db_service, role, content='Test content', channel='user'):
    """Seed one transcript row and return its id."""
    with db_service.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
            (channel, role, content),
        )
        row_id = cursor.lastrowid
        cursor.close()
    return row_id


def _raw_data_graph(db_service, key):
    """Direct lookup in data_graph by key (includes deleted rows)."""
    with db_service.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM data_graph WHERE key = ?",
            (key,),
        )
        row = cursor.fetchone()
        cursor.close()
    return dict(row) if row else None


# ── Flask client fixture ─────────────────────────────────────────────────────

@pytest.fixture
def client(db_service, dgs):
    """
    Flask test client with moments_bp + auth bypassed.

    Patches ``get_shared_db_service`` and ``get_data_graph_service`` so
    the blueprint uses our isolated db_service and DataGraphService.
    """
    app = Flask(__name__)
    app.register_blueprint(moments_bp)
    app.config['TESTING'] = True

    with patch('services.auth_session_service.validate_session', return_value=True), \
         patch('services.database_service.get_shared_db_service', return_value=db_service), \
         patch('services.data_graph_service.get_data_graph_service', return_value=dgs):
        with app.test_client() as c:
            yield c


# ── POST /moments — pin a moment ─────────────────────────────────────────────

class TestPostMoments:

    def test_happy_path_stores_moment_in_data_graph(self, client, db_service):
        """POST /moments with a valid assistant transcript_id stores kind='moment' in data_graph."""
        tid = _insert_transcript(db_service, role='assistant', content='Great insight here')

        resp = client.post(
            '/moments',
            json={'transcript_id': tid},
            content_type='application/json',
        )

        assert resp.status_code == 201
        body = resp.get_json()
        assert 'item' in body
        assert body['item']['key'] == f'moment_{tid}'
        assert body['item']['value'] == 'Great insight here'

        # Verify row persisted in data_graph table
        row = _raw_data_graph(db_service, f'moment_{tid}')
        assert row is not None
        assert row['kind'] == 'moment'
        assert row['source'] == 'pin'

    def test_404_when_transcript_id_not_found(self, client):
        """POST /moments with a nonexistent transcript_id returns 404."""
        resp = client.post(
            '/moments',
            json={'transcript_id': 99999},
            content_type='application/json',
        )

        assert resp.status_code == 404
        assert 'error' in resp.get_json()

    def test_400_when_transcript_row_is_user_role(self, client, db_service):
        """POST /moments rejects pinning a user-role turn — only assistant turns are valid."""
        tid = _insert_transcript(db_service, role='user', content='User message')

        resp = client.post(
            '/moments',
            json={'transcript_id': tid},
            content_type='application/json',
        )

        assert resp.status_code == 400
        body = resp.get_json()
        assert 'assistant' in body['error'].lower()

    def test_idempotent_pin_same_id_twice_no_error(self, client, db_service):
        """POST /moments with the same transcript_id twice returns 201 then 200 (upsert)."""
        tid = _insert_transcript(db_service, role='assistant', content='Reusable insight')

        resp1 = client.post('/moments', json={'transcript_id': tid}, content_type='application/json')
        resp2 = client.post('/moments', json={'transcript_id': tid}, content_type='application/json')

        # First pin creates the row → 201. Re-pin is a no-op upsert → 200.
        assert resp1.status_code == 201
        assert resp2.status_code == 200

        # Exactly one active data_graph row (upsert, no duplicate)
        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM data_graph WHERE kind='moment' AND key=? AND deleted_at IS NULL",
                (f'moment_{tid}',),
            )
            count = cursor.fetchone()[0]
            cursor.close()
        assert count == 1

    def test_400_when_content_type_is_not_json(self, client):
        """POST /moments without application/json Content-Type returns 400."""
        resp = client.post('/moments', data='transcript_id=1')
        assert resp.status_code == 400


# ── GET /moments — list active moments ───────────────────────────────────────

class TestGetMoments:

    def test_returns_all_active_moment_rows(self, client, db_service, dgs):
        """GET /moments returns all non-deleted kind='moment' rows."""
        t1 = _insert_transcript(db_service, role='assistant', content='First insight')
        t2 = _insert_transcript(db_service, role='assistant', content='Second insight')

        for tid in (t1, t2):
            dgs.store(kind='moment', key=f'moment_{tid}', value=f'Content {tid}', source='pin')

        resp = client.get('/moments')

        assert resp.status_code == 200
        body = resp.get_json()
        assert 'items' in body
        assert len(body['items']) == 2

    def test_excludes_deleted_rows(self, client, db_service, dgs):
        """GET /moments excludes rows with deleted_at set."""
        t1 = _insert_transcript(db_service, role='assistant', content='Kept')
        t2 = _insert_transcript(db_service, role='assistant', content='Deleted')

        dgs.store(kind='moment', key=f'moment_{t1}', value='Kept', source='pin')
        dgs.store(kind='moment', key=f'moment_{t2}', value='Deleted', source='pin')

        # Soft-delete the second one directly
        with db_service.connection() as conn:
            conn.execute(
                "UPDATE data_graph SET deleted_at = ? WHERE key = ?",
                (utc_now().isoformat(), f'moment_{t2}'),
            )

        resp = client.get('/moments')

        assert resp.status_code == 200
        items = resp.get_json()['items']
        assert len(items) == 1
        assert items[0]['key'] == f'moment_{t1}'

    def test_returns_empty_list_when_no_moments(self, client):
        """GET /moments returns empty list when data_graph has no moment rows."""
        resp = client.get('/moments')

        assert resp.status_code == 200
        assert resp.get_json()['items'] == []


# ── POST /moments/<id>/forget — soft-delete ─────────────────────────────────

class TestForgetMoment:

    def _seed_moment(self, db_service, dgs, content='Insightful response'):
        """Insert a transcript row and pin it as a moment. Returns transcript_id."""
        tid = _insert_transcript(db_service, role='assistant', content=content)
        dgs.store(kind='moment', key=f'moment_{tid}', value=content, source='pin')
        return tid

    def test_soft_deletes_moment_row(self, client, db_service, dgs):
        """POST /moments/<id>/forget marks the moment row as deleted."""
        tid = self._seed_moment(db_service, dgs)

        resp = client.post(f'/moments/{tid}/forget')

        assert resp.status_code == 200
        assert resp.get_json()['ok'] is True

        row = _raw_data_graph(db_service, f'moment_{tid}')
        assert row is not None
        assert row['deleted_at'] is not None

    def test_forget_nonexistent_moment_returns_404(self, client):
        """POST /moments/<id>/forget on an unknown id returns 404."""
        resp = client.post('/moments/99999/forget')

        assert resp.status_code == 404
        assert 'error' in resp.get_json()

    def test_forgotten_moment_absent_from_list(self, client, db_service, dgs):
        """After forget, moment does not appear in GET /moments."""
        tid = self._seed_moment(db_service, dgs)

        client.post(f'/moments/{tid}/forget')

        resp = client.get('/moments')
        items = resp.get_json()['items']
        assert all(item['key'] != f'moment_{tid}' for item in items)

    def test_second_forget_of_same_moment_returns_404(self, client, db_service, dgs):
        """Forgetting an already-deleted moment returns 404 (not found in active rows)."""
        tid = self._seed_moment(db_service, dgs)

        client.post(f'/moments/{tid}/forget')
        resp2 = client.post(f'/moments/{tid}/forget')

        assert resp2.status_code == 404


# ── GET /moments/search ───────────────────────────────────────────────────────

class TestSearchMoments:

    def _seed_moment(self, db_service, dgs, content, transcript_id=None):
        """Seed one moment row and return (tid, moment_key)."""
        if transcript_id is None:
            transcript_id = _insert_transcript(db_service, role='assistant', content=content)
        moment_key = f'moment_{transcript_id}'
        dgs.store(kind='moment', key=moment_key, value=content, source='pin')
        return transcript_id, moment_key

    def test_missing_query_parameter_returns_400(self, client):
        """GET /moments/search without ?q returns 400."""
        resp = client.get('/moments/search')
        assert resp.status_code == 400

    def test_empty_query_returns_400(self, client):
        """GET /moments/search with empty ?q returns 400."""
        resp = client.get('/moments/search?q=')
        assert resp.status_code == 400

    def test_search_with_no_moments_returns_empty(self, client):
        """GET /moments/search with a valid query returns empty list when no moments exist."""
        resp = client.get('/moments/search?q=cooking')
        assert resp.status_code == 200
        assert resp.get_json()['items'] == []

    def test_search_returns_200_with_moments_present(self, client, db_service, dgs):
        """GET /moments/search returns 200 (non-error) when moments exist, even without vec support."""
        self._seed_moment(db_service, dgs, 'Memorable response about cooking')

        resp = client.get('/moments/search?q=cooking')
        # Result may be empty (no embeddings in test) but must not error
        assert resp.status_code == 200
        assert 'items' in resp.get_json()
