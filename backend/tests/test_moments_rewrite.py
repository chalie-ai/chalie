"""
Tests for the moments rewrite — knowledge-backed pin/list/search/forget.

Coverage:
  POST /moments              — happy path, 404 missing, 400 user role, idempotency
  GET  /moments              — list active, exclude deleted
  POST /moments/<id>/forget  — soft-delete parent + context children (one transaction)
  GET  /moments/search       — moment hit, context rollup, deleted-parent exclusion
  moment_context_service     — _is_enriched, _fetch_context_turns, _enrich_moment, worker skip rules
"""

import contextlib
import json
import sqlite3
import pytest
from datetime import timedelta
from unittest.mock import MagicMock
from flask import Flask

from api.moments import moments_bp
from services.knowledge_service import KnowledgeService
from services.database_service import DatabaseService
from services.time_utils import utc_now

pytestmark = pytest.mark.unit


# ── DB fixtures ──────────────────────────────────────────────────────────────
#
# We build a minimal schema that includes:
#   - knowledge table + FTS5 (identical DDL to production)
#   - transcript table (append-only, no deleted_at column — matches
#     production schema.sql exactly)
#
# This is intentionally NOT using the conftest `db` fixture so we can precisely
# control the schema without running the full SchemaConvergenceService and
# sqlite-vec extension (not available in all environments).

_KNOWLEDGE_DDL = [
    """
    CREATE TABLE IF NOT EXISTS knowledge (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        kind            TEXT    NOT NULL,
        entity          TEXT    NOT NULL DEFAULT 'user',
        key             TEXT    NOT NULL,
        value           TEXT,
        data            TEXT,
        decay_class     TEXT    NOT NULL DEFAULT 'standard',
        confidence      REAL    NOT NULL DEFAULT 0.5,
        reliability     TEXT    NOT NULL DEFAULT 'reliable',
        source          TEXT,
        evidence_count  INTEGER NOT NULL DEFAULT 1,
        created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
        updated_at      TEXT    NOT NULL DEFAULT (datetime('now')),
        last_accessed_at TEXT,
        deleted_at      TEXT,
        search_queries  TEXT    DEFAULT NULL,
        UNIQUE(entity, key)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_knowledge_kind   ON knowledge(kind)",
    "CREATE INDEX IF NOT EXISTS idx_knowledge_entity ON knowledge(entity)",
    "CREATE INDEX IF NOT EXISTS idx_knowledge_key    ON knowledge(key)",
    "CREATE INDEX IF NOT EXISTS idx_knowledge_deleted ON knowledge(deleted_at) WHERE deleted_at IS NULL",
    # Standalone (non-content) FTS5 — matches how KnowledgeService syncs manually.
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
        key, value, kind, entity, search_queries,
        tokenize='porter unicode61'
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
    Isolated SQLite with knowledge + transcript schema.

    Uses a single shared connection (thread-local style) so nested
    ``with db.connection()`` calls inside KnowledgeService all share
    the same connection and can see each other's uncommitted rows.
    """
    db_path = str(tmp_path / "moments_test.db")
    shared_conn = sqlite3.connect(db_path)
    shared_conn.row_factory = sqlite3.Row
    for ddl in _KNOWLEDGE_DDL:
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
def ks(db_service):
    """KnowledgeService backed by the in-memory SQLite.  Embeddings disabled."""
    svc = KnowledgeService(db_service)
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


def _raw_knowledge(db_service, entity, key):
    """Direct lookup bypassing the service (includes deleted rows)."""
    with db_service.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM knowledge WHERE entity = ? AND key = ?",
            (entity, key),
        )
        row = cursor.fetchone()
        cursor.close()
    return dict(row) if row else None


def _count_knowledge_by_entity(db_service, entity, deleted_ok=False):
    """Count knowledge rows for an entity, optionally including deleted ones."""
    with db_service.connection() as conn:
        cursor = conn.cursor()
        if deleted_ok:
            cursor.execute("SELECT COUNT(*) FROM knowledge WHERE entity = ?", (entity,))
        else:
            cursor.execute(
                "SELECT COUNT(*) FROM knowledge WHERE entity = ? AND deleted_at IS NULL",
                (entity,),
            )
        count = cursor.fetchone()[0]
        cursor.close()
    return count


# ── Flask client fixture ─────────────────────────────────────────────────────

@pytest.fixture
def client(db_service):
    """
    Flask test client with moments_bp + auth bypassed.

    Patches ``get_shared_db_service`` so the blueprint's ``_get_db_and_ks()``
    helper returns our isolated ``db_service``.
    """
    from unittest.mock import patch

    app = Flask(__name__)
    app.register_blueprint(moments_bp)
    app.config['TESTING'] = True

    with patch('services.auth_session_service.validate_session', return_value=True), \
         patch('services.database_service.get_shared_db_service', return_value=db_service):
        with app.test_client() as c:
            yield c


# ── POST /moments — pin a moment ─────────────────────────────────────────────

class TestPostMoments:

    def test_happy_path_stores_moment_in_knowledge(self, client, db_service):
        """POST /moments with a valid assistant transcript_id stores kind='moment' in knowledge."""
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

        # Verify row persisted in knowledge table
        row = _raw_knowledge(db_service, 'user', f'moment_{tid}')
        assert row is not None
        assert row['kind'] == 'moment'
        assert row['decay_class'] == 'permanent'
        assert row['source'] == 'user_pin'
        data = json.loads(row['data'])
        assert data['transcript_id'] == tid

    def test_404_when_transcript_id_not_found(self, client, db_service):
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

        # Exactly one knowledge row (upsert, no duplicate)
        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM knowledge WHERE entity='user' AND key=? AND deleted_at IS NULL",
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

    def test_returns_all_active_moment_rows(self, client, db_service, ks):
        """GET /moments returns all non-deleted kind='moment' rows ordered by created_at DESC."""
        # Insert two assistant turns and pin them
        t1 = _insert_transcript(db_service, role='assistant', content='First insight')
        t2 = _insert_transcript(db_service, role='assistant', content='Second insight')

        for tid in (t1, t2):
            ks.store(
                kind='moment', entity='user', key=f'moment_{tid}',
                value=f'Content {tid}',
                data={'transcript_id': tid, 'channel': 'user', 'created_at': utc_now().isoformat()},
                decay_class='permanent', confidence=1.0, source='user_pin',
            )

        resp = client.get('/moments')

        assert resp.status_code == 200
        body = resp.get_json()
        assert 'items' in body
        assert len(body['items']) == 2

    def test_excludes_deleted_rows(self, client, db_service, ks):
        """GET /moments excludes rows with deleted_at set."""
        t1 = _insert_transcript(db_service, role='assistant', content='Kept')
        t2 = _insert_transcript(db_service, role='assistant', content='Deleted')

        ks.store(kind='moment', entity='user', key=f'moment_{t1}',
                 value='Kept', data={'transcript_id': t1, 'channel': 'user', 'created_at': utc_now().isoformat()},
                 decay_class='permanent', confidence=1.0, source='user_pin')
        ks.store(kind='moment', entity='user', key=f'moment_{t2}',
                 value='Deleted', data={'transcript_id': t2, 'channel': 'user', 'created_at': utc_now().isoformat()},
                 decay_class='permanent', confidence=1.0, source='user_pin')

        # Soft-delete the second one directly
        with db_service.connection() as conn:
            conn.execute(
                "UPDATE knowledge SET deleted_at = ? WHERE key = ?",
                (utc_now().isoformat(), f'moment_{t2}'),
            )

        resp = client.get('/moments')

        assert resp.status_code == 200
        items = resp.get_json()['items']
        assert len(items) == 1
        assert items[0]['key'] == f'moment_{t1}'

    def test_returns_empty_list_when_no_moments(self, client):
        """GET /moments returns empty list when knowledge table has no moment rows."""
        resp = client.get('/moments')

        assert resp.status_code == 200
        assert resp.get_json()['items'] == []


# ── POST /moments/<id>/forget — cascade soft-delete ─────────────────────────

class TestForgetMoment:

    def _seed_moment_with_context(self, db_service, ks):
        """Insert a moment + 2 context children, return the transcript_id."""
        tid = _insert_transcript(db_service, role='assistant', content='Insightful response')
        moment_key = f'moment_{tid}'

        ks.store(kind='moment', entity='user', key=moment_key,
                 value='Insightful response',
                 data={'transcript_id': tid, 'channel': 'user', 'created_at': utc_now().isoformat()},
                 decay_class='permanent', confidence=1.0, source='user_pin')

        for n in range(2):
            ks.store(kind='moment_context', entity=moment_key,
                     key=f'{moment_key}_ctx_{n}',
                     value=f'[user] context turn {n}',
                     data={'parent_key': moment_key, 'transcript_id': tid, 'role': 'user', 'ctx_index': n},
                     decay_class='permanent', confidence=1.0, source='moment_enrichment')

        return tid

    def test_soft_deletes_parent_row(self, client, db_service, ks):
        """POST /moments/<id>/forget marks the parent moment row as deleted."""
        tid = self._seed_moment_with_context(db_service, ks)

        resp = client.post(f'/moments/{tid}/forget')

        assert resp.status_code == 200
        assert resp.get_json()['ok'] is True

        parent = _raw_knowledge(db_service, 'user', f'moment_{tid}')
        assert parent is not None
        assert parent['deleted_at'] is not None

    def test_soft_deletes_all_context_rows(self, client, db_service, ks):
        """POST /moments/<id>/forget marks all moment_context children as deleted."""
        tid = self._seed_moment_with_context(db_service, ks)

        client.post(f'/moments/{tid}/forget')

        moment_key = f'moment_{tid}'
        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM knowledge WHERE entity = ? AND deleted_at IS NULL",
                (moment_key,),
            )
            surviving_children = cursor.fetchone()[0]
            cursor.close()

        assert surviving_children == 0

    def test_forget_is_transactional_parent_and_children_deleted_together(self, client, db_service, ks):
        """Both parent and children are deleted in one pass — no partial state."""
        tid = self._seed_moment_with_context(db_service, ks)

        client.post(f'/moments/{tid}/forget')

        moment_key = f'moment_{tid}'
        with db_service.connection() as conn:
            cursor = conn.cursor()
            # All rows for this moment should now have deleted_at set
            cursor.execute(
                "SELECT COUNT(*) FROM knowledge "
                "WHERE (entity = 'user' AND key = ?) OR entity = ?",
                (moment_key, moment_key),
            )
            total_rows = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM knowledge "
                "WHERE ((entity = 'user' AND key = ?) OR entity = ?) AND deleted_at IS NOT NULL",
                (moment_key, moment_key),
            )
            deleted_rows = cursor.fetchone()[0]
            cursor.close()

        # Every row belonging to this moment should be deleted
        assert total_rows > 0
        assert total_rows == deleted_rows

    def test_forget_nonexistent_moment_returns_404(self, client, db_service):
        """POST /moments/<id>/forget on an unknown id returns 404 — nothing was deleted."""
        resp = client.post('/moments/99999/forget')

        assert resp.status_code == 404
        assert 'error' in resp.get_json()

    def test_forget_clears_fts_shadow_table(self, client, db_service, ks):
        """Cascade forget must remove both parent + children from knowledge_fts."""
        tid = self._seed_moment_with_context(db_service, ks)
        moment_key = f'moment_{tid}'

        # Sanity: parent + 2 children present in FTS before forget
        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM knowledge_fts WHERE key LIKE ?", (f'{moment_key}%',))
            before = cursor.fetchone()[0]
            cursor.close()
        assert before >= 3  # parent + 2 context rows

        resp = client.post(f'/moments/{tid}/forget')
        assert resp.status_code == 200

        # FTS shadow table must be empty for this moment — no orphan index entries
        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM knowledge_fts WHERE key LIKE ?", (f'{moment_key}%',))
            after = cursor.fetchone()[0]
            cursor.close()
        assert after == 0


# ── GET /moments/search ───────────────────────────────────────────────────────

class TestSearchMoments:

    def _seed_moment(self, db_service, ks, content, transcript_id=None):
        """Seed one moment row and return (tid, moment_key)."""
        if transcript_id is None:
            transcript_id = _insert_transcript(db_service, role='assistant', content=content)
        moment_key = f'moment_{transcript_id}'
        ks.store(
            kind='moment', entity='user', key=moment_key,
            value=content,
            data={'transcript_id': transcript_id, 'channel': 'user', 'created_at': utc_now().isoformat()},
            decay_class='permanent', confidence=1.0, source='user_pin',
        )
        return transcript_id, moment_key

    def test_moment_hit_surfaces_as_result(self, client, db_service, ks):
        """GET /moments/search?q=<key> exact match on moment key returns the moment."""
        tid, moment_key = self._seed_moment(db_service, ks, 'Memorable response about cooking')

        # Use the exact key as query — guaranteed exact-key hit via Signal 1
        resp = client.get(f'/moments/search?q={moment_key}')

        assert resp.status_code == 200
        items = resp.get_json()['items']
        assert len(items) >= 1
        assert any(item['key'] == moment_key for item in items)

    def test_context_hit_rolled_up_to_parent_not_standalone(self, client, db_service, ks):
        """GET /moments/search returns the parent moment when a context row matches, not the context row itself."""
        tid, moment_key = self._seed_moment(db_service, ks, 'Parent moment content')

        # Add a context row
        ctx_key = f'{moment_key}_ctx_0'
        ks.store(
            kind='moment_context', entity=moment_key, key=ctx_key,
            value=f'[user] {ctx_key}',
            data={'parent_key': moment_key, 'transcript_id': tid, 'role': 'user', 'ctx_index': 0},
            decay_class='permanent', confidence=1.0, source='moment_enrichment',
        )

        # Query the context key directly — should surface the parent, not the ctx row
        resp = client.get(f'/moments/search?q={ctx_key}')

        assert resp.status_code == 200
        items = resp.get_json()['items']
        assert len(items) >= 1
        # Every returned item should be a parent moment key (no '_ctx_' suffix)
        for item in items:
            assert '_ctx_' not in item['key']

    def test_deleted_parent_excluded_even_if_context_row_hits(self, client, db_service, ks):
        """GET /moments/search does not surface deleted parents even when context rows match."""
        tid, moment_key = self._seed_moment(db_service, ks, 'To be deleted moment')

        ctx_key = f'{moment_key}_ctx_0'
        ks.store(
            kind='moment_context', entity=moment_key, key=ctx_key,
            value=f'[user] {ctx_key}',
            data={'parent_key': moment_key, 'transcript_id': tid, 'role': 'user', 'ctx_index': 0},
            decay_class='permanent', confidence=1.0, source='moment_enrichment',
        )

        # Soft-delete the parent moment
        with db_service.connection() as conn:
            conn.execute(
                "UPDATE knowledge SET deleted_at = ? WHERE entity = 'user' AND key = ?",
                (utc_now().isoformat(), moment_key),
            )

        resp = client.get(f'/moments/search?q={moment_key}')

        assert resp.status_code == 200
        items = resp.get_json()['items']
        # Deleted parent must not appear
        assert all(item['key'] != moment_key for item in items)

    def test_missing_query_parameter_returns_400(self, client):
        """GET /moments/search without ?q returns 400."""
        resp = client.get('/moments/search')
        assert resp.status_code == 400


# ── moment_context_service — unit tests for internal functions ────────────────

class TestMomentContextService:
    """
    Tests for the module-level functions in moment_context_service.

    These call the functions directly with a real db_service — no mocking.
    """

    def test_is_enriched_returns_false_when_no_context_rows_exist(self, db_service):
        """_is_enriched returns False when the knowledge table has no moment_context for this key."""
        from services.moment_context_service import _is_enriched

        result = _is_enriched(db_service, 'moment_99')

        assert result is False

    def test_is_enriched_returns_true_when_context_rows_exist(self, db_service, ks):
        """_is_enriched returns True after context rows are stored for this parent key."""
        from services.moment_context_service import _is_enriched

        parent_key = 'moment_42'
        ks.store(
            kind='moment_context', entity=parent_key, key=f'{parent_key}_ctx_0',
            value='[user] some turn',
            data={'parent_key': parent_key, 'transcript_id': 42, 'role': 'user', 'ctx_index': 0},
            decay_class='permanent', confidence=1.0, source='moment_enrichment',
        )

        result = _is_enriched(db_service, parent_key)

        assert result is True

    def test_is_enriched_ignores_deleted_context_rows(self, db_service, ks):
        """_is_enriched returns False when the only context row for this key is deleted."""
        from services.moment_context_service import _is_enriched

        parent_key = 'moment_55'
        ks.store(
            kind='moment_context', entity=parent_key, key=f'{parent_key}_ctx_0',
            value='[user] some turn',
            data={'parent_key': parent_key, 'transcript_id': 55, 'role': 'user', 'ctx_index': 0},
            decay_class='permanent', confidence=1.0, source='moment_enrichment',
        )

        # Soft-delete the context row
        with db_service.connection() as conn:
            conn.execute(
                "UPDATE knowledge SET deleted_at = ? WHERE entity = ? AND kind = 'moment_context'",
                (utc_now().isoformat(), parent_key),
            )

        result = _is_enriched(db_service, parent_key)

        assert result is False

    def test_fetch_context_turns_returns_at_most_10_rows(self, db_service):
        """_fetch_context_turns returns <=10 rows even when more transcript rows exist."""
        from services.moment_context_service import _fetch_context_turns

        # Seed 15 transcript rows in the user channel
        pinned_id = None
        for i in range(15):
            tid = _insert_transcript(db_service, role='assistant', content=f'Turn {i}', channel='user')
            if i == 7:
                pinned_id = tid  # pick middle row as the pinned id

        # Default limits are CTX_BEFORE=5 + CTX_AFTER=5 → at most 10 rows
        turns = _fetch_context_turns(db_service, pinned_id)

        assert len(turns) <= 10

    def test_fetch_context_turns_excludes_pinned_id_itself(self, db_service):
        """_fetch_context_turns never includes the pinned transcript row in its results."""
        from services.moment_context_service import _fetch_context_turns

        _t1 = _insert_transcript(db_service, role='user', content='Before', channel='user')
        pinned = _insert_transcript(db_service, role='assistant', content='Pinned', channel='user')
        _insert_transcript(db_service, role='user', content='After', channel='user')

        turns = _fetch_context_turns(db_service, pinned)

        returned_ids = [t['id'] for t in turns]
        assert pinned not in returned_ids

    def test_fetch_context_turns_only_returns_user_channel(self, db_service):
        """_fetch_context_turns excludes transcript rows from non-user channels."""
        from services.moment_context_service import _fetch_context_turns

        pinned = _insert_transcript(db_service, role='assistant', content='Pinned', channel='user')
        # Insert rows in a different channel — should be excluded
        _insert_transcript(db_service, role='user', content='DMN turn', channel='dmn')
        _insert_transcript(db_service, role='user', content='Scheduled turn', channel='scheduled')
        _insert_transcript(db_service, role='user', content='Real user turn', channel='user')

        turns = _fetch_context_turns(db_service, pinned)

        for turn in turns:
            # All returned rows should come from user channel (we have the id, not channel,
            # but we verify indirectly: only the 'user' channel row should appear, not dmn/scheduled)
            assert turn['content'] != 'DMN turn'
            assert turn['content'] != 'Scheduled turn'

    def test_enrich_moment_stores_context_rows_with_correct_fields(self, db_service, ks):
        """_enrich_moment stores kind='moment_context' rows with correct entity/key/data."""
        from services.moment_context_service import _enrich_moment

        # Seed surrounding transcript rows
        _insert_transcript(db_service, role='user', content='Before the moment', channel='user')
        pinned_id = _insert_transcript(db_service, role='assistant', content='The moment', channel='user')
        _insert_transcript(db_service, role='user', content='After the moment', channel='user')

        moment_key = f'moment_{pinned_id}'
        # Seed the parent moment row — the race guard in _enrich_moment checks
        # that the parent still exists before writing children, so the worker
        # can't silently orphan rows when a forget happens mid-enrichment.
        ks.store(
            kind='moment', entity='user', key=moment_key,
            value='The moment',
            data={'transcript_id': pinned_id, 'channel': 'user', 'created_at': utc_now().isoformat()},
            decay_class='permanent', confidence=1.0, source='user_pin',
        )
        data_json = json.dumps({'transcript_id': pinned_id, 'channel': 'user', 'created_at': utc_now().isoformat()})

        # moment_row tuple: (id, entity, key, value, data, created_at)
        moment_row = (1, 'user', moment_key, 'The moment', data_json, utc_now().isoformat())

        _enrich_moment(db_service, ks, moment_row)

        # Verify context rows were stored
        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT key, value, data FROM knowledge "
                "WHERE entity = ? AND kind = 'moment_context' AND deleted_at IS NULL",
                (moment_key,),
            )
            rows = cursor.fetchall()
            cursor.close()

        assert len(rows) >= 1  # at least the 2 surrounding turns
        for row in rows:
            key, value, data_raw = row['key'], row['value'], row['data']
            assert key.startswith(f'{moment_key}_ctx_')
            assert value.startswith('[')  # prefixed with [role]
            ctx_data = json.loads(data_raw)
            assert ctx_data['parent_key'] == moment_key

    def test_enrich_moment_skips_if_already_enriched(self, db_service, ks):
        """_enrich_moment is idempotent — does not add duplicate context rows."""
        from services.moment_context_service import _enrich_moment

        _insert_transcript(db_service, role='user', content='Before', channel='user')
        pinned_id = _insert_transcript(db_service, role='assistant', content='The moment', channel='user')
        _insert_transcript(db_service, role='user', content='After', channel='user')

        moment_key = f'moment_{pinned_id}'
        # Seed the parent moment so the race guard lets _enrich_moment write.
        ks.store(
            kind='moment', entity='user', key=moment_key,
            value='The moment',
            data={'transcript_id': pinned_id, 'channel': 'user', 'created_at': utc_now().isoformat()},
            decay_class='permanent', confidence=1.0, source='user_pin',
        )
        data_json = json.dumps({'transcript_id': pinned_id})
        moment_row = (1, 'user', moment_key, 'The moment', data_json, utc_now().isoformat())

        # Enrich once
        _enrich_moment(db_service, ks, moment_row)

        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM knowledge WHERE entity = ? AND kind = 'moment_context' AND deleted_at IS NULL",
                (moment_key,),
            )
            count_after_first = cursor.fetchone()[0]
            cursor.close()

        # Enrich again — should skip because _is_enriched returns True
        _enrich_moment(db_service, ks, moment_row)

        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM knowledge WHERE entity = ? AND kind = 'moment_context' AND deleted_at IS NULL",
                (moment_key,),
            )
            count_after_second = cursor.fetchone()[0]
            cursor.close()

        assert count_after_first > 0
        assert count_after_first == count_after_second  # no new rows on second call

    def test_worker_skips_moment_created_less_than_4_hours_ago(self, db_service, ks):
        """_poll_and_enrich skips moments created within the MOMENT_AGE_HOURS window."""
        from services.moment_context_service import _poll_and_enrich
        from unittest.mock import patch

        # Seed a recent moment (created_at = now — less than 4h ago)
        recent_created_at = utc_now().isoformat()
        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO knowledge (kind, entity, key, value, data, decay_class, confidence, source, created_at) "
                "VALUES ('moment', 'user', 'moment_777', 'Some content', ?, 'permanent', 1.0, 'user_pin', ?)",
                (json.dumps({'transcript_id': 777}), recent_created_at),
            )
            cursor.close()

        with patch('services.database_service.get_shared_db_service', return_value=db_service):
            _poll_and_enrich()

        # No context rows should have been created (moment too recent)
        count = _count_knowledge_by_entity(db_service, 'moment_777')
        assert count == 0

    def test_worker_skips_moment_already_enriched(self, db_service, ks):
        """_poll_and_enrich skips moments that already have context rows (idempotency guard)."""
        from services.moment_context_service import _poll_and_enrich
        from unittest.mock import patch

        # Seed an old enough moment (created >4h ago)
        old_created_at = (utc_now() - timedelta(hours=5)).isoformat()
        pinned_id = _insert_transcript(db_service, role='assistant', content='Old moment', channel='user')
        moment_key = f'moment_{pinned_id}'

        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO knowledge (kind, entity, key, value, data, decay_class, confidence, source, created_at) "
                "VALUES ('moment', 'user', ?, 'Old moment', ?, 'permanent', 1.0, 'user_pin', ?)",
                (moment_key, json.dumps({'transcript_id': pinned_id}), old_created_at),
            )
            cursor.close()

        # Pre-seed a context row so _is_enriched returns True
        ks.store(
            kind='moment_context', entity=moment_key, key=f'{moment_key}_ctx_0',
            value='[user] existing context',
            data={'parent_key': moment_key, 'transcript_id': pinned_id, 'role': 'user', 'ctx_index': 0},
            decay_class='permanent', confidence=1.0, source='moment_enrichment',
        )

        count_before = _count_knowledge_by_entity(db_service, moment_key)

        with patch('services.database_service.get_shared_db_service', return_value=db_service):
            _poll_and_enrich()

        count_after = _count_knowledge_by_entity(db_service, moment_key)

        # Count must be unchanged — worker skipped the already-enriched moment
        assert count_after == count_before
