"""
Unit tests for EpisodeConsolidationService.

Uses in-memory SQLite with the full episodes + episodes_vec schema.
All LLM calls and embedding generation are mocked.
"""

import json
import sqlite3
import struct
import uuid
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit

# ── Minimal schema ────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    intent TEXT NOT NULL,
    context TEXT NOT NULL,
    action TEXT NOT NULL,
    emotion TEXT NOT NULL,
    outcome TEXT NOT NULL,
    gist TEXT NOT NULL,
    salience REAL NOT NULL DEFAULT 5.0,
    channel TEXT NOT NULL DEFAULT 'test',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT,
    last_accessed_at TEXT,
    access_count INTEGER DEFAULT 0,
    deleted_at TEXT,
    salience_factors TEXT DEFAULT '{}',
    open_loops TEXT DEFAULT '[]',
    transcript_ids TEXT DEFAULT '[]',
    transcript_id_start INTEGER,
    transcript_id_end INTEGER,
    entities TEXT DEFAULT '[]',
    goal_tags TEXT DEFAULT '[]',
    emotional_valence REAL,
    emotional_arousal REAL,
    consolidated_from TEXT DEFAULT '[]',
    storage_strength REAL DEFAULT 1.0,
    retrieval_weight REAL DEFAULT 1.0
);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_vec USING vec0(
    embedding float[4]
);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    gist, action
);
"""


def _pack(vec) -> bytes:
    return struct.pack(f'{len(vec)}f', *vec)


class _FakeDB:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    @contextmanager
    def connection(self):
        yield self._conn
        self._conn.commit()


@pytest.fixture
def mem_db():
    conn = sqlite3.connect(":memory:")

    # Try to load sqlite-vec; fall back to a plain table that mimics the interface
    _vec_loaded = False
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        _vec_loaded = True
    except Exception:
        pass

    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY,
            intent TEXT NOT NULL,
            context TEXT NOT NULL,
            action TEXT NOT NULL,
            emotion TEXT NOT NULL,
            outcome TEXT NOT NULL,
            gist TEXT NOT NULL,
            salience REAL NOT NULL DEFAULT 5.0,
            channel TEXT NOT NULL DEFAULT 'test',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT,
            last_accessed_at TEXT,
            access_count INTEGER DEFAULT 0,
            deleted_at TEXT,
            salience_factors TEXT DEFAULT '{}',
            open_loops TEXT DEFAULT '[]',
            transcript_ids TEXT DEFAULT '[]',
            transcript_id_start INTEGER,
            transcript_id_end INTEGER,
            entities TEXT DEFAULT '[]',
            goal_tags TEXT DEFAULT '[]',
            emotional_valence REAL,
            emotional_arousal REAL,
            consolidated_from TEXT DEFAULT '[]',
            storage_strength REAL DEFAULT 1.0,
            retrieval_weight REAL DEFAULT 1.0
        )
    """)

    if _vec_loaded:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS episodes_vec USING vec0(embedding float[4])"
        )
    else:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes_vec (
                rowid INTEGER PRIMARY KEY,
                embedding BLOB
            )
        """)

    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(gist, action)"
    )

    conn.commit()

    yield conn
    conn.close()


@pytest.fixture
def fake_db(mem_db):
    return _FakeDB(mem_db)


def _insert_episode(conn, ep_id=None, gist="test gist", entities=None,
                    goal_tags=None, salience=5.0, storage_strength=1.0,
                    retrieval_weight=0.5, consolidated_from=None,
                    open_loops=None, emotional_valence=0.0, emotional_arousal=0.5,
                    embedding=None):
    ep_id = ep_id or str(uuid.uuid4())
    conn.execute("""
        INSERT INTO episodes (
            id, intent, context, action, emotion, outcome, gist,
            salience, channel, entities, goal_tags, storage_strength,
            retrieval_weight, consolidated_from, open_loops,
            emotional_valence, emotional_arousal
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ep_id,
        json.dumps({'type': 'exploration', 'direction': 'test'}),
        json.dumps('test context'),
        'test action',
        json.dumps({'valence': 'neutral', 'intensity': 'low'}),
        'test outcome',
        gist,
        salience,
        'test',
        json.dumps(entities or []),
        json.dumps(goal_tags or []),
        storage_strength,
        retrieval_weight,
        json.dumps(consolidated_from or []),
        json.dumps(open_loops or []),
        emotional_valence,
        emotional_arousal,
    ))
    conn.commit()

    if embedding is not None:
        blob = _pack(embedding)
        cursor = conn.execute("SELECT rowid FROM episodes WHERE id = ?", (ep_id,))
        row = cursor.fetchone()
        if row:
            conn.execute(
                "INSERT OR REPLACE INTO episodes_vec(rowid, embedding) VALUES (?, ?)",
                (row[0], blob)
            )
        conn.commit()

    return ep_id


def _make_llm_response(gist="Consolidated gist across episodes", entities=None,
                       goal_tags=None, open_loops=None):
    data = {
        'intent': {'type': 'reflection', 'direction': 'overall thread'},
        'context': 'Consolidated context',
        'action': 'Multiple related interactions',
        'emotion': {'valence': 'neutral', 'intensity': 'medium'},
        'outcome': 'Consolidated outcome',
        'gist': gist,
        'salience_factors': {'novelty': 2, 'emotional_weight': 1, 'goal_relevance': 2,
                              'decision_made': False, 'open_loop_created': False},
        'open_loops': open_loops or [],
        'entities': entities or [],
        'goal_tags': goal_tags or [],
        'emotional_valence': 0.1,
        'emotional_arousal': 0.5,
    }
    resp = MagicMock()
    resp.text = json.dumps(data)
    return resp


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRunConsolidationCycle:

    def _make_svc(self, fake_db, llm_response, prompt_template="Consolidate: {{source_episodes}}"):
        from services.episode_consolidation_service import EpisodeConsolidationService
        svc = EpisodeConsolidationService(fake_db)
        mock_llm = MagicMock()
        mock_llm.send_message.return_value = llm_response
        svc._llm = mock_llm
        svc._prompt_template = prompt_template
        return svc

    def test_cluster_of_three_produces_super_episode(self, mem_db, fake_db):
        """3 similar episodes (all returned by mock KNN) produce one super episode."""
        ep1 = _insert_episode(mem_db, gist="Episode about Python debugging",
                              embedding=[0.1, 0.2, 0.3, 0.4])
        ep2 = _insert_episode(mem_db, gist="Episode about fixing import errors",
                              embedding=[0.11, 0.21, 0.31, 0.41])
        ep3 = _insert_episode(mem_db, gist="Episode about virtual environments",
                              embedding=[0.12, 0.22, 0.32, 0.42])

        llm_resp = _make_llm_response(gist="Python environment setup arc")
        svc = self._make_svc(fake_db, llm_resp)

        # Mock KNN to return all 3 episodes as a tight cluster
        def mock_find_similar(embedding, exclude_id, exclude_ids):
            all_eps = [
                {'id': ep1, 'rowid': 1, 'gist': 'g1', 'context': 'c', 'intent': '',
                 'outcome': 'o', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 1.0, 'retrieval_weight': 0.5},
                {'id': ep2, 'rowid': 2, 'gist': 'g2', 'context': 'c', 'intent': '',
                 'outcome': 'o', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 1.0, 'retrieval_weight': 0.5},
                {'id': ep3, 'rowid': 3, 'gist': 'g3', 'context': 'c', 'intent': '',
                 'outcome': 'o', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 1.0, 'retrieval_weight': 0.5},
            ]
            return [ep for ep in all_eps if ep['id'] != exclude_id and ep['id'] not in exclude_ids]

        svc._find_similar_episodes = mock_find_similar

        with patch('services.embedding_service.get_embedding_service') as mock_emb:
            mock_emb.return_value.generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
            count = svc.run_consolidation_cycle()

        assert count == 1

        cursor = mem_db.execute(
            "SELECT id, consolidated_from, gist FROM episodes WHERE consolidated_from != '[]' AND consolidated_from IS NOT NULL"
        )
        rows = cursor.fetchall()
        assert len(rows) == 1
        cf = json.loads(rows[0][1])
        assert len(cf) == 3
        assert set(cf) == {ep1, ep2, ep3}

    def test_source_episodes_remain_untouched(self, mem_db, fake_db):
        """After consolidation, source episodes still exist with original data."""
        ep1 = _insert_episode(mem_db, gist="Source 1", storage_strength=2.0,
                              embedding=[0.1, 0.2, 0.3, 0.4])
        ep2 = _insert_episode(mem_db, gist="Source 2", storage_strength=3.0,
                              embedding=[0.1, 0.2, 0.3, 0.4])
        ep3 = _insert_episode(mem_db, gist="Source 3", storage_strength=1.5,
                              embedding=[0.1, 0.2, 0.3, 0.4])

        llm_resp = _make_llm_response(gist="Merged super gist")
        svc = self._make_svc(fake_db, llm_resp)

        def mock_find_similar(embedding, exclude_id, exclude_ids):
            all_eps = [
                {'id': ep1, 'rowid': 1, 'gist': 'Source 1', 'context': '', 'intent': '',
                 'outcome': '', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 2.0, 'retrieval_weight': 0.5},
                {'id': ep2, 'rowid': 2, 'gist': 'Source 2', 'context': '', 'intent': '',
                 'outcome': '', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 3.0, 'retrieval_weight': 0.5},
                {'id': ep3, 'rowid': 3, 'gist': 'Source 3', 'context': '', 'intent': '',
                 'outcome': '', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 1.5, 'retrieval_weight': 0.5},
            ]
            return [ep for ep in all_eps if ep['id'] != exclude_id and ep['id'] not in exclude_ids]

        svc._find_similar_episodes = mock_find_similar

        with patch('services.embedding_service.get_embedding_service') as mock_emb:
            mock_emb.return_value.generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
            svc.run_consolidation_cycle()

        for ep_id in [ep1, ep2, ep3]:
            row = mem_db.execute(
                "SELECT gist, storage_strength FROM episodes WHERE id = ?", (ep_id,)
            ).fetchone()
            assert row is not None, f"Source episode {ep_id} should still exist"

        assert mem_db.execute(
            "SELECT gist FROM episodes WHERE id = ?", (ep1,)
        ).fetchone()[0] == "Source 1"

    def test_consolidated_from_populated_correctly(self, mem_db, fake_db):
        """consolidated_from contains exactly the source episode IDs."""
        source_ids = []
        for i in range(3):
            eid = _insert_episode(mem_db, gist=f"Source {i}",
                                  embedding=[0.1 * (i + 1), 0.2, 0.3, 0.4])
            source_ids.append(eid)

        llm_resp = _make_llm_response(gist="Super gist")
        svc = self._make_svc(fake_db, llm_resp)

        def mock_find_similar(embedding, exclude_id, exclude_ids):
            return [
                {'id': sid, 'rowid': i + 1, 'gist': f'Source {i}', 'context': '',
                 'intent': '', 'outcome': '', 'open_loops': [], 'entities': [],
                 'goal_tags': [], 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 1.0, 'retrieval_weight': 0.5}
                for i, sid in enumerate(source_ids)
                if sid != exclude_id and sid not in exclude_ids
            ]

        svc._find_similar_episodes = mock_find_similar

        with patch('services.embedding_service.get_embedding_service') as mock_emb:
            mock_emb.return_value.generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
            count = svc.run_consolidation_cycle()

        assert count == 1
        row = mem_db.execute(
            "SELECT consolidated_from FROM episodes WHERE consolidated_from != '[]'"
        ).fetchone()
        cf = json.loads(row[0])
        assert set(cf) == set(source_ids)

    def test_already_consolidated_episodes_not_re_consolidated(self, mem_db, fake_db):
        """Episodes already in a super episode's consolidated_from are not re-consolidated."""
        ep1 = _insert_episode(mem_db, gist="Already consolidated 1",
                              embedding=[0.1, 0.2, 0.3, 0.4])
        ep2 = _insert_episode(mem_db, gist="Already consolidated 2",
                              embedding=[0.1, 0.2, 0.3, 0.4])
        ep3 = _insert_episode(mem_db, gist="Already consolidated 3",
                              embedding=[0.1, 0.2, 0.3, 0.4])

        # Create an existing super episode that references ep1, ep2, ep3
        existing_super = str(uuid.uuid4())
        mem_db.execute("""
            INSERT INTO episodes (
                id, intent, context, action, emotion, outcome, gist,
                salience, channel, consolidated_from, retrieval_weight
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            existing_super,
            json.dumps({'type': 'reflection'}),
            json.dumps('ctx'),
            'action',
            json.dumps({'valence': 'neutral'}),
            'outcome',
            'Existing super episode',
            7.0,
            'consolidated',
            json.dumps([ep1, ep2, ep3]),
            1.0,
        ))
        mem_db.commit()

        llm_resp = _make_llm_response(gist="Should not be called")
        svc = self._make_svc(fake_db, llm_resp)

        with patch('services.embedding_service.get_embedding_service') as mock_emb:
            mock_emb.return_value.generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
            count = svc.run_consolidation_cycle()

        assert count == 0
        svc._llm.send_message.assert_not_called()

    def test_singleton_episode_not_consolidated(self, mem_db, fake_db):
        """An episode with no similar neighbours is not consolidated."""
        _insert_episode(mem_db, gist="Lone episode", embedding=[0.1, 0.2, 0.3, 0.4])

        llm_resp = _make_llm_response(gist="Should not happen")
        svc = self._make_svc(fake_db, llm_resp)

        svc._find_similar_episodes = MagicMock(return_value=[])

        with patch('services.embedding_service.get_embedding_service') as mock_emb:
            mock_emb.return_value.generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
            count = svc.run_consolidation_cycle()

        assert count == 0
        svc._llm.send_message.assert_not_called()

    def test_storage_strength_sum_capped_at_ten(self, mem_db, fake_db):
        """Super episode storage_strength = sum of sources, capped at 10.0."""
        ep1 = _insert_episode(mem_db, gist="Ep1", storage_strength=4.0,
                              embedding=[0.1, 0.2, 0.3, 0.4])
        ep2 = _insert_episode(mem_db, gist="Ep2", storage_strength=4.0,
                              embedding=[0.1, 0.2, 0.3, 0.4])
        ep3 = _insert_episode(mem_db, gist="Ep3", storage_strength=4.0,
                              embedding=[0.1, 0.2, 0.3, 0.4])

        llm_resp = _make_llm_response(gist="High strength super")
        svc = self._make_svc(fake_db, llm_resp)

        def mock_find_similar(embedding, exclude_id, exclude_ids):
            all_eps = [
                {'id': ep1, 'rowid': 1, 'gist': 'Ep1', 'context': '', 'intent': '',
                 'outcome': '', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 4.0, 'retrieval_weight': 0.5},
                {'id': ep2, 'rowid': 2, 'gist': 'Ep2', 'context': '', 'intent': '',
                 'outcome': '', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 4.0, 'retrieval_weight': 0.5},
                {'id': ep3, 'rowid': 3, 'gist': 'Ep3', 'context': '', 'intent': '',
                 'outcome': '', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 4.0, 'retrieval_weight': 0.5},
            ]
            return [ep for ep in all_eps if ep['id'] != exclude_id and ep['id'] not in exclude_ids]

        svc._find_similar_episodes = mock_find_similar

        with patch('services.embedding_service.get_embedding_service') as mock_emb:
            mock_emb.return_value.generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
            count = svc.run_consolidation_cycle()

        assert count == 1
        row = mem_db.execute(
            "SELECT storage_strength FROM episodes WHERE consolidated_from != '[]'"
        ).fetchone()
        # 4.0 + 4.0 + 4.0 = 12.0, capped at 10.0
        assert row[0] == pytest.approx(10.0)

    def test_storage_strength_sum_below_cap(self, mem_db, fake_db):
        """Super episode storage_strength = exact sum when below 10."""
        ep1 = _insert_episode(mem_db, gist="Ep1", storage_strength=1.5,
                              embedding=[0.1, 0.2, 0.3, 0.4])
        ep2 = _insert_episode(mem_db, gist="Ep2", storage_strength=2.0,
                              embedding=[0.1, 0.2, 0.3, 0.4])
        ep3 = _insert_episode(mem_db, gist="Ep3", storage_strength=1.0,
                              embedding=[0.1, 0.2, 0.3, 0.4])

        llm_resp = _make_llm_response(gist="Normal strength super")
        svc = self._make_svc(fake_db, llm_resp)

        def mock_find_similar(embedding, exclude_id, exclude_ids):
            all_eps = [
                {'id': ep1, 'rowid': 1, 'gist': 'Ep1', 'context': '', 'intent': '',
                 'outcome': '', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 1.5, 'retrieval_weight': 0.5},
                {'id': ep2, 'rowid': 2, 'gist': 'Ep2', 'context': '', 'intent': '',
                 'outcome': '', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 2.0, 'retrieval_weight': 0.5},
                {'id': ep3, 'rowid': 3, 'gist': 'Ep3', 'context': '', 'intent': '',
                 'outcome': '', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 1.0, 'retrieval_weight': 0.5},
            ]
            return [ep for ep in all_eps if ep['id'] != exclude_id and ep['id'] not in exclude_ids]

        svc._find_similar_episodes = mock_find_similar

        with patch('services.embedding_service.get_embedding_service') as mock_emb:
            mock_emb.return_value.generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
            count = svc.run_consolidation_cycle()

        assert count == 1
        row = mem_db.execute(
            "SELECT storage_strength FROM episodes WHERE consolidated_from != '[]'"
        ).fetchone()
        assert abs(row[0] - 4.5) < 0.01
