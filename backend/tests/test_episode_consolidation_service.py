"""
Unit tests for EpisodeConsolidationService.

Uses in-memory SQLite built from schema.sql — the single source of truth for
the episodes table definition. LLM calls and embedding generation are mocked
because we cannot call a real LLM in unit tests.
"""

import json
import re
import sqlite3
import struct
import uuid
from pathlib import Path
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit

_SCHEMA_PATH = Path(__file__).parent.parent / "schema.sql"


def _build_schema(conn: sqlite3.Connection) -> None:
    """Apply the real production schema.sql to an in-memory connection.

    Tries sqlite-vec first; if unavailable, skips vec0 statements so the rest
    of the schema (episodes table, FTS5, indexes) still applies cleanly.
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
        conn.executescript(sql)
    else:
        for stmt in re.split(r';', sql):
            s = stmt.strip()
            if not s or 'vec0' in s.lower():
                continue
            try:
                conn.execute(s)
            except Exception:
                pass
        conn.commit()


_VEC_DIM = 768  # must match schema.sql episodes_vec float[768]


def _pack(vec) -> bytes:
    return struct.pack(f'{len(vec)}f', *vec)


def _make_embedding(seed: list) -> list:
    """Return a 768-element float list. Pads seed with zeros to reach _VEC_DIM."""
    padded = list(seed) + [0.0] * (_VEC_DIM - len(seed))
    return padded[:_VEC_DIM]


class _FakeDB:
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
    _build_schema(conn)

    # When sqlite-vec is unavailable the vec0 tables are absent.  The
    # consolidation tests that exercise KNN (via _find_similar_episodes) mock
    # that method directly, so the missing vec table doesn't affect them.
    # For tests that insert embeddings via _insert_episode, we need a fallback
    # plain table only if episodes_vec wasn't created.
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','shadow')"
    ).fetchall()}
    if 'episodes_vec' not in tables:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes_vec (
                rowid INTEGER PRIMARY KEY,
                embedding BLOB
            )
        """)
        conn.commit()

    yield conn
    conn.close()


@pytest.fixture
def fake_db(mem_db):
    return _FakeDB(mem_db)


def _insert_episode(conn, ep_id=None, gist="test gist", salience=5,
                    storage_strength=1.0, retrieval_weight=0.5,
                    consolidated_from=None, emotional_valence=0.0,
                    emotional_arousal=0.5, embedding=None, channel='test',
                    transcript_id_start=None, transcript_id_end=None):
    ep_id = ep_id or str(uuid.uuid4())
    conn.execute("""
        INSERT INTO episodes (
            id, gist, salience, channel, storage_strength,
            retrieval_weight, consolidated_from,
            emotional_valence, emotional_arousal,
            transcript_id_start, transcript_id_end
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ep_id,
        gist,
        salience,
        channel,
        storage_strength,
        retrieval_weight,
        json.dumps(consolidated_from or []),
        emotional_valence,
        emotional_arousal,
        transcript_id_start,
        transcript_id_end,
    ))
    conn.commit()

    if embedding is not None:
        blob = _pack(_make_embedding(embedding))
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
            mock_emb.return_value.generate_embedding.return_value = _make_embedding([0.1, 0.2, 0.3, 0.4])
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
            mock_emb.return_value.generate_embedding.return_value = _make_embedding([0.1, 0.2, 0.3, 0.4])
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
            mock_emb.return_value.generate_embedding.return_value = _make_embedding([0.1, 0.2, 0.3, 0.4])
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
                id, gist, salience, channel, consolidated_from, retrieval_weight
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            existing_super,
            'Existing super episode',
            7,
            'consolidated',
            json.dumps([ep1, ep2, ep3]),
            1.0,
        ))
        mem_db.commit()

        llm_resp = _make_llm_response(gist="Should not be called")
        svc = self._make_svc(fake_db, llm_resp)

        with patch('services.embedding_service.get_embedding_service') as mock_emb:
            mock_emb.return_value.generate_embedding.return_value = _make_embedding([0.1, 0.2, 0.3, 0.4])
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
            mock_emb.return_value.generate_embedding.return_value = _make_embedding([0.1, 0.2, 0.3, 0.4])
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
            mock_emb.return_value.generate_embedding.return_value = _make_embedding([0.1, 0.2, 0.3, 0.4])
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
            mock_emb.return_value.generate_embedding.return_value = _make_embedding([0.1, 0.2, 0.3, 0.4])
            count = svc.run_consolidation_cycle()

        assert count == 1
        row = mem_db.execute(
            "SELECT storage_strength FROM episodes WHERE consolidated_from != '[]'"
        ).fetchone()
        assert abs(row[0] - 4.5) < 0.01

    def test_super_episode_inherits_transcript_range_from_sources(self, mem_db, fake_db):
        """Super episode transcript_id_start/end spans all source episodes."""
        ep1 = _insert_episode(mem_db, gist="Ep1", embedding=[0.1, 0.2, 0.3, 0.4],
                              transcript_id_start=1, transcript_id_end=10)
        ep2 = _insert_episode(mem_db, gist="Ep2", embedding=[0.1, 0.2, 0.3, 0.4],
                              transcript_id_start=8, transcript_id_end=20)
        ep3 = _insert_episode(mem_db, gist="Ep3", embedding=[0.1, 0.2, 0.3, 0.4],
                              transcript_id_start=18, transcript_id_end=30)

        llm_resp = _make_llm_response(gist="Transcript range test")
        svc = self._make_svc(fake_db, llm_resp)

        def mock_find_similar(embedding, exclude_id, exclude_ids):
            all_eps = [
                {'id': ep1, 'rowid': 1, 'gist': 'Ep1', 'context': '', 'intent': '',
                 'outcome': '', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 1.0, 'retrieval_weight': 0.5,
                 'transcript_id_start': 1, 'transcript_id_end': 10, 'channel': 'chat'},
                {'id': ep2, 'rowid': 2, 'gist': 'Ep2', 'context': '', 'intent': '',
                 'outcome': '', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 1.0, 'retrieval_weight': 0.5,
                 'transcript_id_start': 8, 'transcript_id_end': 20, 'channel': 'chat'},
                {'id': ep3, 'rowid': 3, 'gist': 'Ep3', 'context': '', 'intent': '',
                 'outcome': '', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 1.0, 'retrieval_weight': 0.5,
                 'transcript_id_start': 18, 'transcript_id_end': 30, 'channel': 'chat'},
            ]
            return [ep for ep in all_eps if ep['id'] != exclude_id and ep['id'] not in exclude_ids]

        svc._find_similar_episodes = mock_find_similar

        with patch('services.embedding_service.get_embedding_service') as mock_emb:
            mock_emb.return_value.generate_embedding.return_value = _make_embedding([0.1, 0.2, 0.3, 0.4])
            count = svc.run_consolidation_cycle()

        assert count == 1
        row = mem_db.execute(
            "SELECT transcript_id_start, transcript_id_end FROM episodes "
            "WHERE consolidated_from != '[]' AND consolidated_from IS NOT NULL"
        ).fetchone()
        assert row[0] == 1
        assert row[1] == 30

    def test_super_episode_channel_from_majority_vote(self, mem_db, fake_db):
        """Super episode channel is the most common channel among source episodes."""
        ep1 = _insert_episode(mem_db, gist="Ep1", embedding=[0.1, 0.2, 0.3, 0.4],
                              channel='voice')
        ep2 = _insert_episode(mem_db, gist="Ep2", embedding=[0.1, 0.2, 0.3, 0.4],
                              channel='chat')
        ep3 = _insert_episode(mem_db, gist="Ep3", embedding=[0.1, 0.2, 0.3, 0.4],
                              channel='chat')

        llm_resp = _make_llm_response(gist="Channel test")
        svc = self._make_svc(fake_db, llm_resp)

        def mock_find_similar(embedding, exclude_id, exclude_ids):
            all_eps = [
                {'id': ep1, 'rowid': 1, 'gist': 'Ep1', 'context': '', 'intent': '',
                 'outcome': '', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 1.0, 'retrieval_weight': 0.5,
                 'transcript_id_start': None, 'transcript_id_end': None, 'channel': 'voice'},
                {'id': ep2, 'rowid': 2, 'gist': 'Ep2', 'context': '', 'intent': '',
                 'outcome': '', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 1.0, 'retrieval_weight': 0.5,
                 'transcript_id_start': None, 'transcript_id_end': None, 'channel': 'chat'},
                {'id': ep3, 'rowid': 3, 'gist': 'Ep3', 'context': '', 'intent': '',
                 'outcome': '', 'open_loops': [], 'entities': [], 'goal_tags': [],
                 'emotional_valence': 0.0, 'emotional_arousal': 0.5,
                 'salience': 5.0, 'storage_strength': 1.0, 'retrieval_weight': 0.5,
                 'transcript_id_start': None, 'transcript_id_end': None, 'channel': 'chat'},
            ]
            return [ep for ep in all_eps if ep['id'] != exclude_id and ep['id'] not in exclude_ids]

        svc._find_similar_episodes = mock_find_similar

        with patch('services.embedding_service.get_embedding_service') as mock_emb:
            mock_emb.return_value.generate_embedding.return_value = _make_embedding([0.1, 0.2, 0.3, 0.4])
            count = svc.run_consolidation_cycle()

        assert count == 1
        row = mem_db.execute(
            "SELECT channel FROM episodes WHERE consolidated_from != '[]' AND consolidated_from IS NOT NULL"
        ).fetchone()
        assert row[0] == 'chat'
