"""
Unit tests for Commit C — super-episode pipeline.

Covers:
  - walk_up_to_apex  (episodic_retrieval_service)
  - find_super_candidates  (episodic_service)
  - Retrieval apex-promotion  (episodic_retrieval_service.retrieve)

Database strategy: in-memory SQLite built from the real schema.sql so that
column definitions never drift from production.

LLM, embedding service, and sqlite-vec are not required — tests either inject
raw blobs directly into episodes_vec or skip the vec-dependent path gracefully.
"""

import json
import re
import sqlite3
import struct
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

_SCHEMA_PATH = Path(__file__).parent.parent / "schema.sql"


# ── Schema helpers ─────────────────────────────────────────────────────────────


def _try_load_vec(conn: sqlite3.Connection) -> bool:
    """Return True if sqlite-vec loaded successfully."""
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except Exception:
        return False


def _build_schema(conn: sqlite3.Connection, *, vec: bool) -> None:
    """Apply schema.sql, optionally skipping vec0 statements."""
    sql = _SCHEMA_PATH.read_text()
    if vec:
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


# ── Fake DB wrapper ────────────────────────────────────────────────────────────


class _FakeDB:
    """Thin wrapper satisfying db_service.connection() context-manager API."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    @contextmanager
    def connection(self):
        yield self._conn
        self._conn.commit()


# ── Embedding helpers ──────────────────────────────────────────────────────────


def _pack(floats: list[float]) -> bytes:
    """Pack a float list into a sqlite-vec binary blob."""
    return struct.pack(f'{len(floats)}f', *floats)


# Embedding dimension matching schema.sql episodes_vec float[768]
_EMB_DIM = 768


def _unit(value: float = 1.0) -> list[float]:
    """Return a _EMB_DIM-dimensional unit vector with ``value`` in the first slot."""
    import math
    v = [0.0] * _EMB_DIM
    v[0] = value
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _sim_vec(*, angle_deg: float = 0.0) -> list[float]:
    """Return a _EMB_DIM unit vector at ``angle_deg`` from the x-axis (2-plane)."""
    import math
    rad = math.radians(angle_deg)
    v = [0.0] * _EMB_DIM
    v[0] = math.cos(rad)
    if _EMB_DIM >= 2:
        v[1] = math.sin(rad)
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def mem_db():
    """In-memory SQLite built from real schema.sql — no sqlite-vec required."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _build_schema(conn, vec=False)
    yield conn
    conn.close()


@pytest.fixture
def mem_db_vec():
    """In-memory SQLite with sqlite-vec loaded.

    Yields (conn, vec_available) so tests can skip gracefully.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    vec = _try_load_vec(conn)
    _build_schema(conn, vec=vec)
    yield conn, vec
    conn.close()


@pytest.fixture
def fake_db(mem_db):
    return _FakeDB(mem_db)


@pytest.fixture
def fake_db_vec(mem_db_vec):
    conn, vec = mem_db_vec
    return _FakeDB(conn), conn, vec


@pytest.fixture
def episodic_svc(fake_db):
    from services.episodic_service import EpisodicService
    return EpisodicService(fake_db)


# ── Helpers to seed episodes directly via SQL ─────────────────────────────────


def _insert_episode(
    conn: sqlite3.Connection,
    *,
    episode_id: Optional[str] = None,
    gist: str = "Test gist",
    salience: int = 5,
    channel: str = "test",
    consolidated_into: Optional[str] = None,
    consolidated_from: Optional[list] = None,
    deleted_at: Optional[str] = None,
    transcript_ids: Optional[list] = None,
    transcript_id_start: Optional[int] = None,
    transcript_id_end: Optional[int] = None,
    emotional_valence: float = 0.0,
    emotional_arousal: float = 0.0,
) -> str:
    """Insert a minimal episode row. Returns the episode UUID."""
    eid = episode_id or str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO episodes (
            id, gist, salience, channel,
            transcript_ids, transcript_id_start, transcript_id_end,
            emotional_valence, emotional_arousal,
            consolidated_from, consolidated_into, deleted_at,
            storage_strength, retrieval_weight
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, 1.0)
        """,
        (
            eid, gist, salience, channel,
            json.dumps(transcript_ids or []),
            transcript_id_start,
            transcript_id_end,
            emotional_valence,
            emotional_arousal,
            json.dumps(consolidated_from or []),
            consolidated_into,
            deleted_at,
        ),
    )
    conn.commit()
    return eid


def _insert_embedding(conn: sqlite3.Connection, episode_id: str, emb: list[float]) -> None:
    """Insert an embedding blob into episodes_vec for the given episode_id.

    Silently skips if sqlite-vec is not loaded (vec0 table absent).
    """
    try:
        row = conn.execute("SELECT rowid FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        if row:
            conn.execute(
                "INSERT OR REPLACE INTO episodes_vec(rowid, embedding) VALUES (?, ?)",
                (row[0], _pack(emb)),
            )
            conn.commit()
    except Exception:
        pass  # sqlite-vec not loaded — tests that need it will skip


# ═══════════════════════════════════════════════════════════════════════════════
# walk_up_to_apex
# ═══════════════════════════════════════════════════════════════════════════════


def _walk(episode_id: str, fake_db) -> Optional[dict]:
    """Call walk_up_to_apex with the test DB injected."""
    from services.episodic_service import EpisodicService
    from services.episodic_retrieval_service import walk_up_to_apex

    # walk_up_to_apex does a local import of get_shared_db_service inside
    # EpisodicService.__init__, so we patch at the database_service level.
    with patch('services.database_service.get_shared_db_service', return_value=fake_db):
        return walk_up_to_apex(episode_id)


class TestWalkUpToApex:
    """Unit tests for episodic_retrieval_service.walk_up_to_apex()."""

    def test_leaf_no_consolidated_into_returns_itself(self, mem_db, fake_db):
        """A leaf episode with consolidated_into=NULL is its own apex."""
        eid = _insert_episode(mem_db, gist="leaf gist", channel="ch1")
        result = _walk(eid, fake_db)
        assert result is not None
        assert result['id'] == eid

    def test_one_hop_returns_super(self, mem_db, fake_db):
        """leaf → super (one hop) → returns super."""
        leaf_id = _insert_episode(mem_db, gist="leaf", channel="ch1")
        super_id = _insert_episode(mem_db, gist="super", channel="ch1")
        mem_db.execute(
            "UPDATE episodes SET consolidated_into = ? WHERE id = ?",
            (super_id, leaf_id),
        )
        mem_db.commit()

        result = _walk(leaf_id, fake_db)
        assert result is not None
        assert result['id'] == super_id

    def test_two_hops_returns_apex(self, mem_db, fake_db):
        """leaf → super → super-super (two hops) → returns apex."""
        leaf_id = _insert_episode(mem_db, gist="leaf", channel="ch1")
        super_id = _insert_episode(mem_db, gist="super", channel="ch1")
        apex_id = _insert_episode(mem_db, gist="apex", channel="ch1")

        mem_db.execute("UPDATE episodes SET consolidated_into = ? WHERE id = ?", (super_id, leaf_id))
        mem_db.execute("UPDATE episodes SET consolidated_into = ? WHERE id = ?", (apex_id, super_id))
        mem_db.commit()

        result = _walk(leaf_id, fake_db)
        assert result is not None
        assert result['id'] == apex_id

    def test_cycle_logs_error_and_returns(self, mem_db, fake_db, caplog):
        """Cycle (a→b→a) is detected, logs an error, does not hang."""
        import logging
        a_id = _insert_episode(mem_db, gist="a", channel="ch1")
        b_id = _insert_episode(mem_db, gist="b", channel="ch1")

        mem_db.execute("UPDATE episodes SET consolidated_into = ? WHERE id = ?", (b_id, a_id))
        mem_db.execute("UPDATE episodes SET consolidated_into = ? WHERE id = ?", (a_id, b_id))
        mem_db.commit()

        with caplog.at_level(logging.ERROR, logger='services.episodic_retrieval_service'):
            result = _walk(a_id, fake_db)

        # Does not raise, does not hang, returns something
        assert result is not None
        assert 'cycle' in caplog.text.lower()

    def test_nonexistent_episode_returns_none(self, fake_db):
        """Walking a non-existent episode ID returns None gracefully."""
        result = _walk("00000000-0000-0000-0000-000000000000", fake_db)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# find_super_candidates
# ═══════════════════════════════════════════════════════════════════════════════


class TestFindSuperCandidates:
    """Unit tests for episodic_service.find_super_candidates().

    All tests require sqlite-vec (episodes_vec table) — they skip when unavailable.
    """

    def _insert_apex(
        self,
        conn: sqlite3.Connection,
        emb: list[float],
        *,
        channel: str = "ch1",
        gist: str = "test",
    ) -> str:
        eid = _insert_episode(conn, gist=gist, channel=channel)
        _insert_embedding(conn, eid, emb)
        return eid

    def _find(self, conn: sqlite3.Connection, channel: str) -> list[list[str]]:
        """Call find_super_candidates with the test DB injected."""
        from services.episodic_service import find_super_candidates
        fake_db = _FakeDB(conn)
        with patch('services.database_service.get_shared_db_service', return_value=fake_db):
            return find_super_candidates(channel)

    def test_empty_channel_returns_empty(self, mem_db_vec):
        """No episodes → empty list."""
        conn, vec = mem_db_vec
        if not vec:
            pytest.skip("sqlite-vec not available")

        result = self._find(conn, "empty_channel")
        assert result == []

    def test_tight_cluster_returns_one_cluster(self, mem_db_vec):
        """3 embeddings at angle 0° from each other form a tight cluster."""
        conn, vec = mem_db_vec
        if not vec:
            pytest.skip("sqlite-vec not available")

        # All three identical (cosine = 1.0 > 0.90 threshold)
        base = _sim_vec(angle_deg=0.0)
        e1 = self._insert_apex(conn, base, gist="ep1")
        e2 = self._insert_apex(conn, base, gist="ep2")
        e3 = self._insert_apex(conn, base, gist="ep3")

        clusters = self._find(conn, "ch1")
        assert len(clusters) == 1
        assert sorted(clusters[0]) == sorted([e1, e2, e3])

    def test_loose_embeddings_returns_empty(self, mem_db_vec):
        """3 embeddings far apart produce no cluster."""
        conn, vec = mem_db_vec
        if not vec:
            pytest.skip("sqlite-vec not available")

        # 60° apart → cosine ≈ 0.5, well below 0.90 threshold
        self._insert_apex(conn, _sim_vec(angle_deg=0.0), gist="ep1")
        self._insert_apex(conn, _sim_vec(angle_deg=60.0), gist="ep2")
        self._insert_apex(conn, _sim_vec(angle_deg=120.0), gist="ep3")

        clusters = self._find(conn, "ch1")
        assert clusters == []

    def test_two_tight_clusters_both_emitted(self, mem_db_vec):
        """Two disjoint tight clusters in the same channel are both emitted with no overlap."""
        conn, vec = mem_db_vec
        if not vec:
            pytest.skip("sqlite-vec not available")

        # Cluster A: angle 0° (cosine ≈ 1.0 between all three)
        base_a = _sim_vec(angle_deg=0.0)
        a1 = self._insert_apex(conn, base_a, gist="a1")
        a2 = self._insert_apex(conn, base_a, gist="a2")
        a3 = self._insert_apex(conn, base_a, gist="a3")

        # Cluster B: angle 90° (orthogonal to A; within-group cosine = 1.0)
        base_b = _sim_vec(angle_deg=90.0)
        b1 = self._insert_apex(conn, base_b, gist="b1")
        b2 = self._insert_apex(conn, base_b, gist="b2")
        b3 = self._insert_apex(conn, base_b, gist="b3")

        clusters = self._find(conn, "ch1")
        assert len(clusters) == 2
        # All 6 IDs are accounted for, no overlap
        all_emitted = [eid for cl in clusters for eid in cl]
        assert sorted(all_emitted) == sorted([a1, a2, a3, b1, b2, b3])

    def test_already_consolidated_episodes_excluded(self, mem_db_vec):
        """Episodes with consolidated_into set (i.e. non-apex) are not in the pool."""
        conn, vec = mem_db_vec
        if not vec:
            pytest.skip("sqlite-vec not available")

        base = _sim_vec(angle_deg=0.0)

        # Insert 3 tight episodes but mark 2 as consolidated (non-apex)
        super_id = _insert_episode(conn, gist="super", channel="ch1")
        self._insert_apex(conn, base, gist="leaf1")
        e2 = _insert_episode(conn, gist="leaf2", channel="ch1", consolidated_into=super_id)
        _insert_embedding(conn, e2, base)
        e3 = _insert_episode(conn, gist="leaf3", channel="ch1", consolidated_into=super_id)
        _insert_embedding(conn, e3, base)

        clusters = self._find(conn, "ch1")
        # Only leaf1 and super_id are apex; 2 items < SUPER_EPISODE_MIN_CLUSTER=3 → no cluster
        assert clusters == []


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval apex-promotion
# ═══════════════════════════════════════════════════════════════════════════════


def _retrieve_with_db(query_text, query_embedding, channel, k, fake_db, *, fts_hits=None, vector_hits=None):
    """Call retrieve() with fake search lanes and test DB injected.

    retrieve() takes query_text as the only positional arg; query_embedding and
    all other params are keyword-only (enforced by the ``*`` in its signature).
    """
    from services.episodic_retrieval_service import retrieve

    with patch('services.episodic_retrieval_service._fts_search', return_value=fts_hits or []), \
         patch('services.episodic_retrieval_service._vector_search', return_value=vector_hits or []), \
         patch('services.database_service.get_shared_db_service', return_value=fake_db):
        return retrieve(query_text, query_embedding=query_embedding, channel=channel, k=k)


class TestRetrievalApexPromotion:
    """Unit tests for episodic_retrieval_service.retrieve() apex-promotion behaviour.

    Because sqlite-vec is required for vector_search, these tests mock the
    internal search lanes and inject episodes directly, then verify that
    walk_up_to_apex is applied and the apex is returned rather than the leaf.
    """

    def _make_hit(self, eid, gist, *, consolidated_into=None, vector_distance=0.05):
        return {
            'id': eid,
            'gist': gist,
            'salience': 5,
            'channel': 'ch1',
            'created_at': '2026-01-01T00:00:00+00:00',
            'last_accessed_at': None,
            'retrieval_weight': 1.0,
            'emotional_valence': 0.0,
            'emotional_arousal': 0.0,
            'consolidated_into': consolidated_into,
            'text_rank': None,
            'vector_distance': vector_distance,
        }

    def test_leaf_hit_promoted_to_super(self, mem_db, fake_db):
        """Vector-search hit on leaf → retrieve() returns the super, not the leaf."""
        leaf_id = _insert_episode(mem_db, gist="leaf", channel="ch1")
        super_id = _insert_episode(mem_db, gist="super", channel="ch1")

        mem_db.execute("UPDATE episodes SET consolidated_into = ? WHERE id = ?", (super_id, leaf_id))
        mem_db.commit()

        leaf_hit = self._make_hit(leaf_id, "leaf", consolidated_into=super_id)
        results = _retrieve_with_db(
            "test query", [0.0] * _EMB_DIM, "ch1", 10, fake_db,
            vector_hits=[leaf_hit],
        )

        returned_ids = [r['id'] for r in results]
        assert super_id in returned_ids, f"Expected super_id={super_id} in {returned_ids}"
        assert leaf_id not in returned_ids, "Leaf should be replaced by its apex"

    def test_two_hop_chain_returns_apex(self, mem_db, fake_db):
        """leaf → super → super-super chain: retrieve() returns the outermost apex."""
        leaf_id = _insert_episode(mem_db, gist="leaf", channel="ch1")
        super_id = _insert_episode(mem_db, gist="super", channel="ch1")
        apex_id = _insert_episode(mem_db, gist="apex", channel="ch1")

        mem_db.execute("UPDATE episodes SET consolidated_into = ? WHERE id = ?", (super_id, leaf_id))
        mem_db.execute("UPDATE episodes SET consolidated_into = ? WHERE id = ?", (apex_id, super_id))
        mem_db.commit()

        leaf_hit = self._make_hit(leaf_id, "leaf", consolidated_into=super_id)
        results = _retrieve_with_db(
            "test query", [0.0] * _EMB_DIM, "ch1", 10, fake_db,
            vector_hits=[leaf_hit],
        )

        returned_ids = [r['id'] for r in results]
        assert apex_id in returned_ids, f"Expected apex_id={apex_id} in {returned_ids}"
        assert leaf_id not in returned_ids
        assert super_id not in returned_ids

    def test_apex_deduplication(self, mem_db, fake_db):
        """Two leaves from the same super emit the super only once."""
        leaf1_id = _insert_episode(mem_db, gist="leaf1", channel="ch1")
        leaf2_id = _insert_episode(mem_db, gist="leaf2", channel="ch1")
        super_id = _insert_episode(mem_db, gist="super", channel="ch1")

        mem_db.execute("UPDATE episodes SET consolidated_into = ? WHERE id = ?", (super_id, leaf1_id))
        mem_db.execute("UPDATE episodes SET consolidated_into = ? WHERE id = ?", (super_id, leaf2_id))
        mem_db.commit()

        leaf_hits = [
            self._make_hit(leaf1_id, "leaf1", consolidated_into=super_id),
            self._make_hit(leaf2_id, "leaf2", consolidated_into=super_id),
        ]
        results = _retrieve_with_db(
            "test query", [0.0] * _EMB_DIM, "ch1", 10, fake_db,
            vector_hits=leaf_hits,
        )

        returned_ids = [r['id'] for r in results]
        assert returned_ids.count(super_id) == 1, "Super should appear exactly once"
