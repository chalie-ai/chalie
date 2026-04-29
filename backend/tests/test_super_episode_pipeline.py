"""
Feature tests for the super-episode pipeline.

Covers:
  - walk_up_to_apex  (episodic_retrieval_service)
  - find_super_candidates  (episodic_service)
  - Retrieval apex-promotion  (episodic_retrieval_service.retrieve)
  - Super-episode write contract — store_episode() + set_consolidated_into()

Database strategy: the shared `db` fixture from conftest builds a
fully-migrated SQLite database from the real schema.sql + SchemaConvergenceService
and patches get_shared_db_service so every production service sees the same
real database.  No mocks, no fakes.

For tests that require sqlite-vec (find_super_candidates, vector search), a
skip guard is applied when the extension is unavailable.  FTS5-backed retrieval
tests run unconditionally — they exercise the real _fts_search lane and real
apex promotion logic.
"""

import json
import math
import struct
import uuid

import pytest

pytestmark = pytest.mark.unit


# ── Embedding helpers ──────────────────────────────────────────────────────────

# The shared `db` fixture (conftest) runs SchemaConvergenceService with
# embedding_dimensions=256 — episodes_vec is 256-dimensional in test DBs.
_EMB_DIM = 256


def _pack(floats: list[float]) -> bytes:
    """Pack a float list into a sqlite-vec binary blob."""
    return struct.pack(f'{len(floats)}f', *floats)


def _sim_vec(*, angle_deg: float = 0.0) -> list[float]:
    """Return a _EMB_DIM unit vector at ``angle_deg`` from the x-axis (2-plane)."""
    rad = math.radians(angle_deg)
    v = [0.0] * _EMB_DIM
    v[0] = math.cos(rad)
    if _EMB_DIM >= 2:
        v[1] = math.sin(rad)
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _sqlite_vec_available(db) -> bool:
    """Check whether sqlite-vec is loaded (episodes_vec is accessible)."""
    try:
        db.execute("SELECT COUNT(*) FROM episodes_vec")
        return True
    except Exception:
        return False


# ── DB helpers ─────────────────────────────────────────────────────────────────


def _insert_episode(
    db,
    *,
    episode_id: str = None,
    gist: str = "Test gist",
    salience: int = 5,
    channel: str = "test",
    consolidated_into: str = None,
    consolidated_from: list = None,
    deleted_at: str = None,
    transcript_ids: list = None,
    transcript_id_start: int = None,
    transcript_id_end: int = None,
    emotional_valence: float = 0.0,
    emotional_arousal: float = 0.0,
) -> str:
    """Insert a minimal episode row directly via SQL. Returns the episode UUID."""
    eid = episode_id or str(uuid.uuid4())
    db.execute(
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
    db.commit()
    return eid


def _insert_embedding(db, episode_id: str, emb: list[float]) -> None:
    """Insert an embedding blob into episodes_vec for the given episode_id.

    Silently skips if sqlite-vec is not loaded.
    """
    try:
        row = db.execute("SELECT rowid FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        if row:
            db.execute(
                "INSERT OR REPLACE INTO episodes_vec(rowid, embedding) VALUES (?, ?)",
                (row[0], _pack(emb)),
            )
            db.commit()
    except Exception:
        pass  # sqlite-vec not loaded — tests that need it will skip


def _sync_fts(db, episode_id: str, gist: str) -> None:
    """Sync an episode row into the FTS5 external-content index."""
    try:
        row = db.execute("SELECT rowid FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        if row:
            db.execute(
                "INSERT INTO episodes_fts(rowid, gist) VALUES (?, ?)",
                (row[0], gist),
            )
            db.commit()
    except Exception:
        pass


# ── walk_up_to_apex ────────────────────────────────────────────────────────────


class TestWalkUpToApex:
    """Feature tests for episodic_retrieval_service.walk_up_to_apex().

    Uses the shared `db` fixture — no manual DB wiring needed.
    walk_up_to_apex accepts an optional db kwarg; we pass None so it
    resolves the shared singleton (already patched by the db fixture).
    """

    def test_leaf_no_consolidated_into_returns_itself(self, db):
        """A leaf episode with consolidated_into=NULL is its own apex."""
        from services.episodic_retrieval_service import walk_up_to_apex

        eid = _insert_episode(db, gist="leaf gist", channel="ch1")
        result = walk_up_to_apex(eid)
        assert result is not None
        assert result['id'] == eid

    def test_one_hop_returns_super(self, db):
        """leaf → super (one hop) → returns super."""
        from services.episodic_retrieval_service import walk_up_to_apex

        leaf_id = _insert_episode(db, gist="leaf", channel="ch1")
        super_id = _insert_episode(db, gist="super", channel="ch1")
        db.execute(
            "UPDATE episodes SET consolidated_into = ? WHERE id = ?",
            (super_id, leaf_id),
        )
        db.commit()

        result = walk_up_to_apex(leaf_id)
        assert result is not None
        assert result['id'] == super_id

    def test_two_hops_returns_apex(self, db):
        """leaf → super → super-super (two hops) → returns apex."""
        from services.episodic_retrieval_service import walk_up_to_apex

        leaf_id = _insert_episode(db, gist="leaf", channel="ch1")
        super_id = _insert_episode(db, gist="super", channel="ch1")
        apex_id = _insert_episode(db, gist="apex", channel="ch1")

        db.execute("UPDATE episodes SET consolidated_into = ? WHERE id = ?", (super_id, leaf_id))
        db.execute("UPDATE episodes SET consolidated_into = ? WHERE id = ?", (apex_id, super_id))
        db.commit()

        result = walk_up_to_apex(leaf_id)
        assert result is not None
        assert result['id'] == apex_id

    def test_cycle_logs_error_and_returns(self, db, caplog):
        """Cycle (a→b→a) is detected, logs an error, does not hang."""
        import logging
        from services.episodic_retrieval_service import walk_up_to_apex

        a_id = _insert_episode(db, gist="a", channel="ch1")
        b_id = _insert_episode(db, gist="b", channel="ch1")

        db.execute("UPDATE episodes SET consolidated_into = ? WHERE id = ?", (b_id, a_id))
        db.execute("UPDATE episodes SET consolidated_into = ? WHERE id = ?", (a_id, b_id))
        db.commit()

        with caplog.at_level(logging.ERROR, logger='services.episodic_retrieval_service'):
            result = walk_up_to_apex(a_id)

        # Does not raise, does not hang, returns something
        assert result is not None
        assert 'cycle' in caplog.text.lower()

    def test_nonexistent_episode_returns_none(self, db):
        """Walking a non-existent episode ID returns None gracefully."""
        from services.episodic_retrieval_service import walk_up_to_apex

        result = walk_up_to_apex("00000000-0000-0000-0000-000000000000")
        assert result is None


# ── find_super_candidates ──────────────────────────────────────────────────────


class TestFindSuperCandidates:
    """Feature tests for episodic_service.find_super_candidates().

    All tests require sqlite-vec (episodes_vec table) — they skip when unavailable.
    Uses the shared `db` fixture which patches get_shared_db_service globally.
    """

    def _insert_apex(self, db, emb: list[float], *, channel: str = "ch1", gist: str = "test") -> str:
        eid = _insert_episode(db, gist=gist, channel=channel)
        _insert_embedding(db, eid, emb)
        return eid

    def test_empty_channel_returns_empty(self, db):
        """No episodes → empty list."""
        if not _sqlite_vec_available(db):
            pytest.skip("sqlite-vec not available")

        from services.episodic_service import find_super_candidates
        result = find_super_candidates("empty_channel")
        assert result == []

    def test_tight_cluster_returns_one_cluster(self, db):
        """3 embeddings at angle 0° from each other form a tight cluster."""
        if not _sqlite_vec_available(db):
            pytest.skip("sqlite-vec not available")

        from services.episodic_service import find_super_candidates

        base = _sim_vec(angle_deg=0.0)
        e1 = self._insert_apex(db, base, gist="ep1")
        e2 = self._insert_apex(db, base, gist="ep2")
        e3 = self._insert_apex(db, base, gist="ep3")

        clusters = find_super_candidates("ch1")
        assert len(clusters) == 1
        assert sorted(clusters[0]) == sorted([e1, e2, e3])

    def test_loose_embeddings_returns_empty(self, db):
        """3 embeddings far apart produce no cluster."""
        if not _sqlite_vec_available(db):
            pytest.skip("sqlite-vec not available")

        from services.episodic_service import find_super_candidates

        # 60° apart → cosine ≈ 0.5, well below 0.90 threshold
        self._insert_apex(db, _sim_vec(angle_deg=0.0), gist="ep1")
        self._insert_apex(db, _sim_vec(angle_deg=60.0), gist="ep2")
        self._insert_apex(db, _sim_vec(angle_deg=120.0), gist="ep3")

        clusters = find_super_candidates("ch1")
        assert clusters == []

    def test_two_tight_clusters_both_emitted(self, db):
        """Two disjoint tight clusters in the same channel are both emitted with no overlap."""
        if not _sqlite_vec_available(db):
            pytest.skip("sqlite-vec not available")

        from services.episodic_service import find_super_candidates

        # Cluster A: angle 0° (cosine ≈ 1.0 between all three)
        base_a = _sim_vec(angle_deg=0.0)
        a1 = self._insert_apex(db, base_a, gist="a1")
        a2 = self._insert_apex(db, base_a, gist="a2")
        a3 = self._insert_apex(db, base_a, gist="a3")

        # Cluster B: angle 90° (orthogonal to A; within-group cosine = 1.0)
        base_b = _sim_vec(angle_deg=90.0)
        b1 = self._insert_apex(db, base_b, gist="b1")
        b2 = self._insert_apex(db, base_b, gist="b2")
        b3 = self._insert_apex(db, base_b, gist="b3")

        clusters = find_super_candidates("ch1")
        assert len(clusters) == 2
        all_emitted = [eid for cl in clusters for eid in cl]
        assert sorted(all_emitted) == sorted([a1, a2, a3, b1, b2, b3])

    def test_already_consolidated_episodes_excluded(self, db):
        """Episodes with consolidated_into set (i.e. non-apex) are not in the pool."""
        if not _sqlite_vec_available(db):
            pytest.skip("sqlite-vec not available")

        from services.episodic_service import find_super_candidates

        base = _sim_vec(angle_deg=0.0)

        super_id = _insert_episode(db, gist="super", channel="ch1")
        self._insert_apex(db, base, gist="leaf1")
        e2 = _insert_episode(db, gist="leaf2", channel="ch1", consolidated_into=super_id)
        _insert_embedding(db, e2, base)
        e3 = _insert_episode(db, gist="leaf3", channel="ch1", consolidated_into=super_id)
        _insert_embedding(db, e3, base)

        clusters = find_super_candidates("ch1")
        # Only leaf1 and super_id are apex; 2 items < SUPER_EPISODE_MIN_CLUSTER=3 → no cluster
        assert clusters == []

    def test_transitive_chain_forms_one_component(self, db):
        """A-B edge + B-C edge (but NO A-C edge) still forms one component.

        Connected-components clustering: transitive neighbourhoods capture chains
        of related episodes as a single memory group.
        """
        if not _sqlite_vec_available(db):
            pytest.skip("sqlite-vec not available")

        from services.episodic_service import find_super_candidates

        # A–B cosine ≈ 0.966 (15° apart) → above 0.90
        # B–C cosine ≈ 0.966 (B at 15°, C at 30°)
        # A–C cosine ≈ 0.866 (30° apart) → BELOW 0.90
        a = self._insert_apex(db, _sim_vec(angle_deg=0.0), gist="a")
        b = self._insert_apex(db, _sim_vec(angle_deg=15.0), gist="b")
        c = self._insert_apex(db, _sim_vec(angle_deg=30.0), gist="c")

        clusters = find_super_candidates("ch1")
        # Connected components via A–B and B–C edges → single cluster of 3.
        assert len(clusters) == 1
        assert sorted(clusters[0]) == sorted([a, b, c])


# ── Retrieval apex-promotion — real FTS lane, no mocks ─────────────────────────


class TestRetrievalApexPromotion:
    """Feature tests for episodic_retrieval_service.retrieve() apex-promotion.

    Episodes are stored via EpisodicService.store_episode() (which syncs FTS)
    and then retrieved via the real retrieve() function.  The FTS lane finds
    the leaf; apex promotion returns the super.  No mocks.
    """

    @pytest.fixture
    def episodic_svc(self, db):
        from services.database_service import get_shared_db_service
        from services.episodic_service import EpisodicService
        return EpisodicService(get_shared_db_service())

    def test_leaf_hit_promoted_to_super(self, db, episodic_svc):
        """FTS hit on a leaf episode → retrieve() returns its super, not the leaf."""
        from services.episodic_retrieval_service import retrieve

        # Store leaf and super via the real service (syncs FTS)
        leaf_id = episodic_svc.store_episode({
            'gist': 'watering the garden on a sunny afternoon',
            'salience': 5,
            'channel': 'ch-apex',
        })
        super_id = episodic_svc.store_episode({
            'gist': 'outdoor activities super episode',
            'salience': 7,
            'channel': 'ch-apex',
        })

        # Wire leaf → super
        db.execute(
            "UPDATE episodes SET consolidated_into = ? WHERE id = ?",
            (super_id, leaf_id),
        )
        db.commit()

        results = retrieve('watering garden', channel='ch-apex', k=10)
        returned_ids = [r['id'] for r in results]

        assert super_id in returned_ids, (
            f"Expected super_id={super_id!r} in results. "
            f"Apex promotion should replace the leaf hit with its super. "
            f"Got: {returned_ids!r}"
        )
        assert leaf_id not in returned_ids, "Leaf should be replaced by its apex"

    def test_two_hop_chain_returns_apex(self, db, episodic_svc):
        """leaf → super → super-super chain: retrieve() returns the outermost apex."""
        from services.episodic_retrieval_service import retrieve

        leaf_id = episodic_svc.store_episode({
            'gist': 'morning coffee ritual',
            'salience': 5,
            'channel': 'ch-chain',
        })
        super_id = episodic_svc.store_episode({
            'gist': 'daily routines',
            'salience': 6,
            'channel': 'ch-chain',
        })
        apex_id = episodic_svc.store_episode({
            'gist': 'lifestyle patterns',
            'salience': 8,
            'channel': 'ch-chain',
        })

        db.execute("UPDATE episodes SET consolidated_into = ? WHERE id = ?", (super_id, leaf_id))
        db.execute("UPDATE episodes SET consolidated_into = ? WHERE id = ?", (apex_id, super_id))
        db.commit()

        results = retrieve('morning coffee', channel='ch-chain', k=10)
        returned_ids = [r['id'] for r in results]

        assert apex_id in returned_ids, (
            f"Expected apex_id={apex_id!r} in results. Two-hop chain must resolve to apex. "
            f"Got: {returned_ids!r}"
        )
        assert leaf_id not in returned_ids
        assert super_id not in returned_ids

    def test_apex_deduplication(self, db, episodic_svc):
        """Two leaves from the same super emit the super only once."""
        from services.episodic_retrieval_service import retrieve

        leaf1_id = episodic_svc.store_episode({
            'gist': 'reading fiction novels late at night',
            'salience': 5,
            'channel': 'ch-dedup',
        })
        leaf2_id = episodic_svc.store_episode({
            'gist': 'reading science books at night',
            'salience': 5,
            'channel': 'ch-dedup',
        })
        super_id = episodic_svc.store_episode({
            'gist': 'reading habit super episode',
            'salience': 7,
            'channel': 'ch-dedup',
        })

        db.execute("UPDATE episodes SET consolidated_into = ? WHERE id = ?", (super_id, leaf1_id))
        db.execute("UPDATE episodes SET consolidated_into = ? WHERE id = ?", (super_id, leaf2_id))
        db.commit()

        results = retrieve('reading at night', channel='ch-dedup', k=10)
        returned_ids = [r['id'] for r in results]

        assert returned_ids.count(super_id) == 1, (
            f"Super should appear exactly once; got count={returned_ids.count(super_id)!r}"
        )


# ── Super-episode write contract — store_episode() + set_consolidated_into() ───


class TestSuperEpisodeWriteContract:
    """store_episode() must persist consolidated_from exactly as supplied, and
    set_consolidated_into() must persist the back-pointer exactly as supplied.

    Pure mechanical — no LLM, no embedding service, no collaborators beyond DB.
    """

    @pytest.fixture
    def episodic_svc(self, db):
        from services.database_service import get_shared_db_service
        from services.episodic_service import EpisodicService
        return EpisodicService(get_shared_db_service())

    def test_store_episode_persists_consolidated_from_list(self, db, episodic_svc):
        """A super-episode dict with consolidated_from=['a','b','c'] must land
        in the DB as a JSON array containing those three IDs, preserving order."""
        src_ids = ['src-aaa', 'src-bbb', 'src-ccc']
        super_ep = {
            'gist': 'Three sibling episodes unified into one super',
            'salience': 7,
            'channel': 'ch-write-contract',
            'consolidated_from': src_ids,
        }

        new_id = episodic_svc.store_episode(super_ep)

        row = db.execute(
            "SELECT consolidated_from FROM episodes WHERE id = ?", (new_id,)
        ).fetchone()
        assert row is not None, "Super-episode row was not inserted"

        stored = json.loads(row[0])
        assert stored == src_ids, (
            f"consolidated_from round-trip failed: stored={stored!r} expected={src_ids!r}. "
            "This is the exact failure mode reported by nightly 103 "
            "('Super-episodes were created but contained 0 source references')."
        )

    def test_store_episode_empty_consolidated_from_defaults_to_empty_list(
        self, db, episodic_svc,
    ):
        """A plain (leaf) episode with no consolidated_from key must land as
        an empty JSON array, not NULL and not a string literal '[]'.

        Ensures readers that do `json.loads(row['consolidated_from'])` never
        hit a None or a JSON parse error on leaf rows."""
        leaf_ep = {
            'gist': 'A plain leaf episode',
            'salience': 5,
            'channel': 'ch-write-contract',
        }

        new_id = episodic_svc.store_episode(leaf_ep)

        row = db.execute(
            "SELECT consolidated_from FROM episodes WHERE id = ?", (new_id,)
        ).fetchone()
        assert row is not None
        assert row[0] is not None, "consolidated_from must default to '[]', not NULL"
        assert json.loads(row[0]) == [], (
            f"consolidated_from must default to empty list; got {row[0]!r}"
        )

    def test_set_consolidated_into_persists_backpointer(self, db, episodic_svc):
        """set_consolidated_into(leaf_id, super_id) must land the super UUID
        verbatim in the leaf row's consolidated_into column."""
        leaf_id = _insert_episode(db, gist='leaf to be consolidated', channel='ch-bp')
        super_id = 'super-xyz-123'

        # Sanity: initially NULL
        row = db.execute(
            "SELECT consolidated_into FROM episodes WHERE id = ?", (leaf_id,)
        ).fetchone()
        assert row[0] is None, "leaf must start with consolidated_into=NULL"

        episodic_svc.set_consolidated_into(leaf_id, super_id)

        row = db.execute(
            "SELECT consolidated_into FROM episodes WHERE id = ?", (leaf_id,)
        ).fetchone()
        assert row[0] == super_id, (
            f"set_consolidated_into failed to persist back-pointer: "
            f"expected {super_id!r}, got {row[0]!r}. Apex-traversal in retrieve() "
            "relies on this column — a NULL here means the super is unreachable "
            "from its leaves."
        )

    def test_set_consolidated_into_applied_to_all_sources_in_cluster(
        self, db, episodic_svc,
    ):
        """End-to-end contract of the loop in SuperEpisodeEncoderProcessor.send():

            for src_id in cluster_ids:
                episodic_svc.set_consolidated_into(src_id, new_id)

        After the loop, every source leaf must point back to the super.
        The unrelated leaf must NOT gain a back-pointer."""
        src_a = _insert_episode(db, gist='src a', channel='ch-cluster')
        src_b = _insert_episode(db, gist='src b', channel='ch-cluster')
        src_c = _insert_episode(db, gist='src c', channel='ch-cluster')
        unrelated = _insert_episode(db, gist='unrelated', channel='ch-cluster')

        super_ep = {
            'gist': 'Super of a, b, c',
            'salience': 7,
            'channel': 'ch-cluster',
            'consolidated_from': [src_a, src_b, src_c],
        }
        new_id = episodic_svc.store_episode(super_ep)

        for src_id in [src_a, src_b, src_c]:
            episodic_svc.set_consolidated_into(src_id, new_id)

        # Every clustered source now points back to the super
        rows = db.execute(
            "SELECT id, consolidated_into FROM episodes WHERE id IN (?, ?, ?)",
            (src_a, src_b, src_c),
        ).fetchall()
        assert len(rows) == 3
        for eid, ci in rows:
            assert ci == new_id, f"source {eid} back-pointer {ci!r} != {new_id!r}"

        # The unrelated leaf is untouched
        row = db.execute(
            "SELECT consolidated_into FROM episodes WHERE id = ?", (unrelated,)
        ).fetchone()
        assert row[0] is None, "Back-pointer leaked onto a non-cluster leaf; scoping broken."

        # The super itself is an apex and carries the source IDs in consolidated_from
        row = db.execute(
            "SELECT consolidated_into, consolidated_from FROM episodes WHERE id = ?",
            (new_id,),
        ).fetchone()
        assert row[0] is None, "super must be an apex (consolidated_into=NULL)"
        assert sorted(json.loads(row[1])) == sorted([src_a, src_b, src_c])
