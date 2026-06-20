"""
Feature tests for the super-episode (memory hierarchy) pipeline.

Covers the count-triggered, density-clustered roll-up:

  - walk_up_to_apex                         (episodic_retrieval_service)
  - find_super_candidates count trigger     (episodic_service)
  - consolidation write contract            (subconscious_worker._step_consolidate)
      level=1 super stamping · level=2 era digests · tombstone marker on children
  - collapsed-tree retrieval                (episodic_retrieval_service.retrieve)
  - new clustering dependency resolution    (sklearn / scipy / umap / numba / llvmlite)

Database strategy: the shared ``db`` fixture (conftest) builds a fully-migrated
SQLite database from the real schema.sql + SchemaConvergenceService (256-dim
episodes_vec) and patches get_shared_db_service so every production service sees
the same real database. No mocks of in-process production code.

The ONLY sanctioned seam is the LLM network boundary (Pattern H): the
super-episode encoder makes one provider call per cluster. We swap
``Providers._resolve`` for the duration of a block with a *real*
``ProviderClient`` subclass via a plain try/finally context manager — never
``unittest.mock``. Everything else runs real: ``find_super_candidates``'s
UMAP→HDBSCAN clustering, the full ``SubconsciousWorker._step_consolidate`` →
``_consolidate_channel`` → ``_write_super_episode`` orchestration,
``MessageProcessor.process`` prompt assembly + ACT loop, ``store_episode``,
``set_consolidated_into``, the level stamp and the tombstone write all execute.

Tests that need sqlite-vec (any clustering / roll-up test) skip when the vec
extension is unavailable. FTS5-backed retrieval tests run unconditionally.

═══════════════════════════════════════════════════════════════════════════════
RED-FIRST NOTE (read before "fixing" a failure)
═══════════════════════════════════════════════════════════════════════════════
Written ahead of the coder. The count-trigger / clustering / level / tombstone
tests are RED on arrival because the replaced algorithm does not yet exist:

  * find_super_candidates still fires at SUPER_EPISODE_MIN_CLUSTER=3 with a
    0.90-cosine connected-components pass — it has no >=50-apex count trigger, so
    it returns clusters below the trigger (the count-trigger tests expect []).
  * the roll-up write never stamps episodes.level (schema default 0), so every
    super lands at level 0 — the level==1 / level==2 assertions are the proof the
    latent bug is fixed.
  * the roll-up write never sets tombstoned_at on consolidated children — the
    tombstone-marker assertion is RED until _write_super_episode writes it.
  * the level-2 "era" recursion does not exist yet.

The dependency-import smoke test is GREEN on arrival (the wheels are installed in
the gate environment) and stands as a regression lock that the new prod deps
resolve. The KEEP-classes (walk_up_to_apex, collapsed-tree retrieval, the
write-contract round-trip) are algorithm-agnostic and stay GREEN.
"""

from __future__ import annotations

import contextlib
import json
import math
import sqlite3
import struct
import uuid
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, cast

import pytest

from services.llm_clients.base import ProviderClient
from services.provider_api import ProviderApiRequest, ProviderApiResponse
from services.providers import Providers
from services.time_utils import parse_utc

if TYPE_CHECKING:
    from services.subconscious_worker import SubconsciousWorker
    from services.database_service import DatabaseService

pytestmark = pytest.mark.unit


# ── Embedding helpers ──────────────────────────────────────────────────────────

# The shared `db` fixture (conftest) runs SchemaConvergenceService with
# embedding_dimensions=256 — episodes_vec is 256-dimensional in test DBs.
_EMB_DIM = 256


def _pack(floats: list[float]) -> bytes:
    return struct.pack(f'{len(floats)}f', *floats)


def _sim_vec(*, angle_deg: float = 0.0) -> list[float]:
    rad = math.radians(angle_deg)
    v = [0.0] * _EMB_DIM
    v[0] = math.cos(rad)
    if _EMB_DIM >= 2:
        v[1] = math.sin(rad)
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _topic_vec(axis: int, *, jitter_axis: int, jitter: float = 0.0) -> list[float]:
    """An L2-normalised _EMB_DIM vector dominated by one ``axis`` with a small
    component on a second ``jitter_axis``.

    Identical-axis vectors are mutually cosine ~1.0 (one tight topic); distinct
    primary axes are mutually orthogonal (cosine 0.0 — separated topics). The
    per-leaf jitter gives a topic a little intra-cluster spread so a density
    clusterer sees real points instead of one degenerate stacked point.
    """
    v = [0.0] * _EMB_DIM
    v[axis] = 1.0
    if jitter:
        v[jitter_axis] = jitter
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _sqlite_vec_available(db: sqlite3.Connection) -> bool:
    """Check whether sqlite-vec is loaded (episodes_vec is accessible)."""
    try:
        db.execute("SELECT COUNT(*) FROM episodes_vec")
        return True
    except Exception:
        return False


# ── DB helpers ─────────────────────────────────────────────────────────────────


def _insert_episode(
    db: sqlite3.Connection,
    *,
    episode_id: str | None = None,
    gist: str = "Test gist",
    salience: int = 5,
    channel: str = "test",
    level: int = 0,
    consolidated_into: str | None = None,
    consolidated_from: list[str] | None = None,
    deleted_at: str | None = None,
    transcript_ids: list[int] | None = None,
    transcript_id_start: int | None = None,
    transcript_id_end: int | None = None,
    emotional_valence: float = 0.0,
    emotional_arousal: float = 0.0,
) -> str:
    """Insert a minimal episode row directly via SQL. Returns the episode UUID."""
    eid = episode_id or str(uuid.uuid4())
    db.execute(
        """
        INSERT INTO episodes (
            id, gist, salience, channel, level,
            transcript_ids, transcript_id_start, transcript_id_end,
            emotional_valence, emotional_arousal,
            consolidated_from, consolidated_into, deleted_at,
            storage_strength, retrieval_weight
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, 1.0)
        """,
        (
            eid, gist, salience, channel, level,
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


def _insert_embedding(db: sqlite3.Connection, episode_id: str, emb: list[float]) -> None:
    """Insert an embedding blob into episodes_vec for the given episode_id.

    Raises if sqlite-vec is not loaded — callers gate on _sqlite_vec_available.
    """
    row = db.execute("SELECT rowid FROM episodes WHERE id = ?", (episode_id,)).fetchone()
    if row:
        db.execute(
            "INSERT OR REPLACE INTO episodes_vec(rowid, embedding) VALUES (?, ?)",
            (row[0], _pack(emb)),
        )
        db.commit()


# ── LLM network-boundary seam (Pattern H — the ONLY sanctioned mock) ───────────
# Mirrors tests/test_tkt926_per_source_profiles.py. A real ProviderClient
# subclass swapped at the class level via try/finally — no unittest.mock — so the
# full ACT loop, prompt assembly and the whole _write_super_episode orchestration
# run real; only the network call to the model is replaced.

class _FakeLLMService(ProviderClient):
    """Stand-in for the resolved provider client. Real ProviderClient subclass so
    Providers.send still runs prompt assembly, tool building and pre-flight."""

    CONTENT_FIELD_LABEL = "message.content"

    def __init__(self, send_fn: Callable[[ProviderApiRequest], ProviderApiResponse]) -> None:
        self._send_fn = send_fn

    def get_context_limit(self) -> int:
        return 200_000

    def estimate_request_tokens(self, dto: ProviderApiRequest) -> int:
        return 1

    def send(self, dto: ProviderApiRequest) -> ProviderApiResponse:
        return self._send_fn(dto)


@contextlib.contextmanager
def _inject_fake_client(
    send_fn: Callable[[ProviderApiRequest], ProviderApiResponse],
) -> Generator[None, None, None]:
    """Swap Providers._resolve at the class level for the block. No unittest.mock."""
    original = Providers._resolve
    setattr(Providers, '_resolve', lambda self, *_a, **_kw: _FakeLLMService(send_fn))
    try:
        yield
    finally:
        setattr(Providers, '_resolve', original)


def _super_ep_envelope(gist: str) -> ProviderApiResponse:
    """The SuperEpisodeEncoder's expected JSON object envelope. The worker
    overwrites 'channel'/'consolidated_from'/'salience' itself; the model owns
    only the gist + affect. The response carries no tool calls, so this ends the
    ACT loop."""
    return ProviderApiResponse(
        text=json.dumps({
            "gist": gist,
            "emotional_valence": 0.0,
            "emotional_arousal": 0.0,
            "has_open_loop": False,
        }),
        model="test-model",
        provider="mock",
        tool_calls=None,
    )


def _worker() -> SubconsciousWorker:
    from services.subconscious_worker import SubconsciousWorker
    return SubconsciousWorker(tick_sec=10, idle_window_sec=60)


def _db_service() -> DatabaseService:
    from services.database_service import get_shared_db_service
    return get_shared_db_service()


# ── Roll-up DB inspection helpers ──────────────────────────────────────────────


def _super_episodes(db: sqlite3.Connection, channel: str) -> list[sqlite3.Row]:
    """Super/era rows (consolidated_from non-empty) currently live in a channel."""
    return db.execute(
        "SELECT id, gist, consolidated_from, level FROM episodes "
        "WHERE channel=? AND deleted_at IS NULL "
        "  AND consolidated_from IS NOT NULL AND consolidated_from != '[]'",
        (channel,),
    ).fetchall()


def _apex_leaf_count(db: sqlite3.Connection, channel: str, *, level: int = 0) -> int:
    """Count apex leaves (consolidated_into IS NULL, not deleted) at a level."""
    return cast(int, db.execute(
        "SELECT COUNT(*) FROM episodes "
        "WHERE channel=? AND consolidated_into IS NULL AND deleted_at IS NULL "
        "  AND level=?",
        (channel, level),
    ).fetchone()[0])


def _seed_apex_cluster(
    db: sqlite3.Connection,
    channel: str,
    *,
    count: int,
    axis: int,
    jitter_axis: int,
    level: int = 0,
    gist_prefix: str = "shard",
    salience: int = 7,
) -> list[str]:
    """Seed ``count`` apex leaves on one tight topic (shared primary ``axis``,
    small per-leaf jitter). Returns the list of episode ids."""
    ids = []
    for i in range(count):
        eid = _insert_episode(
            db, gist=f"{channel} {gist_prefix} {axis}-{i}",
            channel=channel, salience=salience, level=level,
        )
        _insert_embedding(
            db, eid,
            _topic_vec(axis, jitter_axis=jitter_axis, jitter=0.05 * ((i % 5) + 1)),
        )
        ids.append(eid)
    return ids


# ══════════════════════════════════════════════════════════════════════════════
# KEEP — walk_up_to_apex (algorithm-agnostic back-pointer traversal)
# ══════════════════════════════════════════════════════════════════════════════


class TestWalkUpToApex:
    """Feature tests for episodic_retrieval_service.walk_up_to_apex().

    walk_up_to_apex accepts an optional db kwarg; we pass None so it resolves the
    shared singleton (already patched by the db fixture).
    """

    def test_leaf_no_consolidated_into_returns_itself(self, db: sqlite3.Connection) -> None:
        """A leaf episode with consolidated_into=NULL is its own apex."""
        from services.episodic_retrieval_service import walk_up_to_apex

        eid = _insert_episode(db, gist="leaf gist", channel="ch1")
        result = walk_up_to_apex(eid)
        assert result is not None
        assert result['id'] == eid

    def test_one_hop_returns_super(self, db: sqlite3.Connection) -> None:
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

    def test_two_hops_returns_apex(self, db: sqlite3.Connection) -> None:
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

    def test_cycle_logs_error_and_returns(self, db: sqlite3.Connection, caplog: pytest.LogCaptureFixture) -> None:
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


# ══════════════════════════════════════════════════════════════════════════════
# NEW (RED) — count trigger + UMAP/HDBSCAN clustering + level/tombstone roll-up
# ══════════════════════════════════════════════════════════════════════════════


class TestCountTriggeredRollup:
    """The replacement consolidation: a count trigger (>=50 apex leaves per
    channel) drives UMAP→HDBSCAN clustering; clusters roll up into level-1
    super-episodes (children tombstoned + back-pointed); level-1 nodes roll up
    recursively into level-2 era digests; HDBSCAN noise stays leaf apexes.

    Drives the REAL production hot path: find_super_candidates (clustering) and
    SubconsciousWorker._step_consolidate (orchestration → _write_super_episode).
    The only seam is the LLM network boundary (Pattern H).
    """

    def test_count_trigger_does_not_fire_below_fifty_apexes(self, db: sqlite3.Connection) -> None:
        """Below the 50-apex count trigger NO consolidation happens — even with
        49 near-identical (cosine ~1.0) apex leaves on an allowlisted channel.

        Acceptance criterion (build §2 AC-1): the count trigger fires at >=50
        apexes, NOT on a small tight cluster. Asserted against the literal 50,
        not a production constant, so a constant typo cannot make the test agree
        with the bug. The replaced 0.90/min-3 algorithm fires here, so this is
        RED until the count trigger lands.

        Two observable proofs: (1) the clustering entry point returns no
        candidates below the trigger; (2) driving the real consolidation tick
        writes zero super-episodes.
        """
        if not _sqlite_vec_available(db):
            pytest.skip("sqlite-vec not available")

        from services.episodic_service import find_super_candidates

        seeded = _seed_apex_cluster(
            db, "user", count=49, axis=0, jitter_axis=200, gist_prefix="below",
        )
        assert len(seeded) == 49
        # AC-1: literal 50-apex trigger — 49 < 50 must NOT fire.
        assert _apex_leaf_count(db, "user") == 49

        assert find_super_candidates("user") == [], (
            "find_super_candidates fired below the 50-apex count trigger — it "
            "must NO-OP (return []) until a channel reaches >=50 apex leaves"
        )

        def _send(_dto: ProviderApiRequest) -> ProviderApiResponse:
            return _super_ep_envelope("should never be summarised below trigger")

        with _inject_fake_client(_send):
            _worker()._step_consolidate()

        assert _super_episodes(db, "user") == [], (
            "a super-episode was written below the 50-apex count trigger; the "
            "count trigger must gate the whole roll-up, not just the clusterer"
        )
        # No leaf was consolidated — all 49 remain apex.
        assert _apex_leaf_count(db, "user") == 49

    def test_count_trigger_fires_at_fifty_and_stamps_level_one(self, db: sqlite3.Connection) -> None:
        """At >=50 apex leaves the count trigger fires: UMAP→HDBSCAN clusters the
        leaves, the worker writes super-episode parents, the children back-point
        at their parent, and each parent is stamped level=1.

        Seeds 50 apex leaves spanning two well-separated topics (orthogonal
        primary axes) so density clustering finds coherent groups rather than one
        blob. RED today on two counts: (a) the count trigger doesn't exist, and
        (b) episodes.level is never written, so a parent lands at the schema
        default 0 instead of 1 (the latent-bug fix this ticket proves).
        """
        if not _sqlite_vec_available(db):
            pytest.skip("sqlite-vec not available")

        from services.episodic_service import find_super_candidates

        topic_a = _seed_apex_cluster(db, "user", count=25, axis=0, jitter_axis=200,
                                     gist_prefix="alpha")
        topic_b = _seed_apex_cluster(db, "user", count=25, axis=1, jitter_axis=201,
                                     gist_prefix="beta")
        assert _apex_leaf_count(db, "user") == 50  # AC-1: at the literal trigger

        # The clustering entry point yields >=1 candidate group at the trigger.
        groups = find_super_candidates("user")
        assert len(groups) >= 1, (
            "find_super_candidates returned no clusters at 50 apex leaves — the "
            "count-triggered UMAP→HDBSCAN pass must emit candidate groups"
        )

        def _send(_dto: ProviderApiRequest) -> ProviderApiResponse:
            return _super_ep_envelope("Consolidated reflection of a topic.")

        with _inject_fake_client(_send):
            _worker()._step_consolidate()

        supers = _super_episodes(db, "user")
        assert len(supers) >= 1, (
            "no super-episode parent was written at >=50 apex leaves"
        )

        seeded = set(topic_a) | set(topic_b)
        for _sid, _gist, cfrom_json, level in supers:
            assert level == 1, (
                f"super-episode parent must be stamped level=1, got {level!r}. "
                "episodes.level is never written today — this is the latent bug."
            )
            children = json.loads(cfrom_json)
            assert children, "super-episode carries an empty consolidated_from"
            # Every referenced child is one of the seeded leaves and now points
            # back at this parent.
            for child_id in children:
                assert child_id in seeded
            rows = db.execute(
                f"SELECT consolidated_into FROM episodes WHERE id IN "
                f"({','.join('?' * len(children))})",
                children,
            ).fetchall()
            assert all(r[0] == _sid for r in rows), (
                "consolidated children must back-point at their super-episode"
            )

    def test_clusters_are_topically_pure_and_noise_stays_leaf(self, db: sqlite3.Connection) -> None:
        """HDBSCAN clusters are topically coherent and genuine outliers are left
        as leaf apexes — never force-assigned into a cluster.

        Seeds two coherent topics (25 each, orthogonal axes) plus a handful of
        off-topic outliers each on its own far-apart axis. Acceptance criterion
        (build §2 AC-2): topically-pure clusters with noise left as leaves.

        Purity is asserted implementation-robustly: each super-episode's source
        set is a SUBSET of exactly one seeded topic (no parent mixes topics), and
        the outliers keep consolidated_into IS NULL (no parent created for them).
        RED until the density-clustering roll-up with noise handling lands.
        """
        if not _sqlite_vec_available(db):
            pytest.skip("sqlite-vec not available")

        topic_a = set(_seed_apex_cluster(db, "user", count=25, axis=0,
                                         jitter_axis=200, gist_prefix="aaa"))
        topic_b = set(_seed_apex_cluster(db, "user", count=25, axis=1,
                                         jitter_axis=201, gist_prefix="bbb"))

        # Genuine outliers: each on a distinct, isolated axis so no two are dense
        # enough to form a cluster of their own (HDBSCAN min_cluster_size).
        outliers = []
        for k, axis in enumerate((50, 60, 70)):
            oid = _insert_episode(db, gist=f"user outlier {axis}", channel="user",
                                  salience=7)
            _insert_embedding(db, oid, _topic_vec(axis, jitter_axis=axis + 1,
                                                  jitter=0.01))
            outliers.append(oid)

        assert _apex_leaf_count(db, "user") == 53

        def _send(_dto: ProviderApiRequest) -> ProviderApiResponse:
            return _super_ep_envelope("Consolidated reflection of one topic.")

        with _inject_fake_client(_send):
            _worker()._step_consolidate()

        supers = _super_episodes(db, "user")
        assert supers, "no super-episode formed at >=50 apex leaves"

        for _sid, _gist, cfrom_json, level in supers:
            assert level == 1, (
                f"a topically-pure roll-up must still land at level=1 (it decays "
                f"at the wrong tau otherwise), got {level!r}"
            )
            children = set(json.loads(cfrom_json))
            in_a = children & topic_a
            in_b = children & topic_b
            assert not (in_a and in_b), (
                f"super-episode mixed two distinct topics — clustering is not "
                f"topically pure. topic_a∩={len(in_a)} topic_b∩={len(in_b)}"
            )
            assert children <= topic_a or children <= topic_b, (
                "super-episode consolidated a non-topic (outlier) leaf"
            )

        # Noise points are NOT consolidated — they stay leaf apexes.
        for oid in outliers:
            row = db.execute(
                "SELECT consolidated_into, level FROM episodes WHERE id = ?",
                (oid,),
            ).fetchone()
            assert row[0] is None, (
                f"outlier {oid} was force-assigned into a cluster — HDBSCAN noise "
                "(label -1) must remain a leaf apex"
            )
            assert row[1] == 0, "outlier must remain a level-0 leaf"

    def test_recursive_era_digest_forms_at_level_two(self, db: sqlite3.Connection) -> None:
        """When >=25 level-1 nodes accumulate they roll up recursively into a
        level-2 "era" digest whose level-1 children are tombstoned + back-pointed.

        Acceptance criterion (build §2 AC-3): summaries land at level=1, eras at
        level=2. Seeds 25 pre-existing level-1 apex nodes (representing supers
        rolled up on earlier ticks) on one tight topic, then drives the real
        consolidation tick. RED until the level-2 era recursion exists.
        """
        if not _sqlite_vec_available(db):
            pytest.skip("sqlite-vec not available")

        level1_ids = set(_seed_apex_cluster(
            db, "user", count=25, axis=0, jitter_axis=200, level=1,
            gist_prefix="era-source",
        ))
        assert _apex_leaf_count(db, "user", level=1) == 25

        def _send(_dto: ProviderApiRequest) -> ProviderApiResponse:
            return _super_ep_envelope("Era digest of a season of reflections.")

        with _inject_fake_client(_send):
            _worker()._step_consolidate()

        eras = db.execute(
            "SELECT id, consolidated_from FROM episodes "
            "WHERE channel='user' AND level=2 AND deleted_at IS NULL "
            "  AND consolidated_from IS NOT NULL AND consolidated_from != '[]'",
        ).fetchall()
        assert eras, (
            "no level-2 era digest formed from >=25 level-1 nodes — the recursive "
            "roll-up does not exist yet"
        )

        for era_id, cfrom_json in eras:
            children = json.loads(cfrom_json)
            assert children, "era digest carries empty consolidated_from"
            for child_id in children:
                assert child_id in level1_ids, (
                    "era digest must consolidate level-1 nodes, not raw leaves"
                )
                row = db.execute(
                    "SELECT consolidated_into, tombstoned_at FROM episodes WHERE id = ?",
                    (child_id,),
                ).fetchone()
                assert row[0] == era_id, "level-1 child must back-point at its era"
                assert row[1] is not None, (
                    "level-1 child rolled into an era must be tombstoned"
                )

    def test_rollup_children_are_tombstoned_with_tz_aware_utc(self, db: sqlite3.Connection) -> None:
        """Every leaf consolidated into a super-episode is marked with a
        timezone-aware UTC ``tombstoned_at`` alongside its ``consolidated_into``
        back-pointer.

        Acceptance criterion (build §2 AC-4): children tombstoned on roll-up. The
        actual >30d hard-delete is the decay engine's job (a separate ticket) —
        this test asserts only the marker and the back-pointer. RED until
        _write_super_episode writes tombstoned_at (today it writes only the
        back-pointer, so tombstoned_at stays NULL).
        """
        if not _sqlite_vec_available(db):
            pytest.skip("sqlite-vec not available")

        topic_a = set(_seed_apex_cluster(db, "user", count=25, axis=0,
                                         jitter_axis=200, gist_prefix="tomb-a"))
        topic_b = set(_seed_apex_cluster(db, "user", count=25, axis=1,
                                         jitter_axis=201, gist_prefix="tomb-b"))
        assert _apex_leaf_count(db, "user") == 50

        def _send(_dto: ProviderApiRequest) -> ProviderApiResponse:
            return _super_ep_envelope("Consolidated reflection of a topic.")

        with _inject_fake_client(_send):
            _worker()._step_consolidate()

        consolidated = db.execute(
            "SELECT id, consolidated_into, tombstoned_at FROM episodes "
            "WHERE channel='user' AND consolidated_into IS NOT NULL",
        ).fetchall()
        assert consolidated, "no leaf was consolidated at >=50 apex leaves"

        seeded = topic_a | topic_b
        for child_id, into, tombstoned_at in consolidated:
            assert child_id in seeded
            assert into is not None, "consolidated child lost its back-pointer"
            assert tombstoned_at is not None, (
                f"consolidated child {child_id} has no tombstoned_at — roll-up "
                "must mark children for the decay engine's scheduled delete"
            )
            dt = parse_utc(tombstoned_at)
            assert dt.tzinfo is not None, (
                f"tombstoned_at must be timezone-aware UTC, got {tombstoned_at!r}"
            )


def test_clustering_dependencies_import() -> None:
    """The new production clustering stack resolves in the runtime environment.

    Acceptance criterion (build §2 AC-5): scikit-learn / scipy / umap-learn /
    numba / llvmlite install AND import cleanly, and HDBSCAN is the sklearn
    built-in the rewrite targets. GREEN on arrival (wheels installed) — a
    regression lock that a dropped or broken dep is caught at the unit gate
    rather than at runtime in the consolidation worker.
    """
    import importlib

    for mod in ("sklearn", "scipy", "numba", "llvmlite", "umap"):
        importlib.import_module(mod)

    from sklearn.cluster import HDBSCAN  # noqa: F401 — resolution is the assertion


# ══════════════════════════════════════════════════════════════════════════════
# KEEP — collapsed-tree retrieval (TKT-923 contract, algorithm-agnostic)
# ══════════════════════════════════════════════════════════════════════════════


class TestConsolidatedLeafSurfacesAsItself:
    """Feature tests for episodic_retrieval_service.retrieve() under the
    collapsed-tree contract.

    Episodes are stored via EpisodicService.store_episode() (which syncs FTS) and
    retrieved via the real retrieve(). The FTS lane finds the matching row, and
    retrieval hydrates that row AS-IS at its own level — the old apex-promotion
    that replaced a leaf hit with its super is gone. No mocks.

    This class pins what is unique to the super-episode pipeline: a leaf that
    already carries a consolidated_into back-pointer is no longer swallowed by its
    apex — when the query matches only the leaf, the leaf surfaces and the
    unrelated super does not.
    """

    @pytest.fixture
    def episodic_svc(self, db: sqlite3.Connection) -> object:
        from services.database_service import get_shared_db_service
        from services.episodic_service import EpisodicService
        return EpisodicService(get_shared_db_service())

    def test_consolidated_leaf_surfaces_at_its_own_level(self, db: sqlite3.Connection, episodic_svc: object) -> None:
        """A leaf with consolidated_into set still surfaces as ITSELF when the
        query matches the leaf and NOT its super."""
        from services.episodic_retrieval_service import retrieve
        from services.episodic_service import EpisodicService
        from typing import cast as tcast

        svc = tcast(EpisodicService, episodic_svc)

        leaf_id = svc.store_episode({
            'gist': 'watering the garden on a sunny afternoon',
            'salience': 5,
            'channel': 'ch-collapsed',
        })
        super_id = svc.store_episode({
            'gist': 'quarterly budget spreadsheet review meeting',
            'salience': 7,
            'channel': 'ch-collapsed',
        })

        svc.set_consolidated_into(leaf_id, super_id)

        raw = retrieve('watering garden', channel='ch-collapsed', k=10)
        results = tcast(list[dict[str, object]], raw)
        returned_ids = [r['id'] for r in results]

        assert leaf_id in returned_ids, (
            f"Consolidated leaf was swallowed by its apex. Under collapsed-tree "
            f"retrieval a leaf surfaces at its own level even when "
            f"consolidated_into is set. Got: {returned_ids!r}"
        )
        assert super_id not in returned_ids, (
            f"The unrelated super was promoted in despite not matching the query "
            f"— apex-promotion should be gone. Got: {returned_ids!r}"
        )

    def test_consolidated_leaf_appears_once(self, db: sqlite3.Connection, episodic_svc: object) -> None:
        """A single matching consolidated leaf is returned exactly once — the
        collapsed-tree hydration de-duplicates by id across the FTS and vector
        lanes."""
        from services.episodic_retrieval_service import retrieve
        from services.episodic_service import EpisodicService
        from typing import cast as tcast

        svc = tcast(EpisodicService, episodic_svc)

        leaf_id = svc.store_episode({
            'gist': 'reading fiction novels late at night',
            'salience': 5,
            'channel': 'ch-collapsed-dedup',
        })
        super_id = svc.store_episode({
            'gist': 'reading habit super episode',
            'salience': 7,
            'channel': 'ch-collapsed-dedup',
        })

        svc.set_consolidated_into(leaf_id, super_id)

        raw = retrieve('reading fiction at night', channel='ch-collapsed-dedup', k=10)
        results = tcast(list[dict[str, object]], raw)
        returned_ids = [r['id'] for r in results]

        assert returned_ids.count(leaf_id) == 1, (
            f"Matching leaf should appear exactly once; "
            f"got count={returned_ids.count(leaf_id)!r} in {returned_ids!r}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# KEEP (EXTENDED) — super-episode write contract + level/tombstone roll-up effects
# ══════════════════════════════════════════════════════════════════════════════


class TestSuperEpisodeWriteContract:
    """store_episode() must persist consolidated_from exactly as supplied, and the
    roll-up must back-point + level-stamp + tombstone every source child.

    The round-trip tests are pure DB mechanics. The roll-up effects test drives
    the real consolidation tick (LLM seam only) so it captures the level=1 stamp
    and tombstone marker on the production hot path.
    """

    @pytest.fixture
    def episodic_svc(self, db: sqlite3.Connection) -> object:
        from services.database_service import get_shared_db_service
        from services.episodic_service import EpisodicService
        return EpisodicService(get_shared_db_service())

    def test_store_episode_persists_consolidated_from_list(self, db: sqlite3.Connection, episodic_svc: object) -> None:
        """A super-episode dict with consolidated_from=['a','b','c'] must land in
        the DB as a JSON array containing those three IDs, preserving order."""
        from services.episodic_service import EpisodicService
        from typing import cast as tcast

        svc = tcast(EpisodicService, episodic_svc)
        src_ids = ['src-aaa', 'src-bbb', 'src-ccc']
        super_ep: dict[str, object] = {
            'gist': 'Three sibling episodes unified into one super',
            'salience': 7,
            'channel': 'ch-write-contract',
            'consolidated_from': src_ids,
        }

        new_id = svc.store_episode(super_ep)

        row = db.execute(
            "SELECT consolidated_from FROM episodes WHERE id = ?", (new_id,)
        ).fetchone()
        assert row is not None, "Super-episode row was not inserted"

        stored = json.loads(row[0])
        assert stored == src_ids, (
            f"consolidated_from round-trip failed: stored={stored!r} "
            f"expected={src_ids!r}."
        )

    def test_set_consolidated_into_persists_backpointer(self, db: sqlite3.Connection, episodic_svc: object) -> None:
        """set_consolidated_into(leaf_id, super_id) must land the super UUID
        verbatim in the leaf row's consolidated_into column."""
        from services.episodic_service import EpisodicService
        from typing import cast as tcast

        svc = tcast(EpisodicService, episodic_svc)
        leaf_id = _insert_episode(db, gist='leaf to be consolidated', channel='ch-bp')
        super_id = 'super-xyz-123'

        row = db.execute(
            "SELECT consolidated_into FROM episodes WHERE id = ?", (leaf_id,)
        ).fetchone()
        assert row[0] is None, "leaf must start with consolidated_into=NULL"

        svc.set_consolidated_into(leaf_id, super_id)

        row = db.execute(
            "SELECT consolidated_into FROM episodes WHERE id = ?", (leaf_id,)
        ).fetchone()
        assert row[0] == super_id, (
            f"set_consolidated_into failed to persist back-pointer: "
            f"expected {super_id!r}, got {row[0]!r}."
        )

    def test_rollup_parent_is_level_one_and_children_tombstoned(self, db: sqlite3.Connection) -> None:
        """End-to-end write contract of one roll-up: the new super-episode parent
        carries level=1 and every source child gets BOTH consolidated_into and a
        tombstoned_at marker.

        Drives the real SubconsciousWorker consolidation tick (LLM seam only) over
        a >=50-apex channel. RED until the count trigger fires AND
        _write_super_episode stamps level + tombstone (today the parent lands at
        the schema default level 0 and tombstoned_at stays NULL)."""
        if not _sqlite_vec_available(db):
            pytest.skip("sqlite-vec not available")

        topic_a = set(_seed_apex_cluster(db, "user", count=25, axis=0,
                                         jitter_axis=200, gist_prefix="wc-a"))
        topic_b = set(_seed_apex_cluster(db, "user", count=25, axis=1,
                                         jitter_axis=201, gist_prefix="wc-b"))
        seeded = topic_a | topic_b
        assert _apex_leaf_count(db, "user") == 50

        def _send(_dto: ProviderApiRequest) -> ProviderApiResponse:
            return _super_ep_envelope("Consolidated reflection of a topic.")

        with _inject_fake_client(_send):
            _worker()._step_consolidate()

        supers = _super_episodes(db, "user")
        assert supers, "no super-episode written at >=50 apex leaves"

        for sid, _gist, cfrom_json, level in supers:
            assert level == 1, (
                f"roll-up parent must be level=1, got {level!r}"
            )
            for child_id in json.loads(cfrom_json):
                assert child_id in seeded
                row = db.execute(
                    "SELECT consolidated_into, tombstoned_at FROM episodes WHERE id = ?",
                    (child_id,),
                ).fetchone()
                assert row[0] == sid, "child must back-point at the super-episode"
                assert row[1] is not None, (
                    "child rolled into a super must carry a tombstoned_at marker"
                )
