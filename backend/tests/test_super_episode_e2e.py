"""
End-to-end integration test for super-episode creation.

Exercises the entire ``_maybe_trigger_super_episode`` pipeline against real
services — no mocking of ANY collaborator:

    find_super_candidates      → real sqlite-vec cosine search
    _fetch_transcript_spans    → real transcript table read
    SuperEpisodeEncoderProcessor.send()
                               → real LLM provider (configured in test env)
    EmbeddingService.generate_embedding()
                               → real ONNX encoder
    compute_novelty            → real vector math against prior episodes
    compute_salience           → real formula
    EpisodicService.store_episode()
                               → real INSERT into episodes + episodes_vec + FTS
    EpisodicService.set_consolidated_into()
                               → real UPDATE back-pointer

What is asserted:
  * A new super-episode row lands in the DB
  * ``consolidated_from`` on the super contains exactly the 3 source IDs
  * Each source leaf has ``consolidated_into`` set to the super's ID
  * The super itself is an apex (``consolidated_into IS NULL``)

This is the assertion the nightly 103 judge hallucinated as "empty source
references" — here we lock the invariant in a deterministic harness so a
real regression would fail fast locally, not after an hour-long nightly run.

Marked ``integration`` because it requires:
  * sqlite-vec loaded (real vec search)
  * A configured LLM provider (real Ollama / Anthropic / OpenAI response)
  * The ONNX embedding encoder present on disk
"""

import json
import struct
import uuid

import pytest

pytestmark = pytest.mark.integration


# Real production + the real EmbeddingService (gte-modernbert-base) are 768-d.
# The shared conftest `db` fixture pins 256-d for speed in unit tests, so we
# cannot reuse it here — we build our own 768-d database below.
_EMB_DIM = 768


def _pack(floats: list[float]) -> bytes:
    return struct.pack(f'{len(floats)}f', *floats)


def _unit_vec_at_angle(angle_deg: float) -> list[float]:
    """Return a 768-d unit vector at ``angle_deg`` from the x-axis (2-plane)."""
    import math
    rad = math.radians(angle_deg)
    v = [0.0] * _EMB_DIM
    v[0] = math.cos(rad)
    v[1] = math.sin(rad)
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _insert_transcript(conn, channel: str, role: str, content: str) -> int:
    """Insert a transcript row, return its auto-assigned id."""
    conn.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        (channel, role, content),
    )
    row = conn.execute(
        "SELECT id FROM transcript WHERE channel = ? ORDER BY id DESC LIMIT 1",
        (channel,),
    ).fetchone()
    conn.commit()
    return row[0]


def _has_vec_support(conn) -> bool:
    """True if episodes_vec exists and responds to a trivial query."""
    try:
        conn.execute("SELECT rowid FROM episodes_vec LIMIT 0").fetchone()
        return True
    except Exception:
        return False


@pytest.fixture
def prod_dim_db(tmp_path):
    """A fully-converged SQLite database at production dimensions (768).

    Bypasses the shared `db` fixture (which pins 256-d) because this test
    exercises the real EmbeddingService (768-d ONNX encoder) end-to-end —
    dimension mismatch between the embedding service and episodes_vec would
    silently drop the super-episode's own vector.

    Patches the shared db singleton so every service in the call chain
    (EpisodicService, _maybe_trigger_super_episode, find_super_candidates)
    sees this database.
    """
    import shutil
    from unittest.mock import patch

    import services.database_service as _db_mod
    from services.database_service import DatabaseService
    from services.schema_convergence_service import SchemaConvergenceService

    db_path = str(tmp_path / 'e2e.db')
    db_service = DatabaseService(db_path)
    SchemaConvergenceService(db_service, embedding_dimensions=768).converge()

    # Flush WAL into the main file so later connections see the full schema.
    with db_service.connection() as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # Clear any stale thread-local connection so the next _get_connection
    # opens the test file.
    _db_mod._local.conn = None
    _db_mod._local.db_path = None

    # Inject as the process-wide singleton so any `get_shared_db_service()`
    # lookup inside transcript_service / episodic_service sees this db.
    original = _db_mod._shared_db_service
    _db_mod._shared_db_service = db_service

    # Reset data_graph singleton in case any path touches it.
    import services.data_graph_service as _dgs_mod
    original_dgs_instance = _dgs_mod._instance
    _dgs_mod._instance = None

    conn = db_service._get_connection()
    try:
        yield conn, db_service
    finally:
        db_service.close_pool()
        _db_mod._shared_db_service = original
        _db_mod._local.conn = None
        _db_mod._local.db_path = None
        _dgs_mod._instance = original_dgs_instance


class TestMaybeTriggerSuperEpisodeEndToEnd:
    """End-to-end assertion that ``_maybe_trigger_super_episode`` produces a
    super-episode with consolidated_from populated AND back-pointers set on
    every source leaf.

    This test exercises the exact path the nightly 103 scenario runs through,
    in-process and in seconds, against the real production stack.
    """

    def test_cluster_of_three_produces_super_with_consolidated_from_and_backpointers(
        self, prod_dim_db,
    ):
        """Seed 3 tight-cosine apex leaves + transcript rows. Trigger the
        super-episode pipeline. Assert the super row has all 3 source IDs in
        consolidated_from and each leaf has consolidated_into set to the super.
        """
        conn, db_service = prod_dim_db

        # Guard: the integration test requires real sqlite-vec for clustering.
        if not _has_vec_support(conn):
            pytest.skip("sqlite-vec not available in test environment")

        from services.embedding_service import EmbeddingService
        from services.episodic_service import EpisodicService
        from services.transcript_service import _maybe_trigger_super_episode

        channel = f"e2e-super-{uuid.uuid4().hex[:8]}"

        # ── Seed 3 transcript rows so _fetch_transcript_spans has content ──
        t1 = _insert_transcript(conn, channel, 'user',
                                'I fixed the JWT signing secret rotation bug in auth-service today')
        t2 = _insert_transcript(conn, channel, 'user',
                                'Rolled out the new JWT rotation to all pods, zero downtime')
        t3 = _insert_transcript(conn, channel, 'user',
                                'Wrote the post-mortem on the original JWT 401 errors from last week')

        episodic_svc = EpisodicService(db_service)
        emb_svc = EmbeddingService()

        # ── Seed 3 apex episodes with tight embeddings (0°, 5°, 10°) ──
        # Cosines: 0-5≈0.996, 5-10≈0.996, 0-10≈0.985 — all well above the
        # SUPER_EPISODE_THRESHOLD=0.90 floor, guaranteeing a 3-node cluster.
        src_embs = [
            _unit_vec_at_angle(0.0),
            _unit_vec_at_angle(5.0),
            _unit_vec_at_angle(10.0),
        ]
        src_episodes = [
            {
                'gist': 'User shipped the JWT signing secret rotation fix',
                'salience': 7,
                'channel': channel,
                'transcript_ids': [t1],
                'transcript_id_start': t1,
                'transcript_id_end': t1,
                'emotional_valence': 0.2,
                'emotional_arousal': 0.4,
            },
            {
                'gist': 'User rolled out the JWT rotation with zero downtime',
                'salience': 6,
                'channel': channel,
                'transcript_ids': [t2],
                'transcript_id_start': t2,
                'transcript_id_end': t2,
                'emotional_valence': 0.3,
                'emotional_arousal': 0.3,
            },
            {
                'gist': 'User wrote the post-mortem on the JWT 401 incident',
                'salience': 7,
                'channel': channel,
                'transcript_ids': [t3],
                'transcript_id_start': t3,
                'transcript_id_end': t3,
                'emotional_valence': -0.1,
                'emotional_arousal': 0.5,
            },
        ]
        src_ids = [
            episodic_svc.store_episode(ep, embedding=emb)
            for ep, emb in zip(src_episodes, src_embs)
        ]

        # Baseline: no episode in this channel has consolidated_into set yet
        pre = conn.execute(
            "SELECT COUNT(*) FROM episodes "
            "WHERE channel = ? AND consolidated_into IS NOT NULL",
            (channel,),
        ).fetchone()[0]
        assert pre == 0, "pre-condition: no consolidated leaves"

        # ── Fire the pipeline ──
        _maybe_trigger_super_episode(channel, db_service, episodic_svc, emb_svc)

        # ── Assert exactly one super-episode was created ──
        super_rows = conn.execute(
            """
            SELECT id, consolidated_from, consolidated_into, gist
            FROM episodes
            WHERE channel = ?
              AND consolidated_into IS NULL
              AND id NOT IN (?, ?, ?)
              AND deleted_at IS NULL
            """,
            (channel, *src_ids),
        ).fetchall()
        assert len(super_rows) == 1, (
            f"Expected exactly 1 new super-episode for channel {channel!r}, "
            f"found {len(super_rows)}. Rows: {super_rows}"
        )
        super_id, consolidated_from_raw, super_ci, super_gist = super_rows[0]

        # ── Assert consolidated_from contains all 3 source IDs ──
        consolidated_from = json.loads(consolidated_from_raw or '[]')
        assert sorted(consolidated_from) == sorted(src_ids), (
            f"Super-episode {super_id} has consolidated_from={consolidated_from!r}, "
            f"expected {sorted(src_ids)!r}. This is the exact 103-nightly failure "
            "mode: super created but source references empty."
        )

        # ── Assert the super is an apex (its own consolidated_into is NULL) ──
        assert super_ci is None, (
            f"Super-episode must itself be an apex (consolidated_into=NULL), "
            f"got {super_ci!r}"
        )

        # ── Assert every source leaf has a back-pointer to the super ──
        leaf_rows = conn.execute(
            "SELECT id, consolidated_into FROM episodes WHERE id IN (?, ?, ?)",
            tuple(src_ids),
        ).fetchall()
        assert len(leaf_rows) == 3
        for leaf_id, ci in leaf_rows:
            assert ci == super_id, (
                f"Leaf {leaf_id} has consolidated_into={ci!r}, expected {super_id!r}. "
                "Apex-traversal in retrieve() walks this column — a missing "
                "back-pointer means the leaf cannot reach its super."
            )

        # ── Sanity: the super's gist is non-empty (LLM actually produced one) ──
        assert super_gist and len(super_gist) > 10, (
            f"Super gist is empty/trivial — LLM response may not have parsed: {super_gist!r}"
        )
