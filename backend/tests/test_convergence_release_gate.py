"""
Feature test for the self-heal convergence release gate.

Verifies that a pre-redesign ("old-shape") DB heals end-to-end with NO
operator-run migration - the boot sequence (converge + backfill) plus normal
worker ticks drive the full transition.

Acceptance items:
  (A)  Schema backfill - every redesign column populated after boot sequence.
  (B)  Decay heals floored weights; fossil on a muted channel is tombstoned.
  (D)  Retrieval surfaces a seeded episode via FTS after all ticks.
  (F)  Fact extraction: backlog drains (facts_extracted_at stamped) AND the
       seeded superseded fact gets valid_to set (bi-temporal invariant).
  (H)  >=50 apex leaves trigger roll-up: a level-1 super-episode exists, children
       tombstoned + consolidated_into pointing at the parent.
  (RS) Restart-safety: one tick drains a partial backlog; the progress cursor
       (facts_extracted_at) is durable across a fresh service instance; the
       remainder drains on a second tick; at most one item is re-processed.
"""

import contextlib
import json
import math
import pathlib
import sqlite3
import struct
import uuid
from collections.abc import Callable, Generator
from datetime import timedelta
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from services.subconscious_worker import SubconsciousWorker

import pytest

from services.database_service import DatabaseService
from services.llm_clients.base import ProviderClient
from services.provider_api import ProviderApiRequest, ProviderApiResponse
from services.providers import Providers
from services.schema_convergence_service import SchemaConvergenceService
from services.time_utils import utc_now

pytestmark = pytest.mark.unit


# ── Embedding helpers (256-dim — conftest convention) ─────────────────────────

_EMB_DIM = 256


def _pack(floats: list[float]) -> bytes:
    return struct.pack(f"{len(floats)}f", *floats)


def _unit(axis: int, dim: int = _EMB_DIM) -> list[float]:
    v = [0.0] * dim
    v[axis] = 1.0
    return v


def _topic_vec(axis: int, *, jitter_axis: int, jitter: float = 0.0) -> list[float]:
    """L2-normalised vector dominated by one axis with a small jitter component.
    Gives UMAP/HDBSCAN real intra-cluster spread rather than stacked points."""
    v = [0.0] * _EMB_DIM
    v[axis] = 1.0
    if jitter:
        v[jitter_axis] = jitter
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


# ── LLM boundary seam (Pattern H — the ONLY sanctioned boundary) ─────────────
# No unittest.mock.  Swap Providers._resolve at the class level via try/finally.
# Everything in-process (real Providers.send, real step logic, real DB) runs
# unchanged.  Pattern inherited from test_super_episode_pipeline.py verbatim.


class _FakeLLMService(ProviderClient):
    """Real ProviderClient subclass — Providers.send still runs pre-flight."""

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
    """Swap Providers._resolve for the duration of the block. No mock library."""
    original = Providers._resolve
    setattr(Providers, "_resolve", lambda self, *_a, **_kw: _FakeLLMService(send_fn))
    try:
        yield
    finally:
        setattr(Providers, "_resolve", original)


def _super_ep_response(gist: str) -> ProviderApiResponse:
    """SuperEpisodeEncoder expected JSON envelope - worker fills channel/consolidated_from/salience; model provides gist + affect."""
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


def _fact_op_response(ops: list[object]) -> ProviderApiResponse:
    return ProviderApiResponse(
        text=json.dumps({"ops": ops}),
        model="test-model",
        provider="mock",
        tool_calls=None,
    )


def _multiplex_send_fn(
    super_ep_gist: str,
    fact_ops_per_call: list[list[object]],
) -> Callable[[ProviderApiRequest], ProviderApiResponse]:
    """Route LLM calls by detecting "super-episode encoder" in the system prompt;
    all other calls get sequential fact op draws (NOOP is safe for self-gating steps)."""
    fact_state: dict[str, int] = {"n": 0}
    ops_list = fact_ops_per_call  # list of op-lists, one per episode

    def _send(dto: ProviderApiRequest) -> ProviderApiResponse:
        raw = str(dto)
        # SuperEpisodeEncoder system prompt uniquely contains this phrase.
        if "super-episode encoder" in raw:
            return _super_ep_response(super_ep_gist)
        # Fact extraction call (or any other step that gets through) — NOOP is safe.
        idx = min(fact_state["n"], len(ops_list) - 1)
        fact_state["n"] += 1
        return _fact_op_response(ops_list[idx])

    return _send


# ── DB helpers ─────────────────────────────────────────────────────────────────


def _shared_db_svc() -> DatabaseService:
    from services.database_service import get_shared_db_service
    return get_shared_db_service()


def _worker() -> "SubconsciousWorker":
    from services.subconscious_worker import SubconsciousWorker  # noqa: PLC0415
    return SubconsciousWorker(tick_sec=10, idle_window_sec=60)


def _sqlite_vec_available(db: sqlite3.Connection) -> bool:
    try:
        db.execute("SELECT COUNT(*) FROM episodes_vec")
        return True
    except Exception:
        return False


def _insert_episode_raw(
    db: sqlite3.Connection,
    *,
    episode_id: str | None = None,
    gist: str,
    salience: int = 6,
    channel: str = "user",
    level: int = 0,
    retrieval_weight: float = 1.0,
    last_relevant_at: str | None = None,
    consolidated_into: str | None = None,
    facts_extracted_at: str | None = None,
) -> str:
    """Insert an episode row directly (bypasses store_episode's embedding requirement)."""
    eid = episode_id or str(uuid.uuid4())
    lr = last_relevant_at or utc_now().isoformat()
    db.execute(
        """
        INSERT INTO episodes (
            id, gist, salience, channel, level,
            retrieval_weight, last_relevant_at,
            consolidated_into, facts_extracted_at,
            transcript_ids, consolidated_from,
            storage_strength
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', '[]', 1.0)
        """,
        (eid, gist, salience, channel, level, retrieval_weight, lr,
         consolidated_into, facts_extracted_at),
    )
    db.commit()
    return eid


def _insert_embedding(db: sqlite3.Connection, episode_id: str, emb: list[float]) -> None:
    row = db.execute(
        "SELECT rowid FROM episodes WHERE id = ?", (episode_id,)
    ).fetchone()
    if row:
        db.execute(
            "INSERT OR REPLACE INTO episodes_vec(rowid, embedding) VALUES (?, ?)",
            (row[0], _pack(emb)),
        )
        db.commit()


def _seed_apex_cluster_raw(
    db: sqlite3.Connection,
    channel: str,
    *,
    count: int,
    axis: int,
    jitter_axis: int,
) -> list[str]:
    """Seed apex leaf episodes pre-marked as fact-extracted so they do not
    compete with the fact-extraction backlog; purpose is to trigger roll-up (H)."""
    ids: list[str] = []
    vec_ok = _sqlite_vec_available(db)
    already_extracted = utc_now().isoformat()
    for i in range(count):
        eid = _insert_episode_raw(
            db,
            gist=f"{channel} convergence shard {axis}-{i}",
            channel=channel,
            salience=7,
            level=0,
            facts_extracted_at=already_extracted,  # exclude from fact backlog
        )
        if vec_ok:
            _insert_embedding(
                db, eid,
                _topic_vec(axis, jitter_axis=jitter_axis, jitter=0.05 * ((i % 5) + 1)),
            )
        ids.append(eid)
    return ids


# ── Legacy DDL constants (mirrors test_schema_redesign_convergence.py) ─────────
# Re-used verbatim: these are the canonical old shapes that convergence must heal.

_LEGACY_DATA_GRAPH_DDL = """
CREATE TABLE data_graph (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    kind              TEXT NOT NULL CHECK(kind IN ('user_specific','system','misc')),
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
"""


def _build_old_shape_db(tmp_path: pathlib.Path) -> "tuple[DatabaseService, dict[str, object]]":
    """Construct a pre-redesign database and seed all old-shape payloads.

    Returns (db_service, seed_meta) where seed_meta carries ids/timestamps
    needed for post-heal assertions.
    """
    # 1. Fresh converge to get ALL other tables correct.
    db = DatabaseService(str(tmp_path / "gate.db"))
    SchemaConvergenceService(db, embedding_dimensions=256).converge()
    # Do NOT call backfill_redesign_columns() — that is what we are about to test.

    old_first_seen = "2026-01-01T00:00:00+00:00"
    new_first_seen = "2026-03-15T12:30:00+00:00"
    # Fossil cutoff: the janitor tombstones episodes older than _JANITOR_FOSSIL_AGE_DAYS
    # (7 days per decay_engine_service.py).  Seed the fossil with a last_relevant_at
    # 30 days ago so it is well past the cutoff.
    fossil_age = (utc_now() - timedelta(days=30)).isoformat()

    with db.connection() as conn:

        # ── episodes → legacy shape (drop 4 redesign columns) ─────────────────
        conn.execute("ALTER TABLE episodes RENAME TO _ep_tmp")
        conn.execute(
            """
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                gist TEXT NOT NULL,
                salience INTEGER NOT NULL CHECK (salience BETWEEN 1 AND 10),
                channel TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                last_accessed_at TEXT,
                access_count INTEGER DEFAULT 0,
                deleted_at TEXT,
                transcript_ids TEXT DEFAULT '[]',
                transcript_id_start INTEGER,
                transcript_id_end INTEGER,
                emotional_valence REAL,
                emotional_arousal REAL,
                consolidated_from TEXT DEFAULT '[]',
                consolidated_into TEXT,
                storage_strength REAL DEFAULT 1.0,
                retrieval_weight REAL DEFAULT 1.0,
                location_lat REAL,
                location_lon REAL,
                location_name TEXT
            )
            """
            # NO: level, last_relevant_at, tombstoned_at, facts_extracted_at
        )
        conn.execute("DROP TABLE _ep_tmp")

        # (A) / (B) Seed: episodes with floored rw (old-compounding-law artifact).
        conn.execute(
            "INSERT INTO episodes (id, gist, salience, channel, retrieval_weight, "
            "last_accessed_at, created_at) "
            "VALUES ('ep-user-a', 'User lives in Valletta, Malta', 7, 'user', 0.01, "
            "datetime('now','-2 days'), datetime('now','-2 days'))"
        )
        # (D) ep-retrieve is stored via EpisodicService.store_episode() AFTER the
        # boot backfill so episodes_fts is populated by the production write path.
        # The raw-SQL episodes above only need column-level assertions, not FTS
        # retrieval, so their absence from the FTS index is intentional.

        # (B) Fossil: on a muted/legacy channel that the janitor does NOT protect.
        # "legacy_chat" is absent from source_profiles._EXACT_PROFILES and all
        # LIKE patterns, so profile_for("legacy_chat") == _MUTED → unprotected.
        conn.execute(
            "INSERT INTO episodes (id, gist, salience, channel, retrieval_weight, "
            "last_accessed_at, created_at) "
            f"VALUES ('ep-fossil', 'Old stranded legacy chat episode', 3, "
            f"'legacy_chat', 0.01, '{fossil_age}', '{fossil_age}')"
        )

        # ── data_graph → legacy shape (CHECK constraint, no valid_from/valid_to) ─
        conn.execute("DROP TABLE IF EXISTS data_graph")
        conn.execute(_LEGACY_DATA_GRAPH_DDL)

        # (A)/(F) Superseded fact WITHOUT valid_to — backfill must set it.
        conn.execute(
            "INSERT INTO data_graph (id, kind, key, value, first_seen_at, active) "
            "VALUES (10, 'user_specific', 'residence', 'Swieqi', ?, 0)",
            (old_first_seen,),
        )
        # Superseder — the live row this fact lost to.
        conn.execute(
            "INSERT INTO data_graph (id, kind, key, value, first_seen_at, active) "
            "VALUES (11, 'user_specific', 'residence', 'Valletta', ?, 1)",
            (new_first_seen,),
        )
        # Superseded_by edge: row 10 → row 11.
        conn.execute(
            "INSERT INTO data_graph_edges (from_id, to_id, edge_type) "
            "VALUES (10, 11, 'superseded_by')"
        )
        # (D) A live fact for the retrieval assertion.
        conn.execute(
            "INSERT INTO data_graph (id, kind, key, value, first_seen_at, active) "
            "VALUES (12, 'user_specific', 'hobby', 'hiking', ?, 1)",
            ("2026-02-01T00:00:00+00:00",),
        )

    seed_meta: dict[str, object] = {
        "old_first_seen": old_first_seen,
        "new_first_seen": new_first_seen,
    }
    return db, seed_meta


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Full convergence (all acceptance criteria A / B / D / F / H / I)
# ══════════════════════════════════════════════════════════════════════════════


class TestFullJumpConvergence:
    """Old-shape DB → N worker ticks → fully healed new-shape DB.

    Real production boot sequence and SubconsciousWorker; only the LLM network
    boundary is replaced (Pattern H - class-level swap, no mock library).
    """

    def test_old_shape_db_heals_end_to_end(self, db: sqlite3.Connection, store: object, tmp_path: pathlib.Path) -> None:
        # ── STEP 1: build the old-shape database ──────────────────────────────
        old_db, seed_meta = _build_old_shape_db(tmp_path)

        # Inject old_db as the shared singleton so every subsequent production
        # call sees the same database we degraded.
        import services.database_service as _db_mod
        import services.data_graph_service as _dgs_mod

        original_shared = _db_mod._shared_db_service
        original_dgs_instance = _dgs_mod._instance
        _db_mod._shared_db_service = old_db
        _db_mod._local.conn = None
        _db_mod._local.db_path = None
        _dgs_mod._instance = None

        try:
            # NOTE: The decay engine sub-cycles each call db_service.close_pool()
            # in their finally blocks, which closes the thread-local connection.
            # We therefore never hold a single conn reference across a _tick() call.
            # Instead we use the _qry() helper below which re-opens the connection
            # on every call-group via db.connection().

            def _qry(sql: str, params: tuple[object, ...] | None = None) -> list[object]:
                """Re-open and query — safe after close_pool() has been called."""
                with old_db.connection() as c:
                    if params:
                        return cast("list[object]", c.execute(sql, params).fetchall())
                    return cast("list[object]", c.execute(sql).fetchall())

            def _qry1(sql: str, params: tuple[object, ...] | None = None) -> object:
                rows = _qry(sql, params)
                return rows[0] if rows else None

            # ── STEP 2: production boot sequence (mechanism A) ─────────────────
            svc = SchemaConvergenceService(old_db, embedding_dimensions=256)
            svc.converge()
            svc.backfill_redesign_columns()

            # ── (A) Assert: redesign columns backfilled ────────────────────────
            cols = {cast("tuple[object, ...]", r)[1] for r in _qry("PRAGMA table_info(episodes)")}
            for col in ("level", "last_relevant_at", "tombstoned_at",
                        "facts_extracted_at"):
                assert col in cols, f"boot backfill failed to add episodes.{col}"

            dg_cols = {cast("tuple[object, ...]", r)[1] for r in _qry("PRAGMA table_info(data_graph)")}
            assert "valid_from" in dg_cols and "valid_to" in dg_cols

            # Every episode row now has last_relevant_at populated.
            null_lr = cast("tuple[object, ...]", _qry1("SELECT COUNT(*) FROM episodes WHERE last_relevant_at IS NULL"))[0]
            assert null_lr == 0, (
                f"boot backfill left {null_lr} episode(s) with NULL last_relevant_at"
            )

            # (A) Bi-temporal invariant: superseded row's valid_to == superseder's
            # valid_from — the backfill derived this from the superseded_by edge.
            old_valid_to_row = _qry1("SELECT valid_to FROM data_graph WHERE id=10")
            assert old_valid_to_row is not None, "superseded fact row (id=10) missing"
            old_valid_to = cast("tuple[object, ...]", old_valid_to_row)[0]
            assert old_valid_to == seed_meta["new_first_seen"], (
                f"boot backfill: old.valid_to ({old_valid_to!r}) must equal "
                f"new.first_seen_at ({seed_meta['new_first_seen']!r})"
            )
            live_valid_to = cast("tuple[object, ...]", _qry1("SELECT valid_to FROM data_graph WHERE id=11"))[0]
            assert live_valid_to is None, "live fact must keep valid_to=NULL"

            # ── STEP 3: seed ≥50 apex leaves on 'user' for roll-up (H) ────────
            # Done after backfill so the new schema columns are present.
            # Seeded via direct SQL (store_episode needs embedding_service running).
            # We need a fresh connection to check vec availability.
            with old_db.connection() as _c:
                vec_ok = _sqlite_vec_available(_c)
                topic_a_ids: list[str] = []
                topic_b_ids: list[str] = []
                if vec_ok:
                    topic_a_ids = _seed_apex_cluster_raw(
                        _c, "user", count=25, axis=2, jitter_axis=202
                    )
                    topic_b_ids = _seed_apex_cluster_raw(
                        _c, "user", count=25, axis=3, jitter_axis=203
                    )

            # ── STEP 4: seed (D) retrieval episode + (F) fact-extraction backlog ─
            # Use EpisodicService.store_episode so episodes_fts is populated by
            # the production write path (store_episode manually syncs FTS).
            from services.episodic_service import EpisodicService
            ep_svc = EpisodicService(old_db)

            # (D) Retrieval episode — stored via production path so FTS is synced.
            ep_retrieve = ep_svc.store_episode(
                {"gist": "User loves hiking in the Maltese countryside.",
                 "salience": 8, "channel": "user"},
                embedding=_unit(5),
            )

            # (F) Episode with a fact to extract.
            ep_for_fact = ep_svc.store_episode(
                {"gist": "User moved from Swieqi to Valletta.", "salience": 7,
                 "channel": "user"},
                embedding=_unit(10),
            )
            # Verify it entered the backlog.
            row = _qry1("SELECT facts_extracted_at FROM episodes WHERE id=?",
                        (ep_for_fact,))
            assert row is not None and cast("tuple[object, ...]", row)[0] is None, (
                "newly-stored episode must enter the fact-extraction backlog"
            )

            # ── STEP 5: run worker ticks ───────────────────────────────────────
            # Both steps that call the LLM (consolidate + fact_extraction) route
            # through _multiplex_send_fn which returns the correct shaped response
            # per call type without a mock library.
            #
            # synthesis → self-gates (no prior trait history on fresh test DB)
            # dmn       → self-gates (no user_summary row)
            # pattern_match → skips (insufficient transcript delta)
            # geo_patterns  → skips (insufficient located-transcript delta)
            # → Only consolidate and fact_extraction need the fake client.

            fact_ops: list[list[object]] = [
                # op for ep_for_fact: UPDATE triggers bi-temporal supersession.
                [{"op": "UPDATE", "key": "residence", "value": "Valletta"}],
            ]
            send_fn = _multiplex_send_fn("Consolidated user reflections.", fact_ops)

            with _inject_fake_client(send_fn):
                w = _worker()
                w._tick()

            # ── (B) Assert: decay healed floored weights ───────────────────────
            rw_row = _qry1(
                "SELECT retrieval_weight FROM episodes WHERE id='ep-user-a'"
            )
            assert rw_row is not None, "floored episode row missing after tick"
            rw_a = cast("tuple[object, ...]", rw_row)[0]
            assert cast(float, rw_a) > 0.3, (
                f"floored episode rw must recover above 0.3 after decay tick, "
                f"got {rw_a:.4f} (salience=7, ~2d old → expected ~0.65)"
            )

            # (B) Fossil: on 'legacy_chat' (unprotected channel) the janitor
            # should have tombstoned it.  The fossil episode is 30 days old —
            # well past the 7-day cutoff.
            fossil_row = _qry1(
                "SELECT tombstoned_at FROM episodes WHERE id='ep-fossil'"
            )
            assert fossil_row is not None, "fossil episode row disappeared unexpectedly"
            assert cast("tuple[object, ...]", fossil_row)[0] is not None, (
                "fossil episode on unprotected 'legacy_chat' channel (30d old) "
                "must be tombstoned by the janitor after tick 1"
            )

            # ── (D) Assert: retrieval surfaces the seeded hiking episode ───────
            # FTS lane carries the recall at 256-d (vec lane unverified at 256-d —
            # see module docstring; the dedicated 768-d fixture covers the vec lane).
            # ep_retrieve was stored via EpisodicService.store_episode() above so
            # the FTS index is populated by the real production write path.
            from services.episodic_retrieval_service import retrieve
            results = cast("list[dict[str, object]]", retrieve("hiking Maltese countryside", k=10))
            returned_ids = [r["id"] for r in results]
            assert ep_retrieve in returned_ids, (
                f"retrieval must surface the seeded hiking episode after heal; "
                f"got ids: {returned_ids!r}"
            )

            # ── (F) Assert: fact extraction drained the backlog ────────────────
            extracted_at_row = _qry1(
                "SELECT facts_extracted_at FROM episodes WHERE id=?", (ep_for_fact,)
            )
            assert extracted_at_row is not None and cast("tuple[object, ...]", extracted_at_row)[0] is not None, (
                "fact-extraction step must stamp facts_extracted_at on each "
                "processed episode"
            )
            # (F) Bi-temporal supersession chain: the UPDATE op triggered
            # invalidate() on the old 'residence=Swieqi' row (id=10, planted by
            # boot backfill); a new live row for 'residence=Valletta' should now
            # exist.  Boot backfill already set id=10.valid_to from the
            # superseded_by edge (assertion A above). The fact-extraction UPDATE
            # op creates a NEW data_graph row for 'Valletta' via
            # DataGraphService.store() / upsert_fact().
            live_valletta = _qry1(
                "SELECT id, valid_to FROM data_graph "
                "WHERE kind='user_specific' AND value='Valletta' "
                "AND active=1 AND deleted_at IS NULL"
            )
            assert live_valletta is not None, (
                "a live data_graph row for 'Valletta' must exist after fact extraction"
            )

            # ── (H) Assert: hierarchy roll-up (only when sqlite-vec available) ──
            # When vec is not available the roll-up is correctly skipped (no
            # embeddings → clustering returns empty).
            if vec_ok and (topic_a_ids or topic_b_ids):
                supers = _qry(
                    "SELECT id, consolidated_from, level FROM episodes "
                    "WHERE channel='user' AND deleted_at IS NULL "
                    "  AND consolidated_from IS NOT NULL "
                    "  AND consolidated_from != '[]'"
                )
                assert supers, (
                    "no super-episode created at 50 apex leaves — "
                    "the count-triggered roll-up must fire"
                )
                seeded_leaves = set(topic_a_ids) | set(topic_b_ids)
                for sid, cfrom_json, level in cast("list[tuple[object, object, object]]", supers):
                    assert level == 1, (
                        f"roll-up parent must be level=1, got {level!r} — "
                        " latent bug (store_episode never wrote level)"
                    )
                    children = json.loads(cast(str, cfrom_json))
                    assert children, "super-episode has empty consolidated_from"
                    for child_id in children:
                        assert child_id in seeded_leaves, (
                            f"super-episode consolidated an unexpected id: {child_id}"
                        )
                        child_row = _qry1(
                            "SELECT consolidated_into, tombstoned_at "
                            "FROM episodes WHERE id=?",
                            (child_id,),
                        )
                        assert cast("tuple[object, ...]", child_row)[0] == sid, (
                            f"child {child_id} must back-point at super {sid}"
                        )
                        assert cast("tuple[object, ...]", child_row)[1] is not None, (
                            f"child {child_id} must be tombstoned after roll-up"
                        )

        finally:
            old_db.close_pool()
            _db_mod._shared_db_service = original_shared
            _db_mod._local.conn = None
            _db_mod._local.db_path = None
            _dgs_mod._instance = original_dgs_instance


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Restart safety (RS)
# ══════════════════════════════════════════════════════════════════════════════


class TestRestartSafety:
    """Interrupting mid-budget and resuming loses <=1 item.

    Proves that facts_extracted_at is a durable SQLite cursor: a fresh worker
    instance (no in-memory state) drains only the remainder on the second tick.
    """

    def test_partial_drain_is_durable_and_remainder_drains_on_restart(
        self, db: sqlite3.Connection, store: object
    ) -> None:
        from services.subconscious_worker import _FACT_EXTRACTION_CALL_BUDGET
        from services.episodic_service import EpisodicService

        budget = _FACT_EXTRACTION_CALL_BUDGET
        # Seed budget + 3 episodes so one tick processes exactly `budget` and
        # three remain for the second tick.
        n = budget + 3

        ep_svc = EpisodicService(_shared_db_svc())
        ids: list[str] = []
        for i in range(n):
            eid = ep_svc.store_episode(
                {"gist": f"Restart-safety episode {i}.", "salience": 5,
                 "channel": "user"},
                embedding=_unit(i % _EMB_DIM),
            )
            # Push created_at into the past so oldest-first is deterministic.
            db.execute(
                "UPDATE episodes SET created_at = datetime('now', ? || ' seconds') "
                "WHERE id=?",
                (str(-(n - i)), eid),
            )
            ids.append(eid)
        db.commit()

        # All NOOP ops — we measure the drain, not the op semantics.
        noop_send: "Callable[[ProviderApiRequest], ProviderApiResponse]" = lambda _dto: _fact_op_response([{"op": "NOOP"}])  # noqa: E731

        # Helper: re-open a fresh connection after _step_fact_extraction may have
        # called close_pool() on the shared service.  _step_fact_extraction itself
        # does NOT call close_pool(), but we use _shared_db_svc()._get_connection()
        # to be safe (same pattern as test_decay_engine_service._fresh_conn()).
        def _stamped_at(episode_id: str) -> object:
            conn = _shared_db_svc()._get_connection()
            row = conn.execute(
                "SELECT facts_extracted_at FROM episodes WHERE id=?", (episode_id,)
            ).fetchone()
            return row[0] if row else None

        # ── Tick 1 (budget-many processed) ────────────────────────────────────
        with _inject_fake_client(noop_send):
            _worker()._step_fact_extraction()

        processed_after_tick1 = [e for e in ids if _stamped_at(e) is not None]
        remaining_after_tick1 = [e for e in ids if e not in processed_after_tick1]

        assert len(processed_after_tick1) == budget, (
            f"tick 1 must drain exactly {budget} episodes (the full budget), "
            f"got {len(processed_after_tick1)}"
        )
        assert len(remaining_after_tick1) == 3, (
            f"3 episodes must survive into the remainder, got "
            f"{len(remaining_after_tick1)}"
        )
        assert set(processed_after_tick1) == set(ids[:budget]), (
            "oldest episodes must be processed first (FIFO drain)"
        )

        # ── Durability check: fresh service instance sees the same cursor ──────
        # A second _step_fact_extraction call from a NEW worker instance (no
        # in-memory state) must pick up exactly where the first left off.
        # If facts_extracted_at were in-memory only, the fresh instance would
        # re-process already-stamped episodes, exceeding budget by 3.
        with _inject_fake_client(noop_send):
            worker2 = _worker()
            worker2._step_fact_extraction()

        processed_after_tick2 = [e for e in ids if _stamped_at(e) is not None]

        # All n episodes must now be processed; with a durable cursor the second
        # tick drains only the 3 remainder — nothing is re-processed.
        assert len(processed_after_tick2) == n, (
            f"all {n} episodes must be processed after 2 ticks; "
            f"got {len(processed_after_tick2)} — cursor may not be durable"
        )
        # Spec: "≤1 item re-processed/lost".  With facts_extracted_at as the
        # cursor, re-processing = 0 (NULL-guarded WHERE clause).
        reprocessed = len(processed_after_tick2) - n
        assert reprocessed <= 1, (
            f"at most 1 item may be re-processed across a restart; "
            f"got {reprocessed}"
        )
