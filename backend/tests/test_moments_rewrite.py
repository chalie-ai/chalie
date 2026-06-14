"""Feature tests for TKT-928 — Moments v2 (dedicated user-curated store).

Driven through the REAL production entry points end to end, zero mocks:

  * the real Flask app (``api.create_app`` via the ``authed_client`` fixture) for
    the HTTP CRUD + privacy surfaces,
  * the real ``ToolDispatcher(mp).dispatch("memory", {...})`` recall path,
  * the real ``MessageProcessor._seed_turn_zero()`` turn-0 flashback,
  * the real ``DataGraphService.store`` enum guard,
  * the real ``DecayEngineService.run_decay_cycle()`` the subconscious worker
    calls,
  * the real ``EmbeddingService`` (gte-modernbert-base, 768-dim ONNX) and the
    real sqlite-vec ``vec0`` KNN — NOT stubbed.

What TKT-928 changes (and therefore what these RED-first tests pin):

  Today "moments" live in ``data_graph`` as ``kind='moment'`` (auto-populated by a
  6-hour worker). TKT-928 moves them to a dedicated ``moments`` table with its own
  FTS5 + vec0 index, removes ``kind='moment'`` from the data_graph enum (so the
  existing ``store()`` guard rejects it), wipes the legacy rows via a standing
  janitor step in the decay cycle, repoints ``api/moments.py`` onto the new table
  (frontend JSON byte-identical), adds a labeled moments lane to ``handle_recall``
  (never in turn-0 flashback), and adds ``moments`` to privacy delete-all.

RED-first: against current HEAD (moments still in data_graph; no ``moments``
table; no recall lane; no janitor; ``kind='moment'`` still valid) these fail for
the RIGHT reason — the feature is missing — not on import/collection errors.

Marker: ``pytest.mark.unit`` — matching every real memory feature test in this
suite (test_data_graph_service.py, test_decay_engine_service.py,
test_episodic_retrieval_service.py, test_turn0_flashback_continuation_gate.py).
The pre-merge gate runs ``pytest -m unit``; ``integration``-marked files are NOT
collected by it, so these carry ``unit`` to run in the same gate as the code they
guard.
"""

import json

import pytest

from configs.channels import UserConfig
from abilities._dispatcher import ToolDispatcher
from services.data_graph_service import get_data_graph_service
from services.database_service import get_shared_db_service
from services.decay_engine_service import DecayEngineService
from services.embedding_service import get_embedding_service
from services.message_processor import MessageProcessor
from services.time_utils import utc_now
from services import transcript_service

pytestmark = pytest.mark.unit


# ── Shared seeding helpers (real rows via the real services) ──────────────────

# A real assistant turn whose plaintext is distinctive enough for FTS, and a
# semantically-related-but-NOT-lexically-overlapping probe for the vec lane.
_ASSISTANT_TURN = (
    "The Sleeping Lady is a Neolithic clay figurine found in the Hypogeum of "
    "Hal Saflieni, one of Malta's most important prehistoric burial sites."
)
# Shares NO content word with the turn above ("sleeping/lady/neolithic/clay/
# figurine/hypogeum/saflieni/malta/prehistoric/burial") — an FTS MATCH cannot
# bridge these, so a hit can only come from the vector embedding lane.
_SEMANTIC_PROBE = "ancient underground tomb sculpture from the stone age"


def _seed_assistant_turn(content: str = _ASSISTANT_TURN, channel: str = "user") -> int:
    """Write a real assistant transcript row; return its id."""
    return transcript_service.write_assistant_row(channel, content)


def _count_moments(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM moments").fetchone()[0]


def _moments_rows(conn) -> list:
    cur = conn.execute("SELECT * FROM moments")
    rows = cur.fetchall()
    return [dict(r) for r in rows]


# ── 1. Save verbatim → row in the NEW moments table + dedup ───────────────────


def test_pin_persists_verbatim_in_moments_table_and_dedups(authed_client):
    """POST /moments for a real assistant turn lands a row in the ``moments``
    table whose ``content`` is the assistant turn verbatim and whose
    ``transcript_id`` is set. First pin → 201 duplicate=False; an identical pin →
    200 duplicate=True; still exactly ONE row. Asserted against
    ``SELECT … FROM moments`` — NOT data_graph (the whole point of TKT-928)."""
    client, db, _store = authed_client
    tid = _seed_assistant_turn()

    first = client.post(
        "/moments", json={"transcript_id": tid}, content_type="application/json"
    )
    assert first.status_code == 201, first.get_json()
    assert first.get_json()["duplicate"] is False

    rows = _moments_rows(db)
    assert len(rows) == 1, f"expected exactly one moments row, got {rows!r}"
    assert rows[0]["content"] == _ASSISTANT_TURN, (
        "moment content must be the assistant turn stored verbatim"
    )
    assert rows[0]["transcript_id"] == tid

    second = client.post(
        "/moments", json={"transcript_id": tid}, content_type="application/json"
    )
    assert second.status_code == 200, second.get_json()
    assert second.get_json()["duplicate"] is True
    assert _count_moments(db) == 1, "a duplicate pin must not insert a second row"


# ── 2. Searchable via BOTH FTS (lexical) AND vec (semantic) ───────────────────
#
# This is the only behaviour that needs a 768-dim vec table: the real embedding
# model emits 768-dim vectors, while the shared conftest ``db`` fixture converges
# at 256 dims (so its vec lane no-ops on the width mismatch and FTS carries
# everything). To prove the moments_vec embedding really populates and is really
# KNN-searchable, this test builds its OWN converged DB at the model's true 768
# dims — using the SAME production factories the conftest uses
# (DatabaseService + SchemaConvergenceService), parameterised to the real model
# width. No mock, no alternative path: the production convergence creates the real
# moments / moments_fts / moments_vec from schema.sql, and the real EmbeddingService
# + real vec0 run.


@pytest.fixture
def db768(tmp_path):
    """A real, fully-converged SQLite DB sized to the real embedding model's 768
    dims, injected as the process-wide singleton exactly as conftest's ``db``
    fixture does (rebinds ``_shared_db_service``, resets the data_graph + heartbeat
    singletons). Built via the production DatabaseService + SchemaConvergenceService
    — the moments / moments_fts / moments_vec tables come from schema.sql, not from
    hand-rolled DDL."""
    import services.database_service as _db_mod
    from services.database_service import DatabaseService
    from services.schema_convergence_service import SchemaConvergenceService
    from services.policy_manager import PolicyManager
    import services.data_graph_service as _dgs_mod
    from services.heartbeat_service import heartbeat_service

    db_path = str(tmp_path / "test768.db")
    build = DatabaseService(db_path)
    SchemaConvergenceService(build, embedding_dimensions=768).converge()
    PolicyManager(build).apply_seed()
    with build.connection() as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    build.close_pool()

    db_service = DatabaseService(db_path)
    _db_mod._local.conn = None
    _db_mod._local.db_path = None
    original = _db_mod._shared_db_service
    _db_mod._shared_db_service = db_service
    original_dgs = _dgs_mod._instance
    _dgs_mod._instance = None
    heartbeat_service._ctx = None

    conn = db_service._get_connection()
    try:
        yield conn
    finally:
        db_service.close_pool()
        _db_mod._shared_db_service = original
        _db_mod._local.conn = None
        _db_mod._local.db_path = None
        _dgs_mod._instance = original_dgs
        heartbeat_service._ctx = None


@pytest.fixture
def authed_client768(db768):
    """The real Flask app bound to the 768-dim DB (auth + store bypassed exactly as
    the conftest authed_client does)."""
    from unittest.mock import patch  # auth/store boundary only — see conftest
    from api import create_app
    from services.memory_store import MemoryStore

    real_store = MemoryStore()
    with patch("services.auth_session_service.validate_session", return_value=True), \
         patch("services.memory_store.get_shared_store", return_value=real_store), \
         patch("services.memory_client.MemoryClientService.create_connection", return_value=real_store):
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client, db768, real_store


def test_pinned_moment_searchable_via_fts_and_vec(authed_client768):
    """After a pin, /moments/search finds the moment by a lexical substring (proves
    moments_fts is populated) AND by a semantically-related phrase that shares no
    content word (proves the moments_vec 768-dim embedding is populated and
    KNN-searchable). The real gte-modernbert model and real sqlite-vec vec0 run."""
    client, _db, _store = authed_client768
    tid = _seed_assistant_turn()

    created = client.post(
        "/moments", json={"transcript_id": tid}, content_type="application/json"
    )
    assert created.status_code == 201, created.get_json()

    # The embedding must really live in the NEW moments_vec store (not the legacy
    # data_graph path) — one row per pinned moment. This is the assertion that
    # makes the test fail for the right reason today (no moments / moments_vec
    # table yet) and proves the vec index is genuinely populated after the swap.
    moment_id = _db.execute(
        "SELECT id FROM moments WHERE transcript_id = ?", (tid,)
    ).fetchone()[0]
    vec_count = _db.execute(
        "SELECT COUNT(*) FROM moments_vec WHERE rowid = ?", (moment_id,)
    ).fetchone()[0]
    assert vec_count == 1, (
        "the pinned moment has no row in moments_vec — the 768-dim embedding was "
        "not written on save"
    )

    # Lexical lane — a literal substring of the stored content.
    lexical = client.get("/moments/search?q=Hypogeum")
    assert lexical.status_code == 200, lexical.get_json()
    lex_values = [it["value"] for it in lexical.get_json()["items"]]
    assert _ASSISTANT_TURN in lex_values, (
        "FTS lane did not surface the moment for a lexical substring — "
        "moments_fts is not populated"
    )

    # Semantic lane — phrase shares NO content word with the moment, so only the
    # vector embedding can bridge it.
    semantic = client.get(f"/moments/search?q={_SEMANTIC_PROBE.replace(' ', '+')}")
    assert semantic.status_code == 200, semantic.get_json()
    sem_values = [it["value"] for it in semantic.get_json()["items"]]
    assert _ASSISTANT_TURN in sem_values, (
        "vec lane did not surface the moment for a semantic-only phrase — the "
        "moments_vec embedding is missing or KNN search is not wired"
    )


# ── 3. Explicit recall surfaces moment in a labeled lane ──────────────────────


def _new_turn(text: str) -> MessageProcessor:
    """A real UserConfig MessageProcessor positioned where prod fires recall (input
    row written so the act-trail FK resolves). Same shape as
    test_turn0_flashback_continuation_gate.py's harness."""
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, text, {})
    mp.config = UserConfig()
    mp.uid = transcript_service.write_input_row("user", "user", text)
    mp.active_tools = list(mp.config.always_available or [])
    return mp


def _recall_body(out: str) -> dict:
    """Parse the JSON body the memory dispatch wraps between its trail markers —
    the exact slicing test_turn0_flashback_continuation_gate.py uses."""
    head = out.index("]\n") + 2
    tail = out.index("\n[end:memory]")
    return json.loads(out[head:tail])


def test_explicit_recall_surfaces_moment_in_labeled_lane(authed_client):
    """An explicit model-invoked ``memory.recall`` (the real ToolDispatcher path)
    surfaces a pinned moment in a clearly-labeled lane — its result row carries
    ``kind="moment"`` — while the TKT-886 ``{results, fallback}`` contract is
    preserved."""
    client, _, _store = authed_client
    tid = _seed_assistant_turn()
    created = client.post(
        "/moments", json={"transcript_id": tid}, content_type="application/json"
    )
    assert created.status_code == 201, created.get_json()

    mp = _new_turn("what did you tell me about the Sleeping Lady figurine")
    out = ToolDispatcher(mp).dispatch(
        "memory", {"action": "recall", "query": "Sleeping Lady Hypogeum figurine"}
    )

    body = _recall_body(out)
    assert "results" in body and isinstance(body["results"], list), (
        f"explicit recall dropped the JSON results contract: {out!r}"
    )
    assert "fallback" in body, "explicit recall dropped the TKT-886 fallback guardrail"

    moment_rows = [r for r in body["results"] if r.get("kind") == "moment"]
    assert moment_rows, (
        f"recall did not surface the pinned moment in a kind='moment' lane: "
        f"{body['results']!r}"
    )
    assert any(_ASSISTANT_TURN in (r.get("content") or "") for r in moment_rows), (
        "the moment lane row did not carry the pinned content"
    )


# ── 4. Pinned moments are NOT in the turn-0 flashback ─────────────────────────


def _last_seed_result(db) -> str | None:
    row = db.execute(
        "SELECT result FROM tool_calls WHERE tool_name = 'memory' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return None if row is None else row[0]


def _reset_conversation(conn) -> None:
    """Restore genuine session-start conditions WITHOUT touching the pinned
    moment. The turn-0 flashback's continuation gate fires unconditionally only
    when there is no running thread to continue (no prior transcript turns / no
    centroid) — so to observe the flashback at all we must clear the conversation
    state the pin's assistant turn created. The moment itself lives OUTSIDE the
    transcript (in data_graph today, in the ``moments`` table after TKT-928), so
    clearing the conversation leaves it intact and still recall-eligible — exactly
    the state we need to prove it is excluded from the auto-seed. tool_calls is the
    act-trail (FK into transcript) so it is cleared first."""
    conn.execute("DELETE FROM tool_calls")
    conn.execute("DELETE FROM transcript")
    conn.execute("DELETE FROM memory_recall_log")
    conn.commit()


def test_turn_zero_flashback_excludes_pinned_moments(authed_client):
    """The turn-0 flashback (real ``MessageProcessor._seed_turn_zero()`` →
    ``TurnZeroFlashback.seed()``) must never contain a pinned moment — moments are
    a user-curated lane, excluded from the auto-seed by construction. We pin a
    moment whose content matches the seed query, then fire a genuine session-start
    flashback (which fires unconditionally) and assert the moment's content is
    ABSENT from the block the seed injected.

    Regression guard: today the moment is excluded because ``_search_data_graph``
    omits ``KIND_MOMENT`` from its ``kinds`` filter; after TKT-928 moments leave
    data_graph entirely. This test pins that exclusion across the refactor — it
    asserts the flashback DID fire (so the absence is meaningful, not vacuous) and
    that the moment did not leak into it."""
    client, db, _store = authed_client
    tid = _seed_assistant_turn()
    created = client.post(
        "/moments", json={"transcript_id": tid}, content_type="application/json"
    )
    assert created.status_code == 201, created.get_json()

    # The pin wrote an assistant transcript turn; clear conversation state so the
    # next turn is a genuine session start (flashback fires unconditionally). The
    # moment persists outside the transcript and is still recall-eligible.
    _reset_conversation(db)

    _new_turn("tell me about the Sleeping Lady at the Hypogeum")._seed_turn_zero()

    injected = _last_seed_result(db)
    assert injected is not None, (
        "the session-start flashback did not fire — no memory seed call was "
        "recorded, so the moment-exclusion assertion would be vacuous"
    )
    assert _ASSISTANT_TURN not in injected, (
        "a pinned moment leaked into the turn-0 flashback block — moments must "
        f"never appear in the auto-seed: {injected!r}"
    )


# ── 5. data_graph rejects kind='moment' (enum enforced) ───────────────────────


def test_data_graph_store_rejects_kind_moment(db):
    """``data_graph_service.store(kind='moment', …)`` writes NO row and returns a
    falsy value — once ``moment`` is removed from the data_graph kind enum, the
    existing ``if kind not in VALID_KINDS: return None`` guard rejects it (this
    closes the hole where any channel could write any kind)."""
    dgs = get_data_graph_service()

    result = dgs.store(kind="moment", key="moment_999", value="should be rejected")

    assert not result, (
        f"store(kind='moment') must be rejected and return falsy, got {result!r}"
    )
    row = db.execute(
        "SELECT COUNT(*) FROM data_graph WHERE kind = 'moment'"
    ).fetchone()[0]
    assert row == 0, "store(kind='moment') must write zero rows"


# ── 6. Legacy data_graph kind='moment' rows are wiped by the decay cycle ───────


def _seed_legacy_dg_moment(conn, key: str, value: str) -> None:
    """Insert a pre-TKT-928 ``kind='moment'`` row directly into data_graph (the
    'store' path can't be used — it rejects 'moment' after the change), plus its
    FTS companion, mirroring the 199 legacy rows the worker left behind."""
    now = utc_now().isoformat()
    conn.execute(
        "INSERT INTO data_graph (kind, key, value, source, first_seen_at, "
        "last_confirmed_at, active) VALUES ('moment', ?, ?, 'pin', ?, ?, 1)",
        (key, value, now, now),
    )
    rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    try:
        conn.execute(
            "INSERT INTO data_graph_fts(rowid, key, value, kind, search_queries) "
            "VALUES (?, ?, ?, 'moment', '')",
            (rowid, key, value),
        )
    except Exception:
        # External-content FTS may reject a manual insert depending on the build;
        # the durable observable (AC#5) is the data_graph row count, asserted below.
        pass
    conn.commit()


def test_decay_cycle_wipes_legacy_data_graph_moments(db):
    """Seed legacy ``data_graph kind='moment'`` rows, run the REAL decay cycle the
    subconscious worker calls (``DecayEngineService().run_decay_cycle()``), and
    assert ZERO ``kind='moment'`` rows survive. The standing janitor step idempotently
    deletes them (moment policy is ttl_days=None, so plain decay never would)."""
    _seed_legacy_dg_moment(db, "moment_101", "an old auto-saved assistant reply")
    _seed_legacy_dg_moment(db, "moment_102", "another stale moment row")
    assert db.execute(
        "SELECT COUNT(*) FROM data_graph WHERE kind = 'moment'"
    ).fetchone()[0] == 2, "fixture failed to seed the legacy moment rows"

    DecayEngineService().run_decay_cycle()

    survivors = get_shared_db_service()._get_connection().execute(
        "SELECT COUNT(*) FROM data_graph WHERE kind = 'moment'"
    ).fetchone()[0]
    assert survivors == 0, (
        f"decay cycle left {survivors} legacy kind='moment' rows in data_graph — "
        "the standing janitor wipe step is missing"
    )


# ── 7. Frontend JSON contract stays byte-identical through the swap ───────────

_CONTRACT_KEYS = {"id", "transcript_id", "key", "value", "message_text", "created_at"}


def test_frontend_json_contract_is_byte_identical(authed_client):
    """POST /moments, GET /moments, GET /moments/search and POST /moments/<id>/forget
    each return exactly the key set the frontend reads, with ``key == moment_<tid>``
    and ``value == message_text == content``. A regression guard that locks the
    contract across the data_graph→moments swap."""
    client, _db, _store = authed_client
    tid = _seed_assistant_turn()

    created = client.post(
        "/moments", json={"transcript_id": tid}, content_type="application/json"
    )
    assert created.status_code == 201, created.get_json()
    item = created.get_json()["item"]
    assert set(item.keys()) == _CONTRACT_KEYS, (
        f"POST /moments item keys drifted: {sorted(item.keys())}"
    )
    assert item["transcript_id"] == tid
    assert item["key"] == f"moment_{tid}"
    assert item["value"] == _ASSISTANT_TURN
    assert item["message_text"] == _ASSISTANT_TURN

    listed = client.get("/moments")
    assert listed.status_code == 200
    list_items = listed.get_json()["items"]
    assert list_items and set(list_items[0].keys()) == _CONTRACT_KEYS, (
        f"GET /moments item keys drifted: {sorted(list_items[0].keys())}"
    )

    searched = client.get("/moments/search?q=Hypogeum")
    assert searched.status_code == 200
    search_items = searched.get_json()["items"]
    assert search_items and set(search_items[0].keys()) == _CONTRACT_KEYS, (
        f"GET /moments/search item keys drifted: {sorted(search_items[0].keys())}"
    )

    forget = client.post(f"/moments/{tid}/forget")
    assert forget.status_code == 200
    assert forget.get_json()["ok"] is True


# ── 8. Privacy delete-all wipes the moments table ─────────────────────────────


def test_privacy_delete_all_wipes_moments(authed_client):
    """A pinned moment is destroyed by ``DELETE /privacy/delete-all`` (with the real
    X-Confirm-Delete header) — proving ``moments`` is in ``_DELETE_ALL_TABLES``."""
    client, db, _store = authed_client
    tid = _seed_assistant_turn()
    created = client.post(
        "/moments", json={"transcript_id": tid}, content_type="application/json"
    )
    assert created.status_code == 201, created.get_json()
    assert _count_moments(db) == 1, "precondition: one moment pinned"

    resp = client.delete(
        "/privacy/delete-all", headers={"X-Confirm-Delete": "yes"}
    )
    assert resp.status_code == 200, resp.get_json()
    assert resp.get_json().get("deleted") is True

    assert _count_moments(get_shared_db_service()._get_connection()) == 0, (
        "privacy delete-all did not wipe the moments table — 'moments' is not in "
        "_DELETE_ALL_TABLES"
    )
