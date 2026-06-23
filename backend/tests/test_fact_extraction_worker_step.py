"""Feature tests — worker fact-extraction step.

Tests the subconscious-worker step that closes the router gap: hard facts reach
``data_graph`` from the worker WITHOUT the chat model calling ``memory.store``.
The step selects episodes WHERE facts_extracted_at IS NULL oldest-first, runs one
small constrained LLM call per episode, and emits a single Mem0 op (ADD / UPDATE
/ DELETE / NOOP) — UPDATE/DELETE on a temporally-overlapping contradiction performs
bi-temporal invalidation (old active=0, old.valid_to == new.valid_from, fast-decay
tombstone). Unparseable model output → NOOP + counter.

Two contract points are pinned to FROZEN names: if the coder diverges, the test
reds at the seam, which IS the coordination signal (do not paper over it):

  * Entry point         : ``SubconsciousWorker._step_fact_extraction()`` follows the
                          existing ``_step_*`` convention. Surfaced under step key
                          ``fact_extraction`` (build doc 140; nightly 122 gates).
  * Op transport : ``MessageProcessor.process("", FactExtractionConfig)`` called, then
                   ``parse_fact_ops(text)`` (configs/channels/fact_extraction.py).
                   ``kind`` forced to ``user_specific`` inside the parser.

LLM boundary (Pattern H): resolved provider client (``Providers._resolve``) swapped
via try/finally context manager — never ``unittest.mock``, entirely real objects.

Tests touching both MemoryStore and data_graph take the ``store`` fixture so neither
leaks to a real path.
"""

import contextlib
import json
import sqlite3
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, cast

import pytest

from services.database_service import DatabaseService
from services.episodic_service import EpisodicService
from services.llm_clients.base import ProviderClient
from services.provider_api import ProviderApiResponse
from services.providers import Providers
from services.subconscious_worker import SubconsciousWorker

if TYPE_CHECKING:
    from typing import Protocol

    class _PatchableProviders(Protocol):
        _resolve: Callable[..., ProviderClient]

pytestmark = pytest.mark.unit

_DIM = 256  # conftest vec tables are 256-dim (suite convention)


# ── 256-dim embedding helpers (test_episodic_retrieval_service.py precedent) ───

def _unit(index: int, dim: int = _DIM) -> list[float]:
    v = [0.0] * dim
    v[index] = 1.0
    return v


# ── LLM network-boundary seam (test_pattern_match_processor.py precedent) ──────

class _FakeLLMService(ProviderClient):

    CONTENT_FIELD_LABEL = "message.content"

    def __init__(self, send_fn: Callable[[object], ProviderApiResponse]) -> None:
        self._send_fn = send_fn

    def get_context_limit(self) -> int:
        return 200_000

    def estimate_request_tokens(self, dto: object) -> int:
        return 1  # pre-flight over-cap check never triggers

    def send(self, dto: object) -> ProviderApiResponse:
        return self._send_fn(dto)


@contextlib.contextmanager
def _inject_fake_client(send_fn: Callable[[object], ProviderApiResponse]) -> Generator[None, None, None]:
    """Swap Providers._resolve at the class level for the block. No unittest.mock."""
    original = Providers._resolve
    cast("_PatchableProviders", Providers)._resolve = lambda self, *_a, **_kw: _FakeLLMService(send_fn)
    try:
        yield
    finally:
        cast("_PatchableProviders", Providers)._resolve = original


def _ops_response(ops: list[dict[str, object]] | str) -> ProviderApiResponse:
    text = ops if isinstance(ops, str) else json.dumps({"ops": ops})
    return ProviderApiResponse(
        text=text, model="test-model", provider="mock", tool_calls=None,
    )


def _sequenced_sender(responses: list[ProviderApiResponse]) -> Callable[[object], ProviderApiResponse]:
    """The fact step makes one LLM call per unprocessed episode; one entry per
    episode lets each candidate get its own op decision — last entry repeats.
    """
    state: dict[str, int] = {"n": 0}

    def _send(_dto: object) -> ProviderApiResponse:
        i = min(state["n"], len(responses) - 1)
        state["n"] += 1
        return responses[i]

    return _send


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _seed_episode(db: sqlite3.Connection, gist: str, *, channel: str = "user", emb_index: int = 3, salience: int = 6) -> str:
    """Seed one episode via the PRODUCTION write path (EpisodicService).

    store_episode never sets facts_extracted_at, so the row enters the backlog
    (facts_extracted_at IS NULL) exactly as a freshly-encoded episode would.
    Returns the episode id.
    """
    es = EpisodicService(_db_service_from_conn())
    return es.store_episode(
        {"gist": gist, "salience": salience, "channel": channel},
        embedding=_unit(emb_index),
    )


def _db_service_from_conn() -> DatabaseService:
    """Resolve the singleton DatabaseService the ``db`` fixture injected."""
    from services.database_service import get_shared_db_service
    return get_shared_db_service()


def _facts_extracted_at(db: sqlite3.Connection, episode_id: str) -> object:
    row = db.execute(
        "SELECT facts_extracted_at FROM episodes WHERE id=?", (episode_id,)
    ).fetchone()
    return row[0] if row else None


def _backlog_ids(db: sqlite3.Connection) -> list[str]:
    """Episode ids still awaiting fact extraction, oldest-first."""
    return [
        r[0] for r in db.execute(
            "SELECT id FROM episodes WHERE facts_extracted_at IS NULL "
            "AND deleted_at IS NULL ORDER BY created_at ASC, id ASC"
        ).fetchall()
    ]


def _fact_by_value(db: sqlite3.Connection, kind: str, value: str, *, active: int) -> dict[str, object] | None:
    """Fetch a data_graph fact for kind+value at the given active flag, or None.

    Queried by VALUE (which the fact step writes verbatim) rather than key:
    DataGraphService.store() LUT-canonicalises keys (e.g. 'cat_name'→'pets'),
    so asserting on the emitted key would be brittle against LUT contents. The
    behaviour under test is "the fact landed / was demoted", which the value +
    active flag capture exactly and refactor-safely.
    """
    row = db.execute(
        "SELECT id, key, value, active, valid_from, valid_to, retrieval_weight "
        "FROM data_graph WHERE kind=? AND value=? AND active=? "
        "AND deleted_at IS NULL ORDER BY id DESC LIMIT 1",
        (kind, value, active),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0], "key": row[1], "value": row[2], "active": row[3],
        "valid_from": row[4], "valid_to": row[5], "retrieval_weight": row[6],
    }


def _active_fact(db: sqlite3.Connection, kind: str, value: str) -> dict[str, object] | None:
    return _fact_by_value(db, kind, value, active=1)


def _superseded_fact(db: sqlite3.Connection, kind: str, value: str) -> dict[str, object] | None:
    """The demoted (active=0) row carrying ``value`` — note deleted_at IS NULL
    still holds: supersession sets active=0 + valid_to, never deleted_at."""
    return _fact_by_value(db, kind, value, active=0)


def _seed_active_fact(db: sqlite3.Connection, kind: str, key: str, value: str, *, retrieval_weight: float = 1.0) -> dict[str, object]:
    """Seed a pre-existing live fact through the PRODUCTION write path.

    Using DataGraphService.store() (not raw SQL) means the seed key is
    LUT-canonicalised identically to what the worker's later UPDATE op resolves
    to — so the exact-key match that triggers temporal supersession actually
    fires. retrieval_weight is then set directly (store() always seeds 1.0) so
    the fast-decay halving assertion has a known starting point.
    """
    from services.data_graph_service import get_data_graph_service
    res = get_data_graph_service().store(kind, key, value, source="seed")
    assert res is not None, "seed store() must land a row"
    db.execute(
        "UPDATE data_graph SET retrieval_weight=? WHERE id=?",
        (retrieval_weight, res["id"]),
    )
    db.commit()
    return res


def _seed_raw_fact(db: sqlite3.Connection, kind: str, key: str, value: str, *, retrieval_weight: float = 1.0) -> None:
    """Insert a pre-existing live fact via raw SQL with the key VERBATIM.

    Used by the DELETE test: DataGraphService.invalidate() matches the key
    verbatim (no LUT canonicalisation), so the seed must carry the exact key the
    worker's DELETE op names — store()'s canonicalisation would move it.
    """
    db.execute(
        "INSERT INTO data_graph (kind, key, value, active, first_seen_at, "
        "last_confirmed_at, valid_from, source, retrieval_weight) "
        "VALUES (?, ?, ?, 1, datetime('now'), datetime('now'),"
        "datetime('now'), 'test', ?)",
        (kind, key, value, retrieval_weight),
    )
    db.commit()


# ── Tests ──────────────────────────────────────────────────────────────────────

def _worker() -> SubconsciousWorker:
    return SubconsciousWorker(tick_sec=10, idle_window_sec=60)


def test_facts_routed_to_data_graph_without_chat_model_memory_store(db: sqlite3.Connection, store: object) -> None:
    """Acceptance 1 — a new episode (facts_extracted_at IS NULL) has its facts
    routed into data_graph by the WORKER step, and is marked processed.

    The chat model never runs here; no memory.store tool is invoked. The only
    LLM call is the worker's constrained fact-extraction op. This pins the
    router gap closed: facts reach data_graph from the subconscious, not the
    hot path.
    """
    ep = _seed_episode(db, "User's cat is named Tom.")
    assert _facts_extracted_at(db, ep) is None

    send = _sequenced_sender([
        _ops_response([{"op": "ADD", "key": "cat_name", "value": "Tom"}]),
    ])
    with _inject_fake_client(send):
        result = _worker()._step_fact_extraction()

    fact = _active_fact(db, "user_specific", "Tom")
    assert fact is not None, (
        "fact-extraction step must route the extracted fact into data_graph "
        f"without the chat model; step returned: {result!r}"
    )
    assert fact["value"] == "Tom"
    assert fact["key"], "the landed fact must carry a (canonicalised) key"
    # Episode is marked processed → it leaves the backlog.
    assert _facts_extracted_at(db, ep) is not None, (
        "episode must be stamped facts_extracted_at after processing"
    )
    assert ep not in _backlog_ids(db)


def test_add_op_novel_fact_creates_new_row(db: sqlite3.Connection, store: object) -> None:
    """Acceptance 2 — ADD on a fact with no prior contradiction inserts exactly
    one new live data_graph row; nothing is superseded."""
    ep = _seed_episode(db, "User lives in Valletta.")

    send = _sequenced_sender([
        _ops_response([{"op": "ADD", "key": "residence", "value": "Valletta"}]),
    ])
    with _inject_fake_client(send):
        _worker()._step_fact_extraction()

    fact = _active_fact(db, "user_specific", "Valletta")
    assert fact is not None and fact["value"] == "Valletta"
    # No supersession occurred — there is no demoted row carrying this value.
    assert _superseded_fact(db, "user_specific", "Valletta") is None
    assert _facts_extracted_at(db, ep) is not None


def test_update_op_contradiction_bitemporally_invalidates_old_fact(db: sqlite3.Connection, store: object) -> None:
    """Acceptance 3 — UPDATE/DELETE on a temporally-overlapping contradiction
    performs bi-temporal invalidation:

      * old row active=0,
      * old.valid_to == new.valid_from (the validity interval is closed),
      * supersedes / superseded_by audit edges exist,
      * fast-decay tombstone: old retrieval_weight halved,
      * the episode is stamped processed.

    Uses a genuinely *temporal* user_specific key ('residence' → canonical
    'residence', contradiction policy = temporal supersession). The spec's
    flagship cat→dog case is a coexist key ('pet'→'pets': a user may own a cat
    AND a dog, so two values append) and is covered by the nightly episode
    scenario (G5), not by single-key fact-store supersession — residence
    (Valletta → Swieqi) is the correct vehicle for THIS path, mirroring the
    existing test_data_graph_service.py supersession test.

    RED until the step routes the op through store()'s supersession path AND
    _apply_temporal_supersession sets valid_to ().
    """
    _seed_active_fact(db, "user_specific", "residence", "Valletta",
                      retrieval_weight=1.0)
    ep = _seed_episode(db, "User moved from Valletta to Swieqi.")

    send = _sequenced_sender([
        _ops_response([{"op": "UPDATE", "key": "residence", "value": "Swieqi"}]),
    ])
    with _inject_fake_client(send):
        _worker()._step_fact_extraction()

    live = _active_fact(db, "user_specific", "Swieqi")
    old = _superseded_fact(db, "user_specific", "Valletta")
    assert live is not None and live["value"] == "Swieqi", (
        "the contradicting fact must become the live row"
    )
    assert old is not None and old["value"] == "Valletta", (
        "the prior fact must be demoted (active=0), not deleted"
    )
    # Bi-temporal interval closed at the new fact's event-time start.
    assert old["valid_to"] is not None, "old fact's valid_to must be set"
    assert live["valid_from"] is not None, "new fact's valid_from must be set"
    assert old["valid_to"] == live["valid_from"], (
        f"old.valid_to ({old['valid_to']!r}) must equal new.valid_from "
        f"({live['valid_from']!r})"
    )
    # Fast-decay tombstone — retrieval_weight halved from the seeded 1.0.
    assert old["retrieval_weight"] == pytest.approx(0.5), (
        f"superseded fact must fast-decay (rw halved), got {old['retrieval_weight']}"
    )
    # Audit edges present.
    edges = {
        (e[0], e[1], e[2]) for e in db.execute(
            "SELECT from_id, to_id, edge_type FROM data_graph_edges"
        ).fetchall()
    }
    assert (live["id"], old["id"], "supersedes") in edges
    assert (old["id"], live["id"], "superseded_by") in edges
    assert _facts_extracted_at(db, ep) is not None


def test_noop_op_on_known_fact_writes_nothing(db: sqlite3.Connection, store: object) -> None:
    """Acceptance 4 — NOOP (duplicate / already-known fact) writes no new
    data_graph row and supersedes nothing, yet still marks the episode
    processed (it leaves the backlog)."""
    _seed_active_fact(db, "user_specific", "residence", "Valletta")
    before = db.execute("SELECT COUNT(*) FROM data_graph").fetchone()[0]
    ep = _seed_episode(db, "User mentions they live in Valletta again.")

    send = _sequenced_sender([_ops_response([{"op": "NOOP"}])])
    with _inject_fake_client(send):
        _worker()._step_fact_extraction()

    after = db.execute("SELECT COUNT(*) FROM data_graph").fetchone()[0]
    assert after == before, f"NOOP must not write any data_graph row ({before}->{after})"
    # Still the single original live row; nothing demoted.
    assert _superseded_fact(db, "user_specific", "Valletta") is None
    assert _facts_extracted_at(db, ep) is not None, (
        "a NOOP episode is still fully processed and must leave the backlog"
    )


def test_delete_op_bitemporally_invalidates_the_revoked_fact(db: sqlite3.Connection, store: object) -> None:
    """Acceptance 3 (DELETE leg) — a DELETE op invalidates the named live fact
    bi-temporally: active=0, valid_to set, retrieval_weight halved, but the row
    is NOT hard-deleted (audit + bi-temporal history kept). DELETE matches the
    key VERBATIM (DataGraphService.invalidate — no LUT canonicalisation), so the
    fact is seeded with the exact key the worker revokes.
    """
    # Verbatim-key seed (invalidate does not canonicalise).
    _seed_raw_fact(db, "user_specific", "owns_car", "Tesla Model 3",
                   retrieval_weight=1.0)
    ep = _seed_episode(db, "User sold their car and no longer owns one.")

    send = _sequenced_sender([
        _ops_response([{"op": "DELETE", "key": "owns_car", "value": "Tesla Model 3"}]),
    ])
    with _inject_fake_client(send):
        _worker()._step_fact_extraction()

    # No live row for that exact key remains.
    assert _active_fact(db, "user_specific", "Tesla Model 3") is None, (
        "DELETE must remove the fact from the live set"
    )
    # The row is invalidated (active=0 + valid_to), not hard-deleted.
    old = _superseded_fact(db, "user_specific", "Tesla Model 3")
    assert old is not None, "revoked fact must be kept for bi-temporal history"
    assert old["valid_to"] is not None, "DELETE must close the validity interval"
    assert old["retrieval_weight"] == pytest.approx(0.5), (
        f"revoked fact must fast-decay (rw halved), got {old['retrieval_weight']}"
    )
    assert _facts_extracted_at(db, ep) is not None


def test_unparseable_model_output_is_noop_never_corruption(db: sqlite3.Connection, store: object) -> None:
    """Acceptance 5 — unparseable LLM output → safe NOOP: zero data_graph
    writes, the episode is still marked processed (loud counter, never a
    corrupting partial write or an exception that aborts the tick).

    The durable, refactor-safe assertions are the data effects. The per-step
    counter (§4.6.3) surfaces in the step's return detail; we assert softly on
    it without locking its exact phrasing.
    """
    before = db.execute("SELECT COUNT(*) FROM data_graph").fetchone()[0]
    ep = _seed_episode(db, "User said something the model garbled.")

    # Not JSON — parse_fact_ops raises ValueError → counted, degraded to NOOP.
    send = _sequenced_sender([_ops_response("not-json <broken> {op: ADD,,,}")])
    with _inject_fake_client(send):
        result = _worker()._step_fact_extraction()

    after = db.execute("SELECT COUNT(*) FROM data_graph").fetchone()[0]
    assert after == before, (
        f"unparseable output must NEVER write to data_graph ({before}->{after})"
    )
    # Episode is still consumed from the backlog (NOOP marks it processed).
    assert _facts_extracted_at(db, ep) is not None, (
        "unparseable → NOOP must still stamp facts_extracted_at so the episode "
        "is not retried forever"
    )
    # Loud counter surfaced (soft — phrasing not locked).
    assert result is not None
    assert any(tok in str(result).lower() for tok in ("noop", "unparse", "skip")), (
        f"step detail should surface the safe-default counter; got {result!r}"
    )


def test_backlog_drains_at_budget_oldest_first(db: sqlite3.Connection, store: object) -> None:
    """Acceptance 6 — more unprocessed episodes than the per-tick budget (~20):
    exactly budget-many are processed this tick, OLDEST-FIRST, and the remainder
    stay in the backlog (facts_extracted_at IS NULL) for a later tick.

    Mechanism 3: the same budgeted drain serves fresh data and a 30k backlog.
    Tick stays under budget (acceptance: "tick stays under budget").
    """
    from services.subconscious_worker import _FACT_EXTRACTION_CALL_BUDGET

    budget = _FACT_EXTRACTION_CALL_BUDGET
    n = budget + 5
    # Seed n episodes with strictly increasing created_at so oldest-first is
    # unambiguous (store_episode stamps created_at = now; nudge each row back).
    ids: list[str] = []
    for i in range(n):
        ep = _seed_episode(db, f"Fact-bearing episode number {i}.", emb_index=i + 1)
        # Push created_at into the past, decreasing, so index 0 is oldest.
        db.execute(
            "UPDATE episodes SET created_at = datetime('now', ? || ' seconds') "
            "WHERE id=?",
            (str(-(n - i)), ep),
        )
        ids.append(ep)
    db.commit()

    # Every op is a NOOP — we are measuring the DRAIN, not the op semantics, so
    # no data_graph writes are needed; processing == facts_extracted_at stamped.
    send = _sequenced_sender([_ops_response([{"op": "NOOP"}])])
    with _inject_fake_client(send):
        _worker()._step_fact_extraction()

    processed = [e for e in ids if _facts_extracted_at(db, e) is not None]
    remaining = _backlog_ids(db)

    assert len(processed) == budget, (
        f"exactly {budget} episodes must process this tick, got {len(processed)}"
    )
    assert len(remaining) == n - budget, (
        f"{n - budget} episodes must remain in the backlog, got {len(remaining)}"
    )
    # Oldest-first: the budget-many processed are exactly ids[0:budget].
    assert set(processed) == set(ids[:budget]), (
        "the budget must drain the OLDEST episodes first; newer ones wait"
    )
    assert set(remaining) == set(ids[budget:])
