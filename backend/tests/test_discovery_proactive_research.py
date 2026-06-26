"""Feature tests for proactive autonomous research (the Auto Research / discovery loop).

Every test drives a real production entry point against the shared `db`
fixture's real SQLite (the new `discovery_runs` table auto-applies via
SchemaConvergenceService from schema.sql) — zero mocks except the single
sanctioned network-boundary seam (`_inject_fake_client`, a real ProviderClient
subclass swapped at the class level; precedent: test_fact_extraction_worker_step.py).

Wires under test, each a real cross-step trace:
1. Channel routing + recall inclusion — a store on the discovery channel is
   filed as kind='discovery' (memory_retrieval.handle_store) and surfaces through
   the real recall path (handle_recall → _search_data_graph kinds list).
2. The worker step — SubconsciousWorker._step_discovery fires the real
   MessageProcessor under DiscoveryConfig, the post-turn hook persists one run
   with its grounding, the run's full output is read live from the transcript by
   turn, and the durable 6h clock then throttles the next tick.
3. The Brain observability endpoints — the two locked read contracts plus the
   assistant-only transcript join and the 404 path.
"""

import contextlib
import sqlite3
from collections.abc import Callable, Generator
from datetime import timedelta
from typing import TYPE_CHECKING, cast

import pytest

from services.llm_clients.base import ProviderClient
from services.provider_api import ProviderApiResponse
from services.providers import Providers

if TYPE_CHECKING:
    from typing import Protocol

    class _PatchableProviders(Protocol):
        _resolve: Callable[..., ProviderClient]

pytestmark = pytest.mark.unit


# ── LLM network-boundary seam (the ONLY sanctioned mock) ───────────────────────
# A real ProviderClient subclass swapped at the class level via try/finally — no
# unittest.mock — so the whole ACT loop, prompt assembly, transcript writes and
# the post-turn hook run for real; only the network call to the model is replaced.

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
    original = Providers._resolve
    cast("_PatchableProviders", Providers)._resolve = lambda self, *_a, **_kw: _FakeLLMService(send_fn)
    try:
        yield
    finally:
        cast("_PatchableProviders", Providers)._resolve = original


def _text_response(text: str) -> ProviderApiResponse:
    """A plain final answer — no tool calls, so it ends the ACT loop in one step."""
    return ProviderApiResponse(text=text, model="test-model", provider="mock", tool_calls=None)


# ── 1. Channel routing + recall inclusion ──────────────────────────────────────

def test_discovery_store_routes_to_kind_discovery_and_recall_surfaces_it(db: sqlite3.Connection) -> None:
    """A store on the discovery channel files kind='discovery' and recall returns it.

    Drives the two production memory entry points the loop uses: handle_store
    (channel → kind routing) and handle_recall (the kinds-list wire in
    _search_data_graph). Regression guard: drop discovery from either the kind
    policy or the recall kinds list and this fails.
    """
    from services.memory_retrieval import handle_store, handle_recall
    from services.source_profiles import CHANNEL_DISCOVERY

    result = handle_store(CHANNEL_DISCOVERY, {
        "key": "webb_deep_field",
        "value": "The James Webb telescope released new deep-field images of distant galaxies.",
    })
    assert result.status == "success", f"discovery store failed: {result.body}"

    # A same-topic memory on an already-recalled kind. It is the canary that
    # proves retrieval is live in this env: whenever recall works at all this row
    # surfaces, so an empty result means the embedding/FTS backend is absent
    # (a real env gap → skip) rather than the discovery row being filtered out.
    assert handle_store("user", {
        "key": "telescope_interest",
        "value": "The user follows James Webb telescope deep-field imagery closely.",
    }).status == "success"

    # Channel routing is deterministic — assert it straight off the DB, no recall
    # ranking involved: the row landed as a discovery, not the default kind.
    row = db.execute(
        "SELECT kind FROM data_graph WHERE key = 'webb_deep_field' AND deleted_at IS NULL"
    ).fetchone()
    assert row is not None, "discovery store wrote no data_graph row"
    assert row[0] == "discovery", f"expected kind='discovery', got {row[0]!r}"

    # Recall inclusion runs through the real ranked retrieval (handle_recall →
    # _search_data_graph kinds list). With the canary guaranteeing a live backend
    # surfaces *something*, an empty result is an env gap (skip); a non-empty
    # result that omits 'discovery' is the wire being cut (hard fail).
    recall = handle_recall(None, "user", {"query": "James Webb telescope deep-field images"})
    assert recall.status == "success"
    hits = cast("list[dict[str, object]]", cast("dict[str, object]", recall.body)["results"])
    if not hits:
        pytest.skip("FTS/vec surfaced no hit — embedding service unavailable in this env")
    assert "discovery" in {h.get("kind") for h in hits}, (
        f"discovery kind absent from recall results: {[h.get('kind') for h in hits]}"
    )


# ── 2. The worker step: fire → persist → transcript join → throttle ────────────

def test_discovery_step_fires_persists_run_then_throttles(db: sqlite3.Connection, store: object) -> None:
    """One tick fires the loop, persists the run with its grounding, and then throttles.

    The durable clock is forced past its interval so the first call fires
    deterministically; the run that lands proves the worker → DiscoveryConfig →
    MessageProcessor → PersistDiscoveryRunHook chain end-to-end, and the second
    call proves the 6h self-throttle.
    """
    from services.data_graph_service import get_data_graph_service, KIND_SYSTEM
    from services.time_utils import utc_now
    from services.subconscious_worker import (
        SubconsciousWorker,
        _DISCOVERY_TIMESTAMP,
        _DISCOVERY_INTERVAL,
    )
    from services import discovery_runs, compaction_persistence

    summary = "The user is a space-exploration enthusiast building an observatory."
    compacted = "Yesterday we discussed the James Webb telescope's latest imagery."
    answer = "I noticed JWST released fresh deep-field images — worth a look when you're free."

    # Grounding, seeded via the same paths the worker reads at dispatch time: the
    # MAIN checkpoint lives in the compactions table (for_turn_id None), exactly
    # what get_compaction("user", None) returns at dispatch.
    get_data_graph_service().store(kind=KIND_SYSTEM, key="user_summary", value=summary, source="test:seed")
    compaction_persistence.write_compaction("user", None, 1, compacted)

    # Force the clock due so the step fires regardless of any prior run state.
    _DISCOVERY_TIMESTAMP.persist(utc_now() - _DISCOVERY_INTERVAL - timedelta(minutes=1))

    with _inject_fake_client(lambda _dto: _text_response(answer)):
        fired = SubconsciousWorker(tick_sec=10, idle_window_sec=60)._step_discovery()
    assert fired == "fired", f"expected the step to fire, got {fired!r}"

    runs = discovery_runs.list_runs(10)
    assert len(runs) == 1, f"expected exactly one persisted run, got {len(runs)}"
    assert runs[0]["researched"] == answer

    detail = discovery_runs.get_run_detail(cast("int", runs[0]["id"]))
    assert detail is not None
    assert detail["user_summary"] == summary, "grounding user summary not captured at dispatch"
    assert detail["compacted_summary"] == compacted, "grounding compaction not captured at dispatch"
    # Transcript is read live from the turn — the loop's assistant output, joined.
    assert detail["transcript"] == answer

    # The clock advanced, so the next tick self-throttles and writes no new run.
    throttled = SubconsciousWorker(tick_sec=10, idle_window_sec=60)._step_discovery()
    assert throttled.startswith("skip"), f"expected a throttle skip, got {throttled!r}"
    assert len(discovery_runs.list_runs(10)) == 1, "throttled tick still persisted a run"


# ── 3. Brain observability endpoints (locked read contract) ────────────────────

def test_research_observability_endpoints_expose_runs_and_detail(
    authed_client: "tuple[object, sqlite3.Connection, object]",
) -> None:
    """The two read endpoints honour the locked contract incl. the assistant-only join.

    Seeded through the production write helpers (Transcript writers + record_run,
    exactly what the hook calls), then read back over real Flask routes.
    """
    from flask.testing import FlaskClient
    from services.transcript_service import Transcript
    from services.source_profiles import CHANNEL_DISCOVERY
    from services import discovery_runs

    client = cast("FlaskClient", authed_client[0])

    input_id = Transcript.write_input_row(CHANNEL_DISCOVERY, "discovery", "background research task")
    turn_id = Transcript.turn_id_of_row(input_id)
    Transcript.write_assistant_row(CHANNEL_DISCOVERY, "Found one.", turn_id=turn_id)
    Transcript.write_assistant_row(CHANNEL_DISCOVERY, "Found two.", turn_id=turn_id)
    blob = "Found one.\n\nFound two."
    run_id = discovery_runs.record_run(
        turn_id=turn_id, user_summary="US", compacted_summary="CS", researched=blob,
    )
    assert run_id is not None

    listing = client.get("/system/observability/research")
    assert listing.status_code == 200
    body = listing.get_json()
    assert "generated_at" in body
    runs = body["runs"]
    assert isinstance(runs, list) and len(runs) == 1
    assert runs[0]["researched"] == blob

    detail = client.get(f"/system/observability/research/{run_id}")
    assert detail.status_code == 200
    run = detail.get_json()["run"]
    assert run["user_summary"] == "US"
    assert run["compacted_summary"] == "CS"
    # Assistant rows only — the input row is excluded from the joined blob.
    assert run["transcript"] == blob

    missing = client.get("/system/observability/research/999999")
    assert missing.status_code == 404
