"""Feature tests for proactive autonomous research (the Auto Research / discovery loop).

Every test drives a real production entry point against the shared `db`
fixture's real SQLite — zero mocks except the single sanctioned network-boundary
seam (`_inject_fake_client`, a real ProviderClient subclass swapped at the class
level; precedent: test_fact_extraction_worker_step.py).

Wires under test, each a real cross-step trace:
1. Channel routing + recall inclusion — a store on the discovery channel is
   filed as kind='discovery' (memory_retrieval.handle_store) and surfaces through
   the real recall path (handle_recall → _search_data_graph kinds list).
2. The worker step — SubconsciousWorker._step_discovery fires the real
   MessageProcessor under DiscoveryConfig (a UserConfig split: it READS the user
   channel's history but WRITES its rows to the discovery channel), the output
   lands on the discovery channel as one turn, the model's context is grounded in
   the live user spine, and the durable 6h clock then throttles the next tick.
3. The thread API surface — ConfigType.DISCOVERY routes /api/threads and
   /api/thread onto the discovery channel (the Brain Auto Research view's
   replacement for the removed observability endpoints).
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
# the read/write-channel split run for real; only the network call to the model
# is replaced.

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


# ── 2. The worker step: fire → write to discovery channel → ground on user ─────

def test_discovery_step_writes_discovery_turn_grounded_on_user_history(
    db: sqlite3.Connection, store: object,
) -> None:
    """One fired tick writes the loop's output to the discovery channel as one
    turn, while the model's context is grounded in the live user channel.

    The durable clock is forced past its interval so the first call fires
    deterministically; the rows that land prove the worker → DiscoveryConfig
    (UserConfig split: read_channel='user', writes to 'discovery') →
    MessageProcessor chain end-to-end, the captured provider dto proves the
    user-channel history was read (the split's whole point), and the second call
    proves the 6h self-throttle.
    """
    from services.subconscious_worker import (
        SubconsciousWorker,
        _DISCOVERY_INTERVAL,
        _DISCOVERY_TIMESTAMP,
    )
    from services.time_utils import utc_now
    from services.transcript_service import Transcript

    user_msg = "We were just talking about the James Webb telescope's deep-field imagery."
    user_reply = "Right — the new deep-field plates resolved galaxies from 13 billion years ago."
    answer = "JWST just dropped fresh deep-field plates — worth a look when you're free."

    # Seed a live MAIN-spine user turn (input + settled reply) so the discovery
    # loop's read_channel='user' has real history to ground on — exactly the
    # grounding the old metadata dict used to thread in manually.
    user_input_id = Transcript.write_input_row("user", "user", user_msg)
    user_turn = Transcript.turn_id_of_row(user_input_id)
    Transcript.write_assistant_row("user", user_reply, turn_id=user_turn)

    # Force the clock due so the step fires regardless of any prior run state.
    _DISCOVERY_TIMESTAMP.persist(utc_now() - _DISCOVERY_INTERVAL - timedelta(minutes=1))

    captured: list[object] = []

    def _send(dto: object) -> ProviderApiResponse:
        captured.append(dto)
        return _text_response(answer)

    with _inject_fake_client(_send):
        fired = SubconsciousWorker(tick_sec=10, idle_window_sec=60)._step_discovery()
    assert fired == "fired", f"expected the step to fire, got {fired!r}"

    # The loop's output landed on the DISCOVERY channel as exactly one turn.
    disc_rows = db.execute(
        "SELECT turn_id, role, content FROM transcript WHERE channel = 'discovery' ORDER BY id"
    ).fetchall()
    assert disc_rows, "discovery step wrote no transcript rows"
    assert len({r[0] for r in disc_rows}) == 1, "expected a single discovery turn"
    assistant_blobs = [cast("str", r[2]) for r in disc_rows if r[1] == "assistant"]
    assert any(answer in blob for blob in assistant_blobs), (
        f"discovery assistant rows did not carry the answer: {assistant_blobs}"
    )

    # The model's context was grounded in the USER channel (the split's point):
    # the seeded user message surfaces in the assembled user prompt. Discovery's
    # own input row lives on the discovery channel and never appears here.
    assert captured, "no provider dto was captured"
    user_prompt = cast("list[dict[str, object]]", getattr(captured[0], "messages"))[0]["content"]
    assert user_msg in cast("str", user_prompt), (
        "discovery did not read the user channel — seeded user message absent from the prompt"
    )

    # The clock advanced, so the next tick self-throttles and writes no new turn.
    throttled = SubconsciousWorker(tick_sec=10, idle_window_sec=60)._step_discovery()
    assert throttled.startswith("skip"), f"expected a throttle skip, got {throttled!r}"
    after = db.execute(
        "SELECT DISTINCT turn_id FROM transcript WHERE channel = 'discovery'"
    ).fetchall()
    assert len(after) == 1, "throttled tick still wrote a new discovery turn"


# ── 3. Thread API surface for discovery (Brain Auto Research view) ─────────────

def test_discovery_thread_api_lists_and_returns_discovery_turns(
    authed_client: "tuple[object, sqlite3.Connection, object]",
) -> None:
    """ConfigType.DISCOVERY routes the thread API onto the discovery channel.

    Seeded through the production Transcript writers (exactly what the loop
    writes), then read back over real Flask routes — the surface the Brain Auto
    Research view now uses in place of the removed observability endpoints.
    """
    from flask.testing import FlaskClient

    from services.source_profiles import CHANNEL_DISCOVERY
    from services.transcript_service import Transcript

    client = cast("FlaskClient", authed_client[0])

    input_id = Transcript.write_input_row(CHANNEL_DISCOVERY, "user", "background research task")
    turn_id = Transcript.turn_id_of_row(input_id)
    Transcript.write_assistant_row(CHANNEL_DISCOVERY, "Found one.", turn_id=turn_id)
    Transcript.write_assistant_row(CHANNEL_DISCOVERY, "Found two.", turn_id=turn_id)

    listing = client.get("/api/threads?type=discovery")
    assert listing.status_code == 200
    feed = listing.get_json()
    turn_ids = [t["turn_id"] for t in feed["threads"]]
    assert turn_id in turn_ids, f"discovery turn {turn_id} absent from feed {turn_ids}"
    listed = next(t for t in feed["threads"] if t["turn_id"] == turn_id)
    assert listed["preview"], "feed row carried no preview"

    detail = client.get(f"/api/thread/{turn_id}?type=discovery")
    assert detail.status_code == 200
    assistant_texts = {
        cast("str", m["content"])
        for m in detail.get_json()["messages"]
        if m["role"] == "assistant"
    }
    assert {"Found one.", "Found two."} <= assistant_texts, (
        f"discovery detail missing assistant rows: {assistant_texts}"
    )
