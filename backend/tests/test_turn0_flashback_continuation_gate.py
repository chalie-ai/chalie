"""Feature tests for TKT-924 — turn-0 flashback + continuation gate + render.

Driven through the REAL production hot-path entry point that fires the flashback:
``MessageProcessor._seed_turn_zero()``. That is the method that, in production,
issues the framework ``memory(action=recall, _auto=True)`` before the first LLM
turn (message_processor.py:369). TKT-924 wraps it in a *continuation gate*: the
auto-recall must fire only on session start or topic shift, and be SKIPPED for a
continuation message ("yes, do that") whose embedding sits close to the running
conversation centroid.

Zero mocks. Real ``UserConfig`` MessageProcessor, real ``ToolDispatcher`` (called
from inside ``_seed_turn_zero``), real embedding model (the gate's centroid math
runs on real 768-dim vectors), real ``DataGraphService`` / ``EpisodicService``,
real SQLite via the ``db`` fixture, real ``memory_recall_log`` writes.

The single durable observable for the gate is the ``memory_recall_log`` row the
seed recall writes under ``caller='seed'`` (frozen by scenario lock, schema.sql
:447). The gate's decision is "did a NEW caller='seed' row appear for this turn?"
— so each test drives one or more sequential turns (each a fresh MessageProcessor,
sharing the persisted conversation transcript exactly as production does) and
counts the seed rows the gate let through.

Render behaviours (curated flashback block vs JSON) are read back from the
``tool_calls.result`` the seed dispatch recorded for the ``memory`` call, and from
the explicit-recall dispatch return — the exact strings production injects.

Status note (RED-first): at the time of writing, ``_seed_turn_zero`` fires the
seed UNCONDITIONALLY on every turn and renders the recall as JSON for both the
seed and the explicit path. Tests 2/3/4/5 therefore FAIL against current HEAD
(they assert the gate and the curated render that the coder is adding). Tests 1
and 6 are regression pins that may be GREEN now.
"""

import json

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.data_graph_service import get_data_graph_service
from services.database_service import get_shared_db_service
from services.episodic_service import EpisodicService
from services.message_processor import MessageProcessor
from services.transcript_service import write_input_row

pytestmark = pytest.mark.unit


# ── Harness ──────────────────────────────────────────────────────────────────


def _new_turn(text: str) -> MessageProcessor:
    """A real UserConfig MessageProcessor positioned exactly where the turn-0
    seed fires in production: input row written (anchors the act-trail FK), config
    attached, active_tools seeded. This is the same shape the ACT loop's _setup()
    hands to _seed_turn_zero()."""
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, text, {})
    mp.config = UserConfig()
    mp.uid = write_input_row("user", "user", text)
    mp.active_tools = list(mp.config.always_available or [])
    return mp


def _seed_count(db, caller: str = "seed") -> int:
    """How many flashback (caller='seed') recall rows the gate has let through."""
    row = db.execute(
        "SELECT COUNT(*) FROM memory_recall_log WHERE caller = ?", (caller,)
    ).fetchone()
    return row[0]


def _last_seed_result(db) -> str | None:
    """The exact string the seed's memory dispatch recorded into the act-trail —
    i.e. the content injected before the model's first turn. None if no memory
    seed call was recorded (gate skipped)."""
    row = db.execute(
        "SELECT result FROM tool_calls WHERE tool_name = 'memory' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return None if row is None else row[0]


#: The conftest builds the vec tables at 256 dims (tests/conftest.py:50), so a
#: production-faithful episode must be seeded with a 256-dim embedding — production
#: always passes one (transcript_service.py:455, subconscious_worker.py:330). Same
#: precedent as tests/test_episodic_retrieval_service.py:33-41.
_VEC_DIM = 256


def _unit(index: int, dim: int = _VEC_DIM) -> list:
    """A 256-dim unit basis vector for the fixture's vec tables, matching the
    conftest convention used across the episodic-retrieval suite."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _write_compaction(channel: str, body: str) -> None:
    """Persist a chat-history compaction living-doc exactly as production does — a
    transcript row with role='compaction' (compaction_persistence.get_compaction
    reads the newest such row). The terse-message gate reads its '- Now —' section
    (the bullet living-doc format the chat-history compactor writes, system_message
    _prompt.py:263-269)."""
    db = get_shared_db_service()
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO transcript (channel, role, content) VALUES (?, 'compaction', ?)",
            (channel, body),
        )
        conn.commit()


# ── 1. Flashback fires on session start ──────────────────────────────────────


def test_flashback_fires_on_session_start(db):
    """Turn 0 of a fresh conversation — no prior turns, no centroid — must fire
    the flashback: exactly one caller='seed' row appears, written by the seed
    recall the gate let through."""
    assert _seed_count(db) == 0, "fixture leaked a prior seed row"

    _new_turn("what's the latest on my Gozo ferry booking")._seed_turn_zero()

    assert _seed_count(db) == 1, (
        "session-start flashback did not fire — no caller='seed' row written"
    )


# ── 2. Continuation message does NOT re-fire ─────────────────────────────────


def test_continuation_message_does_not_refire_flashback(db):
    """A continuation message that stays on the running topic ('yes, do that'
    after a Gozo-trip turn) embeds close to the conversation centroid, so the gate
    must SKIP the flashback — no new caller='seed' row beyond the session-start
    one."""
    # Turn 1: session start — flashback fires (centroid is now established).
    _new_turn(
        "can you help me plan the family trip to Gozo and book the ferry"
    )._seed_turn_zero()
    assert _seed_count(db) == 1, "session-start flashback should have fired"

    # Turn 2: a bare continuation of the SAME thread. Close to centroid → skip.
    _new_turn("yes, do that")._seed_turn_zero()

    assert _seed_count(db) == 1, (
        "continuation message re-fired the flashback — the centroid gate let a "
        "second caller='seed' row through for 'yes, do that'"
    )


# ── 3. Topic shift DOES re-fire ──────────────────────────────────────────────


def test_topic_shift_refires_flashback(db):
    """A message on a clearly different topic embeds far from the conversation
    centroid, so the gate must RE-FIRE the flashback — a new caller='seed' row
    appears for the shifted turn."""
    # Establish a centroid firmly about the Gozo trip.
    _new_turn(
        "can you help me plan the family trip to Gozo and book the ferry"
    )._seed_turn_zero()
    _new_turn("which hotel near the Gozo harbour is best for the kids")._seed_turn_zero()
    before = _seed_count(db)
    # (turn 2 may or may not have fired; whatever the count, the shift must add one)

    # Hard topic shift — nothing to do with the trip.
    _new_turn(
        "remind me how to configure my home server's nginx reverse proxy for TLS"
    )._seed_turn_zero()

    assert _seed_count(db) == before + 1, (
        "topic-shift message did NOT re-fire the flashback — its embedding is far "
        "from the Gozo centroid yet the gate suppressed the seed recall"
    )


# ── 4. Terse message resolves topic via living-doc 'Now' ─────────────────────


def test_terse_message_resolves_topic_via_living_doc_now(db):
    """A terse message (< ~8 tokens) carries no topic on its own, so the gate
    composes its embed text with the compaction living-doc '## Now' section. Proof
    that the 'Now' section is actually consulted: the SAME terse message yields
    DIFFERENT gate decisions depending only on what '## Now' says relative to the
    conversation centroid.

    Both branches share an identical conversation centroid (one Gozo-trip turn)
    and an identical terse follow-up ('yes please'). The only difference is the
    living-doc 'Now':
      * continuation 'Now' (still about the trip) → composite near centroid → SKIP
      * shifted 'Now' (about an unrelated work deadline) → composite far → RE-FIRE
    If the gate ignored 'Now', both branches would decide identically.
    """
    # --- Branch A: 'Now' continues the established topic → terse should SKIP.
    _new_turn(
        "can you help me plan the family trip to Gozo and book the ferry"
    )._seed_turn_zero()
    base_a = _seed_count(db)
    _write_compaction(
        "user",
        "- Person — User is planning a family holiday.\n"
        "- Now — User is finalising the Gozo ferry booking and hotel for the family trip.\n"
        "- Last — User asked about the ferry.\n",
    )
    _new_turn("yes please")._seed_turn_zero()
    fired_when_now_matches = _seed_count(db) - base_a

    assert fired_when_now_matches == 0, (
        "terse message re-fired even though the living-doc 'Now' kept it on-topic "
        "— the 'Now' composite was not used to keep it near the centroid"
    )

    # --- Branch B: identical terse message, but 'Now' has shifted → RE-FIRE.
    # Fresh conversation state so the centroid is the SAME single Gozo turn.
    # tool_calls.transcript_id is an un-cascaded FK onto transcript(id)
    # (schema.sql:518), so the act-trail rows Branch A recorded must be cleared
    # before the transcript rows they reference.
    db.execute("DELETE FROM tool_calls")
    db.execute("DELETE FROM transcript")
    db.execute("DELETE FROM memory_recall_log")
    db.commit()

    _new_turn(
        "can you help me plan the family trip to Gozo and book the ferry"
    )._seed_turn_zero()
    base_b = _seed_count(db)
    _write_compaction(
        "user",
        "- Person — User is a backend engineer.\n"
        "- Now — User is debugging a production nginx TLS outage on the home server "
        "before a hard deadline.\n"
        "- Last — User mentioned the server.\n",
    )
    _new_turn("yes please")._seed_turn_zero()
    fired_when_now_shifted = _seed_count(db) - base_b

    assert fired_when_now_shifted == 1, (
        "terse message did NOT re-fire even though the living-doc 'Now' had shifted "
        "to an unrelated topic — the 'Now' section was ignored when building the "
        "terse-message embed"
    )


# ── 5. Injected flashback is the curated render block, not JSON ───────────────


def test_seed_renders_curated_block_not_json(db):
    """The seed no longer injects the raw recall JSON; it injects a curated bundle
    (≤5 live facts as bullets + ≤3 episodes as 'On <date>: <one-liner>', supers
    preferred). We seed FTS-findable facts + an episode, fire the session-start
    flashback, and read back the exact string the seed recorded into the act-trail.
    """
    # A live fact (data_graph) and an episode the recall can actually surface.
    get_data_graph_service().store(
        kind="user_specific", key="ferry_provider",
        value="Gozo Channel Line is the user's preferred ferry operator",
        source="test:seed",
    )
    es = EpisodicService(get_shared_db_service())
    # Seed an episodes_vec row (256-dim, fixture width) the production way — every
    # production caller passes an embedding (transcript_service.py:455,
    # subconscious_worker.py:330) — so the episode actually exists in the vector
    # store. Under the 256-dim harness the runtime seed query embeds at 768 and the
    # vector lane no-ops on the width mismatch (logged, non-fatal), so the FTS lane
    # is what carries the episode. The seed recall filters episodes by the
    # processor's channel (UserConfig().channel == 'user'), and FTS5 ANDs the query
    # terms — so the episode is stored on the 'user' channel and its gist contains
    # every content token of the seed query below.
    es.store_episode(
        {
            "gist": (
                "Remind me about the Gozo ferry booking — user booked it for the "
                "family trip on Saturday"
            ),
            "salience": 8, "channel": "user",
        },
        embedding=_unit(7),
    )

    _new_turn("remind me about the Gozo ferry booking")._seed_turn_zero()

    injected = _last_seed_result(db)
    assert injected is not None, "the session-start seed recorded no memory call"

    # It must NOT be the raw recall JSON contract: the {"results": [...]} array
    # shape (and the per-result 'relevance'/'confidence' keys) is what the JSON
    # render emits and what the curated block replaces.
    assert '"results"' not in injected, (
        f"seed still injected the raw recall JSON envelope: {injected!r}"
    )
    assert '"confidence"' not in injected and '"relevance"' not in injected, (
        f"seed leaked JSON result fields instead of curated prose: {injected!r}"
    )

    # It MUST carry the curated dated-episode one-liner marker. The episode was
    # seeded with a real created_at, so the render formats it as 'On <date>:'.
    assert "On " in injected, (
        f"curated block missing the 'On <date>:' episode one-liner: {injected!r}"
    )


# ── 6. Explicit memory.recall keeps the TKT-886 JSON contract ────────────────


def test_explicit_recall_keeps_json_contract(db):
    """Regression pin: an explicit, model-invoked memory.recall (NO _auto) still
    returns the TKT-886 {results, fallback} JSON body unchanged — the curated
    render is the SEED path only. This may be GREEN at HEAD; it must stay green."""
    get_data_graph_service().store(
        kind="user_specific", key="residence", value="Valletta", source="test:seed",
    )
    mp = _new_turn("where do I live")

    out = ToolDispatcher(mp).dispatch(
        "memory", {"action": "recall", "query": "residence city Valletta"}
    )

    # The structured JSON body is intact: a parseable 'results' array plus the
    # explicit-recall 'fallback' guardrail (both absent from the curated seed).
    head = out.index("]\n") + 2
    tail = out.index("\n[end:memory]")
    body = json.loads(out[head:tail])
    assert "results" in body, f"explicit recall dropped the JSON 'results' key: {out!r}"
    assert "fallback" in body, f"explicit recall dropped the 'fallback' guardrail: {out!r}"
    assert isinstance(body["results"], list)
    assert any(r.get("id") == "residence" for r in body["results"]), (
        f"live fact did not surface in explicit recall: {body['results']!r}"
    )
