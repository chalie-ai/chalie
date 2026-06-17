"""Feature tests for TKT-924 - turn-0 flashback + continuation gate + render.

The gate (in ``MessageProcessor._seed_turn_zero``) fires the auto memory recall
only on session start or topic shift, and skips it for continuation messages
whose embedding sits close to the running conversation centroid.

Observable: ``memory_recall_log`` rows with ``caller='seed'`` (schema.sql:447).
Each test drives sequential turns sharing the persisted transcript and counts
seed rows the gate let through.

Status (RED-first): ``_seed_turn_zero`` currently fires unconditionally. Tests
2/3/4/5 FAIL at HEAD; tests 1 and 6 are regression pins that may be GREEN now.
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
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, text, {})
    mp.config = UserConfig()
    mp.uid = write_input_row("user", "user", text)
    mp.active_tools = list(mp.config.always_available or [])
    return mp


def _seed_count(db, caller: str = "seed") -> int:
    row = db.execute(
        "SELECT COUNT(*) FROM memory_recall_log WHERE caller = ?", (caller,)
    ).fetchone()
    return row[0]


def _last_seed_result(db) -> str | None:
    """Returns the act-trail result string the seed recall injected, or None if
    the gate skipped the seed call."""
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
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _write_compaction(channel: str, body: str) -> None:
    """Persist a compaction row (role='compaction') so the terse-message gate can
    read the '- Now -' section when composing the embed text."""
    db = get_shared_db_service()
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO transcript (channel, role, content) VALUES (?, 'compaction', ?)",
            (channel, body),
        )
        conn.commit()


# ── 1. Flashback fires on session start ──────────────────────────────────────


def test_flashback_fires_on_session_start(db):
    """Turn 0 with no prior turns and no centroid must fire the flashback:
    exactly one caller='seed' row appears."""
    assert _seed_count(db) == 0, "fixture leaked a prior seed row"

    _new_turn("what's the latest on my Gozo ferry booking")._seed_turn_zero()

    assert _seed_count(db) == 1, (
        "session-start flashback did not fire — no caller='seed' row written"
    )


# ── 2. Continuation message does NOT re-fire ─────────────────────────────────


def test_continuation_message_does_not_refire_flashback(db):
    """A continuation on the same topic embeds close to the centroid; the gate
    must SKIP the flashback - no new caller='seed' row beyond the session-start."""
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
    """A message on a different topic embeds far from the centroid; the gate
    must RE-FIRE the flashback - a new caller='seed' row appears."""
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
    """A terse message carries no topic alone; the gate composes its embed with
    the compaction '- Now -' section. Proof: the same terse message ('yes please')
    against the same centroid yields DIFFERENT decisions depending only on '- Now -':
    continuation Now -> SKIP; shifted Now (unrelated deadline) -> RE-FIRE."""
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
    """The seed injects a curated bundle (live facts as bullets + episodes as
    'On <date>: <one-liner>') rather than the raw recall JSON envelope."""
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
    """Regression pin: explicit memory.recall (no _auto) returns the TKT-886
    {results, fallback} JSON body unchanged - curated render is the seed path only."""
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
