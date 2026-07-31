"""Test memory.recall guardrail behavior, silent turn-0 seed, and per-lane
relative-floor telemetry. All exercised through the real production
path via DispatchService.dispatch("memory") with zero mocks.

Pinned behaviors:
1. A recall that finds nothing is a LOUD no-results error for EVERY caller —
   explicit and turn-0 seed alike — carrying the hard rule (memory = past state,
   find_tools = ground truth); no fan-out.
2. A recall that surfaces memories carries the same hard rule in its result set,
   regardless of caller; that positive path needs the live embedding pipeline and
   is covered by live-fire, not this suite.
3. Recall-log uses floor_cut_count / final_rrf_count fields under the correct caller.
"""

import sqlite3

import pytest

from configs.channels import UserConfig
from controllers.message_processor import MessageProcessor

pytestmark = pytest.mark.unit

_HINT_LEAD = "HARD RULE: these results are memories"


def _build_user_mp(text: str) -> MessageProcessor:
    """Builds a real MessageProcessor with active_tools seeded — mirrors
    _setup()'s own seed without running the full turn chain. The constructor is
    "constructed inert" — it never allocates a turn or writes the anchoring input
    row itself (that's begin()'s job, run on a background drive thread we don't
    want here). Replay the exact synchronous half of begin() (turn allocation +
    the anchoring input row) through the mp's own transcript_service, so mp.uid
    anchors the act-trail FK exactly as it would mid-turn — no private-field
    poking, no invented API."""
    mp = MessageProcessor(UserConfig(), raw_input=text)
    mp.active_tools = list(mp.config.always_available or [])
    with mp.db.transaction():
        mp.turn_id = mp.transcript_service.allocate_turn()
        mp.uid = mp.transcript_service.append_input(mp.raw_input)
        mp.current_transcript_id = mp.uid
    return mp


def _tool_names_recorded(db: sqlite3.Connection, transcript_id: int) -> list[str]:
    rows = db.execute(
        "SELECT tool_name FROM tool_calls WHERE transcript_id = ?",
        (transcript_id,),
    ).fetchall()
    return [r[0] for r in rows]


def _last_recall_log(db: sqlite3.Connection, caller: str | None = None) -> dict[str, object] | None:
    if caller is None:
        row = db.execute(
            "SELECT caller, query, episode_count, floor_cut_count, final_rrf_count "
            "FROM memory_recall_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
    else:
        row = db.execute(
            "SELECT caller, query, episode_count, floor_cut_count, final_rrf_count "
            "FROM memory_recall_log WHERE caller = ? ORDER BY id DESC LIMIT 1",
            (caller,),
        ).fetchone()
    if row is None:
        return None
    return {
        "caller": row[0],
        "query": row[1],
        "episode_count": row[2],
        "floor_cut_count": row[3],
        "final_rrf_count": row[4],
    }


def test_explicit_recall_carries_guardrail_and_fires_no_fanout(db: sqlite3.Connection) -> None:
    mp = _build_user_mp("what is my home wifi password")

    out = mp.dispatch_service.dispatch(
        "memory", {"action": "recall", "query": "what is my home wifi password"}
    )

    # An empty explicit recall is a loud no-results ERROR, never a quiet
    # success with zero rows — a weak model reads status=success as "the call
    # worked, move on" and settles on fabricated content.
    assert "code=no-results" in out
    assert "No results found." in out

    # The guardrail hint is present and routes to tool discovery — it must
    # never name tools that are not in the registry (the document subsystem
    # is deleted; its stale mention sent the model to a dead tool).
    assert _HINT_LEAD in out
    assert "`find_tools`" in out
    assert "document" not in out.lower()

    # Fan-out is gone: the removed code dispatched document.search + schedule.search,
    # each of which would have recorded a tool_calls row under this transcript.
    from typing import cast
    names = _tool_names_recorded(db, cast(int, mp.uid))
    assert "memory" in names, f"the recall itself was not recorded: {names!r}"
    assert "document" not in names, f"document.search fan-out still firing: {names!r}"
    assert "schedule" not in names, f"schedule.search fan-out still firing: {names!r}"

    # The recall wrote the NEW per-lane telemetry (radius columns are gone).
    tel = _last_recall_log(db, caller="llm_recall")
    assert tel is not None, "explicit recall wrote no telemetry row"
    assert tel["caller"] == "llm_recall"
    assert tel["query"] == "what is my home wifi password"
    # New floor/result counters exist and are sane (non-negative ints).
    assert cast(int, tel["floor_cut_count"]) >= 0
    assert cast(int, tel["final_rrf_count"]) >= 0


def test_turn0_seed_recall_errors_on_empty_and_logs_seed_telemetry(db: sqlite3.Connection) -> None:
    mp = _build_user_mp("what did we talk about at home last week")

    out = mp.dispatch_service.dispatch(
        "memory",
        {"action": "recall", "query": "what did we talk about at home last week", "_auto": True},
    )

    # The seed no longer special-cases: an empty recall is a loud no-results error
    # for every caller, carrying the hard rule so the model pivots to live tools.
    assert "code=no-results" in out, f"empty seed must error like any recall: {out!r}"
    assert _HINT_LEAD in out, f"empty seed must carry the hard rule: {out!r}"
    from typing import cast
    # ...and never fans out to the other stores.
    names = _tool_names_recorded(db, cast(int, mp.uid))
    assert "document" not in names and "schedule" not in names, (
        f"seed recall fanned out to other stores: {names!r}"
    )

    # The seed wrote a telemetry row under caller='seed' with the new fields.
    tel = _last_recall_log(db, caller="seed")
    assert tel is not None, "seed recall wrote no telemetry row"
    assert tel["caller"] == "seed"
    assert cast(int, tel["floor_cut_count"]) >= 0
    assert cast(int, tel["final_rrf_count"]) >= 0


def _drain_search_index() -> None:
    """Drive the real async search-expander pipeline synchronously against the
    bound test DB — the exact production code path, no mocks. In prod the
    search_expander_worker daemon does this continuously; a test must do it
    explicitly because no worker runs under pytest."""
    from services.search_expander_service import SearchExpanderService
    svc = SearchExpanderService()
    svc._self_heal()
    item = svc._dequeue()
    while item is not None:
        svc._process(item)
        item = svc._dequeue()


def test_invalidated_fact_never_surfaces_in_recall(db: sqlite3.Connection) -> None:
    """A bi-temporally superseded data_graph fact (valid_to set) must never surface in recall."""
    import json
    from typing import cast

    from models.fact import FactRow

    FactRow.store("residence", "Valletta", source="test:seed")
    _drain_search_index()

    def _recall_residence_rows() -> list[object]:
        out = _build_user_mp("where do I live").dispatch_service.dispatch(
            "memory", {"action": "recall", "query": "residence city Valletta"}
        )
        # A fully-empty recall is now a loud no-results error — for this
        # test's purpose that IS the empty result set.
        if "code=no-results" in out:
            return []
        head = out.index("]\n") + 2
        tail = out.index("\n[end:memory]")
        return cast("list[object]", json.loads(out[head:tail])["results"])

    # While live, the fact surfaces.
    live_rows = _recall_residence_rows()
    assert any(cast("dict[str, object]", r).get("id") == "residence" for r in live_rows), (
        f"live fact did not surface before invalidation: {live_rows!r}"
    )

    # Bi-temporally close the fact — the exact data state ticket-F supersession
    # produces (valid_to set on the row).
    db.execute("UPDATE data_graph SET valid_to = datetime('now') WHERE key = 'residence'")
    db.commit()

    # The invalidated fact must be gone — never resurface a superseded value.
    after_rows = _recall_residence_rows()
    assert not any(cast("dict[str, object]", r).get("id") == "residence" for r in after_rows), (
        f"invalidated (valid_to-set) fact still surfaced in recall: {after_rows!r}"
    )
