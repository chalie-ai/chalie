"""Test memory.recall guardrail behavior, silent turn-0 seed, and per-lane
relative-floor telemetry. All exercised through the real production
path via ToolDispatcher.dispatch("memory") with zero mocks.

Pinned behaviors:
1. Explicit recall carries a fallback guardrail naming document/schedule tools but fires no fan-out.
2. Turn-0 auto-seed (_auto=True) is silent — no guardrail hint, no fan-out.
3. Recall-log uses floor_cut_count / final_rrf_count fields under the correct caller.
"""

import sqlite3

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.message_processor import MessageProcessor

pytestmark = pytest.mark.unit

_HINT_LEAD = "If you cannot find the information in memory"


def _build_user_mp(text: str) -> MessageProcessor:
    """Builds a real MessageProcessor (constructor pre-allocates the input row, so
    uid anchors act-trail FK) with active_tools seeded — mirrors _setup()'s own seed
    without running the full turn chain."""
    mp = MessageProcessor(UserConfig(), raw_input=text)
    mp.active_tools = list(mp.config.always_available or [])
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

    out = ToolDispatcher(mp).dispatch(
        "memory", {"action": "recall", "query": "what is my home wifi password"}
    )

    # The guardrail is present and names the exact fallback tools + action.
    assert _HINT_LEAD in out
    assert "`document` (action: search)" in out
    assert "`schedule` (action: search)" in out

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


def test_turn0_seed_recall_is_silent_and_logs_seed_telemetry(db: sqlite3.Connection) -> None:
    mp = _build_user_mp("what did we talk about at home last week")

    out = ToolDispatcher(mp).dispatch(
        "memory",
        {"action": "recall", "query": "what did we talk about at home last week", "_auto": True},
    )

    # The silent seed never nags the model with the fallback hint...
    assert _HINT_LEAD not in out, f"seed recall leaked the guardrail hint: {out!r}"
    from typing import cast
    # ...and never fans out to the other stores either.
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


def test_invalidated_fact_never_surfaces_in_recall(db: sqlite3.Connection) -> None:
    """A bi-temporally superseded data_graph fact (valid_to set) must never surface in recall."""
    import json
    from typing import cast

    from services.data_graph_service import get_data_graph_service

    get_data_graph_service().store(
        kind="user_specific", key="residence", value="Valletta", source="test:seed",
    )

    def _recall_residence_rows() -> list[object]:
        out = ToolDispatcher(_build_user_mp("where do I live")).dispatch(
            "memory", {"action": "recall", "query": "residence city Valletta"}
        )
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
