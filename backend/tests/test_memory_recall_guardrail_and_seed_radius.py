"""Feature test: memory.recall guardrail + the 30%-narrower turn-0 seed radius.

Two behaviour changes are proven here against the REAL production chokepoint —
``ToolDispatcher(mp).dispatch("memory", …)`` — the single path BOTH the model's
explicit recall and the turn-0 auto-seed take in production:

1. **Fan-out removed → guardrail in.** ``memory.recall`` no longer silently
   dispatches ``document.search`` / ``schedule.search`` behind the model's back.
   An EXPLICIT recall instead carries a guardrail line steering the model to
   those stores ON ITS OWN judgement, naming the exact tools (``document`` /
   ``schedule``, action ``search``). The silent turn-0 seed (``_auto=True``)
   carries NO hint and fires NO fan-out. We assert the fan-out is gone by proving
   no ``document`` / ``schedule`` rows were recorded on the act-trail — the
   removed code dispatched those tools through a ``ToolDispatcher``, which records
   one ``tool_calls`` row per dispatch.

2. **Seed radius cut 30%.** The turn-0 seed retrieves at
   ``SEED_RADIUS_BASELINE = 0.35`` (``RECALL_RADIUS_BASELINE 0.5 − 30%``); an
   explicit recall stays at ``0.5``. We read this back from the real
   ``memory_recall_log`` telemetry the recall writes.

Zero mocks: real ``UserConfig`` MessageProcessor, real ``ToolDispatcher``, real
embedding model, real DataGraph, real SQLite. The only thing asserted is what the
production code actually wrote to the db and returned to the loop.
"""

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.memory import MemoryAbility
from configs.channels import UserConfig
from services.message_processor import MessageProcessor
from services.transcript_service import write_input_row

pytestmark = pytest.mark.unit

_HINT_LEAD = "If you cannot find the information in memory"


def _build_user_mp(text: str) -> MessageProcessor:
    """A real UserConfig MessageProcessor in the state a recall dispatches from:
    an input row written (so ``uid`` anchors the act-trail FK) and ``active_tools``
    seeded — the exact shape ``_seed_turn_zero`` and the ACT loop fire from."""
    parent = object.__new__(MessageProcessor)
    MessageProcessor.__init__(parent, text, {})
    parent.config = UserConfig()
    parent.uid = write_input_row("user", "user", text)
    parent.active_tools = list(parent.config.always_available or [])
    return parent


def _tool_names_recorded(db, transcript_id: int) -> list:
    rows = db.execute(
        "SELECT tool_name FROM tool_calls WHERE transcript_id = ?",
        (transcript_id,),
    ).fetchall()
    return [r[0] for r in rows]


def test_explicit_recall_carries_guardrail_and_fires_no_fanout(db):
    """An explicit (model-invoked) recall returns the fallback guardrail naming
    the real tools, records its own ``memory`` call but NO ``document`` /
    ``schedule`` fan-out, and logs the wide ``0.5`` radius."""
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
    names = _tool_names_recorded(db, mp.uid)
    assert "memory" in names, f"the recall itself was not recorded: {names!r}"
    assert "document" not in names, f"document.search fan-out still firing: {names!r}"
    assert "schedule" not in names, f"schedule.search fan-out still firing: {names!r}"

    # Explicit recall retrieves at the WIDE baseline — untouched by the seed cut.
    tel = db.execute(
        "SELECT caller, input_radius FROM memory_recall_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert tel is not None, "explicit recall wrote no telemetry row"
    assert tel[0] == "llm_recall"
    assert abs(tel[1] - 0.5) < 1e-9, f"explicit radius drifted from 0.5: {tel[1]}"


def test_turn0_seed_recall_is_silent_and_30pct_narrower(db):
    """The turn-0 auto-seed (``_auto=True``) carries NO guardrail, fires no
    fan-out, and retrieves at the 30%-narrower ``0.35`` seed radius."""
    mp = _build_user_mp("what did we talk about at home last week")

    out = ToolDispatcher(mp).dispatch(
        "memory",
        {"action": "recall", "query": "what did we talk about at home last week", "_auto": True},
    )

    # The silent seed never nags the model with the fallback hint...
    assert _HINT_LEAD not in out, f"seed recall leaked the guardrail hint: {out!r}"
    # ...and never fans out to the other stores either.
    names = _tool_names_recorded(db, mp.uid)
    assert "document" not in names and "schedule" not in names, (
        f"seed recall fanned out to other stores: {names!r}"
    )

    # The seed retrieves at SEED_RADIUS_BASELINE = 0.35.
    tel = db.execute(
        "SELECT caller, input_radius FROM memory_recall_log WHERE caller = 'seed' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert tel is not None, "seed recall wrote no telemetry row"
    assert tel[0] == "seed"
    assert abs(tel[1] - 0.35) < 1e-9, f"seed radius is not 0.35: {tel[1]}"

    # Lock the exact 30% relationship so a future baseline edit can't silently
    # break the cut without tripping this test.
    assert abs(
        MemoryAbility.SEED_RADIUS_BASELINE - MemoryAbility.RECALL_RADIUS_BASELINE * 0.7
    ) < 1e-9, (
        f"seed baseline {MemoryAbility.SEED_RADIUS_BASELINE} is not 30% below "
        f"recall baseline {MemoryAbility.RECALL_RADIUS_BASELINE}"
    )
