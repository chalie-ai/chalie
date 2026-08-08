"""Test recall behavior: empty recall returns clear empty-body success,
no fan-out. All exercised through the real production path via
DispatchService.dispatch("recall") with zero mocks.

Pinned behaviors:
1. A recall that finds nothing returns status=success with a clear "No existing
   memory found." body — no fan-out.
2. A recall that surfaces memories carries the same body format, regardless of
   caller; that positive path needs the live embedding pipeline and is covered
   by live-fire, not this suite.
"""

import sqlite3

import pytest

from configs.channels import UserConfig
from controllers.message_processor import MessageProcessor

pytestmark = pytest.mark.unit


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


def test_explicit_recall_returns_no_memory_and_fires_no_fanout(db: sqlite3.Connection) -> None:
    mp = _build_user_mp("what is my home wifi password")

    out = mp.dispatch_service.dispatch(
        "recall", {"query": "what is my home wifi password"}
    )

    # An empty explicit recall returns status=success with a clear body — the new
    # model never returns code=no-results; it just surfaces the empty-body result.
    assert "status=success" in out
    assert "No existing memory found." in out
    # The old guardrail hint and find_tools mention are gone with the new recall.
    assert "HARD RULE" not in out
    assert "find_tools" not in out
    assert "document" not in out.lower()

    # Fan-out is gone: the removed code dispatched document.search + schedule.search,
    # each of which would have recorded a tool_calls row under this transcript.
    from typing import cast
    names = _tool_names_recorded(db, cast(int, mp.uid))
    assert "recall" in names, f"the recall itself was not recorded: {names!r}"
    assert "document" not in names, f"document.search fan-out still firing: {names!r}"
    assert "schedule" not in names, f"schedule.search fan-out still firing: {names!r}"


def test_turn0_seed_recall_returns_no_memory_and_fires_no_fanout(db: sqlite3.Connection) -> None:
    mp = _build_user_mp("what did we talk about at home last week")

    out = mp.dispatch_service.dispatch(
        "recall",
        {"query": "what did we talk about at home last week", "_auto": True},
    )

    # Empty seed returns the same success-with-empty-body as any recall.
    assert "status=success" in out, f"empty seed must return success: {out!r}"
    assert "No existing memory found." in out, f"empty seed must show empty body: {out!r}"
    from typing import cast
    # ...and never fans out to the other stores.
    names = _tool_names_recorded(db, cast(int, mp.uid))
    assert "document" not in names and "schedule" not in names, (
        f"seed recall fanned out to other stores: {names!r}"
    )
