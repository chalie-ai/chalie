"""Feature test: every transcript row a thread writes points at that thread.

A thread is a row of its own, made by the thread's first transcript row; every
row after it — the question, a message sent into the working turn, the answers,
a later reply — points at the same thread, and a new thread gets its own. The
test drives the real send endpoint (``POST /api/threads/<turn_id>``), the real
``MessageProcessor`` loop and the real transcript writer against the real
SQLite database, with the LLM network boundary scripted by the provider double
the join tests use.
"""

import sqlite3
from typing import cast

import pytest

from tests.test_turn_join import (
    AuthedClient,
    _drain,
    _label_thread,
    _post,
    _Provider,
    _result,
    _say,
    _start_working_turn,
    provider,  # noqa: F401 — the provider double fixture, used by name below
)

pytestmark = pytest.mark.unit

_CHANNEL = "user"


def _thread_of_each_row(db: sqlite3.Connection, turn_id: int) -> set[int | None]:
    rows = db.execute(
        "SELECT thread_id FROM transcript WHERE channel = ? AND turn_id = ?",
        (_CHANNEL, turn_id),
    ).fetchall()
    return {r["thread_id"] for r in rows}


def _thread(db: sqlite3.Connection, turn_id: int) -> int:
    row = db.execute(
        "SELECT id FROM threads WHERE channel = ? AND turn_id = ?", (_CHANNEL, turn_id)
    ).fetchone()
    assert row is not None, f"turn {turn_id} has no thread"
    return int(row["id"])


def test_every_row_a_thread_writes_points_at_it_and_a_new_thread_gets_its_own(
    authed_client: AuthedClient,
    provider: _Provider,  # noqa: F811
) -> None:
    """A question, a message sent into it while it works, its answers and a
    later reply all point at one thread; the next new thread gets another."""
    client, db, _store = authed_client
    provider.script(_say("First answer."), _say("Updated answer."))
    opened = _start_working_turn(client, provider, "question")
    turn_id = cast("int", opened["turn_id"])
    _result(_post(client, turn_id, "one more thing", join="1"))
    provider.release.set()
    _drain()
    _label_thread(turn_id)
    provider.script(_say("Reply answer."))
    _result(_post(client, turn_id, "follow up"))
    _drain()
    provider.script(_say("Other answer."))
    other = cast("int", _result(_post(client, -1, "another question"))["turn_id"])
    _drain()

    contents = {
        r["content"]
        for r in db.execute(
            "SELECT content FROM transcript WHERE channel = ? AND turn_id = ?",
            (_CHANNEL, turn_id),
        )
    }
    assert {
        "question",
        "one more thing",
        "Updated answer.",
        "follow up",
        "Reply answer.",
    } <= contents
    assert _thread_of_each_row(db, turn_id) == {_thread(db, turn_id)}, (
        "a row of the thread does not point at it"
    )
    assert _thread_of_each_row(db, other) == {_thread(db, other)}
    assert _thread(db, other) != _thread(db, turn_id), (
        "a new thread did not get its own row"
    )
