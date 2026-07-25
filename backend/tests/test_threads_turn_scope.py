"""Regression guard for ``Transcript.turn_scope_ids`` — the turn-scope resolver
the thread-block render path (``TurnSerializerService``'s row projection) uses to widen a
set of transcript ids into every id of the turn(s) they belong to.

The multi-turn branch was previously hand-rolled in the API layer with a
per-pair loop; migrating it onto the model introduced a row-value ``IN`` clause
that MUST parenthesise each ``(channel, turn_id)`` element — a flat ``?,?`` list
raises ``sqlite3.OperationalError: IN(...) element has 1 term - expected 2``.
These tests drive the real model against the DB fixture so that regression can
never return silently.
"""

import sqlite3

import pytest

from models.transcript import Transcript

pytestmark = pytest.mark.integration


def _seed(db: sqlite3.Connection, channel: str, turn_id: int | None, role: str = "user") -> int:
    """Insert one transcript row with an explicit ``turn_id`` and return its id."""
    db.execute(
        "INSERT INTO transcript (channel, role, content, turn_id, xml_migrated) "
        "VALUES (?, ?, 'x', ?, 1)",
        (channel, role, turn_id),
    )
    return int(db.execute("SELECT last_insert_rowid()").fetchone()[0])


def test_widens_across_multiple_turns(db: sqlite3.Connection) -> None:
    """A mix of ids from two turns resolves to EVERY id of both turns — the
    row-value ``IN`` branch that raised before the parenthesised-element fix."""
    a1 = _seed(db, "user", 10)
    a2 = _seed(db, "user", 10, role="assistant")
    b1 = _seed(db, "user", 20)
    b2 = _seed(db, "user", 20, role="assistant")
    db.commit()

    # Pass one id from each turn; expect the full span of both turns back.
    scope = Transcript.turn_scope_ids([a1, b1])

    assert sorted(scope) == sorted([a1, a2, b1, b2])


def test_null_turn_ids_fall_back_to_input(db: sqlite3.Connection) -> None:
    """Rows with a NULL turn_id (skip_transcript channels) have no turn to widen
    into, so the input list is returned unchanged rather than dropped."""
    n1 = _seed(db, "subagent", None, role="subagent")
    n2 = _seed(db, "subagent", None, role="subagent")
    db.commit()

    assert Transcript.turn_scope_ids([n1, n2]) == [n1, n2]


def test_empty_input_short_circuits(db: sqlite3.Connection) -> None:
    assert Transcript.turn_scope_ids([]) == []
