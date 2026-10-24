"""Feature test: which delegate turn a tool call is credited with spawning.

``Transcript.delegate_turns`` answers, for a batch of ``tool_calls`` ids, "which
watchable child turn did each one spawn?" — read off the child's input row, whose
``tool_call_id`` holds the caller's id. The ``tool_call_id`` column is unindexed,
so the read is bounded to the batch's time window (earliest open to latest end,
plus slack for a child row that lands after its call ended) and walks the
``(channel, created_at)`` index instead. The window has to be tight enough to be
fast and loose enough never to lose a real child; each test here is a real
situation where getting that wrong either hides a live link or invents one.

Real rows go into the real, schema-built ``transcript`` table (``db``) and the
real query runs against them. Rows are inserted with explicit ``created_at``
because the boundary cases need stamps production never writes by hand.
"""

import sqlite3

import pytest

from models.transcript import Transcript

pytestmark = pytest.mark.unit

_CHANNEL = "delegate:web_search"
_CALL_ID = 7


def _child_row(
    db: sqlite3.Connection, created_at: str, *, call_id: int = _CALL_ID, channel: str = _CHANNEL,
    turn_id: int = 3,
) -> None:
    """A delegate input row stamped with *call_id*, written at *created_at* in the
    shape SQLite's ``datetime('now')`` gives the column (whole-second UTC)."""
    db.execute(
        "INSERT INTO transcript (channel, role, content, tool_call_id, turn_id, created_at) "
        "VALUES (?, 'web_search', 'find it', ?, ?, ?)",
        (channel, str(call_id), turn_id, created_at),
    )
    db.commit()


def _spawned(call: tuple[str, str | None], call_id: int = _CALL_ID) -> dict[str, object] | None:
    """What the real lookup credits *call_id* with, given its (opened, ended)."""
    return Transcript.delegate_turns({call_id: call}).get(call_id)


def test_child_row_stamped_in_the_second_the_call_opened_is_found(db: sqlite3.Connection) -> None:
    """The call opens at ``10:00:00.9`` (microsecond stamp) and the child's row is
    stamped ``10:00:00`` — SQLite truncates to whole seconds, so the row looks
    EARLIER than the call that caused it. It must still be found."""
    _child_row(db, "2026-03-01 10:00:00")

    found = _spawned(("2026-03-01T10:00:00.900000+00:00", "2026-03-01T10:00:05+00:00"))

    assert found == {"channel": _CHANNEL, "turn_id": 3}


def test_child_row_landing_after_a_crash_backfilled_call_is_found(db: sqlite3.Connection) -> None:
    """A call the boot sweep backfilled after a crash has ``ended_at`` equal to its
    ``created_at`` even though the child it spawned wrote its row a few seconds
    later. That row must still resolve, not vanish with the crash."""
    _child_row(db, "2026-03-01 10:00:15")

    found = _spawned(("2026-03-01T10:00:00+00:00", "2026-03-01T10:00:00+00:00"))

    assert found == {"channel": _CHANNEL, "turn_id": 3}


def test_call_still_running_resolves_to_a_child_written_long_after_it_opened(db: sqlite3.Connection) -> None:
    """A call with no ``ended_at`` yet has no upper bound — a long-running
    delegate's pill must keep pointing at its child however late the row landed,
    even when the same batch holds a call that finished minutes earlier."""
    _child_row(db, "2026-03-01 10:30:00", call_id=7, turn_id=3)
    _child_row(db, "2026-03-01 10:00:02", call_id=8, turn_id=4)

    found = Transcript.delegate_turns({
        7: ("2026-03-01T10:00:00+00:00", None),
        8: ("2026-03-01T10:00:01+00:00", "2026-03-01T10:00:05+00:00"),
    })

    assert found == {7: {"channel": _CHANNEL, "turn_id": 3}, 8: {"channel": _CHANNEL, "turn_id": 4}}


@pytest.mark.parametrize(
    "child_stamp",
    ["2026-03-01 08:00:00", "2026-03-01 12:00:00"],
    ids=["hours-before-the-call", "hours-after-it-ended"],
)
def test_matching_stamp_outside_the_calls_window_is_not_credited_to_it(
    db: sqlite3.Connection, child_stamp: str,
) -> None:
    """A row carrying the call's id but written hours before the call opened or
    after it ended cannot have been spawned by it, so it is not linked."""
    _child_row(db, child_stamp)

    assert _spawned(("2026-03-01T10:00:00+00:00", "2026-03-01T10:00:05+00:00")) is None


def test_each_call_is_credited_only_with_the_child_stamped_with_its_own_id(db: sqlite3.Connection) -> None:
    """In a batch, each call is credited with its OWN child, on that child's own
    channel: another call's child is not borrowed, and a call nothing was stamped
    for resolves to no child."""
    _child_row(db, "2026-03-01 10:00:02", call_id=7, turn_id=3)
    _child_row(db, "2026-03-01 10:00:03", call_id=8, turn_id=4, channel="delegate:pim")
    opened, ended = "2026-03-01T10:00:00+00:00", "2026-03-01T10:00:10+00:00"

    found = Transcript.delegate_turns({7: (opened, ended), 8: (opened, ended), 10: (opened, ended)})

    assert found == {
        7: {"channel": _CHANNEL, "turn_id": 3},
        8: {"channel": "delegate:pim", "turn_id": 4},
    }


def test_call_timestamps_are_compared_in_utc(db: sqlite3.Connection) -> None:
    """``tool_calls.created_at`` is an aware ISO stamp and the column default is a
    naive one; the child row's stamp is UTC. A naive stamp is read as UTC, and an
    offset stamp is converted to it — ``12:00+02:00`` is ``10:00`` UTC, so the row
    written at ``10:00:05`` belongs to it."""
    _child_row(db, "2026-03-01 10:00:05")

    naive = _spawned(("2026-03-01 10:00:00", "2026-03-01 10:00:09"))
    offset = _spawned(("2026-03-01T12:00:00+02:00", "2026-03-01T12:00:09+02:00"))

    assert naive == offset == {"channel": _CHANNEL, "turn_id": 3}


def test_unparseable_call_timestamp_fails_loudly(db: sqlite3.Connection) -> None:
    """A call whose ``created_at`` cannot be read raises rather than quietly
    resolving to "no child" — the pill would otherwise lose its link with nothing
    to say why."""
    _child_row(db, "2026-03-01 10:00:05")

    with pytest.raises(ValueError):
        _spawned(("not-a-timestamp", None))
