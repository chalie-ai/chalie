"""Feature tests — off-turn GC of orphaned ``tool_calls`` rows.

Retention deletes ``transcript`` rows out from under the tool calls that
anchored to them, leaving dangling ``transcript_id``s no live read can reach
(every read joins ``transcript``). ``ToolCall.decay`` — fronted by the
``ToolCallGcService`` Decayable registered on the ``DecayEngine`` — reaps those
orphans once they age past the 14-day window, and NOTHING else: a still-anchored
call survives however old, and a young orphan waits out the window.

Real migrated SQLite (the ``db`` template), real model/engine, zero mocks. The
orphan rows are seeded exactly as reality produces them — a tool call whose
``transcript_id`` has no ``transcript`` row — with foreign-key enforcement off
for seeding, mirroring the drifted live state that carries such orphans.
"""

import sqlite3
from datetime import timedelta
from typing import cast

import pytest

from models.tool_call import ToolCall
from services.decay_engine_service import DecayEngineService
from services.time_utils import utc_now
from services.tool_call_gc_service import ToolCallGcService

pytestmark = pytest.mark.unit


def _insert_call(db: sqlite3.Connection, transcript_id: int, created_sql: str) -> int:
    """Insert one tool_calls row anchored to ``transcript_id`` with ``created_at``
    given by a SQL expression (a literal or ``datetime('now', ...)``); return its
    id. Foreign keys are left to the caller (off, so a dangling id is allowed)."""
    cur = db.execute(
        "INSERT INTO tool_calls (transcript_id, tool_name, params, result, summary, "
        f"created_at, ended_at, state) VALUES (?, 'web_search', '{{}}', 'r', 's', "
        f"{created_sql}, NULL, 'done')",
        (transcript_id,),
    )
    return cast(int, cur.lastrowid)


def _seed(db: sqlite3.Connection) -> dict[str, int]:
    """Seed the age×anchor matrix and return the row ids by scenario key.

    Two orphans past the window (``orphan_15d``, ``orphan_400d``) must be reaped;
    everything else — a young orphan, an ancient still-anchored call, a fresh
    anchored call — must survive.
    """
    db.execute("PRAGMA foreign_keys=OFF")  # seed dangling-anchor orphans as reality has them
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES ('user', 'user', 'hi')"
    )
    live_transcript = cast(int, cur.lastrowid)
    # A transcript row that will be purged, orphaning the call anchored to it.
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES ('user', 'user', 'bye')"
    )
    doomed_transcript = cast(int, cur.lastrowid)

    ids = {
        # orphan (transcript never existed) + past the window → REAPED
        "orphan_15d": _insert_call(db, 900001, "datetime('now', '-15 days')"),
        "orphan_400d": _insert_call(db, 900002, "datetime('now', '-400 days')"),
        # orphan but still inside the window → SURVIVES
        "orphan_13d": _insert_call(db, 900003, "datetime('now', '-13 days')"),
        # still anchored, however old → SURVIVES
        "anchored_100d": _insert_call(db, live_transcript, "datetime('now', '-100 days')"),
        # anchored + fresh → SURVIVES
        "anchored_now": _insert_call(db, live_transcript, "datetime('now')"),
        # anchored-then-orphaned + old → REAPED once its transcript is purged below
        "will_orphan_30d": _insert_call(db, doomed_transcript, "datetime('now', '-30 days')"),
    }
    db.execute("DELETE FROM transcript WHERE id = ?", (doomed_transcript,))
    db.commit()
    return ids


def _insert_call_at(db: sqlite3.Connection, transcript_id: int, created_iso: str) -> int:
    """Insert one tool_calls row with ``created_at`` bound as a literal ISO string
    — the exact production shape ``utc_now().isoformat()`` emits — and return its
    id. Distinct from :func:`_insert_call` (which inlines a SQL expression)."""
    cur = db.execute(
        "INSERT INTO tool_calls (transcript_id, tool_name, params, result, summary, "
        "created_at, ended_at, state) VALUES (?, 'web_search', '{}', 'r', 's', ?, NULL, 'done')",
        (transcript_id, created_iso),
    )
    return cast(int, cur.lastrowid)


def _surviving_ids(db: sqlite3.Connection) -> set[int]:
    return {row[0] for row in db.execute("SELECT id FROM tool_calls")}


class TestDecay:
    def test_reaps_only_aged_orphans(self, db: sqlite3.Connection) -> None:
        ids = _seed(db)
        deleted = ToolCall.decay()

        assert deleted == 3  # orphan_15d, orphan_400d, will_orphan_30d
        survivors = _surviving_ids(db)
        assert survivors == {ids["orphan_13d"], ids["anchored_100d"], ids["anchored_now"]}

    def test_keeps_anchored_calls_however_old(self, db: sqlite3.Connection) -> None:
        ids = _seed(db)
        ToolCall.decay()
        # The 100-day-old call is far past the window but still anchored — kept.
        assert ids["anchored_100d"] in _surviving_ids(db)

    def test_keeps_young_orphans(self, db: sqlite3.Connection) -> None:
        ids = _seed(db)
        ToolCall.decay()
        # Orphaned but only 13 days old — inside the 14-day window, kept.
        assert ids["orphan_13d"] in _surviving_ids(db)

    def test_idempotent(self, db: sqlite3.Connection) -> None:
        _seed(db)
        assert ToolCall.decay() == 3
        assert ToolCall.decay() == 0  # nothing left past the window
        assert ToolCall.decay() == 0

    def test_returns_zero_on_clean_table(self, db: sqlite3.Connection) -> None:
        # A single well-anchored fresh call — nothing to sweep.
        cur = db.execute(
            "INSERT INTO transcript (channel, role, content) VALUES ('user', 'user', 'hi')"
        )
        _insert_call(db, cast(int, cur.lastrowid), "datetime('now')")
        db.commit()
        assert ToolCall.decay() == 0


class TestServiceDelegation:
    def test_gc_service_delegates_to_model(self, db: sqlite3.Connection) -> None:
        ids = _seed(db)
        deleted = ToolCallGcService().decay()

        assert deleted == 3
        assert _surviving_ids(db) == {
            ids["orphan_13d"], ids["anchored_100d"], ids["anchored_now"],
        }


class TestEngineIntegration:
    def test_decay_cycle_reaps_aged_orphans(self, db: sqlite3.Connection) -> None:
        # Proves the sweep is actually wired into the registered decay cycle,
        # not merely callable in isolation.
        ids = _seed(db)
        DecayEngineService().run_once()

        survivors = _surviving_ids(db)
        assert ids["orphan_15d"] not in survivors
        assert ids["orphan_400d"] not in survivors
        assert ids["will_orphan_30d"] not in survivors
        assert ids["orphan_13d"] in survivors
        assert ids["anchored_100d"] in survivors


class TestProductionTimestampFormat:
    """Rows land with ``created_at = utc_now().isoformat()`` — ISO-8601 carrying
    microseconds and a ``+00:00`` offset, not SQLite's space format. The window
    cutoff must parse THAT shape or the sweep silently no-ops in production while
    space-format fixtures stay green. This locks the real format into the suite."""

    def test_reaps_aged_orphan_written_in_iso_offset_format(self, db: sqlite3.Connection) -> None:
        db.execute("PRAGMA foreign_keys=OFF")
        old_iso = "2020-06-01T12:00:00.500000+00:00"  # far past the window
        young_iso = (utc_now() - timedelta(days=2)).isoformat()  # inside the window
        old_orphan = _insert_call_at(db, 900101, old_iso)
        young_orphan = _insert_call_at(db, 900102, young_iso)
        db.commit()

        assert ToolCall.decay() == 1
        survivors = _surviving_ids(db)
        assert old_orphan not in survivors  # ISO-offset timestamp parsed and aged out
        assert young_orphan in survivors
