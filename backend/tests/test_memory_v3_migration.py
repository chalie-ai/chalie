# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test for the Memory v3 migration (Slice E) — the consolidator loop.

The migration is a thin loop that fires the consolidator over every transcript
turn in order. This test spies on ``MemoryConsolidatorService.consolidate`` (the
consolidator itself is covered by test_memory_v3_consolidator.py) and asserts the
loop visits every non-excluded turn in (channel, turn_id) order, skips excluded
channels, and repairs legacy NULL turn_id.
"""

import pytest

from services.memory_consolidator_service import MemoryConsolidatorService
from services.memory_v3_migration import MemoryV3Migration

pytestmark = pytest.mark.unit


def _seed_turn(db, channel: str, turn_id: int, content: str = "x") -> None:
    db.execute(
        "INSERT INTO transcript (channel, role, content, turn_id) "
        "VALUES (?, ?, ?, ?)",
        (channel, "user", content, turn_id),
    )


def test_migration_replays_every_turn_in_order_skipping_excluded(
    db, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_turn(db, "user", 1, "first")
    _seed_turn(db, "user", 2, "second")
    _seed_turn(db, "dmn", 1, "reflection")
    # Excluded channels must be skipped.
    _seed_turn(db, "delegate:web_search", 1, "delegate work")
    _seed_turn(db, "memory_consolidator", 1, "own output")
    db.commit()

    calls: list[tuple[str, int]] = []

    def spy(self, channel: str, turn_id: int) -> str:
        calls.append((channel, turn_id))
        return f"{channel}:{turn_id} consolidated"

    monkeypatch.setattr(MemoryConsolidatorService, "consolidate", spy)

    counts = MemoryV3Migration().run()

    assert calls == [("dmn", 1), ("user", 1), ("user", 2)]
    assert counts["turns_consolidated"] == 3


def test_migration_repairs_legacy_null_turn_id(db, monkeypatch: pytest.MonkeyPatch) -> None:
    # A legacy NULL turn_id row is repaired to -id so it joins the loop.
    db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("user", "user", "legacy"),
    )
    db.commit()

    monkeypatch.setattr(
        MemoryConsolidatorService, "consolidate", lambda self, c, t: f"{c}:{t} consolidated"
    )
    MemoryV3Migration().run()

    row = db.execute(
        "SELECT turn_id FROM transcript WHERE content = 'legacy'"
    ).fetchone()
    assert row is not None
    assert row[0] is not None and int(row[0]) < 0
