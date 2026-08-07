# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test for the Memory v3 one-time migration (Slice E).

Seeds old-shape ``data_graph`` + ``episodes`` rows and a legacy NULL ``turn_id``
transcript row on the real ``db`` fixture, runs the migration, and asserts the
new Graph/Map stores are backfilled (active facts only; iteration from level;
provenance from transcript_ids) and the legacy turn_id is repaired.
"""

import json

import pytest

from models.memory_graph import MemoryGraphRow
from models.memory_map import MemoryMapRow
from services.memory_v3_migration import MemoryV3Migration

pytestmark = pytest.mark.unit


def test_migration_backfills_graph_and_map_and_repairs_turn_id(db) -> None:
    db.execute(
        "INSERT INTO data_graph (kind, key, value) VALUES (?, ?, ?)",
        ("user_specific", "residence", "Lisbon"),
    )
    db.execute(
        "INSERT INTO data_graph (kind, key, value) VALUES (?, ?, ?)",
        ("user_specific", "partner", "Ana"),
    )
    # An inactive (superseded) row must be skipped.
    db.execute(
        "INSERT INTO data_graph (kind, key, value, active) VALUES (?, ?, ?, 0)",
        ("user_specific", "old", "gone"),
    )
    db.execute(
        "INSERT INTO episodes (gist, channel, level, salience, transcript_ids) "
        "VALUES (?, ?, ?, ?, ?)",
        ("moved to Lisbon", "user", 0, 6, "[10, 11]"),
    )
    db.execute(
        "INSERT INTO episodes (gist, channel, level, salience) VALUES (?, ?, ?, ?)",
        ("era: settled abroad", "user", 1, 6),
    )
    # Legacy NULL turn_id row.
    db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("user", "user", "hi"),
    )
    db.commit()

    counts = MemoryV3Migration().run()
    assert counts["graph_rows"] >= 2
    assert counts["map_rows"] >= 2
    assert counts["turns_backfilled"] >= 1

    # Active facts -> Graph (subject = kind.key); the inactive one is skipped.
    residence = MemoryGraphRow.by_subject("user_specific.residence")
    assert residence is not None and residence.contents == "Lisbon"
    assert MemoryGraphRow.by_subject("user_specific.partner") is not None
    assert MemoryGraphRow.by_subject("user_specific.old") is None

    # Episodes -> Map; iteration = level + 1; provenance carried from transcript_ids.
    maps = MemoryMapRow.recent(limit=20)
    by_content = {m.contents: m for m in maps}
    assert "moved to Lisbon" in by_content
    assert "era: settled abroad" in by_content
    assert by_content["moved to Lisbon"].iteration == 1
    assert by_content["era: settled abroad"].iteration == 2
    assert json.loads(by_content["moved to Lisbon"].sourced_from) == [10, 11]

    # Legacy NULL turn_id repaired to -id (< 0).
    row = db.execute(
        "SELECT turn_id FROM transcript WHERE content = 'hi'"
    ).fetchone()
    assert row is not None
    assert row[0] is not None and int(row[0]) < 0
