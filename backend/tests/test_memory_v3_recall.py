# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the Memory v3 recall service (Slice D of the redesign).

Drives the real Graph FTS (subject) + Map vec0 (cues) lanes on the real
``db`` fixture with zero mocks of the code under test. The only substitution is
the embedding model (a deterministic 768-d stand-in at the ``get_embedding_service``
seam) so the query vector and the indexed cue vectors come from one source.
Tests that must tell candidates apart take ``lexical_embedder``, whose distances
move with word overlap; the rest keep the equidistant ``fixed_embedder``.
``SearchExpanderService._process_row`` is driven directly so indexing is
synchronous and deterministic.
"""

import json
import sqlite3

import pytest

from contracts.search_config import config_for_table
from models.memory_graph import MemoryGraphRow
from models.memory_map import MemoryMapRow
from services.memory_recall_service import MemoryRecallService
from services.search_expander_service import SearchExpanderService

# No module-level embedder: each test names the one whose distances it relies on,
# so the two never race to patch the same seam.
pytestmark = [pytest.mark.unit]


def _index(table: str, rowid: int) -> None:
    """Drive the real SearchExpanderService indexing for one row, synchronously."""
    cfg = config_for_table(table)
    assert cfg is not None
    SearchExpanderService()._process_row(table, rowid, cfg)


def test_graph_fts_lane_returns_subject_match(
    db: sqlite3.Connection, fixed_embedder: None
) -> None:
    g = MemoryGraphRow(subject="user.residence", contents="Lisbon")
    g.save()
    assert isinstance(g.id, int)
    _index("memory_graph", g.id)

    result = MemoryRecallService().recall("residence", k_graph=3, k_map=0)
    assert len(result["graph"]) == 1
    assert result["graph"][0]["subject"] == "user.residence"
    assert result["graph"][0]["contents"] == "Lisbon"
    assert result["map"] == []


def test_map_vector_lane_excludes_retired_rows(
    db: sqlite3.Connection, fixed_embedder: None
) -> None:
    # A has the highest iteration; retirement must beat salience regardless.
    a = MemoryMapRow(contents="moved to Lisbon years ago", cues="relocation", iteration=5)
    a.save()
    b = MemoryMapRow(contents="settled in Lisbon", cues="relocation", iteration=2)
    b.save()
    # C derives from A -> A leaves the searchable pool (retired, lineage-only).
    c = MemoryMapRow(
        contents="still in Lisbon now",
        cues="relocation",
        iteration=1,
        derived_from=json.dumps([a.id]),
    )
    c.save()
    for row in (a, b, c):
        assert isinstance(row.id, int)
        _index("memory_map", row.id)

    hits = MemoryRecallService().recall("relocation", k_graph=0, k_map=3)["map"]

    assert {h["contents"] for h in hits} == {"settled in Lisbon", "still in Lisbon now"}


def test_map_recall_matches_the_situation_not_the_narrative(
    db: sqlite3.Connection, lexical_embedder: None
) -> None:
    """The whole point of the phase: an episode is found by the situation it is
    ABOUT, not by the words it is written in. The decoy's *contents* are the
    query verbatim — under the old contents-keyed lane it would win outright."""
    wanted = MemoryMapRow(
        contents="his wife never sent the list so he drove back twice",
        source="Grocery trip in Zabbar in May 2026",
        cues="grocery shopping",
    )
    wanted.save()
    decoy = MemoryMapRow(
        contents="grocery shopping downtown was pleasant",
        source="Afternoon walk in June 2026",
        cues="mortgage refinancing",
    )
    decoy.save()
    for row in (wanted, decoy):
        assert isinstance(row.id, int)
        _index("memory_map", row.id)

    hits = MemoryRecallService().recall("grocery shopping", k_graph=0, k_map=2)["map"]

    assert hits[0]["contents"] == "his wife never sent the list so he drove back twice"
    assert hits[0]["source"] == "Grocery trip in Zabbar in May 2026"


def test_map_recall_ranks_by_distance_not_iteration(
    db: sqlite3.Connection, lexical_embedder: None
) -> None:
    """A heavily-consolidated memory no longer outranks the one that fits the
    moment — ``iteration`` is lineage bookkeeping, not relevance."""
    consolidated = MemoryMapRow(
        contents="a much-consolidated memory about money",
        source="Mortgage review in June 2026",
        cues="mortgage refinancing",
        iteration=9,
    )
    consolidated.save()
    fitting = MemoryMapRow(
        contents="forgot half the list again",
        source="Grocery trip in May 2026",
        cues="grocery shopping",
        iteration=1,
    )
    fitting.save()
    for row in (consolidated, fitting):
        assert isinstance(row.id, int)
        _index("memory_map", row.id)

    hits = MemoryRecallService().recall("grocery shopping", k_graph=0, k_map=2)["map"]

    assert [h["iteration"] for h in hits] == [1, 9]


def test_recall_returns_empty_on_miss_without_raising(
    db: sqlite3.Connection, fixed_embedder: None
) -> None:
    result = MemoryRecallService().recall("nothing matches this query")
    assert result == {"graph": [], "map": []}
