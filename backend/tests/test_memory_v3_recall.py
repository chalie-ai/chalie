# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the Memory v3 recall service (Slice D of the redesign).

Drives the real Graph FTS (subject) + Map vec0 (contents) lanes on the real
``db`` fixture with zero mocks of the code under test. The only substitution is
the embedding model (a deterministic 768-d stand-in at the ``get_embedding_service``
seam) so the query vector and the indexed contents vectors come from one source.
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

pytestmark = pytest.mark.unit

_DIM = 768


def _unit(index: int) -> list[float]:
    vec = [0.0] * _DIM
    vec[index] = 1.0
    return vec


class _FixedEmbedder:
    """Deterministic stand-in for the embedding model: every text maps to the
    same 768-d unit vector, matching the vec-table dimension. All map rows end
    up equidistant from the query, so ranking is decided by ``iteration``."""

    def generate_embedding(self, text: str, mp: object = None) -> list[float]:
        return _unit(0)


@pytest.fixture
def fixed_embedder(monkeypatch: pytest.MonkeyPatch) -> _FixedEmbedder:
    emb = _FixedEmbedder()
    monkeypatch.setattr(
        "services.embedding_service.get_embedding_service", lambda: emb
    )
    monkeypatch.setattr(
        "services.memory_recall_service.get_embedding_service", lambda: emb
    )
    return emb


def _index(table: str, rowid: int) -> None:
    """Drive the real SearchExpanderService indexing for one row, synchronously."""
    cfg = config_for_table(table)
    assert cfg is not None
    SearchExpanderService()._process_row(table, rowid, cfg)


def test_graph_fts_lane_returns_subject_match(fixed_embedder: _FixedEmbedder, db: sqlite3.Connection) -> None:
    g = MemoryGraphRow(subject="user.residence", contents="Lisbon")
    g.save()
    assert isinstance(g.id, int)
    _index("memory_graph", g.id)

    result = MemoryRecallService().recall("residence", k_graph=3, k_map=0)
    assert len(result["graph"]) == 1
    assert result["graph"][0]["subject"] == "user.residence"
    assert result["graph"][0]["contents"] == "Lisbon"
    assert result["map"] == []


def test_map_vector_lane_excludes_retired_rows_and_ranks_by_iteration(
    fixed_embedder: _FixedEmbedder, db: sqlite3.Connection
) -> None:
    # A has the HIGHEST iteration; if it weren't retired it would rank first.
    a = MemoryMapRow(contents="moved to Lisbon years ago", iteration=5)
    a.save()
    b = MemoryMapRow(contents="settled in Lisbon", iteration=2)
    b.save()
    # C derives from A -> A leaves the searchable pool (retired, lineage-only).
    c = MemoryMapRow(
        contents="still in Lisbon now", iteration=1, derived_from=json.dumps([a.id])
    )
    c.save()
    for row in (a, b, c):
        assert isinstance(row.id, int)
        _index("memory_map", row.id)

    hits = MemoryRecallService().recall("Lisbon", k_graph=0, k_map=3)["map"]

    # A (iteration 5) must NOT surface — it's retired by C's derived_from.
    iterations = [h["iteration"] for h in hits]
    assert 5 not in iterations
    # Remaining pool {B(2), C(1)} ranked by iteration DESC -> B before C.
    assert iterations == [2, 1]
    assert hits[0]["contents"] == "settled in Lisbon"


def test_recall_returns_empty_on_miss_without_raising(fixed_embedder: _FixedEmbedder, db: sqlite3.Connection) -> None:
    result = MemoryRecallService().recall("nothing matches this query")
    assert result == {"graph": [], "map": []}
