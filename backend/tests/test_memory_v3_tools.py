# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the Memory v3 consolidator tools (Slice B).

Exercises the four abilities end-to-end on the real ``db`` fixture: each tool's
param bag (validation) + run() (store mutation), driving the real
SearchExpanderService indexing where a vector lane is involved. The embedding
model is stubbed deterministically (same seam as test_memory_v3_recall.py).
"""

import json
import sqlite3
from typing import TypeVar, cast

import pytest

from abilities.delete_graph import DeleteGraph
from abilities.recall import Recall
from abilities.save_graph import SaveGraph
from abilities.save_map import SaveMap
from abilities._result import ToolResult
from contracts.params.delete_graph_params_bag import DeleteGraphParamsBag
from contracts.params.param_bag import ParamBag
from contracts.params.recall_params_bag import RecallParamsBag
from contracts.params.save_graph_params_bag import SaveGraphParamsBag
from contracts.params.save_map_params_bag import SaveMapParamsBag
from contracts.search_config import config_for_table
from models.memory_graph import MemoryGraphRow
from models.memory_map import MemoryMapRow
from services.search_expander_service import SearchExpanderService

pytestmark = pytest.mark.unit

_DIM = 768


def _unit(index: int) -> list[float]:
    vec = [0.0] * _DIM
    vec[index] = 1.0
    return vec


class _FixedEmbedder:
    def generate_embedding(self, text: str, mp: object = None) -> list[float]:
        return _unit(0)


@pytest.fixture
def fixed_embedder(monkeypatch: pytest.MonkeyPatch) -> _FixedEmbedder:
    emb = _FixedEmbedder()
    monkeypatch.setattr(
        "services.embedding_service.get_embedding_service", lambda: emb
    )
    monkeypatch.setattr(
        "services.memory_v3_recall_service.get_embedding_service", lambda: emb
    )
    return emb


def _index(table: str, rowid: int) -> None:
    cfg = config_for_table(table)
    assert cfg is not None
    SearchExpanderService()._process_row(table, rowid, cfg)


_T = TypeVar("_T", bound=ParamBag)


def _bag(bag_cls: type[_T], params: dict[str, object]) -> _T:
    bag = bag_cls.from_params(params)
    assert not isinstance(bag, ToolResult)
    return cast("_T", bag)


def test_save_graph_writes_subject_keyed_upsert(fixed_embedder: _FixedEmbedder, db: sqlite3.Connection) -> None:
    SaveGraph().run(_bag(SaveGraphParamsBag, {"subject": "user.residence", "contents": "Lisbon"}))
    row = MemoryGraphRow.by_subject("user.residence")
    assert row is not None and row.contents == "Lisbon"

    # Re-saving the same subject overwrites (one row, new contents).
    SaveGraph().run(_bag(SaveGraphParamsBag, {"subject": "user.residence", "contents": "Porto"}))
    rows = MemoryGraphRow.iterate().get()
    assert len(rows) == 1
    assert rows[0].contents == "Porto"


def test_save_map_computes_iteration_from_derived_parents(fixed_embedder: _FixedEmbedder, db: sqlite3.Connection) -> None:
    SaveMap().run(_bag(SaveMapParamsBag, {"contents": "first episode"}))
    first = MemoryMapRow.recent(limit=1)[0]
    assert first.iteration == 1
    assert first.id is not None

    SaveMap().run(
        _bag(SaveMapParamsBag, {"contents": "follow-up", "derived_from": [first.id]})
    )
    derived = MemoryMapRow.recent(limit=1)[0]
    assert derived.iteration == 2
    assert json.loads(derived.derived_from) == [first.id]


def test_delete_graph_removes_a_fact(fixed_embedder: _FixedEmbedder, db: sqlite3.Connection) -> None:
    SaveGraph().run(_bag(SaveGraphParamsBag, {"subject": "pet", "contents": "cat Tom"}))
    assert MemoryGraphRow.by_subject("pet") is not None

    DeleteGraph().run(_bag(DeleteGraphParamsBag, {"subject": "pet"}))
    assert MemoryGraphRow.by_subject("pet") is None


def test_recall_tool_returns_readable_summary(fixed_embedder: _FixedEmbedder, db: sqlite3.Connection) -> None:
    SaveGraph().run(
        _bag(SaveGraphParamsBag, {"subject": "user.residence", "contents": "Lisbon"})
    )
    g = MemoryGraphRow.by_subject("user.residence")
    assert g is not None
    assert isinstance(g.id, int)
    _index("memory_graph", g.id)

    res = Recall().run(_bag(RecallParamsBag, {"query": "residence"}))
    assert res.status == "success"
    assert isinstance(res.body, str)
    assert "Lisbon" in res.body
