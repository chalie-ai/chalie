# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the four Memory v3 memory tools (Slice B).

Exercises the four abilities end-to-end on the real ``db`` fixture: each tool's
param bag (validation) + run() (store mutation), driving the real
SearchExpanderService indexing where a vector lane is involved. The embedding
model is stubbed deterministically (the shared ``fixed_embedder`` fixture).
"""

import json
import sqlite3
from typing import TYPE_CHECKING, TypeVar, cast

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
from services.dispatch_service import DispatchService
from services.search_expander_service import SearchExpanderService

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("fixed_embedder")]


def _index(table: str, rowid: int) -> None:
    cfg = config_for_table(table)
    assert cfg is not None
    SearchExpanderService()._process_row(table, rowid, cfg)


_T = TypeVar("_T", bound=ParamBag)


def _bag(bag_cls: type[_T], params: dict[str, object]) -> _T:
    bag = bag_cls.from_params(params)
    assert not isinstance(bag, ToolResult)
    return cast("_T", bag)


def test_save_graph_writes_subject_keyed_upsert(db: sqlite3.Connection) -> None:
    SaveGraph().run(_bag(SaveGraphParamsBag, {"subject": "user.residence", "contents": "Lisbon"}))
    row = MemoryGraphRow.by_subject("user.residence")
    assert row is not None and row.contents == "Lisbon"

    # Re-saving the same subject overwrites (one row, new contents).
    SaveGraph().run(_bag(SaveGraphParamsBag, {"subject": "user.residence", "contents": "Porto"}))
    rows = MemoryGraphRow.all().get()
    assert len(rows) == 1
    assert rows[0].contents == "Porto"


def test_save_map_computes_iteration_from_derived_parents(db: sqlite3.Connection) -> None:
    SaveMap().run(
        _bag(
            SaveMapParamsBag,
            {
                "contents": "first episode",
                "source": "Grocery trip in May 2026",
                "cues": ["shopping"],
            },
        )
    )
    first = MemoryMapRow.all().order_by("id DESC").limit(1).get()[0]
    assert first.iteration == 1
    assert first.id is not None

    SaveMap().run(
        _bag(
            SaveMapParamsBag,
            {
                "contents": "follow-up",
                "source": "Shopping trips in 2026",
                "cues": ["shopping", "dead phone battery"],
                "derived_from": [first.id],
            },
        )
    )
    derived = MemoryMapRow.all().order_by("id DESC").limit(1).get()[0]
    assert derived.iteration == 2
    assert json.loads(derived.derived_from) == [first.id]


def test_save_map_stores_source_verbatim_and_joins_cues(db: sqlite3.Connection) -> None:
    source = "Grocery trip in Zabbar in May 2026"
    SaveMap().run(
        _bag(
            SaveMapParamsBag,
            {
                "contents": "forgot half the list",
                "source": source,
                "cues": ["shopping", "incomplete shopping list"],
            },
        )
    )
    row = MemoryMapRow.all().order_by("id DESC").limit(1).get()[0]
    assert row.source == source
    assert row.cues == "shopping, incomplete shopping list"


def test_save_map_rejects_more_than_five_cues(db: sqlite3.Connection) -> None:
    """Overflow is refused, never silently trimmed — only the model can choose
    which five tags a memory is found by."""
    rejected = SaveMapParamsBag.from_params(
        {
            "contents": "an episode",
            "source": "somewhere",
            "cues": ["one", "two", "three", "four", "five", "six"],
        }
    )
    assert isinstance(rejected, ToolResult)
    assert rejected.code == "invalid-param"
    assert MemoryMapRow.all().get() == []


@pytest.mark.parametrize("missing", ["source", "cues"])
def test_save_map_refuses_an_episode_missing_source_or_cues(
    db: sqlite3.Connection, missing: str
) -> None:
    """A row without cues can never be recalled, so the write is rejected at the
    contract rather than landing an unreachable episode."""
    params: dict[str, object] = {
        "contents": "an episode",
        "source": "somewhere",
        "cues": ["shopping"],
    }
    del params[missing]

    rejected = SaveMapParamsBag.from_params(params)
    assert isinstance(rejected, ToolResult)
    assert rejected.code == "missing-params"
    assert MemoryMapRow.all().get() == []


def test_delete_graph_removes_a_fact(db: sqlite3.Connection) -> None:
    SaveGraph().run(_bag(SaveGraphParamsBag, {"subject": "pet", "contents": "cat Tom"}))
    assert MemoryGraphRow.by_subject("pet") is not None

    DeleteGraph().run(_bag(DeleteGraphParamsBag, {"subject": "pet"}))
    assert MemoryGraphRow.by_subject("pet") is None


def test_recall_renders_source_and_memory_for_both_halves(db: sqlite3.Connection) -> None:
    SaveGraph().run(
        _bag(SaveGraphParamsBag, {"subject": "user.residence", "contents": "Lisbon"})
    )
    g = MemoryGraphRow.by_subject("user.residence")
    assert g is not None
    assert isinstance(g.id, int)
    _index("memory_graph", g.id)

    SaveMap().run(
        _bag(
            SaveMapParamsBag,
            {
                "contents": "forgot half the list",
                "source": "Grocery trip in Zabbar in May 2026",
                "cues": ["residence", "incomplete shopping list"],
            },
        )
    )
    m = MemoryMapRow.all().order_by("id DESC").limit(1).get()[0]
    assert isinstance(m.id, int)
    _index("memory_map", m.id)

    res = Recall().run(_bag(RecallParamsBag, {"query": "residence"}))
    assert res.status == "success"
    assert res.body == [
        {"source": "user.residence", "memory": "Lisbon"},
        {"source": "Grocery trip in Zabbar in May 2026", "memory": "forgot half the list"},
    ]


def test_recall_never_shows_the_model_its_own_cues(db: sqlite3.Connection) -> None:
    """Cues are the search key and nothing else — an author that reads back its
    previous tags starts copying them instead of describing the situation."""
    SaveMap().run(
        _bag(
            SaveMapParamsBag,
            {
                "contents": "forgot half the list",
                "source": "Grocery trip in Zabbar in May 2026",
                "cues": ["marmalade", "aubergine"],
            },
        )
    )
    m = MemoryMapRow.all().order_by("id DESC").limit(1).get()[0]
    assert isinstance(m.id, int)
    _index("memory_map", m.id)

    res = Recall().run(_bag(RecallParamsBag, {"query": "marmalade"}))
    rendered = DispatchService(mp=cast("MessageProcessor", None))._render("recall", res)

    assert "forgot half the list" in rendered
    assert "marmalade" not in rendered
    assert "aubergine" not in rendered
    assert "cues" not in rendered


def test_recall_rejects_an_oversized_query_as_invalid_param(
    db: sqlite3.Connection
) -> None:
    """Loud rejection, never a silent trim: the service raises and the ability
    surfaces it as the save_map-shaped invalid-param error, verbatim."""
    query = " ".join(["zebra"] * 100)  # 599 scrubbed chars
    res = Recall().run(_bag(RecallParamsBag, {"query": query}))
    assert res.status == "error"
    assert res.code == "invalid-param"
    assert res.body == (
        "Recall query is too long, maximum of 500 characters allowed. "
        "Use more fine-grained queries to load relevent memories"
    )


def test_recall_stopword_padding_does_not_count_toward_the_cap(
    db: sqlite3.Connection
) -> None:
    """The gate measures the stop-word-scrubbed query: a raw query over 500
    chars, padded with stop-words, scrubs well under the cap and proceeds to
    a normal recall — not rejected, not trimmed."""
    core = " ".join(["apple"] * 40)  # 239 scrubbed chars
    query = core + " " + " ".join(["the", "of", "to", "and"] * 20)  # +279 stop-word chars
    assert len(query) > 500
    res = Recall().run(_bag(RecallParamsBag, {"query": query}))
    assert res.status == "success"
    assert res.body == "No existing memory found."
