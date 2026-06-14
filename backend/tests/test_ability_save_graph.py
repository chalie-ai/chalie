# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""save_graph-specific business-logic tests migrated from the per-ability
conformance file removed in TKT-975. Covers whitespace-only key rejection, happy
path row storage, geo-pass provenance stamping, same-fact deduplication, and the
per-turn budget cap loud path.
"""

import json

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.save_graph import SaveGraph
from configs.channels.geo_pattern import GeoConfig
from configs.channels.pattern import PatternConfig, _pattern_init_instance_state
from services.message_processor import MessageProcessor
from tests._tool_result_harness import body, head, seed_transcript

pytestmark = pytest.mark.unit


# ── Fixtures / helpers ──────────────────────────────────────────────────────────


def _seed_transcript(db) -> int:
    return seed_transcript(db, channel="pattern_match", content="remember a fact")


def _mp(db) -> MessageProcessor:
    """A real MessageProcessor on the pattern config, bound to a real transcript
    anchor so the dispatcher's act-trail write lands on a real row. The
    per-instance budget/dedupe state is initialised exactly as production does
    via ``_pattern_init_instance_state``."""
    mp = MessageProcessor("remember a fact")
    mp.config = PatternConfig(0, 1)
    mp.active_tools = list(mp.config.always_available or [])
    mp.uid = _seed_transcript(db)
    _pattern_init_instance_state(mp)
    return mp


def _head(rendered: str) -> str:
    return head(rendered, "save_graph")


def _body(rendered: str) -> str:
    return body(rendered, "save_graph")


def _rows(db, *, kind=None, key=None, source="pattern_match") -> list:
    sql = "SELECT id, kind, key, value, source FROM data_graph WHERE source=?"
    params: list = [source]
    if kind is not None:
        sql += " AND kind=?"
        params.append(kind)
    if key is not None:
        sql += " AND key=?"
        params.append(key)
    return db.execute(sql, params).fetchall()


def _geo_mp(db) -> MessageProcessor:
    """A real MessageProcessor on the GEO config — the other background pass that
    reaches save_graph. GeoConfig.get_user_prompt lazily inits the same
    per-instance budget/dedupe state PatternConfig does, so we drive it once to
    initialise exactly as production does (no hand-set attrs)."""
    mp = MessageProcessor("detect a geo fact")
    mp.config = GeoConfig(0, 1)
    mp.active_tools = list(mp.config.always_available or [])
    mp.uid = seed_transcript(db, channel="geo_pattern", content="detect a geo fact")
    mp.config.get_user_prompt(mp)  # production lazy-init of save_graph state
    return mp


# ── whitespace-only key slips the truthiness pre-gate → run() rejects it ────────


def test_whitespace_only_key_is_missing_params(db):
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch(
        "save_graph",
        {"kind": "user_specific", "key": "   ", "value": "Lisbon", "act_summary": "x"},
    )
    assert "[save_graph(status=error, code=missing-params" in out
    assert "code=error]" not in out
    assert "key" in out
    assert _rows(db) == []


# ── happy path → success, body {"saved":1,...}, real row with source ────────────


def test_happy_path_stores_one_row(db):
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch(
        "save_graph",
        {"kind": "misc", "key": "fav_drink", "value": "espresso", "act_summary": "x"},
    )
    h = _head(out)
    assert "status=success" in h
    b = json.loads(_body(out))
    assert b == {"saved": 1, "kind": "misc", "key": "fav_drink"}
    rows = _rows(db, kind="misc", key="fav_drink")
    assert len(rows) == 1
    assert rows[0][4] == "pattern_match"


# ── geo pass → same row, provenance stamped 'geo_pattern' not 'pattern_match' ───


def test_geo_pass_stamps_geo_pattern_provenance(db):
    """When save_graph runs under the GEO pass (GeoConfig), the stored fact's
    provenance must be 'geo_pattern', not the pattern pass's 'pattern_match'.

    This is the per-source provenance fix: the two background passes build
    separate MessageProcessors, and the row each writes must record WHICH pass
    produced it so a downstream reader can tell a geo-derived fact from a
    behavioural one. The pattern pass (above) stays 'pattern_match'."""
    mp = _geo_mp(db)
    out = ToolDispatcher(mp).dispatch(
        "save_graph",
        {"kind": "place", "key": "gym", "value": "Fortress Fitness", "act_summary": "x"},
    )
    assert "status=success" in _head(out)

    geo_rows = _rows(db, kind="place", key="gym", source="geo_pattern")
    assert len(geo_rows) == 1, (
        "the geo pass must stamp source='geo_pattern' on the fact it writes"
    )
    # And it must NOT have been recorded under the pattern pass's provenance.
    assert _rows(db, kind="place", key="gym", source="pattern_match") == []


# ── same fact twice in one mp → second is deduped, exactly one row ──────────────


def test_same_fact_twice_dedupes(db):
    mp = _mp(db)
    params = {"kind": "misc", "key": "fav_drink", "value": "espresso", "act_summary": "x"}
    first = ToolDispatcher(mp).dispatch("save_graph", dict(params))
    assert json.loads(_body(first)) == {"saved": 1, "kind": "misc", "key": "fav_drink"}

    second = ToolDispatcher(mp).dispatch("save_graph", dict(params))
    h = _head(second)
    assert "status=success" in h
    b = json.loads(_body(second))
    assert b == {"saved": 0, "deduped": 1, "key": "fav_drink"}

    # Exactly one row landed despite two dispatches.
    assert len(_rows(db, kind="misc", key="fav_drink")) == 1


# ── budget cap → capped=true success, body {"saved":0,"skipped":1}, no row ──────


def test_budget_cap_is_loud_capped(db):
    mp = _mp(db)
    # Drive the real counter attribute straight to the cap — no mock.
    setattr(mp, SaveGraph.BUDGET_COUNTER_ATTR, SaveGraph.BUDGET_CAP)
    out = ToolDispatcher(mp).dispatch(
        "save_graph",
        {"kind": "misc", "key": "over_cap", "value": "nope", "act_summary": "x"},
    )
    h = _head(out)
    assert "status=success" in h
    assert "capped=true" in h
    b = json.loads(_body(out))
    assert b["saved"] == 0
    assert b["skipped"] == 1
    assert "note" in b
    # Nothing was stored.
    assert _rows(db, kind="misc", key="over_cap") == []
