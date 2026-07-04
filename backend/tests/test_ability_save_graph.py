# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""save_graph-specific business-logic tests migrated from the per-ability"""

import json
import sqlite3

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.save_graph import SaveGraph
from configs.channels.geo_pattern import GeoConfig
from configs.channels.pattern import PatternConfig
from services.act_trail import ActTrail
from services.message_processor import MessageProcessor
from tests._tool_result_harness import body, head

pytestmark = pytest.mark.unit


# ── Fixtures / helpers ──────────────────────────────────────────────────────────


def _mp(db: sqlite3.Connection) -> MessageProcessor:
    # Real MessageProcessor construction: config first, then turn_id/raw_input.
    # PatternConfig.skip_input_row is False, so the constructor itself seeds this
    # turn's anchoring input row (mp.uid/mp.turn_id) exactly as production's
    # _setup does — no manual row seeding or private-field poking needed.
    mp = MessageProcessor(PatternConfig(0, 1), raw_input="remember a fact")
    mp.active_tools = list(mp.config.always_available or [])
    return mp


def _head(rendered: str) -> str:
    return head(rendered, "save_graph")


def _body(rendered: str) -> str:
    return body(rendered, "save_graph")


def _rows(db: sqlite3.Connection, *, kind: str | None = None, key: str | None = None, source: str = "pattern_match") -> list[sqlite3.Row]:
    sql = "SELECT id, kind, key, value, source FROM data_graph WHERE source=?"
    params: list[object] = [source]
    if kind is not None:
        sql += " AND kind=?"
        params.append(kind)
    if key is not None:
        sql += " AND key=?"
        params.append(key)
    return db.execute(sql, params).fetchall()


def _geo_mp(db: sqlite3.Connection) -> MessageProcessor:
    """A real MessageProcessor on the GEO config — the other background pass that"""
    mp = MessageProcessor(GeoConfig(0, 1), raw_input="detect a geo fact")
    mp.active_tools = list(mp.config.always_available or [])
    return mp


# ── whitespace-only key slips the truthiness pre-gate → run() rejects it ────────


def test_whitespace_only_key_is_missing_params(db: sqlite3.Connection) -> None:
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


def test_happy_path_stores_one_row(db: sqlite3.Connection) -> None:
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


def test_geo_pass_stamps_geo_pattern_provenance(db: sqlite3.Connection) -> None:
    """When save_graph runs under the GEO pass (GeoConfig), the stored fact's"""
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


def test_same_fact_twice_dedupes(db: sqlite3.Connection) -> None:
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


def test_budget_cap_is_loud_capped(db: sqlite3.Connection) -> None:
    mp = _mp(db)
    # Fill this turn's durable trail to the cap with real save_graph rows via the
    # production writer (the exact call the dispatcher makes), so the DB-derived
    # budget counts BUDGET_CAP prior calls for the turn — no in-memory counter,
    # no mock.
    trail = ActTrail()
    for i in range(SaveGraph.BUDGET_CAP):
        trail.record(
            tool_name="save_graph",
            params={"kind": "misc", "key": f"prior_{i}", "value": "v"},
            result="[save_graph(status=success)]\n{\"saved\":1}\n[end:save_graph]",
            transcript_id=mp.uid,
        )
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
