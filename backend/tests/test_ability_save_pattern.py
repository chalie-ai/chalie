# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import json
import sqlite3

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.save_pattern import SavePattern, _VALID_FREQUENCIES
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
    mp = MessageProcessor(PatternConfig(0, 1), raw_input="look for patterns")
    mp.active_tools = list(mp.config.always_available or [])
    return mp


def _head(rendered: str) -> str:
    return head(rendered, "save_pattern")


def _body(rendered: str) -> str:
    return body(rendered, "save_pattern")


def _rows(db: sqlite3.Connection, *, name: str | None = None, source: str = "pattern_match") -> list[sqlite3.Row]:
    sql = (
        "SELECT id, kind, key, value, source FROM data_graph "
        "WHERE kind='behavioral_pattern' AND source=?"
    )
    params: list[object] = [source]
    if name is not None:
        sql += " AND key=?"
        params.append(name)
    return db.execute(sql, params).fetchall()


def _geo_mp(db: sqlite3.Connection) -> MessageProcessor:
    """A real MessageProcessor on the GEO config — the other background pass that"""
    mp = MessageProcessor(GeoConfig(0, 1), raw_input="look for geo patterns")
    mp.active_tools = list(mp.config.always_available or [])
    return mp


def _valid_params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "name": "morning_run",
        "frequency": "weekday",
        "time_anchor": "07:00",
        "summary": "user goes for a run every weekday morning",
        "evidence_transcript_ids": [12, 18],
        "act_summary": "x",
    }
    params.update(overrides)
    return params


# ── whitespace-only summary slips the truthiness pre-gate → run() rejects it ────


def test_whitespace_only_summary_is_missing_params(db: sqlite3.Connection) -> None:
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("save_pattern", _valid_params(summary="   "))
    assert "[save_pattern(status=error, code=missing-params" in out
    assert "code=error]" not in out
    assert "summary" in out
    assert _rows(db) == []


# ── invalid name → loud invalid-param with example hint ─────────────────────────


def test_invalid_name_is_invalid_param(db: sqlite3.Connection) -> None:
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("save_pattern", _valid_params(name="Morning Run"))
    h = _head(out)
    assert "status=error" in h
    assert "code=invalid-param" in h
    assert "code=error]" not in out
    hint_line = next(ln for ln in out.splitlines() if ln.startswith("hint:"))
    # The hint must carry a minimal, fully-valid example.
    assert "name=" in hint_line
    assert "evidence_transcript_ids" in hint_line
    assert _rows(db, name="Morning Run") == []


# ── invalid frequency → invalid-param with the full 5-frequency valid ladder ────


def test_invalid_frequency_is_invalid_param(db: sqlite3.Connection) -> None:
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("save_pattern", _valid_params(frequency="hourly"))
    h = _head(out)
    assert "status=error" in h
    assert "code=invalid-param" in h
    assert "code=error]" not in out
    valid_line = next(ln for ln in out.splitlines() if ln.startswith("valid:"))
    for freq in _VALID_FREQUENCIES:
        assert freq in valid_line
    assert _rows(db) == []


# ── insufficient evidence → invalid-param mentioning the 2-id minimum ───────────


def test_insufficient_evidence_is_invalid_param(db: sqlite3.Connection) -> None:
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch(
        "save_pattern", _valid_params(evidence_transcript_ids=[7])
    )
    h = _head(out)
    assert "status=error" in h
    assert "code=invalid-param" in h
    assert "code=error]" not in out
    assert "2" in out
    assert _rows(db) == []


# ── happy path → success, body {"saved":1,...}, real row with source ────────────


def test_happy_path_stores_one_row(db: sqlite3.Connection) -> None:
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("save_pattern", _valid_params())
    h = _head(out)
    assert "status=success" in h
    b = json.loads(_body(out))
    assert b["saved"] == 1
    assert b["name"] == "morning_run"
    assert b["confidence"] == 7.0
    assert isinstance(b["row_id"], int)
    assert b == {
        "saved": 1,
        "name": "morning_run",
        "confidence": 7.0,
        "row_id": b["row_id"],
    }
    rows = _rows(db, name="morning_run")
    assert len(rows) == 1
    assert rows[0][4] == "pattern_match"
    stored = json.loads(rows[0][3])
    assert stored["confidence"] == 7.0


# ── geo pass → behavioral_pattern row stamped 'geo_pattern' provenance ──────────


def test_geo_pass_stamps_geo_pattern_provenance(db: sqlite3.Connection) -> None:
    """When save_pattern runs under the GEO pass (GeoConfig), the behavioral"""
    mp = _geo_mp(db)
    out = ToolDispatcher(mp).dispatch(
        "save_pattern",
        _valid_params(name="harbour_gym", summary="user trains at the harbour gym daily"),
    )
    assert "status=success" in _head(out)

    geo_rows = _rows(db, name="harbour_gym", source="geo_pattern")
    assert len(geo_rows) == 1, (
        "the geo pass must stamp source='geo_pattern' on the pattern it writes"
    )
    assert _rows(db, name="harbour_gym", source="pattern_match") == []


# ── reinforce same name → success body carries reinforced:1, exactly one row ─────


def test_reinforce_same_name(db: sqlite3.Connection) -> None:
    mp = _mp(db)
    first = ToolDispatcher(mp).dispatch("save_pattern", _valid_params())
    assert json.loads(_body(first))["saved"] == 1

    second = ToolDispatcher(mp).dispatch(
        "save_pattern", _valid_params(evidence_transcript_ids=[20, 22])
    )
    h = _head(second)
    assert "status=success" in h
    b = json.loads(_body(second))
    assert b["saved"] == 1
    assert b["reinforced"] == 1
    assert b["confidence"] == 10.0  # 7 + 7 capped at 10

    # Exactly one row landed despite two dispatches.
    assert len(_rows(db, name="morning_run")) == 1


# ── budget cap → capped=true success, body {"saved":0,"skipped":1}, no row ──────


def test_budget_cap_is_loud_capped(db: sqlite3.Connection) -> None:
    mp = _mp(db)
    # Fill this turn's durable trail to the cap with real save_pattern rows via
    # the production writer (the exact call the dispatcher makes), so the
    # DB-derived budget counts BUDGET_CAP prior calls for the turn — no in-memory
    # counter, no mock.
    trail = ActTrail()
    for i in range(SavePattern.BUDGET_CAP):
        trail.record(
            tool_name="save_pattern",
            params={"name": f"prior_{i}"},
            result="[save_pattern(status=success)]\n{\"saved\":1}\n[end:save_pattern]",
            transcript_id=mp.uid,
        )
    out = ToolDispatcher(mp).dispatch("save_pattern", _valid_params(name="over_cap"))
    h = _head(out)
    assert "status=success" in h
    assert "capped=true" in h
    b = json.loads(_body(out))
    assert b["saved"] == 0
    assert b["skipped"] == 1
    assert "note" in b
    # Nothing was stored.
    assert _rows(db, name="over_cap") == []
