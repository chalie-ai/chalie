# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for save_graph's ToolResult contract (TKT-912).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint against a real ``MessageProcessor``
configured with the real ``PatternConfig`` (the config that carries save_graph
on its ``always_available`` scope), the real ``AbilityRegistry`` resolution, the
real production ``run()`` writing against the real ``db`` fixture's
``DataGraphService``, and the real ``ActTrail`` write.

save_graph is in ``PolicyManager.INTERNAL`` — it bypasses the policy gate, so
dispatch needs no policy flip.

THE regressions this file pins (TKT-912 "bad looks like"):
  * Four failures used to masquerade as SUCCESS — ``ok({"error": "invalid_kind"})``,
    ``ok({"error": "empty_key"})``, ``ok({"error": "empty_value"})`` and
    ``ok({"error": "store_failed"})``. Now invalid kind is a loud
    ``code=invalid-param`` and missing/blank params are ``code=missing-params``.
  * The per-turn budget cap returned a silent ``ok({"budget_exceeded": True})``.
    Now it carries ``meta capped=true`` and a body that says nothing was stored.
"""

import json

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.save_graph import ALLOWED_KINDS, SaveGraph
from configs.channels.geo_pattern import GeoConfig
from configs.channels.pattern import PatternConfig, _pattern_init_instance_state
from services.act_trail import ActTrail
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


# ── missing params (absent) → dispatcher pre-gate missing-params, no write ──────


def test_missing_kind_is_missing_params(db):
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch(
        "save_graph", {"key": "residence", "value": "Lisbon", "act_summary": "x"}
    )
    assert "[save_graph(status=error, code=missing-params" in out
    assert "code=error]" not in out
    assert "kind" in out
    assert "[end:save_graph]" in out
    assert _rows(db) == []


def test_missing_key_is_missing_params(db):
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch(
        "save_graph", {"kind": "user_specific", "value": "Lisbon", "act_summary": "x"}
    )
    assert "[save_graph(status=error, code=missing-params" in out
    assert "key" in out
    assert _rows(db) == []


def test_missing_value_is_missing_params(db):
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch(
        "save_graph", {"kind": "user_specific", "key": "residence", "act_summary": "x"}
    )
    assert "[save_graph(status=error, code=missing-params" in out
    assert "value" in out
    assert _rows(db) == []


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


# ── invalid kind → loud invalid-param with valid ladder + example hint ──────────


def test_invalid_kind_is_invalid_param(db):
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch(
        "save_graph",
        {
            "kind": "behavioral_pattern",
            "key": "x",
            "value": "y",
            "act_summary": "x",
        },
    )
    head = _head(out)
    assert "status=error" in head
    assert "code=invalid-param" in head
    assert "code=error]" not in out
    valid_line = next(ln for ln in out.splitlines() if ln.startswith("valid:"))
    for kind in ALLOWED_KINDS:
        assert kind in valid_line
    assert "behavioral_pattern" not in valid_line
    hint_line = next(ln for ln in out.splitlines() if ln.startswith("hint:"))
    # The hint must carry a minimal, fully-valid example.
    assert "kind=user_specific" in hint_line
    assert "key=residence" in hint_line
    assert "value=Lisbon" in hint_line
    # Nothing written.
    assert _rows(db, kind="behavioral_pattern") == []


# ── happy path → success, body {"saved":1,...}, real row with source ────────────


def test_happy_path_stores_one_row(db):
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch(
        "save_graph",
        {"kind": "misc", "key": "fav_drink", "value": "espresso", "act_summary": "x"},
    )
    head = _head(out)
    assert "status=success" in head
    body = json.loads(_body(out))
    assert body == {"saved": 1, "kind": "misc", "key": "fav_drink"}
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
    head = _head(second)
    assert "status=success" in head
    body = json.loads(_body(second))
    assert body == {"saved": 0, "deduped": 1, "key": "fav_drink"}

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
    head = _head(out)
    assert "status=success" in head
    assert "capped=true" in head
    body = json.loads(_body(out))
    assert body["saved"] == 0
    assert body["skipped"] == 1
    assert "note" in body
    # Nothing was stored.
    assert _rows(db, kind="misc", key="over_cap") == []


# ── no legacy markers in any envelope ───────────────────────────────────────────


def test_no_legacy_markers_anywhere(db):
    mp = _mp(db)
    setattr(mp, SaveGraph.BUDGET_COUNTER_ATTR, 0)
    envelopes = [
        ToolDispatcher(mp).dispatch(
            "save_graph", {"key": "k", "value": "v", "act_summary": "x"}
        ),
        ToolDispatcher(mp).dispatch(
            "save_graph",
            {"kind": "behavioral_pattern", "key": "k", "value": "v", "act_summary": "x"},
        ),
        ToolDispatcher(mp).dispatch(
            "save_graph",
            {"kind": "misc", "key": "ok_key", "value": "ok_val", "act_summary": "x"},
        ),
    ]
    # Now flip the counter to the cap and add the capped envelope.
    setattr(mp, SaveGraph.BUDGET_COUNTER_ATTR, SaveGraph.BUDGET_CAP)
    envelopes.append(
        ToolDispatcher(mp).dispatch(
            "save_graph",
            {"kind": "misc", "key": "capped_key", "value": "v", "act_summary": "x"},
        )
    )
    for out in envelopes:
        assert '"error": "invalid_kind"' not in out
        assert '"error":"invalid_kind"' not in out
        assert "budget_exceeded" not in out
        assert "already_stored" not in out
        assert "store_failed" not in out
        assert "code=error]" not in out
        assert "code=error," not in out
        assert "code=error)" not in out


# ── an error dispatch still records an act-trail row ─────────────────────────────


def test_error_dispatch_writes_act_trail(db):
    mp = _mp(db)
    ToolDispatcher(mp).dispatch(
        "save_graph",
        {"kind": "behavioral_pattern", "key": "x", "value": "y", "act_summary": "x"},
    )
    trail = ActTrail().fetch_by_transcript_id(mp.uid)
    assert any(
        "[save_graph(status=error, code=invalid-param" in r["result"] for r in trail
    )
