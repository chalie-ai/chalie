# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for save_pattern's ToolResult contract (TKT-913).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint against a real ``MessageProcessor``
configured with the real ``PatternConfig`` (the config that carries save_pattern
on its ``always_available`` scope), the real ``AbilityRegistry`` resolution, the
real production ``run()`` writing against the real ``db`` fixture's data graph,
and the real ``ActTrail`` write.

save_pattern is in ``PolicyManager.INTERNAL`` — it bypasses the policy gate, so
dispatch needs no policy flip.

THE regressions this file pins (TKT-913 "bad looks like"):
  * Four failures used to masquerade as SUCCESS — ``ok({"error": "invalid_name"})``,
    ``ok({"error": "invalid_frequency"})``, ``ok({"error": "empty_summary"})`` and
    ``ok({"error": "insufficient_evidence"})``. Now an invalid name/frequency/
    evidence is a loud ``code=invalid-param`` and a blank summary is
    ``code=missing-params``.
  * The happy path returned ``ok({"ok": True, ...})`` with no count echo. Now it
    returns ``ok({"saved": 1, ...})`` (and ``"reinforced": 1`` on a reinforce),
    in error-vocabulary parity with save_graph.
  * The per-turn budget cap returned a silent ``ok({"budget_exceeded": True})``.
    Now it carries ``meta capped=true`` and a body that says nothing was stored.
"""

import json

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities._registry import AbilityRegistry
from abilities.save_pattern import SavePattern, _VALID_FREQUENCIES
from configs.channels.pattern import PatternConfig, _pattern_init_instance_state
from services.act_trail import ActTrail
from services.message_processor import MessageProcessor

pytestmark = pytest.mark.unit


# ── Fixtures / helpers ──────────────────────────────────────────────────────────


def _seed_transcript(db) -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("pattern_match", "user", "user goes for a run every weekday morning"),
    )
    db.commit()
    return cur.lastrowid


def _mp(db) -> MessageProcessor:
    """A real MessageProcessor on the pattern config, bound to a real transcript
    anchor so the dispatcher's act-trail write lands on a real row. The
    per-instance budget/decay state is initialised exactly as production does
    via ``_pattern_init_instance_state``."""
    mp = MessageProcessor("look for patterns")
    mp.config = PatternConfig(0, 1)
    mp.active_tools = list(mp.config.always_available or [])
    mp.uid = _seed_transcript(db)
    _pattern_init_instance_state(mp)
    return mp


def _head(rendered: str) -> str:
    line = rendered.splitlines()[0]
    assert line.startswith("[save_pattern(")
    return line


def _body(rendered: str) -> str:
    head = rendered.index("]\n") + 2
    tail = rendered.index("\n[end:save_pattern]")
    return rendered[head:tail]


def _rows(db, *, name=None) -> list:
    sql = (
        "SELECT id, kind, key, value, source FROM data_graph "
        "WHERE kind='behavioral_pattern' AND source='pattern_match'"
    )
    params: list = []
    if name is not None:
        sql += " AND key=?"
        params.append(name)
    return db.execute(sql, params).fetchall()


def _valid_params(**overrides) -> dict:
    params = {
        "name": "morning_run",
        "frequency": "weekday",
        "time_anchor": "07:00",
        "summary": "user goes for a run every weekday morning",
        "evidence_transcript_ids": [12, 18],
        "act_summary": "x",
    }
    params.update(overrides)
    return params


# ── missing params (absent) → dispatcher pre-gate missing-params, no write ──────


def test_missing_name_is_missing_params(db):
    mp = _mp(db)
    params = _valid_params()
    del params["name"]
    out = ToolDispatcher(mp).dispatch("save_pattern", params)
    assert "[save_pattern(status=error, code=missing-params" in out
    assert "code=error]" not in out
    assert "name" in out
    assert "[end:save_pattern]" in out
    assert _rows(db) == []


def test_absent_evidence_is_missing_params(db):
    mp = _mp(db)
    params = _valid_params()
    del params["evidence_transcript_ids"]
    out = ToolDispatcher(mp).dispatch("save_pattern", params)
    assert "[save_pattern(status=error, code=missing-params" in out
    assert "code=error]" not in out
    assert _rows(db) == []


def test_empty_list_evidence_is_missing_params(db):
    """An empty ``evidence_transcript_ids`` list is falsy — the truthiness
    pre-gate rejects it before run() as missing-params."""
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch(
        "save_pattern", _valid_params(evidence_transcript_ids=[])
    )
    assert "[save_pattern(status=error, code=missing-params" in out
    assert _rows(db) == []


# ── whitespace-only summary slips the truthiness pre-gate → run() rejects it ────


def test_whitespace_only_summary_is_missing_params(db):
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("save_pattern", _valid_params(summary="   "))
    assert "[save_pattern(status=error, code=missing-params" in out
    assert "code=error]" not in out
    assert "summary" in out
    assert _rows(db) == []


# ── invalid name → loud invalid-param with example hint ─────────────────────────


def test_invalid_name_is_invalid_param(db):
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("save_pattern", _valid_params(name="Morning Run"))
    head = _head(out)
    assert "status=error" in head
    assert "code=invalid-param" in head
    assert "code=error]" not in out
    hint_line = next(ln for ln in out.splitlines() if ln.startswith("hint:"))
    # The hint must carry a minimal, fully-valid example.
    assert "name=" in hint_line
    assert "evidence_transcript_ids" in hint_line
    assert _rows(db, name="Morning Run") == []


# ── invalid frequency → invalid-param with the full 5-frequency valid ladder ────


def test_invalid_frequency_is_invalid_param(db):
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("save_pattern", _valid_params(frequency="hourly"))
    head = _head(out)
    assert "status=error" in head
    assert "code=invalid-param" in head
    assert "code=error]" not in out
    valid_line = next(ln for ln in out.splitlines() if ln.startswith("valid:"))
    for freq in _VALID_FREQUENCIES:
        assert freq in valid_line
    assert _rows(db) == []


# ── insufficient evidence → invalid-param mentioning the 2-id minimum ───────────


def test_insufficient_evidence_is_invalid_param(db):
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch(
        "save_pattern", _valid_params(evidence_transcript_ids=[7])
    )
    head = _head(out)
    assert "status=error" in head
    assert "code=invalid-param" in head
    assert "code=error]" not in out
    assert "2" in out
    assert _rows(db) == []


# ── happy path → success, body {"saved":1,...}, real row with source ────────────


def test_happy_path_stores_one_row(db):
    mp = _mp(db)
    out = ToolDispatcher(mp).dispatch("save_pattern", _valid_params())
    head = _head(out)
    assert "status=success" in head
    body = json.loads(_body(out))
    assert body["saved"] == 1
    assert body["name"] == "morning_run"
    assert body["confidence"] == 7.0
    assert isinstance(body["row_id"], int)
    assert body == {
        "saved": 1,
        "name": "morning_run",
        "confidence": 7.0,
        "row_id": body["row_id"],
    }
    rows = _rows(db, name="morning_run")
    assert len(rows) == 1
    assert rows[0][4] == "pattern_match"
    stored = json.loads(rows[0][3])
    assert stored["confidence"] == 7.0


# ── reinforce same name → success body carries reinforced:1, exactly one row ─────


def test_reinforce_same_name(db):
    mp = _mp(db)
    first = ToolDispatcher(mp).dispatch("save_pattern", _valid_params())
    assert json.loads(_body(first))["saved"] == 1

    second = ToolDispatcher(mp).dispatch(
        "save_pattern", _valid_params(evidence_transcript_ids=[20, 22])
    )
    head = _head(second)
    assert "status=success" in head
    body = json.loads(_body(second))
    assert body["saved"] == 1
    assert body["reinforced"] == 1
    assert body["confidence"] == 10.0  # 7 + 7 capped at 10

    # Exactly one row landed despite two dispatches.
    assert len(_rows(db, name="morning_run")) == 1


# ── budget cap → capped=true success, body {"saved":0,"skipped":1}, no row ──────


def test_budget_cap_is_loud_capped(db):
    mp = _mp(db)
    # Drive the real counter attribute straight to the cap — no mock.
    setattr(mp, SavePattern.BUDGET_COUNTER_ATTR, SavePattern.BUDGET_CAP)
    out = ToolDispatcher(mp).dispatch("save_pattern", _valid_params(name="over_cap"))
    head = _head(out)
    assert "status=success" in head
    assert "capped=true" in head
    body = json.loads(_body(out))
    assert body["saved"] == 0
    assert body["skipped"] == 1
    assert "note" in body
    # Nothing was stored.
    assert _rows(db, name="over_cap") == []


# ── no legacy markers in any envelope ───────────────────────────────────────────


def test_no_legacy_markers_anywhere(db):
    mp = _mp(db)
    setattr(mp, SavePattern.BUDGET_COUNTER_ATTR, 0)
    envelopes = [
        ToolDispatcher(mp).dispatch("save_pattern", _valid_params(name="Morning Run")),
        ToolDispatcher(mp).dispatch("save_pattern", _valid_params(frequency="hourly")),
        ToolDispatcher(mp).dispatch("save_pattern", _valid_params(summary="   ")),
        ToolDispatcher(mp).dispatch(
            "save_pattern", _valid_params(evidence_transcript_ids=[7])
        ),
        ToolDispatcher(mp).dispatch("save_pattern", _valid_params(name="ok_pattern")),
    ]
    # Now flip the counter to the cap and add the capped envelope.
    setattr(mp, SavePattern.BUDGET_COUNTER_ATTR, SavePattern.BUDGET_CAP)
    envelopes.append(
        ToolDispatcher(mp).dispatch("save_pattern", _valid_params(name="capped_one"))
    )
    for out in envelopes:
        assert '"error": "invalid_name"' not in out
        assert '"error":"invalid_name"' not in out
        assert "invalid_frequency" not in out
        assert "empty_summary" not in out
        assert "insufficient_evidence" not in out
        assert "budget_exceeded" not in out
        assert '"ok": true' not in out
        assert '"ok":true' not in out
        assert "code=error]" not in out
        assert "code=error," not in out
        assert "code=error)" not in out


# ── an error dispatch still records an act-trail row ─────────────────────────────


def test_error_dispatch_writes_act_trail(db):
    mp = _mp(db)
    ToolDispatcher(mp).dispatch("save_pattern", _valid_params(name="Morning Run"))
    trail = ActTrail().fetch_by_transcript_id(mp.uid)
    assert any(
        "[save_pattern(status=error, code=invalid-param" in r["result"] for r in trail
    )


# ── registry still resolves save_pattern with its metadata ──────────────────────


def test_save_pattern_registered_with_metadata():
    ability = AbilityRegistry.get("save_pattern")
    assert ability.get_name() == "save_pattern"
    assert ability.get_summary()
    assert len(ability.get_examples()) >= 6
