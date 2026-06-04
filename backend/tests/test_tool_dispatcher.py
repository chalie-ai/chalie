# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for ToolDispatcher (spec §4.2 / §5) — the single tool-call
chokepoint that replaced ``Ability.use``.

Real hot path, zero mocks: a real ``mp``-shaped context dispatches a genuinely
registered ability through the live ``AbilityRegistry`` resolution, the real
``PolicyManager.wrap`` gate, the real ``Ability.run``, and the real ``ActTrail``
write. The ``db`` fixture binds ``ActTrail()`` to a real SQLite database so the
recorded outcome is read back exactly as the ACT loop reconstructs the trail.

``find_tools`` is the probe: it is a registered, INTERNAL (always-allowed) tool
whose empty-query branch returns a deterministic tag string without touching the
network or the embedding model — so the test exercises the dispatch ORCHESTRATION
(match → bind → gate → execute → run → record → return str) end-to-end without a
real LLM. The unknown-tool case proves dispatch records EVERY outcome, not only
successes (so the model never retries a non-existent tool forever).
"""

import threading

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import DmnConfig
from services.act_trail import ActTrail

pytestmark = pytest.mark.unit


def _seed_transcript(db) -> int:
    """Insert a real transcript anchor row (tool_calls.transcript_id FK) and
    return its id — the trail has no anchor to hang rows off without it."""
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("dmn", "user", "find me a tool"),
    )
    db.commit()
    return cur.lastrowid


class _MP:
    """Minimal real MP-shaped context — exactly what dispatch reads off the live
    processor: ``config`` (policy channel + emitter gate) and ``uid`` (the
    transcript anchor the trail records against). ``DISCOVERABLE`` / ``active_tools``
    are what an ACT-loop processor exposes to find_tools; supplied for realism."""

    def __init__(self, uid: int):
        self.config = DmnConfig()        # broadcast_to=None → emitter is a real no-op
        self.uid = uid
        self.DISCOVERABLE: list[str] = []
        self.active_tools: list[str] = []
        self.cancel_event = threading.Event()


def test_dispatch_runs_real_registered_tool_through_gate_and_records(db):
    """A registered tool resolves from the real registry, passes the real
    PolicyManager gate, runs, and its outcome is written to the real trail."""
    transcript_id = _seed_transcript(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch("find_tools", {"query": ""})

    # Real find_tools output: the deterministic empty-query tag.
    assert isinstance(result, str)
    assert "find_tools" in result
    assert "query-required" in result

    # The dispatch recorded exactly one outcome against the transcript anchor,
    # rendered in the invariant trail shape.
    rows = ActTrail().fetch_by_transcript_id(transcript_id)
    assert [r["tool_name"] for r in rows] == ["find_tools"]
    assert ActTrail.render(rows[0]).startswith("[find_tools]")
    assert "query-required" in rows[0]["result"]


def test_dispatch_records_unknown_tool_outcome(db):
    """An unknown tool returns a graceful 'Unknown tool' string AND records it —
    dispatch records every outcome so the model never retries forever (§5)."""
    transcript_id = _seed_transcript(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch("no_such_ability_xyz", {"action": "noop"})

    assert result == "Unknown tool: no_such_ability_xyz"

    rows = ActTrail().fetch_by_transcript_id(transcript_id)
    assert [r["tool_name"] for r in rows] == ["no_such_ability_xyz"]
    assert rows[0]["result"] == "Unknown tool: no_such_ability_xyz"
