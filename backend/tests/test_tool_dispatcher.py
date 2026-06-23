# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for ToolDispatcher - the single tool-call chokepoint.

``find_tools`` (INTERNAL, always-allowed) is the probe: its empty-query branch
returns a deterministic string without touching the network or embedding model,
so the test exercises dispatch orchestration end-to-end without a real LLM.
The unknown-tool case proves dispatch records EVERY outcome so the model never
retries a non-existent tool forever.
"""

import sqlite3
import threading
from typing import cast

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import DmnConfig
from services.act_trail import ActTrail

pytestmark = pytest.mark.unit


def _seed_transcript(db: sqlite3.Connection) -> int:
    """Insert a transcript anchor row; tool_calls.transcript_id FK requires it."""
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("dmn", "user", "find me a tool"),
    )
    db.commit()
    return cast(int, cur.lastrowid)


class _MP:
    """Minimal MP-shaped context: config (DmnConfig, no-op emitter), uid (trail
    anchor), DISCOVERABLE/active_tools (exposed to find_tools for realism)."""

    def __init__(self, uid: int) -> None:
        self.config = DmnConfig()        # broadcast_to=None → emitter is a real no-op
        self.uid = uid
        self.DISCOVERABLE: list[str] = []
        self.active_tools: list[str] = []
        self.cancel_event = threading.Event()


def test_dispatch_runs_real_registered_tool_through_gate_and_records(db: sqlite3.Connection) -> None:
    transcript_id = _seed_transcript(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch("find_tools", {"query": ""})

    # Real find_tools output: the deterministic no-params error, rendered in the
    # canonical envelope by the dispatcher with a stable kebab-case code.
    assert isinstance(result, str)
    assert result.startswith("[find_tools(status=error, code=missing-params)]")
    assert "code=error]" not in result
    assert result.endswith("[end:find_tools]")

    # The dispatch recorded exactly one outcome against the transcript anchor,
    # rendered in the invariant trail shape.
    rows = ActTrail().fetch_by_transcript_id(transcript_id)
    assert [r["tool_name"] for r in rows] == ["find_tools"]
    assert ActTrail.render(rows[0]).startswith("[find_tools]")
    assert "missing-params" in cast(str, rows[0]["result"])


def test_dispatch_records_unknown_tool_outcome(db: sqlite3.Connection) -> None:
    """Unknown tool returns a graceful string AND records it so the model never
    retries a non-existent tool forever."""
    transcript_id = _seed_transcript(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch("no_such_ability_xyz", {"action": "noop"})

    assert result == "Unknown tool: no_such_ability_xyz"

    rows = ActTrail().fetch_by_transcript_id(transcript_id)
    assert [r["tool_name"] for r in rows] == ["no_such_ability_xyz"]
    assert rows[0]["result"] == "Unknown tool: no_such_ability_xyz"
