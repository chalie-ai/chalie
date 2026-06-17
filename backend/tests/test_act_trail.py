# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0



import pytest

from services.act_trail import ActTrail

pytestmark = pytest.mark.unit


def _seed_transcript(db) -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("user", "user", "what's the weather in Malta?"),
    )
    db.commit()
    return cur.lastrowid


def test_round_trips_a_real_tool_calls_row(db):
    transcript_id = _seed_transcript(db)
    trail = ActTrail()

    trail.record(
        tool_name="weather",
        params={"location": "Malta"},
        result="sunny, 27°C",
        transcript_id=transcript_id,
    )
    trail.record(
        tool_name="tool_chain_compactor",
        params={},
        result="summarised the trail",
        transcript_id=transcript_id,
    )

    rows = trail.fetch_by_transcript_id(transcript_id)

    # Oldest→newest by autoincrement id, raw params/result preserved.
    assert [r["tool_name"] for r in rows] == ["weather", "tool_chain_compactor"]
    assert rows[0]["result"] == "sunny, 27°C"

    # render() emits the invariant "[tool_name] params → result" shape.
    assert ActTrail.render(rows[0]) == '[weather] {"location": "Malta"} → sunny, 27°C'
    assert ActTrail.render(rows[1]) == "[tool_chain_compactor] {} → summarised the trail"


def test_record_is_a_silent_noop_without_a_transcript_id(db):
    """Delegates with skip_transcript pass transcript_id=None; the NOT NULL FK
    means record must skip rather than raise (spec §4c / F2)."""
    before = db.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]

    ActTrail().record(
        tool_name="web_search",
        params={"query": "anything"},
        result="…",
        transcript_id=None,
    )

    after = db.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
    assert after == before


def test_fetch_returns_empty_for_an_unknown_transcript(db):
    assert ActTrail().fetch_by_transcript_id(999999) == []
