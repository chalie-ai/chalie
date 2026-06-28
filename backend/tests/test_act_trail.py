# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0



import logging
import sqlite3

import pytest

from services.act_trail import ActTrail

pytestmark = pytest.mark.unit


def _seed_transcript(db: sqlite3.Connection) -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("user", "user", "what's the weather in Malta?"),
    )
    db.commit()
    from typing import cast
    return cast(int, cur.lastrowid)


def test_round_trips_a_real_tool_calls_row(db: sqlite3.Connection) -> None:
    transcript_id = _seed_transcript(db)
    trail = ActTrail()

    trail.record(
        tool_name="weather",
        params={"location": "Malta"},
        result="sunny, 27°C",
        transcript_id=transcript_id,
    )
    trail.record(
        tool_name="chat_history_compactor",
        params={},
        result="summarised the trail",
        transcript_id=transcript_id,
    )

    rows = trail.fetch_by_transcript_id(transcript_id)

    # Oldest→newest by autoincrement id, raw params/result preserved.
    assert [r["tool_name"] for r in rows] == ["weather", "chat_history_compactor"]
    assert rows[0]["result"] == "sunny, 27°C"

    # render() emits the invariant "[tool_name] params → result" shape.
    assert ActTrail.render(rows[0]) == '[weather] {"location": "Malta"} → sunny, 27°C'
    assert ActTrail.render(rows[1]) == "[chat_history_compactor] {} → summarised the trail"

    # The anchor filter is load-bearing: a sibling transcript_id that owns no
    # tool_calls reads back empty, even though THIS transcript has two rows. A
    # dropped/loosened WHERE clause would leak these two into the unknown read.
    assert trail.fetch_by_transcript_id(transcript_id + 1) == []


def test_record_is_a_silent_noop_without_a_transcript_id(
    db: sqlite3.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    """Delegates that skip the transcript pass transcript_id=None; the NOT NULL
    FK means record must short-circuit on the guard BEFORE the write — not let a
    NULL insert raise and get swallowed by the non-fatal except. A clean skip
    logs nothing at WARNING; a swallowed IntegrityError would."""
    before = db.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]

    with caplog.at_level(logging.WARNING):
        ActTrail().record(
            tool_name="web_search",
            params={"query": "anything"},
            result="…",
            transcript_id=None,
        )

    after = db.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
    assert after == before
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
