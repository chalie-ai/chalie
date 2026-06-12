# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for TKT-947 — durable tool_calls retention.

Three real-world behaviours asserted here:

B. Rich-card pipeline end-to-end: a tool_calls row written by ActTrail survives
   after the turn (no purge), and SegmentService.build() over the same transcript
   IDs returns a rich segment — not an orphan-tag text fallback.  This is the
   direct proof that the purge-before-segment-build bug class is fixed.

C. Janitor: DecayEngineService._purge_tool_calls() deletes rows older than 7 days
   and leaves recent rows intact.

D. review_tool_calls narration filter: a narration row seeded inside the window
   must be absent from the ability's results; a normal tool row must be present.

All tests use the real ``db`` fixture (SchemaConvergenceService-converged schema,
zero hand-rolled DDL), zero mocks, utc_now() for all datetimes.
"""

import json
from datetime import timedelta

import pytest

from services.act_trail import ActTrail
from services.time_utils import utc_now

pytestmark = pytest.mark.unit


# ── helpers ───────────────────────────────────────────────────────────────────


def _seed_transcript(db, channel="user", role="user", content="test") -> int:
    db.execute(
        "INSERT INTO transcript (channel, role, content, xml_migrated) "
        "VALUES (?, ?, ?, 1)",
        (channel, role, content),
    )
    return db.execute("SELECT last_insert_rowid()").fetchone()[0]


# ── B. Rich-card pipeline: row survives + SegmentService builds rich segment ──


_WEATHER_DICT = {
    "location": "Valletta, MT",
    "condition": "Sunny",
    "temperature_c": 27.0,
    "temperature_f": 80.6,
    "feels_like_c": 25.0,
    "humidity_pct": 55,
    "wind_kmh": 12.0,
    "wind_direction": "N",
    "visibility_km": 20.0,
    "uv_index": 7,
    "precip_mm": 0.0,
    "observation_time": "2026-06-12T10:00",
    "is_raining": False,
    "is_daylight": True,
    "is_hot": True,
    "is_cold": False,
    "is_windy": False,
    "is_clear": True,
    "forecast_tomorrow_condition": "Mostly sunny",
    "forecast_tomorrow_max_c": 28.0,
    "forecast_tomorrow_min_c": 20.0,
    "forecast_tomorrow_precip_chance_pct": 5,
    "forecast_tomorrow_precip_mm": 0.0,
}

_RICH_RESULT = (
    json.dumps(_WEATHER_DICT)
    + "\n\n"
    + "This tool supports rich-media rendering. "
    + "wrap your synthesis in <span id='weather_1'>your synthesis here</span>."
)


def test_tool_calls_row_survives_turn_and_segment_service_builds_rich_card(db):
    """The tool_calls row written by ActTrail must persist after the turn and
    allow SegmentService.build() to produce a rich segment.

    Before TKT-947: _purge_ephemeral_tool_calls() deleted the row before
    SegmentService ran, so the span tag became an orphan and no card was produced.
    After TKT-947: every row is durable; SegmentService reads it and returns
    type=rich with the correct payload.
    """
    tid = _seed_transcript(db, content="What's the weather in Valletta?")
    db.commit()

    # Write via the real production path (ActTrail.record — the only write path)
    ActTrail().record(
        tool_name="weather",
        params={"location": "Valletta"},
        result=_RICH_RESULT,
        transcript_id=tid,
    )

    # Assert the row is present (survives — no purge removed it)
    rows = db.execute(
        "SELECT id, tool_name FROM tool_calls WHERE transcript_id = ?", (tid,)
    ).fetchall()
    assert len(rows) == 1, (
        "tool_calls row was not found — purge must have been re-introduced or "
        "ActTrail.record failed silently"
    )
    assert rows[0][1] == "weather"

    # Now assert SegmentService.build() produces a rich segment
    assistant_content = "Here is the weather. <span id='weather_1'>Sunny, 27°C.</span>"
    from services.segment_service import SegmentService

    segments = SegmentService.build(assistant_content, [tid])

    rich = [s for s in segments if s["type"] == "rich"]
    assert len(rich) == 1, (
        f"Expected 1 rich segment, got {len(rich)}. Segments: {segments}. "
        "If this is 0, the span tag became an orphan — the row was purged before "
        "SegmentService ran (the TKT-947 regression is back)."
    )
    assert rich[0]["tag"] == "weather_1"
    assert rich[0]["synthesis"] == "Sunny, 27°C."
    assert rich[0]["payload"]["location"] == "Valletta, MT"
    assert rich[0]["payload"]["temperature_c"] == pytest.approx(27.0)


# ── C. Janitor: 7-day time-based purge ────────────────────────────────────────


def test_janitor_deletes_old_rows_and_keeps_recent(db):
    """DecayEngineService._purge_tool_calls() must delete rows older than 7 days
    and leave rows within the 7-day window intact.

    Seeded directly via SQL (matching prod schema) so the test does not depend on
    the janitor's own transport or the ActTrail write path — the assertion is
    purely about what the purge deletes.
    """
    tid = _seed_transcript(db)
    db.commit()

    old_ts = (utc_now() - timedelta(days=8)).isoformat()
    recent_ts = (utc_now() - timedelta(days=1)).isoformat()

    db.execute(
        "INSERT INTO tool_calls (transcript_id, tool_name, params, result, created_at) "
        "VALUES (?, 'memory', '{}', 'old result', ?)",
        (tid, old_ts),
    )
    db.execute(
        "INSERT INTO tool_calls (transcript_id, tool_name, params, result, created_at) "
        "VALUES (?, 'search', '{}', 'recent result', ?)",
        (tid, recent_ts),
    )
    db.commit()

    before = db.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
    assert before == 2

    from services.decay_engine_service import DecayEngineService

    deleted = DecayEngineService()._purge_tool_calls()

    assert deleted == 1, (
        f"Expected 1 deleted row (the 8-day-old one), got {deleted}. "
        "Either the janitor is not using time-based deletion or the cutoff is wrong."
    )

    remaining = db.execute(
        "SELECT tool_name FROM tool_calls"
    ).fetchall()
    assert len(remaining) == 1
    assert remaining[0][0] == "search", (
        "The recent row was deleted — janitor is too aggressive"
    )


# ── D. review_tool_calls filters narration rows ───────────────────────────────


def test_review_tool_calls_excludes_narration_rows(db):
    """A narration row in the review window must be absent from results;
    a normal tool call in the same window must be present.

    Narration rows (tool_name='narration') are mid-loop LLM text blobs.
    Decision 4 keeps them in tool_calls for the trail but filters them from
    review_tool_calls so they do not pollute the user-facing tool audit.
    """
    from abilities._dispatcher import ToolDispatcher
    from configs.channels import UserConfig
    from tests._tool_result_harness import MP, seed_transcript

    anchor_ts = "2026-04-07T14:30:00+00:00"
    inside_ts = "2026-04-07T14:28:00+00:00"

    tid = seed_transcript(db, content="review recent tools")

    # Seed a narration row inside the window
    db.execute(
        "INSERT INTO tool_calls (transcript_id, tool_name, params, result, created_at) "
        "VALUES (?, 'narration', '{}', 'Some mid-loop text the model emitted', ?)",
        (tid, inside_ts),
    )
    # Seed a real tool call inside the window
    db.execute(
        "INSERT INTO tool_calls (transcript_id, tool_name, params, result, created_at) "
        "VALUES (?, 'weather', '{\"location\": \"Malta\"}', '[weather(status=success)]\\n{\"temperature_c\": 27}\\n[end:weather]', ?)",
        (tid, inside_ts),
    )
    db.commit()

    mp = MP(tid, UserConfig({}))
    rendered = ToolDispatcher(mp).dispatch("review_tool_calls", {"date_time": anchor_ts})

    # narration must not appear in the results
    assert "narration" not in rendered, (
        "narration row appeared in review_tool_calls output — filter was removed"
    )
    # the real tool call must appear
    assert "weather" in rendered, (
        "weather tool call was not found in review_tool_calls output"
    )
