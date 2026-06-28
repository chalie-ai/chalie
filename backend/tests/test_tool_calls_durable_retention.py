# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for tool_calls durability and unified retention.

Two guarantees, both pinned against the production hot path:
  * Rich-card rendering: a recorded tool_calls row survives its turn so
    SegmentService / the live broadcast can build the rich segment from it.
  * Unified retention: a tool_calls audit row lives and dies with its transcript
    turn. The transcript GC reaps it exactly when it reaps the turn — below the
    compaction watermark and uncited by any live episode. The old standalone
    7-day janitor (_purge_tool_calls) is gone; there is one cleanup path.
"""

import json
import sqlite3
from typing import cast

import pytest

from services.act_trail import ActTrail
from services.decay_engine_service import DecayEngineService
from services.transcript_service import Transcript

pytestmark = pytest.mark.unit


# ── helpers ───────────────────────────────────────────────────────────────────


def _seed_transcript(db: sqlite3.Connection, channel: str = "user", role: str = "user", content: str = "test") -> int:
    db.execute(
        "INSERT INTO transcript (channel, role, content, xml_migrated) "
        "VALUES (?, ?, ?, 1)",
        (channel, role, content),
    )
    return cast(int, db.execute("SELECT last_insert_rowid()").fetchone()[0])


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


def test_tool_calls_row_survives_turn_and_segment_service_builds_rich_card(db: sqlite3.Connection) -> None:
    """This is the direct proof that 's purge-before-segment-build bug is fixed."""
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
    assert cast(tuple[object, ...], rows[0])[1] == "weather"

    # Now assert SegmentService.build() produces a rich segment
    assistant_content = "Here is the weather. <span id='weather_1'>Sunny, 27°C.</span>"
    from services.segment_service import SegmentService

    segments = SegmentService.build(assistant_content, [tid])

    rich = [s for s in segments if s["type"] == "rich"]
    assert len(rich) == 1, (
        f"Expected 1 rich segment, got {len(rich)}. Segments: {segments}. "
        "If this is 0, the span tag became an orphan — the row was purged before "
        "SegmentService ran (the  regression is back)."
    )
    assert rich[0]["tag"] == "weather_1"
    assert rich[0]["synthesis"] == "Sunny, 27°C."
    assert cast(dict[str, object], rich[0]["payload"])["location"] == "Valletta, MT"
    assert cast(dict[str, object], rich[0]["payload"])["temperature_c"] == pytest.approx(27.0)


# ── B2. Live broadcast path: span pairs with tool_calls after assistant persist ──


def test_broadcast_turn_result_pairs_span_after_assistant_row_persisted(db: sqlite3.Connection) -> None:
    """The live WS broadcast must pair the span tag with the turn's tool_calls
    even though the assistant row is already persisted when it runs.

    Regression (observed live 2026-06-12, transcript 6263): _broadcast_turn_result
    resolved the turn's transcript id via get_recent("user", limit=1), which
    returns the newest channel row of ANY role — the just-persisted assistant
    row, not the user row tool_calls are keyed to.  The tool_calls fetch came
    back empty, the span became an orphan, and the card degraded to plain text
    on the live path (page refresh rendered fine).  The fix resolves backwards
    from the assistant row via resolve_tool_call_transcript_ids — the same
    shared function the refresh path uses.
    """
    import time

    # Chain model: the tool anchors to the assistant STEP row, the synthesis span
    # lands on the FINAL row, and all three rows share the turn_id the input row
    # opened. _broadcast_turn_result resolves TURN-WIDE from the newest channel
    # row (the final assistant row) so the final-row span pairs with the step
    # row's tool. Seeded through the production writers + ActTrail.record — the
    # same path the live chain takes.
    uid = Transcript.write_input_row("user", "user", "What's the weather in Valletta?")
    turn = Transcript.turn_id_of_row(uid)
    step_id = Transcript.write_assistant_row("user", "Let me check the weather.", turn_id=turn)
    ActTrail().record(
        tool_name="weather",
        params={"location": "Valletta"},
        result=_RICH_RESULT,
        transcript_id=step_id,
    )

    # Final synthesis row persisted BEFORE the broadcast runs — reproducing the
    # live ordering where the newest channel row is the turn's final assistant row.
    assistant_content = "<span id='weather_1'>Sunny, 27°C.</span>"
    Transcript.write_assistant_row("user", assistant_content, turn_id=turn)

    # Real consumer at the WS boundary: the broker pushes JSON strings to the
    # connected object exactly as it does to the browser connection.
    received: list[dict[str, object]] = []

    class _Receiver:
        def send(self, raw: str) -> None:
            received.append(json.loads(raw))

    from services.websocket_broker import WebSocketBroker

    broker = WebSocketBroker()
    receiver = _Receiver()
    broker.connect(receiver)
    try:
        from api.chat import _broadcast_turn_result

        _broadcast_turn_result(assistant_content, "req-test", time.time())
    finally:
        broker.disconnect(receiver)

    messages = [m for m in received if m.get("type") == "message"]
    assert len(messages) == 1, f"Expected 1 message event, got: {received}"
    rich = [s for s in cast(list[dict[str, object]], messages[0]["segments"]) if s["type"] == "rich"]
    assert len(rich) == 1, (
        f"Expected 1 rich segment on the live broadcast, got segments: "
        f"{messages[0]['segments']}. If 0, the transcript-id resolution picked "
        "the assistant row again and the span orphaned (the live-path regression "
        "is back)."
    )
    assert rich[0]["tag"] == "weather_1"
    assert cast(dict[str, object], rich[0]["payload"])["location"] == "Valletta, MT"
    done = [m for m in received if m.get("type") == "done"]
    assert len(done) == 1, "broadcast must still emit the done event"


# ── C. Unified retention: tool_calls die with their transcript turn ───────────


def test_transcript_gc_reaps_tool_calls_of_obsolete_turns_only(db: sqlite3.Connection) -> None:
    """A tool_calls row lives and dies with its turn — no separate clock.

    Drives the real decay entry point (the all-channels transcript GC) and pins
    every side of the unified rule in one run:
      * below watermark + uncited       -> the turn AND its tool_calls are reaped
      * below watermark + episode-cited  -> both survive (evidence never orphaned)
      * above the watermark              -> both survive (still live history)
    """
    channel = "user"

    # Two old turns through the production writer; one tool call recorded on each
    # via ActTrail.record — the only production tool_calls write path.
    obsolete = Transcript.write_input_row(channel, "user", "obsolete turn with a tool call")
    cited = Transcript.write_input_row(channel, "user", "turn an episode still cites")
    ActTrail().record(tool_name="weather", params={"location": "Valletta"}, result=_RICH_RESULT, transcript_id=obsolete)
    ActTrail().record(tool_name="memory", params={}, result="recalled fact", transcript_id=cited)

    # A live episode cites the second turn — the GC must protect its tool call too.
    db.execute(
        "INSERT INTO episodes (id, gist, salience, channel, transcript_ids) VALUES (?, ?, ?, ?, ?)",
        ("ep-cite", "covers the cited turn", 5, channel, json.dumps([cited])),
    )
    db.commit()

    # The compaction watermark row, written exactly as the compactor does.
    Transcript.write_input_row(channel, "compaction", "## Voice\nliving checkpoint")

    # A post-watermark turn with its own tool call — still live, must survive.
    live = Transcript.write_input_row(channel, "user", "post-watermark turn")
    ActTrail().record(tool_name="search", params={}, result="fresh result", transcript_id=live)

    # Real decay entry point — the all-channels discovery path, same as prod.
    DecayEngineService()._cleanup_transcript()

    def _tool_calls_on(tid: int) -> int:
        return cast(int, db.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE transcript_id = ?", (tid,)
        ).fetchone()[0])

    assert _tool_calls_on(obsolete) == 0, "tool_calls of the reaped obsolete turn must be gone"
    assert _tool_calls_on(cited) == 1, "an episode-cited turn's tool_calls must survive"
    assert _tool_calls_on(live) == 1, "tool_calls above the watermark must survive"


# ── D. review_tool_calls filters narration rows ───────────────────────────────


def test_review_tool_calls_excludes_narration_rows(db: sqlite3.Connection) -> None:
    """A narration row in the review window must be absent from results;
    a normal tool call in the same window must be present.

    Narration rows (tool_name='narration') are mid-loop LLM text blobs.
    They are kept in tool_calls for the trail but filtered from
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
