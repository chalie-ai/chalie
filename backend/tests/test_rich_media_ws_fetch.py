"""Integration tests for the WS-boundary tool_calls fetch path.

Regression class: _fetch_tool_calls_for_recent_user_turn(channel) queries
    SELECT id FROM transcript WHERE channel = ? AND role = 'user'
SubagentProcessor writes input rows with role='subagent', not role='user'.
If the WS handler receives a message whose topic='subagent' (because the
message came from a SubagentProcessor), it passes 'subagent' as the channel.
The query finds no user-role row for that channel and returns [].

This means a subagent calling weather twice in one turn will NOT have its
tool_calls rows found, so _attach_segments() receives an empty list and
every span becomes an orphan — cards never render.

These tests assert the current behaviour so the arbiter can evaluate whether
to fix the fetch query or surface this as a known limitation.

Additionally tested:
- Action-button path: _attach_segments with empty tool_calls produces a
  single text segment, not an error.
- Two consecutive turns (two separate ActDispatcherService instances) each
  start ordinals at 1 (implicit reset via fresh instantiation).
"""

import json
import pytest

pytestmark = pytest.mark.integration


_WEATHER_DICT = {
    "location": "London, GB",
    "condition": "Partly cloudy",
    "temperature_c": 12.4,
    "temperature_f": 54.3,
    "feels_like_c": 10.1,
    "humidity_pct": 78,
    "wind_kmh": 14.2,
    "wind_direction": "WSW",
    "visibility_km": None,
    "uv_index": None,
    "precip_mm": 0.0,
    "observation_time": "2026-05-02T14:00",
    "is_raining": False,
    "is_daylight": True,
    "is_hot": False,
    "is_cold": False,
    "is_windy": False,
    "is_clear": False,
    "forecast_tomorrow_condition": "Slight rain",
    "forecast_tomorrow_max_c": 14.0,
    "forecast_tomorrow_min_c": 9.0,
    "forecast_tomorrow_precip_chance_pct": 70,
    "forecast_tomorrow_precip_mm": 3.2,
}


def _make_tool_result(ordinal: int, location: str = "London, GB") -> str:
    payload = dict(_WEATHER_DICT, location=location)
    return json.dumps(payload) + f"\n\n<span id='weather_{ordinal}'>"


class TestWsFetchToolCallsForUserChannel:
    """_fetch_tool_calls_for_recent_user_turn resolves tool_calls correctly
    when the transcript row has role='user' (standard UMP path)."""

    def test_user_channel_returns_weather_tool_calls(self, db):
        # Seed a user-channel transcript row (role='user') with a weather tool_call.
        db.execute(
            "INSERT INTO transcript (channel, role, content, xml_migrated) "
            "VALUES ('user', 'user', 'What is the weather?', 1)"
        )
        transcript_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, ephemeral, created_at) "
            "VALUES (?, 'weather', '{}', ?, 1, datetime('now'))",
            (transcript_id, _make_tool_result(1)),
        )
        db.commit()

        from api.websocket import _fetch_tool_calls_for_recent_user_turn
        rows = _fetch_tool_calls_for_recent_user_turn("user")

        assert len(rows) == 1
        assert rows[0]["tool_name"] == "weather"
        assert "<span id='weather_1'>" in rows[0]["result"]

    def test_user_channel_two_weather_calls_both_returned(self, db):
        # Two weather tool_calls in the same turn must both be returned for
        # multi-card pairing to work.
        db.execute(
            "INSERT INTO transcript (channel, role, content, xml_migrated) "
            "VALUES ('user', 'user', 'Compare London and Tokyo', 1)"
        )
        transcript_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, ephemeral, created_at) "
            "VALUES (?, 'weather', '{\"location\":\"London\"}', ?, 1, datetime('now', '-1 second'))",
            (transcript_id, _make_tool_result(1, "London, GB")),
        )
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, ephemeral, created_at) "
            "VALUES (?, 'weather', '{\"location\":\"Tokyo\"}', ?, 1, datetime('now'))",
            (transcript_id, _make_tool_result(2, "Tokyo, JP")),
        )
        db.commit()

        from api.websocket import _fetch_tool_calls_for_recent_user_turn
        rows = _fetch_tool_calls_for_recent_user_turn("user")

        assert len(rows) == 2
        tags_found = {r["tool_name"] for r in rows}
        assert tags_found == {"weather"}
        results_combined = " ".join(r["result"] for r in rows)
        assert "<span id='weather_1'>" in results_combined
        assert "<span id='weather_2'>" in results_combined


class TestWsFetchToolCallsForSubagentChannel:
    """Asserts observed behaviour when channel='subagent'.

    SubagentProcessor writes input rows with role='subagent', NOT role='user'.
    The fetch query selects WHERE role='user', so it cannot find subagent rows.
    This test documents the known gap: subagent weather cards will not render
    via the WS live path.
    """

    def test_subagent_channel_role_subagent_not_found_by_user_query(self, db):
        # Seed a subagent-channel transcript row as SubagentProcessor writes it
        # (role='subagent', channel='subagent').
        db.execute(
            "INSERT INTO transcript (channel, role, content, xml_migrated) "
            "VALUES ('subagent', 'subagent', 'internal task', 1)"
        )
        transcript_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, params, result, ephemeral, created_at) "
            "VALUES (?, 'weather', '{}', ?, 1, datetime('now'))",
            (transcript_id, _make_tool_result(1)),
        )
        db.commit()

        from api.websocket import _fetch_tool_calls_for_recent_user_turn
        rows = _fetch_tool_calls_for_recent_user_turn("subagent")

        # KNOWN BUG: the query requires role='user'; subagent rows have
        # role='subagent'. Result is empty — weather cards from subagent turns
        # will not be paired and will render as orphan text segments instead.
        assert rows == [], (
            "Expected empty list because _fetch_tool_calls_for_recent_user_turn "
            "cannot resolve subagent tool_calls (role='subagent' != 'user'). "
            "If this assertion fails, the bug has been fixed — update this test."
        )


class TestAttachSegmentsActionButtonPath:
    """Action-button path: _attach_segments with empty tool_calls must produce
    a single text segment, not raise and not produce a rich segment."""

    def test_empty_tool_calls_produces_single_text_segment(self):
        from api.websocket import _attach_segments

        message_evt = {"type": "message", "content": ""}
        content = "Here is your result."
        _attach_segments(message_evt, content, [])

        assert "segments" in message_evt
        segments = message_evt["segments"]
        assert len(segments) == 1
        assert segments[0]["type"] == "text"
        assert segments[0]["content"] == "Here is your result."

    def test_empty_tool_calls_empty_content_no_segments_crash(self):
        from api.websocket import _attach_segments

        message_evt = {"type": "message", "content": ""}
        _attach_segments(message_evt, "", [])

        # parse("", []) returns [] — _attach_segments falls back to
        # a single text segment with the raw content ("").
        assert "segments" in message_evt
        # Either empty list or a single empty-text segment — must not raise.
        assert isinstance(message_evt["segments"], list)

    def test_html_content_with_no_tool_calls_emits_text_segment(self):
        from api.websocket import _attach_segments

        content = "<p>It is <b>sunny</b> today.</p>"
        message_evt = {"type": "message"}
        _attach_segments(message_evt, content, [])

        assert "segments" in message_evt
        assert all(s["type"] == "text" for s in message_evt["segments"])


class TestOrdinalCounterAcrossConsecutiveTurns:
    """Two consecutive turns must each start ordinals at 1.

    ActDispatcherService is instantiated fresh per turn inside
    MessageProcessor.send(). The fresh instance has _turn_ordinals = {}
    so ordinals restart at 1 each turn without an explicit reset call.
    """

    def test_two_dispatcher_instances_each_start_at_ordinal_1(self):
        from services.act_dispatcher_service import ActDispatcherService

        ordinals = []

        def _capture(channel, action):
            ordinals.append(action.get("_rich_media_ordinal"))
            return "ok"

        # Turn 1: fresh dispatcher, dispatch weather once → ordinal 1
        dispatcher_turn1 = ActDispatcherService()
        dispatcher_turn1.handlers["weather"] = _capture
        dispatcher_turn1.dispatch_action("user", {"type": "weather"})

        # Turn 2: new dispatcher (simulates new turn) → ordinal must restart at 1
        dispatcher_turn2 = ActDispatcherService()
        dispatcher_turn2.handlers["weather"] = _capture
        dispatcher_turn2.dispatch_action("user", {"type": "weather"})

        # Both dispatches must have produced ordinal 1, confirming fresh instances
        # do not share state.
        assert ordinals == [1, 1], (
            f"Expected [1, 1] (ordinal resets on new dispatcher instance), got {ordinals}"
        )
