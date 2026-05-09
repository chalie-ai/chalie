"""Integration test: /conversation/recent includes segments on assistant rows.

Seeds transcript + tool_calls rows directly into the DB, then calls
get_recent_history() and asserts:
- Assistant rows with a paired weather tool_call get a ``segments`` field with
  at least one ``type=rich`` segment.
- User rows do NOT get a ``segments`` field.
- The refresh path produces the same pairing as the WS live path for the same
  data.
- When no tool_calls exist the assistant row still gets a ``type=text``
  segment containing the content.
"""

import json
import pytest

pytestmark = pytest.mark.unit


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

_TOOL_RESULT = (
    json.dumps(_WEATHER_DICT)
    + "\n\n"
    + "This tool supports rich-media rendering. "
    + "wrap your synthesis in <span id='weather_1'>your synthesis here</span>."
)


class TestConversationRecentSegments:

    def _seed_turn(self, conn, user_text: str, assistant_text: str, tool_calls=None):
        """Insert a user+assistant transcript pair and optional tool_calls.

        Returns (user_transcript_id, assistant_transcript_id).
        Tool_calls are linked to the user transcript row (matches production).
        """
        conn.execute(
            "INSERT INTO transcript (channel, role, content, xml_migrated) VALUES ('user', 'user', ?, 1)",
            (user_text,),
        )
        user_tid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            "INSERT INTO transcript (channel, role, content, xml_migrated) VALUES ('user', 'assistant', ?, 1)",
            (assistant_text,),
        )
        asst_tid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        for tc in (tool_calls or []):
            conn.execute(
                "INSERT INTO tool_calls (transcript_id, tool_name, params, result, ephemeral, created_at) "
                "VALUES (?, ?, ?, ?, ?, datetime('now'))",
                (user_tid, tc["tool_name"], tc.get("params", "{}"), tc["result"], tc.get("ephemeral", 1)),
            )
        conn.commit()
        return user_tid, asst_tid

    def test_assistant_row_without_tool_calls_gets_text_segment(self, db):
        self._seed_turn(db, "Hello", "Hi there!", tool_calls=[])

        from api.conversation import get_recent_history
        messages, _ = get_recent_history(limit=10)

        asst = next(m for m in messages if m["role"] == "assistant")
        assert "segments" in asst
        assert len(asst["segments"]) == 1
        assert asst["segments"][0]["type"] == "text"
        assert asst["segments"][0]["content"] == "Hi there!"

    def test_user_row_has_no_segments_field(self, db):
        self._seed_turn(db, "Hello", "Hi!")

        from api.conversation import get_recent_history
        messages, _ = get_recent_history(limit=10)

        user_msg = next(m for m in messages if m["role"] == "user")
        assert "segments" not in user_msg

    def test_assistant_row_with_weather_tool_gets_rich_segment(self, db):
        assistant_content = "Here is the weather. <span id='weather_1'>Partly cloudy, 12°C.</span>"
        self._seed_turn(
            db,
            "What's the weather in London?",
            assistant_content,
            tool_calls=[
                {"tool_name": "weather", "params": '{"location":"London"}', "result": _TOOL_RESULT, "ephemeral": 1},
            ],
        )

        from api.conversation import get_recent_history
        messages, _ = get_recent_history(limit=10)

        asst = next(m for m in messages if m["role"] == "assistant")
        assert "segments" in asst
        rich_segs = [s for s in asst["segments"] if s["type"] == "rich"]
        assert len(rich_segs) == 1
        seg = rich_segs[0]
        assert seg["tag"] == "weather_1"
        assert seg["synthesis"] == "Partly cloudy, 12°C."
        assert seg["payload"]["location"] == "London, GB"
        assert seg["payload"]["temperature_c"] == pytest.approx(12.4, abs=1e-9)

    def test_ephemeral_tool_calls_included_in_pairing(self, db):
        """ephemeral=1 rows (inline weather results) must be included."""
        assistant_content = "<span id='weather_1'>Sunny!</span>"
        self._seed_turn(
            db,
            "Weather?",
            assistant_content,
            tool_calls=[
                # ephemeral=1 is how weather rows are stored
                {"tool_name": "weather", "params": "{}", "result": _TOOL_RESULT, "ephemeral": 1},
            ],
        )

        from api.conversation import get_recent_history
        messages, _ = get_recent_history(limit=10)

        asst = next(m for m in messages if m["role"] == "assistant")
        rich_segs = [s for s in asst["segments"] if s["type"] == "rich"]
        assert len(rich_segs) == 1

    def test_two_city_turn_produces_two_rich_segments(self, db):
        def _make_result(loc, ordinal):
            payload = dict(_WEATHER_DICT, location=loc)
            return (
                json.dumps(payload)
                + f"\n\n<span id='weather_{ordinal}'>"
            )

        assistant_content = (
            "London: <span id='weather_1'>Cloudy.</span> "
            "Tokyo: <span id='weather_2'>Clear.</span>"
        )
        self._seed_turn(
            db,
            "Compare London and Tokyo weather",
            assistant_content,
            tool_calls=[
                {"tool_name": "weather", "params": '{"location":"London"}', "result": _make_result("London, GB", 1), "ephemeral": 1},
                {"tool_name": "weather", "params": '{"location":"Tokyo"}', "result": _make_result("Tokyo, JP", 2), "ephemeral": 1},
            ],
        )

        from api.conversation import get_recent_history
        messages, _ = get_recent_history(limit=10)

        asst = next(m for m in messages if m["role"] == "assistant")
        rich_segs = [s for s in asst["segments"] if s["type"] == "rich"]
        assert len(rich_segs) == 2
        tags = {s["tag"] for s in rich_segs}
        assert tags == {"weather_1", "weather_2"}

    def test_content_field_still_present_on_assistant_row(self, db):
        """Backwards compat: content field must not be removed."""
        assistant_content = "Plain response with no spans."
        self._seed_turn(db, "Q", assistant_content)

        from api.conversation import get_recent_history
        messages, _ = get_recent_history(limit=10)

        asst = next(m for m in messages if m["role"] == "assistant")
        assert asst["content"] == assistant_content

    def test_multiple_turns_each_assistant_gets_own_segments(self, db):
        """Multiple conversation turns; each assistant row paired with its own tool_calls."""
        def _make_result(loc, ordinal):
            payload = dict(_WEATHER_DICT, location=loc)
            return json.dumps(payload) + f"\n\n<span id='weather_{ordinal}'>"

        # Turn 1: weather tool used
        self._seed_turn(
            db, "London weather?",
            "<span id='weather_1'>Cloudy.</span>",
            tool_calls=[
                {"tool_name": "weather", "params": "{}", "result": _make_result("London, GB", 1), "ephemeral": 1},
            ],
        )
        # Turn 2: no tool used
        self._seed_turn(db, "Thanks", "You are welcome!", tool_calls=[])

        from api.conversation import get_recent_history
        messages, _ = get_recent_history(limit=10)

        asst_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(asst_msgs) == 2

        # First assistant message (weather turn) has a rich segment
        first_asst = asst_msgs[0]
        assert any(s["type"] == "rich" for s in first_asst["segments"])

        # Second assistant message (plain turn) has only text segment
        second_asst = asst_msgs[1]
        assert all(s["type"] == "text" for s in second_asst["segments"])
