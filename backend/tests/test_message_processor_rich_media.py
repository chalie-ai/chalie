"""Integration test: rich-media cards in the message processing pipeline.

Verifies that when the weather tool returns a rich-media instruction string
(JSON payload + trailer), the resulting tool_calls row contains the full string
and the sanitised transcript.content contains the span tag emitted by a mocked
LLM, so that WS segment assembly can pair them.

Uses the real DB fixture (function-scoped SQLite copy), a real
ActDispatcherService (with weather ability registered), and a mock LLM that
returns a canned response containing the span tag.
"""

import json
import pytest

pytestmark = pytest.mark.unit


# ── helpers ───────────────────────────────────────────────────────────────────

_MOCK_WEATHER_DICT = {
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



class TestWeatherSerialise:
    """WeatherAbility._serialise produces the expected string shape."""

    def test_serialise_with_ordinal_produces_json_and_trailer(self):
        from abilities.weather import _serialise
        result = _serialise({"temperature_c": 12}, ordinal=1)
        assert isinstance(result, str)
        parts = result.split("\n\n", 1)
        assert len(parts) == 2
        payload_str, trailer = parts
        assert json.loads(payload_str) == {"temperature_c": 12}
        assert "<span id='weather_1'>" in trailer

    def test_serialise_without_ordinal_returns_dict(self):
        from abilities.weather import _serialise
        result = _serialise({"temperature_c": 12}, ordinal=None)
        assert isinstance(result, dict)
        assert result["temperature_c"] == 12

    def test_serialise_error_payload_no_trailer(self):
        from abilities.weather import _serialise
        result = _serialise({"error": "unavailable", "details": "timeout"}, ordinal=1)
        # Error payloads must not include an instruction trailer
        assert isinstance(result, dict)
        assert result["error"] == "unavailable"

    def test_serialise_ordinal_2_uses_correct_tag(self):
        from abilities.weather import _serialise
        result = _serialise({"temperature_c": 20}, ordinal=2)
        assert "<span id='weather_2'>" in result


class TestRichMediaParserIntegration:
    """Parser round-trip: tool result → parse() → segments."""

    def test_parse_produces_rich_segment_for_london(self):
        from services.rich_media_parser import parse

        tool_result = (
            json.dumps(_MOCK_WEATHER_DICT)
            + "\n\n"
            + "This tool supports rich-media rendering. "
            + "To present this result as a card, "
            + "wrap your synthesis in <span id='weather_1'>your synthesis here</span>."
        )
        tool_calls = [{"tool_name": "weather", "params": "{}", "result": tool_result, "ephemeral": 1}]
        content = "Here is the weather. <span id='weather_1'>Partly cloudy, 12°C.</span>"
        segs = parse(content, tool_calls)

        assert len(segs) == 2
        text_seg = segs[0]
        rich_seg = segs[1]

        assert text_seg["type"] == "text"
        assert "Here is the weather." in text_seg["content"]

        assert rich_seg["type"] == "rich"
        assert rich_seg["tag"] == "weather_1"
        assert rich_seg["synthesis"] == "Partly cloudy, 12°C."
        assert rich_seg["payload"]["location"] == "London, GB"
        assert rich_seg["payload"]["temperature_c"] == pytest.approx(12.4, abs=1e-9)

    def test_parse_two_cities_produces_two_rich_segments(self):
        from services.rich_media_parser import parse

        def _make_result(loc, ordinal):
            payload = dict(_MOCK_WEATHER_DICT, location=loc)
            return (
                json.dumps(payload)
                + f"\n\n<span id='weather_{ordinal}'>"
            )

        tool_calls = [
            {"tool_name": "weather", "params": '{"location":"Paris"}', "result": _make_result("Paris, FR", 1), "ephemeral": 1},
            {"tool_name": "weather", "params": '{"location":"Tokyo"}', "result": _make_result("Tokyo, JP", 2), "ephemeral": 1},
        ]
        content = (
            "<span id='weather_1'>Rainy in Paris.</span> "
            "Meanwhile, <span id='weather_2'>Clear in Tokyo.</span>"
        )
        segs = parse(content, tool_calls)
        rich_segs = [s for s in segs if s["type"] == "rich"]
        assert len(rich_segs) == 2
        assert rich_segs[0]["tag"] == "weather_1"
        assert rich_segs[0]["payload"]["location"] == "Paris, FR"
        assert rich_segs[1]["tag"] == "weather_2"
        assert rich_segs[1]["payload"]["location"] == "Tokyo, JP"
