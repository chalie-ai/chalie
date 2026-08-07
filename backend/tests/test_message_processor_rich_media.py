"""Integration test: rich-media cards in the message processing pipeline."""

import json
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

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

    def test_render_with_ordinal_produces_json_and_trailer(self) -> None:
        from services.dispatch_service import DispatchService
        from abilities._result import ToolResult

        tr = ToolResult.ok({"temperature_c": 12}, rich={"temperature_c": 12})
        rendered = DispatchService(mp=cast("MessageProcessor", None))._render("weather", tr, 1)
        inner = rendered.split("\n", 1)[1].rsplit("\n", 1)[0]

        payload_str, trailer = inner.split("\n\n", 1)
        assert json.loads(payload_str) == {"temperature_c": 12}
        assert "<span id='weather_1'>" in trailer

    def test_render_without_ordinal_has_no_trailer(self) -> None:
        from services.dispatch_service import DispatchService
        from abilities._result import ToolResult

        tr = ToolResult.ok({"temperature_c": 12}, rich={"temperature_c": 12})
        rendered = DispatchService(mp=cast("MessageProcessor", None))._render("weather", tr, None)
        inner = rendered.split("\n", 1)[1].rsplit("\n", 1)[0]
        # No ordinal → body is the bare payload JSON, no instruction trailer.
        assert json.loads(inner) == {"temperature_c": 12}
        assert "<span" not in rendered

    def test_render_error_payload_no_trailer(self) -> None:
        from services.dispatch_service import DispatchService
        from abilities._result import ToolResult

        tr = ToolResult.err("unavailable", code="provider-unreachable", details="timeout")
        rendered = DispatchService(mp=cast("MessageProcessor", None))._render("weather", tr, 1)
        # Error envelopes never carry a rich-media instruction trailer.
        assert "status=error" in rendered
        assert "unavailable" in rendered
        assert "<span" not in rendered

    def test_render_ordinal_2_uses_correct_tag(self) -> None:
        from services.dispatch_service import DispatchService
        from abilities._result import ToolResult

        tr = ToolResult.ok({"temperature_c": 20}, rich={"temperature_c": 20})
        rendered = DispatchService(mp=cast("MessageProcessor", None))._render("weather", tr, 2)
        assert "<span id='weather_2'>" in rendered


class TestRichMediaParserIntegration:
    """Parser round-trip: tool result → parse() → segments."""

    def test_parse_produces_rich_segment_for_london(self) -> None:
        from services.rich_media_parser import RichMediaParser

        tool_result = (
            json.dumps(_MOCK_WEATHER_DICT)
            + "\n\n"
            + "This tool supports rich-media rendering. "
            + "To present this result as a card, "
            + "wrap your synthesis in <span id='weather_1'>your synthesis here</span>."
        )
        tool_calls = [{"tool_name": "weather", "params": "{}", "result": tool_result, "ephemeral": 1}]
        content = "Here is the weather. <span id='weather_1'>Partly cloudy, 12°C.</span>"
        segs = RichMediaParser.parse(content, tool_calls)

        assert len(segs) == 2
        text_seg = segs[0]
        rich_seg = segs[1]

        assert text_seg["type"] == "text"
        assert "Here is the weather." in cast(str, text_seg["content"])

        assert rich_seg["type"] == "rich"
        assert rich_seg["tag"] == "weather_1"
        assert rich_seg["synthesis"] == "Partly cloudy, 12°C."
        assert cast("dict[str, object]", rich_seg["payload"])["location"] == "London, GB"
        assert cast("dict[str, object]", rich_seg["payload"])["temperature_c"] == pytest.approx(12.4, abs=1e-9)

    def test_parse_two_cities_produces_two_rich_segments(self) -> None:
        from services.rich_media_parser import RichMediaParser

        def _make_result(loc: str, ordinal: int) -> str:
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
        segs = RichMediaParser.parse(content, tool_calls)
        rich_segs = [s for s in segs if s["type"] == "rich"]
        assert len(rich_segs) == 2
        assert rich_segs[0]["tag"] == "weather_1"
        assert cast("dict[str, object]", rich_segs[0]["payload"])["location"] == "Paris, FR"
        assert rich_segs[1]["tag"] == "weather_2"
        assert cast("dict[str, object]", rich_segs[1]["payload"])["location"] == "Tokyo, JP"
