"""
Behavioural tests for tool_output_utils — pure formatting functions.

Both format_tool_result and build_tool_telemetry are deterministic with zero
collaborators, so they qualify for unit tests per tester.md.
"""

import pytest

from services.tool_output_utils import format_tool_result, build_tool_telemetry

pytestmark = pytest.mark.unit


# ── format_tool_result ──────────────────────────────────────────────────────

class TestFormatToolResult:
    def test_str_passthrough(self):
        assert format_tool_result("hello") == "hello"

    def test_non_dict_falls_back_to_str(self):
        assert format_tool_result(42) == "42"

    def test_empty_results_shows_message(self):
        assert format_tool_result({"results": [], "message": "nope"}) == "nope"

    def test_search_results_formats_numbered_list(self):
        result = format_tool_result({
            "results": [
                {"title": "First", "snippet": "desc", "url": "http://a"},
                {"title": "Second", "snippet": "", "url": ""},
            ],
        })
        lines = result.split("\n")
        assert lines[0] == "1. First"
        assert lines[1] == "   desc"
        assert lines[2] == "   http://a"
        assert "2. Second" in result

    def test_search_results_shows_count_when_present(self):
        result = format_tool_result({
            "results": [{"title": "X"}],
            "count": 1,
        })
        assert "1 results returned." in result

    def test_page_content_with_truncation(self):
        result = format_tool_result({
            "content": "some text",
            "truncated": True,
            "char_count": 500,
        })
        assert "some text" in result
        assert "truncated to 500 chars" in result

    def test_page_content_error(self):
        result = format_tool_result({"content": "", "error": "timeout"})
        assert result == "Error: timeout"

    def test_generic_dict_skips_budget_remaining(self):
        result = format_tool_result({"a": 1, "budget_remaining": 500})
        assert "a: 1" in result
        assert "budget_remaining" not in result

    def test_generic_dict_lists_serialized(self):
        result = format_tool_result({"items": [1, 2]})
        assert "items: [1, 2]" in result


# ── build_tool_telemetry ────────────────────────────────────────────────────

class TestBuildToolTelemetry:
    def test_extracts_location_fields(self):
        raw = {
            "location": {"lat": 51.5, "lon": -0.1},
            "location_name": "London, UK",
            "local_time": "14:30",
            "locale": "en-GB",
            "language": "en",
            "device": {},
        }
        out = build_tool_telemetry(raw)
        assert out["lat"] == pytest.approx(51.5, abs=1e-9)
        assert out["lon"] == pytest.approx(-0.1, abs=1e-9)
        assert out["city"] == "London"
        assert out["country"] == "UK"
        assert out["time"] == "14:30"
        assert out["locale"] == "en-GB"
        assert out["language"] == "en"

    def test_handles_missing_optional_fields(self):
        raw = {}
        out = build_tool_telemetry(raw)
        assert out["lat"] is None
        assert out["city"] == ""
        assert out["country"] == ""
        assert out["time"] == ""
        assert "device_class" not in out

    def test_includes_device_context_when_present(self):
        raw = {"device": {"class": "mobile", "platform": "ios", "pwa": True}}
        out = build_tool_telemetry(raw)
        assert out["device_class"] == "mobile"
        assert out["platform"] == "ios"
        assert out["pwa"] is True
