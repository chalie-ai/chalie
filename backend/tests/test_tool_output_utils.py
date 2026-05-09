"""
Behavioural tests for tool_output_utils — telemetry construction.

build_tool_telemetry is deterministic with zero collaborators, qualifying
for unit tests per tester.md.
"""

import pytest

from services.tool_output_utils import build_tool_telemetry

pytestmark = pytest.mark.unit


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
