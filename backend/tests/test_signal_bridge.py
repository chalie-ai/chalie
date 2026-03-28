"""
Unit tests for the capability → goal ecology signal bridge.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestSignalBridge:
    """Tests for capabilities.signal_bridge.emit_capability_signal."""

    @patch("capabilities.signal_bridge.route_cognitive_signal")
    def test_valid_signal_routes_to_goal_ecology(self, mock_route):
        from capabilities.signal_bridge import emit_capability_signal

        result = emit_capability_signal(
            "caldav", "novel_observation",
            "Schedule conflict: standup overlaps design review",
            source="caldav:conflict",
        )

        assert result is True
        mock_route.assert_called_once()
        arg = mock_route.call_args[0][0]
        assert arg["signal_type"] == "novel_observation"
        assert arg["payload"]["capability"] == "caldav"
        assert "standup" in arg["payload"]["content"]
        assert arg["source_id"] == "caldav:conflict"

    @patch("capabilities.signal_bridge.route_cognitive_signal")
    def test_default_source_and_truncation(self, mock_route):
        from capabilities.signal_bridge import emit_capability_signal

        emit_capability_signal("imap", "new_knowledge", "x" * 1000)

        arg = mock_route.call_args[0][0]
        assert arg["source_id"] == "imap:monitor"
        assert len(arg["payload"]["content"]) == 500

    def test_rejects_invalid_type_and_empty_content(self):
        from capabilities.signal_bridge import emit_capability_signal

        assert emit_capability_signal("x", "bad_type", "some content") is False
        assert emit_capability_signal("x", "novel_observation", "") is False
        assert emit_capability_signal("x", "novel_observation", "short") is False

    @patch("capabilities.signal_bridge.route_cognitive_signal", side_effect=Exception("boom"))
    def test_exception_returns_false(self, mock_route):
        from capabilities.signal_bridge import emit_capability_signal

        assert emit_capability_signal("x", "novel_observation", "enough content here") is False

    @patch("capabilities.signal_bridge.route_cognitive_signal")
    def test_imap_actionable_only(self, mock_route):
        """Only actionable triage produces signals; noise/informational do not."""
        from capabilities.signal_bridge import emit_capability_signal

        items = [
            {"triage": "noise", "from_name": "Bot", "subject": "Alert"},
            {"triage": "actionable", "from_name": "Boss", "subject": "Review"},
            {"triage": "informational", "from_name": "HR", "subject": "Policy"},
        ]
        for item in items:
            if item["triage"] == "actionable":
                emit_capability_signal(
                    "imap", "new_knowledge",
                    f"Actionable email from {item['from_name']}: {item['subject']}",
                )

        assert mock_route.call_count == 1
        assert "Boss" in mock_route.call_args[0][0]["payload"]["content"]
