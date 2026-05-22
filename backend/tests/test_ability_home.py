"""Unit tests for HomeCapability._tool_trigger_automation name-fallback.

Coverage:
- Direct entity_id (automation.*) passes through without a list call.
- Friendly-name exact match resolves to entity_id + adds resolved_from.
- Friendly-name case-insensitive substring match resolves to entity_id.
- No match returns an error dict; REST trigger is NOT called.
- Multiple matches return an error listing both names; REST trigger is NOT called.
- Empty automation_id returns the existing "automation_id is required" error.

All tests use @pytest.mark.unit and never touch a real network.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_AUTOMATIONS_PAYLOAD = {
    "automations": [
        {
            "entity_id": "automation.good_morning",
            "state": "on",
            "friendly_name": "Good Morning Routine",
            "last_triggered": None,
        }
    ],
    "count": 1,
}

_AUTOMATIONS_MULTI = {
    "automations": [
        {
            "entity_id": "automation.good_morning",
            "state": "on",
            "friendly_name": "Good Morning Routine",
            "last_triggered": None,
        },
        {
            "entity_id": "automation.good_morning_extra",
            "state": "on",
            "friendly_name": "Good Morning Extra",
            "last_triggered": None,
        },
    ],
    "count": 2,
}

_TRIGGER_OK = {"status": "ok", "automation_id": "automation.good_morning", "action": "triggered"}


def _make_connected_capability():
    """Return a HomeCapability instance wired as connected, with no real I/O."""
    from capabilities.home_capability.capability import HomeCapability

    cap = HomeCapability.__new__(HomeCapability)
    from capabilities.base import AbstractCapability
    AbstractCapability.__init__(cap)
    cap._manifest_cache = None
    cap._ws_handler = MagicMock()
    cap._url = "http://fake-ha.local:8123"
    cap._token = "fake-token"
    cap._verify_ssl = False
    cap._connected = True
    return cap


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestTriggerAutomationHappyPath:
    def test_direct_entity_id_skips_list_call(self):
        """automation.* prefix → trigger called directly without listing automations."""
        cap = _make_connected_capability()
        with (
            patch(
                "capabilities.home_capability.capability.rest.trigger_automation",
                return_value=_TRIGGER_OK,
            ) as mock_trigger,
            patch(
                "capabilities.home_capability.capability.rest.list_automations"
            ) as mock_list,
        ):
            result = cap._tool_trigger_automation(
                topic="", params={"automation_id": "automation.good_morning"}
            )

        mock_list.assert_not_called()
        mock_trigger.assert_called_once_with(
            cap._url, cap._token, cap._verify_ssl, "automation.good_morning"
        )
        assert result["status"] == "ok"
        assert "resolved_from" not in result

    def test_friendly_name_exact_match_resolves_and_adds_resolved_from(self):
        """Friendly-name exact match returns success with resolved_from."""
        cap = _make_connected_capability()
        with (
            patch(
                "capabilities.home_capability.capability.rest.list_automations",
                return_value=_AUTOMATIONS_PAYLOAD,
            ),
            patch(
                "capabilities.home_capability.capability.rest.trigger_automation",
                return_value=_TRIGGER_OK,
            ) as mock_trigger,
        ):
            result = cap._tool_trigger_automation(
                topic="", params={"automation_id": "Good Morning Routine"}
            )

        mock_trigger.assert_called_once_with(
            cap._url, cap._token, cap._verify_ssl, "automation.good_morning"
        )
        assert result["status"] == "ok"
        assert result["resolved_from"] == "Good Morning Routine"

    def test_friendly_name_substring_case_insensitive(self):
        """Lowercase substring match resolves to entity_id; resolved_from is present."""
        cap = _make_connected_capability()
        with (
            patch(
                "capabilities.home_capability.capability.rest.list_automations",
                return_value=_AUTOMATIONS_PAYLOAD,
            ),
            patch(
                "capabilities.home_capability.capability.rest.trigger_automation",
                return_value=_TRIGGER_OK,
            ) as mock_trigger,
        ):
            result = cap._tool_trigger_automation(
                topic="", params={"automation_id": "good morning"}
            )

        mock_trigger.assert_called_once_with(
            cap._url, cap._token, cap._verify_ssl, "automation.good_morning"
        )
        assert result["status"] == "ok"
        assert result.get("resolved_from") == "good morning"


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


class TestTriggerAutomationErrorPath:
    def test_no_match_returns_error_and_trigger_not_called(self):
        """Unknown name returns error dict; REST trigger must NOT be called."""
        cap = _make_connected_capability()
        with (
            patch(
                "capabilities.home_capability.capability.rest.list_automations",
                return_value={"automations": [], "count": 0},
            ),
            patch(
                "capabilities.home_capability.capability.rest.trigger_automation"
            ) as mock_trigger,
        ):
            result = cap._tool_trigger_automation(
                topic="", params={"automation_id": "nonexistent"}
            )

        mock_trigger.assert_not_called()
        assert "error" in result
        assert "nonexistent" in result["error"]

    def test_multi_match_returns_error_with_names_and_trigger_not_called(self):
        """Multiple matches return error listing both names; trigger NOT called."""
        cap = _make_connected_capability()
        with (
            patch(
                "capabilities.home_capability.capability.rest.list_automations",
                return_value=_AUTOMATIONS_MULTI,
            ),
            patch(
                "capabilities.home_capability.capability.rest.trigger_automation"
            ) as mock_trigger,
        ):
            result = cap._tool_trigger_automation(
                topic="", params={"automation_id": "good morning"}
            )

        mock_trigger.assert_not_called()
        assert "error" in result
        assert "Good Morning Routine" in result["error"]
        assert "Good Morning Extra" in result["error"]

    def test_empty_automation_id_returns_required_error(self):
        """Empty string automation_id returns the existing required-field error."""
        cap = _make_connected_capability()
        with patch(
            "capabilities.home_capability.capability.rest.trigger_automation"
        ) as mock_trigger:
            result = cap._tool_trigger_automation(
                topic="", params={"automation_id": ""}
            )

        mock_trigger.assert_not_called()
        assert result == {"error": "automation_id is required"}
