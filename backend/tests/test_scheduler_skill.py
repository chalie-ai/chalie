"""
Tests for backend/services/innate_skills/scheduler_skill.py

Covers the _create() grace-window behaviour: past-due timestamps within
_PAST_DUE_GRACE_SECONDS are bumped forward to now+5s; timestamps older
than the grace window are still hard-rejected.

Result format (new): [schedule(action=create)]\\n<json_body>\\n[end:schedule]
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from services.innate_skills.scheduler_skill import handle_scheduler
from services.time_utils import utc_now
from tests._tag_helpers import assert_both_markers, extract_json

pytestmark = pytest.mark.unit


# ── Grace-bump behaviour ───────────────────────────────────────────────────────

class TestCreatePastDueGrace:
    """_create() should tolerate due_at that is slightly in the past."""

    def test_create_past_due_within_grace_bumps_forward(self, db):
        """due_at = now - 5s should be bumped to now+5s instead of rejected."""
        due_at = (utc_now() - timedelta(seconds=5)).isoformat()
        with patch("services.scheduler_service.embed_scheduled_item", return_value=None):
            raw = handle_scheduler("user", {
                "action": "create",
                "message": "Check the oven",
                "due_at": due_at,
            })

        assert_both_markers("schedule", raw)
        result = extract_json("schedule", raw)
        assert result["status"] == "success", f"Expected success, got: {result}"
        assert result["action_performed"] == "create"

        # The returned due_at should be close to now+5s (allow ±5s for test timing)
        returned_due_at = datetime.fromisoformat(result["record"]["due_at"])
        now_at_assert = utc_now()
        low = now_at_assert + timedelta(seconds=3)
        high = now_at_assert + timedelta(seconds=10)
        assert low <= returned_due_at <= high, (
            f"Bumped due_at {returned_due_at.isoformat()} not in expected range "
            f"[{low.isoformat()}, {high.isoformat()}]"
        )

    def test_create_past_due_at_grace_boundary_bumps_forward(self, db):
        """due_at = now - 119s is just inside the 120s grace and should succeed."""
        due_at = (utc_now() - timedelta(seconds=119)).isoformat()
        with patch("services.scheduler_service.embed_scheduled_item", return_value=None):
            raw = handle_scheduler("user", {
                "action": "create",
                "message": "Take medication",
                "due_at": due_at,
            })

        assert_both_markers("schedule", raw)
        result = extract_json("schedule", raw)
        assert result["status"] == "success", f"Expected success, got: {result}"
        assert result["action_performed"] == "create"

    def test_create_past_due_beyond_grace_rejects(self, db):
        """due_at = now - 121s exceeds the 120s grace and must be hard-rejected."""
        due_at = (utc_now() - timedelta(seconds=121)).isoformat()
        with patch("services.scheduler_service.embed_scheduled_item", return_value=None):
            raw = handle_scheduler("user", {
                "action": "create",
                "message": "Old reminder",
                "due_at": due_at,
            })

        assert_both_markers("schedule", raw)
        result = extract_json("schedule", raw)
        assert result["status"] == "error", f"Expected error, got: {result}"
        assert "due_at must be in the future" in result["error"]
