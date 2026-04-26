"""
Tests for backend/services/innate_skills/scheduler_skill.py

Covers the _create() grace-window behaviour: past-due timestamps within
_PAST_DUE_GRACE_SECONDS are bumped forward to now+5s; timestamps older
than the grace window are still hard-rejected.

Also covers the atomic dedup guarantee: back-to-back _create() calls with
the same message must result in exactly one row in scheduled_items.
"""

import json
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from unittest.mock import patch

from services.innate_skills.scheduler_skill import handle_scheduler
from services.time_utils import utc_now

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

        result = json.loads(raw)
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

        result = json.loads(raw)
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

        result = json.loads(raw)
        assert result["status"] == "error", f"Expected error, got: {result}"
        assert "due_at must be in the future" in result["error"]


# ── Atomic dedup guarantee ────────────────────────────────────────────────────

class TestCreateDedup:
    """Rapid back-to-back _create() calls with identical message → one row only."""

    def test_sequential_same_message_produces_one_row(self, db):
        """Two sequential calls with identical message produce exactly one DB row.

        Both calls must return status=success.  The second call returns
        note='already_existed' and the same record id as the first.
        """
        due_at = (utc_now() + timedelta(hours=1)).isoformat()
        params = {
            "action": "create",
            "message": "Call the dentist",
            "due_at": due_at,
        }

        with patch("services.scheduler_service.embed_scheduled_item", return_value=None):
            raw1 = handle_scheduler("user", params)
            raw2 = handle_scheduler("user", params)

        r1 = json.loads(raw1)
        r2 = json.loads(raw2)

        assert r1["status"] == "success", f"First call failed: {r1}"
        assert r2["status"] == "success", f"Second call failed: {r2}"

        # Second call must be identified as a dedup hit
        assert r2.get("note") == "already_existed", (
            f"Expected note='already_existed' on second call, got: {r2}"
        )

        # Both calls must return the same record id
        assert r1["record"]["id"] == r2["record"]["id"], (
            f"Different ids returned: {r1['record']['id']} vs {r2['record']['id']}"
        )

        # Exactly one row must exist in the database
        row_count = db.execute(
            "SELECT COUNT(*) FROM scheduled_items WHERE message = 'Call the dentist' AND status = 'pending'"
        ).fetchone()[0]
        assert row_count == 1, f"Expected 1 row, found {row_count}"

    def test_concurrent_same_message_produces_one_row(self, db):
        """Two threads calling _create() simultaneously produce exactly one DB row.

        Uses a ThreadPoolExecutor to maximise the chance of hitting the race
        window.  Both futures must succeed; exactly one must have no 'note' key
        (the winner) and exactly one must carry note='already_existed' (the loser).
        """
        due_at = (utc_now() + timedelta(hours=2)).isoformat()
        params = {
            "action": "create",
            "message": "Call the dentist concurrent",
            "due_at": due_at,
        }

        results = []
        with patch("services.scheduler_service.embed_scheduled_item", return_value=None):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(handle_scheduler, "user", params) for _ in range(2)]
                for f in as_completed(futures):
                    results.append(json.loads(f.result()))

        assert all(r["status"] == "success" for r in results), (
            f"One or more calls failed: {results}"
        )

        dedup_hits = [r for r in results if r.get("note") == "already_existed"]
        assert len(dedup_hits) >= 1, (
            "Expected at least one dedup hit across concurrent calls, "
            f"but got: {[r.get('note') for r in results]}"
        )

        # Exactly one row must exist in the database
        row_count = db.execute(
            "SELECT COUNT(*) FROM scheduled_items "
            "WHERE message = 'Call the dentist concurrent' AND status = 'pending'"
        ).fetchone()[0]
        assert row_count == 1, f"Expected 1 row, found {row_count}"
