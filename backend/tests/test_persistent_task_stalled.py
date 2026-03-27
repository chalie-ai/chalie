"""
Tests for stalled state handling in PersistentTaskService.

Migrated from mock_db to real SQLite via the shared `db` fixture.
"""

import pytest
import json
from datetime import datetime, timezone, timedelta
from services.persistent_task_service import (
    PersistentTaskService,
    VALID_TRANSITIONS,
    DEFAULT_EXPIRY_DAYS,
)
from services.database_service import get_shared_db_service

pytestmark = pytest.mark.unit


def _seed_account(db):
    """Insert a master_account row so FK constraints are satisfied."""
    db.execute(
        "INSERT OR IGNORE INTO master_account (id, username, password_hash) "
        "VALUES (1, 'testuser', 'hash')"
    )
    db.commit()


def _insert_task(db, **overrides):
    """Insert a persistent_task directly and return its id."""
    defaults = dict(
        account_id=1,
        goal="Test background task",
        scope=None,
        status="accepted",
        priority=5,
        progress="{}",
        result=None,
        result_artifact=None,
        iterations_used=0,
        max_iterations=20,
        fatigue_budget=15.0,
    )
    defaults.update(overrides)

    if 'expires_at' not in defaults:
        defaults['expires_at'] = (
            datetime.now(timezone.utc) + timedelta(days=DEFAULT_EXPIRY_DAYS)
        ).isoformat()

    db.execute(
        """INSERT INTO persistent_tasks
           (account_id, goal, scope, status, priority, progress, result,
            result_artifact, iterations_used, max_iterations, fatigue_budget, expires_at)
           VALUES (:account_id, :goal, :scope, :status, :priority, :progress,
                   :result, :result_artifact, :iterations_used, :max_iterations,
                   :fatigue_budget, :expires_at)""",
        defaults,
    )
    db.commit()
    return db.execute("SELECT last_insert_rowid()").fetchone()[0]


class TestStalledState:

    @pytest.fixture(autouse=True)
    def _seed(self, db):
        _seed_account(db)

    @pytest.fixture
    def service(self, db):
        return PersistentTaskService(get_shared_db_service())

    def test_stall_task_sets_status_and_records_reason(self, db, service):
        task_id = _insert_task(
            db,
            status="in_progress",
            progress=json.dumps({"last_summary": "Working", "coverage_estimate": 0.6}),
            iterations_used=20,
            max_iterations=20,
        )

        result = service.stall_task(task_id, "Iteration budget exhausted")
        assert result is True

        row = db.execute(
            "SELECT status, progress FROM persistent_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert row['status'] == 'stalled'
        p = json.loads(row['progress'])
        assert p["stall_reason"] == "Iteration budget exhausted"
        assert p["last_summary"] == "Working"

    def test_stalled_is_valid_from_in_progress(self):
        assert "stalled" in VALID_TRANSITIONS["in_progress"]

    def test_stalled_is_terminal(self):
        assert "stalled" not in VALID_TRANSITIONS

    def test_stalled_tasks_not_eligible(self, db, service):
        """Stalled tasks should not be returned by get_eligible_task."""
        _insert_task(db, status="stalled")

        result = service.get_eligible_task()
        assert result is None

    def test_checkpoint_auto_stalls_at_max(self, db, service):
        """When iterations_used reaches max_iterations, checkpoint should auto-stall."""
        # Insert task at max_iterations - 1 so checkpoint pushes it to max
        task_id = _insert_task(
            db,
            status="in_progress",
            progress=json.dumps({"last_summary": "Done"}),
            iterations_used=19,
            max_iterations=20,
        )

        result = service.checkpoint(task_id=task_id, progress={"last_summary": "Final"})
        assert result is True

        row = db.execute(
            "SELECT status, progress FROM persistent_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert row['status'] == 'stalled'
        p = json.loads(row['progress'])
        assert p.get('stall_reason') == 'Iteration budget exhausted'
