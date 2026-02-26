"""
Tests for backend/services/persistent_task_service.py

Covers: state machine transitions, CRUD operations, Jaccard duplicate detection,
rate limiting, auto-expiry, and checkpoint/completion logic.
"""

import pytest
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, call

from services.persistent_task_service import (
    PersistentTaskService,
    _jaccard_similarity,
    VALID_TRANSITIONS,
    MAX_ACTIVE_TASKS,
    MAX_CYCLES_PER_HOUR,
    DEFAULT_EXPIRY_DAYS,
    DUPLICATE_SIMILARITY_THRESHOLD,
)
from tests.helpers import make_task_row


@pytest.mark.unit
class TestPersistentTaskService:

    @pytest.fixture
    def mock_db(self):
        """Provides (db, cursor) pair wired for db.connection() context manager."""
        db = MagicMock()
        cursor = MagicMock()
        conn = MagicMock()
        conn.cursor.return_value = cursor
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=conn)
        ctx.__exit__ = MagicMock(return_value=False)
        db.connection.return_value = ctx
        return db, cursor

    @pytest.fixture
    def service(self, mock_db):
        db, _ = mock_db
        return PersistentTaskService(db)

    # ── State Machine Transitions ────────────────────────────────────

    def test_transition_proposed_to_accepted_succeeds(self, service, mock_db):
        """Valid transition proposed -> accepted should return (True, message)."""
        _, cursor = mock_db
        row = make_task_row(task_id=1, status='proposed')
        cursor.fetchone.return_value = row

        success, msg = service.transition(1, 'accepted')

        assert success is True
        assert 'accepted' in msg

    def test_transition_proposed_to_completed_fails(self, service, mock_db):
        """Invalid transition proposed -> completed should return (False, message)."""
        _, cursor = mock_db
        row = make_task_row(task_id=1, status='proposed')
        cursor.fetchone.return_value = row

        success, msg = service.transition(1, 'completed')

        assert success is False
        assert "Cannot transition" in msg

    def test_transition_proposed_to_in_progress_fails(self, service, mock_db):
        """Invalid transition proposed -> in_progress should return (False, message)."""
        _, cursor = mock_db
        row = make_task_row(task_id=1, status='proposed')
        cursor.fetchone.return_value = row

        success, msg = service.transition(1, 'in_progress')

        assert success is False
        assert "Cannot transition" in msg

    # ── accept_task ──────────────────────────────────────────────────

    def test_accept_task_enforces_max_active_limit(self, service, mock_db):
        """accept_task should reject when MAX_ACTIVE_TASKS active tasks exist."""
        _, cursor = mock_db

        # First call: get_task for the proposed task
        proposed_row = make_task_row(task_id=10, account_id=1, status='proposed')
        # Second+ calls: get_active_tasks returns 5 accepted tasks
        active_rows = [
            make_task_row(task_id=i, account_id=1, status='accepted')
            for i in range(1, 6)
        ]

        # get_task queries once (fetchone), get_active_tasks queries once (fetchall)
        cursor.fetchone.return_value = proposed_row
        cursor.fetchall.return_value = active_rows

        success, msg = service.accept_task(10)

        assert success is False
        assert str(MAX_ACTIVE_TASKS) in msg

    def test_accept_task_succeeds_under_limit(self, service, mock_db):
        """accept_task should succeed when fewer than MAX_ACTIVE_TASKS are active."""
        _, cursor = mock_db

        proposed_row = make_task_row(task_id=10, account_id=1, status='proposed')
        # Only 3 accepted tasks -- under the limit of 5
        active_rows = [
            make_task_row(task_id=i, account_id=1, status='accepted')
            for i in range(1, 4)
        ]

        # get_task (for accept_task) returns proposed, get_active_tasks returns 3 tasks,
        # then get_task (inside transition) returns proposed again for validation
        cursor.fetchone.return_value = proposed_row
        cursor.fetchall.return_value = active_rows

        success, msg = service.accept_task(10)

        assert success is True
        assert 'accepted' in msg

    # ── Checkpoint ───────────────────────────────────────────────────

    def test_checkpoint_increments_iterations(self, service, mock_db):
        """checkpoint should execute UPDATE with iterations_used + 1."""
        _, cursor = mock_db

        progress = {'last_summary': 'Step 1 done', 'coverage_estimate': 0.3}
        result = service.checkpoint(task_id=1, progress=progress)

        assert result is True
        # Verify the SQL update was executed
        executed_sql = cursor.execute.call_args[0][0]
        assert 'iterations_used = iterations_used + 1' in executed_sql
        assert 'progress' in executed_sql

    # ── Jaccard Similarity / Duplicate Detection ─────────────────────

    def test_jaccard_exact_match_returns_one(self):
        """Identical strings should yield Jaccard similarity of 1.0."""
        assert _jaccard_similarity("buy groceries today", "buy groceries today") == 1.0

    def test_jaccard_disjoint_returns_zero(self):
        """Completely disjoint word sets should yield 0.0."""
        assert _jaccard_similarity("alpha beta gamma", "delta epsilon zeta") == 0.0

    def test_find_duplicate_above_threshold(self, service, mock_db):
        """find_duplicate should return matching task when similarity exceeds threshold."""
        _, cursor = mock_db

        # get_active_tasks returns one task with a very similar goal
        existing_row = make_task_row(
            task_id=5,
            account_id=1,
            status='in_progress',
            goal="research best Python testing frameworks",
        )
        cursor.fetchall.return_value = [existing_row]
        # get_active_tasks does not call fetchone, only fetchall
        cursor.fetchone.return_value = None

        result = service.find_duplicate(
            account_id=1,
            goal="research best Python testing frameworks please",
        )

        assert result is not None
        assert result['id'] == 5
        # Verify the similarity is actually above threshold
        sim = _jaccard_similarity(
            "research best Python testing frameworks",
            "research best Python testing frameworks please",
        )
        assert sim > DUPLICATE_SIMILARITY_THRESHOLD

    def test_find_duplicate_below_threshold_returns_none(self, service, mock_db):
        """find_duplicate should return None when no task exceeds similarity threshold."""
        _, cursor = mock_db

        existing_row = make_task_row(
            task_id=5,
            account_id=1,
            status='in_progress',
            goal="research Python testing",
        )
        cursor.fetchall.return_value = [existing_row]
        cursor.fetchone.return_value = None

        result = service.find_duplicate(
            account_id=1,
            goal="buy groceries for dinner tonight",
        )

        assert result is None
        # Verify the similarity is actually below threshold
        sim = _jaccard_similarity(
            "research Python testing",
            "buy groceries for dinner tonight",
        )
        assert sim <= DUPLICATE_SIMILARITY_THRESHOLD

    # ── Rate Limiting ────────────────────────────────────────────────

    def test_check_rate_limit_blocks_at_max_cycles(self, service, mock_db):
        """check_rate_limit should return False when cycles_this_hour >= MAX_CYCLES_PER_HOUR."""
        _, cursor = mock_db

        recent_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        progress = {
            'cycles_this_hour': MAX_CYCLES_PER_HOUR,
            'last_cycle_at': recent_time,
        }
        row = make_task_row(task_id=1, status='in_progress', progress=progress)
        cursor.fetchone.return_value = row

        result = service.check_rate_limit(task_id=1)

        assert result is False

    def test_check_rate_limit_allows_under_limit(self, service, mock_db):
        """check_rate_limit should return True when cycles_this_hour < MAX_CYCLES_PER_HOUR."""
        _, cursor = mock_db

        recent_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        progress = {
            'cycles_this_hour': 1,
            'last_cycle_at': recent_time,
        }
        row = make_task_row(task_id=1, status='in_progress', progress=progress)
        cursor.fetchone.return_value = row

        result = service.check_rate_limit(task_id=1)

        assert result is True

    def test_check_rate_limit_resets_after_one_hour(self, service, mock_db):
        """check_rate_limit should allow execution when last_cycle_at is over 1 hour ago."""
        _, cursor = mock_db

        old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        progress = {
            'cycles_this_hour': MAX_CYCLES_PER_HOUR + 5,  # would block if not reset
            'last_cycle_at': old_time,
        }
        row = make_task_row(task_id=1, status='in_progress', progress=progress)
        cursor.fetchone.return_value = row

        result = service.check_rate_limit(task_id=1)

        assert result is True

    # ── CRUD Operations ──────────────────────────────────────────────

    def test_create_task_returns_dict_with_proposed_status(self, service, mock_db):
        """create_task should INSERT and return a task dict with 'proposed' status."""
        _, cursor = mock_db

        now = datetime.now(timezone.utc)
        # First fetchone: the INSERT...RETURNING call
        insert_row = (42, now)
        # Second fetchone: the get_task call after insert
        full_row = make_task_row(task_id=42, account_id=1, goal="Learn Rust", status='proposed')

        cursor.fetchone.side_effect = [insert_row, full_row]

        result = service.create_task(account_id=1, goal="Learn Rust")

        assert result is not None
        assert result['id'] == 42
        assert result['status'] == 'proposed'
        assert result['goal'] == "Learn Rust"

    def test_get_task_returns_none_when_not_found(self, service, mock_db):
        """get_task should return None when the task does not exist."""
        _, cursor = mock_db
        cursor.fetchone.return_value = None

        result = service.get_task(999)

        assert result is None

    def test_get_task_returns_dict_when_found(self, service, mock_db):
        """get_task should return a fully populated task dict when found."""
        _, cursor = mock_db
        row = make_task_row(
            task_id=7,
            account_id=2,
            goal="Write documentation",
            status='in_progress',
            priority=3,
            iterations_used=5,
            max_iterations=20,
        )
        cursor.fetchone.return_value = row

        result = service.get_task(7)

        assert result is not None
        assert result['id'] == 7
        assert result['account_id'] == 2
        assert result['goal'] == "Write documentation"
        assert result['status'] == 'in_progress'
        assert result['priority'] == 3
        assert result['iterations_used'] == 5
        assert result['max_iterations'] == 20

    # ── Auto-Expiry ──────────────────────────────────────────────────

    def test_expire_stale_tasks_returns_count(self, service, mock_db):
        """expire_stale_tasks should return the number of tasks that were expired."""
        _, cursor = mock_db
        # Simulate 3 tasks being expired (RETURNING id gives 3 rows)
        cursor.fetchall.return_value = [(10,), (11,), (12,)]

        count = service.expire_stale_tasks()

        assert count == 3
        executed_sql = cursor.execute.call_args[0][0]
        assert "status = 'expired'" in executed_sql
        assert 'RETURNING id' in executed_sql

    # ── Completion ───────────────────────────────────────────────────

    def test_complete_task_sets_status_and_result(self, service, mock_db):
        """complete_task should UPDATE status to completed and store the result."""
        _, cursor = mock_db

        result = service.complete_task(
            task_id=1,
            result="Successfully gathered all data",
            artifact={"summary": "3 sources found"},
        )

        assert result is True
        executed_sql = cursor.execute.call_args[0][0]
        assert "status = 'completed'" in executed_sql
        params = cursor.execute.call_args[0][1]
        assert params[0] == "Successfully gathered all data"
        assert json.loads(params[1]) == {"summary": "3 sources found"}
        assert params[2] == 1
