import pytest
import json
from unittest.mock import MagicMock
from services.persistent_task_service import PersistentTaskService, VALID_TRANSITIONS
from tests.helpers import make_task_row

pytestmark = pytest.mark.unit


class TestStalledState:

    @pytest.fixture
    def mock_db(self):
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

    def test_stall_task_sets_status_and_records_reason(self, service, mock_db):
        _, cursor = mock_db
        row = make_task_row(task_id=1, status="in_progress",
            progress={"last_summary": "Working", "coverage_estimate": 0.6},
            iterations_used=20, max_iterations=20)
        cursor.fetchone.return_value = row
        result = service.stall_task(1, "Iteration budget exhausted")
        assert result is True
        sql = cursor.execute.call_args[0][0]
        assert "stalled" in sql
        params = cursor.execute.call_args[0][1]
        p = json.loads(params[0])
        assert p["stall_reason"] == "Iteration budget exhausted"
        assert p["last_summary"] == "Working"
        assert params[1] == 1

    def test_stalled_is_valid_from_in_progress(self):
        assert "stalled" in VALID_TRANSITIONS["in_progress"]

    def test_stalled_is_terminal(self):
        assert "stalled" not in VALID_TRANSITIONS

    def test_stalled_tasks_not_eligible(self, service, mock_db):
        _, cursor = mock_db
        cursor.fetchone.return_value = None
        service.get_eligible_task()
        sql = cursor.execute.call_args[0][0]
        assert "stalled" not in sql

    def test_checkpoint_auto_stalls_at_max(self, service, mock_db):
        _, cursor = mock_db
        row = make_task_row(task_id=1, status="in_progress",
            progress={"last_summary": "Done"},
            iterations_used=20, max_iterations=20)
        cursor.fetchone.return_value = row
        result = service.checkpoint(task_id=1, progress={"last_summary": "Final"})
        assert result is True
        calls = [str(c) for c in cursor.execute.call_args_list]
        assert any("stalled" in c for c in calls)
