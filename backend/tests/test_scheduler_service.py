"""
Tests for backend/services/scheduler_service.py
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sqlite3
import services.scheduler_service as scheduler_svc


@pytest.mark.unit
class TestSchedulerService:
    """Test scheduler service background worker."""

    @pytest.fixture(autouse=True)
    def setup_temp_db(self, tmp_path, monkeypatch):
        """Setup temporary database for each test."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = data_dir / "scheduler.db"
        monkeypatch.setattr(scheduler_svc, '_DATA_DIR', data_dir)
        monkeypatch.setattr(scheduler_svc, '_DB_PATH', db_path)
        yield db_path

    def _create_item(self, db_path, item_id="test1", due_at=None, recurrence=None):
        """Helper to create a test item."""
        if due_at is None:
            due_at = (datetime.now() - timedelta(minutes=1)).isoformat()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scheduled_items (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL DEFAULT 'reminder',
                message TEXT NOT NULL,
                due_at TEXT NOT NULL,
                recurrence TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_fired_at TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status_due ON scheduled_items(status, due_at)
        """)
        cursor.execute("""
            INSERT INTO scheduled_items (id, type, message, due_at, recurrence, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (item_id, "reminder", "Test message", due_at, recurrence, "pending"))
        conn.commit()
        conn.close()

    def test_poll_finds_due_items(self, setup_temp_db):
        """Poll should find due items."""
        self._create_item(setup_temp_db, due_at=(datetime.now() - timedelta(minutes=5)).isoformat())

        with patch('services.scheduler_service.PromptQueue') as mock_queue:
            scheduler_svc._poll_and_fire()

            # Queue should have been created
            assert mock_queue.called

    def test_poll_fires_item_to_queue(self, setup_temp_db):
        """Poll should enqueue due items."""
        self._create_item(setup_temp_db, due_at=(datetime.now() - timedelta(minutes=5)).isoformat())

        with patch('services.scheduler_service.PromptQueue') as mock_queue_class:
            mock_queue = MagicMock()
            mock_queue_class.return_value = mock_queue

            scheduler_svc._poll_and_fire()

            # Enqueue should have been called
            assert mock_queue.enqueue.called

    def test_poll_marks_fired(self, setup_temp_db):
        """Fired items should be marked with status='fired'."""
        self._create_item(setup_temp_db, item_id="fired1", due_at=(datetime.now() - timedelta(minutes=5)).isoformat())

        with patch('services.scheduler_service.PromptQueue'):
            scheduler_svc._poll_and_fire()

        # Check status in DB
        conn = sqlite3.connect(str(setup_temp_db))
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM scheduled_items WHERE id = ?", ("fired1",))
        row = cursor.fetchone()
        conn.close()

        assert row[0] == "fired"

    def test_recurrence_daily(self, setup_temp_db):
        """Daily recurrence should add 1 day."""
        due_at = datetime(2024, 1, 15, 10, 0)
        next_due = scheduler_svc._calculate_next_due(due_at, "daily")
        assert next_due == datetime(2024, 1, 16, 10, 0)

    def test_recurrence_weekly(self, setup_temp_db):
        """Weekly recurrence should add 7 days."""
        due_at = datetime(2024, 1, 15, 10, 0)
        next_due = scheduler_svc._calculate_next_due(due_at, "weekly")
        assert next_due == datetime(2024, 1, 22, 10, 0)

    def test_recurrence_monthly(self, setup_temp_db):
        """Monthly recurrence should add 1 month."""
        due_at = datetime(2024, 1, 15, 10, 0)
        next_due = scheduler_svc._calculate_next_due(due_at, "monthly")
        assert next_due == datetime(2024, 2, 15, 10, 0)

    def test_recurrence_monthly_clamping(self, setup_temp_db):
        """Jan 31 → Feb should clamp to Feb 28."""
        due_at = datetime(2024, 1, 31, 10, 0)
        next_due = scheduler_svc._calculate_next_due(due_at, "monthly")
        assert next_due.month == 2
        assert next_due.day == 29  # 2024 is leap year

    def test_recurrence_weekdays(self, setup_temp_db):
        """Weekdays should skip Saturday/Sunday."""
        # Friday
        friday = datetime(2024, 1, 19, 10, 0)
        next_due = scheduler_svc._calculate_next_due(friday, "weekdays")
        # Should be Monday (skip Sat/Sun)
        assert next_due.weekday() < 5

    def test_no_recurrence_one_time(self, setup_temp_db):
        """No recurrence should not create new item."""
        next_item = scheduler_svc._create_recurrence(
            {"id": "test", "due_at": "2024-01-15T10:00:00", "recurrence": None},
            datetime.now()
        )
        assert next_item is None

    def test_fire_item_metadata(self, setup_temp_db):
        """Fired item should have correct metadata."""
        item = {
            "id": "test",
            "type": "reminder",
            "message": "Test message",
            "due_at": "2024-01-15T10:00:00"
        }

        with patch('services.scheduler_service.PromptQueue') as mock_queue_class:
            mock_queue = MagicMock()
            mock_queue_class.return_value = mock_queue

            scheduler_svc._fire_item(item, datetime.now())

            # Check enqueue call
            call_args = mock_queue.enqueue.call_args[0][0]
            assert call_args['metadata']['source'] == "reminder"
            assert call_args['metadata']['destination'] == "web"

    def test_poll_empty_table(self, setup_temp_db):
        """Empty table should not error."""
        # Create empty DB
        conn = sqlite3.connect(str(setup_temp_db))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scheduled_items (
                id TEXT PRIMARY KEY,
                type TEXT,
                message TEXT,
                due_at TEXT,
                recurrence TEXT,
                status TEXT,
                created_at TEXT,
                last_fired_at TEXT
            )
        """)
        conn.commit()
        conn.close()

        # Should not error
        with patch('services.scheduler_service.PromptQueue'):
            scheduler_svc._poll_and_fire()
