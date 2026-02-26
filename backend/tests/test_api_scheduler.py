"""
Unit tests for the Scheduler API blueprint (api/scheduler.py).

All tests mock the database connection and auth session — no external
dependencies required.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from flask import Flask
from api.scheduler import scheduler_bp


# Column order returned by every SELECT in the blueprint
_COLS = [
    "id", "item_type", "message", "due_at", "recurrence",
    "window_start", "window_end", "status", "topic",
    "created_by_session", "created_at", "last_fired_at", "group_id", "is_prompt",
]


def _make_row(
    item_id="abc12345",
    item_type="notification",
    message="Test reminder",
    due_at=None,
    recurrence=None,
    window_start=None,
    window_end=None,
    status="pending",
    topic="general",
    created_by_session=None,
    created_at=None,
    last_fired_at=None,
    group_id="abc12345",
    is_prompt=False,
):
    """Build a 14-element tuple matching the column order the blueprint expects."""
    now = datetime.now(timezone.utc)
    return (
        item_id,
        item_type,
        message,
        due_at or now + timedelta(hours=1),
        recurrence,
        window_start,
        window_end,
        status,
        topic,
        created_by_session,
        created_at or now,
        last_fired_at,
        group_id,
        is_prompt,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def _mock_db():
    """Return (mock_db_service, mock_connection, mock_cursor)."""
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)

    db = MagicMock()
    db.connection.return_value = ctx
    return db, conn, cursor


@pytest.mark.unit
class TestSchedulerAPI:
    """Tests for every scheduler blueprint route."""

    @pytest.fixture
    def client(self):
        app = Flask(__name__)
        app.register_blueprint(scheduler_bp)
        app.config["TESTING"] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self):
        with patch(
            "services.auth_session_service.validate_session", return_value=True
        ):
            yield

    # ----- helpers -----

    def _future_dt(self, hours=1):
        return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()

    def _past_dt(self, hours=1):
        return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

    # ----- GET /scheduler -----

    def test_list_returns_items_with_pagination(self, client, _mock_db):
        mock_db, _, cursor = _mock_db
        cursor.fetchone.return_value = (2,)  # total count
        cursor.fetchall.return_value = [_make_row(), _make_row(item_id="def67890")]

        with patch(
            "services.database_service.get_shared_db_service", return_value=mock_db
        ):
            resp = client.get("/scheduler")

        assert resp.status_code == 200
        body = resp.get_json()
        assert "items" in body
        assert body["total"] == 2
        assert "limit" in body
        assert "offset" in body
        assert len(body["items"]) == 2

    def test_list_filters_by_pending_status(self, client, _mock_db):
        mock_db, _, cursor = _mock_db
        cursor.fetchone.return_value = (1,)
        cursor.fetchall.return_value = [_make_row(status="pending")]

        with patch(
            "services.database_service.get_shared_db_service", return_value=mock_db
        ):
            resp = client.get("/scheduler?status=pending")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["total"] == 1
        # Verify the WHERE clause was parameterised with the status value
        execute_calls = cursor.execute.call_args_list
        count_sql = execute_calls[0][0][0]
        assert "WHERE status" in count_sql

    def test_list_rejects_invalid_status(self, client, _mock_db):
        mock_db, _, cursor = _mock_db

        with patch(
            "services.database_service.get_shared_db_service", return_value=mock_db
        ):
            resp = client.get("/scheduler?status=bogus")

        assert resp.status_code == 400
        assert "status" in resp.get_json()["error"]

    # ----- POST /scheduler -----

    def test_create_returns_201(self, client, _mock_db):
        mock_db, _, cursor = _mock_db
        cursor.fetchone.return_value = _make_row(message="Buy groceries")

        with patch(
            "services.database_service.get_shared_db_service", return_value=mock_db
        ):
            resp = client.post(
                "/scheduler",
                json={"message": "Buy groceries", "due_at": self._future_dt()},
            )

        assert resp.status_code == 201
        body = resp.get_json()
        assert body["item"]["message"] == "Buy groceries"
        assert body["item"]["status"] == "pending"

    def test_create_without_message_returns_400(self, client, _mock_db):
        mock_db, _, cursor = _mock_db

        with patch(
            "services.database_service.get_shared_db_service", return_value=mock_db
        ):
            resp = client.post(
                "/scheduler",
                json={"due_at": self._future_dt()},
            )

        assert resp.status_code == 400
        assert "message" in resp.get_json()["error"]

    def test_create_with_past_due_at_returns_400(self, client, _mock_db):
        mock_db, _, cursor = _mock_db

        with patch(
            "services.database_service.get_shared_db_service", return_value=mock_db
        ):
            resp = client.post(
                "/scheduler",
                json={"message": "Late reminder", "due_at": self._past_dt()},
            )

        assert resp.status_code == 400
        assert "future" in resp.get_json()["error"]

    @pytest.mark.parametrize(
        "recurrence",
        ["daily", "weekly", "monthly", "weekdays", "hourly", "interval:60"],
    )
    def test_create_accepts_valid_recurrence(self, client, _mock_db, recurrence):
        mock_db, _, cursor = _mock_db
        cursor.fetchone.return_value = _make_row(recurrence=recurrence)

        payload = {
            "message": "Recurring item",
            "due_at": self._future_dt(),
            "recurrence": recurrence,
        }
        # hourly recurrence may include windows; add them for that case
        if recurrence == "hourly":
            payload["window_start"] = "09:00"
            payload["window_end"] = "17:00"

        with patch(
            "services.database_service.get_shared_db_service", return_value=mock_db
        ):
            resp = client.post("/scheduler", json=payload)

        assert resp.status_code == 201, (
            f"Expected 201 for recurrence={recurrence}, got {resp.status_code}: "
            f"{resp.get_json()}"
        )

    def test_create_window_on_non_hourly_returns_400(self, client, _mock_db):
        mock_db, _, cursor = _mock_db

        with patch(
            "services.database_service.get_shared_db_service", return_value=mock_db
        ):
            resp = client.post(
                "/scheduler",
                json={
                    "message": "Bad window",
                    "due_at": self._future_dt(),
                    "recurrence": "daily",
                    "window_start": "09:00",
                    "window_end": "17:00",
                },
            )

        assert resp.status_code == 400
        assert "hourly" in resp.get_json()["error"]

    def test_create_window_start_without_end_returns_400(self, client, _mock_db):
        mock_db, _, cursor = _mock_db

        with patch(
            "services.database_service.get_shared_db_service", return_value=mock_db
        ):
            resp = client.post(
                "/scheduler",
                json={
                    "message": "Missing end",
                    "due_at": self._future_dt(),
                    "recurrence": "hourly",
                    "window_start": "09:00",
                },
            )

        assert resp.status_code == 400
        assert "window_end" in resp.get_json()["error"]

    # ----- GET /scheduler/<id> -----

    def test_get_item_returns_404_when_not_found(self, client, _mock_db):
        mock_db, _, cursor = _mock_db
        cursor.fetchone.return_value = None

        with patch(
            "services.database_service.get_shared_db_service", return_value=mock_db
        ):
            resp = client.get("/scheduler/nonexistent")

        assert resp.status_code == 404

    def test_get_item_returns_item_when_found(self, client, _mock_db):
        mock_db, _, cursor = _mock_db
        cursor.fetchone.return_value = _make_row(item_id="xyz99999")

        with patch(
            "services.database_service.get_shared_db_service", return_value=mock_db
        ):
            resp = client.get("/scheduler/xyz99999")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["item"]["id"] == "xyz99999"
        assert body["item"]["status"] == "pending"

    # ----- PUT /scheduler/<id> -----

    def test_update_pending_item(self, client, _mock_db):
        mock_db, _, cursor = _mock_db
        updated_row = _make_row(
            item_id="upd00001",
            message="Updated message",
            item_type="prompt",
            is_prompt=True,
        )
        cursor.fetchone.return_value = updated_row

        with patch(
            "services.database_service.get_shared_db_service", return_value=mock_db
        ):
            resp = client.put(
                "/scheduler/upd00001",
                json={
                    "message": "Updated message",
                    "due_at": self._future_dt(),
                    "item_type": "prompt",
                },
            )

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["item"]["message"] == "Updated message"
        assert body["item"]["item_type"] == "prompt"

    def test_update_returns_404_for_non_pending(self, client, _mock_db):
        mock_db, _, cursor = _mock_db
        # RETURNING yields no row when item is not pending
        cursor.fetchone.return_value = None

        with patch(
            "services.database_service.get_shared_db_service", return_value=mock_db
        ):
            resp = client.put(
                "/scheduler/fired_item",
                json={
                    "message": "Try updating",
                    "due_at": self._future_dt(),
                },
            )

        assert resp.status_code == 404
        assert "not pending" in resp.get_json()["error"]

    # ----- DELETE /scheduler/<id> -----

    def test_cancel_pending_item(self, client, _mock_db):
        mock_db, _, cursor = _mock_db
        cursor.rowcount = 1

        with patch(
            "services.database_service.get_shared_db_service", return_value=mock_db
        ):
            resp = client.delete("/scheduler/cancel01")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "cancelled"
        assert body["id"] == "cancel01"

    def test_cancel_returns_404_for_non_pending(self, client, _mock_db):
        mock_db, _, cursor = _mock_db
        cursor.rowcount = 0

        with patch(
            "services.database_service.get_shared_db_service", return_value=mock_db
        ):
            resp = client.delete("/scheduler/already_fired")

        assert resp.status_code == 404
        assert "not pending" in resp.get_json()["error"]

    # ----- DELETE /scheduler/history -----

    def test_prune_history_returns_deleted_count(self, client, _mock_db):
        mock_db, _, cursor = _mock_db
        cursor.rowcount = 7

        with patch(
            "services.database_service.get_shared_db_service", return_value=mock_db
        ):
            resp = client.delete("/scheduler/history")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["deleted"] == 7
