"""
Unit tests for the Scheduler API blueprint (api/scheduler.py).

Uses the real `db` fixture (fully-migrated SQLite per test) instead of
mock_db.  Auth is still bypassed via a mock on validate_session.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from flask import Flask
from api.scheduler import scheduler_bp


def _future_iso(hours=1):
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _past_iso(hours=1):
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _insert_item(db, **overrides):
    """Insert a scheduled_items row and return the id.

    Provides sensible defaults for every column so callers only need to
    specify the fields they care about.
    """
    now = datetime.now(timezone.utc).isoformat()
    defaults = dict(
        id="abc12345",
        item_type="notification",
        message="Test reminder",
        due_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        recurrence=None,
        window_start=None,
        window_end=None,
        status="pending",
        channel="general",
        created_by_session=None,
        created_at=now,
        last_fired_at=None,
        group_id="abc12345",
        is_prompt=0,
    )
    defaults.update(overrides)
    d = defaults
    db.execute(
        """
        INSERT INTO scheduled_items
          (id, item_type, message, due_at, recurrence,
           window_start, window_end, status, channel,
           created_by_session, created_at, last_fired_at, group_id, is_prompt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            d["id"], d["item_type"], d["message"], d["due_at"],
            d["recurrence"], d["window_start"], d["window_end"],
            d["status"], d["channel"], d["created_by_session"],
            d["created_at"], d["last_fired_at"], d["group_id"],
            d["is_prompt"],
        ),
    )
    db.commit()
    return d["id"]


@pytest.mark.unit
class TestSchedulerAPI:
    """Tests for every scheduler blueprint route."""

    @pytest.fixture
    def client(self, db):
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

    @pytest.fixture(autouse=True)
    def suppress_embed(self):
        """Prevent the background embedding thread from running during tests."""
        with patch("services.scheduler_service.embed_scheduled_item"):
            yield

    # ----- GET /scheduler -----

    def test_list_returns_items_with_pagination(self, client, db):
        _insert_item(db, id="item1")
        _insert_item(db, id="item2", group_id="item2")

        resp = client.get("/scheduler")

        assert resp.status_code == 200
        body = resp.get_json()
        assert "items" in body
        assert body["total"] == 2
        assert "limit" in body
        assert "offset" in body
        assert len(body["items"]) == 2

    # ----- POST /scheduler -----

    def test_create_returns_201(self, client, db):
        resp = client.post(
            "/scheduler",
            json={"message": "Buy groceries", "due_at": _future_iso()},
        )

        assert resp.status_code == 201
        body = resp.get_json()
        assert body["item"]["message"] == "Buy groceries"
        assert body["item"]["status"] == "pending"

        # Verify the row actually exists in the database
        row = db.execute(
            "SELECT * FROM scheduled_items WHERE message = ?",
            ("Buy groceries",),
        ).fetchone()
        assert row is not None

    @pytest.mark.parametrize(
        "recurrence",
        ["daily", "hourly", "interval:60"],
    )
    def test_create_accepts_valid_recurrence(self, client, db, recurrence):
        payload = {
            "message": "Recurring item",
            "due_at": _future_iso(),
            "recurrence": recurrence,
        }
        if recurrence == "hourly":
            payload["window_start"] = "09:00"
            payload["window_end"] = "17:00"

        resp = client.post("/scheduler", json=payload)

        assert resp.status_code == 201, (
            f"Expected 201 for recurrence={recurrence}, got {resp.status_code}: "
            f"{resp.get_json()}"
        )

    def test_create_window_on_non_hourly_returns_400(self, client, db):
        resp = client.post(
            "/scheduler",
            json={
                "message": "Bad window",
                "due_at": _future_iso(),
                "recurrence": "daily",
                "window_start": "09:00",
                "window_end": "17:00",
            },
        )

        assert resp.status_code == 400
        assert "hourly" in resp.get_json()["error"]

    # ----- GET /scheduler/<id> -----

    def test_get_item_returns_item_when_found(self, client, db):
        _insert_item(db, id="xyz99999", group_id="xyz99999")

        resp = client.get("/scheduler/xyz99999")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["item"]["id"] == "xyz99999"
        assert body["item"]["status"] == "pending"

    # ----- PUT /scheduler/<id> -----

    def test_update_pending_item(self, client, db):
        _insert_item(db, id="upd00001", group_id="upd00001")

        resp = client.put(
            "/scheduler/upd00001",
            json={
                "message": "Updated message",
                "due_at": _future_iso(),
                "item_type": "prompt",
            },
        )

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["item"]["message"] == "Updated message"
        assert body["item"]["item_type"] == "prompt"

        # Verify actual DB state
        row = db.execute(
            "SELECT message, item_type FROM scheduled_items WHERE id = ?",
            ("upd00001",),
        ).fetchone()
        assert row["message"] == "Updated message"
        assert row["item_type"] == "prompt"

    def test_update_returns_404_for_non_pending(self, client, db):
        _insert_item(db, id="fired_item", status="fired", group_id="fired_item")

        resp = client.put(
            "/scheduler/fired_item",
            json={
                "message": "Try updating",
                "due_at": _future_iso(),
            },
        )

        assert resp.status_code == 404
        assert "not pending" in resp.get_json()["error"]

    # ----- DELETE /scheduler/<id> -----

    def test_cancel_pending_item(self, client, db):
        _insert_item(db, id="cancel01", group_id="cancel01")

        resp = client.delete("/scheduler/cancel01")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "cancelled"
        assert body["id"] == "cancel01"

        # Verify actual DB state
        row = db.execute(
            "SELECT status FROM scheduled_items WHERE id = ?",
            ("cancel01",),
        ).fetchone()
        assert row["status"] == "cancelled"

    # ----- DELETE /scheduler/history -----

    def test_prune_history_returns_deleted_count(self, client, db):
        # Insert 7 old, non-pending items that qualify for pruning.
        # The blueprint deletes WHERE status IN ('fired','failed','cancelled')
        # AND created_at < datetime('now', '-30 days')
        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        for i in range(7):
            _insert_item(
                db,
                id=f"old{i}",
                status="fired",
                created_at=old_date,
                group_id=f"old{i}",
            )

        resp = client.delete("/scheduler/history")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["deleted"] == 7

        # Verify all rows are gone
        count = db.execute("SELECT COUNT(*) FROM scheduled_items").fetchone()[0]
        assert count == 0
