import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone, timedelta
from typing import cast
from unittest.mock import patch

import pytest
from flask.testing import FlaskClient

from api.scheduler import scheduler_ns
from tests.restx_test_app import mount_namespace


def _future_local_iso(hours: int = 1) -> str:
    """A naive local wall-clock ISO string — the wire shape ``start_at`` takes
    (see ``api/dto/scheduler_item.py``: parsed via ``parse_local``, never a UTC
    offset the caller computes itself)."""
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).replace(tzinfo=None).isoformat()


def _insert_item(db: sqlite3.Connection, **overrides: object) -> str:
    now = datetime.now(timezone.utc).isoformat()
    defaults: dict[str, object] = dict(
        id="abc12345",
        item_type="prompt",
        message="Test reminder",
        start_at=now,
        due_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        cron_dom=None,
        cron_hour=None,
        cron_minute=None,
        status="pending",
        channel="general",
        created_by_session=None,
        created_at=now,
        enabled=1,
        group_id="abc12345",
    )
    defaults.update(overrides)
    d = defaults
    db.execute(
        """
        INSERT INTO scheduled_items
          (id, item_type, message, start_at, due_at, cron_dom, cron_hour, cron_minute,
           status, channel,
           created_by_session, created_at, enabled, group_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            d["id"], d["item_type"], d["message"], d["start_at"], d["due_at"],
            d["cron_dom"], d["cron_hour"], d["cron_minute"],
            d["status"], d["channel"], d["created_by_session"],
            d["created_at"], d["enabled"], d["group_id"],
        ),
    )
    db.commit()
    return cast(str, d["id"])


@pytest.mark.unit
class TestSchedulerAPI:
    """Tests for every scheduler namespace route, against the cron recurrence
    model (TKT-1434): the wire body carries ``day``/``hour``/``minute``
    (nullable ints, NULL = "every") + optional local ``start_at``; the server
    computes ``due_at`` via ``services.cron_schedule.next_due_from``. There is
    no ``recurrence`` keyword or ``is_prompt``/``due_at`` write field anymore —
    both are gone from the DTO and the schema."""

    @pytest.fixture
    def client(self, db: sqlite3.Connection) -> FlaskClient:
        app = mount_namespace(scheduler_ns)
        app.config["TESTING"] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self) -> Iterator[None]:
        with patch("services.auth_session_service.validate_session", return_value=True):
            yield

    @pytest.fixture(autouse=True)
    def suppress_embed(self) -> Iterator[None]:
        """Prevent the background embedding thread from running during tests."""
        with patch("services.scheduler_service.embed_scheduled_item"):
            yield

    # ----- GET /scheduler -----

    def test_list_returns_items_as_bare_list(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        _insert_item(db, id="item1")
        _insert_item(db, id="item2", group_id="item2")

        resp = client.get("/api/scheduler")

        assert resp.status_code == 200
        body = resp.get_json()
        assert isinstance(body, list)
        assert len(body) == 2

    # ----- POST /scheduler -----

    @pytest.mark.parametrize(
        ("day", "hour", "minute"),
        [
            (None, None, None),  # E E E — every minute
            (None, None, 30),    # E E F — every hour at :30
            (None, 9, 0),        # E F F — every day at 09:00
            (15, 9, 0),          # F F F — monthly on day 15 at 09:00
        ],
    )
    def test_create_persists_row_for_every_legal_cron_shape(
        self, client: FlaskClient, db: sqlite3.Connection,
        day: int | None, hour: int | None, minute: int | None,
    ) -> None:
        resp = client.post(
            "/api/scheduler",
            json={"message": "Buy groceries", "day": day, "hour": hour, "minute": minute},
        )

        assert resp.status_code == 201, resp.get_json()
        body = resp.get_json()
        assert body["message"] == "Buy groceries"
        assert body["status"] == "pending"
        assert body["day"] == day
        assert body["hour"] == hour
        assert body["minute"] == minute
        assert body["due_at"]  # a real materialized fire instant

        row = db.execute(
            "SELECT cron_dom, cron_hour, cron_minute FROM scheduled_items WHERE message = ?",
            ("Buy groceries",),
        ).fetchone()
        assert row is not None
        assert (row["cron_dom"], row["cron_hour"], row["cron_minute"]) == (day, hour, minute)

    # ----- Every-prefix violations reject with 422, matching the ability layer -----

    @pytest.mark.parametrize(
        ("day", "hour", "minute"),
        [
            (None, 3, None),   # E F E — minute cannot be 'every' when hour is fixed
            (15, None, None),  # F E E — day fixed forces hour+minute fixed
            (15, None, 30),    # F E F
            (15, 9, None),     # F F E
        ],
    )
    def test_create_rejects_illegal_cron_shape_with_422(
        self, client: FlaskClient, db: sqlite3.Connection,
        day: int | None, hour: int | None, minute: int | None,
    ) -> None:
        before = db.execute("SELECT COUNT(*) FROM scheduled_items").fetchone()[0]

        resp = client.post(
            "/api/scheduler",
            json={"message": "Illegal shape", "day": day, "hour": hour, "minute": minute},
        )

        assert resp.status_code == 422
        assert resp.get_json()["error"] == "Validation failed"
        after = db.execute("SELECT COUNT(*) FROM scheduled_items").fetchone()[0]
        assert after == before, "an illegal cron shape must persist nothing"

    # ----- GET /scheduler/<id> -----

    def test_get_item_returns_item_when_found(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        _insert_item(db, id="xyz99999", group_id="xyz99999")

        resp = client.get("/api/scheduler/xyz99999")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["id"] == "xyz99999"
        assert body["status"] == "pending"

    # ----- PUT /scheduler/<id> -----

    def test_update_pending_item(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        _insert_item(db, id="upd00001", group_id="upd00001")

        resp = client.put(
            "/api/scheduler/upd00001",
            json={"message": "Updated message", "day": None, "hour": 18, "minute": 0},
        )

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["message"] == "Updated message"
        assert body["hour"] == 18
        assert body["minute"] == 0

        row = db.execute(
            "SELECT message, cron_hour, cron_minute FROM scheduled_items WHERE id = ?",
            ("upd00001",),
        ).fetchone()
        assert row["message"] == "Updated message"
        assert row["cron_hour"] == 18
        assert row["cron_minute"] == 0

    def test_update_returns_404_for_non_pending(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        _insert_item(db, id="fired_item", status="fired", group_id="fired_item")

        resp = client.put(
            "/api/scheduler/fired_item",
            json={"message": "Try updating"},
        )

        assert resp.status_code == 404
        assert "not pending" in resp.get_json()["error"]

    # ----- DELETE /scheduler/<id> -----

    def test_cancel_pending_item(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        _insert_item(db, id="cancel01", group_id="cancel01")

        resp = client.delete("/api/scheduler/cancel01")

        assert resp.status_code == 204
        assert resp.data == b""

        row = db.execute(
            "SELECT status FROM scheduled_items WHERE id = ?",
            ("cancel01",),
        ).fetchone()
        assert row["status"] == "cancelled"

    # ----- DELETE /scheduler/history -----

    def test_prune_history(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        for i in range(7):
            _insert_item(
                db,
                id=f"old{i}",
                status="fired",
                created_at=old_date,
                group_id=f"old{i}",
            )

        resp = client.delete("/api/scheduler/history")

        assert resp.status_code == 204

        count = db.execute("SELECT COUNT(*) FROM scheduled_items").fetchone()[0]
        assert count == 0
