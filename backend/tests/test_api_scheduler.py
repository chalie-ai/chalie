import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import cast
from unittest.mock import patch

import pytest
from flask.testing import FlaskClient

from api.scheduler import scheduler_ns
from tests.restx_test_app import mount_namespace


def _insert_item(db: sqlite3.Connection, **overrides: object) -> int:
    """Insert a row against the prompt-only ``scheduled_items`` schema
    (TKT-1434) and return its auto-assigned integer ``id``."""
    now = datetime.now(timezone.utc).isoformat()
    defaults: dict[str, object] = dict(
        message="Test reminder",
        start_at=now,
        cron_dom=None,
        cron_hour=None,
        cron_minute=None,
        enabled=1,
        channel="general",
        created_by_session=None,
        created_at=now,
    )
    defaults.update(overrides)
    d = defaults
    cur = db.execute(
        """
        INSERT INTO scheduled_items
          (message, start_at, cron_dom, cron_hour, cron_minute,
           enabled, channel, created_by_session, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            d["message"], d["start_at"], d["cron_dom"], d["cron_hour"], d["cron_minute"],
            d["enabled"], d["channel"], d["created_by_session"], d["created_at"],
        ),
    )
    db.commit()
    return cast(int, cur.lastrowid)


@pytest.mark.unit
class TestSchedulerAPI:
    """Tests for the scheduler namespace against the prompt-only dumb-cron
    model (TKT-1434): ``scheduled_items`` is one row per schedule, forever —
    no ``item_type``/``due_at``/``status``/``group_id``/``turn_id``. ``id`` is
    the SQLite auto-increment PK, also the schedule's ``turn_id`` on the
    ``schedule`` channel. Delete is a hard ``DELETE`` (no soft-cancel state)."""

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
        _insert_item(db, message="item1")
        _insert_item(db, message="item2")

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
        assert isinstance(body["id"], int)
        assert body["message"] == "Buy groceries"
        assert body["day"] == day
        assert body["hour"] == hour
        assert body["minute"] == minute
        assert body["enabled"] == 1

        row = db.execute(
            "SELECT cron_dom, cron_hour, cron_minute FROM scheduled_items WHERE id = ?",
            (body["id"],),
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
        item_id = _insert_item(db, message="Check on this")

        resp = client.get(f"/api/scheduler/{item_id}")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["id"] == item_id
        assert body["message"] == "Check on this"

    def test_get_item_returns_404_when_missing(self, client: FlaskClient) -> None:
        resp = client.get("/api/scheduler/999999")

        assert resp.status_code == 404

    # ----- PUT /scheduler/<id> -----

    def test_update_replaces_message_and_cron_fields(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        item_id = _insert_item(db, message="Original")

        resp = client.put(
            f"/api/scheduler/{item_id}",
            json={"message": "Updated message", "day": None, "hour": 18, "minute": 0},
        )

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["message"] == "Updated message"
        assert body["hour"] == 18
        assert body["minute"] == 0

        row = db.execute(
            "SELECT message, cron_hour, cron_minute FROM scheduled_items WHERE id = ?",
            (item_id,),
        ).fetchone()
        assert row["message"] == "Updated message"
        assert row["cron_hour"] == 18
        assert row["cron_minute"] == 0

    def test_update_returns_404_for_unknown_id(self, client: FlaskClient) -> None:
        resp = client.put(
            "/api/scheduler/999999",
            json={"message": "Try updating", "day": None, "hour": 9, "minute": 0},
        )

        assert resp.status_code == 404

    # ----- DELETE /scheduler/<id> — hard delete, id never reissued -----

    def test_delete_hard_deletes_row_and_id_is_never_reissued(
        self, client: FlaskClient, db: sqlite3.Connection
    ) -> None:
        item_id = _insert_item(db, message="Cancel me")

        resp = client.delete(f"/api/scheduler/{item_id}")

        assert resp.status_code == 204
        assert resp.data == b""
        assert db.execute(
            "SELECT 1 FROM scheduled_items WHERE id = ?", (item_id,)
        ).fetchone() is None

        # AUTOINCREMENT: a freshly created row never reuses the deleted id.
        new_id = _insert_item(db, message="Fresh schedule")
        assert new_id != item_id
        assert new_id > item_id

    def test_delete_returns_404_for_unknown_id(self, client: FlaskClient) -> None:
        resp = client.delete("/api/scheduler/999999")

        assert resp.status_code == 404
