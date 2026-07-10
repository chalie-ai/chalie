import sqlite3
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from flask.testing import FlaskClient

from api.scheduler import scheduler_ns
from tests.helpers import insert_scheduled_item
from tests.restx_test_app import mount_namespace


@pytest.mark.unit
class TestSchedulerAPI:
    """Tests for the scheduler namespace against the real 5-field crontab
    engine: ``scheduled_items`` is one row per schedule, forever — no
    ``item_type``/``due_at``/``status``/``group_id``/``turn_id``. ``id`` is
    the SQLite auto-increment PK, also the schedule's ``turn_id`` on the
    ``schedule`` channel. Delete is a hard ``DELETE`` (no soft-cancel state).

    There is no every-prefix invariant any more — any combination of the five
    ``minute``/``hour``/``day``/``month``/``weekday`` crontab fields is legal;
    only a malformed or out-of-range expression is rejected (422).
    """

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
        insert_scheduled_item(db, message="item1")
        insert_scheduled_item(db, message="item2")

        resp = client.get("/api/scheduler")

        assert resp.status_code == 200
        body = resp.get_json()
        assert isinstance(body, list)
        assert len(body) == 2

    # ----- POST /scheduler -----

    @pytest.mark.parametrize(
        ("day", "hour", "minute", "month", "weekday"),
        [
            ("*", "*", "*", "*", "*"),          # every minute
            ("*", "*", "30", "*", "*"),         # every hour at :30
            ("*", "9", "0", "*", "*"),           # every day at 09:00
            ("15", "9", "0", "*", "*"),          # monthly on day 15 at 09:00
            ("*", "*", "*/5", "*", "*"),         # NEW: every 5 minutes (step)
            ("*", "*", "0,15,30,45", "*", "*"),  # NEW: comma-union
            ("*", "*", "*", "*", "1-5"),         # NEW: weekdays only (range)
            ("13", "*", "*", "*", "5"),          # NEW: Vixie OR — 13th OR any Friday
        ],
    )
    def test_create_persists_row_for_any_legal_crontab_shape(
        self, client: FlaskClient, db: sqlite3.Connection,
        day: str, hour: str, minute: str, month: str, weekday: str,
    ) -> None:
        resp = client.post(
            "/api/scheduler",
            json={
                "message": "Buy groceries",
                "day": day, "hour": hour, "minute": minute,
                "month": month, "weekday": weekday,
            },
        )

        assert resp.status_code == 201, resp.get_json()
        body = resp.get_json()
        assert isinstance(body["id"], int)
        assert body["message"] == "Buy groceries"
        assert body["day"] == day
        assert body["hour"] == hour
        assert body["minute"] == minute
        assert body["month"] == month
        assert body["weekday"] == weekday
        assert body["enabled"] == 1

        row = db.execute(
            "SELECT cron_dom, cron_hour, cron_minute, cron_month, cron_dow "
            "FROM scheduled_items WHERE id = ?",
            (body["id"],),
        ).fetchone()
        assert row is not None
        assert (row["cron_dom"], row["cron_hour"], row["cron_minute"], row["cron_month"], row["cron_dow"]) == (
            day, hour, minute, month, weekday,
        )

    # ----- Malformed / out-of-range crontab fields reject with 422 -----

    @pytest.mark.parametrize(
        ("day", "hour", "minute", "month", "weekday"),
        [
            ("*", "*", "60", "*", "*"),   # minute out of range (0-59)
            ("*", "24", "*", "*", "*"),   # hour out of range (0-23)
            ("0", "*", "*", "*", "*"),    # day out of range (dom lower bound is 1)
            ("*", "*", "*", "13", "*"),   # month out of range (1-12)
            ("*", "*", "*", "*", "mon"),  # weekday: numeric only, no name tokens
        ],
    )
    def test_create_rejects_invalid_cron_expression_with_422(
        self, client: FlaskClient, db: sqlite3.Connection,
        day: str, hour: str, minute: str, month: str, weekday: str,
    ) -> None:
        before = db.execute("SELECT COUNT(*) FROM scheduled_items").fetchone()[0]

        resp = client.post(
            "/api/scheduler",
            json={
                "message": "Illegal shape",
                "day": day, "hour": hour, "minute": minute,
                "month": month, "weekday": weekday,
            },
        )

        assert resp.status_code == 422
        assert resp.get_json()["error"] == "Validation failed"
        after = db.execute("SELECT COUNT(*) FROM scheduled_items").fetchone()[0]
        assert after == before, "an invalid cron expression must persist nothing"

    # ----- Eager 1-1 gist: a schedule is born with its prompt as its label -----

    def test_create_seeds_thread_gist_from_prompt(
        self, client: FlaskClient, db: sqlite3.Connection
    ) -> None:
        resp = client.post(
            "/api/scheduler",
            json={"message": "Water the plants", "day": "*", "hour": "9", "minute": "0"},
        )
        assert resp.status_code == 201, resp.get_json()
        item_id = resp.get_json()["id"]

        # The gist is written synchronously at create time (no fork-time LLM
        # wait): keyed (channel='schedule', turn_id=id), content == the prompt.
        row = db.execute(
            "SELECT gist FROM thread_gist WHERE channel = ? AND turn_id = ?",
            ("schedule", item_id),
        ).fetchone()
        assert row is not None, "create must seed a thread_gist row"
        assert row["gist"] == "Water the plants"

        # …and it surfaces on /turns as that schedule's label.
        turns = client.get("/api/scheduler/turns").get_json()
        mine = next(t for t in turns if t["turn_id"] == item_id)
        assert mine["gist"] == "Water the plants"

    # ----- GET /scheduler/<id> -----

    def test_get_item_returns_item_when_found(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        item_id = insert_scheduled_item(db, message="Check on this")

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
        item_id = insert_scheduled_item(db, message="Original")

        resp = client.put(
            f"/api/scheduler/{item_id}",
            json={"message": "Updated message", "day": "*", "hour": "18", "minute": "0"},
        )

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["message"] == "Updated message"
        assert body["hour"] == "18"
        assert body["minute"] == "0"

        row = db.execute(
            "SELECT message, cron_hour, cron_minute FROM scheduled_items WHERE id = ?",
            (item_id,),
        ).fetchone()
        assert row["message"] == "Updated message"
        assert row["cron_hour"] == "18"
        assert row["cron_minute"] == "0"

    def test_update_returns_404_for_unknown_id(self, client: FlaskClient) -> None:
        resp = client.put(
            "/api/scheduler/999999",
            json={"message": "Try updating", "day": "*", "hour": "9", "minute": "0"},
        )

        assert resp.status_code == 404

    # ----- DELETE /scheduler/<id> — hard delete, id never reissued -----

    def test_delete_hard_deletes_row_and_id_is_never_reissued(
        self, client: FlaskClient, db: sqlite3.Connection
    ) -> None:
        item_id = insert_scheduled_item(db, message="Cancel me")

        resp = client.delete(f"/api/scheduler/{item_id}")

        assert resp.status_code == 204
        assert resp.data == b""
        assert db.execute(
            "SELECT 1 FROM scheduled_items WHERE id = ?", (item_id,)
        ).fetchone() is None

        # AUTOINCREMENT: a freshly created row never reuses the deleted id.
        new_id = insert_scheduled_item(db, message="Fresh schedule")
        assert new_id != item_id
        assert new_id > item_id

    def test_delete_returns_404_for_unknown_id(self, client: FlaskClient) -> None:
        resp = client.delete("/api/scheduler/999999")

        assert resp.status_code == 404
