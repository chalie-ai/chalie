import sqlite3
from typing import cast

import pytest
from flask.testing import FlaskClient

from tests.helpers import insert_scheduled_item


def _unwrap_listing(body: "dict[str, object]") -> "list[dict[str, object]]":
    """Assert the success listing envelope shape and return the result array."""
    assert body.get("success") is True
    assert "error" not in body
    return cast("list[dict[str, object]]", body["result"])


def _unwrap_single(body: "dict[str, object]") -> "dict[str, object]":
    """Assert the success single-resource envelope shape and return the result dict."""
    assert body.get("success") is True
    assert "error" not in body
    return cast("dict[str, object]", body["result"])


def _find_turn(client: FlaskClient, item_id: object) -> "dict[str, object]":
    """The ``/scheduler/turns`` row for one schedule — its ``turn_id`` IS the
    schedule's id on the ``schedule`` channel."""
    body = cast("dict[str, object]", client.get("/api/scheduler/turns").get_json())
    return next(t for t in _unwrap_listing(body) if t["turn_id"] == item_id)


def _unwrap_error(body: "dict[str, object]") -> str:
    """Assert the error envelope shape and return the error message."""
    assert body.get("success") is False
    assert body.get("result") == []
    return cast(str, body["error"])


@pytest.mark.unit
class TestSchedulerAPI:
    """Tests for the scheduler Endpoint/Action routes against the real 5-field
    crontab engine: ``scheduled_items`` is one row per schedule, forever — no
    ``item_type``/``due_at``/``status``/``group_id``/``turn_id``. ``id`` is
    the SQLite auto-increment PK, also the schedule's ``turn_id`` on the
    ``schedule`` channel. Delete is a hard ``DELETE`` (no soft-cancel state).

    There is no every-prefix invariant any more — any combination of the five
    ``minute``/``hour``/``day``/``month``/``weekday`` crontab fields is legal;
    only a malformed or out-of-range expression is rejected (422).
    """

    # ----- GET /scheduler/all -----

    def test_list_returns_items_in_envelope(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        client, db, _store = authed_client
        insert_scheduled_item(db, message="item1")
        insert_scheduled_item(db, message="item2")

        resp = client.get("/api/scheduler/all")

        assert resp.status_code == 200
        body = cast("dict[str, object]", resp.get_json())
        items = _unwrap_listing(body)
        assert "pagination" in body
        assert len(items) == 2

    # ----- POST /scheduler/-1 -----

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
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object],
        day: str, hour: str, minute: str, month: str, weekday: str,
    ) -> None:
        client, db, _store = authed_client

        resp = client.post(
            "/api/scheduler/-1",
            json={
                "message": "Buy groceries",
                "day": day, "hour": hour, "minute": minute,
                "month": month, "weekday": weekday,
            },
        )

        assert resp.status_code == 201, resp.get_json()
        body = _unwrap_single(cast("dict[str, object]", resp.get_json()))
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
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object],
        day: str, hour: str, minute: str, month: str, weekday: str,
    ) -> None:
        client, db, _store = authed_client
        before = db.execute("SELECT COUNT(*) FROM scheduled_items").fetchone()[0]

        resp = client.post(
            "/api/scheduler/-1",
            json={
                "message": "Illegal shape",
                "day": day, "hour": hour, "minute": minute,
                "month": month, "weekday": weekday,
            },
        )

        assert resp.status_code == 422
        body = cast("dict[str, object]", resp.get_json())
        error = _unwrap_error(body)
        assert error, "the base contract's 422 envelope must carry a real error message"
        after = db.execute("SELECT COUNT(*) FROM scheduled_items").fetchone()[0]
        assert after == before, "an invalid cron expression must persist nothing"

    # ----- 1-1 gist: generated on the first fire, prompt is the fallback -----

    def test_create_leaves_gist_unset_and_turns_falls_back_to_the_prompt(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        client, db, _store = authed_client

        resp = client.post(
            "/api/scheduler/-1",
            json={"message": "Water the plants", "day": "*", "hour": "9", "minute": "0"},
        )
        assert resp.status_code == 201, resp.get_json()
        item_id = _unwrap_single(cast("dict[str, object]", resp.get_json()))["id"]

        # Creating a schedule writes no gist: the label is generated from the
        # fired thread on the first fire, keyed (channel='schedule', turn_id=id).
        # Seeding the raw prompt here would be indistinguishable from a real
        # gist and would permanently suppress generation.
        row = db.execute(
            "SELECT gist FROM thread_gist WHERE channel = ? AND turn_id = ?",
            ("schedule", item_id),
        ).fetchone()
        assert row is None, "create must not write a gist — it is generated on the first fire"

        # Until then /turns carries gist=None and the prompt as preview, which
        # is what the caller renders as the fallback title.
        mine = _find_turn(client, item_id)
        assert mine["gist"] is None
        assert mine["preview"] == "Water the plants"

    def test_turns_surfaces_the_generated_gist_as_the_label(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        client, db, _store = authed_client
        item_id = insert_scheduled_item(db, message="Water the plants")

        # Stand in for the first fire's gist ingest, which writes on
        # (channel='schedule', turn_id=id) — the schedule's own turn (§13.1).
        db.execute(
            "INSERT INTO thread_gist (channel, turn_id, gist, created_at) VALUES (?, ?, ?, ?)",
            ("schedule", item_id, "Watering reminder", "2026-07-18T00:00:00+00:00"),
        )
        db.commit()

        mine = _find_turn(client, item_id)
        assert mine["gist"] == "Watering reminder", "the stored gist is the schedule's label"
        assert mine["preview"] == "Water the plants", "the prompt stays available as preview"

    # ----- GET /scheduler/<id> -----

    def test_get_item_returns_item_when_found(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        client, db, _store = authed_client
        item_id = insert_scheduled_item(db, message="Check on this")

        resp = client.get(f"/api/scheduler/{item_id}")

        assert resp.status_code == 200
        body = _unwrap_single(cast("dict[str, object]", resp.get_json()))
        assert body["id"] == item_id
        assert body["message"] == "Check on this"

    def test_get_item_returns_404_when_missing(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        client, _db, _store = authed_client

        resp = client.get("/api/scheduler/999999")

        assert resp.status_code == 404

    # ----- POST /scheduler/<id> (update) — legacy PUT is now a 405 -----

    def test_update_replaces_message_and_cron_fields(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        client, db, _store = authed_client
        item_id = insert_scheduled_item(db, message="Original")

        # The legacy PUT verb is retired for this resource — the base contract
        # answers every unimplemented method with the uniform 405 envelope.
        put_resp = client.put(
            f"/api/scheduler/{item_id}",
            json={"message": "Updated message", "day": "*", "hour": "18", "minute": "0"},
        )
        assert put_resp.status_code == 405
        assert _unwrap_error(cast("dict[str, object]", put_resp.get_json())) == "Method not allowed"

        resp = client.post(
            f"/api/scheduler/{item_id}",
            json={"message": "Updated message", "day": "*", "hour": "18", "minute": "0"},
        )

        assert resp.status_code == 200
        body = _unwrap_single(cast("dict[str, object]", resp.get_json()))
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

    def test_update_returns_404_for_unknown_id(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        client, _db, _store = authed_client

        resp = client.post(
            "/api/scheduler/999999",
            json={"message": "Try updating", "day": "*", "hour": "9", "minute": "0"},
        )

        assert resp.status_code == 404

    # ----- DELETE /scheduler/<id> — hard delete, id never reissued -----

    def test_delete_hard_deletes_row_and_id_is_never_reissued(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        client, db, _store = authed_client
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

    def test_delete_returns_404_for_unknown_id(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        client, _db, _store = authed_client

        resp = client.delete("/api/scheduler/999999")

        assert resp.status_code == 404
