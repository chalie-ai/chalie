import sqlite3
from typing import cast
from unittest.mock import patch

import pytest
from flask.testing import FlaskClient

from api.actions.scheduler.turns import _GISTS_PER_READ
from models.provider_response import ProviderResponse
from tests.helpers import (
    LabelProvider,
    insert_scheduled_item,
    join_named_threads,
    seed_selected_provider,
)

_BUILD_CLIENT = "services.provider_service.build_client"

# A completion the delegate genuinely settles on but that reduces to nothing: the
# think block is never closed, so stripping swallows the lot. Whitespace alone
# cannot stand in — MessageProcessor refuses to settle an empty completion and
# crashes the turn instead, which is a provider failure, not an unusable answer.
_UNUSABLE = "<think>let me consider the wording"


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

    # ----- Read-time 1-1 gist: the listing generates what it finds missing -----

    def _post(self, client: FlaskClient, message: str, hour: str = "9") -> int:
        resp = client.post(
            "/api/scheduler/-1",
            json={"message": message, "day": "*", "hour": hour, "minute": "0"},
        )
        assert resp.status_code == 201, resp.get_json()
        return cast(int, _unwrap_single(cast("dict[str, object]", resp.get_json()))["id"])

    def _turns(self, client: FlaskClient) -> "list[dict[str, object]]":
        body = cast("dict[str, object]", client.get("/api/scheduler/turns").get_json())
        return _unwrap_listing(body)

    def _turn(self, client: FlaskClient, item_id: int) -> "dict[str, object]":
        return next(t for t in self._turns(client) if t["turn_id"] == item_id)

    def _gist_of(self, db: sqlite3.Connection, item_id: int) -> str | None:
        row = db.execute(
            "SELECT gist FROM scheduled_items WHERE id = ?", (item_id,)
        ).fetchone()
        return cast("str | None", row["gist"])

    def _read_and_generate(
        self, client: FlaskClient, label: str
    ) -> tuple["list[dict[str, object]]", LabelProvider, int]:
        """One listing read with the LLM transport stubbed, waited out to
        completion — the read fires generation, the join makes it observable."""
        provider = LabelProvider(label)
        with patch(_BUILD_CLIENT, return_value=provider):
            turns = self._turns(client)
            fired = join_named_threads("schedule-gist")
        return turns, provider, fired

    def test_the_listing_generates_the_label_a_schedule_lacks(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        """Absence at the read is the whole trigger: creating stores no label,
        and the first listing that finds none generates one from the prompt.
        That read still answers labelless — the caller falls back to the
        prompt — and the next one carries the real label."""
        client, db, _store = authed_client
        seed_selected_provider(db)
        message = "Water the plants on the balcony before it gets too hot"
        item_id = self._post(client, message)

        assert self._gist_of(db, item_id) is None, "create must not write a label"

        turns, provider, fired = self._read_and_generate(client, "Balcony Plant Watering")
        assert fired == 1, "a labelless schedule must have its generation fired by the read"

        mine = next(t for t in turns if t["turn_id"] == item_id)
        assert mine["gist"] is None, "the read that kicks generation off answers labelless"
        assert mine["preview"] == message, "the prompt is the caller's fallback"

        assert message in provider.prompts[0], (
            "the delegate must be fed the schedule's own prompt — a schedule has no "
            "transcript to read one from"
        )
        assert self._gist_of(db, item_id) == "Balcony Plant Watering"
        assert self._turn(client, item_id)["gist"] == "Balcony Plant Watering"

    def test_a_schedule_that_already_has_a_label_is_never_regenerated(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        """The guard is absence, not a timer: once a label exists no later read
        may spend another LLM call on it, however many times the dock polls."""
        client, db, _store = authed_client
        seed_selected_provider(db)
        item_id = self._post(client, "Water the plants")
        self._read_and_generate(client, "Plant Watering")
        assert self._gist_of(db, item_id) == "Plant Watering"

        turns, untouched, fired = self._read_and_generate(client, "SHOULD NEVER BE STORED")
        assert fired == 0, "a labelled schedule must fire no generation"
        assert untouched.prompts == [], "the delegate must not run for a labelled schedule"
        assert self._gist_of(db, item_id) == "Plant Watering"
        assert next(t for t in turns if t["turn_id"] == item_id)["gist"] == "Plant Watering"

    def test_rewriting_the_message_drops_the_label_and_the_next_read_regenerates_it(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        """A rewritten prompt must not keep a label describing the old one. The
        edit deletes it; absence then feeds the same one read-time rule. An edit
        that leaves the message alone keeps its label untouched."""
        client, db, _store = authed_client
        seed_selected_provider(db)
        item_id = self._post(client, "Water the plants")
        self._read_and_generate(client, "Plant Watering")

        resp = client.post(
            f"/api/scheduler/{item_id}",
            json={
                "message": "Call the dentist to book a check-up",
                "day": "*", "hour": "9", "minute": "0",
            },
        )
        assert resp.status_code == 200, resp.get_json()
        assert self._gist_of(db, item_id) is None, "a rewritten prompt must clear the stale label"

        self._read_and_generate(client, "Dentist Appointment")
        assert self._gist_of(db, item_id) == "Dentist Appointment"

        # A cron-only edit leaves the message untouched — the label stands, so
        # the next read has nothing to regenerate.
        resp = client.post(
            f"/api/scheduler/{item_id}",
            json={
                "message": "Call the dentist to book a check-up",
                "day": "*", "hour": "18", "minute": "30",
            },
        )
        assert resp.status_code == 200, resp.get_json()
        assert self._gist_of(db, item_id) == "Dentist Appointment"
        _turns, untouched, fired = self._read_and_generate(client, "SHOULD NEVER BE STORED")
        assert fired == 0 and untouched.prompts == []

    def test_one_read_generates_no_more_than_the_per_read_cap(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        """The first read after an upgrade finds EVERY schedule labelless at
        once. Uncapped that is one concurrent LLM call per schedule against a
        single local model, so a read generates a bounded few and the next poll
        takes the rest — every schedule labelled, none of them at once."""
        client, db, _store = authed_client
        seed_selected_provider(db)
        total = _GISTS_PER_READ + 2
        for n in range(total):
            self._post(client, f"Errand number {n}", hour=str(n + 1))

        _turns, provider, fired = self._read_and_generate(client, "An Errand")
        assert fired == _GISTS_PER_READ, "one read must fire exactly the cap, not one per schedule"
        assert len(provider.prompts) == _GISTS_PER_READ

        labelled = db.execute(
            "SELECT COUNT(*) FROM scheduled_items WHERE gist IS NOT NULL"
        ).fetchone()[0]
        assert labelled == _GISTS_PER_READ

        # The backlog drains on later reads rather than being abandoned.
        _turns, _p, fired = self._read_and_generate(client, "An Errand")
        assert fired == total - _GISTS_PER_READ, "the remainder must be picked up by the next read"

    def test_a_generation_failure_still_lists_the_schedule_with_its_prompt(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        """The label is a decoration, never a precondition: with the provider
        down the schedule is still listed, labelless, prompt intact — and no
        placeholder is written to stand in for the label that failed."""
        client, db, _store = authed_client
        seed_selected_provider(db)
        message = "Take the bins out"
        item_id = self._post(client, message)

        with patch(_BUILD_CLIENT, side_effect=ConnectionError("provider unreachable")):
            assert self._turns(client), "the listing must answer even when generation cannot"
            join_named_threads("schedule-gist")

        assert self._gist_of(db, item_id) is None, (
            "a failed generation must store nothing, not a placeholder"
        )
        mine = self._turn(client, item_id)
        assert mine["gist"] is None, "no label generated means no label reported"
        assert mine["preview"] == message, "the prompt stays as the caller's fallback"

    def test_an_unusable_generation_falls_back_to_the_prompt_and_is_never_retried(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        """A model that answers, but with nothing that survives think-stripping
        (an unclosed block swallows everything after its opener), must still
        settle the row. The prompt itself is stored — exactly what the caller
        would have fallen back to — so absence stays a fresh signal: the
        schedule leaves the labelless set and no later poll can spend another
        LLM call on it. No websocket frame either; a schedule label is polled
        for, never pushed."""
        client, db, _store = authed_client
        seed_selected_provider(db)
        message = "Renew the passport"
        item_id = self._post(client, message)
        frames: list[object] = []

        with (
            patch(_BUILD_CLIENT, return_value=LabelProvider(_UNUSABLE)),
            patch("services.websocket.Websocket.broadcast", side_effect=frames.append),
        ):
            self._turns(client)
            assert join_named_threads("schedule-gist") == 1

        assert self._gist_of(db, item_id) == message, (
            "an unusable generation must settle on the prompt, not stay labelless forever"
        )
        assert frames == [], "a schedule label is polled for, never broadcast"

        _turns, untouched, fired = self._read_and_generate(client, "SHOULD NEVER BE STORED")
        assert fired == 0, "a schedule that could not be labelled must never be re-fired"
        assert untouched.prompts == []

    def test_an_unlabellable_schedule_does_not_starve_the_ones_behind_it(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        """Candidates come off one per-read cap, so schedules that cannot be
        labelled must not hold those slots read after read and block the backlog
        behind them. Settling them on their first attempt is what frees the
        queue: the next read reaches the rest instead of re-firing them."""
        client, db, _store = authed_client
        seed_selected_provider(db)
        total = _GISTS_PER_READ + 2
        for n in range(total):
            self._post(client, f"Errand number {n}", hour=str(n + 1))

        with patch(_BUILD_CLIENT, return_value=LabelProvider(_UNUSABLE)):
            self._turns(client)
            assert join_named_threads("schedule-gist") == _GISTS_PER_READ

        _turns, _p, fired = self._read_and_generate(client, "An Errand")
        assert fired == total - _GISTS_PER_READ, (
            "the unlabellable ones must not consume the cap a second time"
        )
        labelless = db.execute(
            "SELECT COUNT(*) FROM scheduled_items WHERE gist IS NULL"
        ).fetchone()[0]
        assert labelless == 0, "no schedule may be left behind an unlabellable one"

    def test_an_edit_landing_mid_generation_is_not_overwritten_with_the_old_label(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]
    ) -> None:
        """Generation reads the prompt at listing time and finishes seconds to
        minutes later. An edit landing in that window clears the label to ask
        for a fresh one — the in-flight label describes text that no longer
        exists, so it must be discarded, not stored where absence can never
        correct it."""
        client, db, _store = authed_client
        seed_selected_provider(db)
        item_id = self._post(client, "Water the plants")

        class EditsMidFlight(LabelProvider):
            def send(self, dto: object) -> "ProviderResponse":
                from models.scheduled_item import ScheduledItem
                ScheduledItem.filter("id", item_id).update(
                    message="Call the dentist", gist=None
                )
                return super().send(dto)

        with patch(_BUILD_CLIENT, return_value=EditsMidFlight("Plant Watering")):
            self._turns(client)
            assert join_named_threads("schedule-gist") == 1

        assert self._gist_of(db, item_id) is None, (
            "a label for the replaced prompt must not be stored"
        )
        self._read_and_generate(client, "Dentist Appointment")
        assert self._gist_of(db, item_id) == "Dentist Appointment"

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
