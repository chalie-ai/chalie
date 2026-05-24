"""
Feature tests for the WorldState API surface.

  - GET /system/observability/world-state  (Cognition endpoint)
  - Signal push via world_state.push_signal() reflected in rendered output
  - Telemetry rows in the ``telemetry`` table reflected in rendered output
  - Schedule items seeded into DB appear in rendered output + inputs
  - bg_process rows seeded into DB appear in rendered output + inputs
  - Full-lifecycle integration test (@pytest.mark.integration)

All tests use real production code + real SQLite from schema.sql.
Zero mocks of Chalie services.
"""


import uuid
from datetime import timedelta

import pytest

from services.world_state import world_state
from services.time_utils import utc_now


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _future_iso(minutes: int = 60) -> str:
    return (utc_now() + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def _recent_iso(seconds: int = 30) -> str:
    return (utc_now() - timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")


def _reset_world_state(db_conn=None):
    """Wipe all in-memory fragments so each test starts from a clean singleton.

    When ``db_conn`` is supplied the telemetry table is cleared too — the
    /health and observability endpoints both read it.
    """
    world_state.set("signals", {})
    if db_conn is not None:
        from services.heartbeat_service import heartbeat_service
        heartbeat_service._ctx = None
        db_conn.execute("DELETE FROM telemetry")
        db_conn.commit()


def _seed_telemetry(db_conn, ctx: dict) -> None:
    """Persist a flat key/value snapshot into the telemetry table."""
    from services.heartbeat_service import heartbeat_service
    heartbeat_service._ctx = None
    flat = heartbeat_service._flatten(ctx)
    db_conn.execute("DELETE FROM telemetry")
    db_conn.executemany(
        "INSERT INTO telemetry (key, value) VALUES (?, ?)",
        list(flat.items()),
    )
    db_conn.commit()
    heartbeat_service._ctx = None


# ---------------------------------------------------------------------------
# GET /system/observability/world-state — empty state
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWorldStateObservabilityEmpty:
    """When no data has been pushed, the endpoint returns empty rendered string
    and four correctly-keyed inputs with zero entries."""

    def test_empty_state_returns_200(self, authed_client):
        client, db_conn, _ = authed_client
        _reset_world_state(db_conn)

        resp = client.get("/system/observability/world-state")
        assert resp.status_code == 200

    def test_empty_state_rendered_is_empty_string(self, authed_client):
        client, db_conn, _ = authed_client
        _reset_world_state(db_conn)

        data = client.get("/system/observability/world-state").get_json()
        assert data["rendered"] == ""


# ---------------------------------------------------------------------------
# GET /system/observability/world-state — telemetry
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWorldStateTelemetryRendering:
    """Telemetry pushed to the singleton appears verbatim in the rendered block."""

    def test_telemetry_location_appears_in_rendered(self, authed_client):
        client, db_conn, _ = authed_client
        _seed_telemetry(db_conn, {"location_name": "Sliema, MT"})
        world_state.set("signals", {})

        data = client.get("/system/observability/world-state").get_json()
        assert "location_name:Sliema, MT" in data["rendered"]

    def test_telemetry_reflected_in_inputs(self, authed_client):
        client, db_conn, _ = authed_client
        _seed_telemetry(db_conn, {"location_name": "Valletta", "mobility": "stationary"})
        world_state.set("signals", {})

        inputs = client.get("/system/observability/world-state").get_json()["inputs"]
        assert inputs["telemetry"]["location_name"] == "Valletta"
        assert inputs["telemetry"]["mobility"] == "stationary"


# ---------------------------------------------------------------------------
# GET /system/observability/world-state — schedule rows
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWorldStateScheduleRendering:
    """Rows in scheduled_items appear as [schedule] bullets in rendered output
    and in inputs.schedule."""

    def test_pending_scheduled_item_appears_in_rendered_block(self, authed_client):
        client, db_conn, _ = authed_client
        _reset_world_state(db_conn)

        item_id = str(uuid.uuid4())
        db_conn.execute(
            "INSERT INTO scheduled_items (id, message, due_at, status, hidden) "
            "VALUES (?, ?, ?, 'pending', 0)",
            (item_id, "Call Mum", _future_iso(120)),
        )
        db_conn.commit()

        data = client.get("/system/observability/world-state").get_json()
        assert "[schedule]" in data["rendered"]
        assert "Call Mum" in data["rendered"]
        assert "due-in:" in data["rendered"]


# ---------------------------------------------------------------------------
# Signal push → rendered output
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWorldStateSignalRendering:
    """push_signal() on the singleton produces a [signal:…] line in rendered output."""

    def test_signal_appears_in_rendered_block(self, authed_client):
        client, db_conn, _ = authed_client
        _reset_world_state(db_conn)
        world_state.push_signal("weather", "Thunderstorm at 18:00")

        data = client.get("/system/observability/world-state").get_json()
        assert "[signal:weather]" in data["rendered"]
        assert "Thunderstorm at 18:00" in data["rendered"]



# ---------------------------------------------------------------------------
# Full lifecycle (integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestWorldStateFullLifecycle:
    """Push all four data sources → GET world-state → assert all sections rendered
    and all inputs populated."""

    def test_all_three_sections_in_render_and_inputs(self, authed_client):
        client, db_conn, _ = authed_client
        _reset_world_state(db_conn)

        # Telemetry
        _seed_telemetry(db_conn, {"location_name": "Sliema, MT", "local_time": "14:00"})

        # Signal
        world_state.push_signal("news_service", "Malta heatwave warning")

        # Schedule
        item_id = str(uuid.uuid4())
        db_conn.execute(
            "INSERT INTO scheduled_items (id, message, due_at, status, hidden) "
            "VALUES (?, ?, ?, 'pending', 0)",
            (item_id, "Call Mum", _future_iso(120)),
        )
        db_conn.commit()

        resp = client.get("/system/observability/world-state")
        assert resp.status_code == 200
        data = resp.get_json()

        rendered = data["rendered"]
        assert "### Background Telemetry,Processes & Signals" in rendered
        assert "[telemetry]" in rendered
        assert "Sliema, MT" in rendered
        assert "[schedule]" in rendered
        assert "Call Mum" in rendered
        assert "[signal:news_service]" in rendered
        assert "Malta heatwave warning" in rendered

        inputs = data["inputs"]
        assert inputs["telemetry"]["location_name"] == "Sliema, MT"
        assert "news_service" in inputs["signals"]
        assert any(r["message"] == "Call Mum" for r in inputs["schedule"])
