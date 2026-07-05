"""Feature tests for the WorldState API surface using real production code + real
SQLite (schema.sql). Zero mocks of Chalie services."""


import sqlite3

import pytest
from flask.testing import FlaskClient

from services.world_state import world_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_world_state(db_conn: sqlite3.Connection | None = None) -> None:

    world_state.set("signals", {})
    if db_conn is not None:
        from services.heartbeat_service import heartbeat_service
        heartbeat_service._ctx = None
        db_conn.execute("DELETE FROM telemetry")
        db_conn.commit()


def _seed_telemetry(db_conn: sqlite3.Connection, ctx: dict[str, object]) -> None:
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
    def test_empty_state_returns_200(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, db_conn, _ = authed_client
        _reset_world_state(db_conn)

        resp = client.get("/api/system/observability/world-state")
        assert resp.status_code == 200

    def test_empty_state_rendered_is_empty_string(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, db_conn, _ = authed_client
        _reset_world_state(db_conn)

        data = client.get("/api/system/observability/world-state").get_json()
        assert data["rendered"] == ""


# ---------------------------------------------------------------------------
# GET /system/observability/world-state — telemetry
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWorldStateTelemetryRendering:
    def test_telemetry_location_appears_in_rendered(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, db_conn, _ = authed_client
        _seed_telemetry(db_conn, {"location_name": "Sliema, MT"})
        world_state.set("signals", {})

        data = client.get("/api/system/observability/world-state").get_json()
        assert "location_name:Sliema, MT" in data["rendered"]

    def test_telemetry_reflected_in_inputs(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, db_conn, _ = authed_client
        _seed_telemetry(db_conn, {"location_name": "Valletta", "mobility": "stationary"})
        world_state.set("signals", {})

        inputs = client.get("/api/system/observability/world-state").get_json()["inputs"]
        assert inputs["telemetry"]["location_name"] == "Valletta"
        assert inputs["telemetry"]["mobility"] == "stationary"


# ---------------------------------------------------------------------------
# Signal push → rendered output
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWorldStateSignalRendering:
    """push_signal() on the singleton produces a [signal:…] line in rendered output."""

    def test_signal_appears_in_rendered_block(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, db_conn, _ = authed_client
        _reset_world_state(db_conn)
        world_state.push_signal("weather", "Thunderstorm at 18:00")

        data = client.get("/api/system/observability/world-state").get_json()
        assert "[signal:weather]" in data["rendered"]
        assert "Thunderstorm at 18:00" in data["rendered"]



# ---------------------------------------------------------------------------
# Full lifecycle (integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestWorldStateFullLifecycle:
    """Push telemetry + signal → GET world-state → assert both sections rendered
    and all inputs populated."""

    def test_telemetry_and_signal_sections_in_render_and_inputs(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, db_conn, _ = authed_client
        _reset_world_state(db_conn)

        # Telemetry
        _seed_telemetry(db_conn, {"location_name": "Sliema, MT", "local_time": "14:00"})

        # Signal
        world_state.push_signal("news_service", "Malta heatwave warning")

        resp = client.get("/api/system/observability/world-state")
        assert resp.status_code == 200
        data = resp.get_json()

        rendered = data["rendered"]
        assert "### Background Telemetry,Processes & Signals" in rendered
        assert "[telemetry]" in rendered
        assert "Sliema, MT" in rendered
        assert "[signal:news_service]" in rendered
        assert "Malta heatwave warning" in rendered

        inputs = data["inputs"]
        assert inputs["telemetry"]["location_name"] == "Sliema, MT"
        assert "news_service" in inputs["signals"]
        assert "schedule" not in inputs
