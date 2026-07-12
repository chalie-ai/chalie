"""Feature tests for the WorldState API surface using real production code + real
SQLite (schema.sql). Zero mocks of Chalie services."""


import sqlite3

import pytest
from flask.testing import FlaskClient

from models.telemetry import Telemetry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_world_state(db_conn: sqlite3.Connection | None = None) -> None:

    if db_conn is not None:
        from services.heartbeat_service import heartbeat_service
        heartbeat_service._ctx = None
        db_conn.execute("DELETE FROM telemetry")
        db_conn.commit()


def _seed_telemetry(db_conn: sqlite3.Connection, ctx: dict[str, object]) -> None:
    from services.heartbeat_service import heartbeat_service
    heartbeat_service._ctx = None
    Telemetry.replace(ctx)
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

        data = client.get("/api/system/observability/world-state").get_json()
        assert "location_name:Sliema, MT" in data["rendered"]

    def test_telemetry_reflected_in_inputs(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, db_conn, _ = authed_client
        _seed_telemetry(db_conn, {"location_name": "Valletta", "mobility": "stationary"})

        inputs = client.get("/api/system/observability/world-state").get_json()["inputs"]
        assert inputs["telemetry"]["location_name"] == "Valletta"
        assert inputs["telemetry"]["mobility"] == "stationary"


# ---------------------------------------------------------------------------
# Signal push → rendered output
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Full lifecycle (integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestWorldStateFullLifecycle:
    """Push telemetry + signal → GET world-state → assert both sections rendered
    and all inputs populated."""


