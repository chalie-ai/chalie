"""Feature tests for the WorldState API surface using real production code + real
SQLite (schema.sql). Zero mocks of Chalie services."""


import sqlite3

import pytest
from flask.testing import FlaskClient



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_world_state(db_conn: sqlite3.Connection | None = None) -> None:

    if db_conn is not None:
        from services.file_mapper_service import FileMapperService
        from services.telemetry_service import TelemetryService
        FileMapperService.get_telemetry_json_path().unlink(missing_ok=True)
        TelemetryService._cache = None


def _seed_telemetry(db_conn: sqlite3.Connection, ctx: dict[str, object]) -> None:
    """Persist a heartbeat snapshot the way POST /health does (the ``db``
    fixture behind ``db_conn`` redirects the snapshot path into tmp)."""
    from services.telemetry_service import TelemetryService
    TelemetryService.write(ctx)


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
        assert resp.get_json()["rendered"] == ""


# ---------------------------------------------------------------------------
# GET /system/observability/world-state — telemetry
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWorldStateTelemetryRendering:
    def test_telemetry_reflected_in_inputs(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, db_conn, _ = authed_client
        _seed_telemetry(db_conn, {"location_name": "Valletta", "mobility": "stationary"})

        body = client.get("/api/system/observability/world-state").get_json()
        inputs = body["inputs"]
        assert inputs["telemetry"]["location_name"] == "Valletta"
        assert inputs["telemetry"]["mobility"] == "stationary"
        # ``rendered`` is the locale line the model sees, not a world block.
        assert body["rendered"] == "User is currently in Valletta. Use this information to better tailor your response."
