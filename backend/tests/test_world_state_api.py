"""
Feature tests for the WorldState API surface.

  - GET /system/observability/world-state  (Cognition endpoint)
  - Signal push via world_state.push_signal() reflected in rendered output
  - Telemetry push via world_state.set("telemetry") reflected in rendered output
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


def _reset_world_state():
    """Wipe all in-memory fragments so each test starts from a clean singleton."""
    world_state.set("telemetry", {})
    world_state.set("signals", {})


# ---------------------------------------------------------------------------
# GET /system/observability/world-state — empty state
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWorldStateObservabilityEmpty:
    """When no data has been pushed, the endpoint returns empty rendered string
    and four correctly-keyed inputs with zero entries."""

    def test_empty_state_returns_200(self, authed_client):
        client, _, _ = authed_client
        _reset_world_state()

        resp = client.get("/system/observability/world-state")
        assert resp.status_code == 200

    def test_empty_state_rendered_is_empty_string(self, authed_client):
        client, _, _ = authed_client
        _reset_world_state()

        data = client.get("/system/observability/world-state").get_json()
        assert data["rendered"] == ""

    def test_empty_state_inputs_has_four_required_keys(self, authed_client):
        client, _, _ = authed_client
        _reset_world_state()

        inputs = client.get("/system/observability/world-state").get_json()["inputs"]
        assert "telemetry" in inputs
        assert "signals" in inputs
        assert "schedule" in inputs
        assert "bg_processes" in inputs

    def test_empty_state_inputs_all_zero_or_empty(self, authed_client):
        client, _, _ = authed_client
        _reset_world_state()

        inputs = client.get("/system/observability/world-state").get_json()["inputs"]
        assert inputs["telemetry"] == {}
        assert inputs["signals"] == {}
        assert inputs["schedule"] == []
        assert inputs["bg_processes"] == []


# ---------------------------------------------------------------------------
# GET /system/observability/world-state — telemetry
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWorldStateTelemetryRendering:
    """Telemetry pushed to the singleton appears verbatim in the rendered block."""

    def test_telemetry_section_header_present(self, authed_client):
        client, _, _ = authed_client
        _reset_world_state()
        world_state.set("telemetry", {"local_time": "10:00", "location": "Valletta"})

        data = client.get("/system/observability/world-state").get_json()
        assert "### Background Telemetry,Processes & Signals" in data["rendered"]
        assert "[telemetry]" in data["rendered"]

    def test_telemetry_location_appears_in_rendered(self, authed_client):
        client, _, _ = authed_client
        _reset_world_state()
        world_state.set("telemetry", {"location": "Sliema, MT"})

        data = client.get("/system/observability/world-state").get_json()
        assert "location:Sliema, MT" in data["rendered"]

    def test_telemetry_reflected_in_inputs(self, authed_client):
        client, _, _ = authed_client
        _reset_world_state()
        world_state.set("telemetry", {"location": "Valletta", "mobility": "stationary"})

        inputs = client.get("/system/observability/world-state").get_json()["inputs"]
        assert inputs["telemetry"]["location"] == "Valletta"
        assert inputs["telemetry"]["mobility"] == "stationary"


# ---------------------------------------------------------------------------
# GET /system/observability/world-state — schedule rows
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWorldStateScheduleRendering:
    """Rows in scheduled_items appear as [schedule] bullets in rendered output
    and in inputs.schedule."""

    def test_pending_scheduled_item_appears_in_inputs_schedule(self, authed_client):
        client, db_conn, _ = authed_client
        _reset_world_state()

        item_id = str(uuid.uuid4())
        db_conn.execute(
            "INSERT INTO scheduled_items (id, message, due_at, status, hidden) "
            "VALUES (?, ?, ?, 'pending', 0)",
            (item_id, "Check-in flight", _future_iso(120)),
        )
        db_conn.commit()

        data = client.get("/system/observability/world-state").get_json()
        messages = [r["message"] for r in data["inputs"]["schedule"]]
        assert "Check-in flight" in messages

    def test_pending_scheduled_item_appears_in_rendered_block(self, authed_client):
        client, db_conn, _ = authed_client
        _reset_world_state()

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
# GET /system/observability/world-state — bg_process rows
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWorldStateBgProcessRendering:
    """Rows in transcript WHERE channel='goal_pursuit' appear as [bg_process(…)]
    lines and in inputs.bg_processes."""

    def test_goal_pursuit_row_appears_in_inputs_bg_processes(self, authed_client):
        client, db_conn, _ = authed_client
        _reset_world_state()

        db_conn.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('goal_pursuit', 'assistant', 'Booking hotel', ?)",
            (_recent_iso(60),),
        )
        db_conn.commit()

        data = client.get("/system/observability/world-state").get_json()
        contents = [r["content"] for r in data["inputs"]["bg_processes"]]
        assert "Booking hotel" in contents

    def test_goal_pursuit_row_appears_in_rendered_block(self, authed_client):
        client, db_conn, _ = authed_client
        _reset_world_state()

        db_conn.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('goal_pursuit', 'assistant', 'Booking holiday flights', ?)",
            (_recent_iso(60),),
        )
        db_conn.commit()

        data = client.get("/system/observability/world-state").get_json()
        assert "[bg_process(" in data["rendered"]
        assert "Booking holiday flights" in data["rendered"]


# ---------------------------------------------------------------------------
# Signal push → rendered output
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWorldStateSignalRendering:
    """push_signal() on the singleton produces a [signal:…] line in rendered output."""

    def test_signal_appears_in_rendered_block(self, authed_client):
        client, _, _ = authed_client
        _reset_world_state()
        world_state.push_signal("weather", "Thunderstorm at 18:00")

        data = client.get("/system/observability/world-state").get_json()
        assert "[signal:weather]" in data["rendered"]
        assert "Thunderstorm at 18:00" in data["rendered"]

    def test_signal_reflected_in_inputs_signals(self, authed_client):
        client, _, _ = authed_client
        _reset_world_state()
        world_state.push_signal("inbox", "3 unread emails")

        inputs = client.get("/system/observability/world-state").get_json()["inputs"]
        assert "inbox" in inputs["signals"]
        assert inputs["signals"]["inbox"]["label"] == "3 unread emails"


# ---------------------------------------------------------------------------
# Full lifecycle (integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestWorldStateFullLifecycle:
    """Push all four data sources → GET world-state → assert all sections rendered
    and all inputs populated."""

    def test_all_four_sections_in_render_and_inputs(self, authed_client):
        client, db_conn, _ = authed_client
        _reset_world_state()

        # Telemetry
        world_state.set("telemetry", {"location": "Sliema, MT", "local_time": "14:00"})

        # Signal
        world_state.push_signal("news_service", "Malta heatwave warning")

        # Schedule
        item_id = str(uuid.uuid4())
        db_conn.execute(
            "INSERT INTO scheduled_items (id, message, due_at, status, hidden) "
            "VALUES (?, ?, ?, 'pending', 0)",
            (item_id, "Call Mum", _future_iso(120)),
        )

        # bg_process
        db_conn.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('goal_pursuit', 'assistant', 'Booking holiday flights', ?)",
            (_recent_iso(60),),
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
        assert "[bg_process(" in rendered
        assert "Booking holiday flights" in rendered
        assert "[signal:news_service]" in rendered
        assert "Malta heatwave warning" in rendered

        inputs = data["inputs"]
        assert inputs["telemetry"]["location"] == "Sliema, MT"
        assert "news_service" in inputs["signals"]
        assert any(r["message"] == "Call Mum" for r in inputs["schedule"])
        assert any(r["content"] == "Booking holiday flights" for r in inputs["bg_processes"])
