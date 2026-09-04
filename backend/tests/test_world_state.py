"""
Feature tests for WorldState using real in-memory SQLite (built from schema.sql
via the ``db`` fixture), zero mocks. Each test uses a fresh WorldState instance
to avoid cross-test contamination from the module singleton.
"""


import sqlite3

import pytest

from services.world_state import WorldState

_HEADER = "### Background Telemetry,Processes"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh() -> WorldState:
    return WorldState()


def _seed_telemetry(db: sqlite3.Connection, ctx: dict[str, object]) -> None:
    """Persist a heartbeat snapshot the way POST /health does (the ``db``
    fixture redirects the snapshot path into this test's tmp dir)."""
    from services.telemetry_service import TelemetryService
    TelemetryService.write(ctx)


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRenderEmpty:
    def test_empty_state_renders_nothing(self, db: sqlite3.Connection) -> None:
        ws = _fresh()
        assert ws.render() == ""


# ---------------------------------------------------------------------------
# Telemetry section
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRenderTelemetry:
    def test_telemetry_renders_exact_block(self, db: sqlite3.Connection) -> None:
        # End-to-end shape: the FE persists a heartbeat with hidden keys
        # (connection, behavioral, saved_at, _location_name_stale) plus a
        # stale local_time string. The rendered block must be exactly the
        # header + one bullet per surviving group, with the stale local_time
        # hidden — not rendered, not recomputed — and no blank line under
        # [telemetry].
        _seed_telemetry(db, {
            "timezone": "Europe/Malta",
            "locale": "en-GB",
            "language": "en-US",
            "local_time": "10:47",                       # hidden key — never rendered
            "device": {"name": "MacBook", "battery": 82, "os": "macOS"},
            "behavioral": {"focus_state": "deep", "tab_count": 7},  # hidden group
            "connection": "4g",                          # hidden key
        })

        expected = (
            f"{_HEADER}\n"
            "[telemetry]\n"
            "* **user**;timezone:Europe/Malta,locale:en-GB,language:en-US\n"
            "* **device**;name:MacBook,battery:82,os:macOS"
        )

        assert _fresh().render() == expected

    def test_location_surfaces_placename_not_raw_coordinates(self, db: sqlite3.Connection) -> None:
        # Privacy posture (): the chat/system telemetry block surfaces the
        # human-readable place name only. The raw GPS coordinates the frontend
        # sends in the nested ``location`` dict stay out of this block — backend
        # consumers (departure advisory, weather, locale_service) read them
        # directly instead. The heartbeat below mirrors what
        # ClientContextService.save() persists after Nominatim resolution: a
        # nested ``location`` dict AND a top-level resolved ``location_name``.
        _seed_telemetry(db, {
            "timezone": "Europe/Malta",
            "location": {"lat": 35.8989, "lon": 14.5146},
            "location_name": "Valletta, Malta",
        })

        result = _fresh().render()

        # The place name reaches the LLM …
        assert "location_name:Valletta, Malta" in result
        # … but the raw coordinates never do.
        assert "**location**" not in result
        assert "35.8989" not in result
        assert "14.5146" not in result
