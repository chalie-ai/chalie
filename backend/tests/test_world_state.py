"""
Feature tests for WorldState using real in-memory SQLite (built from schema.sql
via the ``db`` fixture), zero mocks. Each test uses a fresh WorldState instance
to avoid cross-test contamination from the module singleton.
"""


import sqlite3
import time as _time

import pytest

from services.world_state import WorldState

_HEADER = "### Background Telemetry,Processes & Signals"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh() -> WorldState:
    return WorldState()


def _seed_telemetry(db: sqlite3.Connection, ctx: dict[str, object]) -> None:
    from services.heartbeat_service import heartbeat_service
    heartbeat_service._ctx = None
    flat = heartbeat_service._flatten(ctx)
    db.execute("DELETE FROM telemetry")
    db.executemany(
        "INSERT INTO telemetry (key, value) VALUES (?, ?)",
        list(flat.items()),
    )
    db.commit()
    heartbeat_service._ctx = None


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
        # header + one bullet per surviving group, with local_time recomputed
        # from the IANA timezone and no blank line under [telemetry].
        _seed_telemetry(db, {
            "timezone": "Europe/Malta",
            "locale": "en-GB",
            "language": "en-US",
            "local_time": "10:47",                       # backend overrides from timezone
            "device": {"name": "MacBook", "battery": 82, "os": "macOS"},
            "behavioral": {"focus_state": "deep", "tab_count": 7},  # hidden group
            "connection": "4g",                          # hidden key
        })

        from zoneinfo import ZoneInfo
        from services.time_utils import utc_now
        expected_lt = utc_now().astimezone(ZoneInfo("Europe/Malta")).strftime("%a %d %b %Y %H:%M")

        expected = (
            f"{_HEADER}\n"
            "[telemetry]\n"
            f"* **user**;timezone:Europe/Malta,locale:en-GB,language:en-US,local_time:{expected_lt}\n"
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


# ---------------------------------------------------------------------------
# Signals section
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRenderSignals:
    def test_signals_render_sorted_by_source(self, db: sqlite3.Connection) -> None:
        ws = _fresh()
        ws.push_signal("zzz", "last")
        ws.push_signal("aaa", "first")
        ws.push_signal("mmm", "middle")
        result = ws.render()
        # Each appears as `[signal:src] label` (no bullet prefix), in alphabetical order.
        assert "[signal:aaa] first" in result
        assert "[signal:mmm] middle" in result
        assert "[signal:zzz] last" in result
        assert result.index("[signal:aaa]") < result.index("[signal:mmm]") < result.index("[signal:zzz]")

    def test_expired_signals_pruned_on_render(self, db: sqlite3.Connection) -> None:
        ws = _fresh()
        ws.push_signal("stale", "old news", ttl=0)
        ws.push_signal("fresh", "live news", ttl=3600)
        _time.sleep(0.01)
        result = ws.render()
        assert "[signal:fresh] live news" in result
        assert "stale" not in result
        assert "old news" not in result


# ---------------------------------------------------------------------------
# Full mix — section ordering and structural rules
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRenderFullMix:
    def test_sections_appear_in_fixed_order(self, db: sqlite3.Connection) -> None:
        _seed_telemetry(db, {"local_time": "10:00", "location_name": "Malta"})
        ws = _fresh()
        ws.push_signal("news", "heatwave")

        result = ws.render()
        assert result.startswith(_HEADER)
        idx_telemetry = result.index("[telemetry]")
        idx_signal = result.index("[signal:news]")
        assert idx_telemetry < idx_signal


