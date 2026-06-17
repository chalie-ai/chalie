"""
Feature tests for WorldState using real in-memory SQLite (built from schema.sql
via the ``db`` fixture), zero mocks. Each test uses a fresh WorldState instance
to avoid cross-test contamination from the module singleton.
"""


import time as _time
import uuid
from datetime import timedelta

import pytest

from services.time_utils import utc_now
from services.world_state import WorldState

_HEADER = "### Background Telemetry,Processes & Signals"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh() -> WorldState:
    return WorldState()


def _future_iso(minutes: int) -> str:
    return (utc_now() + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def _past_iso(minutes: int) -> str:
    return (utc_now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def _seed_telemetry(db, ctx: dict) -> None:
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


def _seed_pending(db, message: str, due_minutes_ahead: int, *, recurrence: str | None = None) -> None:
    db.execute(
        "INSERT INTO scheduled_items (id, message, due_at, status, hidden, recurrence) "
        "VALUES (?, ?, ?, 'pending', 0, ?)",
        (str(uuid.uuid4()), message, _future_iso(due_minutes_ahead), recurrence),
    )


def _seed_fired(db, message: str, fired_minutes_ago: int) -> None:
    iso = _past_iso(fired_minutes_ago)
    db.execute(
        "INSERT INTO scheduled_items (id, message, due_at, status, hidden, last_fired_at) "
        "VALUES (?, ?, ?, 'fired', 0, ?)",
        (str(uuid.uuid4()), message, iso, iso),
    )


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRenderEmpty:
    def test_empty_state_renders_nothing(self, db):
        ws = _fresh()
        assert ws.render() == ""


# ---------------------------------------------------------------------------
# Telemetry section
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRenderTelemetry:
    def test_telemetry_renders_exact_block(self, db):
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

    def test_location_surfaces_placename_not_raw_coordinates(self, db):
        # Privacy posture (TKT-557): the chat/system telemetry block surfaces the
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
# Schedule section
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRenderSchedule:
    def test_upcoming_pending_renders_with_due_in_and_repeats(self, db):
        # Single recurring upcoming item — covers the happy path: it appears,
        # gets a positive due-in, and the recurrence is formatted as repeats:every.
        _seed_pending(db, "Daily standup", due_minutes_ahead=60, recurrence="86400")
        db.commit()

        result = _fresh().render()
        assert "[schedule]" in result
        bullet = next(ln for ln in result.splitlines() if "Daily standup" in ln)
        assert bullet.startswith("* Daily standup (due-in:")
        assert "ago" not in bullet.split("(", 1)[1].split(",")[0]
        assert "repeats:every 1d 0h" in bullet

    def test_hidden_items_excluded(self, db):
        db.execute(
            "INSERT INTO scheduled_items (id, message, due_at, status, hidden) "
            "VALUES (?, ?, ?, 'pending', 1)",
            (str(uuid.uuid4()), "Secret task", _future_iso(60)),
        )
        db.commit()

        assert "Secret task" not in _fresh().render()

    def test_upcoming_pending_supersedes_recent_fire_for_same_message(self, db):
        # Same job has both "just fired" and "due again soon" — show the upcoming one.
        _seed_fired(db, "Mail sync", fired_minutes_ago=30)
        _seed_pending(db, "Mail sync", due_minutes_ahead=45)
        db.commit()

        bullets = [ln for ln in _fresh().render().splitlines() if ln.startswith("* Mail sync")]
        assert len(bullets) == 1
        due_in_field = bullets[0].split("(", 1)[1].split(",")[0]
        assert due_in_field.startswith("due-in:") and "ago" not in due_in_field, (
            f"Expected upcoming due-in (no 'ago'), got: {due_in_field!r}"
        )

    def test_repeated_pending_shows_only_next_upcoming(self, db):
        for minutes_ahead in (300, 90, 600):
            _seed_pending(db, "Sync run", due_minutes_ahead=minutes_ahead)
        db.commit()

        bullets = [ln for ln in _fresh().render().splitlines() if ln.startswith("* Sync run")]
        assert len(bullets) == 1
        # Earliest is 90m → "1h Xm"; the 5h and 10h variants must NOT appear.
        assert "due-in:1h" in bullets[0], f"Expected earliest (90m → 1h…), got: {bullets[0]!r}"
        assert "5h" not in bullets[0]
        assert "10h" not in bullets[0]


# ---------------------------------------------------------------------------
# Signals section
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRenderSignals:
    def test_signals_render_sorted_by_source(self, db):
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

    def test_expired_signals_pruned_on_render(self, db):
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
    def test_sections_appear_in_fixed_order(self, db):
        _seed_telemetry(db, {"local_time": "10:00", "location_name": "Malta"})
        ws = _fresh()
        ws.push_signal("news", "heatwave")
        _seed_pending(db, "Team meeting", due_minutes_ahead=60)
        db.commit()

        result = ws.render()
        assert result.startswith(_HEADER)
        idx_telemetry = result.index("[telemetry]")
        idx_schedule = result.index("[schedule]")
        idx_signal = result.index("[signal:news]")
        assert idx_telemetry < idx_schedule < idx_signal


