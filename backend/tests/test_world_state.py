"""
Feature tests for WorldState — the singleton that renders ambient world context
into the literal block consumed by prompt assembly.

Real WorldState class, real in-memory SQLite (built from schema.sql via the
``db`` fixture), zero mocks. Each test uses a fresh WorldState instance to
avoid cross-test contamination from the module singleton.

Per tester.md: every test asserts a real-world behaviour someone depends on —
no plumbing, no shape checks, no "did we store the dict" tests. The render
output is the only contract that matters; everything observable flows through
``WorldState.render()``.
"""

import json
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
    """Create a fresh WorldState instance (not the module singleton)."""
    return WorldState()


def _future_iso(minutes: int) -> str:
    return (utc_now() + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def _past_iso(minutes: int) -> str:
    return (utc_now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def _seed_telemetry(db, ctx: dict) -> None:
    """Persist a heartbeat-shaped dict into the telemetry table (flat key/value).

    Mirrors what ClientContextService.save() does on a /health POST so render
    tests can drive the full pipeline without spinning up the API.
    """
    def _flatten(payload, prefix=""):
        for key, value in payload.items():
            full = f"{prefix}{key}"
            if isinstance(value, dict) and value:
                yield from _flatten(value, prefix=f"{full}.")
            else:
                yield full, json.dumps(value)

    db.execute("DELETE FROM telemetry")
    db.executemany(
        "INSERT INTO telemetry (key, value) VALUES (?, ?)",
        list(_flatten(ctx)),
    )
    db.commit()


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
# bg_process section
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRenderBgProcess:
    def test_recent_subagent_appears_with_last_update(self, db):
        # A row from ~2 minutes ago must surface with a formatted "Xm ago" timestamp,
        # rendered as a [bg_process(...)] line (not a bullet).
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('subagent', 'assistant', 'Researching hotels in Valletta', ?)",
            ((utc_now() - timedelta(seconds=125)).strftime("%Y-%m-%d %H:%M:%S"),),
        )
        db.commit()

        result = _fresh().render()
        line = next(ln for ln in result.splitlines() if "[bg_process(" in ln)
        assert not line.startswith("*")
        assert "last_update:2m ago" in line
        assert "Researching hotels in Valletta" in line

    def test_older_than_24h_excluded_and_other_channels_ignored(self, db):
        # Two rows that must NOT surface: a subagent row >24h old, and a
        # recent row on a non-subagent channel.
        old_iso = _past_iso(1500)  # >24h
        recent_iso = (utc_now() - timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S")
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('subagent', 'assistant', 'Old pursuit', ?)",
            (old_iso,),
        )
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('user', 'assistant', 'regular chat content', ?)",
            (recent_iso,),
        )
        db.commit()

        result = _fresh().render()
        assert "Old pursuit" not in result
        assert "regular chat content" not in result


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
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('subagent', 'assistant', 'Active goal', ?)",
            ((utc_now() - timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S"),),
        )
        db.commit()

        result = ws.render()
        assert result.startswith(_HEADER)
        idx_telemetry = result.index("[telemetry]")
        idx_schedule = result.index("[schedule]")
        idx_bg = result.index("[bg_process(")
        idx_signal = result.index("[signal:news]")
        assert idx_telemetry < idx_schedule < idx_bg < idx_signal


