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
    def test_telemetry_groups_by_top_level_prefix_and_surfaces_every_fe_key(self, db):
        # End-to-end shape: whatever the FE persists into the telemetry table
        # must reach the rendered block, grouped by top-level prefix. The
        # renderer never knows the schema in advance — that's the whole point
        # of the refactor.
        _seed_telemetry(db, {
            "local_time": "10:47",
            "utc_offset": "+02:00",
            "location_name": "Valletta, Malta",
            "device": {"name": "MacBook", "battery": 82, "os": "macOS"},
            "behavioral": {"focus_state": "deep", "tab_count": 7},
        })

        result = _fresh().render()

        assert result.startswith(_HEADER)
        assert "[telemetry]" in result

        # Top-level scalars collapse into a single **user** bullet.
        user_line = next(ln for ln in result.splitlines() if ln.startswith("* **user**"))
        assert "local_time:10:47" in user_line
        assert "utc_offset:+02:00" in user_line
        assert "location_name:Valletta, Malta" in user_line

        # Each nested dict becomes its own bullet, alphabetically sorted.
        device_line = next(ln for ln in result.splitlines() if ln.startswith("* **device**"))
        assert "name:MacBook" in device_line
        assert "battery:82" in device_line
        assert "os:macOS" in device_line

        behavioral_line = next(ln for ln in result.splitlines() if ln.startswith("* **behavioral**"))
        assert "focus_state:deep" in behavioral_line
        assert "tab_count:7" in behavioral_line

        # Bookkeeping keys must never leak.
        assert "saved_at" not in result
        assert "_location_name_stale" not in result


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

    def test_past_due_pending_renders_with_ago_suffix(self, db):
        _seed_pending(db, "Overdue task", due_minutes_ahead=-30)
        db.commit()

        result = _fresh().render()
        bullet = next(ln for ln in result.splitlines() if "Overdue task" in ln)
        assert "due-in:" in bullet
        assert "ago" in bullet

    def test_hidden_items_excluded(self, db):
        db.execute(
            "INSERT INTO scheduled_items (id, message, due_at, status, hidden) "
            "VALUES (?, ?, ?, 'pending', 1)",
            (str(uuid.uuid4()), "Secret task", _future_iso(60)),
        )
        db.commit()

        assert "Secret task" not in _fresh().render()

    def test_fired_older_than_6h_excluded(self, db):
        _seed_fired(db, "Old alarm", fired_minutes_ago=400)  # >6h
        db.commit()

        assert "Old alarm" not in _fresh().render()

    def test_repeated_fires_collapse_to_most_recent_within_6h(self, db):
        # A recurring job fires every hour for 5 hours. The schedule view must
        # surface ONE bullet for the most-recent fire only — not a wall of dupes.
        for minutes_ago in (300, 240, 180, 120, 30):
            _seed_fired(db, "Mail sync", fired_minutes_ago=minutes_ago)
        db.commit()

        bullets = [ln for ln in _fresh().render().splitlines() if ln.startswith("* Mail sync")]
        assert len(bullets) == 1, f"Expected 1 bullet for repeated fires, got {len(bullets)}: {bullets!r}"
        assert "last-fired:30m" in bullets[0]
        assert "5h" not in bullets[0]

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
    def test_recent_goal_pursuit_appears_with_last_update(self, db):
        # A row from ~2 minutes ago must surface with a formatted "Xm ago" timestamp,
        # rendered as a [bg_process(...)] line (not a bullet).
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('goal_pursuit', 'assistant', 'Researching hotels in Valletta', ?)",
            ((utc_now() - timedelta(seconds=125)).strftime("%Y-%m-%d %H:%M:%S"),),
        )
        db.commit()

        result = _fresh().render()
        line = next(ln for ln in result.splitlines() if "[bg_process(" in ln)
        assert not line.startswith("*")
        assert "last_update:2m ago" in line
        assert "Researching hotels in Valletta" in line

    def test_older_than_24h_excluded_and_other_channels_ignored(self, db):
        # Two rows that must NOT surface: a goal_pursuit row >24h old, and a
        # recent row on a non-goal_pursuit channel.
        old_iso = _past_iso(1500)  # >24h
        recent_iso = (utc_now() - timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S")
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('goal_pursuit', 'assistant', 'Old pursuit', ?)",
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

    def test_long_content_is_truncated_with_ellipsis(self, db):
        long_content = "word " * 120  # 600 chars
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('goal_pursuit', 'assistant', ?, ?)",
            (long_content, (utc_now() - timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S")),
        )
        db.commit()

        line = next(ln for ln in _fresh().render().splitlines() if "[bg_process(" in ln)
        assert line.endswith("…")
        assert len(line) <= 240  # 200 content cap + ~30 char prefix


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
            "VALUES ('goal_pursuit', 'assistant', 'Active goal', ?)",
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

    def test_empty_sections_are_fully_omitted(self, db):
        # Only telemetry seeded — schedule, bg_process, signals headers must be absent.
        _seed_telemetry(db, {"local_time": "08:00", "location_name": "Office"})
        result = _fresh().render()
        assert "[schedule]" not in result
        assert "[bg_process" not in result
        assert "[signal:" not in result

    def test_schedule_header_has_no_blank_line_before_first_bullet(self, db):
        # Regression: the [schedule] header used to be followed by a blank line,
        # producing visual gap before the first bullet. The bullet must come next.
        _seed_pending(db, "Doctor appointment", due_minutes_ahead=60)
        db.commit()

        lines = _fresh().render().splitlines()
        sched_idx = next(i for i, ln in enumerate(lines) if "[schedule]" in ln)
        assert lines[sched_idx + 1].startswith("* "), (
            f"Expected first schedule bullet immediately after header, got: {lines[sched_idx + 1]!r}"
        )
