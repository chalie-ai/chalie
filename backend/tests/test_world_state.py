"""
Feature tests for WorldState — in-process singleton.

All tests use the real production WorldState class with real in-memory SQLite
built from schema.sql (via the conftest `db` fixture). Zero mocks of our own code.

Each test isolates WorldState state by using a fresh WorldState instance
(not the module singleton) to avoid cross-test contamination.
"""

import time as _time
import uuid
from datetime import timedelta

import pytest

from services.time_formatter_service import TimeFormatterService
from services.time_utils import utc_now
from services.world_state import WorldState, _fetch_schedule_rows, _fetch_bg_process_rows

_HEADER = "### Background Telemetry,Processes & Signals"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh() -> WorldState:
    """Create a fresh WorldState instance (not the module singleton)."""
    return WorldState()


def _future_iso(minutes: int = 60) -> str:
    return (utc_now() + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def _past_iso(minutes: int = 60) -> str:
    return (utc_now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def _recent_iso(seconds: int = 30) -> str:
    return (utc_now() - timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# set / get roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSetGet:
    def test_set_returns_self_for_chaining(self):
        ws = _fresh()
        result = ws.set("foo", {"x": 1})
        assert result is ws

    def test_chainable_multiple_types(self):
        ws = _fresh()
        ws.set("a", {"v": 1}).set("b", {"v": 2})
        assert ws.get("a") == {"v": 1}
        assert ws.get("b") == {"v": 2}

    def test_get_returns_stored_dict(self):
        ws = _fresh()
        ws.set("telemetry", {"place": "Valletta"})
        assert ws.get("telemetry") == {"place": "Valletta"}

    def test_get_unset_type_returns_empty_dict(self):
        ws = _fresh()
        assert ws.get("nonexistent") == {}

    def test_set_overwrites_prior_value(self):
        ws = _fresh()
        ws.set("telemetry", {"place": "Valletta"})
        ws.set("telemetry", {"place": "Sliema"})
        assert ws.get("telemetry")["place"] == "Sliema"

    def test_arbitrary_type_keys_supported(self):
        ws = _fresh()
        ws.set("custom_type_x", {"data": 42})
        assert ws.get("custom_type_x") == {"data": 42}

    def test_get_returns_copy_not_reference(self):
        ws = _fresh()
        ws.set("foo", {"x": 1})
        retrieved = ws.get("foo")
        retrieved["x"] = 999
        assert ws.get("foo")["x"] == 1


# ---------------------------------------------------------------------------
# push_signal
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPushSignal:
    def test_push_signal_adds_entry_keyed_by_source(self):
        ws = _fresh()
        ws.push_signal("news", "Malta heatwave warning")
        signals = ws.get("signals")
        assert "news" in signals
        assert signals["news"]["label"] == "Malta heatwave warning"

    def test_push_signal_second_push_same_source_overwrites(self):
        ws = _fresh()
        ws.push_signal("news", "first headline")
        ws.push_signal("news", "second headline")
        signals = ws.get("signals")
        assert signals["news"]["label"] == "second headline"

    def test_push_signal_multiple_sources_coexist(self):
        ws = _fresh()
        ws.push_signal("news", "headline")
        ws.push_signal("inbox", "you have 3 emails")
        signals = ws.get("signals")
        assert "news" in signals
        assert "inbox" in signals

    def test_push_signal_expired_entry_pruned_on_get(self):
        ws = _fresh()
        # Push a signal with TTL 0 → immediately expired
        ws.push_signal("stale", "old news", ttl=0)
        _time.sleep(0.01)  # ensure it's past expiry
        signals = ws.get("signals")
        assert "stale" not in signals

    def test_push_signal_non_expired_entry_survives(self):
        ws = _fresh()
        ws.push_signal("live", "live signal", ttl=3600)
        signals = ws.get("signals")
        assert "live" in signals

    def test_push_signal_expires_at_stored(self):
        ws = _fresh()
        before = _time.time()
        ws.push_signal("src", "label", ttl=60)
        signals = ws.get("signals")
        assert signals["src"]["expires_at"] > before
        assert signals["src"]["expires_at"] < before + 61


# ---------------------------------------------------------------------------
# render() — empty state
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRenderEmpty:
    def test_render_empty_state_returns_empty_string(self, db):
        ws = _fresh()
        result = ws.render()
        assert result == ""


# ---------------------------------------------------------------------------
# render() — telemetry section
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRenderTelemetry:
    def test_render_with_only_telemetry_includes_header_and_section(self, db):
        ws = _fresh()
        ws.set("telemetry", {
            "local_time": "10:47",
            "utc_offset": "+02:00",
            "location": "Valletta, MT",
        })
        result = ws.render()
        assert result.startswith(_HEADER)
        assert "[telemetry]" in result
        assert "* **user**;" in result

    def test_render_telemetry_user_line_with_full_fields(self, db):
        ws = _fresh()
        ws.set("telemetry", {
            "local_time": "10:47",
            "utc_offset": "+02:00",
            "location": "Valletta",
            "mobility": "stationary",
        })
        result = ws.render()
        assert "time:10:47+02:00" in result
        assert "location:Valletta" in result
        assert "mobility:stationary" in result

    def test_render_telemetry_no_battery_when_absent(self, db):
        ws = _fresh()
        ws.set("telemetry", {
            "local_time": "10:00",
            "location": "Home",
            "device": {"name": "iPhone"},
        })
        result = ws.render()
        assert "battery" not in result
        assert "name:iPhone" in result

    def test_render_telemetry_no_mobility_when_absent(self, db):
        ws = _fresh()
        ws.set("telemetry", {
            "local_time": "09:00",
            "location": "Office",
        })
        result = ws.render()
        assert "mobility" not in result

    def test_render_telemetry_no_schedule_or_signals_section(self, db):
        ws = _fresh()
        ws.set("telemetry", {
            "local_time": "09:00",
            "location": "Sliema",
        })
        result = ws.render()
        assert "[schedule]" not in result
        assert "[signal:" not in result
        assert "[bg_process" not in result

    def test_render_telemetry_no_space_after_colon_or_comma(self, db):
        ws = _fresh()
        ws.set("telemetry", {
            "local_time": "10:47",
            "utc_offset": "+02:00",
            "location": "Valletta",
            "mobility": "stationary",
        })
        result = ws.render()
        # Find the user bullet
        lines = result.splitlines()
        user_line = next((l for l in lines if "**user**" in l), None)
        assert user_line is not None
        fields_part = user_line.split(";", 1)[1] if ";" in user_line else ""
        # No ': ' or ', ' in the fields part
        assert ": " not in fields_part
        assert ", " not in fields_part

    def test_render_telemetry_device_line_with_battery(self, db):
        ws = _fresh()
        ws.set("telemetry", {
            "local_time": "09:00",
            "device": {"name": "MacBook", "battery": "82%"},
        })
        result = ws.render()
        assert "* **device**;" in result
        assert "name:MacBook" in result
        assert "battery:82%" in result


# ---------------------------------------------------------------------------
# render() — signals section
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRenderSignals:
    def test_render_with_only_signals_has_no_other_sections(self, db):
        ws = _fresh()
        ws.push_signal("news", "Malta heatwave")
        result = ws.render()
        assert "[telemetry]" not in result
        assert "[schedule]" not in result
        assert "[bg_process" not in result
        assert "[signal:news] Malta heatwave" in result

    def test_render_signals_sorted_by_source_key(self, db):
        ws = _fresh()
        ws.push_signal("zzz", "last")
        ws.push_signal("aaa", "first")
        ws.push_signal("mmm", "middle")
        result = ws.render()
        idx_aaa = result.index("[signal:aaa]")
        idx_mmm = result.index("[signal:mmm]")
        idx_zzz = result.index("[signal:zzz]")
        assert idx_aaa < idx_mmm < idx_zzz

    def test_render_signals_format_no_bullet(self, db):
        ws = _fresh()
        ws.push_signal("news", "headline")
        result = ws.render()
        lines = result.splitlines()
        signal_line = next((l for l in lines if "[signal:news]" in l), None)
        assert signal_line is not None
        assert not signal_line.startswith("*")

    def test_render_signals_space_between_bracket_and_label(self, db):
        ws = _fresh()
        ws.push_signal("src", "the label")
        result = ws.render()
        assert "[signal:src] the label" in result


# ---------------------------------------------------------------------------
# render() — schedule section (requires real DB)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRenderSchedule:
    def test_future_item_appears_with_due_in(self, db):
        item_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO scheduled_items (id, message, due_at, status, hidden) "
            "VALUES (?, ?, ?, 'pending', 0)",
            (item_id, "Call Mum", _future_iso(120)),
        )
        db.commit()

        ws = _fresh()
        result = ws.render()
        assert "[schedule]" in result
        assert "Call Mum" in result
        assert "due-in:" in result

    def test_past_due_item_renders_with_ago_suffix(self, db):
        item_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO scheduled_items (id, message, due_at, status, hidden) "
            "VALUES (?, ?, ?, 'pending', 0)",
            (item_id, "Overdue task", _past_iso(30)),
        )
        db.commit()

        ws = _fresh()
        result = ws.render()
        assert "Overdue task" in result
        assert "due-in:" in result
        assert "ago" in result

    def test_recurring_item_includes_repeats_field(self, db):
        item_id = str(uuid.uuid4())
        # recurrence is stored as seconds in a string
        db.execute(
            "INSERT INTO scheduled_items (id, message, due_at, status, hidden, recurrence) "
            "VALUES (?, ?, ?, 'pending', 0, ?)",
            (item_id, "Daily standup", _future_iso(60), "86400"),
        )
        db.commit()

        ws = _fresh()
        result = ws.render()
        assert "Daily standup" in result
        assert "repeats:every" in result
        assert "1d 0h" in result

    def test_recently_fired_item_appears_within_24h(self, db):
        item_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO scheduled_items (id, message, due_at, status, hidden, last_fired_at) "
            "VALUES (?, ?, ?, 'fired', 0, ?)",
            (item_id, "Morning alarm", _past_iso(1), _past_iso(1)),
        )
        db.commit()

        ws = _fresh()
        result = ws.render()
        assert "Morning alarm" in result

    def test_hidden_item_does_not_appear(self, db):
        item_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO scheduled_items (id, message, due_at, status, hidden) "
            "VALUES (?, ?, ?, 'pending', 1)",
            (item_id, "Secret task", _future_iso(60)),
        )
        db.commit()

        ws = _fresh()
        result = ws.render()
        assert "Secret task" not in result

    def test_non_recurring_item_no_repeats_field(self, db):
        item_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO scheduled_items (id, message, due_at, status, hidden) "
            "VALUES (?, ?, ?, 'pending', 0)",
            (item_id, "One-off task", _future_iso(30)),
        )
        db.commit()

        ws = _fresh()
        result = ws.render()
        assert "One-off task" in result
        assert "repeats" not in result

    def test_fired_item_older_than_24h_does_not_appear(self, db):
        item_id = str(uuid.uuid4())
        old_fired = _past_iso(minutes=1500)  # >24h ago
        db.execute(
            "INSERT INTO scheduled_items (id, message, due_at, status, hidden, last_fired_at) "
            "VALUES (?, ?, ?, 'fired', 0, ?)",
            (item_id, "Old alarm", old_fired, old_fired),
        )
        db.commit()

        ws = _fresh()
        result = ws.render()
        assert "Old alarm" not in result


# ---------------------------------------------------------------------------
# render() — bg_process section (requires real DB)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRenderBgProcess:
    def test_recent_goal_pursuit_row_appears(self, db):
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('goal_pursuit', 'assistant', 'Researching hotels in Valletta', ?)",
            (_recent_iso(seconds=120),),
        )
        db.commit()

        ws = _fresh()
        result = ws.render()
        assert "[bg_process(" in result
        assert "last_update:" in result
        assert "Researching hotels in Valletta" in result

    def test_bg_process_format_no_bullet(self, db):
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('goal_pursuit', 'assistant', 'Active pursuit', ?)",
            (_recent_iso(seconds=60),),
        )
        db.commit()

        ws = _fresh()
        result = ws.render()
        lines = result.splitlines()
        bg_line = next((l for l in lines if "[bg_process(" in l), None)
        assert bg_line is not None
        assert not bg_line.startswith("*")

    def test_bg_process_older_than_24h_excluded(self, db):
        old = _past_iso(minutes=1500)  # >24h
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('goal_pursuit', 'assistant', 'Old pursuit', ?)",
            (old,),
        )
        db.commit()

        ws = _fresh()
        result = ws.render()
        assert "Old pursuit" not in result

    def test_non_goal_pursuit_channel_not_included(self, db):
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('user', 'assistant', 'regular chat content', ?)",
            (_recent_iso(seconds=60),),
        )
        db.commit()

        ws = _fresh()
        result = ws.render()
        assert "regular chat content" not in result

    def test_last_update_field_uses_time_formatter(self, db):
        # Seed a row from exactly ~2 minutes ago (should render as '2m ago')
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('goal_pursuit', 'assistant', 'Finding flights', ?)",
            (_recent_iso(seconds=125),),
        )
        db.commit()

        ws = _fresh()
        result = ws.render()
        # Should contain 'last_update:2m ago' (125s ≈ 2m)
        assert "last_update:2m ago" in result


# ---------------------------------------------------------------------------
# render() — full-mix section ordering
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRenderFullMix:
    def test_section_order_telemetry_schedule_bgprocess_signals(self, db):
        """Sections must appear in fixed order: telemetry → schedule → bg_process → signals."""
        # Seed all four sections
        ws = _fresh()
        ws.set("telemetry", {"local_time": "10:00", "location": "Malta"})
        ws.push_signal("news", "heatwave")

        item_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO scheduled_items (id, message, due_at, status, hidden) "
            "VALUES (?, ?, ?, 'pending', 0)",
            (item_id, "Team meeting", _future_iso(60)),
        )
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('goal_pursuit', 'assistant', 'Active goal', ?)",
            (_recent_iso(seconds=30),),
        )
        db.commit()

        result = ws.render()

        idx_telemetry = result.index("[telemetry]")
        idx_schedule = result.index("[schedule]")
        idx_bg = result.index("[bg_process(")
        idx_signal = result.index("[signal:news]")

        assert idx_telemetry < idx_schedule < idx_bg < idx_signal

    def test_render_starts_with_exact_header(self, db):
        ws = _fresh()
        ws.set("telemetry", {"local_time": "08:00", "location": "Home"})
        result = ws.render()
        assert result.startswith(_HEADER)

    def test_empty_sections_fully_omitted(self, db):
        """Only non-empty sections appear — no phantom headers."""
        ws = _fresh()
        ws.set("telemetry", {"local_time": "08:00", "location": "Office"})
        # No schedule, no bg_process, no signals
        result = ws.render()
        assert "[schedule]" not in result
        assert "[bg_process" not in result
        assert "[signal:" not in result

    def test_render_has_blank_line_after_telemetry_header(self, db):
        ws = _fresh()
        ws.set("telemetry", {"local_time": "08:00", "location": "Office"})
        result = ws.render()
        lines = result.splitlines()
        telem_idx = next(i for i, l in enumerate(lines) if "[telemetry]" in l)
        # Next line after [telemetry] must be blank
        assert lines[telem_idx + 1] == ""

    def test_render_schedule_has_blank_line_after_header(self, db):
        item_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO scheduled_items (id, message, due_at, status, hidden) "
            "VALUES (?, ?, ?, 'pending', 0)",
            (item_id, "Doctor appointment", _future_iso(60)),
        )
        db.commit()

        ws = _fresh()
        result = ws.render()
        lines = result.splitlines()
        sched_idx = next(i for i, l in enumerate(lines) if "[schedule]" in l)
        assert lines[sched_idx + 1] == ""

    def test_bullet_uses_star_single_space(self, db):
        ws = _fresh()
        ws.set("telemetry", {"local_time": "09:00", "location": "Home"})
        result = ws.render()
        lines = result.splitlines()
        bullet_lines = [l for l in lines if l.startswith("* ")]
        assert len(bullet_lines) >= 1
        for line in bullet_lines:
            assert line.startswith("* "), f"Expected '* ' prefix, got: {line!r}"

    def test_semicolon_separates_entity_from_fields(self, db):
        ws = _fresh()
        ws.set("telemetry", {
            "local_time": "10:00",
            "location": "Sliema",
        })
        result = ws.render()
        user_line = next((l for l in result.splitlines() if "**user**" in l), None)
        assert user_line is not None
        assert "**user**;" in user_line
