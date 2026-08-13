"""Unit tests for the pure memory hygiene helpers and the model window queries.

Drives `pending_windows`, `render_listing`, `parse_window_bounds` on the real
implementation, and exercises `MemoryGraphRow.updated_in_window`,
`MemoryGraphRow.earliest_created_at`, `MemoryMapRow.generated_in_window`, and
`MemoryMapRow.earliest_created_at` on a seeded in-memory SQLite database
(provided by the ``db`` fixture from ``conftest``).

No network, no LLM calls, no class scaffolding — every function is pure and
testable in isolation.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from models.memory_graph import MemoryGraphRow
from models.memory_map import MemoryMapRow
from services.memory_hygiene_service import (
    parse_window_bounds,
    pending_windows,
    render_listing,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso(dt: datetime) -> str:
    """Return an ISO-8601 UTC string with the +00:00 offset (no space, 'T'
    between date and time)."""
    return dt.astimezone(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# pending_windows
# ---------------------------------------------------------------------------


class TestPendingWindows:
    """Tests for the pure pending_windows function."""

    def test_three_day_chain_is_contiguous_and_non_overlapping(
        self,
    ) -> None:
        """A 3-day catch-up chain: each start == previous end, no overlaps,
        first window anchored exactly at covered_end."""
        tz = timezone.utc
        # covered_end at 04:00 so the first boundary is the next day.
        covered_end = datetime(2026, 1, 1, 4, 0, 0, tzinfo=tz)
        # now_utc past the 3rd window end + grace, before the 4th window end + grace.
        now_utc = datetime(2026, 1, 4, 12, 0, 0, tzinfo=tz)

        windows = pending_windows(covered_end, now_utc, tz)
        assert len(windows) == 3

        # First window anchored at covered_end.
        assert windows[0][0] == covered_end

        # Contiguous: each start == previous end.
        for i in range(len(windows) - 1):
            assert windows[i + 1][0] == windows[i][1]

        # Non-overlapping: end <= next start.
        for i in range(len(windows) - 1):
            assert windows[i][1] <= windows[i + 1][0]

    def test_all_bounds_are_tz_aware_utc(self) -> None:
        """Every bound returned by pending_windows must be tz-aware with
        utcoffset == 0 (UTC)."""
        tz = timezone.utc
        covered_end = datetime(2026, 1, 1, 0, 0, 0, tzinfo=tz)
        now_utc = datetime(2026, 1, 3, 12, 0, 0, tzinfo=tz)

        windows = pending_windows(covered_end, now_utc, tz)
        for start, end in windows:
            assert start.tzinfo is not None
            assert start.utcoffset() == timedelta(0)
            assert end.tzinfo is not None
            assert end.utcoffset() == timedelta(0)

    def test_bounds_are_utc_even_for_non_utc_tz(self) -> None:
        """Boundaries computed in a non-UTC tz come back normalized to UTC —
        the bounds become lexicographic TEXT comparisons against UTC-stamped
        columns, so a local-offset isoformat would silently mis-window."""
        tz = ZoneInfo("Europe/Malta")
        # 2026-06-01 02:00 UTC == 04:00 CEST — exactly a local boundary.
        covered_end = datetime(2026, 6, 1, 2, 0, 0, tzinfo=timezone.utc)
        now_utc = datetime(2026, 6, 3, 12, 0, 0, tzinfo=timezone.utc)

        windows = pending_windows(covered_end, now_utc, tz)
        assert windows
        for start, end in windows:
            assert start.utcoffset() == timedelta(0)
            assert end.utcoffset() == timedelta(0)
            assert start.isoformat().endswith("+00:00")
            assert end.isoformat().endswith("+00:00")

    def test_currently_open_window_is_excluded(self) -> None:
        """The window whose grace has not yet expired is excluded."""
        tz = timezone.utc
        covered_end = datetime(2026, 1, 1, 0, 0, 0, tzinfo=tz)
        # The first window ends at 04:00. Set now to 2 min after — within the
        # 5-minute grace — so this (and every later) window is excluded.
        now_utc = datetime(2026, 1, 1, 4, 2, 0, tzinfo=tz)

        windows = pending_windows(covered_end, now_utc, tz)
        assert windows == []

    def test_grace_edge_excluded_at_4_minutes(self) -> None:
        """now = end + 4 min is within the 5-minute grace → window excluded."""
        tz = timezone.utc
        window_end = datetime(2026, 1, 2, 4, 0, 0, tzinfo=tz)
        # Grace is 5 minutes; 4 min in is still graced.
        now_utc = window_end + timedelta(minutes=4)
        covered_end = datetime(2026, 1, 1, 4, 0, 0, tzinfo=tz)

        windows = pending_windows(covered_end, now_utc, tz)
        assert windows == []

    def test_grace_edge_included_at_6_minutes(self) -> None:
        """now = end + 6 min exceeds the 5-minute grace → window included."""
        tz = timezone.utc
        window_end = datetime(2026, 1, 2, 4, 0, 0, tzinfo=tz)
        now_utc = window_end + timedelta(minutes=6)
        covered_end = datetime(2026, 1, 1, 4, 0, 0, tzinfo=tz)

        windows = pending_windows(covered_end, now_utc, tz)
        assert len(windows) == 1
        assert windows[0] == (datetime(2026, 1, 1, 4, 0, 0, tzinfo=tz), window_end)

    def test_spring_forward_yields_23h_window(self) -> None:
        """Europe/Malta spring-forward on 2026-03-29 gives a 23 h window."""
        tz = ZoneInfo("Europe/Malta")
        # Malta is EET (UTC+1) in winter and EEST (UTC+2) in summer.
        # Spring forward: last Sunday of March 2026 = 2026-03-29.
        #   At 01:00 UTC -> 03:00 local (02:00 local skipped).
        #   First 04:00 local after = 2026-03-29 03:00 UTC.
        covered_end = datetime(2026, 3, 28, 4, 0, 0, tzinfo=tz).astimezone(timezone.utc)
        window_end = datetime(2026, 3, 29, 3, 0, 0, tzinfo=timezone.utc)
        now_utc = window_end + timedelta(hours=1)

        windows = pending_windows(covered_end, now_utc, tz)
        assert len(windows) == 1
        start, end = windows[0]
        # Window duration should be 23 hours.
        duration = end - start
        assert duration == timedelta(hours=23)

    def test_fall_back_yields_25h_window(self) -> None:
        """Europe/Malta fall-back on 2026-10-25 gives a 25 h window."""
        tz = ZoneInfo("Europe/Malta")
        # Fall-back: last Sunday of October 2026 = 2026-10-25.
        #   At 00:00 UTC -> 01:00 local (02:00 local -> 01:00 local repeated).
        #   Before fall-back: 04:00 local = 03:00 UTC.
        #   After fall-back:  04:00 local = 04:00 UTC.
        # The window spans from the last 04:00 local before fall-back
        # (2026-10-24 04:00 EEST = 2026-10-24 02:00 UTC) to the first
        # 04:00 local after fall-back (2026-10-25 04:00 UTC).
        covered_end = datetime(2026, 10, 24, 4, 0, 0, tzinfo=tz).astimezone(timezone.utc)
        window_end = datetime(2026, 10, 25, 4, 0, 0, tzinfo=timezone.utc)
        now_utc = window_end + timedelta(hours=1)

        windows = pending_windows(covered_end, now_utc, tz)
        assert len(windows) == 1
        start, end = windows[0]
        duration = end - start
        assert duration == timedelta(hours=25)


# ---------------------------------------------------------------------------
# render_listing
# ---------------------------------------------------------------------------


class TestRenderListing:
    """Tests for the pure render_listing function."""

    def test_header_line_format(self) -> None:
        """Header line: '# Memories Generated/Updated — {%a %d %b %Y}' of
        window start in tz."""
        tz = ZoneInfo("Europe/Malta")
        window_start = datetime(2026, 5, 15, 4, 30, 0, tzinfo=timezone.utc)
        window_end = datetime(2026, 5, 16, 4, 0, 0, tzinfo=timezone.utc)
        result = render_listing(window_start, window_end, [], [], tz)
        lines = result.splitlines()
        header_date = window_start.astimezone(tz).strftime("%a %d %b %Y")
        assert lines[0] == f"# Memories Generated/Updated — {header_date}"

    def test_window_line_format(self) -> None:
        """Second line: 'window: {start.isoformat()} → {end.isoformat()}'."""
        tz = timezone.utc
        window_start = datetime(2026, 5, 15, 4, 0, 0, tzinfo=tz)
        window_end = datetime(2026, 5, 16, 4, 0, 0, tzinfo=tz)
        result = render_listing(window_start, window_end, [], [], tz)
        lines = result.splitlines()
        expected = f"window: {window_start.isoformat()} → {window_end.isoformat()}"
        assert lines[1] == expected

    def test_empty_graph_section(self) -> None:
        """When graph_rows is empty the section shows '(none this window)'."""
        tz = timezone.utc
        window_start = datetime(2026, 5, 15, 4, 0, 0, tzinfo=tz)
        window_end = datetime(2026, 5, 16, 4, 0, 0, tzinfo=tz)
        result = render_listing(window_start, window_end, [], [], tz)
        assert "(none this window)" in result

    def test_empty_map_section(self) -> None:
        """When map_rows is empty the section shows '(none this window)'."""
        tz = timezone.utc
        window_start = datetime(2026, 5, 15, 4, 0, 0, tzinfo=tz)
        window_end = datetime(2026, 5, 16, 4, 0, 0, tzinfo=tz)
        # Graph has a row, map is empty. Real model instances, unsaved — the
        # renderer only reads attributes.
        graph_rows = [MemoryGraphRow(subject="user.city", contents="Lisbon")]
        result = render_listing(window_start, window_end, graph_rows, [], tz)
        assert "(none this window)" in result

    def test_graph_rows_render_as_single_json_object(self) -> None:
        """Graph rows are rendered as ONE single-line JSON object
        {subject: contents}."""
        tz = timezone.utc
        window_start = datetime(2026, 5, 15, 4, 0, 0, tzinfo=tz)
        window_end = datetime(2026, 5, 16, 4, 0, 0, tzinfo=tz)

        graph_rows = [
            MemoryGraphRow(subject="user.city", contents="Lisbon"),
            MemoryGraphRow(subject="user.pet", contents="cat Tom"),
        ]
        result = render_listing(window_start, window_end, graph_rows, [], tz)
        lines = result.splitlines()
        # The graph rows section should have exactly one JSON line.
        json_line = None
        in_graph_section = False
        for line in lines:
            if line == "## Graph Memories":
                in_graph_section = True
                continue
            if line.startswith("## "):
                in_graph_section = False
                continue
            if in_graph_section and line.strip() and not line.startswith("**"):
                json_line = line
                break
        assert json_line is not None
        parsed = json.loads(json_line)
        assert parsed == {"user.city": "Lisbon", "user.pet": "cat Tom"}

    def test_map_rows_render_format(self) -> None:
        """Each map row renders as '[id {id} · {HH:MM in tz}] {contents}'."""
        tz = ZoneInfo("Europe/Malta")
        window_start = datetime(2026, 5, 15, 4, 0, 0, tzinfo=timezone.utc)
        window_end = datetime(2026, 5, 16, 4, 0, 0, tzinfo=timezone.utc)

        # 2026-05-15 09:30 UTC = 11:30 CEST (Europe/Malta).
        generated_at = datetime(2026, 5, 15, 9, 30, 0, tzinfo=timezone.utc).isoformat()
        map_rows = [MemoryMapRow(id=42, generated_at=generated_at, contents="went to the park")]
        result = render_listing(window_start, window_end, [], map_rows, tz)
        lines = result.splitlines()
        # Find the map row line.
        map_line = None
        for line in lines:
            if line.startswith("[id "):
                map_line = line
                break
        assert map_line is not None
        assert map_line == "[id 42 · 11:30] went to the park"

    def test_round_trip_parse_window_bounds(self) -> None:
        """parse_window_bounds(render_listing(s, e, rows, rows, tz)) == (s, e)."""
        tz = ZoneInfo("Europe/Malta")
        window_start = datetime(2026, 5, 15, 4, 0, 0, tzinfo=timezone.utc)
        window_end = datetime(2026, 5, 16, 4, 0, 0, tzinfo=timezone.utc)

        graph_rows = [MemoryGraphRow(subject="user.city", contents="Lisbon")]
        map_rows = [
            MemoryMapRow(id=1, generated_at=window_start.isoformat(), contents="first episode")
        ]

        rendered = render_listing(window_start, window_end, graph_rows, map_rows, tz)
        parsed = parse_window_bounds(rendered)
        assert parsed is not None
        parsed_start, parsed_end = parsed
        assert parsed_start == window_start
        assert parsed_end == window_end


# ---------------------------------------------------------------------------
# parse_window_bounds
# ---------------------------------------------------------------------------


class TestParseWindowBounds:
    """Tests for the pure parse_window_bounds function."""

    def test_no_window_line_returns_none(self) -> None:
        """Content with no 'window:' line returns None."""
        content = "some random text\nno window marker here"
        assert parse_window_bounds(content) is None

    def test_malformed_line_no_separator_returns_none(self) -> None:
        """A 'window:' line without ' → ' separator returns None."""
        content = "window: 2026-01-01T00:00:00+00:00 2026-01-02T00:00:00+00:00"
        assert parse_window_bounds(content) is None

    def test_unparseable_garbage_returns_none(self) -> None:
        """One side unparseable garbage → None."""
        content = "window: NOT-A-DATE → 2026-01-02T00:00:00+00:00"
        assert parse_window_bounds(content) is None

    def test_sentinel_shaped_timestamp_rejected(self) -> None:
        """One side '0001-01-01T00:00:00+00:00' is sentinel-shaped and must be
        rejected (returns None)."""
        content = "window: 0001-01-01T00:00:00+00:00 → 2026-01-02T00:00:00+00:00"
        assert parse_window_bounds(content) is None

    def test_valid_bounds_returned(self) -> None:
        """Valid 'window:' line returns the (start, end) pair."""
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        content = f"window: {start.isoformat()} → {end.isoformat()}"
        result = parse_window_bounds(content)
        assert result is not None
        parsed_start, parsed_end = result
        assert parsed_start == start
        assert parsed_end == end


# ---------------------------------------------------------------------------
# Model window queries
# ---------------------------------------------------------------------------


class TestMemoryGraphRowWindowQuery:
    """Tests for MemoryGraphRow.updated_in_window and earliest_created_at."""

    def test_half_open_interval_includes_start_excludes_end(
        self, db: sqlite3.Connection
    ) -> None:
        """A row stamped exactly at start IS returned; a row exactly at end
        is NOT — the interval is half-open [start, end)."""
        start = datetime(2026, 3, 1, 4, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 2, 4, 0, 0, tzinfo=timezone.utc)
        start_iso = _utc_iso(start)
        end_iso = _utc_iso(end)

        # Row exactly at start → must be included.
        row_start = MemoryGraphRow(subject="test.at_start", contents="boundary start")
        row_start.save()
        # Manually stamp last_updated_at to the boundary value.
        db.execute(
            "UPDATE memory_graph SET last_updated_at = ? WHERE subject = ?",
            (start_iso, "test.at_start"),
        )

        # Row exactly at end → must be excluded.
        row_end = MemoryGraphRow(subject="test.at_end", contents="boundary end")
        row_end.save()
        db.execute(
            "UPDATE memory_graph SET last_updated_at = ? WHERE subject = ?",
            (end_iso, "test.at_end"),
        )

        # Row inside the window → must be included.
        inside = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        row_inside = MemoryGraphRow(subject="test.inside", contents="middle")
        row_inside.save()
        db.execute(
            "UPDATE memory_graph SET last_updated_at = ? WHERE subject = ?",
            (_utc_iso(inside), "test.inside"),
        )

        db.commit()

        results = MemoryGraphRow.updated_in_window(start_iso, end_iso)
        subjects = {r.subject for r in results}
        assert "test.at_start" in subjects
        assert "test.inside" in subjects
        assert "test.at_end" not in subjects

    def test_results_ordered_ascending(self, db: sqlite3.Connection) -> None:
        """Results are ordered by last_updated_at ASC."""
        timestamps = [
            datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 1, 6, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 1, 8, 0, 0, tzinfo=timezone.utc),
        ]
        for ts in timestamps:
            row = MemoryGraphRow(subject=f"test.ordered.{ts.hour}", contents=f"content {ts.hour}")
            row.save()
            db.execute(
                "UPDATE memory_graph SET last_updated_at = ? WHERE subject = ?",
                (_utc_iso(ts), f"test.ordered.{ts.hour}"),
            )
        db.commit()

        start = datetime(2026, 3, 1, 4, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 2, 4, 0, 0, tzinfo=timezone.utc)
        results = MemoryGraphRow.updated_in_window(_utc_iso(start), _utc_iso(end))
        hours = [int(r.subject.split(".")[-1]) for r in results]
        assert hours == [6, 8, 10]

    def test_earliest_created_at_returns_none_on_empty_table(self) -> None:
        """On an empty table, earliest_created_at returns None."""
        # Wipe any existing rows to ensure clean state.
        MemoryGraphRow._bound_connection().execute("DELETE FROM memory_graph")
        MemoryGraphRow._bound_connection().commit()
        assert MemoryGraphRow.earliest_created_at() is None

    def test_earliest_created_at_returns_minimum_after_seeding(self, db: sqlite3.Connection) -> None:
        """After seeding, earliest_created_at returns the minimum created_at."""
        # Wipe table first.
        MemoryGraphRow._bound_connection().execute("DELETE FROM memory_graph")
        MemoryGraphRow._bound_connection().commit()

        ts1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 3, 1, 6, 0, 0, tzinfo=timezone.utc)
        ts3 = datetime(2026, 3, 1, 8, 0, 0, tzinfo=timezone.utc)
        for ts in (ts1, ts2, ts3):
            row = MemoryGraphRow(subject=f"test.min.{ts.hour}", contents=f"content {ts.hour}")
            row.save()
            db.execute(
                "UPDATE memory_graph SET created_at = ? WHERE subject = ?",
                (_utc_iso(ts), f"test.min.{ts.hour}"),
            )
        db.commit()

        earliest = MemoryGraphRow.earliest_created_at()
        assert earliest is not None
        assert earliest == _utc_iso(ts2)


class TestMemoryMapRowWindowQuery:
    """Tests for MemoryMapRow.generated_in_window and earliest_created_at."""

    def test_half_open_interval_includes_start_excludes_end(
        self, db: sqlite3.Connection
    ) -> None:
        """A row stamped exactly at start IS returned; a row exactly at end
        is NOT — the interval is half-open [start, end)."""
        start = datetime(2026, 3, 1, 4, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 2, 4, 0, 0, tzinfo=timezone.utc)
        start_iso = _utc_iso(start)
        end_iso = _utc_iso(end)

        # Row exactly at start → included.
        row_start = MemoryMapRow(contents="boundary start")
        row_start.save()
        db.execute(
            "UPDATE memory_map SET generated_at = ? WHERE id = ?",
            (start_iso, row_start.id),
        )

        # Row exactly at end → excluded.
        row_end = MemoryMapRow(contents="boundary end")
        row_end.save()
        db.execute(
            "UPDATE memory_map SET generated_at = ? WHERE id = ?",
            (end_iso, row_end.id),
        )

        # Row inside the window → included.
        inside = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        row_inside = MemoryMapRow(contents="middle")
        row_inside.save()
        db.execute(
            "UPDATE memory_map SET generated_at = ? WHERE id = ?",
            (_utc_iso(inside), row_inside.id),
        )

        db.commit()

        results = MemoryMapRow.generated_in_window(start_iso, end_iso)
        contents = {r.contents for r in results}
        assert "boundary start" in contents
        assert "middle" in contents
        assert "boundary end" not in contents

    def test_results_ordered_ascending(self, db: sqlite3.Connection) -> None:
        """Results are ordered by generated_at ASC."""
        timestamps = [
            datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 1, 6, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 1, 8, 0, 0, tzinfo=timezone.utc),
        ]
        rows: list[MemoryMapRow] = []
        for ts in timestamps:
            row = MemoryMapRow(contents=f"content {ts.hour}")
            row.save()
            rows.append(row)
            db.execute(
                "UPDATE memory_map SET generated_at = ? WHERE id = ?",
                (_utc_iso(ts), row.id),
            )
        db.commit()

        start = datetime(2026, 3, 1, 4, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 2, 4, 0, 0, tzinfo=timezone.utc)
        results = MemoryMapRow.generated_in_window(_utc_iso(start), _utc_iso(end))
        contents = [r.contents for r in results]
        assert contents == ["content 6", "content 8", "content 10"]

    def test_earliest_created_at_returns_none_on_empty_table(self) -> None:
        """On an empty table, earliest_created_at returns None."""
        MemoryMapRow._bound_connection().execute("DELETE FROM memory_map")
        MemoryMapRow._bound_connection().commit()
        assert MemoryMapRow.earliest_created_at() is None

    def test_earliest_created_at_returns_minimum_after_seeding(self, db: sqlite3.Connection) -> None:
        """After seeding, earliest_created_at returns the minimum created_at."""
        MemoryMapRow._bound_connection().execute("DELETE FROM memory_map")
        MemoryMapRow._bound_connection().commit()

        ts1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 3, 1, 6, 0, 0, tzinfo=timezone.utc)
        ts3 = datetime(2026, 3, 1, 8, 0, 0, tzinfo=timezone.utc)
        for ts in (ts1, ts2, ts3):
            row = MemoryMapRow(contents=f"content {ts.hour}")
            row.save()
            db.execute(
                "UPDATE memory_map SET created_at = ?, generated_at = ? WHERE id = ?",
                (_utc_iso(ts), _utc_iso(ts), row.id),
            )
        db.commit()

        earliest = MemoryMapRow.earliest_created_at()
        assert earliest is not None
        assert earliest == _utc_iso(ts2)


class TestProductionTimestampFormat:
    """Rows written through the models' production save paths carry
    'T'-shaped isoformat timestamps (no space between date and time) in
    last_updated_at / generated_at / created_at."""

    def test_memory_graph_row_carry_t_shaped_timestamps(self, db: sqlite3.Connection) -> None:
        """MemoryGraphRow.save() produces T-shaped ISO timestamps."""
        row = MemoryGraphRow(subject="test.timestamp.graph", contents="check format")
        row.save()
        db.commit()

        cur = db.execute(
            "SELECT last_updated_at, created_at FROM memory_graph WHERE subject = ?",
            ("test.timestamp.graph",),
        )
        recorded = cur.fetchone()
        assert recorded is not None
        last_updated, created_at = recorded
        # Both must contain 'T' (not a space) between date and time.
        assert "T" in last_updated
        assert " " not in last_updated
        assert "T" in created_at
        assert " " not in created_at

    def test_memory_map_row_carry_t_shaped_timestamps(self, db: sqlite3.Connection) -> None:
        """MemoryMapRow.save() produces T-shaped ISO timestamps."""
        row = MemoryMapRow(contents="check format map")
        row.save()
        db.commit()

        cur = db.execute(
            "SELECT generated_at, created_at FROM memory_map WHERE id = ?",
            (row.id,),
        )
        recorded = cur.fetchone()
        assert recorded is not None
        generated_at, created_at = recorded
        assert "T" in generated_at
        assert " " not in generated_at
        assert "T" in created_at
        assert " " not in created_at
