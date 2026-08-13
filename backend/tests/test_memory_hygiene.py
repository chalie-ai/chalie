"""Tests for memory hygiene — the pure helpers, and the whole ``tick()`` pass.

The first half drives `pending_windows`, `render_listing`, `parse_window_bounds`
on the real implementation, and exercises `MemoryGraphRow.updated_in_window`,
`MemoryGraphRow.earliest_created_at`, `MemoryMapRow.generated_in_window`, and
`MemoryMapRow.earliest_created_at` on a seeded in-memory SQLite database
(provided by the ``db`` fixture from ``conftest``). Every function there is
pure and testable in isolation — no network, no LLM calls, no scaffolding.

The second half (from ``TestFirstRunChainStart`` down) drives
``MemoryHygieneService.tick()`` end to end: the real service, the real
``MessageProcessor``, the real prompt/transcript/dispatch spine and the real
SQLite database, with only the LLM network boundary
(``services.provider_service.build_client``) swapped for a scripted double —
the same seam ``test_memory_step.py`` and ``test_context_limit_recovery.py``
use. The double is keyed on the LISTING CONTENT of each request, never on its
tool set: every hygiene pass carries the same four memory tools, so the tools
cannot tell one window's pass from another's.

Time is pinned by patching ``utc_now`` in the service's OWN namespace (it
imports the symbol, so that is the binding the window chain reads), and the
timezone falls back to UTC under test (no heartbeat), which puts every hygiene
boundary at 04:00 UTC. Nothing sleeps and nothing reads the wall clock.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from abilities.recall import Recall
from configs.channels.memory_hygiene import MemoryHygieneConfig
from configs.enums.channels import Channel
from controllers.message_processor import MessageProcessor
from exceptions import ContextLimit
from models.machine_state import MachineStateRow
from models.memory_graph import MemoryGraphRow
from models.memory_map import MemoryMapRow
from models.provider_request import ProviderRequest
from models.provider_response import ProviderResponse
from models.turn_execution import TurnExecution
from services import memory_hygiene_service
from services.memory_hygiene_service import (
    MemoryHygieneService,
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


# ===========================================================================
# MemoryHygieneService.tick() — end-to-end
# ===========================================================================

#: The LLM network boundary — the only substitution any test below makes.
_BUILD_CLIENT = "services.provider_service.build_client"

#: The machine-state key the stable hygiene turn_id is persisted under.
_TURN_ID_KEY = "memory_hygiene:turn_id"

#: Distinctive row contents, so the scripted double can tell which window's
#: listing — or which half of a split one — a request is carrying. Keying on
#: content is the only option available: every pass carries the same four
#: memory tools and the same system prompt.
_ALPHA = "MARKER_ALPHA"
_BETA = "MARKER_BETA"

#: The pinned clock. The locale service falls back to UTC under test (no
#: heartbeat), so every hygiene boundary lands at 04:00 UTC.
_NOW = datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc)
#: ALPHA sits in the first closed window, BETA in the second.
_ALPHA_AT = datetime(2026, 3, 3, 12, 0, 0, tzinfo=timezone.utc)
_BETA_AT = datetime(2026, 3, 4, 12, 0, 0, tzinfo=timezone.utc)
_BOUNDARY_1 = datetime(2026, 3, 4, 4, 0, 0, tzinfo=timezone.utc)
_BOUNDARY_2 = datetime(2026, 3, 5, 4, 0, 0, tzinfo=timezone.utc)


def _window_line(start: datetime, end: datetime) -> str:
    """The listing's coverage line, exactly as ``render_listing`` writes it."""
    return f"window: {start.isoformat()} → {end.isoformat()}"


def _seed_graph_row(db: sqlite3.Connection, subject: str, contents: str, at: datetime) -> None:
    """Write a real graph row through the model's own save path, then backdate
    its timestamps — the same seed shape the window-query tests above use."""
    MemoryGraphRow(subject=subject, contents=contents).save()
    db.execute(
        "UPDATE memory_graph SET created_at = ?, last_updated_at = ? WHERE subject = ?",
        (_utc_iso(at), _utc_iso(at), subject),
    )
    db.commit()


# ── the scripted provider ───────────────────────────────────────────────────


def _body(request: ProviderRequest) -> str:
    """Every message's content, concatenated — the payload as one searchable
    string."""
    return "\n".join(str(message.get("content", "")) for message in request.messages)


def _listing_of(request: ProviderRequest) -> str:
    """The CURRENT window's listing, sliced off the request body at its
    ``window:`` line, or ``""`` when this request carries no listing.

    A hygiene prompt appends this pass's listing verbatim, so its ``window:``
    line is the only one that survives at the start of a line. A listing
    carried forward through ``## Previous Messages`` — or through the
    hand-over / compaction / thread-gist passes — is flattened onto a single
    ``[ts] role: …`` line, so it can never be mistaken for this pass's own.
    That is the same anchoring ``parse_window_bounds`` uses in production.
    """
    lines = _body(request).splitlines()
    for index, line in enumerate(lines):
        if line.startswith("window: "):
            return "\n".join(lines[index:])
    return ""


def _window_of(request: ProviderRequest) -> str:
    """The ``window:`` line of the listing this request carries, or ``""``."""
    listing = _listing_of(request)
    return listing.splitlines()[0] if listing else ""


def _listings(provider: "_ScriptedProvider") -> list[ProviderRequest]:
    """Only the requests that carry a window listing — i.e. the hygiene passes,
    with the hand-over, compaction and thread-gist sub-passes filtered out."""
    return [request for request in provider.requests if _listing_of(request)]


@dataclass(frozen=True)
class _Payload:
    """What a lane matches on: the whole request body, and this pass's listing."""

    body: str
    listing: str


@dataclass(frozen=True)
class _Say:
    """Script entry: answer with prose, which settles the pass."""

    text: str


@dataclass(frozen=True)
class _Call:
    """Script entry: answer with one tool call."""

    name: str
    input: dict[str, str]


@dataclass(frozen=True)
class _Crash:
    """Script entry: fail the send outright, as an unreachable provider would."""

    message: str


@dataclass(frozen=True)
class _OverLimit:
    """Script entry: the provider rejects the payload for length. Raised with no
    turn attached — a client has none; ``ProviderService.send`` attaches its
    own before re-raising."""

    message: str


_Entry = _Say | _Call | _Crash | _OverLimit


def _listing_carries(*present: str, without: tuple[str, ...] = ()) -> Callable[[_Payload], bool]:
    """A matcher over THIS pass's listing only.

    Deliberately not over the whole body: a later window's request carries the
    earlier window's listing inside ``## Previous Messages``, so a body-level
    match would route the second pass into the first pass's lane.
    """
    def match(payload: _Payload) -> bool:
        return (
            bool(payload.listing)
            and all(needle in payload.listing for needle in present)
            and not any(needle in payload.listing for needle in without)
        )
    return match


class _Lane:
    """One keyed script: requests whose payload satisfies ``matches`` are served
    from ``entries``, one entry per call, the last entry repeating."""

    def __init__(self, matches: Callable[[_Payload], bool], entries: list[_Entry]) -> None:
        self.matches = matches
        self._entries = entries
        self.served = 0

    def take(self) -> _Entry:
        entry = self._entries[min(self.served, len(self._entries) - 1)]
        self.served += 1
        return entry


class _ScriptedProvider:
    """A scripted double at the ``build_client`` seam.

    Each request is served by the first lane whose matcher accepts it; anything
    unmatched — the hand-over pass, the history compactor, the thread-gist
    daemon — is served ``default``. ``tick()`` leaves several turn threads
    running at once, so the request log is written under a lock.
    """

    def __init__(
        self, lanes: list[_Lane] | None = None, default: _Entry | None = None,
    ) -> None:
        self._lanes = lanes if lanes is not None else []
        self._default = default if default is not None else _Say("- no changes")
        self._lock = threading.Lock()
        self.requests: list[ProviderRequest] = []

    def get_context_limit(self) -> int:
        return 120_000

    def send(self, request: ProviderRequest) -> ProviderResponse:
        payload = _Payload(body=_body(request), listing=_listing_of(request))
        with self._lock:
            self.requests.append(request)
            lane = next((lane for lane in self._lanes if lane.matches(payload)), None)
            entry = lane.take() if lane is not None else self._default
        if isinstance(entry, _Crash):
            raise RuntimeError(entry.message)
        if isinstance(entry, _OverLimit):
            raise ContextLimit(entry.message)
        if isinstance(entry, _Call):
            return ProviderResponse(
                # Non-empty prose alongside the call: an empty completion would
                # trigger the empty-completion steering path instead.
                text="calling tool",
                model="scripted",
                tool_calls=[{"id": "tc1", "name": entry.name, "input": entry.input}],
                tokens_input=1000,
            )
        return ProviderResponse(text=entry.text, model="scripted", tokens_input=1000)


class _Clock:
    """The pinned ``now`` for the window chain.

    ``memory_hygiene_service`` imports ``utc_now`` into its own namespace, so
    that binding — not ``services.time_utils`` — is what a tick reads. Mutating
    ``now`` between ticks is how a test advances the calendar without sleeping.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, now: datetime) -> None:
        self.now = now
        monkeypatch.setattr(memory_hygiene_service, "utc_now", lambda: self.now)


def _drain_turns(timeout_s: float = 30.0) -> None:
    """Join every turn thread a tick left running.

    ``MemoryHygieneService._run_turn`` joins each pass before the next fires,
    but a settled turn spawns daemons of its own (``thread-gist``,
    ``skill-suggest``) which start turns of their own — and those would
    outlive both the provider patch and this test's database. Loops rather
    than joins once for exactly that reason.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        live = [
            thread for thread in threading.enumerate()
            if thread.name.startswith("turn-")
            or thread.name in ("thread-gist", "skill-suggest")
        ]
        if not live:
            return
        for thread in live:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
    raise AssertionError(
        f"turn threads still running after {timeout_s}s: "
        f"{[t.name for t in threading.enumerate()]}",
    )


def _transcript(db: sqlite3.Connection, role: str) -> list[tuple[int, int, str]]:
    """``(id, turn_id, content)`` for every hygiene-channel row of one role,
    oldest first — the durable record each assertion below is made against."""
    return [
        (int(row[0]), int(row[1]), str(row[2] or ""))
        for row in db.execute(
            "SELECT id, turn_id, content FROM transcript "
            "WHERE channel = ? AND role = ? ORDER BY id",
            (Channel.MEMORY_HYGIENE.value, role),
        ).fetchall()
    ]


# ---------------------------------------------------------------------------
# 1 — the first run anchors the chain on the store itself
# ---------------------------------------------------------------------------


class TestFirstRunChainStart:
    """With no coverage row to read, the chain starts at the store's own
    earliest ``created_at``."""

    pytestmark = [pytest.mark.usefixtures("chat_provider")]

    def test_first_window_starts_exactly_at_the_earliest_stored_row(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A store with one two-day-old row and no hygiene transcript history
        opens its first window at that row's ``created_at``, to the second.

        There is no earlier coverage to resume from, so anchoring anywhere else
        would either re-read memories that never existed yet or silently skip
        the store's oldest day. The assertion is made against the value read
        back out of the database, not against the datetime the test seeded, so
        a format drift between what the model writes and what the service parses
        would fail it.
        """
        _Clock(monkeypatch, _NOW)
        _seed_graph_row(db, "hygiene.alpha", _ALPHA, _ALPHA_AT)

        provider = _ScriptedProvider()
        with patch(_BUILD_CLIENT, return_value=provider):
            status = MemoryHygieneService().tick()
            _drain_turns()

        listings = _listings(provider)
        assert listings, f"no listing request was served (tick said {status!r})"

        stored_created_at = db.execute(
            "SELECT created_at FROM memory_graph WHERE subject = ?", ("hygiene.alpha",),
        ).fetchone()[0]
        assert _window_of(listings[0]) == f"window: {stored_created_at} → {_BOUNDARY_1.isoformat()}", (
            f"the first window did not open on the store's earliest row: "
            f"{_window_of(listings[0])!r}"
        )
        assert _ALPHA in _listing_of(listings[0]), "the seeded row never reached the listing"


# ---------------------------------------------------------------------------
# 2 — an empty store is a no-op, not a turn
# ---------------------------------------------------------------------------


class TestEmptyStore:
    """Nothing stored, nothing covered — the tick must cost nothing."""

    pytestmark = [pytest.mark.usefixtures("chat_provider")]

    def test_empty_store_fires_no_provider_call_and_writes_no_row(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Both memory tables empty and no coverage row: the tick reports
        ``empty store``, calls no provider, and leaves the transcript untouched.

        A fresh install ticks on this branch every day until the first memory is
        written, so a turn fired here would burn a model call — and open a
        hygiene thread — for a day that has nothing in it.
        """
        _Clock(monkeypatch, _NOW)
        db.execute("DELETE FROM memory_graph")
        db.execute("DELETE FROM memory_map")
        db.commit()

        provider = _ScriptedProvider()
        with patch(_BUILD_CLIENT, return_value=provider):
            status = MemoryHygieneService().tick()
            _drain_turns()

        assert status == "empty store"
        assert provider.requests == [], (
            f"an empty store fired {len(provider.requests)} provider call(s)"
        )
        assert db.execute("SELECT COUNT(*) FROM transcript").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# 3 — a two-window catch-up runs on ONE persistent thread
# ---------------------------------------------------------------------------


class TestTwoWindowCatchUp:
    """Two closed windows, one tick, one thread — and a second tick that has
    nothing left to do."""

    pytestmark = [pytest.mark.usefixtures("chat_provider")]

    @staticmethod
    def _seed(db: sqlite3.Connection) -> None:
        _seed_graph_row(db, "hygiene.alpha", _ALPHA, _ALPHA_AT)
        _seed_graph_row(db, "hygiene.beta", _BETA, _BETA_AT)

    @staticmethod
    def _provider() -> _ScriptedProvider:
        return _ScriptedProvider([
            _Lane(_listing_carries(_ALPHA), [_Say("- merged alpha facts into one subject")]),
            _Lane(_listing_carries(_BETA), [_Say("- merged beta facts into one subject")]),
        ])

    def test_both_windows_run_once_each_on_one_persistent_turn(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One tick over two closed windows runs exactly two passes, oldest
        window first, both on the one turn the machine-state key names — and a
        second tick then has nothing pending.

        The thread is the point: hygiene is a running conversation with itself,
        so every pass must fork into the same turn rather than opening a fresh
        one per day. The stable id lives in machine state, and the coverage the
        next tick resumes from is read back out of the listings those passes
        stored — no separate cursor to drift.
        """
        _Clock(monkeypatch, _NOW)
        self._seed(db)
        provider = self._provider()
        service = MemoryHygieneService()

        with patch(_BUILD_CLIENT, return_value=provider):
            status = service.tick()
            _drain_turns()

        assert status == "processed=2 skipped=0"
        assert len(_listings(provider)) == 2, (
            f"expected one pass per window, got {len(_listings(provider))}: "
            f"{[_window_of(r) for r in _listings(provider)]}"
        )
        assert {_window_of(r) for r in _listings(provider)} == {
            _window_line(_ALPHA_AT, _BOUNDARY_1),
            _window_line(_BOUNDARY_1, _BOUNDARY_2),
        }

        # Oldest first, asserted on the input rows the service writes itself —
        # the order it drives the chain in, independent of which turn thread
        # reaches the provider first.
        inputs = _transcript(db, "memory_hygiene")
        assert [parse_window_bounds(content) for _id, _turn, content in inputs] == [
            (_ALPHA_AT, _BOUNDARY_1),
            (_BOUNDARY_1, _BOUNDARY_2),
        ]

        # One thread: both passes share a turn_id, and it is the one persisted.
        turn_ids = {turn for _id, turn, _content in inputs}
        assert len(turn_ids) == 1, f"the two passes opened {len(turn_ids)} threads: {turn_ids}"
        stored = MachineStateRow.newest_active_by_key(_TURN_ID_KEY)
        assert stored is not None, "the hygiene turn_id was never persisted"
        assert int(stored.value or -1) == turn_ids.pop()

        # Both passes settled their conclusion onto that thread.
        settled = [content for _id, _turn, content in _transcript(db, "assistant")]
        assert len(settled) == 2, f"expected 2 settled passes, got {settled!r}"

        served = len(_listings(provider))
        with patch(_BUILD_CLIENT, return_value=provider):
            second = service.tick()
            _drain_turns()

        assert second == "no pending windows"
        assert len(_listings(provider)) == served, (
            "a second tick re-ran a window the stored listings already cover"
        )

    def test_the_later_window_reads_the_earlier_windows_conclusion(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The second pass must open on the first pass's own bullet.

        Consolidation is sequential by nature — the second day's pass has to see
        that yesterday's subjects were already merged, or it re-merges them under
        a fresh synonym. The fork history read is what carries that across, so
        the second request must show ``## Previous Messages`` with the first
        pass's settled line in it.
        """
        _Clock(monkeypatch, _NOW)
        self._seed(db)
        provider = self._provider()

        with patch(_BUILD_CLIENT, return_value=provider):
            MemoryHygieneService().tick()
            _drain_turns()

        listings = _listings(provider)
        assert len(listings) == 2
        assert _window_of(listings[0]) == _window_line(_ALPHA_AT, _BOUNDARY_1), (
            "the windows did not reach the provider oldest-first"
        )
        assert _window_of(listings[1]) == _window_line(_BOUNDARY_1, _BOUNDARY_2)

        second = _body(listings[1])
        assert "## Previous Messages" in second, "the second pass did not fork the thread"
        assert "- merged alpha facts into one subject" in second, (
            "the second pass opened blind to the first pass's conclusion"
        )


# ---------------------------------------------------------------------------
# 4 — a window with no memory activity is skipped in silence
# ---------------------------------------------------------------------------


class TestEmptyWindowSkipped:
    """A day on which nothing was written costs nothing and leaves no trace."""

    pytestmark = [pytest.mark.usefixtures("chat_provider")]

    def test_a_window_with_no_rows_is_skipped_without_a_turn(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A chain of one empty window followed by one populated window runs a
        single pass and reports the skip in its status.

        The empty day must still be *covered* — the next tick resumes from the
        populated window's end, so it is never revisited — but firing a turn on
        an empty listing would ask the model to consolidate nothing, and would
        leave a transcript row claiming the day was worked.
        """
        clock = _Clock(monkeypatch, datetime(2026, 3, 4, 12, 0, 0, tzinfo=timezone.utc))
        _seed_graph_row(db, "hygiene.alpha", _ALPHA, _ALPHA_AT)
        provider = _ScriptedProvider()
        service = MemoryHygieneService()

        # A first, ordinary pass — the cheapest way to lay down real coverage.
        with patch(_BUILD_CLIENT, return_value=provider):
            assert service.tick() == "processed=1 skipped=0"
            _drain_turns()
        served = len(_listings(provider))
        assert served == 1

        # 04 Mar → 05 Mar has no memory activity at all; 05 Mar → 06 Mar does.
        _seed_graph_row(
            db, "hygiene.beta", _BETA, datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc),
        )
        clock.now = datetime(2026, 3, 6, 12, 0, 0, tzinfo=timezone.utc)

        with patch(_BUILD_CLIENT, return_value=provider):
            status = service.tick()
            _drain_turns()

        assert status == "processed=1 skipped=1"

        fresh = _listings(provider)[served:]
        assert len(fresh) == 1, (
            f"the empty window fired a pass: {[_window_of(r) for r in fresh]}"
        )
        assert _window_of(fresh[0]) == _window_line(
            datetime(2026, 3, 5, 4, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 6, 4, 0, 0, tzinfo=timezone.utc),
        )

        empty_window = _window_line(_BOUNDARY_1, _BOUNDARY_2)
        assert not [
            content for _id, _turn, content in _transcript(db, "memory_hygiene")
            if empty_window in content
        ], "the skipped window still wrote a transcript row"


# ---------------------------------------------------------------------------
# 5 — a crashed pass is skipped, never retried
# ---------------------------------------------------------------------------


class TestCrashedPassIsSkipped:
    """One window's turn dying must not stall the chain or re-run forever."""

    pytestmark = [pytest.mark.usefixtures("chat_provider")]

    @staticmethod
    def _provider() -> _ScriptedProvider:
        return _ScriptedProvider([
            _Lane(_listing_carries(_ALPHA), [_Crash("boom")]),
            _Lane(_listing_carries(_BETA), [_Say("- merged beta facts into one subject")]),
        ])

    def test_a_crashed_window_does_not_stop_the_next_one_or_come_back(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The first window's turn crashes; the second still runs, and a second
        tick never re-offers the crashed window.

        Coverage advances off the transcript row the crashed pass already wrote
        when it opened — so a crash is a *skip*, deliberately, not a retry. A
        window that failed once will fail the same way tomorrow, and retrying it
        would wedge the chain on that day forever while every later day goes
        unconsolidated.
        """
        _Clock(monkeypatch, _NOW)
        _seed_graph_row(db, "hygiene.alpha", _ALPHA, _ALPHA_AT)
        _seed_graph_row(db, "hygiene.beta", _BETA, _BETA_AT)
        provider = self._provider()
        service = MemoryHygieneService()

        with patch(_BUILD_CLIENT, return_value=provider):
            service.tick()
            _drain_turns()

        alpha_line = _window_line(_ALPHA_AT, _BOUNDARY_1)
        beta_line = _window_line(_BOUNDARY_1, _BOUNDARY_2)
        assert any(_window_of(r) == alpha_line for r in _listings(provider))
        assert any(_window_of(r) == beta_line for r in _listings(provider)), (
            "the crashed window took the next one down with it"
        )

        crashed = [
            row for row in TurnExecution.filter("channel", Channel.MEMORY_HYGIENE.value).get()
            if row.state == TurnExecution.CRASHED
        ]
        assert len(crashed) == 1, (
            f"expected exactly one crashed pass, got {[r.state for r in crashed]}"
        )
        settled = [content for _id, _turn, content in _transcript(db, "assistant")]
        assert settled == ["- merged beta facts into one subject"]

        served = len(_listings(provider))
        with patch(_BUILD_CLIENT, return_value=provider):
            second = service.tick()
            _drain_turns()

        assert second == "no pending windows"
        assert not [
            r for r in _listings(provider)[served:] if _window_of(r) == alpha_line
        ], "the crashed window was retried"

    def test_the_service_reports_the_crash_it_skipped(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A skipped window is logged by the service, at ERROR, once.

        Silence here is the failure mode that matters: a window quietly dropped
        leaves a hole in the store's consolidation that nothing else reports.
        """
        _Clock(monkeypatch, _NOW)
        _seed_graph_row(db, "hygiene.alpha", _ALPHA, _ALPHA_AT)
        _seed_graph_row(db, "hygiene.beta", _BETA, _BETA_AT)
        provider = self._provider()

        with caplog.at_level("ERROR", logger=memory_hygiene_service.__name__):
            with patch(_BUILD_CLIENT, return_value=provider):
                MemoryHygieneService().tick()
                _drain_turns()

        reported = [
            record for record in caplog.records
            if record.name == memory_hygiene_service.__name__
            and record.levelname == "ERROR"
            and "turn crashed" in record.getMessage()
        ]
        assert len(reported) == 1, (
            f"expected one 'turn crashed' ERROR from the service, got {len(reported)}"
        )


# ---------------------------------------------------------------------------
# 6 — an over-limit window splits in half
# ---------------------------------------------------------------------------


class TestOverLimitWindowSplit:
    """A listing that will not fit is halved and re-run, not dropped."""

    pytestmark = [pytest.mark.usefixtures("chat_provider")]

    @staticmethod
    def _seed(db: sqlite3.Connection) -> None:
        """Two rows in ONE closed window — the smallest set that can be halved."""
        _seed_graph_row(db, "hygiene.alpha", _ALPHA, _ALPHA_AT)
        _seed_graph_row(
            db, "hygiene.beta", _BETA, datetime(2026, 3, 3, 18, 0, 0, tzinfo=timezone.utc),
        )

    @staticmethod
    def _provider() -> _ScriptedProvider:
        """Only the full, two-row listing over-limits; either half fits.

        The rejection is keyed on the listing rather than the whole payload so
        the turn's own recovery machinery (the hand-over pass and the history
        compactor, neither of which carries a listing) can run for real — every
        one of the step loop's re-sends still carries the same two-row listing,
        so the exception survives the recovery and escapes the turn.
        """
        return _ScriptedProvider([
            _Lane(
                _listing_carries(_ALPHA, _BETA),
                [_OverLimit("payload exceeds the model's maximum context length")],
            ),
            _Lane(_listing_carries(_ALPHA, without=(_BETA,)), [_Say("- merged the alpha half")]),
            _Lane(_listing_carries(_BETA, without=(_ALPHA,)), [_Say("- merged the beta half")]),
        ])

    def test_an_over_limit_listing_escapes_the_turns_own_recovery(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A listing the provider keeps rejecting is let out of the turn, not
        spun on — which is what puts the split decision in the service's hands.

        Compaction is the only lever the step loop has, and it cannot shrink a
        listing that is re-appended verbatim on every re-send. The bound turns
        that into a crashed turn carrying the size fault, rather than a thread
        retrying until the process dies.
        """
        _Clock(monkeypatch, _NOW)
        self._seed(db)
        provider = self._provider()

        with patch(_BUILD_CLIENT, return_value=provider):
            MemoryHygieneService().tick()
            _drain_turns()

        both = [
            r for r in _listings(provider)
            if _ALPHA in _listing_of(r) and _BETA in _listing_of(r)
        ]
        assert both, "the full two-row listing was never sent"

        crashed = [
            row for row in TurnExecution.filter("channel", Channel.MEMORY_HYGIENE.value).get()
            if row.state == TurnExecution.CRASHED
        ]
        assert crashed, "the over-limit turn never terminated as crashed"
        assert "context length" in (crashed[0].stop_reason or ""), (
            f"the turn crashed for the wrong reason: {crashed[0].stop_reason!r}"
        )
        assert len(both) < 10, f"the over-limit send was not bounded — {len(both)} attempts"

    def test_the_over_limit_window_is_re_run_as_two_halves(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each half is re-run against the SAME window bounds.

        Halving is the only way a day whose listing outgrew the window gets
        consolidated at all, and both halves must stay stamped with the window
        they came from — a half that re-declares narrower bounds would leave the
        coverage chain claiming less than it covered.
        """
        _Clock(monkeypatch, _NOW)
        self._seed(db)
        provider = self._provider()

        with patch(_BUILD_CLIENT, return_value=provider):
            MemoryHygieneService().tick()
            _drain_turns()

        window = _window_line(_ALPHA_AT, _BOUNDARY_1)
        alpha_only = [
            r for r in _listings(provider)
            if _ALPHA in _listing_of(r) and _BETA not in _listing_of(r)
        ]
        beta_only = [
            r for r in _listings(provider)
            if _BETA in _listing_of(r) and _ALPHA not in _listing_of(r)
        ]
        assert alpha_only, "the first half was never re-run"
        assert beta_only, "the second half was never re-run"
        assert {_window_of(r) for r in alpha_only + beta_only} == {window}, (
            "a half re-declared its own bounds instead of the window's"
        )


# ---------------------------------------------------------------------------
# 7 — a tool-active pass settles, and its conclusion carries forward
# ---------------------------------------------------------------------------


class TestToolActivePassHandsOff:
    """The pass that actually uses its tools still settles a readable bullet."""

    pytestmark = [pytest.mark.usefixtures("chat_provider")]

    def test_a_recall_round_then_a_settle_leaves_the_bullet_on_the_thread(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A pass that recalls first and then answers settles its bullet onto
        the hygiene channel, and the next pass opens with it in view.

        This is the shape every real hygiene pass takes — recall, act, report —
        so the bullet has to survive the tool round: it is the only record of
        what the pass changed, and the only thing the next day's pass can read
        to avoid undoing it.
        """
        clock = _Clock(monkeypatch, datetime(2026, 3, 4, 12, 0, 0, tzinfo=timezone.utc))
        _seed_graph_row(db, "hygiene.alpha", _ALPHA, _ALPHA_AT)
        alpha_lane = _Lane(
            _listing_carries(_ALPHA),
            [
                _Call(Recall.NAME, {"query": "alpha facts"}),
                _Say("- deleted stale subject"),
            ],
        )
        provider = _ScriptedProvider([alpha_lane])
        service = MemoryHygieneService()

        with patch(_BUILD_CLIENT, return_value=provider):
            assert service.tick() == "processed=1 skipped=0"
            _drain_turns()

        assert alpha_lane.served >= 2, (
            f"the pass settled without running the tool round ({alpha_lane.served} round(s))"
        )
        alpha_requests = [r for r in _listings(provider) if _ALPHA in _listing_of(r)]
        assert f"[{Recall.NAME}]" in _body(alpha_requests[1]), (
            "the settle round did not carry the recall it had just made"
        )

        # The tool round leaves its own prose row behind; the bullet is what
        # closes the pass.
        settled = [content for _id, _turn, content in _transcript(db, "assistant")]
        assert settled[-1] == "- deleted stale subject", (
            f"the pass did not settle on its bullet: {settled!r}"
        )

        # The next day's pass forks the same thread and opens on that bullet.
        _seed_graph_row(
            db, "hygiene.beta", _BETA, datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc),
        )
        clock.now = datetime(2026, 3, 6, 12, 0, 0, tzinfo=timezone.utc)
        with patch(_BUILD_CLIENT, return_value=provider):
            service.tick()
            _drain_turns()

        beta_request = next(
            (r for r in _listings(provider) if _BETA in _listing_of(r)), None,
        )
        assert beta_request is not None, "the next window never ran"
        forked = _body(beta_request)
        assert "## Previous Messages" in forked, "the next pass did not fork the thread"
        assert "- deleted stale subject" in forked, (
            "the previous pass's conclusion did not carry forward"
        )


# ---------------------------------------------------------------------------
# 8 — the crash signal the service consolidates on
# ---------------------------------------------------------------------------


class TestCleanTurnCarriesNoCrash:
    """``crash_exception`` is the field ``_consolidate`` branches on."""

    pytestmark = [pytest.mark.usefixtures("chat_provider")]

    def test_a_settled_hygiene_turn_records_no_crash_exception(self) -> None:
        """A hygiene pass that settles cleanly leaves ``crash_exception`` None.

        Driven through the same config and entry point the service uses, and
        joined via ``result()`` — the same join ``_run_turn`` performs — so the
        field is read after the turn has actually terminated. Without the join
        the crash branch could not be told apart from a race.
        """
        listing = render_listing(
            _ALPHA_AT,
            _BOUNDARY_1,
            [MemoryGraphRow(subject="hygiene.alpha", contents=_ALPHA)],
            [],
            timezone.utc,
        )
        provider = _ScriptedProvider()
        with patch(_BUILD_CLIENT, return_value=provider):
            mp = MessageProcessor.process(MemoryHygieneConfig(), listing)
            mp.result()
            _drain_turns()

        assert mp.crash_exception is None
        assert _listings(provider), "the pass never reached the provider"
        assert _window_of(_listings(provider)[0]) == _window_line(_ALPHA_AT, _BOUNDARY_1)
