"""Pure window-math and listing render/parse helpers for the daily memory
consolidation pass.

No service class lives here — that comes in a later task.  All functions are
pure (no I/O, no side-effects) so they can be unit-tested in isolation and
imported by whichever service owns the consolidation loop.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone, tzinfo
from typing import TYPE_CHECKING

from services.time_utils import PARSE_SENTINEL, parse_utc

if TYPE_CHECKING:
    from models.memory_graph import MemoryGraphRow
    from models.memory_map import MemoryMapRow

_HYGIENE_HOUR = 4
_GRACE = timedelta(minutes=5)


def pending_windows(
    covered_end: datetime,
    now_utc: datetime,
    tz: tzinfo,
) -> list[tuple[datetime, datetime]]:
    """Return the list of completed (start, end) windows that still need
    hygiene.

    Each window is bounded by consecutive 04:00 local boundaries.  The first
    window starts exactly at ``covered_end`` (the point at which the last
    pass left off).  A boundary strictly after ``covered_end`` closes one
    window and opens the next.  Windows that end more than ``_GRACE`` ago
    (i.e. ``now_utc >= window_end + _GRACE``) are included; the still-open
    window (the one whose end is in the future) is never returned.

    DST transitions are handled naturally by tz arithmetic — no manual
    offsets — so a spring-forward day yields a 23 h window and a fall-back
    day yields 25 h.
    """
    # Materialise the first local boundary at or after covered_end.
    # covered_end is UTC-aware; we convert to local, snap to 04:00, then back
    # to UTC.
    if covered_end.tzinfo is None:
        covered_end = covered_end.replace(tzinfo=timezone.utc)

    local_covered = covered_end.astimezone(tz)
    candidate = datetime(
        local_covered.year,
        local_covered.month,
        local_covered.day,
        _HYGIENE_HOUR,
        0,
        0,
        tzinfo=tz,
    )
    # If 04:00 today is already at or before covered_end, step to tomorrow.
    if candidate <= covered_end.astimezone(tz):
        candidate += timedelta(days=1)

    # candidate is now the first boundary strictly after covered_end.
    windows: list[tuple[datetime, datetime]] = []
    window_start = covered_end
    boundary = candidate.astimezone(timezone.utc)

    while True:
        window_end = boundary
        # Include this window only if the grace period has expired.
        if now_utc >= window_end + _GRACE:
            windows.append((window_start, window_end))
        else:
            # Once we hit the first window that hasn't expired yet, everything
            # after it is also not expired — stop.
            break
        # Advance to the next boundary.
        local_boundary = boundary.astimezone(tz)
        next_candidate = datetime(
            local_boundary.year,
            local_boundary.month,
            local_boundary.day,
            _HYGIENE_HOUR,
            0,
            0,
            tzinfo=tz,
        )
        if next_candidate <= local_boundary:
            next_candidate += timedelta(days=1)
        boundary = next_candidate.astimezone(timezone.utc)
        window_start = window_end

    return windows


def render_listing(
    window_start: datetime,
    window_end: datetime,
    graph_rows: list["MemoryGraphRow"],
    map_rows: list["MemoryMapRow"],
    tz: tzinfo,
) -> str:
    """Render a human-readable consolidation listing for one window."""
    header_date = window_start.astimezone(tz).strftime("%a %d %b %Y")
    lines: list[str] = [
        f"# Memories Generated/Updated — {header_date}",
        f"window: {window_start.isoformat()} → {window_end.isoformat()}",
        "",
        "## Graph Memories",
        "**Hard Facts, Decisions, Pivots, Preferences, etc...**",
    ]
    if graph_rows:
        obj = {row.subject: row.contents for row in graph_rows}
        lines.append(json.dumps(obj, ensure_ascii=False))
    else:
        lines.append("(none this window)")

    lines.append("")
    lines.append("## Map Memories")
    lines.append("**Episodes, Incidents, Discussions, etc...**")
    if map_rows:
        for row in map_rows:
            generated_local = parse_utc(row.generated_at).astimezone(tz)
            time_str = generated_local.strftime("%H:%M")
            lines.append(f"[id {row.id} · {time_str}] {row.contents}")
    else:
        lines.append("(none this window)")

    return "\n".join(lines)


def parse_window_bounds(content: str) -> tuple[datetime, datetime] | None:
    """Reverse-engineer the (start, end) pair from a listing's ``window:``
    line.

    Returns ``None`` when the line is missing, malformed, or either side
    parses to :data:`~services.time_utils.PARSE_SENTINEL`.
    """
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("window: "):
            rest = stripped[len("window: "):]
            parts = rest.split(" → ")
            if len(parts) != 2:
                return None
            start = parse_utc(parts[0])
            end = parse_utc(parts[1])
            if start == PARSE_SENTINEL or end == PARSE_SENTINEL:
                return None
            return (start, end)
    return None
