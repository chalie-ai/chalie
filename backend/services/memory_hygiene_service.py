"""Pure window-math and listing render/parse helpers for the daily memory
consolidation pass, plus the :class:`MemoryHygieneService` that drives each tick.

All functions are pure (no I/O, no side-effects) so they can be unit-tested
in isolation and imported by whichever service owns the consolidation loop.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone, tzinfo
from typing import TYPE_CHECKING

from services.time_utils import PARSE_SENTINEL, parse_utc, utc_now

if TYPE_CHECKING:
    from models.memory_graph import MemoryGraphRow
    from models.memory_map import MemoryMapRow
    from controllers.message_processor import MessageProcessor

_HYGIENE_HOUR = 4
_GRACE = timedelta(minutes=5)

# The stable hygiene turn_id is persisted here (discovery's mechanism): the
# first fire allocates a fresh MAIN turn and stores its id; every later fire
# forks into that same turn, so all passes cluster as one thread.
_TURN_ID_KEY = "memory_hygiene:turn_id"

logger = logging.getLogger(__name__)


class MemoryHygieneService:
    """Daily consolidation driver.

    One ``tick`` reads the last coverage transcript row, computes pending
    day-windows, and runs one :class:`~controllers.message_processor.MessageProcessor`
    turn per window on the same persistent thread (keyed by
    ``memory_hygiene:turn_id`` in machine state).
    """

    def __init__(self) -> None:
        self._empty_until: datetime | None = None

    def tick(self) -> str:
        """Run one consolidation tick.

        Returns a short status string the cron job logs under ``[MEMORY HYGIENE]``.
        """
        from models.memory_graph import MemoryGraphRow  # noqa: PLC0415
        from models.memory_map import MemoryMapRow  # noqa: PLC0415
        from models.transcript import Transcript  # noqa: PLC0415
        from services.locale_service import get_timezone  # noqa: PLC0415

        tz = get_timezone()

        # ── coverage read ──────────────────────────────────────────────────
        row = (
            Transcript.filter("role", "memory_hygiene")
            .order_by("id DESC")
            .first()
        )
        if row is not None:
            content = str(row.content or "")
            bounds = parse_window_bounds(content)
            if bounds is None:
                head = (content[:200] + "...") if len(content) > 200 else content
                logger.error("[MEMORY HYGIENE] corrupt coverage row content: %s", head)
                return "corrupt coverage"
            covered_end_dt: datetime = bounds[1]  # end of last covered window
        else:
            # First run: covered_end = minimum earliest-created-at across both tables.
            graph_min_raw = MemoryGraphRow.earliest_created_at()
            map_min_raw = MemoryMapRow.earliest_created_at()
            graph_min = parse_utc(graph_min_raw) if graph_min_raw is not None else PARSE_SENTINEL
            map_min = parse_utc(map_min_raw) if map_min_raw is not None else PARSE_SENTINEL
            if graph_min == PARSE_SENTINEL and map_min == PARSE_SENTINEL:
                return "empty store"
            if graph_min == PARSE_SENTINEL:
                covered_end_dt = map_min
            elif map_min == PARSE_SENTINEL:
                covered_end_dt = graph_min
            else:
                covered_end_dt = min(graph_min, map_min)
            if covered_end_dt == PARSE_SENTINEL:
                logger.error("[MEMORY HYGIENE] could not parse earliest created_at from store")
                return "corrupt store timestamps"

        # ── pending chain ──────────────────────────────────────────────────
        windows = pending_windows(covered_end_dt, utc_now(), tz)
        if not windows:
            return "no pending windows"

        # ── _empty_until gate ──────────────────────────────────────────────
        newest_end = windows[-1][1]
        if self._empty_until is not None and newest_end <= self._empty_until:
            return "no new windows since last empty tick"

        processed = 0
        skipped = 0
        fired_any = False

        for window_start, window_end in windows:
            start_iso = window_start.isoformat()
            end_iso = window_end.isoformat()

            graph_rows: list[MemoryGraphRow] = MemoryGraphRow.updated_in_window(start_iso, end_iso)
            map_rows: list[MemoryMapRow] = MemoryMapRow.generated_in_window(start_iso, end_iso)

            if not graph_rows and not map_rows:
                skipped += 1
                continue

            fired_any = True
            processed += self._consolidate(window_start, window_end, graph_rows, map_rows, tz)

        # _empty_until bookkeeping.
        if not fired_any:
            self._empty_until = newest_end
        else:
            self._empty_until = None

        return f"processed={processed} skipped={skipped}"

    def _run_turn(self, listing: str) -> "MessageProcessor":
        """Run one MessageProcessor turn, reusing the persisted turn_id."""
        from models.machine_state import MachineStateRow  # noqa: PLC0415
        from configs.channels.memory_hygiene import MemoryHygieneConfig  # noqa: PLC0415
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        turn_id = self._load_turn_id()
        mp = MessageProcessor.process(
            MemoryHygieneConfig(),
            listing,
            turn_id=turn_id if turn_id is not None else -1,
        )
        if turn_id is None:
            MachineStateRow.store(
                key=_TURN_ID_KEY,
                value=str(mp.turn_id),
                source="memory_hygiene_job",
            )
        return mp

    def _consolidate(
        self,
        window_start: datetime,
        window_end: datetime,
        graph_rows: list["MemoryGraphRow"],
        map_rows: list["MemoryMapRow"],
        tz: tzinfo,
    ) -> int:
        """Recursively consolidate one window, splitting on ContextLimit."""
        from exceptions import ContextLimit  # noqa: PLC0415

        listing = render_listing(window_start, window_end, graph_rows, map_rows, tz)
        mp = self._run_turn(listing)

        if mp.crash_exception is None:
            return 1

        if isinstance(mp.crash_exception, ContextLimit):
            if len(graph_rows) + len(map_rows) == 1:
                logger.error(
                    "[MEMORY HYGIENE] single-row listing still over-limits — "
                    "skipping window %s→%s",
                    window_start.isoformat(),
                    window_end.isoformat(),
                )
                return 0
            left_graph, left_map, right_graph, right_map = self._split_rows(graph_rows, map_rows)
            return (
                self._consolidate(window_start, window_end, left_graph, left_map, tz)
                + self._consolidate(window_start, window_end, right_graph, right_map, tz)
            )

        logger.error(
            "[MEMORY HYGIENE] turn crashed: %s",
            type(mp.crash_exception).__name__,
        )
        return 0

    def _load_turn_id(self) -> int | None:
        """Read the persisted memory-hygiene turn_id, or None on first fire."""
        from models.machine_state import MachineStateRow  # noqa: PLC0415

        row = MachineStateRow.newest_active_by_key(_TURN_ID_KEY)
        if not row or not row.value:
            return None
        try:
            return int(row.value)
        except (TypeError, ValueError):
            return None

    def _split_rows(
        self, graph_rows: list["MemoryGraphRow"], map_rows: list["MemoryMapRow"]
    ) -> tuple[list["MemoryGraphRow"], list["MemoryMapRow"], list["MemoryGraphRow"], list["MemoryMapRow"]]:
        """Split the combined (graph then map) row set into two halves by index."""
        g = len(graph_rows)
        mid = (g + len(map_rows)) // 2
        left_graph = graph_rows[: min(mid, g)]
        left_map = map_rows[: max(0, mid - g)]
        right_graph = graph_rows[min(mid, g):]
        right_map = map_rows[max(0, mid - g):]
        return left_graph, left_map, right_graph, right_map


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
