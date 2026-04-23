"""
WorldState — in-process singleton for ambient world context.

Zero DB writes, zero worker threads, zero polling.
render() is called on demand: once per user turn and once per Cognition
endpoint hit.  Signals are stored in an internal dict, TTL-pruned on read.

Push sites:
  - POST /api/updates/context → world_state.set("telemetry", ctx)
  - POST /api/signals         → world_state.push_signal(source, label)
  - WorldAwarenessService     → world_state.push_signal("news", headline)
  - IMAPHandler               → world_state.push_signal("inbox", summary)

Pull sites (read inside render()):
  - scheduled_items  — upcoming / recently-fired schedule entries
  - transcript WHERE channel='goal_pursuit' — active pursuits

Output format (literal):
  ### Background Telemetry,Processes & Signals
  [telemetry]

  * **user**;time:{HH:MM}{±HH:MM},location:{location},mobility:{mobility}
  * **device**;name:{device-name},battery:{power}
  [schedule]

  * {message} (due-in:{duration})
  [bg_process(last_update:{duration} ago)] {content}
  [signal:{source}] {label}
"""

import logging
import threading
import time as _time

from services.time_utils import utc_now, parse_utc
from services.time_formatter_service import TimeFormatterService

logger = logging.getLogger(__name__)

_SECTION_HEADER = "### Background Telemetry,Processes & Signals"

# Schedule query — pending items due in ≤7 days, or fired in last 24 hours, not hidden
_SCHEDULE_SQL = """
SELECT message, due_at, last_fired_at, recurrence
FROM scheduled_items
WHERE (
    (status = 'pending' AND due_at <= datetime('now', '+7 days'))
    OR (status = 'fired' AND last_fired_at >= datetime('now', '-24 hours'))
) AND hidden = 0
ORDER BY CASE WHEN status = 'pending' THEN 0 ELSE 1 END, due_at ASC
LIMIT 20
"""

# bg_process query — recent goal_pursuit transcript rows (schema uses created_at, not updated_at)
_BG_PROCESS_SQL = """
SELECT content, created_at
FROM transcript
WHERE channel = 'goal_pursuit'
  AND created_at >= datetime('now', '-24 hours')
ORDER BY created_at DESC
LIMIT 10
"""


# ── Render helpers (module-level, pure functions) ─────────────────────────────


def _telemetry_user_fields(ctx: dict) -> list[str]:
    """Extract the user-line fields (time, location, mobility) from telemetry ctx."""
    fields = []
    time_val = ctx.get("local_time") or ctx.get("time")
    offset_val = ctx.get("utc_offset") or ctx.get("timezone_offset")
    if time_val:
        fields.append(f"time:{time_val}{offset_val}" if offset_val else f"time:{time_val}")
    location = ctx.get("location") or ctx.get("place")
    if location:
        fields.append(f"location:{location}")
    mobility = ctx.get("mobility")
    if mobility:
        fields.append(f"mobility:{mobility}")
    return fields


def _telemetry_device_fields(device) -> list[str]:
    """Extract the device-line fields (name, battery) from telemetry ctx.device."""
    if not isinstance(device, dict):
        return []
    fields = []
    name = device.get("name")
    if name:
        fields.append(f"name:{name}")
    battery = device.get("battery")
    if battery is not None:
        fields.append(f"battery:{battery}")
    return fields


def _schedule_due_field(due_at_str: str, now) -> str:
    """Format the due-in field for a scheduled_items row."""
    try:
        diff_secs = (parse_utc(due_at_str) - now).total_seconds()
    except Exception:
        return "due-in:unknown"
    if diff_secs >= 0:
        return f"due-in:{TimeFormatterService.duration(diff_secs)}"
    return f"due-in:{TimeFormatterService.duration(abs(diff_secs))} ago"


def _schedule_last_fired_field(last_fired_str) -> str | None:
    """Format the last-fired field, or None if absent / unparseable."""
    if not last_fired_str:
        return None
    try:
        return f"last-fired:{TimeFormatterService.ago(last_fired_str)}"
    except Exception:
        return None


def _schedule_recurrence_field(recurrence_str) -> str | None:
    """Format the repeats field, or None if absent / non-numeric."""
    if not recurrence_str:
        return None
    try:
        rec_secs = float(recurrence_str)
    except (ValueError, TypeError):
        logger.debug("[WorldState] unparseable recurrence dropped: %r", recurrence_str)
        return None
    return f"repeats:every {TimeFormatterService.duration(rec_secs)}"


class WorldState:
    """In-process singleton. Sole owner of world-state data + rendering.

    Thread-safe via a single internal lock protecting ``_store``.
    """

    def __init__(self):
        self._store: dict = {}           # arbitrary type → dict fragments
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────

    def set(self, type: str, value: dict) -> "WorldState":
        """Store a typed fragment, overwriting any prior value.

        Args:
            type: Arbitrary key, e.g. ``'telemetry'``, ``'signals'``.
            value: The dict to store.

        Returns:
            ``self`` for chaining.
        """
        with self._lock:
            self._store[type] = value
        return self

    def get(self, type: str) -> dict:
        """Retrieve a typed fragment.

        For ``type='signals'``, expired entries are pruned before returning.

        Args:
            type: Key previously passed to :meth:`set`.

        Returns:
            The stored dict, or ``{}`` when unset.
        """
        with self._lock:
            if type == "signals":
                self._prune_signals()
            return dict(self._store.get(type) or {})

    def push_signal(self, source: str, label: str, ttl: int = 3600) -> None:
        """Merge a signal into the signals dict, overwriting the same source.

        Args:
            source: Identifies the originating system (e.g. ``'news'``, ``'inbox'``).
            label: Human-readable signal text.
            ttl: Seconds until this signal expires.  Default 3600 (1 hour).
        """
        with self._lock:
            signals: dict = dict(self._store.get("signals") or {})
            signals[source] = {
                "label": label,
                "expires_at": _time.time() + ttl,
            }
            self._store["signals"] = signals

    def render(self) -> str:
        """Combine in-memory fragments and DB reads into the literal output block.

        Returns:
            Multi-line string starting with the section header, or ``''`` when
            every section is empty.  Raises on DB errors — callers must handle.
        """
        parts = []

        # ── Telemetry ──────────────────────────────────────────────────────
        telemetry_lines = self._render_telemetry()
        if telemetry_lines:
            parts.append("[telemetry]")
            parts.append("")
            parts.extend(telemetry_lines)

        # ── Schedule ───────────────────────────────────────────────────────
        schedule_lines = self._render_schedule()
        if schedule_lines:
            parts.append("[schedule]")
            parts.append("")
            parts.extend(schedule_lines)

        # ── bg_process ─────────────────────────────────────────────────────
        bg_lines = self._render_bg_process()
        parts.extend(bg_lines)

        # ── Signals ────────────────────────────────────────────────────────
        signal_lines = self._render_signals()
        parts.extend(signal_lines)

        if not parts:
            return ""

        return _SECTION_HEADER + "\n" + "\n".join(parts)

    # ── Private render helpers ─────────────────────────────────────────────

    def _render_telemetry(self) -> list[str]:
        """Produce bullet lines for the [telemetry] section."""
        with self._lock:
            ctx = dict(self._store.get("telemetry") or {})
        if not ctx:
            return []

        lines = []
        user_fields = _telemetry_user_fields(ctx)
        if user_fields:
            lines.append("* **user**;" + ",".join(user_fields))
        device_fields = _telemetry_device_fields(ctx.get("device"))
        if device_fields:
            lines.append("* **device**;" + ",".join(device_fields))
        return lines

    def _render_schedule(self) -> list[str]:
        """Produce bullet lines for the [schedule] section from scheduled_items."""
        rows = _fetch_schedule_rows()
        if not rows:
            return []

        now = utc_now()
        lines = []
        for row in rows:
            message = row.get("message") or ""
            fields = [_schedule_due_field(row.get("due_at") or "", now)]
            last_fired_field = _schedule_last_fired_field(row.get("last_fired_at"))
            if last_fired_field:
                fields.append(last_fired_field)
            recurrence_field = _schedule_recurrence_field(row.get("recurrence"))
            if recurrence_field:
                fields.append(recurrence_field)
            lines.append(f"* {message} ({','.join(fields)})")

        return lines

    def _render_bg_process(self) -> list[str]:
        """Produce [bg_process(...)] lines from goal_pursuit transcript rows."""
        rows = _fetch_bg_process_rows()
        lines = []
        for row in rows:
            content = (row.get("content") or "").strip()
            created_at = row.get("created_at") or ""
            if not content:
                continue
            try:
                last_update = TimeFormatterService.ago(created_at)
            except Exception:
                last_update = "unknown"
            if len(content) > 200:
                cut = content.rfind(" ", 0, 200)
                if cut == -1:
                    cut = 200
                content = content[:cut] + "…"
            lines.append(f"[bg_process(last_update:{last_update})] {content}")
        return lines

    def _render_signals(self) -> list[str]:
        """Produce [signal:{source}] lines from the in-memory signals dict."""
        with self._lock:
            self._prune_signals()
            signals = dict(self._store.get("signals") or {})
        if not signals:
            return []
        lines = []
        for source in sorted(signals.keys()):
            label = signals[source].get("label", "")
            lines.append(f"[signal:{source}] {label}")
        return lines

    def _prune_signals(self) -> None:
        """Remove expired signals from the store. Must be called under self._lock."""
        signals = self._store.get("signals")
        if not signals:
            return
        now = _time.time()
        pruned = {
            src: entry
            for src, entry in signals.items()
            if entry.get("expires_at", 0) > now
        }
        self._store["signals"] = pruned


# ── DB helpers (reused by the Cognition endpoint) ──────────────────────────


def _get_db():
    from services.database_service import get_shared_db_service
    return get_shared_db_service()


def _fetch_schedule_rows() -> list[dict]:
    """Execute the schedule query and return rows as list of dicts."""
    db = _get_db()
    with db.connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(_SCHEDULE_SQL)
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        finally:
            cursor.close()


def _fetch_bg_process_rows() -> list[dict]:
    """Execute the bg_process query and return rows as list of dicts."""
    db = _get_db()
    with db.connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(_BG_PROCESS_SQL)
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        finally:
            cursor.close()


# ── Module-level singleton ─────────────────────────────────────────────────

world_state = WorldState()
