"""
WorldState — in-process singleton for ambient world context.

render() is called on demand: once per user turn and once per Cognition
endpoint hit.  Signals are stored in an internal dict, TTL-pruned on read;
telemetry and schedule are read directly from the database.

Push sites:
  - POST /health (heartbeat.js) → ClientContextService.save() →
                                  upserts ``telemetry`` table rows
  - POST /api/signals           → world_state.push_signal(source, label)
  - WorldAwarenessService       → world_state.push_signal("news", headline)
  - IMAPHandler                 → world_state.push_signal("inbox", summary)

Pull sites (read inside render()):
  - telemetry        — flat key/value rows persisted from the FE heartbeat
  - scheduled_items  — upcoming / recently-fired schedule entries

Output format (literal):
  ### Background Telemetry,Processes & Signals
  [telemetry]
  * **user**;<flat top-level keys, k:v comma-separated>
  * **device**;<keys nested under "device", k:v comma-separated>
  * **behavioral**;<keys nested under "behavioral", k:v comma-separated>
  …one bullet per top-level group the FE sent…
  [schedule]
  * {message} (due-in:{duration})
  [signal:{source}] {label}
"""

import json
import logging
import threading
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from services.time_utils import utc_now, parse_utc
from services.time_formatter_service import TimeFormatterService

logger = logging.getLogger(__name__)

_SECTION_HEADER = "### Background Telemetry,Processes & Signals"

# Schedule query — pending items due in ≤7 days, or fired in last 24 hours, not hidden
_SCHEDULE_SQL = """
SELECT message, due_at, last_fired_at, recurrence, status
FROM scheduled_items
WHERE (
    (status = 'pending' AND due_at <= datetime('now', '+7 days'))
    OR (status = 'fired' AND last_fired_at >= datetime('now', '-6 hours'))
) AND hidden = 0
ORDER BY CASE WHEN status = 'pending' THEN 0 ELSE 1 END, due_at ASC
LIMIT 200
"""


# ── Render helpers (module-level, pure functions) ─────────────────────────────


# Top-level telemetry keys that should not be surfaced in the rendered block —
# they are internal bookkeeping or noise the LLM does not need.
_TELEMETRY_HIDDEN_KEYS = {"saved_at", "_location_name_stale", "connection"}

# Top-level dict groups that should not be rendered as their own bullet.
_TELEMETRY_HIDDEN_GROUPS = {"behavioral"}

# Strftime format for the synthesised local_time field — "Sat 02 May 2026 11:35".
_LOCAL_TIME_FORMAT = "%a %d %b %Y %H:%M"


def _format_telemetry_value(value) -> str | None:
    """Render a leaf telemetry value as a string, or None to suppress the field.

    Suppresses null/empty values so absent fields do not clutter the output.
    Booleans render as ``true``/``false`` (LLM-friendly, matches JSON).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value if value else None
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value) if value else None
    if isinstance(value, dict):
        # Nested dicts should have been split into separate rows by the
        # flattener; if one slips through, JSON-encode as a fallback.
        return json.dumps(value, separators=(",", ":")) if value else None
    return str(value)


def _is_hidden_telemetry_key(key: str) -> bool:
    return key in _TELEMETRY_HIDDEN_KEYS or key.startswith("_")


def _render_dict_subfields(d: dict) -> list[str]:
    """Render the visible fields of a nested telemetry dict as ``key:value`` strings."""
    sub_fields = []
    for sub_key, sub_value in d.items():
        if _is_hidden_telemetry_key(sub_key):
            continue
        rendered = _format_telemetry_value(sub_value)
        if rendered is not None:
            sub_fields.append(f"{sub_key}:{rendered}")
    return sub_fields


def _group_telemetry(ctx: dict) -> list[tuple[str, list[str]]]:
    """Group a nested telemetry dict into ``[(group_label, [k:v, ...])]`` entries.

    - Top-level scalar keys collect under the synthetic ``user`` group.
    - Top-level dict keys form their own groups (``device``, ``behavioral``, …).
    - Hidden bookkeeping keys (``saved_at``, ``_location_name_stale``) drop.
    - Empty groups are omitted entirely.
    """
    user_fields: list[str] = []
    grouped: dict[str, list[str]] = {}

    for key, value in ctx.items():
        if _is_hidden_telemetry_key(key):
            continue
        if isinstance(value, dict):
            if key in _TELEMETRY_HIDDEN_GROUPS:
                continue
            sub_fields = _render_dict_subfields(value)
            if sub_fields:
                grouped[key] = sub_fields
            continue
        rendered = _format_telemetry_value(value)
        if rendered is not None:
            user_fields.append(f"{key}:{rendered}")

    out: list[tuple[str, list[str]]] = []
    if user_fields:
        out.append(("user", user_fields))
    for group_name in sorted(grouped.keys()):
        out.append((group_name, grouped[group_name]))
    return out


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


def _keep_extreme(bucket: dict, message: str, row: dict, *, ts_key: str, prefer_lower: bool) -> None:
    """Insert ``row`` into ``bucket[message]`` if it has the more extreme timestamp under ``ts_key``."""
    current = bucket.get(message)
    if current is None:
        bucket[message] = row
        return
    incoming_ts = row.get(ts_key) or ""
    current_ts = current.get(ts_key) or ""
    if (prefer_lower and incoming_ts < current_ts) or (not prefer_lower and incoming_ts > current_ts):
        bucket[message] = row


def _bucket_schedule_rows(rows: list) -> tuple[dict, dict]:
    """Split rows into pending-by-message (earliest due_at) and fired-by-message (latest last_fired_at)."""
    pending_by_msg: dict[str, dict] = {}
    fired_by_msg: dict[str, dict] = {}
    for row in rows:
        message = row.get("message") or ""
        status = row.get("status")
        if status == "pending":
            _keep_extreme(pending_by_msg, message, row, ts_key="due_at", prefer_lower=True)
        elif status == "fired":
            _keep_extreme(fired_by_msg, message, row, ts_key="last_fired_at", prefer_lower=False)
    return pending_by_msg, fired_by_msg


def _order_schedule_messages(pending: dict, fired: dict) -> list[str]:
    """Pending entries first (by due_at asc), then fired-only entries (by last_fired_at desc)."""
    ordered = sorted(pending.keys(), key=lambda m: pending[m].get("due_at") or "")
    seen = set(ordered)
    ordered.extend(sorted(
        (m for m in fired if m not in seen),
        key=lambda m: fired[m].get("last_fired_at") or "",
        reverse=True,
    ))
    return ordered


def _compute_local_time() -> str | None:
    """Render the user's wall-clock time as ``Sat 02 May 2026 11:35``."""
    try:
        from services.locale_service import format_date
        from services.time_utils import utc_now
        return format_date(utc_now(), _LOCAL_TIME_FORMAT, for_ui=True)
    except Exception as exc:
        logger.debug("[WorldState] local_time compute failed: %s", exc)
        return None


def _render_schedule_fields(row: dict, now) -> str:
    """Build the comma-separated field string for one schedule entry."""
    fields = [_schedule_due_field(row.get("due_at") or "", now)]
    last_fired_field = _schedule_last_fired_field(row.get("last_fired_at"))
    if last_fired_field:
        fields.append(last_fired_field)
    recurrence_field = _schedule_recurrence_field(row.get("recurrence"))
    if recurrence_field:
        fields.append(recurrence_field)
    return ",".join(fields)


@dataclass(frozen=True)
class Signal:
    """Typed event pushed by an interface. Short-lived, absorbed and discarded."""

    source: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    received_at: datetime = field(default_factory=utc_now)


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

    def absorb(self, signal: Signal) -> None:
        """Process an incoming typed signal. Updates the snapshot fields atomically.

        Recognised kinds:
        - "user_message" -> updates last_user_message_at
        - "heartbeat"    -> updates last_heartbeat_at
        - "device"       -> sets current_device_class from payload['device_class']
        - "local_time"   -> sets current_local_time from payload['local_time']

        Unknown kinds are silently ignored (forward-compatibility).
        """
        with self._lock:
            if signal.kind == "user_message":
                self._store["world_state:last_user_message_at"] = signal.received_at.isoformat()
            elif signal.kind == "heartbeat":
                self._store["world_state:last_heartbeat_at"] = signal.received_at.isoformat()
            elif signal.kind == "device":
                dc = signal.payload.get("device_class")
                if dc:
                    self._store["world_state:current_device_class"] = dc
            elif signal.kind == "local_time":
                lt = signal.payload.get("local_time")
                if lt:
                    self._store["world_state:current_local_time"] = (
                        lt if isinstance(lt, str) else lt.isoformat()
                    )

    def snapshot(self) -> dict[str, Any]:
        """Read-only snapshot of the four typed ambient fields. Caller treats as immutable.

        Datetime fields are ``None`` when not yet set; once set they return a
        timezone-aware UTC ``datetime``.
        """
        with self._lock:
            raw_msg = self._store.get("world_state:last_user_message_at")
            raw_hb = self._store.get("world_state:last_heartbeat_at")
            raw_lt = self._store.get("world_state:current_local_time")
            return {
                "last_user_message_at": parse_utc(raw_msg) if raw_msg is not None else None,
                "last_heartbeat_at": parse_utc(raw_hb) if raw_hb is not None else None,
                "current_device_class": self._store.get("world_state:current_device_class"),
                "current_local_time": parse_utc(raw_lt) if raw_lt is not None else None,
            }

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
            parts.extend(telemetry_lines)

        # ── Schedule ───────────────────────────────────────────────────────
        schedule_lines = self._render_schedule()
        if schedule_lines:
            parts.append("[schedule]")
            parts.extend(schedule_lines)

        # ── Signals ────────────────────────────────────────────────────────
        signal_lines = self._render_signals()
        parts.extend(signal_lines)

        if not parts:
            return ""

        return _SECTION_HEADER + "\n" + "\n".join(parts)

    # ── Private render helpers ─────────────────────────────────────────────

    def _render_telemetry(self) -> list[str]:
        """Produce bullet lines for the [telemetry] section.

        Reads the latest heartbeat from the ``telemetry`` table (populated by
        ``ClientContextService.save()``) and surfaces every key the frontend
        sent, grouped by top-level prefix.  Top-level scalar keys aggregate
        under the synthetic ``user`` group; nested dicts (``device`` …) form
        their own groups.  ``local_time`` is overwritten with a freshly-computed
        value derived from the stored IANA timezone so it never goes stale.
        """
        from services.heartbeat_service import heartbeat_service
        ctx = dict(heartbeat_service.read())  # shallow copy — _render mutates local_time
        if not ctx:
            return []

        fresh_local_time = _compute_local_time()
        if fresh_local_time:
            ctx["local_time"] = fresh_local_time

        lines = []
        for group_name, fields in _group_telemetry(ctx):
            lines.append(f"* **{group_name}**;" + ",".join(fields))
        return lines

    def _render_schedule(self) -> list[str]:
        """Produce bullet lines for the [schedule] section from scheduled_items.

        Per unique message:
          1. If any pending row exists, surface the earliest-due one.
          2. Else, surface the most-recent fired row (SQL already constrains
             fired rows to the last 6 hours; older fires are excluded).
        """
        rows = _fetch_schedule_rows()
        if not rows:
            return []

        pending_by_msg, fired_by_msg = _bucket_schedule_rows(rows)
        ordered = _order_schedule_messages(pending_by_msg, fired_by_msg)

        now = utc_now()
        lines = []
        for message in ordered:
            row = pending_by_msg.get(message) or fired_by_msg[message]
            lines.append(f"* {message} ({_render_schedule_fields(row, now)})")
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



# ── Module-level singleton ─────────────────────────────────────────────────

world_state = WorldState()
