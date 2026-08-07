import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import cast

from services.durable_timestamp import DurableTimestamp
from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)

_SECTION_HEADER = "### Background Telemetry,Processes"

# Key for the last user-message timestamp. The in-memory ``_store`` dict and the
# durable MemoryStore deliberately share this key — both are the fast in-process
# read path for the same value.
_STORE_KEY_LAST_USER_MESSAGE = "world_state:last_user_message_at"

# Durable dual-write clock so the subconscious user-active gate survives a
# process/container restart. Without it the gate starves: the value lives only
# in the in-memory store and is wiped on every restart.
# Bidirectional dependency: services/durable_timestamp.py owns the persist/
# hydrate mechanism; this module supplies the key pair + provenance.
_DG_KEY_LAST_USER_MESSAGE = "world_state_last_user_message_at"
_SOURCE_LAST_USER_MESSAGE = "world_state"


# ── Render helpers (module-level, pure functions) ─────────────────────────────


# Top-level telemetry keys that should not be surfaced in the rendered block —
# they are internal bookkeeping or noise the LLM does not need.
_TELEMETRY_HIDDEN_KEYS = {"saved_at", "_location_name_stale", "connection"}

# Top-level dict groups that should not be rendered as their own bullet.
# ``location`` carries the raw GPS dict (lat/lon) the frontend heartbeat sends;
# it stays out of the chat/system prompt. Backend consumers read the coordinates
# directly (departure advisory, weather, locale_service); the chat LLM only ever
# sees the resolved ``location_name`` scalar, which renders under the synthetic
# ``user`` group. (The background geo-pattern pass — configs/channels/geo_pattern.py
# — is the one model-facing consumer still given coordinates, to cluster
# location-tagged transcripts into place-based habits.)
_TELEMETRY_HIDDEN_GROUPS = {"behavioral", "location"}

# Strftime format for the synthesised local_time field — "Sat 02 May 2026 11:35".
_LOCAL_TIME_FORMAT = "%a %d %b %Y %H:%M"


def _is_hidden_telemetry_key(key: str) -> bool:
    return key in _TELEMETRY_HIDDEN_KEYS or key.startswith("_")


def _render_dict_subfields(d: dict[str, object]) -> list[str]:
    sub_fields = []
    for sub_key, sub_value in d.items():
        if _is_hidden_telemetry_key(sub_key):
            continue
        rendered = WorldState._format_telemetry_value(sub_value)
        if rendered is not None:
            sub_fields.append(f"{sub_key}:{rendered}")
    return sub_fields


def _group_telemetry(ctx: dict[str, object]) -> list[tuple[str, list[str]]]:
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
        rendered = WorldState._format_telemetry_value(value)
        if rendered is not None:
            user_fields.append(f"{key}:{rendered}")

    out: list[tuple[str, list[str]]] = []
    if user_fields:
        out.append(("user", user_fields))
    for group_name in sorted(grouped.keys()):
        out.append((group_name, grouped[group_name]))
    return out


@dataclass(frozen=True)
class Signal:
    source: str
    kind: str
    payload: dict[str, object] = field(default_factory=dict)
    received_at: datetime = field(default_factory=utc_now)


class WorldState:
    """In-process singleton. Sole owner of world-state data + rendering.

    Thread-safe via a single internal lock protecting ``_store``.
    """

    def __init__(self) -> None:
        """Build a WorldState and hydrate the restart-durable fields.

        ``last_user_message_at`` is loaded from the dual-write store at
        construction so the first read after a process/container restart sees
        the persisted value (mirrors ``IdleGatedJob``'s last-fired hydrate). Hydrate
        failure is non-fatal — the field simply starts unset.
        """
        self._store: dict[str, object] = {}           # arbitrary type → dict fragments
        self._lock = threading.Lock()
        self._user_msg_clock = DurableTimestamp(
            memory_key=_STORE_KEY_LAST_USER_MESSAGE,
            data_graph_key=_DG_KEY_LAST_USER_MESSAGE,
            source=_SOURCE_LAST_USER_MESSAGE,
        )
        self._hydrate_last_user_message_at()

    # ── Public API ─────────────────────────────────────────────────────────

    def set(self, type: str, value: dict[str, object]) -> "WorldState":
        with self._lock:
            self._store[type] = value
        return self

    def get(self, type: str) -> dict[str, object]:
        with self._lock:
            return dict(cast("dict[str, object]", self._store.get(type)) or {})

    def absorb(self, signal: Signal) -> None:
        """Process an incoming typed signal. Updates the snapshot fields atomically.

        Recognised kinds:
        - "user_message" -> updates last_user_message_at
        - "heartbeat"    -> updates last_heartbeat_at
        - "device"       -> sets current_device_class from payload['device_class']
        - "local_time"   -> sets current_local_time from payload['local_time']

        Unknown kinds are silently ignored (forward-compatibility).
        """
        persist_user_message: datetime | None = None
        with self._lock:
            if signal.kind == "user_message":
                self._store[_STORE_KEY_LAST_USER_MESSAGE] = signal.received_at.isoformat()
                persist_user_message = signal.received_at
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
                        lt if isinstance(lt, str) else cast("datetime", lt).isoformat()
                    )

        # Durable write happens outside the lock — the dual-write touches
        # MemoryStore + data_graph and must not block other absorb/snapshot
        # callers. The in-memory store is already updated above; persistence is
        # the restart-survival copy the subconscious user-active gate reads.
        if persist_user_message is not None:
            self._user_msg_clock.persist(persist_user_message)

    def _hydrate_last_user_message_at(self) -> None:
        """Load the durable last-user-message timestamp into the in-memory store.

        Called once from ``__init__`` so a restarted process sees the persisted
        value on its first read. The durable read happens outside the snapshot
        hot path; failure is non-fatal (the field starts unset).
        """
        try:
            hydrated = self._user_msg_clock.load()
        except Exception as exc:
            logger.warning("[WorldState] hydrate last_user_message_at failed: %s", exc)
            return
        if hydrated is not None:
            with self._lock:
                self._store[_STORE_KEY_LAST_USER_MESSAGE] = hydrated.isoformat()

    def snapshot(self) -> dict[str, object]:
        """Read-only snapshot of the four typed ambient fields. Caller treats as immutable.

        Datetime fields are ``None`` when not yet set; once set they return a
        timezone-aware UTC ``datetime``. ``last_user_message_at`` is hydrated
        from durable storage at construction, so the in-memory store is the
        single read source here even after a restart.
        """
        with self._lock:
            raw_msg = self._store.get(_STORE_KEY_LAST_USER_MESSAGE)
            raw_hb = self._store.get("world_state:last_heartbeat_at")
            raw_lt = self._store.get("world_state:current_local_time")
            return {
                "last_user_message_at": parse_utc(cast("str", raw_msg)) if raw_msg is not None else None,
                "last_heartbeat_at": parse_utc(cast("str", raw_hb)) if raw_hb is not None else None,
                "current_device_class": self._store.get("world_state:current_device_class"),
                "current_local_time": parse_utc(cast("str", raw_lt)) if raw_lt is not None else None,
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

        fresh_local_time = WorldState._compute_local_time()
        if fresh_local_time:
            ctx["local_time"] = fresh_local_time

        lines = []
        for group_name, fields in _group_telemetry(ctx):
            lines.append(f"* **{group_name}**;" + ",".join(fields))
        return lines

    # ── Static render helpers ──────────────────────────────────────────────

    @staticmethod
    def _format_telemetry_value(value: object) -> str | None:
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

    @staticmethod
    def _compute_local_time() -> str | None:
        """Return wall-clock time formatted as ``Sat 02 May 2026 11:35``."""
        try:
            from services.locale_service import LocaleService
            from services.time_utils import utc_now
            return LocaleService.format_date(utc_now(), _LOCAL_TIME_FORMAT, for_ui=True)
        except Exception as exc:
            logger.debug("[WorldState] local_time compute failed: %s", exc)
            return None


world_state = WorldState()
