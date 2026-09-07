import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import cast

from services.durable_timestamp import DurableTimestamp
from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class Signal:
    source: str
    kind: str
    payload: dict[str, object] = field(default_factory=dict)
    received_at: datetime = field(default_factory=utc_now)


class WorldState:
    """In-process singleton. Sole owner of world-state data.

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

        Unknown kinds are silently ignored (forward-compatibility) — in
        particular a stray "local_time" signal is dropped: the model's clock
        is the per-line message stamp, not the world state.
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
        """Read-only snapshot of the three typed ambient fields. Caller treats as immutable.

        Datetime fields are ``None`` when not yet set; once set they return a
        timezone-aware UTC ``datetime``. ``last_user_message_at`` is hydrated
        from durable storage at construction, so the in-memory store is the
        single read source here even after a restart.
        """
        with self._lock:
            raw_msg = self._store.get(_STORE_KEY_LAST_USER_MESSAGE)
            raw_hb = self._store.get("world_state:last_heartbeat_at")
            return {
                "last_user_message_at": parse_utc(cast("str", raw_msg)) if raw_msg is not None else None,
                "last_heartbeat_at": parse_utc(cast("str", raw_hb)) if raw_hb is not None else None,
                "current_device_class": self._store.get("world_state:current_device_class"),
            }


world_state = WorldState()
