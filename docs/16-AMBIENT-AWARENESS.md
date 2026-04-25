# WorldState — Ambient Signal Snapshot

## Overview

Chalie maintains a lightweight, in-process ambient snapshot that tracks four typed facts about the current session without any LLM calls, without any database writes, and without holding locks longer than a dict lookup.

The snapshot is populated by `world_state.absorb(signal)` whenever a typed `Signal` arrives from an interface. The result is available to any turn via `world_state.snapshot()`.

---

## Signal Types

| `kind` | Source | What it captures |
|--------|--------|-----------------|
| `user_message` | WebSocket `_handle_chat` | Timestamp of the most recent user turn |
| `heartbeat` | POST `/health` | Timestamp of the most recent client heartbeat |
| `device` | POST `/health` | Current device class (e.g. `phone`, `desktop`, `tablet`) |
| `local_time` | POST `/health` | Client-reported local time as an ISO string |

Unknown `kind` values are silently ignored — the system is forward-compatible with new signal types without any code change.

---

## Signal Dataclass

```python
@dataclass(frozen=True)
class Signal:
    """Typed event pushed by an interface. Short-lived, absorbed and discarded."""
    source: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    received_at: datetime = field(default_factory=utc_now)
```

Signals are immutable and carry a `received_at` timestamp automatically set to `utc_now()` at construction time. Interfaces that need to override the timestamp (e.g. for replayed events) can pass `received_at` explicitly.

---

## WorldState API

```python
# Push an event — modifies the in-process snapshot
world_state.absorb(Signal(source='ws', kind='user_message', payload={'text': text[:200]}))

# Read the snapshot — safe to call from any thread
snap = world_state.snapshot()
# snap['last_user_message_at']  → datetime (UTC) or None
# snap['last_heartbeat_at']     → datetime (UTC) or None
# snap['current_device_class']  → str or None
# snap['current_local_time']    → datetime (UTC) or None
```

All four snapshot fields are `None` when no signal of that kind has been absorbed since boot. Callers must treat `None` as "not yet known" rather than any sentinel datetime.

---

## Where absorb() is Called

| Call site | Signal kind | Trigger |
|-----------|-------------|---------|
| `backend/api/websocket.py` `_handle_chat()` | `user_message` | User sends a chat message |
| `backend/api/system.py` POST `/health` | `heartbeat` | Any heartbeat from the client |
| `backend/api/system.py` POST `/health` | `device` | Heartbeat payload includes `device_class` |
| `backend/api/system.py` POST `/health` | `local_time` | Heartbeat payload includes `local_time` |

Each call is wrapped in `try/except` so a WorldState error never surfaces to the interface layer.

---

## Design Constraints

- **Zero DB.** `absorb()` and `snapshot()` operate entirely on `world_state._store` (an in-process dict). No SQLite writes. No MemoryStore writes.
- **Thread-safe.** Both methods acquire `world_state._lock` for the duration of the operation. Lock is never held across I/O.
- **Immutable signals.** `Signal` is `frozen=True`. The payload dict is shallow-copied by reference; do not mutate it after construction.
- **No inference.** The snapshot records what interfaces reported. No place inference, attention scoring, or energy estimation. Classification of what these facts mean is done at turn-assembly time by the caller of `snapshot()`.

---

## What Was Removed (v0.5.0)

Prior to v0.5.0, three services powered a more complex ambient layer:

- **`AmbientInferenceService`** — rule-based classifier that inferred place, attention, energy, mobility, and tempo from telemetry signals.
- **`SituationModelService`** — assembled inferences into a structured situation snapshot and persisted it to MemoryStore.
- **`PlaceLearningService`** — accumulated place fingerprints in SQLite (`place_fingerprints` table, now auto-dropped by SchemaConvergenceService).

These three services were removed entirely. Their replacement is `Signal` + `absorb()` + `snapshot()`: four typed fields, zero inference at ingest time, zero DB overhead.
