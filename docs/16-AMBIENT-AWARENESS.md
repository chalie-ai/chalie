# Signals & Ambient Awareness

Chalie keeps a lightweight, in-process picture of "what's going on right now" in **WorldState** (`backend/services/world_state.py`). It feeds two things: the `### Background Telemetry, Processes & Signals` block rendered into every system prompt, and the idle gate that decides when background cognition may run.

> **Location privacy.** The telemetry block rendered into every chat/system prompt surfaces the *resolved place name* (e.g. `location_name:Valletta, Malta`), never the raw GPS coordinates the client reports. The latitude/longitude pair the render hides stays backend-internal — consumed directly by the departure advisory, weather lookups, and `locale_service`. The one model-facing consumer of coordinates is the background **geo-pattern** pass, which runs only in the subconscious tick (never a user-facing turn) and is given them to cluster location-tagged transcripts into place-based habits.

There are two ways information enters WorldState.

## 1. The Typed Snapshot — `absorb(Signal)`

Four typed facts about the current session, updated by internal code:

```python
from services.world_state import world_state, Signal

world_state.absorb(Signal(source="api", kind="device", payload={"device_class": "phone"}))

snap = world_state.snapshot()
# snap["last_user_message_at"]   → datetime (UTC) or None
# snap["last_heartbeat_at"]      → datetime (UTC) or None
# snap["current_device_class"]   → str or None
# snap["current_local_time"]     → datetime (UTC) or None
```

| `kind` | Written by | Captures |
|---|---|---|
| `user_message` | The chat turn pipeline | When the user last spoke — **persisted durably** (MemoryStore + data graph) so it survives restarts |
| `heartbeat` | `POST /health` (client heartbeat, ~every 5 min) | Last contact from the client |
| `device` | `POST /health` | Current device class (`phone`, `desktop`, …) |
| `local_time` | `POST /health` | Client-reported local time |

Unknown kinds are ignored, so the snapshot is forward-compatible. The subconscious worker reads `last_user_message_at` to enforce its 30-minute idle gate.

## 2. Freeform Signals — `push_signal()`

Short text signals that appear in the system prompt as `[signal:<source>] <label>` lines:

```python
world_state.push_signal("news", "Tech: new EU AI rules announced", ttl=3600)
```

- **One slot per source** — a new signal from the same source replaces the previous one.
- **TTL** defaults to 1 hour; expired signals are pruned lazily on read.
- Used internally by the hourly news scan and the mail monitor; available to any service.

### External signal API

Outside processes can push signals over HTTP:

```
POST /api/signals
{
  "signal_type": "ambient_context",      // advisory label — does not affect routing
  "source": "my_service",                // becomes the signal slot key
  "content": "Build #4812 failed on main" // becomes the rendered label
}
```

- Auth: session cookie, or a bearer token whose wrapper grants the `signal_type` (or `*`) under `capabilities.signals`.
- Rate limit: 100 signals/min per wrapper. Batch endpoint: `POST /api/signals/batch` (max 50 per request).
- Every accepted signal resolves to `push_signal(source, content, ttl=3600)` — there is no queue, no routing by type, and no per-type consumer. The signal simply becomes ambient context the next time a prompt is assembled.

## What Runs in the Background

| Service | Cadence | What it does |
|---|---|---|
| **Subconscious worker** | Every 5 min, fires only after 30+ min of user idleness | The eight-step cognition tick — consolidation, decay, pattern matching, user-summary synthesis, DMN reflection, capability sync, geo patterns, proactive research (see [04-ARCHITECTURE.md](04-ARCHITECTURE.md#background-cognition)) |
| **World awareness** | Hourly | Derives up to 8 interests from the user's strongest traits and recent topics, fetches matching headlines, pushes a `news` signal — zero LLM calls |
| **Decay engine** | Inside each subconscious tick | Recomputes episode retrieval weights, applies per-kind data-graph decay, deletes expired rows, prunes old transcripts and tool-call records |

All background work degrades gracefully: every step is wrapped at its boundary, a failed step is logged and skipped, and a missing signal means "nothing interesting happened", never an error.

## Hooking In

- **Make ambient context visible to the model** — `push_signal()` (or `POST /api/signals` from outside the process). It will appear in the next assembled prompt for up to its TTL.
- **Read session state** — `world_state.snapshot()` from any thread.
- **Influence the idle gate** — the subconscious worker fires only when `last_user_message_at` is older than 30 minutes; user activity automatically suppresses it.
