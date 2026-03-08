# Ambient Awareness

## Overview

Chalie continuously infers context from browser telemetry and behavioral signals — without asking, without polling, and without any LLM calls at inference time. This ambient layer gives the cognitive runtime situational awareness: where the user is, how focused they are, what device they're on, and how much energy they have. Autonomous actions and cognitive drift use this context to decide when to act and when to stay silent.

**Design constraint:** The inference engine runs in <1ms with zero LLM involvement. Context awareness must never introduce latency or cost into the critical path.

---

## Services

### Ambient Inference Service (`services/ambient_inference_service.py`)

The core deterministic inference engine. Reads browser telemetry and behavioral signals and classifies six context dimensions:

| Dimension | What It Captures | Example Values |
|-----------|-----------------|----------------|
| `place` | Where the user likely is | `home`, `work`, `transit`, `unknown` |
| `attention` | How focused the user appears | `deep_focus`, `active`, `idle`, `distracted` |
| `energy` | Estimated cognitive/physical energy | `high`, `moderate`, `low` |
| `mobility` | Whether the user is moving | `stationary`, `walking`, `transit` |
| `tempo` | Pace of interaction | `slow`, `normal`, `fast` |
| `device_context` | Device type and state | `desktop`, `mobile`, `battery_low`, etc. |

All classification is rule-based, driven by thresholds loaded from `configs/agents/ambient-inference.json`. When `emit_events=True`, the service emits transition events (place change, attention shift, energy change) to the Event Bridge whenever a dimension value changes.

**Key principle:** Geolocation is never used at the GPS coordinate level. Place inference uses coarse signals (network, timezone, battery behavior) and is refined by the Place Learning Service using geohash (~1km precision) after enough observations accumulate.

---

### Place Learning Service (`services/place_learning_service.py`)

Accumulates place fingerprints in the `place_fingerprints` SQLite table. Each fingerprint is a cluster of ambient signals associated with a geohash (~1km cell, never raw coordinates). After 20+ observations for a given fingerprint cluster, the learned pattern overrides the heuristic inference from the Ambient Inference Service.

Over time, Chalie learns that "this network + this timezone + this battery pattern = home" without any explicit user configuration. Place knowledge is entirely emergent from behavioral observation.

---

### Client Context Service (`services/client_context_service.py`)

Manages richer session-level context beyond the moment-to-moment ambient inference:

- **Location history ring buffer** — stores the last 12 place inferences; used to detect transitions
- **Place transition detection** — fires when place dimension changes (e.g., `home → work`)
- **Session re-entry detection** — detects when a user returns after >30 minutes absence; emits `session_resume` event
- **Demographic trait seeding** — infers approximate locale metadata (language, region) from browser locale on first session; stored as low-confidence user traits
- **Circadian data collection** — records hourly interaction counts for behavioral pattern mining

Emits `session_start` and `session_resume` events to the Event Bridge on each WebSocket connection.

---

### Event Bridge Service (`services/event_bridge_service.py`)

The connector layer between ambient context changes and autonomous actions. Raw transition events from the Ambient Inference Service and Client Context Service are gated before any action is taken:

| Gate | Purpose |
|------|---------|
| **Stabilization window (90s)** | A dimension must hold its new value for 90s before it's treated as real — filters transient noise |
| **Per-event cooldowns** | Prevents the same event type from firing repeatedly within a cooldown window |
| **Confidence gating** | Low-confidence inferences don't trigger actions |
| **Aggregation window (60s)** | Multiple events within 60s are bundled into a single action trigger |
| **Focus gate** | When attention = `deep_focus`, all autonomous event-driven actions are suppressed |

Configuration in `configs/agents/event-bridge.json`.

---

## Data Flow

```
Browser telemetry (battery, network, locale, interaction rate)
  |
  v
[Ambient Inference Service] — deterministic, <1ms, zero LLM
  |
  +-- Current context dimensions → injected into context assembly
  |
  +-- Transition events (when a dimension changes)
        |
        v
      [Event Bridge Service]
        |
        +-- Stabilization window (90s)
        +-- Confidence gate
        +-- Focus gate (suppressed if deep_focus)
        +-- Aggregation window (60s bundle)
        |
        v
      Autonomous action dispatch → CognitiveDriftEngine / ProactiveAction

[Place Learning Service] — background, accumulates fingerprints
  |
  +-- After 20+ observations: learned pattern overrides heuristics

[Client Context Service] — fires on WebSocket connect/reconnect
  |
  +-- session_start / session_resume events → Event Bridge
  +-- Location history, circadian counts → behavioral pattern mining
```

---

## Attention Protection

The ambient layer exists to protect attention, not to create more interruptions. The focus gate is the most important gate in the Event Bridge: when the user is in deep focus (high interaction rate, low pause duration, single sustained topic), all event-driven autonomous actions are suppressed entirely. Chalie does not interrupt deep work.

This reflects a core design principle: reducing noise is as valuable as completing tasks.

---

## Privacy Model

- Geolocation is never stored at GPS precision — only geohash (~1km cell)
- Place fingerprints store signal clusters, not raw coordinates
- All data remains local (SQLite); ambient telemetry never leaves the device
- Locale/demographic traits are stored as low-confidence inferences, not assertions
- Users can delete any learned trait via `DELETE /system/observability/traits/<key>`

---

## Related

- **`07-COGNITIVE-ARCHITECTURE.md`** — how attention state gates the mode router and cognitive drift
- **`06-WORKERS.md`** — Client Context Service and Event Bridge lifecycle details
- **`04-ARCHITECTURE.md`** — full service listing under "Ambient Awareness"
