# Ambient Awareness

## Overview

Chalie continuously infers context from browser telemetry and behavioral signals — without asking, without polling, and without any LLM calls at inference time. This ambient layer gives the cognitive runtime situational awareness: where the user is, how focused they are, what device they are on, and how much energy they have. Autonomous actions and cognitive drift use this context to decide when to act and when to stay silent.

The inference engine runs in under 1ms with zero LLM involvement. Context awareness must never introduce latency or cost into the critical path.

---

## Six Dimensions

| Dimension | What It Captures | Example Values |
|-----------|-----------------|----------------|
| `place` | Where the user likely is | `home`, `work`, `transit`, `out` |
| `attention` | How focused the user appears | `deep_focus`, `casual`, `distracted`, `away` |
| `energy` | Estimated cognitive/physical energy | `high`, `moderate`, `low` |
| `mobility` | Whether the user is moving | `stationary`, `commuting`, `traveling` |
| `tempo` | Pace of interaction | `rushed`, `relaxed`, `reflective` |
| `device_context` | Narrative description of device state | `"morning work session"`, `"evening commute"` |

All classification is rule-based and deterministic. When a dimension value changes, a transition event is emitted to the Event Bridge for further gating.

---

## Place Learning

Geolocation is never used at GPS coordinate level. Place inference starts from coarse signals — network fingerprint, timezone, battery behavior, interaction patterns — and is refined over time through behavioral observation.

Place fingerprints accumulate at geohash precision (~1km cell, never raw coordinates). Each fingerprint is a cluster of ambient signals associated with a location. After enough observations for a given cluster, the learned pattern overrides the initial heuristic inference. Chalie learns that "this network + this timezone + this battery pattern = home" without any explicit user configuration.

---

## Session Context

On each connection and reconnection, the system evaluates session-level context beyond moment-to-moment inference:

- **Re-entry detection** — a return after 30+ minutes of absence emits a `session_resume` event
- **Transition detection** — a sustained place change triggers a place transition event
- **Locale-derived trait seeding** — browser locale informs approximate region and language on first session, stored as low-confidence traits, not assertions
- **Circadian data** — hourly interaction counts are recorded for behavioral pattern mining

---

## Event Bridge and Gates

Transition events do not trigger autonomous actions directly. Every event passes through a gating layer before anything fires:

| Gate | Purpose |
|------|---------|
| Stabilization window (90s) | A dimension must hold its new value for 90s before it is treated as real — filters transient noise |
| Per-event cooldowns | Prevents the same event type from firing repeatedly within a cooldown window |
| Confidence gating | Low-confidence inferences do not trigger actions |
| Aggregation window (60s) | Multiple events within 60s are bundled into a single action trigger |
| Focus gate | When attention is `deep_focus`, all autonomous event-driven actions are suppressed entirely |

The sequence is: telemetry arrives, dimensions are classified, a transition is detected, the event enters the gate pipeline, and only if all gates pass does an autonomous action fire.

---

## Attention Protection

The ambient layer exists to protect attention, not to create more interruptions. The focus gate is the most important gate in the pipeline: when the user is in deep focus — high interaction rate, low pause duration, single sustained topic — all event-driven autonomous actions are suppressed entirely. Chalie does not interrupt deep work.

Reducing noise is as valuable as completing tasks.

---

## Privacy Model

- Geolocation is never stored at GPS precision — only geohash (~1km cell)
- Place fingerprints store signal clusters, not raw coordinates
- All data remains local (SQLite); ambient telemetry never leaves the device
- Locale and demographic traits are stored as low-confidence inferences, not assertions
- Any learned trait can be deleted via the Brain dashboard or the privacy API
