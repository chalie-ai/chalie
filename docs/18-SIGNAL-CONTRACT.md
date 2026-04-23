# Signal Contract — Continuous Reasoning Spine

This document defines the contract governing Chalie's signal-driven architecture. It is the governing spec for all service integration decisions.

---

## Governing Principles

### Simplicity Over Cleverness

Every service must be as minimal as possible. Complexity compounds across 40+ services. When in doubt, do less.

### Graceful Isolation

**"Forgetting my name for a split-second doesn't put me in a vegetative state."**

No service failure may cascade into other services. Every signal consumer must operate under the assumption that any signal source may be dead, delayed, or producing garbage. The system degrades gracefully — individual capabilities may temporarily weaken, but the core reasoning loop never stops.

Concrete rules:

- Every signal consumption is wrapped in try/except at the boundary
- Every service has a fail-open default: if it cannot do its job, it returns a neutral result (empty string, no-op, skip), never raises into the caller
- No service holds locks that other services need
- No service writes state that another service must read to function — MemoryStore state is advisory, never mandatory
- "No signal" means "nothing interesting happened", not "something is broken"

### Independent Testability

Every service must be testable in complete isolation. Unit tests use in-memory storage with no shared state and no dependency on another service being initialized. Integration between services is tested by the nightly suite, not by unit tests.

---

## Service Layers

Every service belongs to exactly one of three layers. Failures are contained within a layer — they never cascade across layer boundaries.

| Layer | Analogy | What it does | If it fails... |
|-------|---------|--------------|----------------|
| **Cognitive** | Brain | Memory formation, consolidation, decay, planning, reflection, reasoning | ...you stop reasoning well, but you can still perceive and use tools |
| **Embodiment** | Body/Senses | Ambient awareness, place learning, context tracking, voice I/O, session events | ...you lose situational awareness, but the reasoning loop continues on what it knows |
| **Capability** | Tools/Hands | External tools, document processing, scheduling, list management | ...you lose specific abilities, but you find alternatives or report inability |

Cross-layer rules:

- Cognitive services never import embodiment or capability services at module level — lazy imports only
- Embodiment services write to MemoryStore; cognitive services read from MemoryStore. No direct calls between layers.
- Capability failures surface as "tool unavailable" — the cognitive layer plans around them, never crashes
- A full embodiment outage means ambient signals stop arriving; the cognitive layer treats this as idle, not as an error

---

## Signal Envelope

Signals carry references and summaries, not fat payloads. Every signal includes:

- `signal_type` — what happened (see registered types below)
- `source` — which service emitted it
- `concept_id` / `concept_name` — direct reference to the relevant knowledge node
- `topic` — domain context
- `content` — freeform summary, under 200 characters
- `activation_energy` — 0.0–1.0, how important or urgent
- `timestamp` — when emitted

Emission is always fire-and-forget. Emitters never wait for a response and never instantiate the consumer. A failed emit is logged at DEBUG level and never raised.

---

## Registered Signal Types

| Signal Type | Meaning | Energy Range |
|-------------|---------|--------------|
| `memory_pressure` | Knowledge is fading or contradicted | 0.5–0.7 |
| `new_knowledge` | New concept formed from experience | 0.6 |
| `novel_observation` | Surprising tool output stored to data graph | 0.6 |
| `ambient_context` | Environment changed (place, attention, energy) | From confidence |
| `idle_discovery` | Nothing happened; engine self-seeds | 0.4–0.5 |
| `episode_created` | New narrative episode consolidated | 0.5 |
| `trait_changed` | User trait created, updated, or corrected | 0.3–0.7 |
| `task_state_changed` | Persistent task state transition | 0.5–0.6 |
| `schedule_fired` | Scheduled reminder or task fired | 0.5 |
| `thread_expired` | Conversation thread expired | 0.3 |
| `user_message` | User sent a chat message | 1.0 |
| `goal_inferred` | Recurring topic pattern detected as potential goal | 0.6 |

Adding a new signal type requires: an entry in this table, a nightly test scenario, and documentation of what the consumer should do with it.

---

## Signal Transport

- **Priority queue** — user messages are processed first
- **Background queue** — all other signal types
- **Max depth** — 50 signals; oldest dropped on overflow (background queue only)
- **Debounce** — 30s minimum between processed background signals; user messages bypass
- **Yield points** — background signal processing checks the priority queue before expensive operations; if a user message is waiting, background reasoning aborts

---

## Service Health

Services write periodic heartbeats keyed by service name with a TTL of 2x their expected cycle time. The self-model aggregates these and includes dead services in its noteworthy list. Health is observational, not coercive — no automated restart.

---

## Migration State

Migration from timer-based to signal-driven architecture is partial. The reasoning loop is fully signal-driven. Most other services still run on timers and emit signals on relevant events. Services that run on long natural cycles (6h, 30min) are good candidates to remain timer-driven indefinitely.

---

## Anti-Patterns

**Signal cascades** — Service A emits a signal, B processes it and emits, C processes it and emits back toward A. No circular signal paths. Draw the signal graph before adding a new emission point.

**Signal as RPC** — emitting a signal and waiting for a response. Signals are fire-and-forget. If a response is needed, use a direct function call or a dedicated result queue.

**Mandatory signals** — a service that crashes or errors if it does not receive a signal within N seconds. Timeouts trigger idle or maintenance behavior, not error states.

**Fat signals** — payloads containing full episode text, embeddings, or large data structures. Signals carry references and short summaries; consumers look up full data from storage.

**Signal-driven configuration** — using signals to propagate config changes. Config is read from files or the database at service init or on a slow reload cycle. Signals carry cognitive events, not infrastructure state.

---

## The Spine

Current design: single consumer, single queue. Multi-consumer routing — where each service registers interest in specific signal types — is deferred until enough services are signal-driven that routing becomes necessary. Build it when you need it.
