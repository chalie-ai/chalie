# Signal Contract — Continuous Reasoning Spine

This document defines the contract that governs Chalie's transition from independent timer-based services to a unified signal-driven architecture. It is the governing spec for all migration work.

**Status:** Active — governs all service migration decisions.

---

## 1. Governing Principles

### 1.1 Simplicity Over Cleverness

Every service must be as minimal as possible. Complexity compounds across 40+ services — a 10% increase in complexity per service is a 40x increase in system debugging difficulty. When in doubt, do less.

### 1.2 Graceful Isolation

**"Forgetting my name for a split-second doesn't put me in a vegetative state."**

No service failure may cascade into other services. Every signal consumer must operate under the assumption that any signal source may be dead, delayed, or producing garbage. The system degrades gracefully — individual capabilities may temporarily weaken, but the core reasoning loop never stops.

Concrete rules:
- Every signal consumption is wrapped in try/except at the boundary
- Every service has a **fail-open** default: if it can't do its job, it returns a neutral result (empty string, no-op, skip), never raises into the caller
- No service holds locks that other services need
- No service writes state that another service must read to function (MemoryStore state is advisory, never mandatory)
- A service being dead means its signals stop arriving — consumers treat "no signal" as "nothing interesting happened", not as an error

### 1.3 Independent Testability

Every service must be testable in complete isolation:
- Unit tests use in-memory MemoryStore and `:memory:` SQLite — no shared state
- No test may depend on another service being initialized
- Every service is covered by at least one `chalie-nightly-test` blackbox scenario
- Integration between services is tested by the nightly suite, not by unit tests

### 1.4 Service Layers (Fault Domains)

Every service belongs to exactly one of three layers. Failures are contained within a layer — they never cascade across layer boundaries.

| Layer | Analogy | What it does | If it fails... |
|---|---|---|---|
| **Cognitive** | Brain | Reasoning, memory formation, consolidation, decay, planning, reflection | ...you stop reasoning well, but you still perceive and can still use tools |
| **Embodiment** | Body/Senses | Perception, ambient awareness, place learning, context tracking, voice I/O | ...you lose awareness of surroundings, but you can still think and act on what you know |
| **Capability** | Tools/Hands | External tools, document processing, scheduling, list management | ...you lose specific abilities, but you find alternatives or report inability |

**Cognitive services:**
DecayEngine, SemanticConsolidation, EpisodicMemoryWorker, MemoryChunker, ReasoningLoopService, ContextAssembly, CognitiveTriage, ModeRouter, PlanDecomposition, CriticService, UncertaintyService, ContradictionClassifier, IdleConsolidation, GrowthPattern, AutobiographySynthesis, CuriosityThread/Pursuit, RoutingReflection, SelfModel

**Embodiment services:**
AmbientInference, PlaceLearning, ClientContext, EventBridge, VoiceService, FolderWatcher, TemporalPattern, EpisodicMemoryObserver, ThreadExpiry

**Capability services:**
ToolRegistry, ToolWorker, ToolContainer, ToolConfig, ToolProfile, ToolPerformance, ToolUpdateChecker, ACTLoop, ACTDispatcher, DocumentService, DocumentProcessing, DocumentPurge, SchedulerService, ListService, PersistentTaskWorker, MomentEnrichment, ProfileEnrichment

**Cross-layer rules:**
- Cognitive services never import embodiment or capability services at module level (lazy imports only)
- Embodiment services write to MemoryStore; cognitive services read from MemoryStore. Never direct calls.
- Capability failures surface as "tool unavailable" — the cognitive layer plans around them, never crashes
- A full embodiment outage means ambient signals stop arriving. The cognitive layer treats this as "nothing interesting is happening" (idle), not as an error

### 1.5 Minimal Surface Area

Each service exposes the minimum interface needed:
- One public method for its primary job (e.g., `process()`, `consolidate()`, `decay()`)
- Signal emission is a side-effect, not the primary interface
- No service exposes internal state to other services except through MemoryStore advisory keys

---

## 2. Signal Envelope

All signals flowing through the spine use this format:

```python
@dataclasses.dataclass
class ReasoningSignal:
    signal_type: str          # What happened (see §3)
    source: str               # Who emitted it (service name)
    concept_id: int | None    # Direct concept reference (fast path)
    concept_name: str | None  # Human-readable label
    topic: str | None         # Domain/topic context
    content: str | None       # Freeform payload (< 200 chars)
    activation_energy: float  # 0.0–1.0, how important/urgent
    timestamp: float          # When emitted (epoch)
```

### 2.1 Signal Types (Registered)

| Signal Type | Meaning | Emitter(s) | Energy Range |
|---|---|---|---|
| `memory_pressure` | Knowledge is fading or contradicted | decay_engine, semantic_consolidation | 0.5–0.7 |
| `new_knowledge` | New concept formed from experience | semantic_consolidation | 0.6 |
| `novel_observation` | Surprising tool output stored as episode | experience_assimilation | 0.6 |
| `ambient_context` | Environment changed (place, attention, energy) | event_bridge | From confidence |
| `idle_discovery` | Nothing happened, engine self-seeds | reasoning_loop (internal) | 0.4–0.5 |
| `gist_stored` | Active conversation gists stored | gist_storage | 0.3 |
| `episode_created` | New narrative episode consolidated | episodic_memory_worker | 0.5 |
| `trait_changed` | User trait created, updated, or corrected | user_trait_service | 0.3–0.7 |
| `task_state_changed` | Persistent task state transition | persistent_task_service | 0.5–0.6 |
| `schedule_fired` | Scheduled reminder/task fired | scheduler_service | 0.5 |
| `thread_expired` | Conversation thread expired | thread_expiry_service | 0.3 |
| `curiosity_finding` | Curiosity thread produced a finding | curiosity_pursuit_service | 0.5 |
| `user_message` | User sent a chat message | websocket | 1.0 |
| `goal_inferred` | Recurring topic pattern detected as potential goal | goal_inference_service | 0.6 |

Note: Signal handlers also update the world model cache in MemoryStore (`world_model:items`).
`task_state_changed` and `schedule_fired` trigger incremental cache updates via
`WorldStateService.notify_task_changed()` / `notify_schedule_changed()`. The cache
is fully refreshed from DB during idle periods.

New signal types require:
1. Addition to this table
2. A nightly test scenario
3. Documentation of what the consumer should do with it

### 2.2 Signal Transport

- **Priority queue:** `reasoning:priority` (user messages — processed first)
- **Background queue:** `reasoning:signals` (all other signal types)
- **Pop:** `blpop([priority, signals], timeout=idle_timeout)` — tries priority first
- **Push:** `rpush(key, signal.to_json())`
- **Max depth:** 50 signals (oldest dropped on overflow, background queue only)
- **Debounce:** 30s minimum between processed background signals (user messages bypass)
- **Serialization:** JSON via `dataclasses.asdict()`
- **Yield points:** Background signal processing checks priority queue before expensive operations (LLM calls); if a user message is waiting, background reasoning aborts and the loop picks up the priority signal

### 2.3 Emission Rules

- Emission is always **fire-and-forget** — the emitter never waits for a response
- Emission is always **wrapped in try/except** — a failed emit is logged at DEBUG, never raised
- Emission uses **lazy imports** (`from services.cognitive_drift_engine import ...` or `from services.reasoning_loop_service import ...`) to avoid import cycles
- Emitters never instantiate the consumer — they push to the queue and forget

---

## 3. Service Lifecycle Contract

### 3.1 Registration

Every spine-connected service declares in its module docstring:
```
Emits: signal_type_1, signal_type_2
Consumes: signal_type_3 (via reasoning:signals queue)
Trigger: <timer Ns | signal-driven | request-driven | one-shot>
Fail mode: <fail-open description>
```

### 3.2 Health

Every long-running service writes a heartbeat:
```python
store.set(f"health:{service_name}", str(time.time()), ex=ttl)
```

Where `ttl` is 2x the expected cycle time. The `SelfModelService` (30s cycle) reads these heartbeats and includes dead services in its `noteworthy[]` list. No automated restart — health is observational, not coercive.

### 3.3 Startup Order

Services start in dependency order (managed by `run.py`), but **no service assumes another service is running**. If a dependency isn't ready:
- Queue-based: messages accumulate, processed when consumer starts
- Signal-based: signals accumulate (up to queue cap), processed when consumer starts
- Direct call: try/except, return neutral default

---

## 4. Migration Pattern

### 4.1 Converting a Timer Service to Signal-Responsive

For a service that currently runs on `time.sleep(N)`:

**Before:**
```python
def run(self):
    while True:
        time.sleep(self.interval)
        self._do_work()
```

**After (Phase 1 — emit signals, keep timer):**
```python
def run(self):
    while True:
        time.sleep(self.interval)
        self._do_work()
        # NEW: emit signal if something interesting happened
        if result.is_interesting:
            emit_reasoning_signal(ReasoningSignal(...))
```

**After (Phase 2 — consume signals, remove timer):**
```python
def run_signal_loop(self):
    while True:
        signal = self.store.blpop("service:signals", timeout=self.max_idle)
        if signal:
            self._process_signal(signal)
        else:
            self._idle_maintenance()
```

**Phase 1 is always safe to ship independently.** Phase 2 requires the spine to route signals to the service.

### 4.2 Migration Checklist (Per Service)

- [ ] Service docstring updated with Emits/Consumes/Trigger/Fail-mode
- [ ] Signal emission added (Phase 1)
- [ ] Unit tests pass in isolation
- [ ] Nightly scenario created/updated
- [ ] Timer removed, signal consumption added (Phase 2)
- [ ] Fail-open verified (service killed → system continues)
- [ ] Documented in this file's migration tracker (§5)

---

## 5. Migration Tracker

### Phase 1 Complete (Emits Signals, Keeps Timer)

| Service | Signals Emitted | Timer | Nightly Scenario |
|---|---|---|---|
| **DecayEngineService** | `memory_pressure` | 30min | 966 |
| **SemanticConsolidationService** | `new_knowledge`, `memory_pressure` | Queue-driven | 967 |
| **ExperienceAssimilationService** | `novel_observation` | 60s poll | — |
| **EventBridgeService** | `ambient_context` | Event-driven | 968 |
| **GistStorageService** | `gist_stored` | Request-driven | 970 |
| **EpisodicMemoryWorker** | `episode_created` | Queue-driven | 971 |
| **UserTraitService** | `trait_changed` | Request-driven | 972 |
| **PersistentTaskService** | `task_state_changed` | Request/timer | 973 |
| **SchedulerService** | `schedule_fired` | 60s timer | 974 |
| **ThreadExpiryService** | `thread_expired` | 5min timer | 975 |
| **CuriosityPursuitService** | `curiosity_finding` | 6h timer | 976 |

### Phase 2 Complete (Signal-Driven, No Timer)

| Service | Signals Consumed | Idle Fallback | Nightly Scenario |
|---|---|---|---|
| **ReasoningLoopService** | All signal types | 10min → salient/insight | 965, 968, 969 |

### Not Yet Started

| Service | Current Trigger | Priority | Notes |
|---|---|---|---|
| EpisodicMemoryObserver | 60s timer | — | Could react to gist-stored signals |
| IdleConsolidationService | 5min timer | — | Could react to queue-drain signals |
| GrowthPatternService | 30min timer | — | Could react to trait-change signals |
| AutobiographySynthesis | 6h timer | Low | Long cycle, timer is fine for now |
| PersistentTaskWorker | 30min timer | — | Could react to plan-ready signals |
| RoutingReflectionService | 5min timer | — | Could react to low-confidence routing signals |
| ProfileEnrichmentService | 6h timer | Low | Long cycle, timer is fine |
| TemporalPatternService | 6h timer | Low | Long cycle, timer is fine |
| ToolUpdateChecker | 6h timer | Low | Infrastructure, timer is fine |
| SelfModelService | 30s timer | — | Heartbeat aggregator, timer is natural |
| DocumentPurgeService | 6h timer | Low | Maintenance, timer is fine |
| MomentEnrichmentService | 5min timer | Low | Polling for status, timer is fine |
| FolderWatcherService | 30s timer | Low | OS-level polling, timer is natural |

---

## 6. Anti-Patterns

### 6.1 Signal Cascades
**Bad:** Service A emits signal → Service B processes it and emits signal → Service C processes it and emits signal → Service A processes it.
**Rule:** No circular signal paths. If A emits to B, B must never emit back to A through any chain. Draw the signal graph before adding a new emission point.

### 6.2 Signal as RPC
**Bad:** Service A emits a signal and waits for a response.
**Rule:** Signals are fire-and-forget. If you need a response, use a direct function call or a dedicated result queue (like `bg_llm:result:{job_id}`).

### 6.3 Mandatory Signals
**Bad:** Service B crashes if it doesn't receive a signal from Service A within N seconds.
**Rule:** No signal is mandatory. "No signal" means "nothing interesting happened", never "something is broken". Timeouts trigger idle/maintenance behavior, not error states.

### 6.4 Fat Signals
**Bad:** Signal payload contains the full episode text, embeddings, or large data structures.
**Rule:** Signals carry references (concept_id, topic) and summaries (content < 200 chars). The consumer looks up full data from SQLite/MemoryStore if needed.

### 6.5 Signal-Driven Configuration
**Bad:** Using signals to propagate config changes across services.
**Rule:** Config is read from files/DB at service init or on a slow reload cycle. Signals carry cognitive events, not infrastructure state.

---

## 7. The Spine (Future)

The current architecture has a single consumer (ReasoningLoopService) reading from a single queue (`reasoning:signals`). The future spine will:

1. **Route signals to multiple consumers** — each service registers interest in specific signal types
2. **Priority scheduling** — user-facing signals preempt background maintenance
3. **Backpressure** — slow consumers don't cause queue overflow for fast consumers
4. **Observability** — signal flow is logged and queryable for debugging

This is explicitly **not built yet**. The current single-queue model is sufficient for Phase 1 (emit signals) and the initial Phase 2 conversions. The spine emerges when enough services are signal-driven that routing becomes necessary.

**Build the spine when you need it, not before.**
