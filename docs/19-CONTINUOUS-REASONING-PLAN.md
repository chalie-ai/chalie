# Master Plan — Stage 5: Continuous Reasoning Loop

**Status:** Active implementation plan
**Governs:** The transition from message-response architecture to continuous reasoning
**Dependencies:** Signal Contract (`docs/18-SIGNAL-CONTRACT.md`), Vision (`docs/00-VISION.md`)

---

## The Shift

**Today:** Chalie is a message-response system with background enrichment. User sends message → pipeline processes it → response returned. Between messages, independent timer-based services run maintenance (decay, consolidation, drift). These are separate worlds that share a database.

**Stage 5:** Chalie is a continuous reasoning loop. User messages, ambient signals, memory events, and scheduled triggers are all signals feeding a single PERCEIVE → UPDATE → REASON → ACT → REFLECT cycle. The loop runs always. Responding to the user is one possible action, not the default.

**The key architectural move:** The CognitiveDriftEngine — currently "idle-time thinking" — evolves into the ReasoningLoop — "always thinking, sometimes responding."

---

## What Already Works

Before building anything new, recognize what's already operational:

| Loop Phase | Current Implementation | Quality |
|---|---|---|
| **PERCEIVE** | AmbientInference, ClientContext, EventBridge, WebSocket handler | Good — deterministic, fast |
| **UPDATE** | GistStorage, WorkingMemory, FactExtraction | Good — runs inline with perception |
| **REASON** | CognitiveTriage (user msgs), ModeRouter (non-user), DriftEngine (idle) | Fragmented — three separate reasoning paths |
| **ACT** | DigestWorker (respond), ToolWorker (tools), PersistentTaskWorker (background) | Good — but triggered separately per path |
| **REFLECT** | CriticService, ExperienceAssimilation, ReflectAction | Good — but only post-ACT, not continuous |

**The problem isn't missing pieces — it's that these pieces run in independent silos.** REASON has three separate entry points. ACT has three separate dispatchers. There's no unified loop that says "a signal arrived, what should I do about it?"

---

## Milestones

### Milestone 0: Signal Foundation (COMPLETE)

**What:** Signal emission from cognitive services + single signal consumer.
**Delivered:** CognitiveDriftEngine is signal-driven. DecayEngine, SemanticConsolidation, ExperienceAssimilation, and EventBridge emit signals. AmbientToolAction disabled.
**Commit:** `db2cfa1` on `rc-1.0.1`

---

### Milestone 1: Complete Signal Coverage (COMPLETE)

**Goal:** Every service that produces an interesting event emits a signal. This is Phase 1 of the Signal Contract — emit signals, keep existing timers. Safe to ship incrementally.

**New signal types to add:**

| Signal Type | Emitter | When | Why it matters |
|---|---|---|---|
| `gist_stored` | GistStorageService | New gist stored from conversation | Indicates active conversation; reasoning loop may want to reason about emerging topics |
| `episode_created` | EpisodicMemoryWorker | Episode consolidated from gists | New narrative memory formed — reasoning can connect it to existing knowledge |
| `trait_changed` | UserTraitService | Trait created, updated, or corrected | Identity/preference shift — may trigger autobiography update or goal re-evaluation |
| `task_state_changed` | PersistentTaskService | Task created, completed, failed, paused | Goal progress — reasoning may need to adjust plans or inform user |
| `schedule_fired` | SchedulerService | Reminder/task fires | Time-based event — may need reasoning about whether to interrupt user |
| `thread_expired` | ThreadExpiryService | Conversation thread expired | Conversation ended — may trigger summarization or goal extraction |
| `curiosity_finding` | CuriosityPursuitService | Curiosity thread produced a finding | Self-directed learning complete — may be worth sharing or connecting |
| `queue_idle` | PromptQueue | All queues drained, system idle | System idle — safe for heavy background work (batch consolidation) |

**Implementation:** Each is a 5–15 line addition at the emission point (try/except, lazy import, fire-and-forget). No consumer changes needed — CognitiveDriftEngine already handles unknown signal types gracefully (semantic lookup fallback).

**Rule:** Don't add signals for routine operations (every DB write, every log entry). Only signal when something *cognitively interesting* happened — something that a reasoning loop would want to know about.

---

### Milestone 2: Rename and Expand the Reasoning Loop (COMPLETE)

**Goal:** CognitiveDriftEngine → ReasoningLoopService. Same architecture (blpop, signal processing, action routing), but expanded to handle the new signal types from Milestone 1 and with clearer separation between idle-discovery and signal-responsive behavior.

**What changes:**
- Rename `cognitive_drift_engine.py` → `reasoning_loop_service.py` (with backward-compatible import alias)
- Rename `cognitive_drift_worker` → `reasoning_loop_worker` in `run.py`
- Rename MemoryStore keys: `cognitive_drift_*` → `reasoning_loop_*`
- Add signal-type-specific handlers (dispatch table instead of one-size-fits-all `_process_signal`)
- Each handler is minimal: convert signal → seed → reason, or signal → direct action (no reasoning needed for some signals)

**What doesn't change:**
- blpop loop stays the same
- Spreading activation → synthesis → action routing pipeline stays the same
- All self-regulation (fatigue, cooldown, debounce, richness gate) stays the same
- All autonomous actions stay the same

**Signal dispatch table:**

```
memory_pressure    → _signal_to_seed() → _reason_from_seed()      # Think about fading knowledge
new_knowledge      → _signal_to_seed() → _reason_from_seed()      # Think about new concepts
novel_observation  → _signal_to_seed() → _reason_from_seed()      # Think about surprising findings
ambient_context    → _signal_to_seed() → _reason_from_seed()      # Think about context changes
gist_stored        → _update_active_topics()                       # Track what's being discussed (no reasoning)
episode_created    → _signal_to_seed() → _reason_from_seed()      # Think about new memories
trait_changed      → _note_identity_shift()                        # Flag for autobiography (no reasoning)
task_state_changed → _evaluate_goal_progress()                     # Check if plans need adjustment
schedule_fired     → _evaluate_interruption()                      # Decide whether to surface
thread_expired     → _extract_unresolved_goals()                   # Mine for latent goals
curiosity_finding  → _signal_to_seed() → _reason_from_seed()      # Think about discoveries
queue_idle         → _trigger_batch_maintenance()                  # Safe for heavy work
idle (timeout)     → _handle_idle_signal()                         # Salient/insight discovery
```

**Key insight:** Not every signal needs full reasoning (spreading activation → synthesis → action). Some signals just update internal state. Keep handlers minimal.

---

### Milestone 3: User Messages as Signals (COMPLETE)

**Goal:** User messages enter the same signal loop as everything else. This is the paradigm shift from "message-response" to "continuous reasoning with occasional responses."

**This is the hardest milestone.** It changes the fundamental flow:

**Today:**
```
WebSocket → prompt-queue → PromptQueue thread → digest_worker → response
```

**After:**
```
WebSocket → signal(user_message) → ReasoningLoop → route to response pipeline
```

**Why this matters:** Today, a user message bypasses all reasoning — it goes straight to triage → mode → response. The reasoning loop (drift engine) never sees it. After this change, the reasoning loop is the first thing that sees every signal, including user messages. It can:
- Notice that a user message relates to an ongoing goal
- Decide that a message doesn't need a response (IGNORE, but with goal awareness)
- Correlate a message with recent ambient signals ("they're asking about restaurants — and they just arrived at a new place")
- Update world model before formulating a response

**Implementation approach:**

1. **New signal type: `user_message`**
   - Highest priority (preempts all background signals)
   - Contains: thread_id, message text, triage result (if already triaged), client context snapshot
   - Emitted by WebSocket handler (replaces `rpush("prompt-queue", ...)` for chat messages)

2. **Priority queue**
   - Cannot use simple `blpop` anymore — user messages must preempt background signals
   - Two queues: `reasoning:priority` (user messages, schedule fires) + `reasoning:signals` (everything else)
   - Loop: `blpop(["reasoning:priority", "reasoning:signals"], timeout=idle_timeout)`
   - MemoryStore `blpop` already supports multiple keys with priority (tries first key first)

3. **User message handler in ReasoningLoop**
   - Runs the existing pipeline: triage → context assembly → frontal cortex → response
   - But now it happens *inside* the reasoning loop, with access to the loop's state
   - The loop knows what it was just thinking about, what goals are active, what ambient context looks like
   - This context can enrich the response without additional retrieval

4. **Latency constraint**
   - User messages must be processed within 200ms of arrival (time to first reasoning step, not to response)
   - Background signals being processed when a user message arrives must yield (check priority queue between steps)
   - This is the main engineering challenge

**Risk:** Latency. Today, user messages go straight to the pipeline. Adding a signal hop adds latency. Mitigation: priority queue + yield points + the signal hop itself is just a `rpush`/`blpop` which is <1ms on MemoryStore.

**What this enables (Stage 5 from vision):**
- Goal-level processing: the loop sees "user mentioned X three times this week" because it processes ALL signals
- World model awareness: the loop knows what's happening when it formulates a response
- Unified reasoning: no more three separate reasoning paths

---

### Milestone 4: Goal Inference (COMPLETE)

**Goal:** The reasoning loop detects goals forming across signals — not just explicit requests, but patterns across conversations, schedule items, trait changes, and ambient context.

**Depends on:** Milestone 3 (user messages as signals — the loop needs to see all signals to detect patterns)

**What "goal inference" means concretely:**
- User mentions "birthday" in 3 conversations over 2 weeks → infer goal: "prepare for birthday"
- User creates 4 schedule items all related to "trip" → infer goal: "plan trip"
- User asks about "mortgage rates" twice, then searches for "houses" → infer goal: "explore home buying"
- Ambient context shows user at gym 3x/week → infer pattern, not goal (important distinction)

**Delivered:**
- `GoalInferenceService` — deterministic SQL candidate detection + LLM validation pipeline
- Candidate detection: queries `interaction_log` for topics with ≥3 conversations, ≥5 messages in 14 days
- Filters: existing goals (PersistentTaskService duplicate check), routine topics, recently proposed goals
- LLM validation: names the goal, assigns confidence, explains reasoning
- Creates PROPOSED persistent task with evidence checkpoint, surfaces via proactive notification
- `goal_inferred` signal emitted to reasoning loop for further reasoning about the new goal
- Signal topic tracking: all non-user-message signals accumulate topics in `goal_inference:signal_topics` sorted set (14-day window)
- Idle-time trigger: goal inference runs during idle handler every 6h (configurable cooldown)
- Dispatch table: `goal_inferred` → full reasoning path (`_handle_reasoning_signal`)
- Config: `goal_inference` section in `cognitive-drift.json`
- Nightly scenario: 978

---

### Milestone 5: World Model (COMPLETE)

**Goal:** WorldStateService becomes actively maintained by the reasoning loop, not passively queried.

**Delivered:**
- WorldStateService gains a MemoryStore-backed cache (`world_model:items`) refreshed during idle periods
- Signal handlers in the reasoning loop update the cache incrementally:
  - `_handle_task_state_changed()` → `ws.notify_task_changed()`
  - `_handle_schedule_fired()` → `ws.notify_schedule_changed()`
- `get_world_state()` reads from cache first (temporal + semantic scoring), falls back to DB if stale (> 5min)
- Three new world state sections:
  - **Ambient context**: place, attention, energy, mobility, tempo from `ambient:prev_inferences`
  - **Active topics**: conversation topics from `reasoning_loop:active_topics`
  - **Reasoning focus**: current thought seed from `reasoning_loop:state`
- `get_world_model_summary()` provides structured data for reasoning loop context enrichment
- `_get_loop_context()` includes world model summary — reasoning sees the full world when processing signals
- Nightly scenario: 979

**Architecture**: Cache stores raw item data; temporal scoring is recomputed per-request (always fresh). Semantic scoring uses a single DB connection for all KNN lookups. Cache misses fall back to the existing DB query path. All updates are fail-open.

---

### Milestone 6: Full Spine (Build When Needed)

**Goal:** When we have 5+ signal consumers, build routing infrastructure.

**Not before.** The current single-queue model is sufficient through Milestones 1–3. Only build the spine when routing becomes a bottleneck.

**What the spine provides:**
- Signal-type → consumer routing (pub/sub)
- Priority scheduling across consumers
- Backpressure (slow consumers don't overflow)
- Health monitoring and dead-consumer detection
- Signal flow observability (debug which signal went where)

---

## Sequencing & Dependencies

```
M0 ──── DONE
  │
  v
M1 ──── Signal coverage (can start now, no dependencies)
  │
  v
M2 ──── Rename + expand loop (depends on M1 for new signal types)
  │
  v
M3 ──── User messages as signals (depends on M2 for dispatch table)
  │         │
  │         v
  │       M4 ──── Goal inference (depends on M3 — needs all signals visible)
  │         │
  │         v
  │       M5 ──── World model (depends on M4 — needs goals to track)
  │
  v
M6 ──── Spine (build when 5+ consumers exist, likely during M4/M5)
```

**M1 and M2 can be done in parallel with other work.** They're additive (new signals, rename) — no existing behavior changes.

**M3 is the breaking change.** The message flow fundamentally changes. This needs careful testing and possibly a feature flag to roll back.

**M4 and M5 are the payoff.** This is where Stage 5 capability actually emerges.

---

## What NOT To Build

1. **Don't build the spine early.** Single queue works until it doesn't. Premature routing infrastructure adds complexity with no payoff.

2. **Don't convert all timer services to signal-driven.** 24h calibration cycles (routing regulator, triage calibration, topic stability) are fine on timers. They're maintenance, not reasoning. See Signal Contract §5 for the full assessment.

3. **Don't add more autonomous actions.** The 9 existing actions (8 active + AMBIENT_TOOL disabled) cover the action space. New capabilities come from better signals and better reasoning, not more action types.

4. **Don't parallelize the reasoning loop.** One loop, one thread, sequential signal processing. Parallelism introduces ordering bugs and makes debugging impossible. The loop is fast enough — each signal processes in <1s (excluding LLM calls which are async via BackgroundLLMQueue).

5. **Don't fold services together.** The temptation will be to merge related services (e.g., merge EpisodicMemoryObserver into SemanticConsolidation). Resist. Each service is a minimal, independently testable unit. Merging increases blast radius when something fails.

---

## Success Criteria

Stage 5 is complete when:

1. **All interesting events are signals.** No cognitive work happens without the reasoning loop knowing about it.
2. **User messages flow through the reasoning loop.** The loop sees every signal — user, ambient, memory, scheduled — in a unified stream.
3. **Goals are inferred, not just requested.** The loop detects patterns across signals and proposes goals that the user hasn't explicitly stated.
4. **World model is continuously maintained.** Context assembly pulls from a live world model, not from per-request database queries.
5. **Latency is preserved.** User message → first response byte in <3s (same as today).
6. **Graceful isolation holds.** Killing any single service degrades one capability, not the whole system.

---

## Estimated Scope

| Milestone | Effort | Risk | Sessions |
|---|---|---|---|
| M1: Signal coverage | Low | Low | 1–2 |
| M2: Rename + expand | Medium | Low | 2–3 |
| M3: User messages as signals | High | Medium | 3–5 |
| M4: Goal inference | High | High | 3–5 |
| M5: World model | Medium | Medium | 2–3 |
| M6: Spine (if needed) | Medium | Low | 2–3 |

Total: ~15–20 sessions over several weeks. Each milestone is independently shippable.
