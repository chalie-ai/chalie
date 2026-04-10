# Message Flow — Complete Routing Reference

This document is the single authoritative visual map of how a user message travels through Chalie. Every branch, every storage hit, every LLM call, and every background cycle is shown here.

**Legend**
```
⚡ DET   — Deterministic (no LLM, <10ms)
🧠 LLM   — LLM inference call
📥 M     — MemoryStore READ
📤 M     — MemoryStore WRITE
📥 DB    — SQLite READ
📤 DB    — SQLite WRITE
⏱ ~Xms  — Typical latency
```

---

## 1. Master Overview — All Possible Paths

```
                            ┌──────────────────────┐
                            │  User Message via    │
                            │    /ws  (WebSocket)  │
                            └──────────┬───────────┘
                                       │ spawns daemon thread
                            ┌──────────▼───────────┐
                            │  _handle_chat_       │
                            │  background()        │
                            └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────────────────────────┐
                            │  UserMessageProcessor.instance().process()│
                            │                                           │
                            │  1. UserPromptAssemblyService.build()     │
                            │     (world state, transcript, episodes,   │
                            │      knowledge, self-awareness, files)    │
                            │                                           │
                            │  2. SystemPromptAssemblyService.build()   │
                            │     (identity, directives, frontal-cortex │
                            │      template — cacheable, stable)        │
                            │                                           │
                            │  3. MessageProcessor.send()               │
                            │     ├─ 📤 DB  transcript (user turn)      │
                            │     ├─ 📥 DB  context_window_service      │
                            │     │         .build_messages() — always  │
                            │     │         reconstructed from DB       │
                            │     ├─ 🧠 LLM  Providers.send_messages() │
                            │     │         (frontal-cortex-unified)    │
                            │     ├─ tool loop: dispatch → store to DB  │
                            │     │   → compact at 80% → rebuild msgs   │
                            │     └─ 📤 DB  transcript (assistant turn) │
                            └──────────┬────────────────────────────────┘
                                       │ result dict
                            ┌──────────▼───────────┐
                            │  OutputService.       │
                            │  enqueue_text()       │
                            └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │  📤 M  pub/sub        │
                            │  sse:{request_id}    │
                            │  WS → Client         │
                            └──────────────────────┘

NOTE: Tool calls from the LLM are executed inline by MessageProcessor.send()
via the DB-backed tool loop (max 30 iterations / 15 min). Context is rebuilt
from the database on every iteration — nothing accumulates in memory.

BACKGROUND (always running, independent of user messages):
  PATH D  ──  Persistent Task Worker  (30min ± jitter)   (see §5)
  PATH E  ──  Reasoning Loop          (600s, idle-only)   (see §7)
```

---

## 2. Phase A — Context Assembly (UserPromptAssemblyService)

Runs inside `UserMessageProcessor.process()` on every user message, before the LLM call.

Conversation history is **not** assembled here. It is constructed from the database by `context_window_service.build_messages()` and injected into the messages array by `MessageProcessor.send()` on every iteration.

```
┌─────────────────────────────────────────────────────────────────────┐
│  UserPromptAssemblyService.build()                                  │
│                                                                     │
│  Step 1  World State header                       📥 M  <5ms       │
│          WorldStateService.get_world_state(channel)                 │
│          Includes: time, calendar events, active reminders          │
│          ─────────────────────────────────────────────────          │
│  Step 2  Voice mode guard (per-turn)              ⚡ DET            │
│          If metadata.source == 'voice':                             │
│          Injects TTS instruction (no markdown/formatting)           │
│          ─────────────────────────────────────────────────          │
│  Step 3  System Awareness                         ⚡ DET            │
│          SelfModelService.format_for_prompt()                       │
│          Only non-empty when degradation signals are present        │
│          ─────────────────────────────────────────────────          │
│  Step 4  Current Turn                             📥 DB             │
│          memory_skill.recall_episodes(                              │
│              caller='seed',                                         │
│              baseline_radius=SEED_RADIUS_BASELINE,                  │
│          )                                                          │
│          → dynamic radius (narrow/expand vs prior-turn queries)     │
│          → writes memory_recall_log row (caller='seed')             │
│          → "### Related Memories" (episodic auto-recall)            │
│          → "## User Message" (raw user text)                        │
│          → file tags (images / documents from metadata)             │
│          → nudge tag (if present in metadata)                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  context_window_service.build_messages()          📥 DB  <20ms     │
│                                                                     │
│  Reconstructs the full LLM messages array from DB on every call.   │
│  Nothing accumulates in memory.                                     │
│                                                                     │
│  1. compaction_service.get_compaction(channel)                      │
│     → watermark ID (entries with id ≤ watermark are summarized)     │
│  2. transcript_service.get_recent(channel, since_id=watermark)      │
│     → all entries since watermark (no in-memory limit)              │
│  3. ToolCallService.get_by_transcript_ids([assistant_ids])          │
│     → reconstruct tool_calls lists for assistant entries            │
│  4. Prepend order (if compaction exists):                           │
│     [overflow_content → compacted_text → transcript entries]        │
│     overflow_content placed BEFORE compacted text so the compacted  │
│     summary receives higher recency/attention weight from the LLM   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase B — System Prompt Assembly (SystemPromptAssemblyService)

Builds the stable, cacheable system prompt sent alongside every user message.

```
┌─────────────────────────────────────────────────────────────────────┐
│  SystemPromptAssemblyService.build(type='unified')                  │
│                                                                     │
│  Template: frontal-cortex-unified.md (loaded via load_configs)      │
│                                                                     │
│  Placeholder injection:                                             │
│    {{identity_modulation}}                                          │
│        IdentityService → VoiceMapperService.generate_modulation()   │
│        → tone/personality instruction string                        │
│                                                                     │
│    {{adaptive_directives}}                                          │
│        AdaptiveLayerService.generate_directives()                   │
│        → response style adjustments based on interaction signals    │
│                                                                     │
│  Designed for provider-side prompt caching. Content rarely changes  │
│  between turns (identity vectors + adaptive signals are slow-moving)│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Mode Router — Non-User Flows (Drift, Proactive, Fallback)

### 4a. Mode Router (Deterministic)

Used only for non-user flows (cognitive drift, proactive notifications, fallback). User messages bypass this entirely — they go directly through `UserMessageProcessor`.

```
┌─────────────────────────────────────────────────────────────────────┐
│  ModeRouterService                           ⚡ DET  ~5ms           │
│                                                                     │
│  Signal inputs (all already in memory from prior assembly):         │
│    context_warmth       channel_confidence   has_question_mark      │
│    working_memory_turns fok_score            interrogative_words    │
│    gist_count           is_new_channel       greeting_pattern       │
│    fact_count           world_state_present  explicit_feedback      │
│    intent_type          intent_complexity    intent_confidence      │
│    information_density  implicit_reference   prompt_token_count     │
│                                                                     │
│  Scoring formula (per mode):                                       │
│    score[mode] = base_score + Σ(weight[signal] × signal_value)    │
│    Anti-oscillation: hysteresis dampening from prior mode          │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Tie-breaker?                           ⚡ ONNX  ~5ms        │   │
│  │  Triggered when: top-2 scores within effective_margin       │   │
│  │  Model:   mode-tiebreaker (ONNX classifier)                 │   │
│  │  Output:  pick mode A or B                                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                            UNIFIED
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  FrontalCortexService                        🧠 LLM  ~500ms–2s     │
│                                                                     │
│  Prompt = soul.md + identity-core.md + frontal-cortex-{mode}.md    │
│                                                                     │
│  Context injected:                                                  │
│    • Working memory (channel)                                       │
│    • Chat history                                                   │
│    • Assembled context (semantic retrieval)                         │
│    • Drift gists (if idle thoughts exist)                           │
│    • Context relevance inclusion map (computed dynamically)         │
│                                                                     │
│  Config files:                                                      │
│    UNIFIED      → frontal-cortex-unified.json                       │
│                                                                     │
│  Output: { response: str, confidence: float, mode: str }           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                           Phase D  (§6)
```

### 4b. Tool Loop (User Messages)

Tool calls from user-message LLM responses are executed inline by `MessageProcessor.send()`. The loop runs up to 30 iterations with a 15-minute wall-clock timeout. On each iteration the context window is rebuilt from the database via `context_window_service.build_messages()`. Results are stored to the transcript and tool_calls tables immediately; nothing accumulates in memory.

Background workers (persistent_task_worker, goal_pursuit_processor) use the ACT orchestrator independently via their own `MessageProcessor` subclasses.

---

## 5. Phase D — Post-Response Commit

Runs after every response is generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE D: Post-Response Commit                                      │
│                                                                     │
│  Step 1  Transcript already appended by MessageProcessor.send()     │
│          context_window_service.check_and_compact() runs at 80%    │
│          of provider context limit (both before the first LLM call  │
│          and after each tool-loop iteration)                        │
│                         │                                           │
│  Step 2  Log interaction event                  📤 DB              │
│          Table: interaction_log                                      │
│          Fields: event_type='system_response', mode,               │
│                  confidence, generation_time                        │
│                         │                                           │
│  Step 3  Encode response event                  📤 M  (async)      │
│          EventBusService → ENCODE_EVENT                             │
│          Triggers downstream memory consolidation:                  │
│                                                                     │
│          ┌──────────────────────────────────────────────────────┐  │
│          │  episodic-memory-queue (PromptQueue)                 │  │
│          │    → episodic_memory_worker: episode build  🧠 LLM  │  │
│          │    → 📤 DB  episodes  (with sqlite-vec embedding)    │  │
│          │                                                      │  │
│          │  semantic_consolidation_queue (PromptQueue)          │  │
│          │    → semantic consolidation: concept extract 🧠 LLM │  │
│          │    → 📤 DB  concepts, semantic_relationships         │  │
│          └──────────────────────────────────────────────────────┘  │
│                         │                                           │
│  Step 4  Publish to WebSocket                   📤 M  (pub/sub)    │
│          key: sse:{request_id}                                      │
│          OutputService.enqueue_text() → /ws → client               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Path D — Persistent Task Worker (Background, 30min Cycle)

Operates completely independently of user messages.

```
┌─────────────────────────────────────────────────────────────────────┐
│  persistent_task_worker  (30min ± 30% jitter)                      │
│                                                                     │
│  1. Expire stale tasks                          📥📤 DB            │
│     Table: persistent_tasks                                         │
│     created_at > max_age → mark EXPIRED                            │
│                                                                     │
│  2. Pick eligible task (FIFO within priority)   📥 DB              │
│     State machine: PENDING → RUNNING → COMPLETED                    │
│                                                                     │
│  3. Load task + progress                        📥 DB              │
│     persistent_tasks.progress (JSON as TEXT)                               │
│     Contains: plan DAG, coverage, step statuses                    │
│                                                                     │
│  4. Execution branch:                                               │
│     ┌──────────────────┐      ┌───────────────────────────────┐   │
│     │  HAS PLAN DAG?   │─Yes─►│  Plan-Aware Execution         │   │
│     └────────┬─────────┘      │  Ready steps = steps where    │   │
│              │ No             │  all depends_on are DONE       │   │
│              ▼                │  Execute each ready step       │   │
│     ┌──────────────────┐      │  via bounded ACT loop         │   │
│     │  Flat ACT Loop   │      └───────────────────────────────┘   │
│     │  Iterate toward  │                                           │
│     │  goal directly   │                                           │
│     └──────────────────┘                                           │
│                                                                     │
│  5. Bounded ACT Loop (both branches):           🧠 LLM  per iter  │
│     max_iterations=5, cumulative_timeout=30min                     │
│     Same fatigue model as interactive ACT loop                     │
│                                                                     │
│  6. Atomic checkpoint                           📤 DB              │
│     persistent_tasks.progress (JSON as TEXT, atomic UPDATE)        │
│     Saves: plan, coverage %, step statuses, last results           │
│                                                                     │
│  7. Coverage check                              ⚡ DET             │
│     100% complete → mark COMPLETED                                 │
│                                                                     │
│  8. Adaptive surfacing (optional)                                   │
│     After cycle 2, or coverage jumped > 15%                        │
│     → Proactive message to user                                    │
│     → 📤 M  pub/sub proactive channel                              │
│                                                                     │
│  PLAN DECOMPOSITION (called on task creation):  🧠 LLM  ~300ms    │
│  PlanDecompositionService                                           │
│  Prompt: plan-decomposition.md                                      │
│  Output: { steps: [{ id, description, depends_on: [] }] }          │
│  Validates: Kahn's cycle detection, quality gates (Jaccard <0.7),  │
│             confidence > 0.5, step word count 4-30                 │
│  Stores: persistent_tasks.progress.plan (JSON as TEXT)              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Path E — Reasoning Loop (Background, 600s Idle-Only)

Runs only when all PromptQueues are idle. Signal-driven continuous reasoning.

```
┌─────────────────────────────────────────────────────────────────────┐
│  reasoning_loop_service  (600s idle timeout, signal-driven)        │
│                                                                     │
│  Preconditions:                               ⚡ DET               │
│    All queues idle?   📥 M  (queue lengths = 0)                    │
│    Recent episodes exist? (lookback 168h)  📥 DB                   │
│    Bail if user is in deep focus           📥 M  focus:{channel}   │
│                                                                     │
│  1. Seed Selection (weighted random)          ⚡ DET               │
│     Salient  0.60 │ Insight  0.40                                   │
│     Source: 📥 DB  episodes table (by category)                    │
│                                                                     │
│  2. Spreading Activation (depth ≤ 2)          ⚡ DET               │
│     📥 DB  semantic_concepts, semantic_relationships               │
│     📥📤 M  cognitive_drift_activations  (sorted set)              │
│     📥📤 M  cognitive_drift_concept_cooldowns  (hash)              │
│     Collect top 5 activated concepts                               │
│                                                                     │
│  3. Thought Synthesis                         🧠 LLM  ~100ms       │
│     Prompt: cognitive-drift.md + soul.md                           │
│     Input:  activated concepts + soul axioms                       │
│     Output: thought text                                            │
│                                                                     │
│  4. Store drift gist                          📤 M               │
│     key: gist:{channel}  (30min TTL)                               │
│     Will surface in frontal cortex context on next user message    │
│                                                                     │
│  5. Action Decision Routing                   ⚡ DET               │
│     Scores registered actions:                                      │
│                                                                     │
│     ┌───────────────┬──────────┬─────────────────────────────────┐ │
│     │  Action       │ Priority │  What it does                   │ │
│     ├───────────────┼──────────┼─────────────────────────────────┤ │
│     │  COMMUNICATE  │    10    │  Push thought to user (deferred)│ │
│     │  PLAN         │     7    │  Propose persistent task 🧠 LLM │ │
│     │  SEED_THREAD  │     6    │  Plant new conversation seed    │ │
│     │  REFLECT      │     5    │  Internal memory consolidation  │ │
│     │  RECONCILE    │     4    │  Contradiction resolution       │ │
│     │  AMBIENT_TOOL │     3    │  Context-triggered tool use     │ │
│     │  NOTHING      │     0    │  Always available fallback      │ │
│     └───────────────┴──────────┴─────────────────────────────────┘ │
│                                                                     │
│     Winner selected by score (ties broken by priority)             │
│     PLAN action → calls PlanDecompositionService  🧠 LLM          │
│                → stores in persistent_tasks  📤 DB                 │
│                                                                     │
│  6. Deferred queue                             📤 M               │
│     COMMUNICATE → stores thought for quiet-hours delivery          │
│     Async: flushes when user returns from absence                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Complete Storage Access Map

### MemoryStore Keys Reference

```
Key Pattern                        TTL        Read    Written by
─────────────────────────────────────────────────────────────────────
fok:{channel}                      —          A,B     FOK update service
world_model:items                  —          A       WorldStateService
reasoning_loop:activations         —          E       Reasoning loop
reasoning_loop:cooldowns           —          E       Reasoning loop
sse:{request_id}                   short      /ws     UserMessageProcessor / OutputService

PromptQueues (in-memory, thread-safe):
episodic-memory-queue              —          D       encode event handler
semantic_consolidation_queue       —          D       episodic_memory_worker
```

### SQLite Tables Reference

```
Table                      When Written                    When Read
──────────────────────────────────────────────────────────────────────
interaction_log            Phase D (every message)         observability endpoints
cortex_iterations          ACT loop (background workers)   observability endpoints
episodes                   episodic_memory_worker (async)  frontal_cortex, reasoning loop
concepts                   semantic_consolidation (async)  drift engine, context assembly
semantic_relationships     semantic_consolidation          drift engine
user_traits                IIP hook                        identity service
persistent_tasks           Path D (task worker)            persistent_task_worker
transcript                 MessageProcessor.send()         context_window_service.build_messages()
compactions                context_window_service          context_window_service.build_messages()
tool_calls                 MessageProcessor.send()         context_window_service.build_messages()
place_fingerprints         ambient inference               place_learning_service
```

---

## 9. LLM Call Inventory

Every LLM call in the system, with typical latency and model used.

```
Service                          Model            Prompt                   Latency   Triggered by
──────────────────────────────────────────────────────────────────────────────────────────────────
UserMessageProcessor (unified)   primary model    frontal-cortex-unified   ~500ms-2s User path (via UserPromptAssemblyService + SystemPromptAssemblyService)
context_window_service compact   same as job      inline system prompt     ~500ms-2s At 80% context limit (mid-loop or pre-call); uses same provider as the conversation
ModeRouterService (tiebreaker)   ONNX             mode-tiebreaker model    ~5ms      Non-user flows only
FrontalCortexService (UNIFIED)   primary model    soul + unified.md        ~500ms-2s Non-user flows
ReasoningLoop (thought)          lightweight      cognitive-drift.md       ~100ms    Path E
PlanDecompositionService         lightweight      plan-decomposition.md    ~300ms    On task creation
episodic_memory_worker           lightweight      episodic-memory.md       ~200ms    Phase D async
semantic_consolidation           lightweight      semantic-extract.md      ~200ms    Phase D async

```

**Deterministic paths (zero LLM):**
- IIP hook (regex)
- Intent classifier
- Empty guard / CANCEL detection (inline in unified path)
- Mode router scoring (non-user flows)
- Fatigue budget check in ACT loop
- Termination checks
- Spreading activation in drift engine
- Plan DAG cycle detection (Kahn's)
- FOK / warmth / memory confidence calculations

---

## 10. Latency Profile by Path

```
Path              P50 Latency    Bottleneck
────────────────────────────────────────────────────────────
Unified (user)    1s – 3s        Unified LLM call (primary model)
Unified + skills  2s – 30s       Skill execution (varies — ACT parked)
D — Task Worker   30min cycle    Background, no user wait
E — Drift         300s cycle     Background, no user wait

Component latency breakdown (unified path, typical):
  UserPromptAssembly   <20ms   ── DB reads (transcript + episodes)
  SystemPromptAssembly <10ms   ── MemoryStore + identity vectors
  Unified LLM call     ~800ms  ── Primary model (varies by provider)
  Transcript write     <5ms    ── SQLite WAL write (both turns)
  OutputService        ~1ms    ── MemoryStore pub/sub
  ─────────────────────────────────────────────────────────
  Total (typical)      ~0.85s
```

---

## 11. Architectural Principles Visible in the Flow

| Principle | Where it shows up in the flow |
|-----------|-------------------------------|
| **Attention is sacred** | User messages go direct to LLM — no routing overhead; DB-backed context reconstruction ensures the full conversation is always present without memory growth |
| **Judgment over activity** | Single unified LLM call per iteration; mode router handles non-user flows deterministically |
| **Tool agnosticism** | Tool schemas injected at send time via `Providers._get_tools()` — no tool names hardcoded in the pipeline |
| **Continuity over transactions** | Transcript + compaction + episodes all feed every response; context_window_service reconstructs the full messages array from DB on every iteration |
| **System prompt caching** | `SystemPromptAssemblyService` builds stable content (identity, directives) separately from volatile user-turn content |
| **No memory accumulation** | `context_window_service.build_messages()` reads from DB on every call; only transcript IDs are tracked in memory during the tool loop |

---

*Last updated: 2026-04-09. See `docs/INDEX.md` for the full documentation map.*
