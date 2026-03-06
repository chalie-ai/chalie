# Message Flow Diagrams: Every Path, DB Hit, and LLM Call

Visual and detailed breakdowns of every execution path, database query, and LLM invocation during a message cycle. See also Workflow and [Cognitive Router](07-COGNITIVE-ARCHITECTURE.md).


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
                            │   User Message POST  │
                            │     /chat  (HTTP)    │
                            └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │  WebSocket channel   │
                            │  ws:{request_id}     │
                            │  📤 M  ws_pending    │
                            └──────────┬───────────┘
                                       │ daemon thread
                            ┌──────────▼───────────┐
                            │   digest_worker()    │◄──── background
                            └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │   PHASE A            │
                            │   Ingestion &        │
                            │   Context Assembly   │
                            │   (see §2)           │
                            └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │   PHASE B            │
                            │   Signal Collection  │
                            │   & Triage           │
                            │   (see §3)           │
                            └──────────┬───────────┘
                                       │
                           ┌───────────┴────────────────────┐
                           │         Triage Branch          │
                           │  (CognitiveTriageService)      │
                           └──┬─────────────┬───────────────┘
                              │             │               │
               ┌──────────────▼──┐  ┌───────▼──────┐  ┌───▼──────────────┐
               │  PATH A         │  │  PATH B       │  │  PATH C          │
               │  Social Exit    │  │  ACT →        │  │  RESPOND /       │
               │  CANCEL/IGNORE/ │  │  Tool Worker  │  │  CLARIFY /       │
               │  ACKNOWLEDGE    │  │  (PromptQueue) │  │  ACKNOWLEDGE     │
               └──────┬──────────┘  └──────┬────────┘  └──────┬───────────┘
                      │                    │                    │
               ┌──────▼──────────┐  ┌──────▼────────┐  ┌──────▼───────────┐
               │  Empty response │  │  Background   │  │  Mode Router     │
               │  + WM append    │  │  execution    │  │  (Deterministic) │
               │  📤 M   📤 DB   │  │  (see §5)     │  │  → Generation    │
               └─────────────────┘  └───────────────┘  │  (see §4)        │
                                                        └──────┬───────────┘
                                                               │
                                                        ┌──────▼───────────┐
                                                        │   PHASE D        │
                                                        │   Post-Response  │
                                                        │   Commit (see §6)│
                                                        └──────┬───────────┘
                                                               │
                                                        ┌──────▼───────────┐
                                                        │  📤 M  pub/sub   │
                                                        │  output:{id}     │
                                                        │  WS → Client     │
                                                        └──────────────────┘

BACKGROUND (always running, independent of user messages):
  PATH D  ──  Persistent Task Worker  (30min ± jitter)   (see §7)
  PATH E  ──  Cognitive Drift Engine  (300s, idle-only)   (see §8)
```

---

## 2. Phase A — Ingestion & Context Assembly

Runs immediately for every message, before any routing decision.

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE A: Context Assembly                                          │
│                                                                     │
│  Step 1  IIP Hook (Identity Promotion)            ⚡ DET  <5ms     │
│          Regex: "call me X", "my name is X", …                     │
│          Match → 📤 M  📤 DB  (trait + identity)                   │
│          No match → continue                                        │
│                           │                                         │
│  Step 2  Working Memory                           📥 M              │
│          key: wm:{thread_id}  (list, 4 turns, 24h TTL)             │
│          ─────────────────────────────────────────────────          │
│  Step 3  Gists                                    📥 M              │
│          key: gist:{topic}  (sorted set, 30min TTL)                │
│          ─────────────────────────────────────────────────          │
│  Step 4  Facts                                    📥 M              │
│          key: fact:{topic}:{key}  (24h TTL)                        │
│          ─────────────────────────────────────────────────          │
│  Step 5  World State                              📥 M              │
│          key: world_state:{topic}                                   │
│          ─────────────────────────────────────────────────          │
│  Step 6  FOK (Feeling-of-Knowing) score           📥 M              │
│          key: fok:{topic}  (float 0.0–5.0)                         │
│          ─────────────────────────────────────────────────          │
│  Step 7  Context Warmth                           ⚡ DET            │
│          warmth = (wm_score + gist_score + world_score) / 3        │
│          ─────────────────────────────────────────────────          │
│  Step 8  Memory Confidence                        ⚡ DET            │
│          conf = 0.4×fok + 0.4×warmth + 0.2×density                │
│          is_new_topic → conf *= 0.7                                 │
│          ─────────────────────────────────────────────────          │
│  Step 9  Session / Focus Tracking                 📥📤 M            │
│          topic_streak:{thread_id}  (2h TTL)                        │
│          focus:{thread_id}  (auto-infer after N exchanges)         │
│          Silence gap > 2700s → trigger episodic memory             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase B — Signal Collection & Two-Layer Routing

This phase produces the routing decision in two separate layers.

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1: Intent Classification                   ⚡ DET  ~5ms     │
│                                                                     │
│  IntentClassifierService                                            │
│  Input:  text, topic, warmth, memory_confidence, wm_turns          │
│  Output: { intent_type, complexity, confidence }                   │
│  No external calls — pure heuristics                                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  LAYER 2: Cognitive Triage                                          │
│  CognitiveTriageService  (4-step pipeline)                         │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 2a  Social Filter                  ⚡ DET  ~1ms       │   │
│  │                                                             │   │
│  │  Pattern → Result (no LLM, returns immediately)            │   │
│  │  ─────────────────────────────────────────────────          │   │
│  │  Greeting / positive feedback (short) → ACKNOWLEDGE        │   │
│  │  Cancel / nevermind                   → CANCEL             │   │
│  │  Self-resolved / topic drop           → IGNORE             │   │
│  │  Empty input                          → IGNORE             │   │
│  │                                                             │   │
│  │  If matched ──► PATH A (Social Exit)                       │   │
│  └─────────────────────────┬───────────────────────────────────┘   │
│                            │ not matched                            │
│  ┌─────────────────────────▼───────────────────────────────────┐   │
│  │  Step 2b  Cognitive Triage LLM           🧠 LLM  ~100-300ms │   │
│  │                                                             │   │
│  │  Config:   cognitive-triage.json                           │   │
│  │  Prompt:   cognitive-triage.md                             │   │
│  │  Model:    lightweight (qwen3:4b or smaller)               │   │
│  │  Timeout:  500ms (falls back to heuristics on timeout)     │   │
│  │                                                             │   │
│  │  Context sent to LLM:                                      │   │
│  │    • User text                                             │   │
│  │    • Previous mode + [tools](09-TOOLS.md) used                            │   │
│  │    • Tool summaries (from profile service)                 │   │
│  │    • Working memory summary (last 2 turns)                 │   │
│  │    • context_warmth, memory_confidence, gist_count         │   │
│  │                                                             │   │
│  │  LLM output (JSON):                                        │   │
│  │    branch:             respond | clarify | act             │   │
│  │    mode:               RESPOND|CLARIFY|ACT|ACKNOWLEDGE…    │   │
│  │    [tools](09-TOOLS.md):              ["tool1", …]     (up to 3)          │   │
│  │    skills:             ["recall", …]                       │   │
│  │    confidence_internal: 0.0–1.0                            │   │
│  │    confidence_tool_need: 0.0–1.0                           │   │
│  │    freshness_risk:     0.0–1.0                             │   │
│  └─────────────────────────┬───────────────────────────────────┘   │
│                            │                                        │
│  ┌─────────────────────────▼───────────────────────────────────┐   │
│  │  Step 2c  Self-Eval Sanity Check          ⚡ DET  ~1ms      │   │
│  │                                                             │   │
│  │  • Cap tool list at 3 contextual skills                    │   │
│  │  • Validate skill names                                    │   │
│  │  • Factual question detected → may force ACT               │   │
│  │  • URL in message detected  → may force ACT                │   │
│  │  • Can OVERRIDE LLM result if heuristics detect issues     │   │
│  └─────────────────────────┬───────────────────────────────────┘   │
│                            │                                        │
│  ┌─────────────────────────▼───────────────────────────────────┐   │
│  │  Step 2d  Triage Calibration Log         📤 DB  ~1ms        │   │
│  │  Table: triage_calibration                                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┼───────────────────┐
              │                │                   │
       branch=social    branch=act          branch=respond
              │                │                   │
          PATH A           PATH B              PATH C
```

---

## 4. Path C — RESPOND / CLARIFY / ACKNOWLEDGE

### 4a. Mode Router (Deterministic)

```
┌─────────────────────────────────────────────────────────────────────┐
│  ModeRouterService                           ⚡ DET  ~5ms           │
│                                                                     │
│  Signal inputs (all already in memory from Phase A/B):             │
│    context_warmth       topic_confidence     has_question_mark     │
│    working_memory_turns fok_score            interrogative_words   │
│    gist_count           is_new_topic         greeting_pattern      │
│    fact_count           world_state_present  explicit_feedback     │
│    intent_type          intent_complexity    intent_confidence     │
│    information_density  implicit_reference   prompt_token_count    │
│                                                                     │
│  Scoring formula (per mode):                                       │
│    score[mode] = base_score + Σ(weight[signal] × signal_value)    │
│    Anti-oscillation: hysteresis dampening from prior mode          │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Tie-breaker?                           🧠 LLM  ~100ms      │   │
│  │  Triggered when: top-2 scores within effective_margin       │   │
│  │  Model:   qwen3:4b                                          │   │
│  │  Input:   mode descriptions + context summary               │   │
│  │  Output:  JSON → pick mode A or B                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  📤 DB  routing_decisions table                                     │
│    Fields: mode, scores, tiebreaker_used, margin, signal_snapshot  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
          ┌────────────────────┼───────────────────┐
          │                    │                   │
       RESPOND             CLARIFY           ACKNOWLEDGE
          │                    │                   │
          └────────────────────┴───────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  FrontalCortexService                        🧠 LLM  ~500ms–2s     │
│                                                                     │
│  Prompt = soul.md + identity-core.md + frontal-cortex-{mode}.md    │
│                                                                     │
│  Context injected:                                                  │
│    • Working memory (thread_id)                                     │
│    • Chat history                                                   │
│    • Assembled context (semantic retrieval)                         │
│    • Drift gists (if idle thoughts exist)                           │
│    • [Context relevance](10-CONTEXT-RELEVANCE.md) inclusion map (computed dynamically)         │
│                                                                     │
│  Config files:                                                      │
│    RESPOND      → frontal-cortex-respond.json                      │
│    CLARIFY      → frontal-cortex-clarify.json                      │
│    ACKNOWLEDGE  → frontal-cortex.json (base)                       │
│                                                                     │
│  Output: { response: str, confidence: float, mode: str }           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                           Phase D  (§6)
```

### 4b. ACT Mode — The Action Loop

Triggered when triage `branch=respond` but mode router selects ACT, **or** directly from triage `branch=act` via the internal path in `route_and_generate`.

```
┌─────────────────────────────────────────────────────────────────────┐
│  ActLoopService                                                     │
│  Config: cumulative_timeout=60s  per_action=10s  max_iterations=5  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Iteration N                                                 │  │
│  │                                                              │  │
│  │  1. Generate action plan            🧠 LLM                  │  │
│  │     Prompt: frontal-cortex-act.md                           │  │
│  │     Input:  user text + act_history (prior results)         │  │
│  │     Output: [{ type, params, … }, …]                        │  │
│  │                                                              │  │
│  │  2. Termination check               ⚡ DET                  │  │
│  │     • Fatigue budget exceeded?                               │  │
│  │     • Cumulative timeout reached?                            │  │
│  │     • Max iterations reached?                                │  │
│  │     • No actions in plan?                                    │  │
│  │     • Same action repeated 3× in a row?                     │  │
│  │     If any → exit loop                                       │  │
│  │                                                              │  │
│  │  3. Execute actions                  ⚡/🧠 varies           │  │
│  │     ActDispatcherService                                     │  │
│  │     Chains outputs: result[N] → input[N+1]                  │  │
│  │     Action types:                                            │  │
│  │       recall, memorize, introspect, associate               │  │
│  │       schedule, list, focus, persistent_task                │  │
│  │       (+ external [tools](09-TOOLS.md) via tool_worker thread)             │  │
│  │                                                              │  │
│  │  4. Accumulate fatigue               ⚡ DET                 │  │
│  │     cost *= (1.0 + fatigue_growth_rate × iteration)         │  │
│  │     fatigue += cost                                          │  │
│  │                                                              │  │
│  │  5. Log iteration                    📤 DB                  │  │
│  │     Table: cortex_iterations                                 │  │
│  │     Fields: iteration_number, actions_executed,             │  │
│  │             execution_time_ms, fatigue, mode                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│           │                                                         │
│           └──► repeat if can_continue()                             │
│                                                                     │
│  After loop terminates:                                             │
│  1. Re-route → terminal mode (force previous_mode='ACT')           │
│     Mode router (deterministic, skip_tiebreaker=True)              │
│     Typically selects RESPOND                                       │
│  2. Generate terminal response (FrontalCortex)   🧠 LLM           │
│     act_history passed as context                                   │
│     All-card actions → skip text (mode='IGNORE')                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Path B — ACT → Tool Worker (PromptQueue)

Triggered when `CognitiveTriageService` selects `branch=act` and specific [tools](09-TOOLS.md) are named.

```
┌─────────────────────────────────────────────────────────────────────┐
│  _handle_act_triage()                         ⚡ DET               │
│                                                                     │
│  1. Create cycle record                         📤 DB              │
│     Table: cortex_iterations                                        │
│     Type: 'user_input', source: 'user'                             │
│                                                                     │
│  2. Enqueue tool work                           📤 M  (Queue)      │
│     Queue: tool-queue                                               │
│     Payload:                                                        │
│       cycle_id, topic, text, intent                                │
│       context_snapshot: { warmth, tool_hints, exchange_id }        │
│                                                                     │
│  3. Set WS pending flag                         📤 M               │
│     key: ws_pending:{request_id}  TTL=600s                         │
│     Tells /chat endpoint: tool_worker will deliver response         │
│                                                                     │
│  4. Return empty response (digest_worker done)                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                   WebSocket holds open (polling ws_pending)
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  tool_worker  (daemon thread)                                       │
│                                                                     │
│  1. Dequeue from tool-queue                     📥 M  (Queue)      │
│                                                                     │
│  2. Get relevant [tools](09-TOOLS.md)                          📥 DB              │
│     From triage_selected_tools, or compute via relevance           │
│                                                                     │
│  3. Dispatch each tool                                              │
│     ActDispatcherService (generic, no tool-specific branches)      │
│     Per-tool timeout enforced                                       │
│     Result: { status, result, execution_time }                     │
│                                                                     │
│  4. Post-action critic verification             🧠 LLM  (optional) │
│     CriticService — lightweight LLM                                │
│     Safe actions:         silent correction                         │
│     Consequential actions: pause + escalate to user                │
│                                                                     │
│  5. Log results                                 📤 DB              │
│                                                                     │
│  6. Publish response                            📤 M  (pub/sub)    │
│     key: output:{request_id}                                        │
│     Payload: { metadata: { response, mode, cards, … } }           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                        WebSocket receives pub/sub
                        → streams cards + text to client
```

---

## 6. Phase D — Post-Response Commit

Runs after every response is generated (Paths A, B, C).

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE D: Post-Response Commit                                      │
│                                                                     │
│  Step 1  Append to Working Memory               📤 M               │
│          key: wm:{thread_id}  (append)                              │
│          { role: 'assistant', content, timestamp }                 │
│          Max 4 turns maintained                                     │
│                         │                                           │
│  Step 2  Log interaction event                  📤 DB              │
│          Table: interaction_log                                      │
│          Fields: event_type='system_response', mode,               │
│                  confidence, generation_time                        │
│                         │                                           │
│  Step 3  Onboarding state                       📤 DB              │
│          SparkStateService — increment exchange count               │
│          Table: spark_state                                         │
│                         │                                           │
│  Step 4  Encode response event                  📤 M  (async)      │
│          EventBusService → ENCODE_EVENT                             │
│          Triggers downstream memory consolidation:                  │
│                                                                     │
│          ┌──────────────────────────────────────────────────────┐  │
│          │  memory-chunker-queue (PromptQueue)                  │  │
│          │    → memory_chunker_worker: gist generation 🧠 LLM  │  │
│          │    → 📤 M  gist:{topic}  (sorted set)                │  │
│          │                                                      │  │
│          │  episodic-memory-queue (PromptQueue)                 │  │
│          │    → episodic_memory_worker: episode build  🧠 LLM  │  │
│          │    → 📤 DB  episodes  (with sqlite-vec embedding)    │  │
│          │                                                      │  │
│          │  semantic_consolidation_queue (PromptQueue)          │  │
│          │    → semantic consolidation: concept extract 🧠 LLM │  │
│          │    → 📤 DB  concepts, semantic_relationships         │  │
│          └──────────────────────────────────────────────────────┘  │
│                         │                                           │
│  Step 5  Publish to WebSocket                   📤 M  (pub/sub)    │
│          key: output:{request_id}                                   │
│          /chat endpoint receives, streams to client                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Path D — Persistent Task Worker (Background, 30min Cycle)

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

## 8. Path E — Cognitive Drift Engine (Background, 300s Idle-Only)

Runs only when all PromptQueues are idle. Mimics the brain's Default Mode Network.

```
┌─────────────────────────────────────────────────────────────────────┐
│  cognitive_drift_engine  (300s cycles, idle-gated)                 │
│                                                                     │
│  Preconditions:                               ⚡ DET               │
│    All queues idle?   📥 M  (queue lengths = 0)                    │
│    Recent episodes exist? (lookback 168h)  📥 DB                   │
│    Bail if user is in deep focus           📥 M  focus:{thread_id} │
│                                                                     │
│  1. Seed Selection (weighted random)          ⚡ DET               │
│     Decaying  0.35 │ Recent   0.25 │ Salient 0.15                  │
│     Insight   0.15 │ Random   0.10                                  │
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
│     key: gist:{topic}  (30min TTL)                                  │
│     Will surface in frontal cortex context on next user message    │
│                                                                     │
│  5. Action Decision Routing                   ⚡ DET               │
│     Scores registered actions:                                      │
│                                                                     │
│     ┌──────────────┬──────────┬──────────────────────────────────┐ │
│     │  Action      │ Priority │  What it does                    │ │
│     ├──────────────┼──────────┼──────────────────────────────────┤ │
│     │  COMMUNICATE │    10    │  Push thought to user (deferred) │ │
│     │  SUGGEST     │     8    │  Tool recommendation             │ │
│     │  NURTURE     │     7    │  Engagement nudge                │ │
│     │  PLAN        │     7    │  Propose persistent task 🧠 LLM  │ │
│     │  SEED_THREAD │     6    │  Plant new conversation seed     │ │
│     │  REFLECT     │     5    │  Internal memory consolidation   │ │
│     │  NOTHING     │     0    │  Always available fallback       │ │
│     └──────────────┴──────────┴──────────────────────────────────┘ │
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

## 9. Complete Storage Access Map

### [MemoryStore](08-DATA-SCHEMAS.md) Keys Reference

```
Key Pattern                        TTL        Read    Written by
─────────────────────────────────────────────────────────────────────
wm:{thread_id}                     24h        A,C     D, tool_worker
gist:{topic}                       30min      A,C     Drift, memory_chunker
fact:{topic}:{key}                 24h        A       Frontal cortex
fok:{topic}                        —          A,B     FOK update service
world_state:{topic}                —          A       World state service
topic_streak:{thread_id}           2h         A       Phase A (focus tracking)
focus:{thread_id}                  variable   A,E     FocusSessionService
cognitive_drift_activations        —          E       Drift engine
cognitive_drift_concept_cooldowns  —          E       Drift engine
cognitive_drift_state              —          E       Drift engine
ws_pending:{request_id}            600s       /chat   _handle_act_triage
output:{request_id}                short      /chat   digest_worker, tool_worker

PromptQueues (in-memory, thread-safe):
prompt-queue                       —          —       run.py → digest_worker
tool-queue                         —          B       _handle_act_triage
memory-chunker-queue               —          D       Encode event handler
episodic-memory-queue              —          D       memory_chunker_worker
semantic_consolidation_queue       —          D       episodic_memory_worker
```

### SQLite Tables Reference

```
Table                      When Written                    When Read
──────────────────────────────────────────────────────────────────────
routing_decisions          Phase C (every message)         routing_reflection_service
interaction_log            Phase D (every message)         observability endpoints
cortex_iterations          ACT loop, Path B                observability endpoints
episodes                   memory_chunker (async)          frontal_cortex, drift engine
concepts                   semantic_consolidation (async)  drift engine, context assembly
semantic_relationships     semantic_consolidation          drift engine
user_traits                IIP hook, triage calibration    identity service
triage_calibration         Phase B Step 2d                 triage_calibration_service
persistent_tasks           Path D (task worker)            persistent_task_worker
topics                     Phase A (new topic)             topic_classifier
threads                    session management              session_service
chat_history               Phase D                         frontal_cortex
spark_state                Phase D                         onboarding service
place_fingerprints         ambient inference               place_learning_service
curiosity_threads          drift (SEED_THREAD action)      curiosity_pursuit_service
```

---

## 10. LLM Call Inventory

Every LLM call in the system, with typical latency and model used.

```
Service                      Model            Prompt                   Latency   Triggered by
────────────────────────────────────────────────────────────────────────────────────────────────
TopicClassifierService       lightweight      topic-classifier.md      ~100ms    Every message
CognitiveTriageService       lightweight      cognitive-triage.md      ~100-300ms Every message
ModeRouterService (tiebreaker) qwen3:4b       mode-tiebreaker.md       ~100ms    Close scores only
FrontalCortex (RESPOND)      primary model    soul + respond.md        ~500ms-2s Path C
FrontalCortex (CLARIFY)      primary model    soul + clarify.md        ~500ms-2s Path C
FrontalCortex (ACKNOWLEDGE)  primary model    soul + acknowledge.md    ~500ms-2s Path C
FrontalCortex (ACT plan)     primary model    frontal-cortex-act.md    ~500ms-2s Path C ACT loop
FrontalCortex (terminal)     primary model    mode-specific            ~500ms-2s After ACT loop
CriticService                lightweight      critic.md                ~200ms    Path B (optional)
CognitiveDrift (thought)     lightweight      cognitive-drift.md       ~100ms    Path E
PlanDecompositionService     lightweight      plan-decomposition.md    ~300ms    On task creation
memory_chunker_worker        lightweight      memory-chunker.md        ~100ms    Phase D async
episodic_memory_worker       lightweight      episodic-memory.md       ~200ms    Phase D async
semantic_consolidation       lightweight      semantic-extract.md      ~200ms    Phase D async
RoutingReflectionService     strong model     routing-reflection.md    ~1-2s     Idle-time only
```

**Deterministic paths (zero LLM):**
- IIP hook (regex)
- Intent classifier
- Social filter in cognitive triage
- Mode router scoring
- Fatigue budget check in ACT loop
- Termination checks
- Spreading activation in drift engine
- Plan DAG cycle detection (Kahn's)
- FOK / warmth / memory confidence calculations

---

## 11. Latency Profile by Path

```
Path              P50 Latency    Bottleneck
────────────────────────────────────────────────────────────
A — Social Exit   ~400ms         Topic classifier LLM
B — ACT + [Tools](09-TOOLS.md)   5s – 30s+      Tool execution (external)
C — RESPOND       1s – 3s        Frontal cortex (primary LLM)
C — CLARIFY       1s – 2s        Frontal cortex (primary LLM)
C — ACT Loop      2s – 30s       N × frontal-cortex-act LLMs
D — Task Worker   30min cycle    Background, no user wait
E — Drift         300s cycle     Background, no user wait

Component latency breakdown (Path C RESPOND, typical):
  Context assembly     <10ms   ── [MemoryStore](08-DATA-SCHEMAS.md) reads (all cached)
  Intent classify      ~5ms    ── Deterministic
  Triage LLM           ~200ms  ── qwen3:4b
  Social filter        ~1ms    ── Regex
  Mode router          ~5ms    ── Math, no LLM
  Frontal cortex LLM   ~800ms  ── Primary model (varies by provider)
  Working memory write <5ms    ── [MemoryStore](08-DATA-SCHEMAS.md) append
  DB event log         ~10ms   ── SQLite WAL write
  WS publish           ~1ms    ── [MemoryStore](08-DATA-SCHEMAS.md) pub/sub
  ─────────────────────────────────────────────────────────
  Total (typical)      ~1.1s
```

---

## 12. Five Architectural Principles Visible in the Flow

| Principle | Where it shows up in the flow |
|-----------|-------------------------------|
| **Attention is sacred** | Social filter exits in <1ms — never wastes LLM for greetings; ACT fatigue model prevents runaway tool chains |
| **Judgment over activity** | Two-layer routing: fast social filter first, then LLM triage only if needed; mode router is deterministic not generative |
| **Tool agnosticism** | `ActDispatcherService` routes all [tools](09-TOOLS.md) generically — no tool names anywhere in the Phase B/C infrastructure |
| **Continuity over transactions** | Working memory, gists, episodes, concepts all feed every response; drift gists surface even on next message |
| **Single authority** | `RoutingStabilityRegulator` is the only process that mutates router weights (24h cycle); no tug-of-war possible |

---

*Last updated: 2026-02-27. See `docs/INDEX.md` for the full documentation map.*

## Related Documentation
- [Workflow](05-WORKFLOW.md)
- [Cognitive Router](07-COGNITIVE-ARCHITECTURE.md)
- [Workers](06-WORKERS.md)