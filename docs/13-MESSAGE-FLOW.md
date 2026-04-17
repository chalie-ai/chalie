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
                            ┌──────────▼────────────────────────────────────────┐
                            │  UserMessageProcessor(raw_input, metadata,        │
                            │                       on_narration).send(req_id)  │
                            │                                                   │
                            │  One instance per turn. No singleton. No process()│
                            │                                                   │
                            │  _run_memory_seed()  — episodic auto-recall        │
                            │                                                   │
                            │  ACT loop (see §2):                               │
                            │    getUserPrompt()   — builds literal-text body   │
                            │    _wrap_with_checkpoint()  — envelopes body      │
                            │    [compaction check at 80% — see §3]             │
                            │    getSystemPrompt() — identity + unified template│
                            │    getTools()        — innate skills + discovered  │
                            │    Providers.send_messages()  🧠 LLM              │
                            │    handleTool() per tool_call → ActDispatcher     │
                            │    tool_synthesis DTO, user_steer drain           │
                            │    repeat until text-only response or cap hit     │
                            │                                                   │
                            │  store()  — append_atomic_turn() ONE transaction  │
                            │  postTurn() — 8-service fan-out (see §4)          │
                            └──────────┬────────────────────────────────────────┘
                                       │ final response text
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

NOTE: The LLM receives a single literal-text user message containing a
## Previous Messages block built by getPreviousMessages(). The provider
multi-turn messages[] array is NOT used for history — one element only.

Tool calls from the LLM are dispatched inline by handleTool() via
ActDispatcherService. Nothing accumulates across turns in memory; the
transcript is the persistence layer and is read fresh via getPreviousMessages()
at the start of each ACT iteration.

BACKGROUND (always running, independent of user messages):
  PATH B  ──  DMN Service            (60min idle / 6h cadence)  (see §5)
  PATH C  ──  Goal Pursuit daemon    (spawned per goal)          (see §6)
  PATH D  ──  Scheduled Prompts      (60s poll)                  (see §7)
```

---

## 2. Phase A — ACT Loop (MessageProcessor.send())

Runs inside every `MessageProcessor` subclass. Shown here for `UserMessageProcessor`.

```
┌─────────────────────────────────────────────────────────────────────┐
│  MessageProcessor.send(request_id)                                  │
│                                                                     │
│  Pre-loop (once per turn):                                          │
│    _run_memory_seed()                            📥 DB  <10ms       │
│    memory_skill.recall_episodes(caller='seed')                      │
│    → self._memory_seed (str), self._memory_seed_radius (float)      │
│    → durable DTO appended: {name='memory', ephemeral=0}             │
│                                                                     │
│  Fetch context limit once:                                          │
│    Providers.instance().get_context_limit(job=JOB)  ⚡ DET         │
│    → context_limit (int, fallback=32000)                            │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  ACT iteration body (up to MAX_ITERATIONS, MAX_TIMEOUT):    │   │
│  │                                                              │   │
│  │  getUserPrompt()                         ⚡ DET  <5ms        │   │
│  │    World State block (WorldStateService)    📥 M             │   │
│  │    System Awareness block (SelfModelService)                 │   │
│  │    ## Previous Messages (getPreviousMessages())  📥 DB       │   │
│  │      ← compaction watermark + transcript rows               │   │
│  │      ← durable (ephemeral=0) tool_calls interleaved         │   │
│  │    [memory(radius=X)] seed line                              │   │
│  │    user: <raw_input> [file_tags] [nudge_tag]                 │   │
│  │    ACT loop trail (getActLoopTrail())                        │   │
│  │                                                              │   │
│  │  _wrap_with_checkpoint()                 ⚡ DET  <1ms        │   │
│  │    If compaction exists: prepend Checkpoint/Current State    │   │
│  │    envelope; else pass through verbatim                      │   │
│  │                                                              │   │
│  │  Compaction threshold check (see §3)                        │   │
│  │                                                              │   │
│  │  getSystemPrompt()                       ⚡ DET  <5ms        │   │
│  │    getUserDefinition() + SYSTEM_PROMPT_CLASS().getPrompt()   │   │
│  │                                                              │   │
│  │  getTools()                              ⚡ DET  <1ms        │   │
│  │    NATIVE_TOOLS resolved via tool_schema_service             │   │
│  │    + getDynamicTools() (find_tools discoveries)              │   │
│  │                                                              │   │
│  │  Providers.send_messages(system, messages, tools)  🧠 LLM   │   │
│  │    messages = [{'role':'user','content':user_body}]          │   │
│  │                                                              │   │
│  │  No tool_calls → loop_exited_cleanly = True → break         │   │
│  │                                                              │   │
│  │  If LLM returned narration text alongside tool_calls:       │   │
│  │    tool_synthesis DTO appended (ephemeral=1)                 │   │
│  │    _emit_narration() → on_narration callback → SSE           │   │
│  │                                                              │   │
│  │  For each tool_call: handleTool()        ⚡ DET + varies     │   │
│  │    ActDispatcherService.dispatch_action()                    │   │
│  │    DTO appended to _pending_tool_calls (ephemeral=1)         │   │
│  │    Rendered line appended to _act_trail                      │   │
│  │    find_tools side effect: _discovered_tools extended        │   │
│  │    Exceptions → "ERROR: <tool> failed: <msg>" (never raise)  │   │
│  │                                                              │   │
│  │  Drain steering queue (steer:{request_id})      📥 M        │   │
│  │    user_steer DTOs appended (ephemeral=1)                    │   │
│  │                                                              │   │
│  │  iteration += 1  → goto top of loop                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  store(final_text)                           📤 DB  <5ms           │
│    transcript_service.append_atomic_turn(                           │
│        channel, role, raw_input, llm_response, pending_tool_calls)  │
│    ONE transaction:                                                 │
│      INSERT transcript (role=ROLE, content=raw_input)  → self._uid │
│      INSERT tool_calls sorted by DTO timestamp (all accumulated)    │
│      INSERT transcript (role='assistant', content=llm_response)     │
│    Post-commit daemon threads (outside transaction):                │
│      _embed_entry(input_id)                                         │
│      _embed_entry(assistant_id)                                     │
│      _maybe_trigger_extraction(channel, assistant_id)               │
│                                                                     │
│  postTurn() → 8-service fan-out (see §4)                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mid-ACT Compaction (Two-Stage)

Triggered when the rendered user-message body (including checkpoint envelope) exceeds 80% of the provider's context window. Measured via `estimate_tokens()` before the provider call on each iteration. No DB writes happen mid-loop — all DTOs accumulate in `self._pending_tool_calls` until `store()`.

```
┌─────────────────────────────────────────────────────────────────────┐
│  _check_threshold(user_body, context_limit)      ⚡ DET             │
│                                                                     │
│  Over 80%?                                                          │
│    ├─ YES → Stage 1: _run_stage1_tool_compaction()  🧠 LLM         │
│    │    LLM call against accumulated _act_trail text                │
│    │    Removes ephemeral=1 tool DTOs (except memory + user_steer)  │
│    │    Appends single tool_compaction DTO (ephemeral=1)            │
│    │    Rebuilt user_body re-measured:                              │
│    │      Still over 80%?                                           │
│    │        ├─ YES → Stage 2: _run_stage2_act_restart()  🧠 LLM   │
│    │        │    _run_full_compaction() → LLM summarises channel    │
│    │        │    Upserts compactions table (checkpoint + watermark) │
│    │        │    Appends compaction DTO (ephemeral=0)               │
│    │        │    Collapses pre-restart ephemeral=1 DTOs into        │
│    │        │      single act_restart DTO (ephemeral=1)             │
│    │        │    Resets _act_trail = [], _discovered_tools = []     │
│    │        │    Sets iteration = 0 (loop restarts; MAX_TIMEOUT     │
│    │        │      wall-clock guard keeps ticking)                  │
│    │        └─ NO  → continue to provider call                     │
│    └─ NO  → continue to provider call                              │
│                                                                     │
│  Pseudo-tool DTOs produced (all use invoked_by='system'):          │
│    tool_compaction  ephemeral=1  — Stage 1 summary (audit only)    │
│    compaction       ephemeral=0  — Stage 2 checkpoint (durable)    │
│    act_restart      ephemeral=1  — Stage 2 pre-restart collapse    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase B — Post-Turn Service Fan-Out (UserMessageProcessor.postTurn())

Runs synchronously inside `send()` after `store()` commits, before `send()` returns. The final response is already in the caller's hands (streamed via narration callback); `postTurn()` latency does not delay the user.

Personal facts are captured by the LLM-native memory skill path: the system prompt instructs the LLM to call `memory.store` for personal disclosures, and `memory_skill._handle_store()` delegates to `DataGraphService.store()` which handles contradiction detection internally (temporal_change, true_contradiction, ambiguous) per the kind's policy. The LLM chooses the kind (`user_specific`, `system`, `misc`) via the tool schema enum.

```
┌─────────────────────────────────────────────────────────────────────┐
│  UserMessageProcessor.postTurn()                                    │
│                                                                     │
│  1. ConversationPhaseService         📤 M  (sync)                  │
│     update(user_text, is_user=True)                                 │
│     update(response, is_user=False)  [if response != '']            │
│                         │                                           │
│  2. SituationModelService            📤 M  (sync)                  │
│     update_on_message()                                             │
│                         │                                           │
│  3. SaveSuggestionService  [DEPRECATED]  📥📤 M  (sync)            │
│     3a. detect_save_trigger(user_text) → emit_save_card if flagged  │
│     3b. detect_saveable_content(response) → flag_saveable if hit    │
│                         │                                           │
│  4. Adaptive layer       📤 M  (sync)                              │
│     _detect_fork_response(text)                                     │
│     _store_adaptive_signals(text)                                   │
│                         │                                           │
│  5. DMNService.on_turn() 📤 M  (sync) ← R10 CRITICAL              │
│     Defers DMN idle timer — must fire on every user turn            │
│                         │                                           │
│  6. MetricsService       📤 M  (sync)                              │
│     record_counter('requests_total')                                │
│     record_counter('user_messages_total')                           │
│                         │                                           │
│  7. compaction_service.check_and_compact()   📥📤 DB  (sync)       │
│     End-turn backstop — safety net if mid-loop compaction missed    │
└─────────────────────────────────────────────────────────────────────┘
```

Background subclasses (`DMNMessageProcessor`, `GoalPursuitProcessor`, `ScheduledMessageProcessor`) run a minimal `postTurn()` — only `MetricsService`. No phase updates, no DMN reset.

---

## 5. Path B — DMN Service (Background, 60min Idle / 6h Cadence)

Operates completely independently of user messages.

```
┌─────────────────────────────────────────────────────────────────────┐
│  DMNService timer loop                                              │
│                                                                     │
│  Trigger 1: 60min without user activity (idle)      ⚡ DET         │
│    Context: last 50 high-weight episodes                            │
│                                                                     │
│  Trigger 2: 6h cadence (salience)                   ⚡ DET         │
│    Context: top episodes by retrieval_weight + active goals         │
│                                                                     │
│  Quiet hours (23:00–08:00 local), rate limit (4/24h)               │
│                                                                     │
│  DMNMessageProcessor(raw_input=context, metadata).send()  🧠 LLM  │
│    CHANNEL = 'dmn', ROLE = 'proactive_thought'                      │
│    MAX_ITERATIONS = 15, MAX_TIMEOUT = 300s                          │
│    getUserPrompt(): proactive_thought: <context>                    │
│    postTurn(): metrics only                                         │
│                                                                     │
│  Exits silently if response contains DMN_NO_ACTION                  │
│  Otherwise: OutputService.enqueue_proactive()  📤 M                │
│             → pub/sub → WS → user                                  │
└─────────────────────────────────────────────────────────────────────┘
```

`dmn_service._active_topic()` uses an Option A whitelist: reads only `channel='user'` rows from the transcript, falls back to `'general'` if none exist.

---

## 6. Path C — Goal Pursuit (Background, Per-Goal Daemon Thread)

A daemon thread is spawned by the `goal_pursuit` innate skill when the LLM calls it. Each pursuit is isolated on its own instance with flat channel `'goal_pursuit'`. The `pursuit_id` lives in `metadata` only — never in the channel string.

```
┌─────────────────────────────────────────────────────────────────────┐
│  GoalPursuitProcessor(raw_input=goal, metadata).send()  🧠 LLM     │
│                                                                     │
│  CHANNEL = 'goal_pursuit'   (flat — pursuit_id in metadata only)    │
│  ROLE = 'goal_pursuit'                                              │
│  MAX_ITERATIONS = 50, MAX_TIMEOUT = 7200s (2h)                     │
│                                                                     │
│  getUserPrompt(): goal_pursuit: <goal>  + ACT trail                 │
│  postTurn(): metrics only                                           │
│                                                                     │
│  On completion: OutputService.enqueue_proactive()  📤 M            │
│                 → pub/sub → WS → user                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Path D — Scheduled Prompts (60s Poll)

Fired by the scheduler service when a scheduled item comes due.

```
┌─────────────────────────────────────────────────────────────────────┐
│  ScheduledMessageProcessor(raw_input=prompt, metadata).send()      │
│                                                                     │
│  CHANNEL = 'scheduled'   (flat — item_id in metadata only)          │
│  ROLE = 'scheduled'                                                 │
│  MAX_ITERATIONS = 30, MAX_TIMEOUT = 900s                            │
│  Excludes 'schedule' + 'goal_pursuit' from NATIVE_TOOLS             │
│                                                                     │
│  getUserPrompt(): scheduled: <prompt>  + ACT trail                  │
│  postTurn(): metrics + mark executed (best-effort)                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Storage Access Map

### MemoryStore Keys Reference

```
Key Pattern                        TTL        Read/Written by
─────────────────────────────────────────────────────────────────────
steer:{request_id}                 short      UserMessageProcessor._drain_steering
sse:{request_id}                   short      OutputService / narration callback
active_channel:default             —          reset-thread channel advancement
fok:{channel}                      —          memory system
```

### SQLite Tables Reference

```
Table                      When Written                      When Read
──────────────────────────────────────────────────────────────────────
transcript                 store() → append_atomic_turn()    getPreviousMessages()
tool_calls                 store() → append_atomic_turn()    getPreviousMessages() (ephemeral=0 only)
compactions                _run_full_compaction() (UPSERT)   _wrap_with_checkpoint(), getPreviousMessages()
episodes                   transcript trigger (per-channel)  _run_memory_seed() / memory_skill
                           first at 25 inserts, then +20;
                           window = 25 (20 new + 5 overlap)
data_graph                 DataGraphService.store()           memory_skill, data_graph callers
memory_recall_log          recall_episodes() chokepoint      meta-harness tuning
```

---

## 10. LLM Call Inventory

Every LLM call in the system, with typical latency and model used.

```
Service                          Model         Prompt                   Latency   Triggered by
─────────────────────────────────────────────────────────────────────────────────────────────
UserMessageProcessor (unified)   primary       frontal-cortex-unified   ~500ms-2s User WebSocket path
MessageProcessor compaction      same as JOB   COMPACTION_PROMPT        ~500ms    At 80% ctx threshold (send)
MessageProcessor tool compact    same as JOB   TOOL_COMPACTION_PROMPT   ~200ms    Stage 1 mid-ACT compaction
DMNMessageProcessor              primary       DMNSystemMessagePrompt   ~500ms-2s DMN idle / cadence trigger
GoalPursuitProcessor             primary       GoalPursuitSystemMsgPr.  ~500ms-2s Per-goal daemon thread
ScheduledMessageProcessor        primary       ScheduledSystemMsgPrompt ~500ms-2s Scheduler service (60s poll)
episodic extraction              frontal-ctx   episode-extraction.md    ~500ms-2s rolling per-channel trigger (async, first at 25, then +20)
```

**Deterministic paths (zero LLM):**
- `_run_memory_seed()` — calls `recall_episodes()` direct (no LLM, vector search only)
- `handleTool()` dispatch — routes to skill handler (skill may call LLM internally)
- Compaction threshold check — `estimate_tokens()` character heuristic
- ACT loop termination check
- `postTurn()` synchronous services (phase, situation, adaptive signals)
- DMN timer trigger

---

## 11. Latency Profile by Path

```
Path              P50 Latency    Bottleneck
────────────────────────────────────────────────────────────
User (no tools)   ~0.8s – 2s     Primary model LLM call
User + tools      2s – 30s       Tool execution + extra LLM iterations
DMN               ~1s – 3s       Background, no user wait
Goal pursuit      minutes – 2h   Background, no user wait
Scheduled         ~1s – 5s       Background, no user wait

Component latency (user path, typical, no tools):
  _run_memory_seed           <10ms   ── DB vector search (episodes)
  getUserPrompt              <5ms    ── DB reads (transcript + compaction)
  getSystemPrompt            <5ms    ── file read + placeholder replace
  Primary LLM call           ~800ms  ── Varies by provider / model
  store() → atomic turn      <5ms    ── SQLite WAL write
  postTurn() sync services   <10ms   ── MemoryStore + DB writes
  ─────────────────────────────────────────────────────────
  Total (typical)            ~850ms
```

---

## 12. Architectural Principles Visible in the Flow

| Principle | Where it shows up in the flow |
|-----------|-------------------------------|
| **Per-turn instance, no singleton** | `UserMessageProcessor(raw_input, metadata, on_narration)` constructed fresh each WebSocket message — no `.instance()`, no `.process()` |
| **Literal-text history, not messages[]** | `getPreviousMessages()` assembles a `## Previous Messages` text block; provider sees one-element `messages[]` containing the whole body |
| **Atomic persistence** | `store()` calls `append_atomic_turn()` — single transaction writes input row, all accumulated DTOs, assistant row together; no mid-loop DB writes |
| **Tool agnosticism** | `getTools()` resolves `NATIVE_TOOLS` by name at call time; `find_tools` extends `_discovered_tools` dynamically; no tool-name hardcoding |
| **Per-subclass fan-out** | `postTurn()` is where each subclass fans out its own service work; no shared event bus; `UserMessageProcessor` runs 8 services; background subclasses run 2 |
| **Compaction is in-loop** | Two-stage mid-ACT compaction fires inside `send()` on threshold breach — Stage 1 trims tool trail in place; Stage 2 restarts the loop from a fresh checkpoint |
| **Flat channels** | `CHANNEL` is always a fixed string (`'user'`, `'dmn'`, `'goal_pursuit'`, `'scheduled'`); pursuit_id / item_id live in `metadata` only |

---

*Last updated: 2026-04-11. See `docs/INDEX.md` for the full documentation map.*
