# Message Flow

Chalie handles five distinct message paths. All of them share the same **ACT loop** and **atomic storage** model. The paths differ in what triggers them, how much post-turn work they do, and whether the result goes to the user immediately or arrives as a proactive push.

---

## The Five Paths

| Path | Trigger | Result |
|---|---|---|
| **User** | WebSocket message from the user | Response delivered on the same socket |
| **DMN** | Step 5 of the subconscious worker tick (no own trigger) | Saves findings via memory tool; no chat-UI push |
| **Goal pursuit** | Background daemon spawned per active goal | Proactive push when goal resolves |
| **Scheduled** | Timer fires on a due prompt | Proactive push to client |
| **Episode encoder** | Internal, runs when the transcript tail grows long enough | No user-visible output; consolidates memory |

Each path is an independent orchestrator. There is no shared queue or central dispatcher — each path constructs its own processor instance and runs the loop directly.

---

## User Path

When a message arrives over WebSocket, a daemon thread is spawned immediately so the HTTP layer stays free. The processor runs the full ACT loop, stores the turn atomically, runs post-turn services, and delivers the response.

```
User (WebSocket)
       │
       ▼
  daemon thread spawned
       │
       ▼
  ┌─────────────────────────────┐
  │  _setup()                   │
  │  · write input transcript   │
  │    row (captures uid)       │
  │  · deliberation-score gate  │
  │    (exploration pass, user  │
  │    channel only)            │
  │  · _seed_turn_zero():       │
  │    - memory recall dispatch │
  │      (if memory_seed=True)  │
  │    - document.upload per    │
  │      attachment (if any)    │
  └────────────┬────────────────┘
               │
               ▼
  ┌─────────────────────────────┐
  │  ACT loop  (see below)      │
  └────────────┬────────────────┘
               │
               ▼
  ┌─────────────────────────────┐
  │  Atomic store               │
  │  (input + tools + response  │
  │   in one transaction)       │
  └────────────┬────────────────┘
               │
               ▼
  ┌─────────────────────────────┐
  │  Post-turn fan-out          │
  │  (see below)                │
  └────────────┬────────────────┘
               │
               ▼
  Response + metrics → WebSocket → client
```

**History is literal text, not a messages array.** The previous conversation is assembled as a text block inside the user message body. The provider always receives a single-element messages list. This keeps history portable and independent of provider multi-turn formats.

---

## ACT Loop

Every processor runs the same ACT loop. The loop continues until the model produces a response with no tool calls, a cooperative cancel signal is received (`_cancel_event`), or the processor's iteration cap is reached. User-facing processors (UMP, SubagentProcessor) have no iteration cap — they run until the model finishes or the user stops them. Background processors retain hard caps (DMN=100, EAMP=200, PatternMatch=100, GeoPattern=100). The user can stop an active UMP turn via `POST /chat/interrupt` or a running subagent via `POST /chat/subagent/<sub_id>/stop`. When a turn is cancelled, `_cleanup_cancelled_turn()` deletes all tool_call and transcript rows for that turn — the cancelled turn leaves no trace in the database. If the user sent a new message mid-turn, the frontend concatenates the original + new message (separated by `\n\n`) and starts a fresh turn with the combined text.

```
  ┌──────────────────────────────────────────┐
  │  Build prompt                            │
  │  (world state, past messages, user text, │
  │   accumulated tool trail)                │
  │                 │                        │
  │  Check context size ─── near limit? ──► compaction (see below)
  │                 │                        │
  │  Send to LLM                             │
  │                 │                        │
  │  No tool calls? ──► done                 │
  │                 │                        │
  │  Tool calls? ──► dispatch each tool      │
  │                  · append result         │
  │                    to trail              │
  │                  · record tool in        │
  │                    metrics               │
  │                 │                        │
  │  Check _cancel_event (cooperative stop)   │
  │                 │                        │
  └─────────────────┘ (next iteration)
```

Tool errors are returned to the model as structured result strings. They are never raised to the caller or surfaced to the user directly.

Each tool call is written to `tool_calls` immediately via `Ability.record()` — the trail is the table, not an in-memory list. Ephemeral rows are purged at turn end (`_purge_ephemeral_tool_calls`); durable rows (compaction, thinking) persist.

---

## Mid-ACT Compaction

Two independent thresholds checked once per iteration via `Providers.calculate()`:

| Threshold | Data source | Action |
|-----------|-------------|--------|
| **>90% + trail exists** | `tool_calls` rows for this turn | `_compact_trail()` — summarises trail into one `trail_compaction` row, iteration resets to 0 |
| **>80%** | Prior conversation transcript | `_compact_history()` — summarises prior turns into history watermark, iteration resets to 0 |

The 90% check runs first. Compaction loops (`COMPACTION_CONFIG`, `SUBAGENT_COMPACTION_CONFIG`) hit a recursion guard — if they approach 80% they log a warning and proceed rather than compacting-of-compaction.

**Trail compaction** (`_compact_trail()`): assembles the current trail from `tool_calls` (everything since the last `trail_compaction` row), summarises it with `COMPACTION_CONFIG`, and records the result as a new `trail_compaction` row (`tool_name='trail_compaction'`, `ephemeral=1`). The next `_from_last_compaction()` slice begins at that row — every prior row silently drops out of the assembled trail without a DELETE.

**History compaction** (`_compact_history()`): calls `_run_full_compaction()` which writes a durable `tool_calls` row with `tool_name='compaction'`, `ephemeral=0`, `params={"compacted_up_to_id": <max id>, "status": "success"|"failure"}`, `result=<summary body>`. `compaction_persistence.get_compaction(channel)` returns the most recent success row. The current `'user'`-channel summary is surfaced read-only in the Brain dashboard under **Cognition → Compacted Summary** via `GET /system/observability/compaction`.

**Two compaction kinds must not collide.** History rows use `tool_name='compaction'`; trail rows use `tool_name='trail_compaction'`. The history lookup filters `params.status='success'`, which trail rows never carry — so the two selectors are disjoint.

---

## Post-Turn Fan-Out (User Path Only)

After the turn is stored, `config.post_turn(mp, response_text)` is called if non-None. This is the only post-turn hook. For the user channel it fires proactive skill suggestion (when iteration ≥ 4 and the loop exited cleanly). For all background channels it is `None`.

Metrics (token counts, request counters) are recorded inside the provider send gateway (`Providers._log_after_call`) — not in `post_turn` and not in the loop. Token totals are accumulated per-send so delegate/sub-processor attribution is correct automatically.

Background paths (DMN, pattern match, episode encoder, etc.) have `post_turn=None` and emit nothing to the chat UI (`broadcast_to=None`). (DMN no longer has an idle-timer — it runs as Step 5 of the subconscious worker tick.)

Personal facts are handled inline during the ACT loop: when the model decides to store something, it calls the memory ability directly. Contradiction detection happens at storage time.

---

## Background Paths

### DMN (Reflective Pass — Subconscious Step 5)

DMN no longer has its own daemon, idle trigger, or proactive output channel (v0.6.0). It runs as **Step 5 of the subconscious worker tick** — see `04-ARCHITECTURE.md` for the full five-step ordering. The processor reflects on the user picture and persists findings to `data_graph` via the `memory` tool; nothing is pushed to chat.

```
  Subconscious tick (Step 5)
      │
      ├── load user_summary_long (fallback user_summary, else skip)
      ├── load channel='user' episodes (retrieval_weight ≥ 0.3, 30d window, LIMIT 50)
      │
      ▼
   ACT loop (news / search / browser ALWAYS_AVAILABLE; memory tool for writes)
      │
      ▼
   Findings persisted to data_graph. No chat-UI broadcast.
```

### Goal Pursuit

When the model calls the goal-pursuit tool, a daemon thread is spawned for that goal. It runs an extended ACT loop independently, with a long wall-clock budget. On completion, the result is pushed to the client as a proactive message.

Each goal daemon is fully isolated. Multiple goals can run in parallel on separate threads.

### Scheduled Prompts

A polling loop checks for prompts that are due. When one fires, it runs an ACT loop with a reduced tool set (scheduling and goal-pursuit tools are excluded to prevent recursion). The result is pushed to the client and the item is marked executed.

### Episode Encoder

Runs internally when a channel's transcript tail grows beyond a threshold. Encodes recent transcript windows into episode records for later recall. Produces no user-visible output. The trigger is evaluated per-channel after each turn is stored.

---

## Per-Turn Metrics

Every WebSocket response frame carries a `metrics` block. Token counts span **all** LLM calls in the turn — the main ACT loop, the thinking exploration pass, and any compaction call fired by `_handle_overflow()`. Tool counts record how many times each tool was called. The response time is measured from before the daemon thread is spawned.

```json
{
  "tokens_total": 4820,
  "tools": { "memory": 1, "search": 2 },
  "response_time_s": 1.43
}
```

Action-button responses (no LLM call) carry the block with zero tokens. Error frames carry whatever partial metrics were accumulated before the failure.
