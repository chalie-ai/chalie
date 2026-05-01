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
  │  pre_act()                  │
  │  · memory recall via        │
  │    handle_memory() —        │
  │    canonical tool dispatch, │
  │    stored ephemeral=0       │
  │  · deliberation-score gate  │
  │    (exploration pass)       │
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

Every processor runs the same bounded ACT loop. The loop continues until the model produces a response with no tool calls, or until iteration limits are reached.

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
  │  Drain any mid-turn user steering        │
  │                 │                        │
  └─────────────────┘ (next iteration)
```

Tool errors are returned to the model as structured result strings. They are never raised to the caller or surfaced to the user directly.

All tool call records accumulate in memory during the loop. Nothing is written to the database until the loop finishes.

---

## Mid-ACT Compaction

When the rendered `user_body` exceeds 80% of the provider's context limit, or when the provider returns a `PayloadTooLargeError` (HTTP 413), `_handle_overflow()` fires. There is a single overflow path — no Stage 1 / Stage 2 distinction.

**Overflow path — `_handle_overflow()`:**

1. `_run_full_compaction(exclude_id=self._uid)` is called. This constructs a `ContinuityCompactionProcessor`, builds an LLM input from the previous compaction summary (if any) plus all transcript turns since the last success watermark (excluding the current in-flight turn so the model is not tempted to answer the user question instead of summarising).
2. The LLM response is parsed for `<summary>...</summary>`. The extracted body is the new compaction text.
   - **Success** — writes a `tool_calls` row with `tool_name='compaction'`, `ephemeral=0`, `params={"compacted_up_to_id": <max included id>, "status": "success"}`, `result=<summary body>`. Emits `[COMPACTION] {channel}: continuity success — in=… chars, out=… chars, watermark …→…`.
   - **Failure** (parse error, empty output, LLM error) — writes a row with `status=failure` and `result=''`. Emits `[COMPACTION] {channel}: continuity failure — reason=…`. The lookup ignores failure rows, so the previous success summary remains active. The caller breaks to cap-exit and the turn cannot proceed.
3. On success, in-flight ACT loop state is cleared: `_act_trail`, `_discovered_tools`, `_pending_tool_calls`, and `_thinking_exploration` are all reset. Any `tool_calls` rows with `transcript_id = self._uid AND ephemeral = 1` (in-flight ACT artifacts from the aborted iteration) are deleted. The iteration counter resets to 0 and the ACT loop restarts.

A second `PayloadTooLargeError` after a successful recovery breaks immediately to cap-exit (tracked via `_payload_too_large_recovered`) — the summary itself is too large for the provider.

**Compaction storage** is append-only. Results are `tool_calls` rows, not a separate table. `compaction_persistence.get_compaction(channel)` returns `{compacted_text, compacted_up_to_id, tool_call_id}` for the most recent success row, or `None` if none exists.

**Subagent overflow** is handled differently. `SubagentProcessor._handle_overflow()` overrides the base: it calls `SubagentTrailCompactionProcessor` to compress `self._act_trail` in place, writes a `tool_calls` audit row with `tool_name='subagent_trail_compaction'` and `ephemeral=1`, and continues from the same iteration — no restart, because a subagent is one-shot and cannot rebuild channel history.

---

## Post-Turn Fan-Out (User Path Only)

After the turn is stored, the user processor triggers a set of services synchronously. Because the response is already on the wire (sent via narration callbacks during the loop), this fan-out does not affect perceived latency.

- **Conversation phase tracking** — updates the current phase based on both the user message and the response.
- **Metrics counter** — increments request and user-message totals.

Background paths (DMN, goal pursuit, scheduled) skip all of this. They only update the request counter. (DMN no longer has an idle-timer to reset — it runs as Step 5 of the subconscious worker tick.)

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
