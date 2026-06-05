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
  │    (user channel only;      │
  │     sets thinking_level)    │
  │  · _seed_turn_zero():       │
  │    - thinking dispatch      │
  │      (if thinking_level=    │
  │       'high'; ThinkingAbil- │
  │       ity via ToolDispatch) │
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

Every turn runs the same ACT loop through the one flat `MessageProcessor`; behaviour differs only by the per-channel `ProcessorConfig`. The loop continues until the model produces a response with no tool calls, a cooperative cancel signal is received (`_cancel_event`), or `config.max_iterations` is reached. The user config sets `max_iterations=None` (unbounded) — a user turn runs until the model finishes or the user stops it. Background and delegate configs set explicit caps (DMN=100, external-agent=200, pattern-match=100, geo-pattern=30, delegates=50). The user can stop an active user turn via `POST /chat/interrupt`, or cancel a running async delegate via `POST /chat/subagent/<sub_id>/stop` (a legacy route name). When a turn is cancelled, `_cleanup_cancelled_turn()` deletes all tool_call and transcript rows for that turn — the cancelled turn leaves no trace in the database. If the user sent a new message mid-turn, the frontend concatenates the original + new message (separated by `\n\n`) and starts a fresh turn with the combined text.

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

Each tool call is written to `tool_calls` by `ToolDispatcher.dispatch()` via `ActTrail().record()` — the trail is the table, not an in-memory list. Ephemeral rows are purged at turn end (`_purge_ephemeral_tool_calls`); durable rows (compaction, thinking, memory seed, document uploads) persist across turns.

---

## Mid-ACT Compaction

Compaction relies **solely on context-window size** — there is no turn-count threshold. It is triggered by two conditions, both routed to the SAME chokepoint — `_dispatch_compaction()`:

| Trigger | Condition | How |
|---------|-----------|-----|
| **Window-fit (trim-then-compact)** | `Providers._fit_request()` finds the scaffolded request exceeds the fit cap and drops the oldest rendered rows to make it fit, setting `mp._compaction_pending = True` | After the send returns, `_loop` sees `_compaction_pending` and calls `_dispatch_compaction()` to fold the dropped history into the checkpoint for the next turn |
| **Reactive (413)** | `providers.send()` raises `PayloadTooLargeError` (provider rejects the payload outright) | `_dispatch_compaction()` in the `except` branch, **one-shot** — guarded by `mp._payload_compacted` so a second 413 re-raises instead of looping |

**The fit loop (`Providers._fit_request`).** The request is built, its token size estimated, and compared against `cap = window - max(int(0.10 * window), 8000)` — i.e. always reserve `max(10% of the window, 8k tokens)` of response headroom. If the request already fits, nothing is dropped and `_compaction_pending` stays `False`. If it overflows, the loop calls `mp.drop_oldest_previous_message()` (which advances `mp._history_drop`, trimming one more row off the FRONT of `get_previous_messages()`), rebuilds, and re-estimates — **looping until the request fits or the history is exhausted**. `drop_oldest_previous_message()` returns `False` once every row is dropped, so the loop is guaranteed to terminate (this is the hang the design fixes). On any drop it flags `_compaction_pending` so the loop compacts the dropped rows after the send. Because the watermark is the compaction row's own id, the subsequent compaction folds the *entire* pre-watermark backlog — the overflow turn sees a one-turn partial view, then the next turn reads the dense checkpoint.

There is **no proactive turn-count trigger, no `ContextOverflowError`, no retry counter, no `_compact()` method, and no inline compaction**. `_dispatch_compaction()` fires two INTERNAL, never-discoverable abilities through the normal `ToolDispatcher.dispatch()` chokepoint — the exact machinery as the turn-0 `memory`/`thinking` seeds. Because they go through dispatch, each one **auto-records its `tool_calls` row AND auto-emits the act-trail WebSocket events** (`act_tool_start`/`act_tool_end`) — that is how compaction shows up in the frontend act-trail with no hand-rolled emit. `_dispatch_compaction()` first resets `mp._history_drop = 0`, then fires the two abilities in order:

1. **`tool_chain_compactor`** (fired first) — reads the current turn's act-trail off the bound `mp` via `_render_act_trail()`. If `_has_trail()` is false (no non-compactor row since the last boundary) it silently returns `""` — no LLM call, no boundary. Otherwise it runs `MessageProcessor.process(trail_text, ToolChainCompactionConfig())` and returns the dense handover. The dispatch chain records that handover as the `tool_chain_compactor` tool_calls row — **that row (when its result is non-empty) is the new trail boundary**: `_from_last_compaction()` slices from it, so pre-compacted tool calls drop out of the rendered trail without a DELETE.

2. **`chat_history_compactor`** (fired second) — reads the parent channel's `get_previous_messages()` off `mp`. When the backlog is empty it returns without writing (nothing to fold). Otherwise it carries the prior checkpoint forward as a `## Previous Summary` block, runs `MessageProcessor.process(combined, ChatHistoryCompactionConfig())`, and writes the model's output **verbatim** to the `transcript` table as a durable `role='compaction'` row via `transcript_service.write_input_row(channel, "compaction", summary)`. The new row's own `id` becomes the watermark — `_previous_rows()` reads `id > watermark`, so the next read returns nothing through it. The watermark **always advances** on a non-empty backlog, so compaction can never silently no-op into an infinite loop. There is **no parser**: no `<analysis>`/`<summary>` tags, nothing to trim — whatever the model writes IS the checkpoint.

`get_previous_messages()` renders **every** row since the watermark, minus the leading `mp._history_drop` rows the fit loop has trimmed (`entries = self._previous_rows()[self._history_drop:]`). There is no fixed row cap — the only bound on the rendered slice is whatever the window-fit loop had to drop to make the request fit. `_history_drop` is reset to `0` at the start of each `_fit_request` and again inside `_dispatch_compaction()` (once the dropped rows are folded into the checkpoint, the watermark advances past them and the drop count is no longer meaningful).

Both compaction configs (`ChatHistoryCompactionConfig`, `ToolChainCompactionConfig`) set `thinking_mode = "high"` (a `ClassVar` lever read by `Providers.send()`), carry `suppress_history=True`, expose no tools, and use `channel="compaction"` — a dedicated channel name so the inner compaction send's own `_wrap_with_checkpoint("compaction", …)` lookup is empty (continuity is instead carried explicitly via `## Previous Summary`). Watermark rows are written to the **parent** channel, never to `"compaction"`.

**Checkpoint envelope.** When a compaction row exists for the channel, `_wrap_with_checkpoint()` (called inside `Providers.send()`) wraps the user-prompt body into a `### Checkpoint - What you were previously discussing / doing` block followed by `### Current State - What's happening in the current turn`. No-op when no compaction row exists.

**Two compaction kinds must not collide.** History rows live in `transcript` with `role='compaction'`; trail rows live in `tool_calls` with `tool_name='tool_chain_compactor'`. The watermark lookup (`compaction_persistence.get_compaction`) reads `transcript WHERE role='compaction'` — disjoint from the trail selector. The same summary is surfaced read-only in the Brain dashboard under **Cognition → Compacted Summary** via `GET /system/observability/compaction`.

---

## Post-Turn Fan-Out (User Path Only)

After the turn is stored, `_record()` iterates `config.post_turn_hooks` — a `tuple[PostTurnHook, ...]` (default `()` for background channels). Each hook's `run(mp, response_text)` is called in a failure-isolated loop; one hook raising does not affect the others. For the user channel the tuple contains `ProactiveSuggestionHook` (fires proactive skill suggestion when iteration ≥ 4 and the loop exited cleanly). Background channels have an empty tuple.

Metrics (token counts, request counters) are recorded inside the provider send gateway (`Providers._log_after_call`) — not in post-turn hooks and not in the loop. Token totals are accumulated per-send so delegate/sub-processor attribution is correct automatically.

Background paths (DMN, pattern match, episode encoder, etc.) have `post_turn_hooks=()` and emit nothing to the chat UI (`broadcast_to=None`). (DMN no longer has an idle-timer — it runs as Step 5 of the subconscious worker tick.)

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

Every WebSocket response frame carries a `metrics` block. Token counts span **all** LLM calls in the turn — the main ACT loop, the `ThinkingAbility` exploration pass (when `thinking_level='high'`), and any compaction call fired by `_dispatch_compaction()` (on the window-fit `_compaction_pending` path or the one-shot 413 path). Tool counts record how many times each tool was called. The response time is measured from before the daemon thread is spawned.

```json
{
  "tokens_total": 4820,
  "tools": { "memory": 1, "search": 2 },
  "response_time_s": 1.43
}
```

Action-button responses (no LLM call) carry the block with zero tokens. Error frames carry whatever partial metrics were accumulated before the failure.
