# Message Flow

Chalie handles five distinct message paths. All of them share the same **ACT loop** and **atomic storage** model. The paths differ in what triggers them, how much post-turn work they do, and whether the result goes to the user immediately or arrives as a proactive push.

---

## The Five Paths

| Path | Trigger | Result |
|---|---|---|
| **User** | WebSocket message from the user | Response delivered on the same socket |
| **DMN** | Idle period or periodic cadence, no user activity required | Proactive push to client |
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

When the prompt nears the context window limit, the system compacts before the next LLM call. This is a two-stage process.

**Stage 1 — tool trail summary.** `TrailCompactionProcessor` summarises the accumulated tool trail into a single compact record. Ephemeral tool records are replaced in-place. If the prompt now fits, the loop continues normally.

**Stage 2 — full restart.** If the prompt is still too large after Stage 1, `FullCompactionProcessor` summarises the entire channel history into a durable checkpoint, collapses the pre-restart trail into a single restart marker, and resets the loop to iteration zero. The checkpoint is prepended to every subsequent prompt as "Current State", so the model picks up where it left off.

Both stages produce their own audit records stored with the turn. The wall-clock timeout continues ticking across a Stage 2 restart.

---

## Post-Turn Fan-Out (User Path Only)

After the turn is stored, the user processor triggers a set of services synchronously. Because the response is already on the wire (sent via narration callbacks during the loop), this fan-out does not affect perceived latency.

- **Conversation phase tracking** — updates the current phase based on both the user message and the response.
- **DMN timer reset** — defers the idle-trigger so proactive thoughts don't interrupt an active conversation.
- **Metrics counter** — increments request and user-message totals.

Background paths (DMN, goal pursuit, scheduled) skip all of this. They only update the request counter.

Personal facts are handled inline during the ACT loop: when the model decides to store something, it calls the memory ability directly. Contradiction detection happens at storage time.

---

## Background Paths

### DMN (Proactive Thought)

The DMN path fires on its own schedule — after an idle period with no user activity, and on a periodic cadence to surface high-salience topics. It respects quiet hours and a daily rate cap.

```
  DMN timer
      │
      ├── idle trigger  →  context: recent high-weight episodes
      │
      └── cadence trigger  →  context: top episodes + active goals
              │
              ▼
        ACT loop (lighter iteration cap, no pre-loop exploration)
              │
              ▼
        No-action signal? → exit silently
              │
              ▼
        Proactive push → WebSocket → client
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

Every WebSocket response frame carries a `metrics` block. Token counts span **all** LLM calls in the turn — the main ACT loop, the thinking exploration pass, and any Stage 1 or Stage 2 compaction calls. Tool counts record how many times each tool was called. The response time is measured from before the daemon thread is spawned.

```json
{
  "tokens_total": 4820,
  "tools": { "memory": 1, "search": 2 },
  "response_time_s": 1.43
}
```

Action-button responses (no LLM call) carry the block with zero tokens. Error frames carry whatever partial metrics were accumulated before the failure.
