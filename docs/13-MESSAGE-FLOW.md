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
  │     sets thinking_level —    │
  │     a persisted user over-   │
  │     ride 'medium'/'high'     │
  │     short-circuits it)       │
  │  · _seed_turn_zero():       │
  │    - thinking dispatch      │
  │      (if thinking_level=    │
  │       'high'; ThinkingAbil- │
  │       ity via ToolDispatch) │
  │    - memory recall dispatch │
  │      (if memory_seed=True)  │
  │    - document.upload per    │
  │      attachment, uploaded   │
  │      in parallel at a       │
  │      barrier (if any)       │
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

**Image attachments are uploaded, vision-indexed, and searchable.** When a user turn carries attachments, `_seed_turn_zero()` uploads them **in parallel** through a `ThreadPoolExecutor` barrier — each via its own `ToolDispatcher` issuing a blocking `document.upload` — so the turn-0 context already holds every upload result before the ACT loop starts. A bad attachment path is logged and skipped without aborting its siblings. At upload, `text_extractor` routes images to `vision.describe_image(path, RICH_INDEX_PROMPT)` (the brain's Vision Provider, or RapidOCR when no provider is configured) and feeds the rich description into the existing `extract_text → create_document_artifacts → data_graph.store` pipeline — which embeds (sqlite-vec), FTS5-indexes, and doc2query-expands it — so `document.search` finds the image by its visual content. A textless image with no vision provider persists `status='ready'` (not `failed`). On later turns the model re-invokes the `vision` tool (`image=doc_id`, a new `query`) against the same document. PDFs keep their existing OCR extraction untouched.

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

Compaction relies **solely on context-window size** — there is no turn-count threshold — and it fires **compact-first**: the over-cap request is *never* sent as a partial view, it is compacted BEFORE sending. Both triggers route to the SAME chokepoint — `_dispatch_compaction()`:

| Trigger | Condition | How |
|---------|-----------|-----|
| **Window-fit (compact-first)** | `_loop()` calls `Providers.send(dto)`, which pre-flight-measures the request via `client.estimate_request_tokens(dto)`. When `measured >= cap`, `send()` raises `RequestOverCapError` (from `services.provider_api`) **without calling the provider** | `_loop` catches `RequestOverCapError`, calls `_dispatch_compaction()` FIRST, then rebuilds the collapsed DTO from the DB and re-sends. If `_dispatch_compaction()` made no progress (irreducible), it calls `_force_send(dto)` to let the provider be the source of truth |
| **Irreducible (force send)** | `_dispatch_compaction()` made no progress — the act-trail is already collapsed (an empty handover lands no boundary) **and** the chat-history watermark cannot advance (nothing left to fold) | `_loop` calls `_force_send(dto)`, which calls `client.send(dto)` directly (bypassing the cap check) so the provider receives the request and becomes the source of truth — it succeeds, or fails loud. There is **no** reactive 413 re-compaction path |

**The cap check (`Providers.send`).** On each ACT iteration `_loop()` builds a `ProviderApiRequest` DTO via `_build_send_dto()` and calls `Providers.send(dto)`. Inside `send()`: the client is resolved by `dto.type`, the context window is capped at `MAX_CONTEXT_WINDOW = 200_000`, and `client.estimate_request_tokens(dto)` measures the full request. If `measured >= window - max(int(0.10 * window), 8000)` — i.e. the request leaves less than `max(10% of the window, 8k tokens)` of response headroom — `send()` raises `RequestOverCapError`. `_loop` catches it and compacts first. A post-flight `ResponseOverLimitError` (provider-side size rejection — HTTP 413 on Ollama/Anthropic, `context_length_exceeded` on OpenAI, token-count error on Gemini) is caught by the **same** `except` clause, triggering the same compact-and-retry path. Because every ACT iteration rebuilds the request from the DB (never from in-memory state), advancing the watermark auto-clears both the `## Previous Messages` block and the `tool_calls`-built act-trail, so the rebuilt request fits. `_dispatch_compaction()` returns a **progress** boolean; when it returns `False` (nothing left to collapse — irreducible request) `_loop` calls `_force_send(dto)`, which skips the cap check and lets the provider fail loud rather than looping forever. Convergence is at most ~2 iterations.

There is **no proactive turn-count trigger, no `_fit_request`, no `drop_oldest_previous_message`, no `_compaction_pending`/`_payload_compacted`/`_history_drop` state, no `ContextOverflowError`, no retry counter, no `_compact()` method, and no inline compaction**. `_dispatch_compaction()` fires two INTERNAL, never-discoverable abilities through the normal `ToolDispatcher.dispatch()` chokepoint — the exact machinery as the turn-0 `memory`/`thinking` seeds. Because they go through dispatch, each one **auto-records its `tool_calls` row AND auto-emits the act-trail WebSocket events** (`act_tool_start`/`act_tool_end`) — that is how compaction shows up in the frontend act-trail with no hand-rolled emit. `_dispatch_compaction()` fires the two abilities in order:

1. **`tool_chain_compactor`** (fired first) — reads the current turn's act-trail off the bound `mp` via `_render_act_trail(for_compaction=True)`. If `_has_trail()` is false it silently returns `""` — no LLM call, no boundary. **"Trail" here means genuine agent activity:** both compactor markers AND the automatic turn-0 memory-recall seed (`memory(action='recall', _auto=True)`, identified by `_is_auto_memory_recall`) are excluded from the no-op check and from the compaction input (canonical design §2-3). A model-issued `memory` call has no `_auto` flag and counts as real trail. The per-turn act-trail the model sees (`_render_act_trail()`, default) still shows the auto-recall — only the compaction INPUT drops it. Otherwise it runs `MessageProcessor.process(trail_text, ToolChainCompactionConfig())` and returns the dense handover. The dispatch chain records that handover as the `tool_chain_compactor` tool_calls row — **that row (when its result is non-empty) is the new trail boundary**: `_from_last_compaction()` slices from it, so pre-compacted tool calls drop out of the rendered trail without a DELETE.

2. **`chat_history_compactor`** (fired second) — assembles its compaction input via `_fit_compaction_input(parent, prior)`: it carries the prior checkpoint forward as a `## Previous Summary` block in front of the parent channel's `get_previous_messages()` off `mp`. As a **rare fallback** (design step 4.2), if the bare compaction request itself (Previous Summary + `## Previous Messages` + the compaction system prompt, **no tools**) still exceeds `cap = window - max(10% window, 8k)`, it drops the oldest message one at a time (`get_previous_messages(drop_oldest=…)`) until it fits — this drop-oldest lives **inside the compactor**, never at the send layer. When the backlog is empty (or fully exhausted by drops) `_fit_compaction_input` returns `None` and the compactor returns without writing (nothing to fold). Otherwise it runs `MessageProcessor.process(combined, ChatHistoryCompactionConfig())` and writes the model's output **verbatim** to the `transcript` table as a durable `role='compaction'` row via `transcript_service.write_input_row(channel, "compaction", summary)`. The new row's own `id` becomes the watermark — `_previous_rows()` reads `id > watermark`, so the next read returns nothing through it. The watermark **always advances** on a non-empty backlog, so compaction can never silently no-op into an infinite loop. There is **no parser**: no `<analysis>`/`<summary>` tags, nothing to trim — whatever the model writes IS the checkpoint.

`get_previous_messages(drop_oldest=0)` renders **every** row since the watermark (`entries = self._previous_rows()`); the optional `drop_oldest` argument skips that many leading (oldest) rows (`entries = entries[drop_oldest:]`) and is used **only** by `chat_history_compactor._fit_compaction_input` for the rare 4.2 bare-request fallback. There is no fixed row cap and no persistent drop state — the main send path always renders the full backlog since the watermark, and compaction (advancing the watermark) is what bounds it.

Both compaction configs (`ChatHistoryCompactionConfig`, `ToolChainCompactionConfig`) set `thinking_mode = "high"` (a `ClassVar` read by `_build_send_dto()` via `resolve_thinking_mode`), carry `suppress_history=True`, expose no tools, and use `channel="compaction"` — a dedicated channel name so the inner compaction send's own `_wrap_with_checkpoint("compaction", …)` lookup is empty (continuity is instead carried explicitly via `## Previous Summary`). Watermark rows are written to the **parent** channel, never to `"compaction"`.

**Thinking-mode precedence.** `_build_send_dto()` resolves the level via `resolve_thinking_mode(config_thinking_mode, override, level)` = `config_thinking_mode or override or level` (module-level pure function in `services/providers.py`): a config hard-pin (the two compactors + `ThinkingAbility`, all `"high"`) wins over the persisted user override, which in turn wins over the gate-computed `mp.thinking_level`. The override is a single global setting (`settings` key `thinking_level_override`, values `medium`/`high`; absent ⇒ `auto`) read by `thinking_override_service.get_thinking_override()` and assigned to `mp.thinking_override` in `process()`. When set it reaches the provider on **every** channel (a user `medium` therefore never downgrades the compactors' pinned `high`); on the user channel it also short-circuits the deliberation gate. The composer exposes it bottom-left, alongside a bottom-right context-size indicator fed by `GET /system/context-usage` (last `job_name='user:user'` `tokens_input` ÷ selected-provider `max_tokens`), re-fetched on page load and on every inbound WS message. The indicator keys off `job_name='user:user'` — the main user turn — not `usage_class='chat'`, which is also written by the thinking pre-pass and every web_search/web_browse delegate iteration (they inherit the parent's CHAT policy_channel) and made the value oscillate between the user turn and a delegate's clean-context sub-request.

**Checkpoint envelope.** When a compaction row exists for the channel, `_wrap_with_checkpoint()` (called inside `_build_send_dto()`) wraps the user-prompt body into a `### Checkpoint - What you were previously discussing / doing` block followed by `### Current State - What's happening in the current turn`. No-op when no compaction row exists.

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

Every WebSocket response frame carries a `metrics` block. Token counts span **all** LLM calls in the turn — the main ACT loop, the `ThinkingAbility` exploration pass (when `thinking_level='high'`), and any compaction call fired by `_dispatch_compaction()` on the compact-first over-cap path. Tool counts record how many times each tool was called. The response time is measured from before the daemon thread is spawned.

```json
{
  "tokens_total": 4820,
  "tools": { "memory": 1, "search": 2 },
  "response_time_s": 1.43
}
```

Action-button responses (no LLM call) carry the block with zero tokens. Error frames carry whatever partial metrics were accumulated before the failure.
