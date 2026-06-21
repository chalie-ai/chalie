 # Message Flow
 
Every message — whatever triggered it — runs the same **ACT loop** through the one `MessageProcessor`. What differs per path is the trigger, the channel config, and where the result goes.
 
## The Four Paths
 
 | Path | Trigger | Result |
 |---|---|---|
| **User** | `POST /chat` from the web UI | Reply pushed to every open surface over WebSocket |
| **Subconscious** | 5-minute background tick (idle-gated) | Memory updates only — nothing pushed to chat |
| **Scheduled** | A due reminder or scheduled prompt | Runs the task in the background, then relays the result through a user-channel turn pushed to the client |
| **Delegate** | The model calls `web_search` / `web_browse` / `vision` mid-turn | A synthesized answer returned to the calling turn (async on the user channel) |
 
 ---
 
 ## User Path
 
The client sends messages over **HTTP** (`POST /chat`, multipart form-data with text + optional file attachments). The WebSocket at `/ws` is push-only — the server uses it to stream status, tool activity, and the final reply back to the client, and to echo every received user message on the same channel so all open surfaces stay in sync.
 
 ```
POST /chat
  └─ daemon thread per turn
       └─ MessageProcessor.process(text, UserConfig(metadata), cancel_event)
            └─ _run()      under the per-channel lock:
                 ├─ _setup()    input transcript row · deliberation gate · turn-zero seeds
                 ├─ _step()     one LLM call; tool-bearing steps write their assistant
                 │              row and recurse, a plain-text reply ends the chain
                 └─ _end_turn() post-turn hooks
       └─ WebSocket: {"type":"message", ...} + {"type":"done", ...}
 ```
 
**Turn-zero seeds.** Before the first LLM call, the framework dispatches real tool calls whose results land in the turn's tool trail (never injected into the prompt):
 
- a `memory` recall keyed on the input,
- one `document` upload per attachment — uploads fan out in parallel and are joined at a barrier, so vision/OCR extraction overlaps,
- an internal `thinking` pass, only when the deliberation classifier scores the message *high*.
 
**History is literal text.** Previous conversation is rendered as a `## Previous Messages` block inside the single user message — the provider never receives a multi-turn array.
 
**Interrupt & cancel.** `POST /chat/interrupt` stops the active turn; a cancelled turn deletes its transcript and tool-call rows, leaving no trace. Sending a new message mid-turn cancels the active turn and starts a fresh one with the original + new text combined. Cancellation reaches a turn even while it is parked waiting on an interactive permission prompt: the gate wait polls the cancel event, so a cancel resolves the prompt as denied and the turn unwinds, always releasing the per-channel lock so the replacement turn never deadlocks behind it.
 
**Connection resilience.** Because replies arrive only over the push channel, the client guards the socket on two fronts. A socket that closes cleanly (`onclose`) triggers exponential-backoff reconnection. A *half-open* socket — one a reverse proxy idle-drops at the TCP layer without sending a close frame, so `onclose` never fires and `readyState` stays `OPEN` — is caught by a liveness watchdog: the client stamps the arrival time of every inbound frame and, if none arrives for longer than 1.5× the server's keep-alive ping interval, tears the dead socket down and reconnects. A tab regaining focus re-runs the same staleness check on demand, so a backgrounded tab heals the moment the user returns instead of silently swallowing the next reply. Whichever path tears the socket down, the client also finalizes any in-flight turn locally — it collapses the live progress indicator and drops the turn's pending callbacks. The turn's terminal `done` was delivered to the now-dead socket and is never resent, so without this the progress indicator would hang indefinitely even though the backend had already finished the turn, and a stale callback could later misattribute an unrelated turn's completion.

### Multi-surface sync

A Chalie instance belongs to **one user**, but that user may have it open on several surfaces at once — phone, laptop, multiple tabs. There is one conversation, and every surface mirrors it. Two rules keep them aligned, both riding the single `/ws` broadcast channel that every surface subscribes to:

1. **User messages.** When a surface sends a message, the server stores it and broadcasts a `user_message` echo to every connection. The sending surface already rendered the bubble optimistically, so it recognises its own echo — by a client-minted `echo_id` round-tripped through `POST /chat` — and drops it; every other surface has no such bubble and renders one. The echo carries the text verbatim.
2. **Assistant messages.** The final reply is broadcast as a `message` event to every connection. The surface that started the turn renders it through its turn callbacks; peer surfaces render a plain assistant bubble from the broadcast. Either way the reply shows on all surfaces.

The live per-turn ACT trail (status and tool activity) rides the same channel — the broker fans **every** frame out to all live connections and prunes any socket whose send fails, so one stale connection can neither block delivery nor accumulate. What differs is rendering, not routing: only the surface that started the turn shows the trail, because idle peers drop content-free ACT frames client-side (a render-policy choice) — the trail is ephemeral turn scaffolding, not durable conversation. A surface that joins mid-turn simply shows the user and assistant bubbles once they broadcast.
 
 ---
 
## The ACT Loop
 
```
build request (history + world state + input + tool trail)
  → pre-flight size check ── over cap? ──► compact, rebuild, retry
  → send to LLM
  → no tool calls?  → done, return the text
  → tool calls      → dispatch each via ToolDispatcher → results appended to trail
  → next iteration (until done or cancelled)
```
 
- There is no iteration cap on any channel: the loop terminates only when the model answers in plain text (no tool calls) or the cancel event fires. The user can always interrupt a turn.
- Tool errors are returned to the model as structured result strings; they never crash the loop or surface raw to the user.
- Every tool call is written to the `tool_calls` table as it happens; a row lives and dies with its transcript turn — the decay engine's transcript GC reaps it together with the turn once that turn falls below the compaction watermark and is no longer cited by any live episode.
 
### Compaction
 
Compaction is size-driven only — there is no turn-count or age trigger. The `## Previous Messages` block each turn assembles keeps growing as the conversation continues; when the pre-flight check (or a provider-side size rejection) fires, the loop dispatches a single internal ability through the normal tool chokepoint:
 
1. **`chat_history_compactor`** — summarises the older `## Previous Messages` block; the summary is written to the transcript as a `role='compaction'` row, whose id becomes the new history watermark.
 
Because every iteration rebuilds the request from the database, advancing the watermark automatically shrinks the next request. If nothing is left to compact, the loop force-sends and lets the provider be the source of truth. The latest compaction summary is visible in Brain → Cognition → Compacted Summary.
 
 ---
 
 ## Background Paths
 
**Subconscious tick** — every 5 minutes, gated on 30+ minutes of user idleness, the subconscious worker runs its nine steps (consolidate, fact extraction, decay, pattern match, synthesis, DMN reflection, capability sync, geo patterns, proactive research). See [04-ARCHITECTURE.md](04-ARCHITECTURE.md#background-cognition). Each is a normal `MessageProcessor` turn on its own channel. Most write no transcript rows and none broadcast to chat. History compaction is **not** one of these steps — it runs in-loop during a turn when the request outgrows the context window (the over-cap path above).
 
**Scheduled prompts** — the scheduler worker polls for due items and fires each in two stages, modelled on the delegate tools. Stage one runs the instruction as an independent background turn on its own muted `scheduled` channel (full tool surface, no episodes or facts of its own), persisting the instruction so a fired task is recoverable. Stage two hands that result to an ordinary user-channel turn with `hidden_input=True`, which is what surfaces to the client and is episodically encoded — so the trigger text stays out of the visible conversation while the reply is delivered normally. The two stages run on a daemon thread so the poll never blocks on the LLM work.
 
**Episode encoding** — after turns are stored on an episode-producing channel (`user`, `dmn`, `external-agent:*` — see [04-ARCHITECTURE.md](04-ARCHITECTURE.md#per-source-memory-profiles)), a rolling trigger (every ~20 new transcript rows) encodes the recent window into episodic memory. No user-visible output.
 
 ---
 
 ## Per-Turn Metrics
 
Every final WebSocket frame carries a `metrics` block. Token counts cover **all** LLM calls in the turn — the main loop, any `thinking` pass, and any compaction calls:
 
 ```json
 {
   "tokens_total": 4820,
  "tools": { "memory": 1, "web_search": 1 },
   "response_time_s": 1.43
 }
 ```
