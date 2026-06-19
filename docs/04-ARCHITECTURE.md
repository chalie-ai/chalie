# Architecture
 
 ## What Chalie Is
 
Chalie is a persistent personal AI — a single Python process that keeps thinking between conversations. Memory accumulates and decays over time, background workers reflect on what they've learned while the user is idle, and every conversation builds on everything that came before.
 
The stack:
 
- **Flask + flask-sock** — HTTP API and a push-only WebSocket, default port **31025**
- **SQLite** (WAL mode) — the only database; `sqlite-vec` for vector search, FTS5 for keyword search
- **MemoryStore** — an in-process, thread-safe key/value store with TTL (no Redis)
- **Pluggable LLM providers** — Ollama, Anthropic, OpenAI (+ OpenAI-compatible), Google Gemini
- **Vue 3 + Vite 5** — two SPA builds (`apps/interface` and `apps/brain`) in a pnpm workspace under `frontend/`; TypeScript strict, Pinia, Vue Router, SCSS; Flask serves the compiled `dist/` trees verbatim
 
Everything runs in one process. Workers are daemon threads supervised by a `WorkerManager` that health-checks and restarts them every 5 seconds. Threads communicate through SQLite, MemoryStore, and the WebSocket broker — there are no queues and no inter-process messaging.

**One user, many surfaces.** A Chalie instance serves a single user, but that user may have it open on several devices or tabs at once. There is one conversation; every surface mirrors it. The WebSocket broker (`backend/services/websocket_broker.py`) holds every live connection and fans each event out to all of them, so a message sent or received on any surface appears on all of them. A dropped socket reconnects on its own and rebuilds durable state from the database; missed in-flight events are tolerated because the conversation is reconstructed from persistence, not replayed. See [13-MESSAGE-FLOW.md](13-MESSAGE-FLOW.md#multi-surface-sync) for the user/assistant echo rules.
 
 ---
 
## The Runtime at a Glance
 
`backend/run.py` boots the database, warms the ONNX models, and registers these workers:
 
| Worker | What it does |
 |---|---|
| `rest-api-worker` | Flask HTTP + WebSocket server (the front door) |
| `scheduler-service` | Fires due reminders and scheduled prompts |
| `subconscious-worker` | The background-cognition tick (see below) |
| `world-awareness-service` | Hourly news scan based on the user's interests |
| `moment-context-service` | 6-hourly enrichment of recent conversation moments |
| `folder-watcher-service` | Watches configured folders and ingests new documents |
| `document-purge-service` | Periodic cleanup of deleted documents |
| `search-expander-service` | Generates and embeds paraphrase variants for new memory rows |
| `tmp-cleanup-service` | Temp-file cleanup |
| `mcp-server` / `mcp-client-heartbeat` | Inbound MCP server + outbound MCP connection keepalive |
 
 ---
 
## MessageProcessor — the One Way to Run a Turn
 
Every LLM turn in the system — a user chat message, a background reflection, a compaction, a web-research delegate — runs through the same single class:
 
```python
from services.message_processor import MessageProcessor
 
text = MessageProcessor.process(
    raw_input,              # the message text ("" for background turns)
    config,                 # a ProcessorConfig subclass — defines the channel
    metadata=None,          # optional: attachments, source, hidden_input, ...
    cancel_event=None,      # optional threading.Event for cooperative cancel
)                           # blocks until the turn completes, returns the reply text
 ```
 
There are no `MessageProcessor` subclasses. What varies between channels is **data, not type**: each channel supplies a frozen `ProcessorConfig` (base class in `backend/services/processor_config.py`, concrete configs in `backend/configs/channels/`).
 
A turn has three phases:
 
1. **Setup** — writes the input transcript row, scores the message with a small ONNX deliberation classifier (user channel only), and fires turn-zero seed tool calls: an automatic `memory` recall, one `document` upload per attachment, and — when deliberation resolves *high* — an internal `thinking` pass.
2. **ACT loop** — assembles one user message (conversation history as literal text, world state, the input, the accumulated tool trail) and calls the LLM. Tool calls are dispatched and their results fed back; the loop repeats until the model answers in plain text, the cancel event fires, or `config.max_iterations` is hit. If the request outgrows the context window, compaction fires transparently inside the loop and the request is rebuilt.
3. **Record** — writes the assistant transcript row, then runs `config.post_turn_hooks` in a failure-isolated loop.
 
On the user channel, tool activity is streamed live via WebSocket events (`act_tool_start`, `act_tool_end`, `act_narration`) to the surface that started the turn; the final `message`/`done` events are sent by the caller once `process()` returns. The `message` event — like the `user_message` echo broadcast when the message first arrives — fans out to every open surface, so both sides of the conversation appear everywhere. The live ACT trail is per-turn and bound to the initiating surface; it is deliberately *not* mirrored, being ephemeral turn scaffolding rather than durable conversation.
 
### Defining a new channel
 
Subclass `ProcessorConfig` and implement its three abstract prompt builders — that's the whole registration:
 
```python
@dataclass(frozen=True)
class MyChannelConfig(ProcessorConfig):
    def get_system_prompt(self, mp) -> str: ...
    def get_user_prompt(self, mp) -> str: ...
    def get_user_definition(self, mp) -> str: ...
 
MessageProcessor.process("", MyChannelConfig())
```
 
The config fields that matter most:
 
| Field | Effect |
|---|---|
| `channel` | Transcript/policy scope; e.g. `"user"`, `"dmn"`, `"delegate:web_search"` |
| `always_available` / `discoverable` / `blocked` | The turn's tool surface (see [09-TOOLS.md](09-TOOLS.md)) |
| `max_iterations` | ACT-loop cap; `None` = unbounded (the user channel) |
| `skip_transcript` | `True` = run silently, write no conversation rows |
| `suppress_history` | `True` = no `## Previous Messages` block (all background channels) |
| `memory_seed` | `True` = automatic memory recall at turn zero |
| `broadcast_to` | `'user'` = stream WebSocket events; `None` = silent |
| `post_turn_hooks` | Tuple of `PostTurnHook` objects run after the turn is recorded |
| `SUPPORTS_ASYNC` | ClassVar; `True` exposes an `async` flag on tool schemas (user channel only) |

To add a post-turn side-effect, subclass `PostTurnHook`, implement `run(mp, result_text)`, and add it to the channel's `post_turn_hooks` tuple. Hooks are independent and failure-isolated — never rely on ordering.
 
 ---
 
## Provider Layer
 
Every LLM call flows through one gateway: `Providers().send(dto)` (`backend/services/providers.py`). Callers build a provider-neutral `ProviderApiRequest` (system prompt, messages, tools, thinking level) and get back a normalized `ProviderApiResponse` (text, tool calls, token counts, latency).
 
- **Routing** is by `ProviderType`: `CHAT` resolves to the globally selected provider, `VISION` to the configured vision provider. There is no per-job routing.
- **Pre-flight cap check**: if the measured request leaves less than `max(10% of window, 8k tokens)` of headroom, `send()` raises `RequestOverCapError` *before* calling the provider — the ACT loop catches it, compacts, and retries.
- One thin client per platform lives in `services/llm_clients/` (`anthropic.py`, `openai.py`, `gemini.py`, `ollama.py`); `factory.py` picks the right one from the provider's `platform` string.
 
History reaches the model as a literal `## Previous Messages` text block inside a **single-element** `messages[]` array — Chalie controls exactly what context the model sees on every turn, independent of any provider's multi-turn format.
 
 ---
 
## Memory
 
Four layers, each on a different timescale:
 
| Layer | What it stores | Decays? |
|---|---|---|
| **Transcript** | Append-only record of every turn, channel-scoped, optionally GPS-tagged | Pruned after 90 days |
| **Compaction** | LLM-written continuity summaries; the newest `role='compaction'` transcript row is the history watermark | No |
| **Episodes** | Narrative snapshots extracted from transcript windows (see per-source profiles below), with salience and emotional scores; similar episodes consolidate into super-episodes | Yes — exponential decay on last relevance |
| **Data graph** | Structured facts (`user_specific`, `behavioral_pattern`, `place`, `moment`, `document`, …) with per-kind decay and contradiction/canonicalisation rules | Yes — per-kind policy |
 
Episode retrieval is hybrid: vector KNN + FTS5, reranked by relevance, recency, and salience, with a relative floor that drops weak candidates instead of padding results. Every data-graph write is also expanded asynchronously into paraphrase variants (doc2query) and embedded, so differently-worded questions still hit the right facts.

### Per-source memory profiles

Not every channel contributes to memory the same way. A single **allowlist** (`backend/services/source_profiles.py`) declares, for each transcript source, five orthogonal switches: whether its rows become episodes, feed fact extraction, count as the user's own movement in the geo window, count as user behaviour in the pattern window, and back-fill the user's live location. A channel absent from the table resolves to a fully-muted default, so a new source produces no memory until it is explicitly opted in.

| Source | Episodes | Facts | Geo / pattern = user activity | Notes |
|---|---|---|---|---|
| `user` | yes | yes | yes | The full pipeline. |
| `dmn` | yes | yes | no | The proactive reflection voice — its own episodes and facts, but its loop is not the user moving through the world. |
| `external-agent:*` | yes | yes | no | First-class memory, channel-tagged by agent. |
| `delegate:*`, `skills_building`, `scheduled` | no | no | no | Background loops are muted — their value surfaces through the parent or user-facing turn, not as standalone memory. |

Memory **reads cross channels.** Episode recall never filters by the caller's own channel, so a memory encoded by the proactive voice or an external agent surfaces in an ordinary user turn — exactly as structured facts already do. Because muted channels write no episodes, channel-agnostic recall naturally scopes to the sources that actually hold memory. The caller's channel is recorded only for recall-telemetry provenance.

**Provenance** is stamped at write time, not declared in the profile: a fact records the channel of the episode it came from (`fact_extraction:<channel>`), and a behavioural pattern records which background pass produced it (`pattern_match` for the behavioural pass, `geo_pattern` for the location pass).

 ---
 
## Background Cognition
 
The **subconscious worker** ticks every 5 minutes but only fires when the user has been idle for 30+ minutes and there is something new since the last run. Each tick runs seven steps, each isolated so one failure can't block the rest:
 
1. **Consolidate** — cluster episodes into super-episodes, per episode-producing channel (`user`, `dmn`, each `external-agent:*`); clusters stay channel-scoped and are never pooled across sources
2. **Decay** — run the decay engine over episodes, the data graph, old transcripts, and tool-call records (7-day retention). A fossil janitor tombstones stranded leaves on muted/legacy channels but protects episode-producing channels — the proactive voice never consolidates, so its leaves are permanently apex and must not be reaped
3. **Pattern match** — an LLM pass over new user-behaviour transcripts that records behavioural patterns and facts (`save_pattern` / `save_graph`), then maps patterns to skills
4. **Synthesis** — refresh the running user summary when new traits or patterns have appeared
5. **DMN** — a reflective ACT pass over the user summary and recent episodes; findings are saved to memory, nothing is pushed to chat
6. **Capability sync** — poll connected external services (mail, calendar, contacts)
7. **Geo patterns** — extract location-tied behavioural patterns from GPS-tagged transcripts. The pattern and geo windows (and their cursors) read only the channels the source profiles mark as user activity, so background loops never masquerade as the user being somewhere
 
All of these are ordinary `MessageProcessor.process()` calls with their own channel configs — there is no separate background engine.
 
**Delegate tools** (`web_search`, `web_browse`, `vision`) are focused sub-turns: each spawns its own ACT loop with a fixed minimal tool surface and returns a synthesized answer to the calling turn. On the user channel they can run async (the answer arrives as a follow-up assistant message); a delegate can never spawn another delegate.
 
 ---
 
## Where to Plug In
 
| You want to… | Do this |
|---|---|
| Add a tool the LLM can call | Add an `Ability` under `backend/abilities/` — see [09-TOOLS.md](09-TOOLS.md) |
| Add a new kind of LLM turn | Subclass `ProcessorConfig`, call `MessageProcessor.process()` |
| React after a turn completes | Add a `PostTurnHook` to the channel's config |
| Feed ambient context into prompts | Push a signal into WorldState — see [16-AMBIENT-AWARENESS.md](16-AMBIENT-AWARENESS.md) |
| Connect an external tool server | Add an MCP server via the `mcp_manager` tool or the Brain UI |
| Add an HTTP endpoint | Add a Flask blueprint under `backend/api/` |
| Add a frontend route or page | Add a Vue Router route in `frontend/apps/interface/src/router.ts` or `frontend/apps/brain/src/router.ts`; register any new serve path in `backend/api/__init__.py` (`_register_static_routes`) |
