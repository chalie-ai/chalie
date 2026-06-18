# Architecture
 
 ## What Chalie Is
 
Chalie is a persistent personal AI — a single Python process that keeps thinking between conversations. Memory accumulates and decays over time, background workers reflect on what they've learned while the user is idle, and every conversation builds on everything that came before.
 
The stack:
 
- **Flask + flask-sock** — HTTP API and a push-only WebSocket, default port **31025**
- **SQLite** (WAL mode) — the only database; `sqlite-vec` for vector search, FTS5 for keyword search
- **MemoryStore** — an in-process, thread-safe key/value store with TTL (no Redis)
- **Pluggable LLM providers** — Ollama, Anthropic, OpenAI (+ OpenAI-compatible), Google Gemini
 
Everything runs in one process. Workers are daemon threads supervised by a `WorkerManager` that health-checks and restarts them every 5 seconds. Threads communicate through SQLite, MemoryStore, and the WebSocket broker — there are no queues and no inter-process messaging.
 
 ---
 
## The Runtime at a Glance
 
`backend/run.py` boots the database, warms the ONNX models, and registers these workers:
 
| Worker | What it does |
 |---|---|
| `rest-api-worker` | Flask HTTP + WebSocket server (the front door) |
| `scheduler-service` | Fires due reminders and scheduled prompts |
| `subconscious-worker` | The background-cognition tick (see below) |
| `world-awareness-service` | Hourly news scan based on the user's interests |
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
 
On the user channel, tool activity is streamed live to the browser via WebSocket events (`act_tool_start`, `act_tool_end`, `act_narration`); the final `message`/`done` events are sent by the caller once `process()` returns.
 
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
| `always_available` | Tools pre-injected on the turn; everything else is reached via `find_tools` if it is `DISCOVERABLE` (see [09-TOOLS.md](09-TOOLS.md)) |
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
 
- **Routing** is by `ProviderType` (precedence: `VISION` > `DELEGATE` > `CHAT`):
  - `CHAT` — the globally selected provider (`selected_provider_id` in settings).
  - `VISION` — the configured vision provider (`vision_provider_id`); falls back to the main provider when it supports vision; resolves to "Disabled" only when no vision-capable provider exists.
  - `DELEGATE` — the provider used for subagent/delegate turns (`delegate_provider_id`), covering `web_search`, `web_browse`, and other delegated tool work; defaults to the main provider when no pin is set (there is no "Disabled" state for delegate).
  - `VISUAL_OUTPUT` — reserved, currently unused.
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
| **Episodes** | Narrative snapshots extracted from transcript windows (see per-source profiles below), with salience and emotional scores; episodes form a three-level hierarchy — leaf (0), topic super-episode (1), era digest (2) — via periodic density-clustering roll-ups | Yes — exponential decay on last relevance, per-level tau |
| **Data graph** | Structured facts (`user_specific`, `behavioral_pattern`, `place`, `document`, …) with per-kind decay and contradiction/canonicalisation rules | Yes — per-kind policy |
| **Moments** | User-curated bookmarks of individual assistant replies (the "remember this" pin), in their own table outside the data graph | No — lives until the user deletes it |
 
Episode retrieval is hybrid: vector KNN + FTS5, reranked by relevance, recency, and salience, with a relative floor that drops weak candidates instead of padding results. Every data-graph write is also expanded asynchronously into paraphrase variants (doc2query) and embedded, so differently-worded questions still hit the right facts.

The data-graph kind set is **closed and enforced**: a write with an unrecognised kind is rejected rather than stored, so no channel can invent its own kind. (`moment` used to be a data-graph kind; it is now its own store — see below.)

### Episode hierarchy and roll-up

Episodes sit in a three-level hierarchy. Every extracted episode starts as a **leaf** (`level=0`). When enough leaves accumulate on a channel they are density-clustered into **topic super-episodes** (`level=1`), and when enough of those accumulate they are further clustered into **era digests** (`level=2`).

**Count trigger.** Roll-up fires on a per-channel count, not a similarity floor: once a channel holds at least 50 leaf apexes, clustering runs. The count trigger always eventually fires; a similarity gate can silently never fire on densely-packed embedding spaces.

**Clustering pipeline.** The apex embeddings for a channel are assembled into a matrix, L2-normalised, then reduced to 10 dimensions with UMAP (cosine metric, pinned seed for deterministic output), and finally clustered with HDBSCAN (minimum cluster size 10, euclidean metric on the reduced space). UMAP reduction is mandatory — raw 768-dimensional HDBSCAN collapses into one blob, and PCA produces degenerate results at every dimensionality. HDBSCAN returns a noise label (−1) for genuine outliers; those episodes are **never force-assigned** and remain leaf apexes eligible for the next round.

**Era digests.** After the leaf round, the worker reads the stored summary embeddings of level-1 apexes on the same channel. If at least 25 are present, the same UMAP→HDBSCAN pipeline clusters them into level-2 era digests. There is an intentional one-tick lag: the era round reads level-1 apexes from before the current leaf round, so a newly created super-episode waits at least one tick before it can roll up further.

**Hierarchy write contract.** When a parent is written, `store_episode` stamps its `level` (1 or 2) and `last_relevant_at`. Each child episode receives `consolidated_into` (back-pointer to the parent id) and `tombstoned_at` in a single atomic update — a child can never carry a back-pointer without a tombstone. The per-level decay tau then applies correctly: leaf 14 days, level-1 90 days, level-2 365 days, tombstoned 7 days. Tombstoned episodes are hard-deleted by the janitor after 30 days. The decay and deletion steps are owned by the decay engine (step 2 of the background tick); the roll-up only writes the markers.

**Per-tick cap.** To prevent an overdue roll-up from monopolising one background tick, the worker summarises at most 5 clusters per tick. Remaining clusters roll up on the next qualifying tick.

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

### Moments

A **moment** is an explicit user bookmark of a single assistant reply — the "remember this" pin. Moments live in their own `moments` table (with companion `moments_fts` lexical index and `moments_vec` semantic-KNN index), entirely outside the data graph. Each moment is keyed to one assistant transcript turn (`transcript_id` is unique — one pin per turn) and stores the reply verbatim.

Unlike data-graph facts, a moment carries **no retrieval weight, no decay, and no janitor** on its own table — it persists until the user explicitly forgets it. The content embedding is computed **synchronously when the user pins**, because pinning is a deliberate user action rather than part of the chat turn's hot path.

Moments surface in memory recall as a **clearly-labeled lane**, but only on **explicit recall** — never in the silent turn-zero seed. So a pinned bookmark is available when the user (or the model) deliberately searches memory, yet it never leaks unbidden into the start-of-turn flashback.

A standing step in the decay cycle wipes any leftover `kind='moment'` rows from the data graph (a one-time clean-up of a legacy storage path; idempotent, and a no-op once drained). Moments are also covered by the "delete all my data" privacy path. The `moments` tables are created declaratively by the schema-convergence pass — there is no migration code.

 ---
 
## Background Cognition
 
The **subconscious worker** ticks every 5 minutes but only fires when the user has been idle for 30+ minutes and there is something new since the last run. Each tick runs eight steps, each isolated so one failure can't block the rest:
 
1. **Compact** — proactively fold the user channel's accumulated history into its durable compaction watermark before any other work, so the context window stays small while the user is away. This runs only the channel's compactors through the normal dispatch path (no LLM turn), and is a pure no-op — no LLM call, no watermark write — when nothing new sits past the watermark, so re-running it on a later idle tick costs nothing
2. **Consolidate** — run the hierarchy roll-up on each episode-producing channel (`user`, `dmn`, each `external-agent:*`): a leaf round (level-0 → level-1) fires at 50+ apex leaves, and an era round (level-1 → level-2) fires at 25+ level-1 apexes; both use the UMAP→HDBSCAN pipeline described above; clusters stay channel-scoped and are never pooled across sources
3. **Decay** — run the decay engine over episodes, the data graph, old transcripts, and tool-call records (7-day retention), and wipe any leftover legacy `kind='moment'` rows from the data graph (moments now have their own table). A fossil janitor tombstones stranded leaves on muted/legacy channels but protects episode-producing channels — the proactive voice never consolidates, so its leaves are permanently apex and must not be reaped
4. **Pattern match** — an LLM pass over new user-behaviour transcripts that records behavioural patterns and facts (`save_pattern` / `save_graph`), then maps patterns to skills
5. **Synthesis** — refresh the running user summary when new traits or patterns have appeared
6. **DMN** — a reflective ACT pass over the user summary and recent episodes; findings are saved to memory, nothing is pushed to chat
7. **Capability sync** — poll connected external services (mail, calendar, contacts)
8. **Geo patterns** — extract location-tied behavioural patterns from GPS-tagged transcripts. The pattern and geo windows (and their cursors) read only the channels the source profiles mark as user activity, so background loops never masquerade as the user being somewhere
 
Every step after the compaction pass is an ordinary `MessageProcessor.process()` call with its own channel config; the compaction pass builds a user-channel `MessageProcessor` and runs only its compactors. There is no separate background engine.
 
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
