# System Architecture

## What Chalie Is

Chalie is a persistent cognitive runtime — a single Python process that keeps thinking between conversations. It is not a request-response wrapper around an LLM. Every message flows through a multi-layer memory pipeline, background workers maintain and decay knowledge while the user is idle, and the system forms a continuously evolving model of the person it talks to. Intelligence accumulates over time; it does not reset per session.

The stack: Flask + flask-sock for HTTP and WebSocket, SQLite (WAL mode, sqlite-vec for vector search, FTS5 for keyword search), an in-process thread-safe MemoryStore (no Redis), and a pluggable LLM provider layer supporting Ollama, Anthropic, OpenAI, and Google Gemini. Everything runs in one process.

---

## How a Message Flows

A user message arrives over WebSocket. The handler spawns a daemon thread and runs the turn via `MessageProcessor.process(raw_input, make_user_config(...))` — one flat class, one per-channel config (see Message Processors below). Nothing else touches the message.

`process()` builds a fresh instance and runs `_run()` → `_setup()` → `_loop()` → `_record()`:

1. **Setup & turn-zero seed** — `_setup()` writes the input transcript row (capturing the uid) and runs the deliberation gate: a lightweight ONNX classifier assigns a continuous deliberation score (0.0–1.0) on the row; higher scores nudge the prompt toward more careful reasoning and very high scores trigger a one-shot pre-reasoning exploration pass before the tool loop begins. It then calls `_seed_turn_zero()`, which (user channel) dispatches a `memory` recall keyed on the input and one `document` upload per attachment — each through `Ability.dispatch()`, so the matches and uploaded files land in the turn-0 trail rather than being injected into the prompt.
2. **ACT loop** — `_loop()` assembles a single user message containing the literal conversation history, world state, and the current input, then calls the LLM. If the LLM invokes a tool, the result is appended to the trail and the loop continues. This repeats until the model returns a plain text response, a cooperative cancel signal is received (`_cancel_event`), or `config.max_iterations` is reached. The user config sets `max_iterations=None` (unbounded) — a user turn runs until the model finishes or the user stops it via `POST /chat/interrupt` (or `POST /chat/subagent/<sub_id>/stop` to cancel a running async delegate — a legacy route name). When a turn is cancelled, `_cleanup_cancelled_turn()` deletes all tool_call and transcript rows for that turn so no trace persists. Background and delegate configs set explicit caps (DMN=100, external-agent=200, pattern-match=100, geo-pattern=30, delegates=50).
3. **Atomic write** — `_record()` commits the user turn, every tool call from the loop, and the assistant response in one SQLite transaction, then purges the turn's ephemeral trail rows. Nothing is written to the database mid-loop.
4. **Post-turn fan-out** — `config.post_turn(mp, response)` runs if set (save-suggestion detection on the user channel; `None` for background channels). The response is already on its way to the client before fan-out begins. Metrics are recorded inside the provider send gateway, not here. (Pre-v0.6.0 also reset a DMN idle-timer here; DMN is now driven entirely by the subconscious worker tick.)

```
WebSocket frame
  └─ daemon thread
       └─ MessageProcessor.process(raw_input, make_user_config(…))
            ├─ _setup()  (input row · deliberation gate · _seed_turn_zero:
            │             memory recall + document upload via Ability.dispatch)
            ├─ ACT loop ──────────────────────────────────────────┐
            │    assemble prompt (history + world state + input)  │
            │    → LLM call                                       │
            │    → tool calls → tool results → back to LLM  ─────┘
            ├─ _record() atomic write (transcript + tool_calls, one tx; purge ephemeral)
            └─ post_turn fan-out → response → client
```

See `docs/13-MESSAGE-FLOW.md` for the full turn lifecycle.

---

## Message Processors

`MessageProcessor` is the **single class** behind every LLM turn — there are no per-channel subclasses. What varies between channels is *data, not type*: each channel is described by a frozen [`ProcessorConfig`](backend/services/processor_config.py). The architectural rules are simple:

- **One flat class, one config per channel.** `MessageProcessor.process(raw_input, config)` is the single entry point. The config carries the channel's identity (`channel`, `role`, `usage_class`), its three prompt builders (`build_user_prompt`, `build_user_definition`, `build_system_prompt`), its tool tiers (`always_available`, `discoverable`, `blocked`), and loop control (`max_iterations`, `skip_transcript`, `suppress_history`, …). There is no shared dispatcher, no central router, and no subclass holding channel state.
- **One instance per turn.** `process()` builds a fresh instance, attaches the config, and runs `_run()` → `_setup()` → `_loop()` → `_record()`. All turn state lives on that instance; nothing is shared between turns.
- **Configs are module constants or factories.** Constant channels expose a module-level instance (`DMN_CONFIG`, `EPISODE_ENCODER_CONFIG`, `COMPACTION_CONFIG`, …); per-request channels expose a factory (e.g. `make_user_config(metadata=…)`). All live under `backend/configs/channels/`.
- **Atomic record at the end.** `_record()` persists the assistant row and turn metrics once the ACT loop finishes, then purges the turn's ephemeral trail rows.
- **`Ability.dispatch()` is the single dispatch chokepoint.** Every ACT-loop tool call — user channel, DMN, delegates, action buttons, even the framework's turn-0 seed calls — routes through this one static method (`sanitize → policy → timeout thread → execute → record`). Tool errors return structured strings to the LLM; they never surface to the user or crash the loop.
- **`post_turn` is the only optional hook.** It is a `ProcessorConfig` field (`(mp, response_text) -> None`), not a subclass method; `None` means no fan-out. Channel-specific side-effects hang off it.

History reaches the LLM as a literal `## Previous Messages` text block inside the user message body. The provider always receives a single-element `messages[]` array — not a multi-turn array. This is an intentional design choice: it gives the system full control over what context the model sees on each turn.

Compaction is just another flat turn: `MessageProcessor.process(<input>, COMPACTION_CONFIG)`. **One universal config** serves every channel (delegates included) and **both** compaction kinds — there is no per-channel or per-subagent compaction config. The two kinds differ only in input and output row, never in config: **trail compaction** (>90% context) summarises this turn's tool-call trail into an *ephemeral* `tool_name='trail_compaction'` row; **history compaction** (>80%) summarises older transcript into a *durable* `tool_name='compaction'` row. A recursion guard — `_COMPACTION_CHANNELS = frozenset({"compaction"})` — stops a compaction turn from compacting itself. `compaction_persistence.get_compaction(channel)` reads the latest durable success row; there is no separate `compactions` table. Both kinds currently share a single system-prompt body, `ContinuityCompactionSystemPrompt` (in `system_message_prompt.py`).

Background channels (episode encoders, the user-summary synthesiser, and the compaction turn) set `config.skip_transcript = True`, so they run the full ACT loop without writing to the conversation record.

---

## Memory Hierarchy

Four layers, each optimised for a different timescale and purpose:

| Layer | What it stores | Decays? |
|-------|---------------|---------|
| **Transcript** | Append-only conversation record, channel-scoped | No (pruned after 90 days) |
| **Compaction** | LLM-generated continuity summaries stored as `tool_calls` rows with `tool_name='compaction'`; append-only, failure rows kept as audit | No |
| **Episodes** | Narrative units extracted from transcript windows | Yes — power-law retrieval weight decay |
| **Data Graph** | Structured knowledge (facts, preferences, moments) | Yes — per-kind decay policy |

**Transcript** is the raw record. Each row optionally carries `location_lat`, `location_lon`, and `location_name` — auto-tagged from the client's GPS heartbeat at write time via `locale_service.get_location()`. `get_previous_messages()` renders everything above the compaction watermark as a literal text block. When that block approaches the provider's context limit, compaction fires and summarises the older portion — the latest success summary becomes the new floor. Compaction results are stored as `tool_calls` rows (`tool_name='compaction'`, `ephemeral=0`), not in a separate table. `compaction_persistence.get_compaction(channel)` retrieves the most recent success row via a join on `transcript.channel`; failure rows are recorded for audit but filtered from the lookup so a bad LLM output cannot poison subsequent prompts.

**Episodes** are extracted automatically by a rolling trigger: when enough new transcript lines have accumulated for a channel, a background processor encodes that window into narrative snapshots with emotional valence, arousal, and salience scores. Each episode inherits the dominant location from its transcript window (`location_lat`, `location_lon`, `location_name`) — enabling location-filtered memory recall. Similar episodes consolidate into super-episodes over time. Retrieval uses hybrid vector + FTS5 search, adaptive radius, and apex traversal (following consolidation links upward). All retrieval paths (`_get_episode_raw`, FTS, vector) surface location columns.

**Data Graph** is the knowledge layer. Writes for user-specific facts go through a canonicalisation engine: the key is compared against a set of high-level concepts, and a rule (temporal supersede / coexist additive / immutable block) is applied. This prevents duplicate or contradictory facts from accumulating. The database shape lives in `backend/schema.sql`.

**Query expansion.** Every data-graph write is enqueued to the `SearchExpanderService` — a single boot-time FIFO daemon. It generates paraphrased variants via doc2query, embeds each, and writes them to `expanded_semantic` + `expanded_semantic_vec` keyed back to the source rowid. Recall adds a KNN signal against the variant index so paraphrased questions hit the right facts even when the literal surface form does not match. The daemon is event-driven (not busy-loop) and self-heals on boot by rescanning rows with `search_queries IS NULL`.

---

## Background Reasoning

Chalie keeps thinking when you are not typing. Background workers run as daemon threads in the same process:

- **Subconscious worker** (v0.5.0 §5, extended in v0.6.0) — a single daemon thread (`subconscious-worker`) that is the **sole owner of latent cognition**. It ticks every 5 minutes (`SUBCONSCIOUS_TICK_SEC`). On each tick it checks two gates: the **user-active** gate (`WorldState.snapshot().last_user_message_at` within the last 30 min) and the **already-fired** gate (`subconscious_last_fired_at > last_user_message_at`). The already-fired gate also fires after a process restart when `subconscious_last_fired_at` was hydrated from durable storage but no user message has arrived yet — the previous lifetime already covered the open idle window, so the worker waits for fresh signal before running again. When both pass, it runs **seven steps** in this exact order, each isolated in `_safe_step()` so one failure cannot block the rest: (1) consolidate apex episodes into super-episodes per channel via `MessageProcessor.process("", make_super_episode_config(...))` — gated to `channel='user'` upstream by `transcript_service._maybe_trigger_extraction`, so non-user channels (DMN, scheduled, delegate, …) never produce episodes; (2) `DecayEngineService.run_once()` (the engine instance is built once and cached on the worker so the per-tick config read is amortised); (3) the pattern-match pass (a flat `MessageProcessor` run under `make_pattern_config(...)`) followed immediately by `SkillAssociationService` (see below); (4) user-summary synthesis (`MessageProcessor.process("", make_user_summary_config())`) — gated by `_should_synthesise()`, which returns `''` when no new traits or behavioural patterns have arrived since the last successful run; (5) DMN reflection (`MessageProcessor.process("", DMN_CONFIG)`) — see DMN entry below. The tick-complete log line reads `tick complete: consolidate=… decay=… pattern_match=… synthesis=… dmn=… capability_sync=… geo_patterns=…` and is the canonical ordering signal. Re-entrancy is guarded by a non-blocking lock; concurrent ticks return without doing work. The next tick is anchored to `monotonic()` after the current tick returns, not before — long ticks therefore extend the cycle rather than starve the next gate. State (`subconscious_last_fired_at`) is mirrored to MemoryStore (`subconscious:last_fired_at`) and to `data_graph` (`kind='system'`, `key='subconscious_last_fired_at'`) so the gate state survives restarts. (6) **capability sync** — calls `monitor()` on every connected capability; `MailCapability._do_monitor()` manages per-protocol cadence internally (IMAP every cycle, CalDAV every 3rd ~15 min, CardDAV every 12th ~60 min). The scheduler is not involved in triggering syncs — it only stores calendar event data. (7) **geo pattern extraction** — a flat `MessageProcessor` run under `make_geo_config(cursor, latest)` fires when ≥30 new location-tagged transcripts have accumulated since the last cursor (`data_graph` kind=`system`, key=`geo_pattern_cursor`). Same architecture as the pattern-match pass but reads only `location_lat IS NOT NULL` rows, includes coordinates in the prompt, and focuses the LLM on location-tied behavioural patterns. Uses `save_pattern` and `save_graph` (kind=`place`). No decay sweep — decay is shared with the pattern-match pass.
- **Decay engine** — applies power-law decay to episode retrieval weights and data-graph entries, and purges old transcript entries and tool-call rows. The engine itself (`DecayEngineService`) has no daemon thread of its own; it exposes `run_once()` and is called by the subconscious worker as Step 2 of its tick.
- **Pattern matcher** — the pattern-match pass runs as Step 3 of the subconscious worker tick when ≥50 new transcripts have accumulated since the last cursor (`data_graph` kind=`system`, key=`pattern_match_cursor`). One LLM forward pass over the new transcript window with two processor-scoped tools: `save_pattern` (UPSERT a `behavioral_pattern` row with confidence math: new=7, reinforced=`min(10, prev+7)`, capped at 10; budget 20 calls) and `save_graph` (routes through `DataGraphService.store()` for `user_specific` / `misc` / `moment` / `document` kinds; budget 50 calls). The model emits all calls in parallel; `max_iterations=100` bounds runaway loops. After the pass, an in-place SQL decay sweep subtracts 0.005 from every untouched active pattern's confidence; rows that hit 0.0 flip to `active=0` (soft delete). The user-summary pass reads active behavioral_pattern rows directly when assembling its synopsis prompt — no tool surface for user turns. Immediately after the pattern-match pass completes, **`SkillAssociationService().run_pass()`** fires as Layer 2 of the Self-Refining Skill Library: it reads all active `behavioral_pattern` rows from `data_graph`, reads the skill index from `skills.sqlite`, calls the LLM once to map patterns to skills, and writes personalisation rules into the `skill_associations` table in `skills.sqlite`. A failure in this pass is logged at WARNING and does not block the rest of Step 3 or subsequent steps.
- **DMN (Default Mode Network)** — Step 5 of the subconscious worker tick (v0.6.0). `MessageProcessor.process("", DMN_CONFIG)` runs one ACT pass against `user_summary_long` (fallback `user_summary`, else skip with `status='skipped'`) plus channel='user' episodes (retrieval_weight ≥ 0.3, 30-day window, LIMIT 50). The pass saves findings via the `memory` tool (writes `data_graph` rows) and never broadcasts to the chat UI — there is no `enqueue_proactive` call, no `DMN_NO_ACTION` sentinel, and no proactive output channel. `DMN_CONFIG` uses `DEFAULT_ALWAYS_AVAILABLE` (`find_skills`, `find_tools`, `memory`) and sets `blocked = frozenset({"web_search", "research", "web_browse", "summariser"})` — a background reflection pass may not spawn delegate work; all other tools remain discoverable via `find_tools`. The pre-v0.6.0 `DMNService` daemon (idle/cadence triggers, OutputService.enqueue_proactive) and the `BackgroundLLMProxy` queue (`background_llm_worker.py`, `background_llm_queue.py`) are deleted — the worker tick is the single trigger.
- **Delegate tools** — `web_search`, `research`, `web_browse`, and `summariser` each run a focused ACT turn on a `delegate:<name>` channel (`MessageProcessor.process(goal, config)`, `max_iterations=50`, 600 s deadline). On the user channel they run async (daemon thread, immediate ack, synthesis delivered back via `dispatch_message(..., hidden_input=True)` — the single chat chokepoint in `api/chat.py`); nested delegate calls (e.g. `summariser`→`web_search`) run synchronously inline. They replace the former `subagent` ability and its `SubagentProcessor`. See the "Delegate tools" section below.
- **Scheduled prompts** — the scheduler fires due reminders and timed tasks via `dispatch_message()` with `hidden_input=True` (always starts a fresh user-channel turn).
- **Supporting workers** — world awareness (weather, news), moment context enrichment, document purge, folder watcher, and the `SearchExpanderService` (single FIFO consumer that generates + embeds query variants for every new data-graph row). User-summary synthesis and super-episode consolidation no longer run their own daemons — both are driven by the subconscious worker tick.
- **EmbeddingService** — a module-level singleton that serialises all ONNX inference through a single daemon worker thread via a FIFO queue (`_embedding_queue`). All callers — `generate_embedding(text)`, `generate_embedding_np(text)`, `generate_embeddings_batch(texts)` — check MemoryStore first; cache hits bypass the queue entirely. The worker is started lazily on the first job submission (not at import time) so tests that never call the service never spawn a real ONNX thread. The single-worker model eliminates concurrent `session.run()` calls, which each allocate 500 MB+ of working memory and caused OOM under bulk document ingestion. Session construction routes through `onnx_session.build_session()` (see below).
- **onnx_session.py** — single chokepoint for all ONNX session construction in the process. `choose_providers(model_path)` returns the ordered provider list, applying the Metal 16384 2D-texture ceiling check for CoreML (any initializer dimension exceeding 16384 triggers automatic removal of `CoreMLExecutionProvider`). `build_session(path, opts, providers, log_prefix)` constructs the session and retries with CPU-only on construction failure. `EmbeddingService`, `VoiceService` (`voice.py`), and `Doc2QueryService` all route through this module — no service constructs `ort.InferenceSession` directly.

No worker shares its processor instance with another. Each channel is fully isolated.

### Delegate tools

The four **delegate tools** replace the former `subagent` ability and its `SubagentProcessor` (both removed). Each is a standalone `Ability` (`backend/abilities/{web_search,research,web_browse,summariser}.py`) that builds its **own** `ProcessorConfig` and runs a focused ACT turn via `MessageProcessor.process(goal, config)` — there is no subclass, no `SUBAGENT_TYPES` registry, and no `make_subagent_config()` factory.

| Tool | `always_available` surface | Goal param |
|------|----------------------------|------------|
| `web_search` | `search`, `read` | `query` |
| `research` | `memory`, `search`, `read`, `find_tools` | `goal` |
| `web_browse` | `browser`, `read` | `goal` |
| `summariser` | `read`, `document`, `web_search` | `goal` |

**Shared config shape** (built inside each ability's `run()`): `channel=f"delegate:{NAME}"`, `role=NAME`, `usage_class="subconscious"`, `max_iterations=50`, `discoverable=[]`, `blocked=build_blocked(self._TOOLS)`, plus `skip_transcript=True`, `skip_input_row=True`, `suppress_history=True`, `broadcast_to=None`, `memory_seed=False` — a delegate is a scratch turn, not a conversation.

**Recursion guard.** `_delegate.py`'s `build_blocked()` blocks every delegate name *not* in the caller's own surface, so a delegate can never spawn another delegate — except the one sanctioned step a tool explicitly lists in `always_available` (`summariser`→`web_search`). `DELEGATE_TOOL_NAMES` is the canonical set; `DELEGATE_DEADLINE_SECONDS = 600` is the wall-clock horizon; `delegate_goal(params)` normalises `goal`/`query`; `render_trail(mp)` renders the caller's act-trail into the delegate's user prompt.

**Delivery is channel-decided, not type-decided.** Every delegate sets `ASYNC_CAPABLE = True`, but `Ability.dispatch()` takes the async path only when the *dispatching* channel supports it — `_supports_async_delivery(channel)` is `True` only for `channel == "user"`:

- **Async (called on the user channel):** `dispatch()` spawns a daemon (`_run_async_delegate`), registers a cancel `Event` in the module-level `_active_delegates` registry (keyed by `delegate_id`), and returns an immediate ack. On completion the daemon delivers the synthesis via `dispatch_message(result, channel="user", hidden_input=True)` — the single chat chokepoint — which starts a fresh user-channel turn; the raw result never lands in the transcript, only the synthesized assistant response. The user lists active delegates via `GET /chat/subagents/active` (`Ability.get_active_delegates()`) and cancels one via `POST /chat/subagent/<sub_id>/stop` (`Ability.cancel_delegate(sub_id)` — a legacy route name).
- **Sync (called on any non-user channel):** `dispatch()` runs the delegate inline via `_run_with_timeout(...)` and blocks the parent iteration until it returns. This is the path a nested `summariser`→`web_search` call takes.

`web_browse` replaces the former `web_surfer` subagent type.

---

## ONNX Runtime Selection

### Install-time wheel dispatch

`installer/install.sh` detects the host GPU before installing Python dependencies and swaps the `onnxruntime` wheel accordingly:

| Detected hardware | Wheel installed |
|-------------------|-----------------|
| NVIDIA GPU (`nvidia-smi` present) | `onnxruntime-gpu` |
| AMD GPU (`/dev/kfd` + `amdgpu` kernel module) | `onnxruntime-rocm` (from AMD's manylinux index) |
| Everything else | `onnxruntime` (CPU) |

The CPU wheel is always installed first as a baseline. The GPU wheel replacement only happens after a `pip install --dry-run` confirms the download would succeed — so installs on machines without network access to the GPU index remain on CPU rather than failing. ORT version is pinned at `1.20.1` as a single source of truth in the installer. `backend/pyproject.toml` does not pin `onnxruntime` directly; it carries `rapidocr_onnxruntime` which transitively pulls the CPU wheel for development workflows that bypass the installer.

For air-gapped AMD installs, set `ROCM_PIP_INDEX` to a local mirror before running the installer.

### Runtime provider selection

All session construction goes through `backend/services/onnx_session.py`:

- `choose_providers(model_path)` — returns the ordered execution provider list. On macOS, any ONNX model whose initializer tensors include a dimension exceeding **16384** (the Metal 2D-texture ceiling) has `CoreMLExecutionProvider` stripped automatically. `gte-modernbert-base` trips this limit (vocab embedding is `{50368, 768}`). The check runs at session construction time and emits a `[<prefix>] Dropped CoreMLExecutionProvider: model has dim > 16384` log line when it fires.
- `build_session(path, opts, providers, log_prefix)` — constructs the `InferenceSession`. If construction raises with the chosen providers, it retries with `["CPUExecutionProvider"]` before propagating the error. This makes individual model failures non-fatal for the rest of the process.

`EmbeddingService`, `VoiceService`, and `Doc2QueryService` all call `build_session` — no service constructs `ort.InferenceSession` directly.

### Asset layout

Two distinct on-disk directories separate runtime-downloaded weights from pre-shipped classifier files, with one extra location for pre-shipped sqlite-vec/FTS5 search indexes. All locations are resolved by `FileMapperService` (`backend/services/file_mapper_service.py`) — the single source of truth for the on-disk layout. There are no env-var or CLI-flag overrides.

| Path | Tracked in git | Contents |
|------|----------------|----------|
| `data/models/` | No (gitignored) | Encoder ONNX (`gte-modernbert-base`), `doc2query-small`. Downloaded on first boot or installer step. |
| `resources/voice-models/` | No (gitignored, downloaded by installer) | Kokoro TTS ONNX + voices (`kokoro/kokoro-v1.0.onnx`, `kokoro/voices-v1.0.bin`) and Moonshine STT ONNX (`moonshine/base/encoder_model.onnx`, `moonshine/base/decoder_model_merged.onnx`). Downloaded by `installer/install.sh` at install time. Placed outside `data/` so the files bake into the Docker image and survive `chalie update`. |
| `backend/pre-trained/` | Yes | Per-task classifier meta + `.npz` MLP heads (`deliberation_score/`, `mode_detector/`) plus drift sidecars for pre-shipped search indexes (`abilities_sha.json`, `skills_sha.json`). Cloning the repo is enough to classify on first turn — no GitHub release fetch. |
| `backend/abilities/assets/` and similar `*/assets/` directories | Yes (binary diff suppressed via `.gitattributes`) | Pre-shipped sqlite-vec/FTS5 search indexes (`abilities.sqlite`, `concept_lut.sqlite`, `search_tool_providers.sqlite`, `skills.sqlite`). Built by `python -m utils.build_ability_db` (and equivalents); `skills.sqlite` is built separately by `python -m utils.build_skills_db` from YAML files in `backend/abilities/skills/`. A CI `--check` step compares the per-row sha to the sidecar in `backend/pre-trained/` and fails the build on drift. `abilities.sqlite` indexes **every** registered ability, including `save_pattern` and `save_graph` (1 SUMMARY row + 6–8 EXAMPLE rows each, all embedded at 768 dim). Discovery scoping is enforced at query time via each processor's `DISCOVERABLE` list, not by excluding rows from the index. Drift sidecars: `backend/pre-trained/abilities_sha.json`, `backend/pre-trained/skills_sha.json`. `skills.sqlite` additionally holds the `skill_associations` table (written at runtime by `SkillAssociationService`) — the build script recreates the schema on each rebuild, so rebuild clears any accumulated personalisation data. |

`OnnxInferenceService.__init__(models_dir, pretrained_dir)` takes both. The shared encoder ONNX is resolved against `models_dir`; per-task classifier directories resolve against `pretrained_dir`. The singleton accessor passes `FileMapperService.get_models_path()` and `FileMapperService.get_pretrained_path()` — tests pass tmp dirs to exercise corruption / contract paths.

The pre-shipped `<task>-classifier_meta.json` is the authoritative calibration source for each head — alpha, bucket thresholds, sha256 pin. Missing or corrupt meta files raise at boot rather than falling back to baked-in defaults; per-turn callers (e.g. the user-channel turn) catch the construction error and degrade to a safe default (`thinking_level='low'` for the deliberation gate). If `_preload_models` itself raises before the singleton finishes registering tasks, the outer except annotates `OnnxInferenceService._failed_registrations = [("preload", ...)]` and flips `_ready = True` so the `/health` endpoint can explain the degraded state instead of reporting a generic not-ready.

---

## Tools and Skills

Two loading tiers stack on every user turn and are merged first-seen, so the unconditional tier can never be shadowed by a dynamic entry of the same name:

**Tool scope lives on each channel's `ProcessorConfig`**, not on a base class — there are no `MessageProcessor` subclasses. Three fields control visibility (defaults in `configs/channels/_common.py`):

- `always_available` — ability names pre-injected as native tools on every ACT iteration by `AbilityRegistry.build_tools()`. `DEFAULT_ALWAYS_AVAILABLE` = `["find_skills", "find_tools", "memory"]`.
- `discoverable` — ability names that `find_tools` may surface for this config at runtime. `DEFAULT_DISCOVERABLE` lists every first-party ability a normal channel may reach, including the delegate tools (`research`, `summariser`, `web_browse`, `web_search`). The SQL query inside `find_tools` filters candidates to `WHERE name IN (discoverable - blocked)`, so a config can never discover anything outside its own list.
- `blocked` — `frozenset` of ability names excluded from both `discoverable` and the `find_tools` index. Default: empty. A config sets it to exclude specific tools without redeclaring the full list.

The user and external-agent configs use `DEFAULT_ALWAYS_AVAILABLE` and `DEFAULT_DISCOVERABLE` unchanged (`blocked` empty). `DMN_CONFIG` shares those defaults but sets `blocked = frozenset({"web_search", "research", "web_browse", "summariser"})` — a background reflection pass may not spawn delegate work. `find_skills` is always-available rather than discoverable because, like `find_tools` and `memory`, it is a meta-tool: returning procedural playbooks is infrastructure, not a task-specific capability. The user config maps `skip_input_row = bool(metadata["hidden_input"])`, so scheduled-prompt and external-agent triggers (which pass `hidden_input`) keep the raw trigger out of the user transcript while the synthesized response still lands. The pattern and geo-pattern configs set `always_available = ["save_pattern", "save_graph"]` with `discoverable = []`.

**Discoverable abilities** are never pre-injected. The `find_tools` ability performs semantic search against the abilities index at runtime. When the LLM invokes `find_tools`, the matching abilities become available for the remainder of that ACT loop. All first-party abilities are reachable exclusively through this path — pre-injecting them would bloat context, create staleness bugs, and break tool-agnostic routing.

`find_tools` performs **RRF (Reciprocal Rank Fusion) discovery** exclusively against `backend/abilities/assets/abilities.sqlite`. `abilities.sqlite` indexes **every** registered ability — including `save_pattern` and `save_graph`. Discovery scoping is enforced at query time via the calling processor's `DISCOVERABLE` list, not by selectively excluding rows from the index. Each query runs two independent retrievals — (a) sqlite-vec k-NN (`KNN_DEPTH=30`) over per-entry embeddings and (b) FTS5 BM25 over the same entries — both filtered to `WHERE name IN (allow)`. Results are grouped per ability (best distance / best BM25 score per ability), ranked independently, then fused via `score(a) = Σ 1 / (15 + rank_i(a))` across the two lists. The top-`k` abilities by fused score are returned. The non-standard `RRF_K=15` (vs. the standard 60) is deliberate: at ~17 candidates, k=60 compresses scores into too narrow a band for crisp separation. If `EmbeddingService` fails, a keyword-only fallback queries `ability_search_fts MATCH ? AND name IN (allow)` in `abilities.sqlite` directly. If `abilities.sqlite` is missing, both paths return `[]` with a `[FIND_TOOLS]` WARNING.

`ModeGateService` is a built-but-currently-dormant prompt-steering layer; it **does not gate tool availability**. The design: a small ONNX multi-label classifier (`mode_detector`, eight heads — `research`, `coding`, `brainstorm`, `analyze`, `plan`, `write`, `math`, `converse`) emits per-mode probabilities; `tick()` folds them into a per-mode EMA that snaps up on a fire and decays by 0.75 on a miss (a topic stays "warm" for ~four turns), persists state in MemoryStore under `mode_gate:state` (cleared on `/privacy/delete-all`), and `get_system_prompt_additions()` appends steering directives (long-summary swap on `converse`; brainstorm/research/analyze suffixes; a mandatory-`code_eval` block on `coding`/`math`) once a mode's EMA crosses `STEER_THRESHOLD = 0.6`. The user config already reads this: `configs/channels/user.py` appends `get_system_prompt_additions()` when `mp._mode_gate_cached` is set. **Currently dormant:** the flat refactor dropped the wiring that ran the gate per turn (the old `UserMessageProcessor._get_mode_state()`), so `ModeGateService` is never instantiated or `tick()`-ed in production and `_mode_gate_cached` is never assigned — the classifier never runs on a real turn and no steering ever fires. Tracked by TKT-800 (re-wire in `_setup()` or remove the layer).

Tool results flow through a single render-and-record path (`ToolRenderAndRecordService`) that formats the output and writes it to the `tool_calls` table. Tool infrastructure has no knowledge of specific tools; tools have no knowledge of infrastructure.

**Native tool calling only.** Every provider adapter (`OllamaService`, `AnthropicService`, `OpenAIService`, `GeminiService` in `services/llm_service.py` + `services/ollama_service.py`) reads tool calls exclusively from the response's structured `tool_calls` (or equivalent) field. There is no content-side fallback that scrapes XML/JSON tool-call markup out of `message.content`. If a model fails to populate the structured field, that turn produces zero tool dispatches by design — silent inline-content rescues mask model misbehaviour and were ripped to keep failure modes loud and observable.

Every ability result uses the canonical tag block format from `backend/services/innate_skills/_tag.py` (the only file remaining under `services/innate_skills/` after Phase 4 cutover — kept because it is purely a formatter shared by every ability):

```
[<ability_name>(k1=v1, k2=v2)]
<body>
[end:<ability_name>]
```

This is the single source of truth — no ability constructs its own format string. See `docs/09-TOOLS.md`.

**Turn-zero seeding.** There is no `pre_act()`/`send()` hook — both were removed with the subclasses. Instead, `_setup()` calls `_seed_turn_zero()` after the input transcript row is written (so `self._uid` is populated) but before iteration 0. It performs two declarative, config-gated behaviours, each via `Ability.dispatch()` so the call blocks, records a durable `tool_calls` row, and is rendered into the trail exactly like an LLM-issued call: (a) when `config.memory_seed` is set (the user channel), one `memory` `recall` keyed on the raw input; (b) one `document` `upload` per file in `metadata["attachments"]` (upload is the ingest — no second auto `view`). The model's first turn already sees the memory matches and uploaded documents in its trail — they are not injected into the user prompt.

---

## Ability Framework

`backend/abilities/` is the **sole dispatch path** for every cognitive operation an LLM turn can invoke. Legacy `services/innate_skills/*_skill.py` modules and `tools/<name>.py` modules have been ripped — every dispatchable capability is now an `Ability` subclass. The only files kept under their old paths are companion modules that an `Ability` imports from (`tools/browser/{security,pool,extraction,interaction,credentials}.py`, `tools/search/{router,fetcher,transformers}.py`) and the shared formatter `services/innate_skills/_tag.py`.

### `Ability` ABC and `SearchableAbility`

Every concrete ability subclasses `Ability` (`backend/abilities/_base.py`) and declares:

| Class attribute | Type | Notes |
|-----------------|------|-------|
| `NAME` | `str` | Stable identifier; matches the channel-level tool name. |
| `SEARCH_TOOLTIP` | `str` | 2–5 word description for the `find_tools` index. Required on every non-INTERNAL ability (enforced by `__init_subclass__`). |
| `SUMMARY` | `str` | One sentence used as the SUMMARY row in `abilities.sqlite`. |
| `EXAMPLES` | `list[str]` | 6–8 natural-language phrases for EXAMPLE rows (enforced by `__init_subclass__`). |
| `INPUT_SCHEMA` | `dict` | JSON Schema for `execute()` params. |
| `TIMEOUT` | `int` | Per-call timeout in seconds (default 10). |

Tool scope (always-available vs discoverable) is **not** declared on the `Ability` ABC. It is owned by each channel's `ProcessorConfig` via its `always_available` and `discoverable` lists (see Tools and Skills section).

The `execute(channel, params, telemetry)` method is the sole per-ability execution surface. `pre_dispatch(params) -> dict` is called by `Ability.dispatch()` before policy enforcement — abilities override it to normalise or escalate parameters (e.g. `BashAbility` uses it to upgrade the LLM's self-classification via heuristic inspection). The default is a pass-through. `ASYNC_CAPABLE: ClassVar[bool] = False` opts an ability into async delivery on user-channel calls (daemon thread, immediate ack, result delivered via `dispatch_message` on completion).

`SearchableAbility(Ability, ABC)` (`backend/abilities/_search.py`) is an intermediate base class for abilities that search a vec+FTS5 sqlite database. It provides `rrf_merge()` (`@staticmethod`, pure RRF fusion), `_hybrid_search()` (vec+FTS5 query using `self._DB_PATH`), and `_fts_only_search()` (FTS fallback). Subclasses declare `_DB_PATH` and `_LOG_PREFIX` as `ClassVar`s and implement domain-specific SQL and formatting as methods. `FindToolsAbility` and `FindSkillsAbility` both inherit from it. Module constants `RRF_K=15` and `KNN_DEPTH=30` are importable from `abilities._search`.

### Concrete abilities

`abilities/{bash, browser, calendar, chalie_docs, code_eval, contacts, document, email, file_permissions, file_write, find_skills, find_tools, home, list, mcp_manager, memory, news, place, programming_docs_search, read, research, review_tool_calls, review_transcript, save_graph, save_pattern, schedule, search, search_files, skill_builder, skill_manager, summariser, timer, ubiquiti, weather, web_browse, web_download, web_search}.py`. `abilities.sqlite` indexes every registered ability; `abilities_sha.json` mirrors the per-row sha and is checked in CI. (`save_pattern` / `save_graph` are `SYSTEM` abilities reachable only by the pattern-match and geo-pattern passes — see below.)

Per-ability implementation notes:

- `bash` — safe shell execution via `bash -c`. LLM self-classifies commands into 7 action categories (read, execute, modify_file, web_fetch, installation, remote_execution, compound). `pre_dispatch` applies heuristic escalation-only overrides via `shlex.split` command-word matching and quote-aware compound detection. Destructive patterns (rm -rf /, fork bombs, mkfs, dd) are unconditionally blocked. Environment is sanitised to strip secret-bearing vars. Policy: `bash.read`=allow in chat, all others=ask, all 7=deny for external_agent. DISCOVERABLE only.
- `browser` — `playwright.sync_api` imported unconditionally at module top. Companion modules under `tools/browser/`.
- `calendar` — read ops (`list_events`, `get_event`) query `scheduled_items` via `query_items()` in `abilities/schedule.py` (the single SQL path for that table, filtering `hidden=1, source='mail', item_type='event'`). Write ops (`update_event`) delegate to `MailCapability`'s CalDAV handler. Discoverable in the user + DMN configs only.
- `chalie_docs` — self-reference ability routing "what is chalie", "chalie tools", "release notes", "codebase" queries to chalie.ai URLs via the `read` tool. Enum-gated `query` param (`basics`, `tools`, `releases`, `code-base`). Discoverable in the user + DMN configs.
- `code_eval` — `_RESTRICTED_GLOBALS` built once as `ClassVar[dict]`; a fresh `dict()` copy is taken per call so state never leaks between executions.
- `contacts` — delegates to `MailCapability`'s CardDAV handler closures. Actions: `list`, `get`. Discoverable in the user + DMN configs only. CardDAV contacts are stored in `data_graph` with `kind='user_specific'`, `key='contact:<Display Name>'`, and a JSON value containing `fn`, `given_name`, `family_name`, `nickname`, `emails` (typed: `[{"value": "…", "type": "work"}, …]`), `phones` (typed, same shape), `org`, `title`, and `uid`. IMAP sender contacts continue to use the lightweight `key='contact:<email>'`, `value='<display_name>'` format. Both formats are handled transparently by `contact_resolver.resolve()` and `_parse_contact_row()`.
- `document` — `create_document_artifacts` exposed as both a module-level function and a `classmethod` so `api/documents.py` and `services/folder_watcher_service.py` import the same path.
- `email` — delegates to `MailCapability`'s IMAP/SMTP handler closures. Actions: `search`, `read`, `draft`, `manage`, `send`, `reply`, `forward`. Send/reply/forward require SMTP credentials; reply and forward read the original email internally and return it in the response so the LLM can see the full content it acted on. All three outbound actions are `ask`-gated in chat policy and denied in subconscious. Custom mail servers are supported via `build_custom_provider()` — pass `imap_host`/`imap_port`/`imap_tls` and optional `smtp_host`/`smtp_port`/`smtp_tls`, `caldav_url`/`carddav_url` to the setup endpoint. Discoverable in the user + DMN configs only.
- `find_skills` — inherits `SearchableAbility`. RRF (vec + FTS5) discovery against `abilities/assets/skills.sqlite`. Returns curated and user-created step-by-step tool-calling playbooks. Filters by `enabled=1`. Falls back to FTS-only search when `EmbeddingService` fails. ALWAYS_AVAILABLE on all user-facing processors.
- `skill_builder` — CRUD for user-defined skill playbooks. Actions: create, edit, delete, list. User skills stored as YAML in data/skills/user/. On create/edit, indexed into skills.sqlite for find_skills routing. Only source=user skills can be edited or deleted. Discoverable in the user + DMN configs. Brain Skills tab (/api/skills) provides a separate REST CRUD surface.
- `find_tools` — inherits `SearchableAbility`. RRF discovery against `abilities.sqlite` (see Tools and Skills section). ALWAYS_AVAILABLE on all user-facing processors.
- `home_assistant` (file: `home.py`) — integrates exclusively with Home Assistant via its REST and WebSocket APIs. Chalie does not communicate directly with device protocols (MQTT, Zigbee, Z-Wave, Matter) or third-party cloud APIs — HA is the sole integration point. Delegates to `HomeCapability`'s tool handler closures. Actions: `list_devices`, `get_state`, `control`, `list_automations`, `trigger_automation`, `subscribe_events`. Dual-protocol: REST (`requests`) for the first five actions, persistent WebSocket (`websocket-client` daemon thread) for `subscribe_events` and `_do_monitor()` liveness. Read actions (`list_devices`, `get_state`, `list_automations`, `subscribe_events`) are `allow`-gated; write actions (`control`, `trigger_automation`) are `ask`-gated. Events forwarded via Redis `output:events` pub/sub. Discoverable in the user + DMN configs only.
- `web_search`, `research`, `web_browse`, `summariser` — **delegate tools**: each runs a focused ACT turn on a `delegate:<name>` channel and (on the user channel) delivers its synthesis back asynchronously via `dispatch_message(..., hidden_input=True)`. See the "Delegate tools" section. They replace the removed `subagent` ability.
- `list` — `_DEFAULT_LIST_NAME` as `ClassVar[str]`; handler helpers at module level.
- `memory` — 8 radius constants promoted to `ClassVar` (`RECALL_RADIUS_BASELINE`, `SEED_RADIUS_BASELINE`, etc.) so the meta-harness can patch them by name. Module-level `recall_episodes()` function preserved for importability by the turn-0 seed path (`_seed_turn_zero`).
- `place` — save/list/get/delete named locations (home, work, gym). Stores in `data_graph` with `kind='place'`, `key=<label>`, JSON value `{lat, lon, name, radius_m}`. GPS coordinates read from telemetry at save time. Discoverable in the user config only.
- `news` — `_service` classvar lazily initialised via `_get_service()` classmethod.
- `programming_docs_search` — all 23 `_Source` subclasses and `_ALL_SOURCES`/`_ALIAS_MAP` at module level.
- `read` — `requests` at module top; `_BROWSER_HEADERS`, `_BLOCKED_PATH_PREFIXES`, `_URL_FETCH_TIMEOUT` as `ClassVar`. SSRF guard delegated to shared `abilities/_ssrf.py`.
- `web_download` — streaming file download from URL to `/tmp/chalie_downloads/` (default) or explicit destination. SSRF guard via shared `abilities/_ssrf.py`, scheme validation (http/https only), blocked destination prefixes, 100 MB size cap, retry on `ConnectionError`, SSL verify-then-fallback. Policy: chat=ask, external_agent=deny.
- `review_tool_calls` — returns `dict` directly.
- `schedule` — atomic dedup `INSERT...WHERE NOT EXISTS` preserved verbatim; `_PAST_DUE_GRACE_SECONDS` as `ClassVar[int] = 120`.
- `search` — `_DB` path resolves to `tools/search/assets/search_tool_providers.sqlite`; companion router/fetcher/transformers stay under `tools/search/`. The router scores queries via k-NN over pre-embedded `provider_examples` rows (1 190 total); three thresholds govern dispatch: `_MIN_SCORE=0.50` (below = routing miss, DDG-only fallback), `_WEAK_SCORE=0.60` (below but ≥ 0.50 = routed providers AND DDG appended, `meta["ddg_supplement"]=True`), and `_GAP=0.10` (max score distance from top to still include a secondary provider). Extend the example bank with `backend/utils/seed_routing_examples.py` (INSERT OR IGNORE, idempotent), then regenerate with `cd backend && python -m utils.generate_search_cache`.
- `weather` — Open-Meteo (primary, coordinate-based) + wttr.in (city-name fallback). `_cache` and `_CACHE_TTL=600` as `ClassVar`s so the 10-minute cache is shared. Open-Meteo response also carries `hourly=temperature_2m,weather_code` and `daily=…,sunrise,sunset`; `_extract_hourly_strip` slices the 8 entries starting from the current local hour (matched by `YYYY-MM-DDTHH` prefix against `current.time`) and the payload exposes `sunrise`, `sunset`, `hourly` for the FE ambient-sky card. wttr.in fallback returns these as `None`/`[]` so the FE shape stays stable.

### Pattern-match helpers — `save_pattern` / `save_graph`

`abilities/save_pattern.py` (`SavePattern`) and `abilities/save_graph.py` (`SaveGraph`) are `Ability` subclasses flagged `SYSTEM = True` — the `SYSTEM` flag keeps them out of the policy UI (`AbilityRegistry.policy_visible()` excludes them). They register like any other ability and `abilities.sqlite` indexes them, but they are reachable only because the pattern and geo-pattern configs set `always_available = ["save_pattern", "save_graph"]` (with `discoverable = []`); no other config lists them, so no other channel can surface them and `find_tools` never offers them. They are resolved into native tool schemas by `AbilityRegistry.build_tools()` from the config's `always_available` list — there is no `get_tools()` method on the flat `MessageProcessor`. Per-call budget and decay-tracking state (`_save_pattern_calls`, `_save_graph_calls`, `_touched_pattern_ids`, …) lives on the calling processor instance and is read via `current_processor()`.

### Registry (`backend/abilities/_registry.py`)

Singleton with an `RLock`. Exposes `get(name)`, `all()`, the static `build_tools(mp)` (the native-tool builder — see Tools and Skills), and `policy_visible()`. Lazily walks `backend/abilities/` on first access via shallow `glob("*.py")` (skipping files starting with `_`), then traverses `Ability.__subclasses__()` filtering out abstract classes. The registry is the single source of truth for which abilities are active in the process and is the only path an `Ability` instance is created — no module elsewhere instantiates an ability directly. Tool scope (which channel sees which ability) is not the registry's concern; that belongs to each channel's `ProcessorConfig`.

---

## Chat File Attachments

Chat attachments use HTTP only — the WebSocket is server→client push (see the WebSocket section). The frontend uploads each file via `POST /upload`, which stores it under `/tmp/chalie_*` and returns that `tmp_path`. The send then `POST`s `/chat` with the message text plus an `attachments` array of those paths (capped at 10); `dispatch_message()` forwards them to the processor as `metadata['attachments']`.

On turn 0 — before the first LLM iteration — `_seed_turn_zero()` iterates each `tmp_path` in `metadata['attachments']`. `_read_attachment()` realpath-resolves each path, rejects anything that does not resolve under `/tmp/chalie_` (traversal guard), and base64-encodes it. For each file it then issues **one** blocking tool call through `Ability.dispatch()`:

1. `document(action='upload', name=..., content=..., content_type=...)` — persists to permanent storage, runs extraction synchronously, returns a document ID; the `/tmp` file is deleted afterwards.

Upload *is* the ingest: there is no second auto `document(action='view')` call. Because the seed call routes through `Ability.dispatch()`, it records a `tool_calls` row rendered into the turn-0 trail, so the model's first turn already sees the uploaded document.

Both calls go through `Ability.dispatch()` — policy enforcement, WS tool events, and `tool_calls` audit rows are generated naturally. The results land in the ACT trail so the LLM sees file content on iter-0 of the ACT loop.

Image attachments are handled by the document pipeline: `image_context_service.analyze()` uses the configured vision provider for a comprehensive description when one is set (TKT-715), otherwise local OCR. The `/documents/upload` REST endpoint remains available for the Brain document-management UI.

---

## Ambient Awareness

`WorldState` maintains a lightweight in-process snapshot of four typed facts: last user message timestamp, last heartbeat timestamp, current device class, and client-reported local time. The snapshot is updated via `world_state.absorb(Signal(...))` — called from the WebSocket handler on every user message and from POST `/health` on every client heartbeat.

`absorb()` and `snapshot()` operate on an in-process dict with no database writes and no LLM calls. Unknown signal kinds are silently ignored. All four snapshot fields default to `None` when no signal of that kind has been received since boot.

See `docs/16-AMBIENT-AWARENESS.md` for the full `Signal` dataclass contract and call sites.

---

## Frontend

Four independent single-page applications: the main chat interface, the brain admin dashboard, the onboarding wizard, and the login form. A shared auth-gate module enforces redirect rules uniformly.

The chat interface is built from focused ES6 modules wired together by a thin orchestrator (`app.js`). Modules communicate through constructor injection, callback registration, and custom DOM events. No module references another directly.

**HTML Markup Format:** all LLM-to-client content is a single `content` string of HTML tags from a fixed allowlist. Backend `services.markup.sanitize()` (nh3, the OWASP-aligned ammonia sanitiser) is the single chokepoint — every assistant response is sanitised before reaching the frontend. Tags / attributes outside the allowlist are stripped; the frontend renders the result via `innerHTML` and trusts the chokepoint.

**Rich-Media Segments:** certain tools (currently `weather`, `list`, `timer`, `calendar`, `contacts`) opt into structured card rendering by appending a rich-media instruction trailer to their return string. The trailer tells the LLM to wrap its synthesis in `<span id='<tool>_<N>'>…</span>`. The sanitiser whitelists `<span id>` so the tag survives the nh3 pass intact. `RichMediaParser.parse(content, tool_calls)` (`backend/services/rich_media_parser.py`) then runs at two sites — the WebSocket `message` event assembly and the `/conversation/recent` refresh path — and converts the sanitised text plus the turn's `tool_calls` rows into an ordered `segments` array:

```
[
  {"type": "text",  "content": "…"},
  {"type": "rich",  "tag": "weather_1", "payload": {…}, "synthesis": "…"},
  …
]
```

Both sites produce byte-identical output because both read `tool_calls` from the database (including `ephemeral=1` rows), not from in-memory state.

**Per-ability payload enrichment.** The parser stays tool-agnostic: any card whose runtime state lives outside the LLM-visible tool result implements `Ability.enrich_rich_payload(cls, payload, row)`, called once per rich segment. `TimerAbility` uses it to inject `started_at` from `row.created_at` (the wall-clock anchor stays out of the LLM's reach). `ListAbility` uses it to re-fetch the live list from `ListService.get_list(id)` so checkbox mutations made via the silent-action channel are visible on refresh — without this, `tool_calls.result` would replay a stale snapshot. The default `Ability.enrich_rich_payload` is identity, so cards with no runtime state inherit it for free.

**Policy enforcement.** `Ability.dispatch()` calls `PolicyManager.wrap(channel, permission, callback)` before running any ability. `channel` is a `ProcessorConfig.POLICY_CHANNEL` enum value; `permission` is the ability action string (e.g. `email.send`). Three settings: `internal` (always allowed, hidden in Brain UI), `allow` (proceed), `ask` (surface a WebSocket `permission_request` to the chat UI — only on the `chat` channel; `subconscious` and `external_agent` auto-escalate to deny), `deny` (reject immediately). The gate is a flat `policy(channel, permission, setting)` SQLite table. On a cache miss, `PolicyManager._setting()` inserts an `ask` row lazily — unknown actions default to ask rather than skipping enforcement. Channels with no user present (`subconscious`, `external_agent`) treat `ask` as `deny` automatically, eliminating the latent forever-hang. The static default seed (`backend/abilities/assets/policy_defaults.json`, 404 rows) is applied via `INSERT OR IGNORE` on every boot by `_migrate_policy_table()` in `run.py`, which also copies any legacy `policy_rules` rows into the new table before `SchemaConvergenceService` runs. `_permission_gates` lives in `services/policy_manager.py`. Schema: `policy(channel, permission, setting)` + `policy_blocked_log`. API: `GET /api/policies` (flat triples, `internal` rows excluded) · `PUT /api/policies` (single-cell upsert) · `GET/DELETE /api/policies/blocked` · `POST /api/policies/reset`. `GET /api/policies` also returns a `meta` key — a sorted list of `{action_id, label, category}` objects derived from `POLICY_CATEGORY` / `POLICY_LABELS` class attributes on each Ability. The Brain Policies tab consumes `meta` to render grouped toggles with human-readable labels; no hardcoded lists exist in the frontend.

**Channel gate — non-user isolation.** `Ability.dispatch()` uses `config.channel` to gate channel-specific behaviour. Rich-media ordinal injection is gated on `channel == 'user'`, so on any non-user channel (delegates on `delegate:<name>`, DMN, the background passes) a rich-media tool returns a plain dict with no instruction trailer — a card never escapes into a parent's ACT trail. (`services.rich_media_parser.strip_spans()` remains as a defensive scrubber for stray `<span id='name_N'>…</span>` wrappers, but nothing currently calls it — the ordinal gate is the active mechanism.) See `docs/superpowers/specs/2026-05-02-rich-media-cards-design.md` for the full protocol.

**LLM-emittable tags (8):**
- `<b>`, `<i>`, `<u>` — inline emphasis
- `<h1>` — heading
- `<code>` — code (inline or block, model decides)
- `<p>` — paragraph
- `<ul><li>` — list

The LLM is **NOT** allowed to emit `<a>`. Plain-text URLs are auto-linkified by the frontend via `linkifyjs` so the model cannot inject arbitrary anchors into the rendered DOM.

**Programmatic-only tags (3):**
- `<img src="..." alt="...">` — backend-emitted images (browser screenshots, generated assets). `src` restricted to `http(s):` by nh3.
- `<actions>` / `<action label="..." value="..."/>` — backend-emitted interactive button rows. `<action>` carries chat-action attributes (`label`, `value`) and overlay-action attributes (`execute`, `collect`, `target`, `open-url`, `payload`, `style`).

The system prompt forbids the LLM from emitting programmatic tags or `<a>`.

**Wire shape:**
```json
{"type": "message", "content": "<p>Hello <b>world</b></p>", ...}
```

**Backend module:** `backend/services/markup.py` — `sanitize()` (nh3 chokepoint, accepts mixed plain text + allowlisted HTML and passes both through), `extract_plaintext()` (TTS — strips tags + drops `<actions>` subtree), `actions_to_xml`, `escape_attr`. Zero hand-rolled tokenisation; nh3 owns all parsing. There is no "is this XML?" heuristic and no plaintext-to-`<p>` wrapper — text nodes are valid HTML so wrapping was wrong: it turned mixed-mode model output into entity-escaped literals.
**Frontend module:** `frontend/interface/markup_renderer.js` (innerHTML + linkify + programmatic wiring) + `markup_extract.js` (DOM walk for TTS plaintext). `frontend/interface/vendor/linkify.es.mjs` (linkifyjs 4.3.2, vendored ESM, ~20 KB).
**No boot migration:** legacy markdown→HTML migration was retired with `markdown_xml_migration.py` after the nh3 cutover. Existing transcripts retain their stored content — sanitisation runs on every render.

**Asset versioning:** every static asset reference in served HTML has the version string injected into its filename at response time (e.g. `app.js` becomes `app-0.3.3.js`). Static routes strip the version suffix before the disk lookup, so nothing is renamed on disk. Versioned filenames are used instead of query strings because some service workers and proxies ignore query strings when keying caches. HTML responses themselves are never cached.

**MIME registration:** `backend/api/__init__.py` calls `mimetypes.add_type()` at import time for `.js`, `.mjs`, `.json`, `.css`, and `.html`. Python's `mimetypes` module reads the Windows registry on Windows, which frequently returns `text/plain` (or `None`) for `.js` files due to stale or missing entries — strict browsers (Chrome, Edge) refuse to execute scripts served with the wrong MIME type. Registering the canonical types overrides any bad registry mapping, with no effect on macOS/Linux (where the defaults are already correct).

See `docs/03-WEB-INTERFACE.md` for the full Radiant design system spec.

---

## REST API

The REST API covers conversation, memory, privacy, providers, tools, scheduling, lists, and observability. Endpoints live in `backend/api/`. See individual blueprints for the full surface.

---

## Key Architectural Rules

These are invariants, not conventions. Violating them creates systemic problems.

- **Atomic per-turn persistence.** All database writes for a turn happen in a single transaction at the end of the ACT loop. No mid-loop writes.
- **Literal-text history.** Previous messages are rendered as a text block, not a multi-turn `messages[]` array. The provider always sees one user message.
- **Flat channels.** A channel is a stable string identifier. Additional routing context (goal ID, scheduled item ID) lives in metadata, not in the channel string.
- **Channel-scoped data.** Transcript, compaction, and episode data are keyed by channel. No cross-channel leakage.
- **Tool agnosticism.** No tool-specific logic in triage, dispatch, or frontend rendering. Tools self-declare via manifests. Innate skills are the exception.
- **No external tool pre-injection.** External tools are discovered at runtime via `find_tools` only. Never add tool schemas to the native tools list, system prompt, or pre-loaded context.
- **Clean removal.** When a service, class, or function is removed, delete it completely — file, imports, callers, tests. No hollow passthroughs, no deprecated shims, no re-exports.
- **Model-agnostic.** Different cognitive functions may use different LLM providers. Nothing is hardcoded to a specific model.

---

## Glossary

| Term | Meaning |
|------|---------|
| **MessageProcessor** | Single class for all LLM turns. One instance per turn; channel behaviour comes from a frozen `ProcessorConfig`, not a subclass. |
| **Channel** | Stable string scoping transcript and compaction data (e.g. `user`, `dmn`, `delegate:web_search`). |
| **HTML Markup Format** | Content format: single `content` string of HTML, backend → frontend. Backend `services.markup.sanitize()` (nh3) is the chokepoint. LLM emits 8 formatting tags (no `<a>`); backend programmatically emits `<img>`, `<actions>`, `<action>`. Frontend trusts the chokepoint, auto-linkifies plain-text URLs via `linkifyjs`. Rich-media turns additionally carry a `segments` array (see Rich-Media Segments above). |
| **DMN** | Default Mode Network — Step 5 of the subconscious worker tick. Reflective pass that reads the user synthesis + recent user-channel episodes and saves findings via the memory tool. No chat-UI broadcast. |
| **Episode** | Narrative memory unit extracted from transcript windows. Has salience score and decaying retrieval weight. |
| **Data Graph** | Structured knowledge store with canonicalisation, typed edges, and per-kind decay. |
| **Salience** | Computed importance score [1–10] based on emotional arousal, valence, open loops, and novelty. |
| **Subconscious worker** | 5-minute tick that runs latent cognition (super-episode consolidation, decay, pattern match + skill association, user-summary synthesis, DMN, capability sync, geo pattern extraction) only when the user has been idle for ≥ 30 minutes. Capability sync calls `monitor()` on each connected capability — the scheduler only stores calendar event data. |
| **Behavioural pattern** | A `data_graph` row with `kind='behavioral_pattern'` written by the pattern-match or geo-pattern pass. Content JSON: `name, frequency, time_anchor, summary, confidence, last_seen_at, evidence_transcript_ids`. Confidence starts at 7 on first observation, increments by 7 on reinforce (capped at 10), and decays −0.005 per subconscious tick on untouched rows. Rows at 0 are soft-deleted (`active=0`). On reinforce, the SQL recall-ranking columns are also updated: `evidence_count` increments, `storage_strength` gets a diminishing boost (`0.05 / log2(evidence+1)`, capped at 1.0), `retrieval_weight` resets to 1.0, and `last_accessed_at` is set — mirroring `DataGraphService._reinforce_row()`. |
| **Named place** | A `data_graph` row with `kind='place'` written by `PlaceAbility` or the geo-pattern pass. Content JSON: `lat, lon, name, radius_m`. Stores user-labelled locations (home, work, gym). Used by `ScheduleAbility` to resolve destination coordinates for departure-time reminders. Persistent, no TTL, cosine-supersede on conflict, explicit deletion only. |

---

Testing: `docs/12-TESTING.md`. Development setup: `docs/01-QUICK-START.md`.
