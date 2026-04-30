# System Architecture

## What Chalie Is

Chalie is a persistent cognitive runtime — a single Python process that keeps thinking between conversations. It is not a request-response wrapper around an LLM. Every message flows through a multi-layer memory pipeline, background workers maintain and decay knowledge while the user is idle, and the system forms a continuously evolving model of the person it talks to. Intelligence accumulates over time; it does not reset per session.

The stack: Flask + flask-sock for HTTP and WebSocket, SQLite (WAL mode, sqlite-vec for vector search, FTS5 for keyword search), an in-process thread-safe MemoryStore (no Redis), and a pluggable LLM provider layer supporting Ollama, Anthropic, OpenAI, and Google Gemini. Everything runs in one process.

---

## How a Message Flows

A user message arrives over WebSocket. The handler spawns a daemon thread, constructs a `UserMessageProcessor` for this turn, and calls `send()`. Nothing else touches the message.

Inside `send()`:

1. **Memory seed** — recent episodes are retrieved and attached to the turn context.
2. **Deliberation gate** — a lightweight ONNX classifier reads the message and assigns a continuous deliberation score (0.0–1.0) on the transcript row. Higher scores nudge the prompt toward more careful reasoning; very high scores trigger a one-shot pre-reasoning pass before the tool loop begins.
3. **ACT loop** — the processor assembles a single user message containing the literal conversation history, world state, memory seed, and the current input, then calls the LLM. If the LLM invokes a tool, the result is appended to the trail and the loop continues. This repeats until the LLM returns a plain text response or hits the iteration cap.
4. **Atomic write** — one SQLite transaction commits the user turn, every tool call from the loop, and the assistant response. Nothing is written to the database mid-loop.
5. **Post-turn fan-out** — services that react to a completed turn (conversation phase update, save suggestion detection, DMN timer reset, metrics) run after the atomic write. The response is already on its way to the client before fan-out begins.

```
WebSocket frame
  └─ daemon thread
       └─ UserMessageProcessor.send()
            ├─ memory seed
            ├─ deliberation gate  (classify → optional exploration pass)
            ├─ ACT loop ──────────────────────────────────────────┐
            │    assemble prompt (history + world state + seed)   │
            │    → LLM call                                       │
            │    → tool calls → tool results → back to LLM  ─────┘
            ├─ atomic write (transcript + tool_calls, one tx)
            └─ post-turn fan-out → response → client
```

See `docs/13-MESSAGE-FLOW.md` for the full turn lifecycle.

---

## Message Processors

`MessageProcessor` is the abstract base for every LLM turn in the system. The architectural rules are simple:

- **One class per channel.** User messages, DMN thoughts, goal pursuit, scheduled prompts, compaction calls, and internal encoders each have their own subclass. There is no shared dispatcher or central router.
- **One instance per turn.** All turn state lives on the instance. No singletons, no shared instances.
- **Subclasses hardcode their channel and role.** A processor knows what it is. Context scoping flows from that identity.
- **Atomic store at the end.** `store()` commits everything in one transaction when the ACT loop finishes.
- **`handleTool()` is the single dispatch chokepoint.** Tool errors return structured strings to the LLM; they never surface to the user or crash the loop.
- **`postTurn()` is where channel-specific fan-out lives.** Shared plumbing goes in the base; subclass-specific services go in the subclass.

History reaches the LLM as a literal `## Previous Messages` text block inside the user message body. The provider always receives a single-element `messages[]` array — not a multi-turn array. This is an intentional design choice: it gives the system full control over what context the model sees on each turn.

Compaction also dispatches through the `MessageProcessor` hierarchy. `FullCompactionProcessor` handles full channel checkpoint summarisation; `TrailCompactionProcessor` handles mid-ACT tool-trail summarisation. Both subclasses hardcode `JOB='frontal-cortex-unified'`, `ALWAYS_AVAILABLE=[]`, `DISCOVERABLE=[]`, and `SKIP_TRANSCRIPT_WRITE=True`, and override the recursion guard so a compaction call never triggers a nested compaction. Their system-prompt bodies live as constants in `system_message_prompt.py`. `MetricsAccumulator.merge()` folds the sub-processor's token and tool counts into the parent turn's metrics so per-turn reporting reflects the full cost. Pure SQL helpers for the `compactions` table (`get_compaction`, `get_entries_since`) live in `compaction_persistence.py`.

Internal processors (episode encoders, the user summary synthesiser, and compaction processors) set a flag that suppresses transcript writes — they run the ACT loop without polluting the conversation record.

---

## Memory Hierarchy

Four layers, each optimised for a different timescale and purpose:

| Layer | What it stores | Decays? |
|-------|---------------|---------|
| **Transcript** | Append-only conversation record, channel-scoped | No (pruned after 90 days) |
| **Compaction** | LLM-generated summary of history beyond context limit | No (one per channel) |
| **Episodes** | Narrative units extracted from transcript windows | Yes — power-law retrieval weight decay |
| **Data Graph** | Structured knowledge (facts, preferences, moments) | Yes — per-kind decay policy |

**Transcript** is the raw record. `getPreviousMessages()` renders everything above the compaction watermark as a literal text block. When that block approaches the provider's context limit, compaction fires and summarises the older portion — the summary becomes the new floor.

**Episodes** are extracted automatically by a rolling trigger: when enough new transcript lines have accumulated for a channel, a background processor encodes that window into narrative snapshots with emotional valence, arousal, and salience scores. Similar episodes consolidate into super-episodes over time. Retrieval uses hybrid vector + FTS5 search, adaptive radius, and apex traversal (following consolidation links upward).

**Data Graph** is the knowledge layer. Writes for user-specific facts go through a canonicalisation engine: the key is compared against a set of high-level concepts, and a rule (temporal supersede / coexist additive / immutable block) is applied. This prevents duplicate or contradictory facts from accumulating. The database shape lives in `backend/schema.sql`.

**Query expansion.** Every data-graph write is enqueued to the `SearchExpanderService` — a single boot-time FIFO daemon. It generates paraphrased variants via doc2query, embeds each, and writes them to `expanded_semantic` + `expanded_semantic_vec` keyed back to the source rowid. Recall adds a KNN signal against the variant index so paraphrased questions hit the right facts even when the literal surface form does not match. The daemon is event-driven (not busy-loop) and self-heals on boot by rescanning rows with `search_queries IS NULL`.

---

## Background Reasoning

Chalie keeps thinking when you are not typing. Background workers run as daemon threads in the same process:

- **Subconscious worker** (v0.5.0 §5) — a single daemon thread (`subconscious-worker`) that owns latent cognition. It ticks every 5 minutes (`SUBCONSCIOUS_TICK_SEC`), and on each tick checks two gates: the **user-active** gate (`WorldState.snapshot().last_user_message_at` within the last 30 min) and the **already-fired** gate (`subconscious_last_fired_at > last_user_message_at`). The already-fired gate also fires after a process restart when `subconscious_last_fired_at` was hydrated from durable storage but no user message has arrived yet — the previous lifetime already covered the open idle window, so the worker waits for fresh signal before running again. When both pass, it runs four steps in order, each isolated in its own try/except so one failure cannot block the rest: (1) consolidate apex episodes into super-episodes per channel via `SuperEpisodeEncoderProcessor`, (2) `DecayEngineService.run_once()` (the engine instance is built once and cached on the worker so the per-tick config read is amortised), (3) `PatternMatchProcessor` (see below), (4) `UserSummaryProcessor.send()`. Steps 3 + 4 are LLM-heavy and are skipped when the background-LLM queue is saturated (`bg_llm:queue` depth ≥ `SUBCONSCIOUS_BG_QUEUE_THRESHOLD`, default 20, leaving headroom under `MAX_QUEUE_DEPTH=25`). Re-entrancy is guarded by a non-blocking lock; concurrent ticks return without doing work. The next tick is anchored to `monotonic()` after the current tick returns, not before — long ticks therefore extend the cycle rather than starve the next gate. State (`subconscious_last_fired_at`) is mirrored to MemoryStore (`subconscious:last_fired_at`) and to `data_graph` (`kind='system'`, `key='subconscious_last_fired_at'`) so the gate state survives restarts.
- **Decay engine** — applies power-law decay to episode retrieval weights and data-graph entries, and purges old transcript entries and tool-call rows. The engine itself (`DecayEngineService`) has no daemon thread of its own; it exposes `run_once()` and is called by the subconscious worker as Step 2 of its tick.
- **Pattern matcher** — `PatternMatchProcessor` runs as Step 3 of the subconscious worker tick when ≥50 new transcripts have accumulated since the last cursor (`data_graph` kind=`system`, key=`pattern_match_cursor`). One LLM forward pass over the new transcript window with two processor-scoped tools: `save_pattern` (UPSERT a `behavioral_pattern` row with confidence math: new=7, reinforced=`min(10, prev+7)`, capped at 10; budget 20 calls) and `save_graph` (routes through `DataGraphService.store()` for `user_specific` / `misc` / `moment` / `document` kinds; budget 50 calls). The model emits all calls in parallel; `MAX_ITERATIONS=30` bounds runaway loops. After the pass, an in-place SQL decay sweep subtracts 0.005 from every untouched active pattern's confidence; rows that hit 0.0 flip to `active=0` (soft delete). `UserSummaryProcessor` reads active behavioral_pattern rows directly when assembling its synopsis prompt — no tool surface for user turns.
- **DMN (Default Mode Network)** — after a period of idle time, Chalie initiates a proactive thought using recent or high-salience episodes as context. Uses its own `MessageProcessor` subclass; exits silently when nothing warrants a response.
- **Subagent** — focused background tasks spawned by the `subagent` innate ability. Each runs a `SubagentProcessor` instance with a per-type tool surface and a per-instance wall-clock deadline. When complete the result is injected into the parent's ACT trail (Case A: parent mid-turn via `_pending_steers`) or spawns a synthetic `SubagentReturnProcessor` turn on the user channel (Case B: parent idle). Three types: `web_surfer` (60 min, web search + browse), `summariser` (10 min, read + compress), `general_purpose` (30 min, arbitrary parallel work). The `wait=true` flag caps any type at 300 s and blocks the parent iteration synchronously. See "Subagent" section below.
- **Scheduled prompts** — the scheduler fires due reminders and timed tasks via their own processor subclass.
- **Supporting workers** — world awareness (weather, news), moment context enrichment, document purge, folder watcher, interface health monitor, self-model health signals, and the `SearchExpanderService` (single FIFO consumer that generates + embeds query variants for every new data-graph row). User-summary synthesis and super-episode consolidation no longer run their own daemons — both are driven by the subconscious worker tick.
- **EmbeddingService** — a module-level singleton that serialises all ONNX inference through a single daemon worker thread via a FIFO queue (`_embedding_queue`). All callers — `generate_embedding(text)`, `generate_embedding_np(text)`, `generate_embeddings_batch(texts)` — check MemoryStore first; cache hits bypass the queue entirely. The worker is started lazily on the first job submission (not at import time) so tests that never call the service never spawn a real ONNX thread. The single-worker model eliminates concurrent `session.run()` calls, which each allocate 500 MB+ of working memory and caused OOM under bulk document ingestion. Session construction routes through `onnx_session.build_session()` (see below).
- **onnx_session.py** — single chokepoint for all ONNX session construction in the process. `choose_providers(model_path)` returns the ordered provider list, applying the Metal 16384 2D-texture ceiling check for CoreML (any initializer dimension exceeding 16384 triggers automatic removal of `CoreMLExecutionProvider`). `build_session(path, opts, providers, log_prefix)` constructs the session and retries with CPU-only on construction failure. `EmbeddingService`, `VoiceService` (`voice.py`), and `Doc2QueryService` all route through this module — no service constructs `ort.InferenceSession` directly.

No worker shares its processor instance with another. Each channel is fully isolated.

### Subagent

The `subagent` ability (`backend/abilities/subagent.py`) is the public surface. The `SubagentProcessor` (`backend/services/subagent_processor.py`) is the execution engine.

**Three types, registered in `SUBAGENT_TYPES`:**

| Type | Default budget | Tool surface | Use case |
|------|----------------|--------------|----------|
| `web_surfer` | 60 min | `read`, `search`, `browser`, `news`, `memory`, `find_tools` | Multi-source web research, live page lookups |
| `summariser` | 10 min | `read`, `search`, `document`, `find_tools` | Compress long content before pulling into context |
| `general_purpose` | 30 min | `memory`, `find_tools` | Parallel arbitrary work; use `find_tools` for more capabilities |

**INPUT_SCHEMA:** `prompt` (required string), `type` (enum, default `general_purpose`), `wait` (boolean, default `false`).

**Async return flow (default `wait=false`):**

1. `SubagentAbility.execute()` captures `current_processor()` (a `contextvars.ContextVar` bound by `send()`) as `parent_ref` before spawning the daemon thread — so the reference is stable even if the parent turn finishes first.
2. The daemon runs `SubagentProcessor(...).send()` and wraps the result in a canonical `[subagent.complete(...)]` envelope via `_build_envelope()`.
3. `_deliver_envelope(envelope, parent_ref)` routes:
   - **Case A** (`isinstance(parent_ref, UserMessageProcessor)` is True): appends the envelope to `parent_ref._pending_steers`. `getUserPrompt()` drains `_pending_steers` into the ACT trail at the top of the next iteration.
   - **Case B** (parent idle or non-UMP): spawns a `subagent-return` daemon that runs `SubagentReturnProcessor(envelope).send()`.

**`SubagentReturnProcessor`** is a thin `UserMessageProcessor` subclass (`ROLE='subagent_return'`) defined in `backend/services/user_message_processor.py`. It inherits all UMP behaviour — `ALWAYS_AVAILABLE`, `DISCOVERABLE`, system prompt, `postTurn` fan-out. `ROLE='subagent_return'` causes the input row to be excluded from user-visible history rebuilds.

**`_pending_steers`** is a `list[str]` field on every `MessageProcessor` instance (base class `__init__`). Only `UserMessageProcessor.getUserPrompt()` drains it. Background processors that do not call `getUserPrompt()` never drain pending steers — subagents targeting those channels always fall through to Case B.

**Sync mode (`wait=true`):** `_run_sync()` constructs `SubagentProcessor` inline and calls `.send()` directly from the parent ACT loop iteration. Budget is capped at 300 s regardless of type default. The parent iteration blocks until the subagent returns.

**Wall-clock guard:** `SubagentProcessor` sets `self._deadline = time.time() + timeout_seconds` in `__init__`. The base `send()` loop checks `self._deadline` after each iteration and breaks if exceeded. `ITERATION_TIMEOUT` (1800 s, base class constant) provides an additional per-iteration safety wall that applies to all processors including subagents.

---

## ONNX Runtime Selection

### Install-time wheel dispatch

`installer/install.sh` detects the host GPU before installing Python dependencies and swaps the `onnxruntime` wheel accordingly:

| Detected hardware | Wheel installed |
|-------------------|-----------------|
| NVIDIA GPU (`nvidia-smi` present) | `onnxruntime-gpu` |
| AMD GPU (`/dev/kfd` + `amdgpu` kernel module) | `onnxruntime-rocm` (from AMD's manylinux index) |
| Everything else | `onnxruntime` (CPU) |

The CPU wheel is always installed first as a baseline. The GPU wheel replacement only happens after a `pip install --dry-run` confirms the download would succeed — so installs on machines without network access to the GPU index remain on CPU rather than failing. ORT version is pinned at `1.20.1` as a single source of truth in the installer. `backend/requirements.txt` does not pin `onnxruntime` directly; it carries `rapidocr_onnxruntime` which transitively pulls the CPU wheel for development workflows that bypass the installer.

For air-gapped AMD installs, set `ROCM_PIP_INDEX` to a local mirror before running the installer.

### Runtime provider selection

All session construction goes through `backend/services/onnx_session.py`:

- `choose_providers(model_path)` — returns the ordered execution provider list. On macOS, any ONNX model whose initializer tensors include a dimension exceeding **16384** (the Metal 2D-texture ceiling) has `CoreMLExecutionProvider` stripped automatically. `gte-modernbert-base` trips this limit (vocab embedding is `{50368, 768}`). The check runs at session construction time and emits a `[<prefix>] Dropped CoreMLExecutionProvider: model has dim > 16384` log line when it fires.
- `build_session(path, opts, providers, log_prefix)` — constructs the `InferenceSession`. If construction raises with the chosen providers, it retries with `["CPUExecutionProvider"]` before propagating the error. This makes individual model failures non-fatal for the rest of the process.

`EmbeddingService`, `VoiceService`, and `Doc2QueryService` all call `build_session` — no service constructs `ort.InferenceSession` directly.

### Asset layout

Two distinct on-disk directories separate runtime-downloaded weights from pre-shipped classifier files, with one extra location for pre-shipped sqlite-vec/FTS5 search indexes:

| Path | Tracked in git | Contents |
|------|----------------|----------|
| `backend/data/models/` | No (gitignored) | Encoder ONNX (`gte-modernbert-base`), voice (`kokoro`), `doc2query-small`. Downloaded on first boot or installer step. |
| `backend/data/pre-trained/` | Yes | Per-task classifier meta + `.npz` MLP heads (`deliberation_score/`, `mode_detector/`) plus drift sidecars for pre-shipped search indexes (`abilities_sha.json`). Cloning the repo is enough to classify on first turn — no GitHub release fetch. |
| `backend/abilities/assets/` and similar `*/assets/` directories | Yes (binary diff suppressed via `.gitattributes`) | Pre-shipped sqlite-vec/FTS5 search indexes (`abilities.sqlite`, `concept_lut.sqlite`, `search_tool_providers.sqlite`). Built by `python -m utils.build_ability_db` (and equivalents); a CI `--check` step compares the per-row sha to the sidecar in `data/pre-trained/` and fails the build on drift. `abilities.sqlite` indexes **every** registered ability, including `save_pattern` and `save_graph` (1 SUMMARY row + 6–8 EXAMPLE rows each, all embedded at 768 dim). Discovery scoping is enforced at query time via each processor's `DISCOVERABLE` list, not by excluding rows from the index. Drift sidecar: `backend/data/pre-trained/abilities_sha.json`. |

`OnnxInferenceService.__init__(models_dir, pretrained_dir)` takes both. The shared encoder ONNX is resolved against `models_dir`; per-task classifier directories resolve against `pretrained_dir`. CLI flags `--models-dir` and `--pretrained-dir` (or env `MODELS_DIR` / `PRETRAINED_DIR`) override the defaults.

The pre-shipped `<task>-classifier_meta.json` is the authoritative calibration source for each head — alpha, bucket thresholds, sha256 pin. Missing or corrupt meta files raise at boot rather than falling back to baked-in defaults; per-turn callers (e.g. `UserMessageProcessor`) catch the construction error and degrade to a safe default (`thinking_level='low'` for the deliberation gate). If `_preload_models` itself raises before the singleton finishes registering tasks, the outer except annotates `OnnxInferenceService._failed_registrations = [("preload", ...)]` and flips `_ready = True` so the `/health` endpoint can explain the degraded state instead of reporting a generic not-ready.

---

## Tools and Skills

Two loading tiers stack on every user turn and are merged first-seen, so the unconditional tier can never be shadowed by a dynamic entry of the same name:

**Tool scope lives on each `MessageProcessor` subclass**, not on the `Ability` ABC. Every processor declares two `ClassVar` lists:

- `ALWAYS_AVAILABLE` — ability names pre-injected as native tools on every ACT iteration via `getTools()`.
- `DISCOVERABLE` — ability names that `find_tools` may surface for this processor at runtime. The SQL query inside `find_tools` filters candidates to `WHERE name IN (DISCOVERABLE)`, so a processor can never discover anything outside its own list.

`UserMessageProcessor` sets `ALWAYS_AVAILABLE` to the eight core cognitive abilities (`document`, `find_tools`, `list`, `memory`, `read`, `review_tool_calls`, `schedule`, `subagent`) and `DISCOVERABLE` to the six first-party external abilities (`browser`, `code_eval`, `news`, `programming_docs_search`, `search`, `weather`). Other processors narrow both lists to match their scope — `PatternMatchProcessor`, for example, sets `ALWAYS_AVAILABLE = ["save_pattern", "save_graph"]` and `DISCOVERABLE = []`. `SubagentProcessor` sets `ALWAYS_AVAILABLE` per-instance from `SUBAGENT_TYPES[agent_type]['native_tools']` and `DISCOVERABLE` to the full external set minus a blocklist (`subagent`, `save_graph`, `detect_pattern`).

**Discoverable abilities** are never pre-injected. The `find_tools` ability performs semantic search against the abilities index at runtime. When the LLM invokes `find_tools`, the matching abilities become available for the remainder of that ACT loop. All external (first-party + interface) abilities are reachable exclusively through this path — pre-injecting them would bloat context, create staleness bugs, and break tool-agnostic routing.

`find_tools` performs **RRF (Reciprocal Rank Fusion) discovery** exclusively against `backend/abilities/assets/abilities.sqlite`. `abilities.sqlite` indexes **every** registered ability — including `save_pattern` and `save_graph`. Discovery scoping is enforced at query time via the calling processor's `DISCOVERABLE` list, not by selectively excluding rows from the index. Each query runs two independent retrievals — (a) sqlite-vec k-NN (`KNN_DEPTH=30`) over per-entry embeddings and (b) FTS5 BM25 over the same entries — both filtered to `WHERE name IN (allow)`. Results are grouped per ability (best distance / best BM25 score per ability), ranked independently, then fused via `score(a) = Σ 1 / (15 + rank_i(a))` across the two lists. The top-`k` abilities by fused score are returned. The non-standard `RRF_K=15` (vs. the standard 60) is deliberate: at ~17 candidates, k=60 compresses scores into too narrow a band for crisp separation. If `EmbeddingService` fails, a keyword-only fallback queries `ability_search_fts MATCH ? AND name IN (allow)` in `abilities.sqlite` directly. If `abilities.sqlite` is missing, both paths return `[]` with a `[FIND_TOOLS]` WARNING.

`ModeGateService` runs once per user turn (via `UserMessageProcessor._get_mode_state()`) but **does not gate tool availability**. A small ONNX multi-label classifier (`mode_detector`, eight heads: `research`, `coding`, `brainstorm`, `analyze`, `plan`, `write`, `math`, `converse`) emits per-mode probabilities. Per-mode EMA state snaps up on a fire and decays by 0.75 on a miss, so a topic stays "warm" for roughly four turns before falling below the activation threshold. State persists in MemoryStore under `mode_gate:state` and clears on `/privacy/delete-all`. The mode set powers prompt-steering directives (long-summary swap on `converse`, brainstorm/research/analyze suffixes) and is reserved for future mode-driven features. A single `[MODE-GATE]` log line per turn records probabilities, state transition, and the active mode set.

Tool results flow through a single render-and-record path (`ToolRenderAndRecordService`) that formats the output and writes it to the `tool_calls` table. Tool infrastructure has no knowledge of specific tools; tools have no knowledge of infrastructure.

Every ability result uses the canonical tag block format from `backend/services/innate_skills/_tag.py` (the only file remaining under `services/innate_skills/` after Phase 4 cutover — kept because it is purely a formatter shared by every ability):

```
[<ability_name>(k1=v1, k2=v2)]
<body>
[end:<ability_name>]
```

This is the single source of truth — no ability constructs its own format string. See `docs/09-TOOLS.md` and `docs/15-INTERFACES.md`.

**`pre_act()` hook.** `MessageProcessor.pre_act()` is called from `send()` after the input transcript row is written (so `self._uid` is populated) but before the ACT loop starts. The base implementation is a no-op. `UserMessageProcessor` overrides it to run the memory seed: it calls `handle_memory()` directly (same path as an LLM-invoked recall), stores the result via `ToolRenderAndRecordService(ephemeral=False)` — making the seed a durable, auditable tool call row — and places the canonical tag block in `self._memory_seed` for `getUserPrompt()` to inject verbatim.

---

## Ability Framework

`backend/abilities/` is the **sole dispatch path** for every cognitive operation an LLM turn can invoke. Legacy `services/innate_skills/*_skill.py` modules and `tools/<name>.py` modules have been ripped — every dispatchable capability is now an `Ability` subclass. The only files kept under their old paths are companion modules that an `Ability` imports from (`tools/browser/{security,pool,extraction,interaction,credentials}.py`, `tools/search/{router,fetcher,transformers}.py`) and the shared formatter `services/innate_skills/_tag.py`.

### `Ability` ABC (`backend/abilities/_base.py`)

Every concrete ability subclasses `Ability` and declares:

| Class attribute | Type | Notes |
|-----------------|------|-------|
| `NAME` | `str` | Stable identifier; matches the channel-level tool name. |
| `SUMMARY` | `str` | One sentence used as the SUMMARY row in `abilities.sqlite`. |
| `EXAMPLES` | `list[str]` | 6–8 natural-language phrases for EXAMPLE rows (enforced by `__init_subclass__`). |
| `INPUT_SCHEMA` | `dict` | JSON Schema for `execute()` params. |
| `TIMEOUT` | `int` | Per-call timeout in seconds (default 10). |

Tool scope (always-available vs discoverable) is **not** declared on the `Ability` ABC. It is owned by each `MessageProcessor` subclass via its `ALWAYS_AVAILABLE` and `DISCOVERABLE` `ClassVar` lists (see Tools and Skills section).

The `execute(channel, params, telemetry)` method is the sole dispatch surface. `pre_dispatch` and `post_dispatch` hooks exist for subclass-level lifecycle work.

### Concrete abilities (14 dispatchable)

`abilities/{browser, code_eval, document, find_tools, list, memory, news, programming_docs_search, read, review_tool_calls, schedule, search, subagent, weather}.py`. `abilities.sqlite` indexes all 14 (134 embedded entries total). `abilities_sha.json` mirrors the per-row sha and is checked in CI.

Per-ability implementation notes:

- `browser` — `playwright.sync_api` imported unconditionally at module top. Companion modules under `tools/browser/`.
- `code_eval` — `_RESTRICTED_GLOBALS` built once as `ClassVar[dict]`; a fresh `dict()` copy is taken per call so state never leaks between executions.
- `document` — `create_document_artifacts` exposed as both a module-level function and a `classmethod` so `api/documents.py` and `services/folder_watcher_service.py` import the same path.
- `find_tools` — RRF discovery against `abilities.sqlite` (see Tools and Skills section).
- `subagent` — spawns a daemon thread that runs `SubagentProcessor`. `SubagentProcessor` and `SubagentReturnProcessor` are lazy-imported inside the `_run_async()` closure to avoid import cycles. On completion, `_deliver_envelope()` routes the result: Case A (parent turn active, detected via `current_processor()` contextvar) appends to `parent._pending_steers`; Case B (parent idle) calls `_spawn_return_processor()` which starts a `subagent-return` daemon running `SubagentReturnProcessor(envelope).send()`. `SubagentReturnProcessor` is a thin `UserMessageProcessor` subclass with `ROLE='subagent_return'`; it inherits all UMP behaviour and the `subagent_return` input row is excluded from user-visible history rebuilds.
- `list` — `_DEFAULT_LIST_NAME` as `ClassVar[str]`; handler helpers at module level.
- `memory` — 8 radius constants promoted to `ClassVar` (`RECALL_RADIUS_BASELINE`, `SEED_RADIUS_BASELINE`, etc.) so the meta-harness can patch them by name. Module-level `recall_episodes()` function preserved for importability by the UMP pre-act seed path.
- `news` — `_service` classvar lazily initialised via `_get_service()` classmethod.
- `programming_docs_search` — all 23 `_Source` subclasses and `_ALL_SOURCES`/`_ALIAS_MAP` at module level.
- `read` — `requests` at module top; `_BLOCKED_NETS`, `_BROWSER_HEADERS`, `_BLOCKED_PATH_PREFIXES`, `_URL_FETCH_TIMEOUT` as `ClassVar`.
- `review_tool_calls` — returns `dict` directly.
- `schedule` — atomic dedup `INSERT...WHERE NOT EXISTS` preserved verbatim; `_PAST_DUE_GRACE_SECONDS` as `ClassVar[int] = 120`.
- `search` — `_DB` path resolves to `tools/search/assets/search_tool_providers.sqlite`; companion router/fetcher/transformers stay under `tools/search/`.
- `weather` — Open-Meteo (primary, coordinate-based) + wttr.in (city-name fallback). `_cache` and `_CACHE_TTL=600` as `ClassVar`s so the 10-minute cache is shared.

### Pattern-match helpers — plain classes, not Ability subclasses

`abilities/save_pattern.py` (`SavePattern`) and `abilities/save_graph.py` (`SaveGraph`) are processor-internal helpers used only by `PatternMatchProcessor`. They live at the top level of `abilities/` and are indexed by `abilities.sqlite` — but they are intentionally **plain Python classes** (no `Ability` inheritance) so they are never instantiated by the registry walk and never surface through `find_tools`. Scoping is enforced by `PatternMatchProcessor`'s `ALWAYS_AVAILABLE = ["save_pattern", "save_graph"]` / `DISCOVERABLE = []` — no other processor lists them, so no other channel can reach them. Each helper exposes a `TOOL_SCHEMA` consumed directly by `PatternMatchProcessor.getTools()` (via the `ALWAYS_AVAILABLE` path in `MessageProcessor.getTools()`) and an `execute(args, processor)` method that reads/writes processor state.

### Registry (`backend/abilities/_registry.py`)

Singleton with an `RLock`. Exposes only `get(name)` and `all()`. Lazily walks `backend/abilities/` on first access via shallow `glob("*.py")` (skipping files starting with `_`), then traverses `Ability.__subclasses__()` filtering out abstract classes. The registry is the single source of truth for which abilities are active in the process and is the only path an `Ability` instance is created — no module elsewhere instantiates an ability directly. Tool scope (which processor sees which ability) is not the registry's concern; that belongs to each `MessageProcessor` subclass.

---

## Chat File Attachments

When a WebSocket `chat` frame carries `image_ids` — or when a recent upload/chat-image document exists for the current user — `_resolve_file_tags()` in `backend/api/websocket.py` runs before the `UserMessageProcessor` is constructed. It blocks the turn until every referenced document reaches a terminal state or a shared 10-second deadline expires, then attaches structured tags to `metadata['file_tags']`:

- `[image id=<doc> ocr="..."]` — ready image, OCR text truncated
- `[image id=<doc> status=failed|timeout|not_found]` — image that never reached ready
- `[document id=<doc> summary="..."]` — ready PDF/text document
- `[document id=<doc> status=failed|timeout|not_found]` — document equivalent

`UserMessageProcessor.getUserPrompt()` appends these tags to the turn line so the LLM sees file context inline with the user message. The no-silent-drop invariant applies: every failure mode emits a tag — the model learns a file was attached and what happened to it, never an empty prompt.

Images land via `POST /chat/image` (source_type `chat_image`, triggers OCR + scene analysis). Documents land via `POST /documents/upload` (source_type `upload`, triggers text extraction for PDFs via `document_skill.create_document_artifacts`). The 10-second deadline is shared across every file resolved in the turn — worst case one wait, not N.

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

**Backend module:** `backend/services/markup.py` — `sanitize()` (nh3 chokepoint), `extract_plaintext()` (TTS — strips tags + drops `<actions>` subtree), `wrap_text_xml`, `actions_to_xml`, `is_xml_content`, `escape_text`, `escape_attr`. Zero hand-rolled tokenisation; nh3 owns all parsing.
**Frontend module:** `frontend/interface/markup_renderer.js` (innerHTML + linkify + programmatic wiring) + `markup_extract.js` (DOM walk for TTS plaintext). `frontend/interface/vendor/linkify.es.mjs` (linkifyjs 4.3.2, vendored ESM, ~20 KB).
**No boot migration:** legacy markdown→HTML migration was retired with `markdown_xml_migration.py` after the nh3 cutover. Existing transcripts retain their stored content — sanitisation runs on every render.

**Asset versioning:** every static asset reference in served HTML has the version string injected into its filename at response time (e.g. `app.js` becomes `app-0.3.3.js`). Static routes strip the version suffix before the disk lookup, so nothing is renamed on disk. Versioned filenames are used instead of query strings because some service workers and proxies ignore query strings when keying caches. HTML responses themselves are never cached.

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
| **MessageProcessor** | Abstract base for all LLM turns. One instance per turn, one subclass per channel. |
| **Channel** | Stable string scoping transcript and compaction data (e.g. `user`, `dmn`, `subagent`). |
| **HTML Markup Format** | Content format: single `content` string of HTML, backend → frontend. Backend `services.markup.sanitize()` (nh3) is the chokepoint. LLM emits 8 formatting tags (no `<a>`); backend programmatically emits `<img>`, `<actions>`, `<action>`. Frontend trusts the chokepoint, auto-linkifies plain-text URLs via `linkifyjs`. |
| **DMN** | Default Mode Network — timer-based proactive intelligence that fires during idle periods. |
| **Episode** | Narrative memory unit extracted from transcript windows. Has salience score and decaying retrieval weight. |
| **Data Graph** | Structured knowledge store with canonicalisation, typed edges, and per-kind decay. |
| **Salience** | Computed importance score [1–10] based on emotional arousal, valence, open loops, and novelty. |
| **Subconscious worker** | 5-minute tick that runs latent cognition (super-episode consolidation, decay, pattern match, user-summary synthesis) only when the user has been idle for ≥ 30 minutes. |
| **Behavioural pattern** | A `data_graph` row with `kind='behavioral_pattern'` written by `PatternMatchProcessor`. Content JSON: `name, frequency, time_anchor, summary, confidence, last_seen_at, evidence_transcript_ids`. Confidence starts at 7 on first observation, increments by 7 on reinforce (capped at 10), and decays −0.005 per subconscious tick on untouched rows. Rows at 0 are soft-deleted (`active=0`). |

---

Testing: `docs/12-TESTING.md`. Development setup: `docs/01-QUICK-START.md`. Interface pairing: `docs/15-INTERFACES.md`.
