# System Architecture

## Overview

Chalie is a **persistent cognitive agent** — a continuously running cognitive runtime, not a request-response service. Intelligence emerges from experience: every interaction flows through a memory pipeline that compresses, abstracts, and decays information over time. The system is not a chatbot or assistant wrapper; it is a cognitive runtime that forms memories, runs background processes, exercises judgment, and diverges into a unique identity shaped by its interaction history.

## Core Architecture

### System Type
- **Synthetic cognitive brain** using LLMs to replicate human brain functions
- **Tech Stack**: Python backend, SQLite (WAL mode + sqlite-vec + FTS5), MemoryStore (in-memory, thread-safe), Ollama (configurable LLMs), Vanilla JavaScript frontend (Radiant design system)
- **Core Pattern**: Single-process architecture with daemon threads; all LLM calls run through `MessageProcessor` subclasses

### Communication Pattern
1. Client connects to `/ws` (WebSocket via flask-sock)
2. Client sends `{"type": "message", "text": "..."}` JSON frame
3. Backend spawns daemon thread → `UserMessageProcessor(raw_input, metadata, on_narration).send(request_id)` → `MetricsAccumulator.snapshot()` → response + metrics published via pub/sub → WebSocket frames (`status` → `message` → `done`). The `message` frame carries a `metrics` object: `{tokens_total, tools: {name: count}, response_time_s}`.
4. Drift thoughts, cards, and proactive notifications also arrive over the same `/ws` connection
5. Authentication: Session cookie-based (`@require_session` decorator)

## Code Organization

```
backend/
├── services/          # Business logic (memory, orchestration, routing, embeddings)
├── workers/           # Async workers (digest, memory chunking, consolidation)
├── listeners/         # Input handlers (direct REST API)
├── api/               # REST API blueprints (conversation, memory, proactive, privacy, system)
├── configs/           # Configuration files (connections.json, agent configs, generated/)
├── prompts/           # LLM prompt templates (mode-specific)
├── tools/             # First-party tool modules
├── tests/             # Test suite
└── run.py             # Single-process entry point
```

Frontend applications located separately:
```
frontend/
├── interface/         # Main chat UI (ES6 modules, Radiant design system)
│   ├── app.js         # Thin orchestrator — boot, wiring, small glue (~680 lines)
│   ├── utils.js       # Shared utilities (escHtml, lsGet/lsSet, toast, relativeTime)
│   ├── auth.js        # Authentication & login dialog
│   ├── chat.js        # Message sending orchestration & conversation history
│   ├── image_attach.js    # Image upload, preview strip, analysis tracking
│   ├── document_upload.js # Document upload dialog, processing, synthesis
│   ├── task_strip.js      # Persistent task strip display & polling
│   ├── apps_panel.js      # Interface daemon panel, scope approval, app overlay
│   ├── event_router.js    # WebSocket drift/push event dispatcher
│   ├── notifications.js   # Audio chime, system notifications, push subscription
│   ├── update_system.js   # Update banner & dialog
│   ├── ambient_canvas.js  # Animated background gradient orbs
│   ├── api.js         # REST client
│   ├── ws.js          # WebSocket client
│   ├── renderer.js    # Conversation spine DOM renderer
│   ├── presence.js    # Presence dot state machine
│   ├── voice.js       # Voice I/O (STT + TTS)
│   ├── heartbeat.js   # Client context telemetry
│   ├── ambient.js     # Behavioral sensor (passive)
│   ├── moment_search.js   # Recall search overlay
│   ├── blocks.js      # Universal block-format renderer (JSON block arrays → DOM)
│   ├── activity_panel.js  # Activity/tool-call trace panel
│   └── sw.js          # Service worker (caching, push, share target)
├── brain/             # Admin/cognitive dashboard
├── on-boarding/       # Account setup wizard
├── login/             # Dedicated login form (Keychain-friendly)
└── shared/            # Cross-SPA assets: theme.css, auth-gate.js
```

**Module communication**: Constructor injection for shared services, callback registration for cross-module events, custom DOM events (`chalie:action`, `chalie:speak`, `chalie:pin-moment`) for loose coupling. Modules never reference each other directly — `app.js` wires all connections.

**Asset versioning**: Flask injects a `<script type="importmap">` into `index.html` at serve time, mapping all module imports to `?v=VERSION` URLs. The `VERSION` file is the single source of truth. Service worker uses network-first for JS/CSS (localhost = 0ms latency) with cache fallback for offline/PWA.

**Shared auth gate**: `frontend/shared/auth-gate.js` is the single choke-point for redirect logic. Each SPA sets `window.__chaliePage` and loads the gate; app code awaits `window.chalieGateReady` before bootstrapping. See `docs/03-WEB-INTERFACE.md` for the rule table.

**IMPORTANT**: UI code must exist under `/interface/`, `/brain/`, `/on-boarding/`, `/login/`, or `/shared/` only.

## Key Services

### Core Services (`backend/services/`)

#### Routing & Decision Making
- Mode routing has been removed. All message processing flows through `MessageProcessor` subclasses directly — each channel is its own orchestrator.

#### Response Generation
- Voice modulation is handled inline by `UserMessageProcessor.getSystemPrompt()` — no separate service.

#### Message Processing
- **`message_processor.py`** — Abstract base class for all LLM turns. One instance per turn — no singletons, no `.instance()`, no `.process()`. Constructor: `__init__(raw_input, metadata=None)`. Entry method: `send(request_id=None) -> str`. Per-turn state lives on the instance: `_act_trail`, `_discovered_tools`, `_memory_seed`, `_uid`, `_dispatcher`, `_metrics` (`MetricsAccumulator`). `self._dispatcher` (ActDispatcherService) is created once per turn in `send()` — all tools (innate + external) dispatch through the same handler path. `_register_discovered_tools()` registers find_tools results as handlers mid-turn. `set_turn_start(ts: float)` lets the websocket caller backdate the accumulator start time to before the background thread was spawned. Provides `send()` (full ACT loop), `store()` (atomic persistence), `handleTool()` (single dispatch chokepoint, never re-raises; calls `_metrics.record_tool()` for every tool invocation), `getSystemPrompt()`, `getTools()`, `getPreviousMessages()`. Tool output is rendered and recorded by `ToolRenderAndRecordService`. Subclasses implement `getUserDefinition()` and `getUserPrompt()`. Subclasses override `postTurn()` for per-channel service fan-out. History is delivered as a literal `## Previous Messages` text block inside `getUserPrompt()` — the provider sees a single-element `messages[]`, not a multi-turn array. Two-stage mid-ACT compaction fires at 80% of the provider's context window (Stage 1: tool-trail compaction in place; Stage 2: full checkpoint + ACT loop restart). Every `Providers.send_messages()` call (main ACT loop, thinking exploration, Stage 1 compaction, Stage 2 compaction) feeds its `LLMResponse` into `_metrics.accumulate()` so all token costs for the turn are captured. ContextVar-based `current_processor()` / `bind_current_processor()` allow innate skills to reach the running processor without it being passed as a parameter. **`SKIP_TRANSCRIPT_WRITE: bool = False`** — class-level flag that gates both the input-row write at the top of `send()` and the assistant-row write in `store()`; when True, `_uid` is never set and no transcript rows are produced. Used by internal processors (e.g. `EpisodeEncoderProcessor`, `SuperEpisodeEncoderProcessor`) that must invoke the ACT loop without polluting the transcript.
- **`metrics_accumulator.py`** — `MetricsAccumulator` dataclass. Per-turn token and tool accounting. Accumulates `tokens_input`, `tokens_output`, `tokens_thinking`, `tokens_cache_read`, `tokens_cache_create` across all LLM calls in the turn via `accumulate(llm_response)`. Records tool invocation counts via `record_tool(name)`. `snapshot() -> dict` emits `{tokens_total, tools, response_time_s}` (and `tokens_total_complete: false` when any provider call omitted usage counts). Instantiated in `MessageProcessor.__init__` so exploration-pass and compaction tokens are counted even though they run before the ACT loop `loop_start`.
- **`system_message_prompt.py`** — `SystemMessagePrompt` abstract base + six concrete subclasses: `UnifiedSystemMessagePrompt` (user channel), `DMNSystemMessagePrompt` (DMN channel), `GoalPursuitSystemMessagePrompt` (goal-pursuit channel), `ScheduledSystemMessagePrompt` (scheduled channel), `EpisodeEncoderSystemPrompt` (episode encoder), `SuperEpisodeEncoderSystemPrompt` (super-episode encoder). Every prompt body lives as a Python constant on the subclass via the shared `_SYSTEM_PROMPT` class attribute — no file reads, no fallbacks, no `_FileBackedSystemMessagePrompt`. `_SYSTEM_PROMPT` is declared `@property @abstractmethod` on the base, so a subclass that forgets to override it raises `TypeError` at construction time (Python's ABCMeta machinery). A plain class attribute in the subclass satisfies the contract. `getPrompt()` lives on the base and returns `self._SYSTEM_PROMPT` verbatim. One instance per turn, per `MessageProcessor.getSystemPrompt()` call. The `getUserDefinition()` line is prepended by the base `getSystemPrompt()` method — subclasses only build the body. `UserMessageProcessor.getSystemPrompt()` weaves `{{voice_modulation}}` and `{{adaptive_directives}}` into the Unified template on every turn.
- **`user_message_processor.py`** — `UserMessageProcessor(MessageProcessor)` subclass for the user channel. `CHANNEL='user'`, `ROLE='user'`, `NATIVE_TOOLS=ALL_SKILL_NAMES`, `MAX_ITERATIONS=30`, `MAX_TIMEOUT=900s`, `THINKING_TIMEOUT=600s`. Constructor adds `on_narration: Callable[[str, int], None] | None`. `_run_memory_seed()` auto-seeds episodes once per turn (`caller='seed'`, durable `ephemeral=0` DTO). `_run_thinking_gate()` (base-level, user channel only) classifies deliberation depth after memory seed — `loop_start` is anchored after the gate completes so `MAX_TIMEOUT=900s` covers only ACT loop iterations. When `level='high'`, fires a one-shot exploration pass (same-job LLM call, `tools=[]`, capped at `THINKING_TIMEOUT=600s` via `ThreadPoolExecutor`) and stores the result as a durable `tool_calls` row (`name='thinking'`, `ephemeral=0`); `_wrap_with_exploration()` prepends the already-computed exploration text to the user_body on every ACT iteration (cheap string prefix — the LLM call ran once); excluded from user-facing render by `_NEVER_RENDER_IN_PREVIOUS`. `getUserPrompt()` assembles: world state (`world_state.render()` — empty string skips the block entirely), system awareness, `## Previous Messages`, memory seed line, current turn, ACT trail. `getSystemPrompt()` weaves `{{voice_modulation}}` and `{{adaptive_directives}}` into the unified template — no prompt tail mutation for thinking level (deliberation depth is communicated via the provider-native `thinking_mode` flag on every API call). `getTools()` excludes `rich_render` for voice-mode turns. `postTurn()` runs 8-service fan-out (see `docs/13-MESSAGE-FLOW.md` §4). Personal facts are captured inline by the LLM-native memory skill (`kind='user_specific'`); the memory skill delegates rule determination to the LUT canonicalization engine inside `data_graph_service.store()` — no background trait pipeline, no contradiction classifier.
- **`onnx_inference_service.py`** — Shared gte-modernbert encoder + swappable 2-layer MLP classifier heads. Borrows the encoder session + tokenizer from `embedding_service` (loaded once per process, ~596 MB); each task head is a tiny numpy MLP (~800 KB) loaded from a per-task `.npz` file at first use. Boot-time sha256 pin: `classifier_meta.json::base_encoder_sha256` is verified against the on-disk encoder ONNX via streaming hash; mismatch raises `RuntimeError` and the task is NOT registered. On success emits `[CLASSIFIER BOOT] <task> sha256=… input_dim=… hidden_dim=… num_classes=… activation=…`. `MODEL_REGISTRY` is the single source of truth for downloadable tasks (`thinking_level` only — the `contradiction` head was retired in favour of the deterministic LUT engine in `data_graph_service`); `ensure_models()` fetches `<prefix>-classifier_meta.json` + `<prefix>_head.npz` from the latest `chalie-ai/models` release (tag resolved once per process via the GitHub `releases/latest` API, with a cached fallback to the locally installed meta version if the API is unreachable — never hardcoded). `predict(task, text, extra_features=None)` returns `(label, confidence)` — label strings come directly from the head (e.g. `low`/`medium`/`high`). Weights follow torch `nn.Linear` `(out, in)` layout — forward uses `features @ W1.T + b1`. **Boot-gate visibility:** `_failed_registrations` records `(task, error)` for any task that fails sha256 verification; `ready` returns False whenever the list is non-empty (so `/api/system` fails fast instead of masking a degraded classifier); `degraded` is True when registration completed but at least one task failed. The system endpoint surfaces `status='degraded'` with the failed task list — silent fallback in production is no longer possible.
- **`thinking_level_classifier_service.py`** — Pre-ACT deliberation-depth gate (user channel only). `classify(user_turn, prev_level='none') -> {'level': str, 'confidence': float, 'fallback': bool}`. Level ∈ `{low, medium, high}`. Raw user turn goes straight into the encoder — no prompt template. `prev_level` is encoded as a 4-dim one-hot (`PREV_LEVEL_TO_IDX = {none:0, low:1, medium:2, high:3}`, FROZEN into the trained head's W1) concatenated to the 768-d pooled embedding → 772-d MLP features. Confidence threshold 0.70; below threshold applies sticky fallback (prev_level or `'low'` cold-start, `fallback=True`). **The classifier is the sole decision maker — no anti-demotion guard, no channel counters.** Head label returned verbatim above threshold; below threshold: sticky fallback to prev_level, or `'low'` on cold start. Raw level persisted to `transcript.thinking_level` for continuity. Model: `thinking_level` entry in `onnx_inference_service.MODEL_REGISTRY`, auto-downloaded from the latest `chalie-ai/models` release (tag resolved dynamically via GitHub API — see `onnx_inference_service.py`). Resolved level mapped to `thinking_mode` string (`'low'` → `None`, `'medium'` → `'medium'`, `'high'` → `'high'`) and passed on every provider call via `Providers.send_messages()`. High mode additionally triggers a one-shot exploration pass in `_run_thinking_gate()` (same-job LLM call, `tools=[]`, result stored in `tool_calls` as `name='thinking'`, `ephemeral=0`); the exploration LLM call runs under `THINKING_TIMEOUT=600s` (independent of `MAX_TIMEOUT=900s`) via `ThreadPoolExecutor.result(timeout=600)`.
- **`dmn_message_processor.py`** — `DMNMessageProcessor(MessageProcessor)` subclass. `CHANNEL='dmn'`, `ROLE='proactive_thought'`, `MAX_ITERATIONS=15`, `MAX_TIMEOUT=300s`. Excludes `goal_pursuit` from `NATIVE_TOOLS`. `postTurn()` logs `dmn_reflection` event + metrics only.
- **`goal_pursuit_processor.py`** — `GoalPursuitProcessor(MessageProcessor)` subclass. `CHANNEL='goal_pursuit'` (flat — `pursuit_id` in `metadata` only, never in the channel string). `MAX_ITERATIONS=50`, `MAX_TIMEOUT=7200s`. `postTurn()` logs `goal_pursuit_turn` event + metrics.
- **`scheduled_message_processor.py`** — `ScheduledMessageProcessor(MessageProcessor)` subclass. `CHANNEL='scheduled'` (flat — `item_id` in `metadata` only). Excludes `schedule` and `goal_pursuit` from `NATIVE_TOOLS`. `postTurn()` logs `scheduled_prompt_turn` + metrics + marks item executed.
- **`episode_encoder_processor.py`** — `EpisodeEncoderProcessor(MessageProcessor)` subclass. Internal one-shot processor that encodes a transcript window into episode snapshots. `CHANNEL='episode_encoder'`, `ROLE='episode_encoder'`, `JOB='frontal-cortex-unified'`, `NATIVE_TOOLS=[]`, `MAX_ITERATIONS=1`, `MAX_TIMEOUT=120s`, `SKIP_TRANSCRIPT_WRITE=True`. Constructed with `(transcript_window: str, referenced_episodes: str)`. `getUserPrompt()` formats the window (each line `[id] (timestamp) role: content`) plus any referenced episodes (candidates for update/delete) into a single user message. Returns raw LLM text from `send()` — caller (`transcript_service._trigger_episode_extraction()`) owns JSON parsing and persistence. `postTurn()` is a no-op. `EpisodeEncoderSystemPrompt` is the static cacheable system-message body: instructs the LLM to return a JSON array of snapshot objects with `gist`, `transcript_ids`, `has_open_loop`, `emotional_valence`, `emotional_arousal`, `update_id`, `delete_id`.
- **`super_episode_encoder_processor.py`** — `SuperEpisodeEncoderProcessor(MessageProcessor)` subclass. Internal one-shot processor that consolidates a high-similarity episode cluster into a single super-episode. `CHANNEL='super_episode_encoder'`, `ROLE='super_episode_encoder'`, `JOB='frontal-cortex-unified'`, `NATIVE_TOOLS=[]`, `MAX_ITERATIONS=1`, `MAX_TIMEOUT=120s`, `SKIP_TRANSCRIPT_WRITE=True`. Constructed with `(cluster_str: str)` — pre-formatted cluster of leaf episodes + their transcript spans. Returns raw LLM text; caller (`transcript_service._maybe_trigger_super_episode()`) parses the single JSON object and persists via `EpisodicService.store_episode()` then `set_consolidated_into()` for each leaf. `SuperEpisodeEncoderSystemPrompt` is the static cacheable body.
- **`context_window_service.py`** — Legacy DB-backed context window helper. `build_messages(channel)` reconstructs a multi-turn `messages[]` array from the transcript and tool_calls tables. `check_and_compact()` triggers compaction at 80% of the context limit. **Not used by the `MessageProcessor` subclasses** — the user/DMN/goal-pursuit/scheduled channels use `getPreviousMessages()` (literal-text block) and the two-stage in-loop compaction inside `send()`. Called by the end-turn backstop in `UserMessageProcessor.postTurn()`.
- **`providers.py`** — Thin singleton gateway wrapping provider resolution and LLM send. Resolves the correct LLM provider for a given job (e.g. `frontal-cortex-unified`), sends messages, returns a raw `LLMResponse`. No response parsing. `LLMResponse` fields: `text`, `model`, `provider`, `tokens_input`, `tokens_output`, `tokens_thinking` (OpenAI reasoning_tokens / Gemini thoughts_token_count; `None` for Anthropic which bundles thinking into output_tokens), `tokens_cache_read` (Anthropic only), `tokens_cache_create` (Anthropic only), `tool_calls`, `stop_reason`, `latency_ms`.

#### World State & Time Formatting
- **`world_state.py`** — In-process singleton (`world_state`) that owns all world-state data and rendering. Three public methods: `set(type, value) -> WorldState` (chainable, overwrites prior value for the same type), `get(type) -> dict` (returns `{}` when unset), `render() -> str` (assembles the literal `### Background Telemetry, Processes & Signals` block on demand — returns `''` when every section is empty). Sugar helper `push_signal(source, label, ttl=3600)` updates the internal signals dict keyed by source and prunes expired entries on each read. Zero DB writes, zero worker threads. Telemetry (from `POST /client-context`) and signals (from `/api/signals`, `WorldAwarenessService`, `IMAPHandler`) are set via `world_state.set()`; schedule and bg_process data are pulled live inside `render()` from `scheduled_items` and `transcript WHERE channel='goal_pursuit'` respectively. Process restart wipes telemetry + signals — they re-populate naturally on the next client heartbeat or signal push.
- **`time_formatter_service.py`** — Shared chokepoint for elapsed-time and duration formatting. `TimeFormatterService.duration(seconds)` — compact unit-normalised string: `0–60s → '{X}s'`, `61–3599s → '{X}m'`, `3600–86399s → '{X}h {Y}m'`, `86400+s → '{X}d {Y}h'`. Always non-negative (caller takes absolute value; directional context added at the call site). `TimeFormatterService.ago(past)` — shortcut returning `'{duration} ago'`; accepts a past datetime or an int/float of elapsed seconds; negative deltas clamp to `'0s ago'`. Used by `world_state.render()`, `news.py`, and `introspect_skill.py`. Never call open-coded `_relative_time()` helpers — route through this service.

#### Memory System
- **`transcript_service.py`** — Persistent, channel-scoped, append-only conversation record (SQLite + sqlite-vec); semantic search, keyword fallback, selective embedding (>50 tokens), 90-day TTL pruning; fires per-channel rolling episode extraction trigger (DB-state-driven: fires whenever `COUNT(transcript.id > MAX(episodes.transcript_id_end))` for the channel reaches `_EXTRACTION_THRESHOLD=20`; window = 25 entries, 5-entry overlap with the prior window). Extraction runs in a daemon thread: fetches the 25-entry window, formats it as `[id] (timestamp) role: content` lines, fetches any episodes referenced during those turns via `tool_calls WHERE tool_name='memory'`, calls `EpisodeEncoderProcessor(window_str, referenced_str).send()`, parses the returned JSON array of snapshots, hoists the novelty comparison set once per invocation, then for each snapshot embeds the gist, computes novelty + salience, and upserts via `EpisodicService.store_episode()` or `update_episode()`. Delete-only snapshots (`delete_id` set, all other fields null) go directly to `EpisodicService.soft_delete()`.
- **`compaction_service.py`** — Incremental LLM-powered summarization. Primary interactive path runs through `context_window_service`. `get_compaction()` returns the stored record including `overflow_content`.
- **`episodic_service.py`** — Episode storage + CRUD only (~270 LOC). `EpisodicService` provides `store_episode(episode_data, *, embedding=None)`, `update_episode(episode_id, fields, embedding=None)`, `soft_delete(episode_id)`, `set_consolidated_into(leaf_id, super_id)`, `get_episode_by_id(episode_id)` (bumps activation score). No retrieval, no hybrid search, no format_for_prompt, no inline overlap super-episode creation — all ripped. Module-level cluster + novelty helpers live here because they operate directly on `episodes` / `episodes_vec` and are consumed by `transcript_service` post-extraction: `find_super_candidates(channel)` (greedy pairwise cosine ≥ `SUPER_EPISODE_THRESHOLD`, min size `SUPER_EPISODE_MIN_CLUSTER`); `_fetch_novelty_comparison_set(channel)` returns embedding blobs for the (recent ∪ most-activated) apex pool, up to `NOVELTY_RECENT_LIMIT + NOVELTY_ACTIVATION_LIMIT`; `compute_novelty(new_embedding, prior_embeddings)` returns `1 - max(cosine_sim)` clamped [0,1] (1.0 when no priors). Super episodes are created exclusively via `transcript_service._maybe_trigger_super_episode` using these helpers — no second code path.
- **`episodic_retrieval_service.py`** — Module-level retrieval chokepoint. `retrieve(query_text, *, query_embedding=None, channel=None, radius=0.3, k=10, return_telemetry=False)` returns a ranked list of apex episode dicts. Pipeline: hybrid vector + FTS5 candidate pool → radius filter with adaptive population-aware shrink (`effective = radius / (1 + 0.1 × log2(N + 2))`) → RRF merge → apex traversal (`walk_up_to_apex` follows `consolidated_into` back-pointers without bumping intermediate-hop activation — depth guard `APEX_TRAVERSAL_MAX_DEPTH=20`) → composite rerank (`vector_sim*4 + fts_rank_norm*2 + arousal*1 + recency*1 + salience_norm*1 + retrieval_weight*3`, scaled ×10 for memory_skill confidence bucketing — `emotional_congruence` dropped as a constant has zero discriminative power) → debounced reconsolidation (10 min via MemoryStore, in-place activation bump on surfaced episodes). All callers — `memory_skill.recall_episodes`, `/memory/search` (`api/memory.py`), `/api/query/relevance` + `/api/query/memory` (`api/query.py`) — route here. `_get_episode_raw(episode_id, db=None)` is the side-effect-free read used during apex traversal. The `db=None` kwarg on traversal helpers supports dependency injection for tests without `mock.patch`.
- **`salience_service.py`** — Pure function `compute_salience(valence, arousal, has_open_loop, novelty) -> int [1..10]`. Formula: `emotional = 0.6*arousal + 0.4*abs(valence)`, `raw = SALIENCE_W_EMOTIONAL*emotional + SALIENCE_W_NOVELTY*novelty + SALIENCE_W_OPEN_LOOP*(1.0 if has_open_loop else 0.0)`, `return clamp(round(raw*10), 1, 10)`. Weights from `episodic_constants`. No class, no state.
- **`episodic_constants.py`** — Module-level tunable literals for episodic memory: `SUPER_EPISODE_THRESHOLD=0.90`, `SUPER_EPISODE_MIN_CLUSTER=3`, `NOVELTY_RECENT_LIMIT=100`, `NOVELTY_ACTIVATION_LIMIT=100`, `SALIENCE_W_EMOTIONAL=0.4`, `SALIENCE_W_NOVELTY=0.4`, `SALIENCE_W_OPEN_LOOP=0.2`, `APEX_TRAVERSAL_MAX_DEPTH=20`. Single source of truth for all episodic tuning — change here, pick up everywhere.
- **`data_graph_service.py`** — Unified knowledge graph (`data_graph` table + `data_graph_edges` typed join table); 3 LLM-visible kinds (`user_specific`, `system`, `misc`) + 1 internal (`moment`); dual embeddings (`data_graph_key_vec` + `data_graph_value_vec` weighted 2:1); ACT-R base-level activation scoring; split strength model (`storage_strength` monotonic ↑ + `retrieval_weight` decays); three clocks (`first_seen_at`, `last_confirmed_at`, `last_accessed_at`); 1-hop graph expansion with Granovetter cross-kind bonus + hub dampening; power-law decay per kind policy. `store()` applies the LUT canonicalization engine for `user_specific` writes: key embedding → KNN against `concept_lut.sqlite` (27 high-level concepts, cosine threshold 0.80) → rule-based dispatch (`temporal` supersede / `coexist` additive / `immutable` conflict-block); LUT misses logged to `concept_lut_misses`. `forget()` action supports hard-delete with rule-aware semantics (temporal deletes full version chain; coexist removes single value; immutable deletes the single row). Structured result dicts returned from `store()` and `forget()` for LLM self-correction.
- **`doc2query_service.py`** — Generates potential search queries for data_graph entries at write time using `doc2query/msmarco-t5-small-v1` (77M param T5 via ONNX Runtime directly — no PyTorch/optimum); stored in `search_queries` column; ONNX session unloads after 10 min idle to reclaim ~600MB
- **`list_service.py`** — Deterministic list management (shopping, to-do, chores); perfect recall with full history via `lists`, `list_items`, `list_events` tables
- **`moment_context_service.py`** — Background worker (6h poll): reads recent assistant transcript turns and writes `kind='moment'` rows into `data_graph` via DataGraphService; `api/moments.py` pin endpoint creates moments synchronously

#### Autonomous Behavior
- **`dmn_service.py`** — Default Mode Network: timer-based proactive intelligence. Fires after 60min idle (recent context — last 50 episodes) and every 6h (salience context — top-weight episodes + active goals). Constructs `DMNMessageProcessor(context, metadata).send()` directly — one instance per DMN turn; exits silently when response contains `DMN_NO_ACTION`. Quiet hours (23:00–08:00 local), rate limit (4/24h rolling). Configurable via `CHALIE_DMN_FIRST_IDLE_S` and `CHALIE_DMN_REPEAT_S`.
- **`decay_engine_service.py`** — Periodic decay (30min cycle): power-law `retrieval_weight` decay for episodes (no hard deletes), episode consolidation into super episodes, deferred reconsolidation, knowledge decay, constraint consolidation, transcript cleanup

#### Ambient Awareness
- **`ambient_inference_service.py`** — Deterministic inference engine (<1ms, zero LLM): place, attention, energy, mobility, tempo, device_context from browser telemetry + behavioral signals; thresholds loaded from `configs/agents/ambient-inference.json`; emits transition events (place, attention, energy) to event bridge when `emit_events=True`
- **`place_learning_service.py`** — Accumulates place fingerprints (geohash ~1km, never raw coords) in `place_fingerprints` table; learned patterns override heuristics after 20+ observations
- **`client_context_service.py`** — Rich client context with location history ring buffer (12 entries), place transition detection, session re-entry detection (>30min absence), demographic trait seeding from locale, and circadian hourly interaction counts; emits session_start/session_resume events to event bridge

- **`event_bridge_service.py`** — Connects ambient context changes (place, attention, energy, session) to autonomous actions; enforces stabilization windows (90s), per-event cooldowns, confidence gating, aggregation (60s bundle window), and focus gates; config in `configs/agents/event-bridge.json`

#### Tool Loop & Dispatch
- **`act_dispatcher_service.py`** — Routes tool calls to skill handlers with timeout enforcement; returns structured results with confidence and contextual notes
- **`tool_call_service.py`** — Unified API for all `tool_calls` table writes and reads. `store()` / `store_batch()` accept a `tool_call_id` parameter (LLM-generated call ID) used to reconstruct tool call lists when rebuilding context. `ephemeral` flag controls visibility in Previous Turns (ephemeral records — tool loop results, steers — are excluded; non-ephemeral — file tags, nudges — are included).
- **`goal_pursuit_processor.py`** — `GoalPursuitProcessor(MessageProcessor)` subclass: single `goal` string, daemon thread, `CHANNEL='goal_pursuit'` (flat — `pursuit_id` in `metadata` only), 50 iter / 2h timeout; surfaces result via `OutputService.enqueue_proactive()`
- **`blocks_render_service.py`** — Universal block-format renderer. All content sent over WebSocket uses the block protocol: JSON arrays of typed block objects. No HTML emitted over the wire. `blocks.js` (frontend) handles rendering.

#### Constants & Registries
- **`services/innate_skills/registry.py`** — Authoritative frozenset definitions for all skill membership sets (`ALL_SKILL_NAMES`, `PLANNING_SKILLS`, `COGNITIVE_PRIMITIVES`, `CONTEXTUAL_SKILLS`, `TRIAGE_VALID_SKILLS`, etc.). Single source of truth — all consumers import from here.
- **`services/act_action_categories.py`** — Authoritative frozenset definitions for action behavior categories (`READ_ACTIONS`, `DETERMINISTIC_ACTIONS`, `SAFE_ACTIONS`, `CRITIC_SKIP_READS`, `ACTION_FATIGUE_COSTS`).

#### Tool Integration
- **`tool_registry_service.py`** — Tool discovery, metadata management; loads first-party tools from ToolLibraryService, registers interface tools via HTTP. `execute()` returns `{'text': result}` dicts (same shape as innate skills); `invoke()` wraps in legacy `[TOOL:name]...[/TOOL]` format for output_service notifications
- **`tool_render_and_record_service.py`** — Central render+record for tool output: formats as `[tool_name(key="val",key2=0.3)] result` and writes to `tool_calls` table via `ToolCallService`
- **`tool_config_service.py`** — Tool configuration persistence; OAuth token management
- **`tool_performance_service.py`** — Performance metrics tracking; correctness-biased ranking (50% success_rate, 15% speed, 15% reliability, 10% cost, 10% preference); post-triage tool reranking; user correction propagation; 30-day preference decay
- **`tool_profile_service.py`** — Tool capability profiles powering the `find_tools` innate skill. Profiles include a `keywords` column (comma-separated, 256 char cap). Embeddings are generated from `short_summary + keywords` (not the full profile). Retrieval uses 2-axis scoring: semantic k-NN distance + keyword match count; score = `(distance * 10) - kw_match_count` (lower is better). Single-word keywords use set intersection; multi-word keywords use substring match. Results are labeled excellent/good/fair/weak instead of exposing raw scores. A dynamic TOC (one line per tool, showing its first keyword) is injected into the `find_tools` skill description at startup. Built-in tools use hardcoded profiles from `BUILTIN_TOOL_PROFILES` in `tool_library_service.py`, seeded at startup via `seed_builtin_profiles()` — bypasses the LLM profiler entirely. Interface tools still go through the LLM profiler. Staleness detection uses `manifest_hash` so re-seeding is skipped when nothing changed.

#### Identity & Learning
- Identity is now derived from the data graph (traits, facts, preferences). No dedicated identity service.

#### Infrastructure
- **`database_service.py`** — SQLite connection management (WAL mode, thread-local connections)
- **`schema_convergence_service.py`** — Bidirectional declarative schema management: converges live DB to match `schema.sql` (tables, columns, indexes, virtual tables). Adds anything missing AND drops anything no longer declared. Destructive ops gated by env flag `CHALIE_SCHEMA_ALLOW_DESTRUCTIVE` (default on). Replaces the deleted `migrations/` folder — schema.sql is the only source of truth for shape
- **`memory_store.py`** — MemoryStore: thread-safe, in-memory key-value store with Redis-compatible API
- **`config_service.py`** — JSON file config loader (agent configs, connection names); runtime config (port, host) managed by `runtime_config.py` via CLI args
- **`output_service.py`** — Output queue management for responses
- **`event_bus_service.py`** — Pub/sub event routing

#### Documents & File Management
- **`document_service.py`** — Document CRUD, soft delete with 30-day purge window, dual-layer duplicate detection (SHA-256 hash + cosine similarity on summary embeddings). Documents on disk remain source of truth; search uses data_graph artifacts.
- **`document_skill.py`** stores document content as overlapping 512-1024 char artifacts (`kind='document'`) in data_graph via `create_document_artifacts()`. File uploads extract text via `text_extractor.py` then follow the same artifact path. Memory recall surfaces top-3 document artifacts alongside regular knowledge.

### Innate Skills (`backend/services/innate_skills/`)

Built-in cognitive skills always available to the LLM:
- **`memory_skill.py`** — Four actions: `store` (save fact to data_graph), `recall` (fast search: data_graph + episodes), `reflect` (deep search: top episode + recursive 3-level graph expansion + 2 supporting facts), `forget` (hard-delete with rule-aware semantics — delegates to `data_graph_service.forget()`). `store` returns one of 12 structured response templates so the LLM can self-correct on canonicalization surprises. `TOOL_SCHEMA` description ships all 27 canonical keys + 5 store rules + niche-fact fallback. `recall_episodes()` is the single chokepoint for all episodic retrieval (seed + LLM recall); computes dynamic radius (redundancy narrowing + drift expansion), writes `memory_recall_log` telemetry row per call. Output format: `[id:{id},relevance:{level}] {text}`.
- **`introspect_skill.py`** — Comprehensive internal state report: 4 natural-language scopes (memory health, skill/tool usage, reasoning state, identity); supports "why did you do that?" via audit trail
- **`scheduler_skill.py`** — Create/list/cancel reminders and scheduled tasks (<100ms)
- **`list_skill.py`** — Deterministic list management: add/remove/check items, view, history (<50ms)
- **`goal_pursuit_skill.py`** — Spawn a background goal pursuit: takes a single `goal` string, creates a `GoalPursuitProcessor` daemon thread (`CHANNEL='goal_pursuit'`, `pursuit_id` stored in `metadata`), returns immediately; result surfaces as a proactive message when complete
- **`document_skill.py`** — Document search and management: search (hybrid semantic via sqlite-vec + FTS5 + keyword retrieval), list, view, delete, restore; documents are reference material retrieved via skill, not context assembly
- **`read_skill.py`** — Fetch and read web page content for information gathering and research
- **`find_tools_skill.py`** — Discover registered tools via semantic search against tool capability profiles; discovered tool names compound across tool loop iterations
- **`goals_skill.py`** — Goal management: list, update, complete goals
- **`rich_render_skill.py`** — Emit rich block-format content (charts, tables, structured cards) via the block protocol
- **`notes_skill.py`** — Search past conversation transcript for on-demand retrieval of older context (`notes` alias preserved for backward compat)
- **`review_tool_calls_skill.py`** — Re-read raw tool call records from a previous turn; takes `date_time` parameter, returns all records within ±5 minutes from the `tool_calls` table

## Worker Processes (`backend/workers/`)

### Workers (Daemon Threads)
- **Episodic Memory Worker** — Utility functions for goal emergence detection; episode extraction itself runs inline via the per-channel rolling transcript trigger. Trigger is DB-state-driven: fires whenever `COUNT(transcript.id > MAX(episodes.transcript_id_end))` for the channel reaches 20, independent of process-local state, so restarts never desync from accumulated history.

### Services/Daemons (Daemon Threads)
- **REST API + WebSocket** — Flask app with flask-sock on port 8081
- **DMN Service** — Timer-based proactive intelligence (60min idle → recent context, 6h cadence → salience context); constructs `DMNMessageProcessor(context, metadata).send()` directly — one instance per DMN turn; see service listing above
- **Ambient Inference Service** — Deterministic inference of place, attention, energy, mobility, tempo from browser telemetry (<1ms, zero LLM)
- **Place Learning Service** — Accumulates place fingerprints in SQLite; learned patterns override heuristics after 20+ observations
- **Decay Engine** — Periodic memory decay cycle (30min): power-law `retrieval_weight` decay for episodes (no hard deletes), super episode consolidation (3-5 similar), deferred reconsolidation, data_graph decay (per-kind policy), transcript cleanup, tool_calls purge
- **Tool Synthesis** — Reads recent tool call results from DB and runs a focused ACT loop (memory skill only) to store useful findings into `data_graph` (60min interval)
- **Scheduler Service** — Fires due reminders/tasks (60s poll); due scheduled prompts dispatch via `ScheduledMessageProcessor`
- **Autobiography Synthesis** — Synthesizes user narrative (6h cycle)
- **Profile Enrichment** — Tool profile enrichment (6h cycle, 3 tools/cycle); preference decay; usage-triggered full profile rebuilds
- **Moment Worker** — Reads recent transcript turns and writes `kind='moment'` rows into `data_graph` (6h poll)
- **User Summary Worker** — 30-min cadence; compares `MAX(last_confirmed_at)` of active `kind='user_specific'` rows against the current `user_summary` (`kind='system'`) row and, when traits are newer or the summary is missing, constructs `UserSummaryProcessor().send()` to re-synthesise `user_summary` + `user_summary_long`. No boot fire — cold-start synthesis is handled by the lazy fallback inside `UserMessageProcessor.getUserDefinition()`
- **World Awareness Service** — Pulls ambient world context (weather, news, calendar events); pushes results into the `WorldState` singleton via `world_state.push_signal("news", headline, ttl=3600)`
- **Folder Watcher** — Watches configured local folder for new documents; triggers ingestion pipeline
- **Interface Health Monitor** — Pings all paired interfaces every 30s; marks offline after 3 consecutive failures
- **Background LLM Worker** — Async LLM calls for non-interactive tasks (profile generation, tool profiling, etc.)
- **Self Model Worker** — Monitors system health signals; populates `SelfModelService` degradation indicators
- **Goal Pursuit** — `GoalPursuitProcessor` daemon thread spawned by the `goal_pursuit` innate skill; `CHANNEL='goal_pursuit'` (flat); 50 iter / 2h timeout; no plan phase; surfaces completion via `OutputService.enqueue_proactive()`
- **Document Purge Service** — Hard-deletes documents past their 30-day soft-delete window (6h cycle)
- **VaultService** — AES-256-GCM envelope encryption; PBKDF2-derived KEK wraps a random DEK stored in `vault_config`; unlocked post-login; migrates legacy Fernet data on first unlock

## Data Flow Pipeline

### User Input → Response Pipeline
```
[User Input via WebSocket]
  → [WebSocket handler] spawns daemon thread
    → UserMessageProcessor(raw_input, metadata, on_narration).send(request_id)
      ├─ _run_memory_seed()
      │    (recall_episodes caller='seed' → _memory_seed, durable DTO)
      ├─ _run_thinking_gate()  [user channel only]
      │    ThinkingLevelClassifierService.classify(user_turn, prev_level)
      │    → level ∈ {low, medium, high}; confidence threshold 0.70
      │    → confidence < 0.70: sticky fallback (reuse prev_level or 'low' on cold start)
      │    → classifier is sole decision maker — no anti-demotion guard, no counters
      │    → level persisted raw to transcript.thinking_level column
      │    → thinking_mode = None (low) | 'medium' | 'high'
      │    → level='high': one-shot exploration pass (THINKING_TIMEOUT=600s envelope)
      │         ThreadPoolExecutor.result(timeout=600) — independent of MAX_TIMEOUT
      │         timeout exceeded → WARNING, proceed without exploration
      │         result stored as tool_calls row (name='thinking', ephemeral=0)
      │         _wrap_with_exploration() prepends it to user_body each ACT iter
      │         (logs the [internal_exploration] marker exactly ONCE, pre-loop)
      │         excluded from all user-facing render by _NEVER_RENDER_IN_PREVIOUS
      ├─ loop_start = time.time()  [MAX_TIMEOUT clock anchored AFTER the gate;
      │    exploration runs under THINKING_TIMEOUT; MAX_TIMEOUT = ACT loop only]
      ├─ ACT loop (up to 30 iter / 900s):
      │    getUserPrompt()  — builds literal-text body:
      │      World State, System Awareness,
      │      ## Previous Messages (getPreviousMessages() from DB),
      │      [memory(radius=X)] seed line,
      │      [exploration block if level='high'] (via _wrap_with_exploration)
      │      user: <raw_input> [file_tags] [nudge_tag],
      │      ACT loop trail
      │    _wrap_with_checkpoint() — envelope if compaction exists
      │    [80% ctx threshold → Stage 1 / Stage 2 compaction]
      │    getSystemPrompt() — identity + adaptive + unified template (no prompt mutation)
      │    getTools()       — innate skills + discovered tools
      │    messages = [{'role':'user','content':user_body}]  (single element)
      │    Providers.send_messages(thinking_mode=...) → LLM call
      │      provider-native flag per level:
      │        low → no flag | medium → reasoning_effort/budget_tokens(4096)/think=True
      │        high → reasoning_effort/budget_tokens(16384)/think=True
      │        Ollama: pre-flight /api/show capabilities check per model (cached);
      │          'thinking' in caps → send think=True; absent → log INFO, skip flag
      │    for each tool_call: handleTool() → ActDispatcherService
      │      DTOs accumulated in _pending_tool_calls (no mid-loop DB writes)
      │    drain steer:{request_id} → user_steer DTOs
      │    repeat until text-only response or cap
      ├─ store() → transcript_service.append_atomic_turn()
      │    ONE transaction:
      │      INSERT transcript (ROLE, raw_input) → _uid
      │      INSERT tool_calls (all accumulated DTOs, sorted by timestamp)
      │      INSERT transcript ('assistant', llm_response)
      │    Post-commit daemon threads (embedding + episode extraction)
      └─ postTurn() → 8-service fan-out → OutputService.enqueue_text()
                    → pub/sub → WebSocket → client
```

### Background Processes
```
[Decay Engine] → runs every 1800s (30min)
    ├─ Episodic decay (power-law retrieval_weight, no hard delete)
    ├─ Episode consolidation (super episodes from 3-5 similar)
    ├─ Deferred reconsolidation (pending episodes from retrieval)
    ├─ Knowledge decay (confidence-weighted)
    ├─ Constraint consolidation
    └─ Transcript cleanup (unlinked entries below compaction watermark)

[DMN Service] → timer-based proactive intelligence
    ├─ 60min idle → "recent" (last 50 high-weight episodes)
    ├─ 6h cadence → "salience" (top episodes by retrieval_weight + active goals)
    └─ DMNMessageProcessor(context, metadata).send() → exits silently on DMN_NO_ACTION

```

## Key Architectural Decisions

### Unified Message Processing Path
- **User messages** go directly to `UserMessageProcessor(raw_input, metadata, on_narration).send(request_id)`. One instance per turn; no singleton; no `.process()` entry method.
- **Non-user flows** (DMN, goal pursuit, scheduled prompts) each instantiate their own `MessageProcessor` subclass directly. No central dispatcher. Each channel is its own orchestrator.
- **History as literal text** — `getPreviousMessages()` renders the channel transcript as a `## Previous Messages` block inside the user message body. The provider receives a single-element `messages[]` array — not a multi-turn array with `role='tool'` entries.
- **Atomic persistence** — `store()` calls `transcript_service.append_atomic_turn()` in one transaction at the end of the ACT loop. No mid-loop DB writes.
- **No synthesis step** — the LLM decides whether to respond directly or call tools. No separate ACT/RESPOND split.

### Mode-Specific Prompts
- Each flow type has its own focused prompt template in `backend/prompts/`
- Active templates: `frontal-cortex-unified` (user + goal pursuit), `dmn` (background intelligence), scheduled flows reuse unified
- Focused scope prevents prompt bloat and allows smaller models to handle each function

### Memory Hierarchy
- **Transcript** (SQLite + sqlite-vec, `transcript` table) — Persistent, channel-scoped, append-only conversation record; `getPreviousMessages()` reads entries above the compaction watermark and renders them as a literal `## Previous Messages` text block; written atomically per turn via `transcript_service.append_atomic_turn()`; `thinking_level` column stores the raw classifier output (`low`/`medium`/`high`) on user-role rows for continuity signal across turns
- **Compaction** (SQLite, `compactions` table) — Incremental LLM summarization triggered at 80% of the provider's context limit inside `MessageProcessor.send()`; stores `compacted_text` + `compacted_up_to_id` watermark; keyed by `channel`; surfaced to the LLM via the `### Checkpoint` envelope prepended by `_wrap_with_checkpoint()`
- **Episodes** (SQLite + sqlite-vec, `episodes` table) — Transcript-linked narrative units with power-law `retrieval_weight` decay and `storage_strength` that never decreases; created by a DB-state-driven per-channel transcript trigger (fires when `COUNT(transcript.id > MAX(episodes.transcript_id_end)) >= 20`; window = 25 entries = 20 new + 5 overlap with the prior window); FTS5 index synced on insert; overlapping new episodes consolidate into "super episodes" referencing source episodes; retrieval always routed through `memory_skill.recall_episodes()` for uniform telemetry. `reflect` action traverses the episode graph recursively (up to 3 levels: super → child → transcript).
- **Data Graph** (SQLite + sqlite-vec, `data_graph` + `data_graph_edges` tables) — Unified knowledge graph with 3 LLM-visible kinds (`user_specific`, `system`, `misc`) + 1 internal (`moment`); split strength model (`storage_strength` + `retrieval_weight`); ACT-R activation scoring; dual embeddings (key_vec + value_vec weighted 2:1); FTS5 porter-stemmed search; cosine relevance floor (0.42) drops irrelevant vector candidates; 1-hop graph expansion with typed edges, Granovetter cross-kind bonus, and hub dampening
- **Tool Calls** (SQLite, `tool_calls` table) — Per-turn tool invocations with `ephemeral` flag; all accumulated DTOs written atomically at turn end via `transcript_service.append_atomic_turn()`; durable (`ephemeral=0`) rows (memory auto-seed, full compaction records) are replayed in `getPreviousMessages()` on future turns; ephemeral rows are audit-only and never re-injected into context
- **Lists** (SQLite) — Deterministic ground-truth state (shopping, to-do, chores); perfect recall, no decay, full event history

Each layer optimized for its timescale. Conversation history is rendered by `getPreviousMessages()` as a literal text block inside `getUserPrompt()`. Per-turn additions (world state via `world_state.render()`, episodic auto-seed, voice guard) are assembled directly by `UserMessageProcessor.getUserPrompt()`. Lists injected into all prompts as `{{active_lists}}`.

### Configuration Precedence
```
Environment variables > .env file > JSON config files > hardcoded defaults
```

See `docs/02-PROVIDERS-SETUP.md` for provider configuration.

### Thread-Safe Worker State
- All workers run as daemon threads within a single Python process
- Shared state managed via thread-safe data structures (locks, queues)
- No multiprocessing overhead — lightweight, in-process coordination

### Error Resilience
- All workers catch JSON decode errors from LLM responses
- Log meaningful messages instead of crashing
- Return status strings for graceful degradation

## Safety & Constraints

### Hard Boundaries
- **Prompt hierarchy** immutable (marked as "authoritative and final")
- **Skill registry** fixed at startup (no runtime skill registration)
- **Data scope** parameterized by channel (no cross-channel leakage)
- **Speaker confidence** gates trait storage (unknown speakers = 0.3 penalty)

### Operational Limits
- **User tool loop** (`UserMessageProcessor`): 30 max iterations, 15 min cumulative timeout; mid-loop narration streamed via `on_narration` callback → WebSocket
- **DMN tool loop** (`DMNMessageProcessor`): 15 max iterations, 5 min timeout; exits silently on `DMN_NO_ACTION`
- **Goal pursuit** (`GoalPursuitProcessor`): 50 max iterations, 2h wall-clock timeout; `CHANNEL='goal_pursuit'` (flat); no concurrency cap, no state machine
- **Scheduled** (`ScheduledMessageProcessor`): 30 max iterations, 15 min timeout; `CHANNEL='scheduled'` (flat)

### Anti-Manipulation
- **Identity isolation**: 6 vectors with coherence constraints
- **No vulnerability simulation**: Explicitly forbidden
- **Exponential backoff**: System retreats on silence (opposite of dependency)
- **No flattery optimization**: Soul axiom: "Never optimize by misleading"

## Configuration Files

### Primary Configuration
- **`configs/connections.json`** — SQLite path and MemoryStore settings
- **`configs/agents/*.json`** — LLM settings (model, temperature, timeout) per provider job

### Provider Configuration
- Stored in SQLite `providers` table (not JSON files)
- Runtime configurable via REST API (`/api/providers`)
- Supports: Ollama, Anthropic, OpenAI, Google Gemini

See `docs/02-PROVIDERS-SETUP.md` for detailed setup instructions.

## REST API

### Available Blueprints
- **`user_auth`** — Account creation, login, API key management
- **`conversation`** — Chat endpoint (WebSocket streaming), conversation list/retrieval
- **`memory`** — Memory search, fact management
- **`proactive`** — Outreach/notifications, upcoming tasks
- **`privacy`** — Data deletion, export
- **`system`** — Health, version, settings, observability (memory, tools, tasks, traits)
- **`tools`** — Tool execution, configuration
- **`providers`** — LLM provider configuration
- **`push`** — Push notification subscription
- **`scheduler`** — Reminders and scheduled tasks
- **`lists`** — List management
- **`stubs`** — Placeholder endpoints (calendar, notifications, integrations, permissions) returning 501

### Observability Endpoints (`/system/observability/*`)
- **`routing`** — Mode router decision distribution and recent activity
- **`records`** — Paginated data-graph records. `GET /system/observability/records?source=<episodes|user|system>&q=<str>&offset=<int>` returns up to 250 rows of `{created, last_accessed, key, value}` ordered by `last_accessed_at IS NULL, last_accessed_at DESC, created DESC`.
- **`tools`** — Tool performance stats
- **`tasks`** — Active goal pursuit threads
- **`world-state`** — `GET /system/observability/world-state` returns `{"rendered": "...", "inputs": {"telemetry": {...}, "signals": {...}, "schedule": [...], "bg_processes": [...]}}`. `rendered` is the exact literal block the LLM receives (empty string when nothing salient). `inputs` exposes the raw data behind each section — useful for debugging why the block says what it says. Frontend Brain Cognition tab renders the `rendered` block verbatim (monospace) plus an inputs table.

See API blueprints in `backend/api/` for full reference.

## Testing Strategy

**Feature tests, zero mocks, real-world only.** Every test runs the real production stack — real services, real DB, real models. No `MagicMock`, no `patch`, no `monkeypatch` of production code. See `docs/12-TESTING.md` for the full discipline.

### Test Markers
- `@pytest.mark.unit` — Pure functions only (no IO, no collaborators, no state)
- `@pytest.mark.integration` — Feature tests against the real stack (default for anything touching a service, DB, model, or network)

### Hard Rules
- **3–10 feature tests per service.** More than 10 is a design smell.
- **Nightly YAML scenarios** (`/Volumes/llm/chalie-nightly-test/scenarios/`) are the primary feature tests. Python-level feature tests are a fast-feedback supplement.
- **Coder agent never writes or modifies tests.** The `tester` agent has exclusive ownership.

### Test Organization
```
backend/tests/
├── test_<service>_features.py    # Integration-marked feature tests (real stack)
├── test_<module>_pure_functions.py   # Unit-marked pure-function tests
└── helpers.py                    # Deterministic data factories, NOT mocks
```

Run all tests: `pytest`
Run unit only: `pytest -m unit`
Run feature tests: `pytest -m integration`

## Development Workflow

### Setup
```bash
cd backend
pip install -r requirements.txt
source .venv/bin/activate
cp .env.example .env
```

### Local Development
```bash
# Single command — starts Flask + WebSocket + all daemon threads
python backend/run.py
```

No external services required. SQLite and MemoryStore are embedded — everything runs in one process.

## Deployment Notes

- **No Telemetry**: Zero external calls except to configured LLM/voice providers
- **Local First**: All data stored locally unless external providers configured
- **Encryption**: AES-256-GCM envelope encryption via VaultService (password-derived KEK wraps a random DEK)
- **CORS**: Defaults to localhost, restrict before production

## Interface Layer

External applications can extend Chalie's capabilities by pairing as interfaces. Interfaces expose tool capabilities that Chalie registers in its normal tool pipeline.

### Protocol

Chalie → Interface:
- `GET /health` — periodic liveness check (every 30s)
- `GET /capabilities` — fetch tool manifests
- `POST /execute` — invoke a capability

Interface → Chalie:
- `POST /api/signals` — push events (authenticated via signal_token)

### Pairing

Bluetooth-style: Chalie generates a one-time pairing key (brain dashboard). User enters it into the interface along with Chalie's host:port. Interface calls `POST /api/interfaces/pair`. Both sides exchange connection details.

### Health Monitoring

A daemon thread pings all paired interfaces every 30 seconds. After 3 consecutive failures, an interface is marked offline and its tools become invisible to the LLM. Recovery is automatic on the next successful health check.

### Key Files
- `services/interface_registry_service.py` — Core lifecycle management
- `api/interfaces.py` — REST API for pairing, listing, removal
- `workers/interface_health_worker.py` — Health monitor daemon
- `schema.sql` — Interface tables (interfaces, interface_tools); shape converged on every boot by `SchemaConvergenceService`

## Glossary

- **MessageProcessor**: Abstract base class for all LLM turns. Defines the tool-calling loop, transcript persistence, and context window reconstruction. Subclasses: `UserMessageProcessor`, `DMNMessageProcessor`, `ScheduledMessageProcessor`, `GoalPursuitProcessor`, `EpisodeEncoderProcessor` (internal, `SKIP_TRANSCRIPT_WRITE=True`), `SuperEpisodeEncoderProcessor` (internal, `SKIP_TRANSCRIPT_WRITE=True`), `UserSummaryProcessor` (internal, `SKIP_TRANSCRIPT_WRITE=True`, synthesises stored `user_specific` facts into `user_summary` + `user_summary_long` rows; 30-min cadence worker + lazy fallback from `UserMessageProcessor.getUserDefinition()`).
- **Channel**: Stable string identifier scoping a conversation context in the `transcript` and `compactions` tables. Replaced the former topic/thread distinction.
- **Block Protocol**: Universal content format — all LLM-to-client content is JSON arrays of typed block objects. `blocks_render_service.py` (backend) → `blocks.js` (frontend). No HTML over the wire.
- **DMN (Default Mode Network)**: Timer-based proactive intelligence; fires after 60min idle (recent context) and every 6h (salience); uses `DMNMessageProcessor`; exits on `DMN_NO_ACTION`.
- **GoalPursuitProcessor**: Background `MessageProcessor` subclass. Runs a single goal string for up to 50 iterations / 2h. Channel-isolated, surfaces result as a proactive message.
- **Episode**: Transcript-linked narrative memory unit with `storage_strength` (never decreases) and `retrieval_weight` (power-law decay); created by rolling transcript trigger; consolidates into super episodes.
- **Dynamic Radius**: Per-turn system-determined retrieval radius for episodic recall: `effective = baseline × narrow_factor × expand_factor`. Narrow on redundancy, expand on drift. Never exposed to the LLM.
- **memory_recall_log**: Telemetry table recording every episodic retrieval (caller, radius components, candidate counts). Read by meta-harness for tuning the 8 radius constants.
- **Data Graph**: Unified knowledge graph (`data_graph` + `data_graph_edges` tables) with 3 LLM-visible kinds (user_specific, system, misc) + 1 internal (moment). Split strength model, ACT-R activation, dual embeddings, typed edges. `user_specific` writes go through the LUT canonicalization engine (27-concept `concept_lut.sqlite`). Replaces former `knowledge` table.
- **LUT Canonicalization Engine**: Deterministic conflict-resolution engine inside `data_graph_service.store()`. Key embedding → KNN against `concept_lut.sqlite` (cosine ≥ 0.80) → rule dispatch: `temporal` supersedes previous value; `coexist` stores additively; `immutable` blocks write. Misses logged to `concept_lut_misses`. Replaces the ONNX contradiction classifier.
- **concept_lut_misses**: Observability table recording `user_specific` keys that did not match any of the 27 LUT concepts (kind, key, value_preview, count, first_seen, last_seen). Used for LUT curation.
- **Mode Router**: Deterministic mathematical function selecting engagement mode from observable signals. **Only consulted for non-user flows** (DMN fallback). User turns bypass it entirely.
- **Context Warmth**: Signal (0.0-1.0) measuring how much context is available for the current channel.
- **Salience**: Computed importance score [1..10] produced by `compute_salience(valence, arousal, has_open_loop, novelty)` in `salience_service.py`. Weights from `episodic_constants`: 40% emotional (0.6×arousal + 0.4×|valence|), 40% novelty (1 − max cosine-sim against apex episode set), 20% open-loop boost. Pure function, no class, no state.
