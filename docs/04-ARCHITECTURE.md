# System Architecture

## Overview

Chalie is a **persistent cognitive agent** — a continuously running cognitive runtime, not a request-response service. Intelligence emerges from experience: every interaction flows through a memory pipeline that compresses, abstracts, and decays information over time. The system is not a chatbot or assistant wrapper; it is a cognitive runtime that forms memories, runs background processes, exercises judgment, and diverges into a unique identity shaped by its interaction history.

## Core Architecture

### System Type
- **Synthetic cognitive brain** using LLMs to replicate human brain functions
- **Tech Stack**: Python backend, SQLite (WAL mode + sqlite-vec + FTS5), MemoryStore (in-memory, thread-safe), Ollama (configurable LLMs), Vanilla JavaScript frontend (Radiant design system)
- **Core Pattern**: Single-process architecture with daemon threads; all LLM calls run through `MessageProcessor` subclasses; PromptQueue/digest_worker is a deprecated legacy path pending removal

### Communication Pattern
1. Client connects to `/ws` (WebSocket via flask-sock)
2. Client sends `{"type": "message", "text": "..."}` JSON frame
3. Backend spawns daemon thread → `UserMessageProcessor.process()` → response published via pub/sub → WebSocket frames (`status` → `message` → `done`)
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
├── migrations/        # Database migrations
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
└── on-boarding/       # Account setup wizard
```

**Module communication**: Constructor injection for shared services, callback registration for cross-module events, custom DOM events (`chalie:action`, `chalie:speak`, `chalie:pin-moment`) for loose coupling. Modules never reference each other directly — `app.js` wires all connections.

**Asset versioning**: Flask injects a `<script type="importmap">` into `index.html` at serve time, mapping all module imports to `?v=VERSION` URLs. The `VERSION` file is the single source of truth. Service worker uses network-first for JS/CSS (localhost = 0ms latency) with cache fallback for offline/PWA.

**IMPORTANT**: UI code must exist under `/interface/`, `/brain/`, or `/on-boarding/` only.

## Key Services

### Core Services (`backend/services/`)

#### Routing & Decision Making
- **`mode_router_service.py`** — Deterministic mode routing (~5ms) with signal collection + tie-breaker. **User messages bypass this entirely** — they go directly to `UserMessageProcessor`. The router is only consulted for non-user flows (DMN, fallback).
- `routing_stability_regulator_service.py` — Single authority for router weight mutation. **Service file currently absent** (registered as optional at boot, fails gracefully; inactive).

#### Response Generation
- **`frontal_cortex_service.py`** — Thin legacy facade preserving public API. Actual prompt assembly in `prompt_assembly_service.py`, LLM dispatch in `response_generation_service.py` and ultimately `providers.py`.
- **`voice_mapper_service.py`** — Translates identity vectors to tone instructions

#### Message Processing
- **`message_processor.py`** — Base class for all LLM turns. Handles transcript persistence, LLM invocation via `Providers`, and the standard tool-calling loop. The context window is always reconstructed from the database via `context_window_service.build_messages()` — nothing accumulates in memory. Compaction triggers at 80% of the provider's context limit.
- **`message_processor_v2.py`** — **In-progress refactor** (parallel to legacy base). Introduces: abstract interface (`CHANNEL`/`ROLE`/`getUserPrompt()`/`getUserDefinition()`), per-subclass `NATIVE_TOOLS`/`MAX_ITERATIONS`/`MAX_TIMEOUT`, ContextVar-based current-processor discovery (`current_processor()` / `bind_current_processor()`), and `_memory_query_history` for dynamic radius tuning. Will replace `message_processor.py` at Commit 11 of the refactor.
- **`user_message_processor.py`** — Entry point for all user-initiated messages. Builds user + system prompts, applies voice-mode tool filtering, sends via `MessageProcessor.send()` (full DB-backed tool loop; 30 iter / 15 min), and returns a normalized result dict. Singleton.
- **`dmn_message_processor.py`** — `MessageProcessor` subclass for the Default Mode Network. 15 iter / 5 min. Exits silently when response contains `DMN_NO_ACTION`. Excludes `goal_pursuit` skill.
- **`scheduled_message_processor.py`** — `MessageProcessor` subclass for scheduled prompts (replaces the legacy prompt-queue path).
- **`context_window_service.py`** — DB-backed context window construction. `build_messages(channel)` always reconstructs the full LLM messages array from the transcript and tool_calls tables. `check_and_compact(channel, context_limit, job, pending_content, is_tool_triggered)` triggers compaction at 80% of the context limit and handles overflow: if a pending tool result would exceed the hard limit, it compacts first and stores the tool result as `overflow_content` in the compaction record. Overflow content is placed *before* the compacted summary in the reconstructed messages so the compacted text receives higher recency/attention weight. Compaction always uses the same provider job as the conversation.
- **`providers.py`** — Thin singleton gateway wrapping provider resolution and LLM send. Resolves the correct LLM provider for a given job (e.g. `frontal-cortex-unified`), sends messages, returns a raw `LLMResponse`. No response parsing.
- **`user_prompt_assembly_service.py`** — Builds the per-turn user message: world state header, voice-mode guard, system awareness, and current turn (episodic auto-recall + user message + file/nudge tags). Conversation history is **not** assembled here — it is handled by `context_window_service.build_messages()`. Changes every turn; not cached.
- **`system_prompt_assembly_service.py`** — Builds the stable, cacheable system prompt: identity modulation, adaptive directives, and the frontal-cortex-unified template. Designed for provider-side prompt caching. Previously an empty shell — now fully implemented.

#### Memory System
- **`transcript_service.py`** — Persistent, channel-scoped, append-only conversation record (SQLite + sqlite-vec); semantic search, keyword fallback, selective embedding (>50 tokens), 90-day TTL pruning; fires rolling episode extraction trigger at `id % 25`
- **`compaction_service.py`** — Incremental LLM-powered summarization. Primary interactive path runs through `context_window_service`. `get_compaction()` returns the stored record including `overflow_content`.
- **`episodic_service.py`** — Unified episodic service: SQLite CRUD, hybrid vector + FTS5 retrieval (`retrieve_episodes(query_text, radius, query_embedding, return_telemetry)`), power-law `retrieval_weight` decay, `format_for_prompt()`. All episodic recall (seed + LLM) is routed through `memory_skill.recall_episodes()` to ensure uniform telemetry via `memory_recall_log`.
- **`knowledge_service.py`** — Unified knowledge store (`knowledge` table replaces former `user_traits`, `semantic_concepts`, `procedural_memory`, etc.); stores traits, facts, procedures, preferences, rules, metrics; RRF hybrid search (exact + FTS5 porter-stemmed + vector KNN); NLTK stop-word filtering; doc2query expansion at write time; decay management and prompt injection
- **`doc2query_service.py`** — Generates potential search queries for knowledge entries at write time using `doc2query/msmarco-t5-small-v1` (77M param T5 via ONNX Runtime directly — no PyTorch/optimum); stored in `search_queries` column; ONNX session unloads after 10 min idle to reclaim ~600MB
- **`list_service.py`** — Deterministic list management (shopping, to-do, chores); perfect recall with full history via `lists`, `list_items`, `list_events` tables
- **`moment_service.py`** — Pinned message bookmarks with LLM-enriched context, sqlite-vec semantic search, and salience boosting; moments stored as documents with `source_type='moment'`
- **`moment_enrichment_service.py`** — Background worker (5min poll): generates LLM summaries, seals moments after 4hrs

#### Autonomous Behavior
- **`dmn_service.py`** — Default Mode Network: timer-based proactive intelligence. Fires after 60min idle (recent context — last 50 episodes) and every 6h (salience context — top-weight episodes + active goals). Calls `DMNMessageProcessor().process()` — full tool-loop-based LLM call; exits silently when response contains `DMN_NO_ACTION`. Quiet hours (23:00–08:00 local), rate limit (4/24h rolling). Configurable via `CHALIE_DMN_FIRST_IDLE_S` and `CHALIE_DMN_REPEAT_S`.
- **`decay_engine_service.py`** — Periodic decay (30min cycle): power-law `retrieval_weight` decay for episodes (no hard deletes), episode consolidation into super episodes, deferred reconsolidation, knowledge decay, constraint consolidation, transcript cleanup

#### Ambient Awareness
- **`ambient_inference_service.py`** — Deterministic inference engine (<1ms, zero LLM): place, attention, energy, mobility, tempo, device_context from browser telemetry + behavioral signals; thresholds loaded from `configs/agents/ambient-inference.json`; emits transition events (place, attention, energy) to event bridge when `emit_events=True`
- **`place_learning_service.py`** — Accumulates place fingerprints (geohash ~1km, never raw coords) in `place_fingerprints` table; learned patterns override heuristics after 20+ observations
- **`client_context_service.py`** — Rich client context with location history ring buffer (12 entries), place transition detection, session re-entry detection (>30min absence), demographic trait seeding from locale, and circadian hourly interaction counts; emits session_start/session_resume events to event bridge

- **`event_bridge_service.py`** — Connects ambient context changes (place, attention, energy, session) to autonomous actions; enforces stabilization windows (90s), per-event cooldowns, confidence gating, aggregation (60s bundle window), and focus gates; config in `configs/agents/event-bridge.json`

#### Tool Loop & Dispatch
- **`act_dispatcher_service.py`** — Routes tool calls to skill handlers with timeout enforcement; returns structured results with confidence and contextual notes
- **`act_reflection_service.py`** — Enqueues tool outputs for background experience assimilation
- **`tool_call_service.py`** — Unified API for all `tool_calls` table writes and reads. `store()` / `store_batch()` accept a `tool_call_id` parameter (LLM-generated call ID) used to reconstruct tool call lists when rebuilding context. `ephemeral` flag controls visibility in Previous Turns (ephemeral records — tool loop results, steers — are excluded; non-ephemeral — file tags, nudges — are included).
- **`goal_pursuit_processor.py`** — `GoalPursuitProcessor(MessageProcessor)` subclass: single `goal` string, daemon thread, 50 iter / 2h timeout, channel-isolated (`goal_pursuit:{uuid}`); surfaces result via `OutputService.enqueue_proactive()`; uses `review_tool_calls` skill to recall prior work if context was compacted
- **`blocks_render_service.py`** — Universal block-format renderer. All content sent over WebSocket uses the block protocol: JSON arrays of typed block objects. No HTML emitted over the wire. `blocks.js` (frontend) handles rendering.

#### Constants & Registries
- **`services/innate_skills/registry.py`** — Authoritative frozenset definitions for all skill membership sets (`ALL_SKILL_NAMES`, `PLANNING_SKILLS`, `COGNITIVE_PRIMITIVES`, `CONTEXTUAL_SKILLS`, `TRIAGE_VALID_SKILLS`, etc.). Single source of truth — all consumers import from here.
- **`services/act_action_categories.py`** — Authoritative frozenset definitions for action behavior categories (`READ_ACTIONS`, `DETERMINISTIC_ACTIONS`, `SAFE_ACTIONS`, `CRITIC_SKIP_READS`, `ACTION_FATIGUE_COSTS`).
- **`services/act_memory_keys.py`** — Centralized MemoryStore key patterns for the tool system (deferred cards, tool caches, heartbeat, reflection queue).

#### Tool Integration
- **`tool_registry_service.py`** — Tool discovery, metadata management; loads first-party tools from ToolLibraryService, registers interface tools via HTTP; invokes first-party tools directly in-process
- **`tool_config_service.py`** — Tool configuration persistence; webhook key generation (HMAC-SHA256 + replay protection via X-Chalie-Signature/X-Chalie-Timestamp)
- **`tool_performance_service.py`** — Performance metrics tracking; correctness-biased ranking (50% success_rate, 15% speed, 15% reliability, 10% cost, 10% preference); post-triage tool reranking; user correction propagation; 30-day preference decay
- **`tool_profile_service.py`** — Tool capability profiles powering the `find_tools` innate skill. Profiles include a `keywords` column (comma-separated, 256 char cap). Embeddings are generated from `short_summary + keywords` (not the full profile). Retrieval uses 2-axis scoring: semantic k-NN distance + keyword match count; score = `(distance * 10) - kw_match_count` (lower is better). Single-word keywords use set intersection; multi-word keywords use substring match. Results are labeled excellent/good/fair/weak instead of exposing raw scores. A dynamic TOC (one line per tool, showing its first keyword) is injected into the `find_tools` skill description at startup. Built-in tools use hardcoded profiles from `BUILTIN_TOOL_PROFILES` in `tool_library_service.py`, seeded at startup via `seed_builtin_profiles()` — bypasses the LLM profiler entirely. Interface tools still go through the LLM profiler. Staleness detection uses `manifest_hash` so re-seeding is skipped when nothing changed.
- **Webhook endpoint** (`/api/tools/webhook/<name>`) — External tool triggers with HMAC-SHA256 or simple token auth, 30 req/min rate limit, 512KB payload cap

#### Identity & Learning
- **`identity_service.py`** — 6-dimensional identity vector system with coherence constraints
- **`identity_state_service.py`** — Tracks identity state changes and evolution

#### Infrastructure
- **`database_service.py`** — SQLite connection management (WAL mode, thread-local connections)
- **`schema_convergence_service.py`** — Declarative schema management: converges live DB to match `schema.sql` (tables, columns, indexes, virtual tables)
- **`memory_store.py`** — MemoryStore: thread-safe, in-memory key-value store with Redis-compatible API
- **`config_service.py`** — JSON file config loader (agent configs, connection names); runtime config (port, host) managed by `runtime_config.py` via CLI args
- **`output_service.py`** — Output queue management for responses
- **`event_bus_service.py`** — Pub/sub event routing

#### Documents & File Management
- **`document_service.py`** — Document CRUD, chunk storage, hybrid search (semantic via sqlite-vec + FTS5 + keyword boost via Reciprocal Rank Fusion), soft delete with 30-day purge window, dual-layer duplicate detection (SHA-256 hash + cosine similarity on summary embeddings)
- **`document_processing_service.py`** — Full extraction pipeline: text extraction (pdfplumber, python-docx, python-pptx, trafilatura), regex-based metadata extraction (dates, companies, monetary values, reference numbers, document type heuristic), adaptive chunk sizing by document type, SimHash fingerprinting, language detection (langdetect)

### Innate Skills (`backend/services/innate_skills/`)

Built-in cognitive skills always available to the LLM:
- **`memory_skill.py`** — Unified recall + store across ALL memory layers. `recall_episodes()` is the single chokepoint for all episodic retrieval (seed + LLM recall); computes dynamic radius (redundancy narrowing + drift expansion), writes `memory_recall_log` telemetry row per call.
- **`introspect_skill.py`** — Comprehensive internal state report: 4 natural-language scopes (memory health, skill/tool usage, reasoning state, identity); supports "why did you do that?" via audit trail
- **`scheduler_skill.py`** — Create/list/cancel reminders and scheduled tasks (<100ms)
- **`autobiography_skill.py`** — Retrieve synthesized user narrative with optional section extraction (<500ms)
- **`list_skill.py`** — Deterministic list management: add/remove/check items, view, history (<50ms)
- **`goal_pursuit_skill.py`** — Spawn a background goal pursuit: takes a single `goal` string, creates a `GoalPursuitProcessor` daemon thread in a channel-isolated context (`goal_pursuit:{uuid}`), returns immediately; result surfaces as a proactive message when complete
- **`document_skill.py`** — Document search and management: search (hybrid semantic via sqlite-vec + FTS5 + keyword retrieval), list, view, delete, restore; documents are reference material retrieved via skill, not context assembly
- **`read_skill.py`** — Fetch and read web page content for information gathering and research
- **`find_tools_skill.py`** — Discover registered tools via semantic search against tool capability profiles; discovered tool names compound across tool loop iterations
- **`goals_skill.py`** — Goal management: list, update, complete goals
- **`rich_render_skill.py`** — Emit rich block-format content (charts, tables, structured cards) via the block protocol
- **`notes_skill.py`** — Search past conversation transcript for on-demand retrieval of older context (`notes` alias preserved for backward compat)
- **`review_tool_calls_skill.py`** — Re-read raw tool call records from a previous turn; takes `date_time` parameter, returns all records within ±5 minutes from the `tool_calls` table

## Worker Processes (`backend/workers/`)

### Legacy Queue Workers (Daemon Threads — Deprecated)
- **Digest Worker** (`digest_worker.py`) — **Deprecated, pending removal.** Retained only because a handful of internal callers have not yet been migrated to `MessageProcessor` subclasses. WebSocket user messages bypass it entirely.
- **Episodic Memory Worker** — Utility functions for goal emergence detection; episode extraction itself runs inline via the rolling transcript trigger (`id % 25`).

### Services/Daemons (Daemon Threads)
- **REST API + WebSocket** — Flask app with flask-sock on port 8081
- **DMN Service** — Timer-based proactive intelligence (60min idle → recent context, 6h cadence → salience context); calls `DMNMessageProcessor().process()`; see service listing above
- **Ambient Inference Service** — Deterministic inference of place, attention, energy, mobility, tempo from browser telemetry (<1ms, zero LLM)
- **Place Learning Service** — Accumulates place fingerprints in SQLite; learned patterns override heuristics after 20+ observations
- **Decay Engine** — Periodic memory decay cycle (30min): power-law `retrieval_weight` decay for episodes (no hard deletes), super episode consolidation (3-5 similar), deferred reconsolidation, knowledge decay, transcript cleanup, constraint consolidation
- **Experience Assimilation** — Tool results → episodic memory (60s poll)
- **Scheduler Service** — Fires due reminders/tasks (60s poll); due scheduled prompts dispatch via `ScheduledMessageProcessor`
- **Autobiography Synthesis** — Synthesizes user narrative (6h cycle)
- **Profile Enrichment** — Tool profile enrichment (6h cycle, 3 tools/cycle); preference decay; usage-triggered full profile rebuilds
- **Moment Enrichment** — Enriches pinned moments with LLM summary, seals after 4hrs (5min poll)
- **World Awareness Service** — Pulls ambient world context (weather, news, calendar events) for `WorldStateService`
- **Folder Watcher** — Watches configured local folder for new documents; triggers ingestion pipeline
- **Interface Health Monitor** — Pings all paired interfaces every 30s; marks offline after 3 consecutive failures
- **Background LLM Worker** — Async LLM calls for non-interactive tasks (profile generation, tool profiling, etc.)
- **Self Model Worker** — Monitors system health signals; populates `SelfModelService` degradation indicators
- **Goal Pursuit** — `GoalPursuitProcessor` daemon thread spawned by the `goal_pursuit` innate skill; 50 iter / 2h timeout; no plan phase; surfaces completion via `OutputService.enqueue_proactive()`
- **Document Purge Service** — Hard-deletes documents past their 30-day soft-delete window (6h cycle)
- **VaultService** — AES-256-GCM envelope encryption; PBKDF2-derived KEK wraps a random DEK stored in `vault_config`; unlocked post-login; migrates legacy Fernet data on first unlock

## Data Flow Pipeline

### User Input → Response Pipeline
```
[User Input via WebSocket]
  → [WebSocket handler] spawns daemon thread
    → [UserMessageProcessor.process()]
      ├─ UserPromptAssemblyService.build()
      │    (world state, voice guard, system awareness, episodic recall)
      │    NOTE: conversation history NOT assembled here
      ├─ SystemPromptAssemblyService.build()
      │    (identity, directives, frontal-cortex-unified template)
      ├─ MessageProcessor.send()
      │    ├─ transcript.append(channel, 'user', ...)
      │    ├─ context_window_service.check_and_compact() [pre-call]
      │    ├─ context_window_service.build_messages() → messages array
      │    │    (always reconstructed from DB — compaction + transcript + tool_calls)
      │    ├─ Providers.send_messages() → first LLM call
      │    ├─ Tool loop (standard tool-calling protocol):
      │    │    ├─ transcript.append(channel, 'assistant', ...)
      │    │    ├─ ActDispatcherService.dispatch_action() per tool call
      │    │    ├─ context_window_service.check_and_compact()
      │    │    │    (overflow check: compact before storing if result would exceed limit)
      │    │    ├─ transcript.append(channel, 'tool', result)
      │    │    ├─ ToolCallService.store(..., tool_call_id=tc['id'])
      │    │    ├─ context_window_service.check_and_compact() [post-iter]
      │    │    ├─ context_window_service.build_messages() → rebuilt array
      │    │    ├─ Providers.send_messages() → next LLM call
      │    │    └─ repeat until no tool_calls or cap (30 iter / 15 min)
      │    └─ transcript.append(channel, 'assistant', final response)
      └─ OutputService.enqueue_text() → pub/sub → WebSocket → client

  Background (async, from OutputService):
    → [Transcript Service] (rolling trigger at id % 25)
      → [Episode Extractor] → SQLite episodes table
      → Traits extracted → Knowledge Store
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
    └─ DMNMessageProcessor().process() → exits silently on DMN_NO_ACTION
```

## Key Architectural Decisions

### Unified Message Processing Path
- **User messages bypass the mode router entirely** — they go directly to `UserMessageProcessor` which runs the full tool-calling loop (30 iter / 15 min).
- **Non-user flows** (DMN idle/salience cycles, scheduled prompts) use purpose-built `MessageProcessor` subclasses (`DMNMessageProcessor`, `ScheduledMessageProcessor`, `GoalPursuitProcessor`).
- **No synthesis step** — the LLM decides whether to respond directly or call tools. No separate ACT/RESPOND split.

### Mode-Specific Prompts
- Each flow type has its own focused prompt template in `backend/prompts/`
- Active templates: `frontal-cortex-unified` (user + goal pursuit), `dmn` (background intelligence), scheduled flows reuse unified
- Focused scope prevents prompt bloat and allows smaller models to handle each function

### Deterministic Mode Router (non-user flows only)
- **Scope**: Only consulted for DMN fallback and non-WebSocket flows; user turns never pass through it
- **Signals**: ~17 observable signals from context + NLP (context warmth, question marks, greeting patterns, etc.)
- **Scores**: Each mode gets weighted composite score; highest wins; ONNX tie-breaker for ambiguous cases

### Memory Hierarchy
- **Transcript** (SQLite + sqlite-vec, `transcript` table) — Persistent, channel-scoped, append-only conversation record; `context_window_service.build_messages()` reads all entries above the compaction watermark on every LLM call (no in-memory accumulation)
- **Compaction** (SQLite, `compactions` table) — Incremental LLM summarization triggered at 80% of the provider's context limit; stores compacted text, watermark ID, and `overflow_content`; keyed by `channel`
- **Episodes** (SQLite + sqlite-vec, `episodes` table) — Transcript-linked narrative units with power-law `retrieval_weight` decay and `storage_strength` that never decreases; created by rolling transcript trigger (`id % 25`); consolidate into "super episodes" referencing source episodes; retrieval always routed through `memory_skill.recall_episodes()` for uniform telemetry
- **Knowledge** (SQLite + sqlite-vec, `knowledge` table) — Unified store replacing former `user_traits`, `semantic_concepts`, `procedural_memory` tables; stores traits, facts, procedures, preferences, rules, metrics; RRF hybrid search (exact + FTS5 + vector KNN)
- **Tool Calls** (SQLite, `tool_calls` table) — Per-turn tool invocations with `ephemeral` flag; non-ephemeral calls reconstructed into context window by `context_window_service`
- **Lists** (SQLite) — Deterministic ground-truth state (shopping, to-do, chores); perfect recall, no decay, full event history

Each layer optimized for its timescale. Conversation history is reconstructed by `context_window_service.build_messages()` on every call. Per-turn user additions (world state, episodic recall, voice guard) assembled by `UserPromptAssemblyService`. Lists injected into all prompts as `{{active_lists}}`.

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
- **User tool loop**: 30 max iterations, 15 min cumulative timeout; per-iteration synthesis sent via WebSocket
- **DMN tool loop**: 15 max iterations, 5 min timeout
- **Goal pursuit**: 50 max iterations, 2h wall-clock timeout; no concurrency cap, no state machine

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
- **`system`** — Health, version, settings, observability (routing, memory, tools, identity, tasks, autobiography, traits)
- **`tools`** — Tool execution, configuration
- **`providers`** — LLM provider configuration
- **`push`** — Push notification subscription
- **`scheduler`** — Reminders and scheduled tasks
- **`lists`** — List management
- **`stubs`** — Placeholder endpoints (calendar, notifications, integrations, permissions) returning 501

### Observability Endpoints (`/system/observability/*`)
- **`routing`** — Mode router decision distribution and recent activity
- **`memory`** — Memory layer counts and health indicators
- **`tools`** — Tool performance stats
- **`identity`** — Identity vector states
- **`tasks`** — Active goal pursuit threads
- **`autobiography`** — Current autobiography narrative with delta (changed/unchanged sections)
- **`traits`** (GET) — User traits grouped by category with confidence scores
- **`traits/<key>`** (DELETE) — Remove a specific learned trait (user correction)

See API blueprints in `backend/api/` for full reference.

## Testing Strategy

### Test Markers
- `@pytest.mark.unit` — No external dependencies (fast)
- `@pytest.mark.integration` — Requires SQLite/MemoryStore (slower)

### Test Organization
```
backend/tests/
├── test_services/         # Service unit tests
├── test_workers/          # Worker integration tests
└── fixtures/              # Shared test fixtures
```

Run all tests: `pytest`
Run only unit: `pytest -m unit`
Run with verbose: `pytest -v`

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
- `migrations/012_interfaces.sql` — Database schema

## Glossary

- **MessageProcessor**: Abstract base class for all LLM turns. Defines the tool-calling loop, transcript persistence, and context window reconstruction. Subclasses: `UserMessageProcessor`, `DMNMessageProcessor`, `ScheduledMessageProcessor`, `GoalPursuitProcessor`.
- **Channel**: Stable string identifier scoping a conversation context in the `transcript` and `compactions` tables. Replaced the former topic/thread distinction.
- **Block Protocol**: Universal content format — all LLM-to-client content is JSON arrays of typed block objects. `blocks_render_service.py` (backend) → `blocks.js` (frontend). No HTML over the wire.
- **DMN (Default Mode Network)**: Timer-based proactive intelligence; fires after 60min idle (recent context) and every 6h (salience); uses `DMNMessageProcessor`; exits on `DMN_NO_ACTION`.
- **GoalPursuitProcessor**: Background `MessageProcessor` subclass. Runs a single goal string for up to 50 iterations / 2h. Channel-isolated, surfaces result as a proactive message.
- **Episode**: Transcript-linked narrative memory unit with `storage_strength` (never decreases) and `retrieval_weight` (power-law decay); created by rolling transcript trigger; consolidates into super episodes.
- **Dynamic Radius**: Per-turn system-determined retrieval radius for episodic recall: `effective = baseline × narrow_factor × expand_factor`. Narrow on redundancy, expand on drift. Never exposed to the LLM.
- **memory_recall_log**: Telemetry table recording every episodic retrieval (caller, radius components, candidate counts). Read by meta-harness for tuning the 8 radius constants.
- **Knowledge**: Unified SQLite store (`knowledge` table) for traits, facts, procedures, preferences, rules, metrics — replacing former separate `user_traits`, `semantic_concepts`, `procedural_memory` tables.
- **Mode Router**: Deterministic mathematical function selecting engagement mode from observable signals. **Only consulted for non-user flows** (DMN fallback). User turns bypass it entirely.
- **Context Warmth**: Signal (0.0-1.0) measuring how much context is available for the current channel.
- **Salience**: Computed importance metric (0.1-1.0) based on novelty, emotion, commitment.
