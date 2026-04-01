# System Architecture

## Overview

Chalie is a **persistent cognitive agent** — a continuously running cognitive runtime, not a request-response service. Intelligence emerges from experience: every interaction flows through a memory pipeline that compresses, abstracts, and decays information over time. The system is not a chatbot or assistant wrapper; it is a cognitive runtime that forms memories, runs background processes, exercises judgment, and diverges into a unique identity shaped by its interaction history.

## Core Architecture

### System Type
- **Synthetic cognitive brain** using LLMs to replicate human brain functions
- **Tech Stack**: Python backend, SQLite (WAL mode + sqlite-vec + FTS5), MemoryStore (in-memory, thread-safe), Ollama (configurable LLMs), Vanilla JavaScript frontend (Radiant design system)
- **Core Pattern**: Single-process architecture with daemon threads and PromptQueue, service-oriented design

### Communication Pattern
1. Client connects to `/ws` (WebSocket via flask-sock)
2. Client sends `{"type": "message", "text": "..."}` JSON frame
3. Backend enqueues message → Digest Worker processes → response streamed back as WebSocket frames (`status` → `message` → `done`)
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
│   ├── markdown.js    # Markdown parser (XSS-safe)
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
- **`mode_router_service.py`** — Deterministic mode routing (~5ms) with signal collection + tie-breaker
- **`routing_stability_regulator_service.py`** — Single authority for router weight mutation (24h cycle, ±0.02/day max)
#### Response Generation
- **`frontal_cortex_service.py`** — LLM response generation using mode-specific prompts
- **`voice_mapper_service.py`** — Translates identity vectors to tone instructions

#### Memory System
- **`context_assembly_service.py`** — Unified retrieval from multiple memory layers (transcript + compaction, moments, episodes, knowledge) with weighted budget allocation; working memory now reads from compaction summary + budget-constrained recent transcript entries instead of fixed-size MemoryStore FIFO
- **`transcript_service.py`** — Persistent, topic-scoped, append-only conversation record (SQLite + sqlite-vec); semantic search, keyword fallback, selective embedding (>50 tokens), 90-day TTL pruning
- **`compaction_service.py`** — Incremental LLM-powered summarization; fires when total context exceeds 85% of provider budget; stores compacted text with transcript watermark in `topic_compactions`
- **`episodic_retrieval_service.py`** — Hybrid vector + FTS search for episodes
- **`knowledge_service.py`** — Unified knowledge store (traits, concepts, procedures, relationships) with RRF hybrid search (exact + FTS5 + vector KNN), decay management, and prompt injection
- **`temporal_pattern_service.py`** — Mines hour-of-day and day-of-week distributions from `interaction_log` for behavioral pattern detection; stores discoveries as behavioral traits in knowledge table with generalized labels; 24h background worker cycle
- **`episodic_storage_service.py`** — SQLite CRUD for episodic memories
- **`list_service.py`** — Deterministic list management (shopping, to-do, chores); perfect recall with full history via `lists`, `list_items`, `list_events` tables
- **`moment_service.py`** — Pinned message bookmarks with LLM-enriched context, sqlite-vec semantic search, and salience boosting; moments stored as documents with `source_type='moment'`
- **`moment_enrichment_service.py`** — Background worker (5min poll): collects gists from ±4hr interaction window, generates LLM summaries, seals moments after 4hrs

#### Autonomous Behavior
- **`reasoning_loop_service.py`** — Signal-driven continuous reasoning; dispatches signal types through handlers; falls back to salient/insight discovery on idle; attention-gated
- **`autonomous_actions/`** — Decision routing by priority: CommunicateAction (10), PlanAction (7), AmbientToolAction (6), ReflectAction (5), ReconcileAction (4), NothingAction (-1)
- **`decay_engine_service.py`** — Periodic decay (episodic 0.05/hr, semantic 0.03/hr)

#### Ambient Awareness
- **`ambient_inference_service.py`** — Deterministic inference engine (<1ms, zero LLM): place, attention, energy, mobility, tempo, device_context from browser telemetry + behavioral signals; thresholds loaded from `configs/agents/ambient-inference.json`; emits transition events (place, attention, energy) to event bridge when `emit_events=True`
- **`place_learning_service.py`** — Accumulates place fingerprints (geohash ~1km, never raw coords) in `place_fingerprints` table; learned patterns override heuristics after 20+ observations
- **`client_context_service.py`** — Rich client context with location history ring buffer (12 entries), place transition detection, session re-entry detection (>30min absence), demographic trait seeding from locale, and circadian hourly interaction counts; emits session_start/session_resume events to event bridge

- **`event_bridge_service.py`** — Connects ambient context changes (place, attention, energy, session) to autonomous actions; enforces stabilization windows (90s), per-event cooldowns, confidence gating, aggregation (60s bundle window), and focus gates; config in `configs/agents/event-bridge.json`

#### ACT Loop & Critic
- **`act_orchestrator_service.py`** — Unified, parameterized ACT loop runner. Configurable: `critic_enabled`, `smart_repetition` (embedding-based), `escalation_hints`, `persistent_task_exit`, `deferred_card_context`. Fatigue-free termination model: hard cap (30 iterations), cumulative timeout, semantic repetition, type repetition, no-actions signal.
- **`act_loop_service.py`** — Cognitive iteration manager with action execution, history tracking, and telemetry. Constructor-injected critic and dispatcher. Generic scalar output chaining between sequential actions.
- **`act_dispatcher_service.py`** — Routes actions to skill handlers with timeout enforcement; returns structured results with confidence and contextual notes
- **`critic_service.py`** — Post-action verification via lightweight LLM; safe actions get silent correction, consequential actions pause; EMA-based confidence calibration
- **`act_reflection_service.py`** — Enqueues tool outputs for background experience assimilation
- **`persistent_task_service.py`** — Multi-session background task management with state machine (ACCEPTED → IN_PROGRESS → COMPLETED/PAUSED/CANCELLED/EXPIRED); duplicate detection via Jaccard similarity; rate limiting (3 cycles/hr, 5 active tasks max)
- **`plan_decomposition_service.py`** — LLM-powered goal → step DAG decomposition; validates DAG (Kahn's cycle detection), step quality (4–30 word descriptions, Jaccard dedup), and cost classification (cheap/expensive); plans stored in `persistent_tasks.progress` JSON (stored as TEXT in SQLite); ready-step ordering (shallowest depth, cheapest first)

#### Constants & Registries
- **`services/innate_skills/registry.py`** — Authoritative frozenset definitions for all skill membership sets (`ALL_SKILL_NAMES`, `PLANNING_SKILLS`, `COGNITIVE_PRIMITIVES`, `CONTEXTUAL_SKILLS`, `TRIAGE_VALID_SKILLS`, etc.). Single source of truth — all consumers import from here.
- **`services/act_action_categories.py`** — Authoritative frozenset definitions for action behavior categories (`READ_ACTIONS`, `DETERMINISTIC_ACTIONS`, `SAFE_ACTIONS`, `CRITIC_SKIP_READS`, `ACTION_FATIGUE_COSTS`).
- **`services/act_memory_keys.py`** — Centralized MemoryStore key patterns for the ACT system (deferred cards, tool caches, heartbeat, reflection queue).

#### Tool Integration
- **`tool_registry_service.py`** — Tool discovery, metadata management; loads first-party tools from ToolLibraryService, registers interface tools via HTTP; invokes first-party tools directly in-process
- **`tool_config_service.py`** — Tool configuration persistence; webhook key generation (HMAC-SHA256 + replay protection via X-Chalie-Signature/X-Chalie-Timestamp)
- **`tool_performance_service.py`** — Performance metrics tracking; correctness-biased ranking (50% success_rate, 15% speed, 15% reliability, 10% cost, 10% preference); post-triage tool reranking; user correction propagation; 30-day preference decay
- **`tool_profile_service.py`** — LLM-generated tool capability profiles with `short_summary`, `full_profile`, `triage_triggers`, and `usage_scenarios`; profiles power the `find_tools` innate skill (semantic search against capability embeddings in `tool_capability_profiles_vec`)
- **Webhook endpoint** (`/api/tools/webhook/<name>`) — External tool triggers with HMAC-SHA256 or simple token auth, 30 req/min rate limit, 512KB payload cap

#### Identity & Learning
- **`identity_service.py`** — 6-dimensional identity vector system with coherence constraints
- **`identity_state_service.py`** — Tracks identity state changes and evolution

#### Infrastructure
- **`database_service.py`** — SQLite connection management (WAL mode) and migrations
- **`memory_store.py`** — MemoryStore: thread-safe, in-memory key-value store with Redis-compatible API
- **`config_service.py`** — JSON file config loader (agent configs, connection names); runtime config (port, host) managed by `runtime_config.py` via CLI args
- **`output_service.py`** — Output queue management for responses
- **`event_bus_service.py`** — Pub/sub event routing

#### Topic Classification
- **`topic_classifier_service.py`** — Embedding-based deterministic topic classification with two-signal boundary detection
- **`two_signal_boundary_service.py`** — Self-calibrating topic boundary detector: consecutive + window similarity must both drop below adaptive thresholds (K=1.6), with discourse marker fast path; per-thread state in MemoryStore with 24h TTL

#### Session & Conversation
- **`thread_conversation_service.py`** — MemoryStore-backed conversation thread persistence
- **`thread_service.py`** — Manages conversation threads with expiry
- **`session_service.py`** — Tracks user sessions and topic changes

#### Documents & File Management
- **`document_service.py`** — Document CRUD, chunk storage, hybrid search (semantic via sqlite-vec + FTS5 + keyword boost via Reciprocal Rank Fusion), soft delete with 30-day purge window, dual-layer duplicate detection (SHA-256 hash + cosine similarity on summary embeddings)
- **`document_processing_service.py`** — Full extraction pipeline: text extraction (pdfplumber, python-docx, python-pptx, trafilatura), regex-based metadata extraction (dates, companies, monetary values, reference numbers, document type heuristic), adaptive chunk sizing by document type, SimHash fingerprinting, language detection (langdetect)
- **`document_card_service.py`** — Inline HTML card emission for document search results (source attribution with type badges, confidence indicators), upload confirmations, document previews, and lifecycle events; cyan `#00F0FF` accent

### Innate Skills (`backend/services/innate_skills/` and `backend/skills/`)

Built-in cognitive skills for the ACT loop:
- **`recall_skill.py`** — Unified retrieval across ALL memory layers including user traits (<500ms); supports "what do you know about me?" via `user_traits` layer with broad/specific query modes and confidence labels
- **`memorize_skill.py`** — Store gists and facts (<50ms)
- **`introspect_skill.py`** — Comprehensive internal state report: 4 natural-language scopes (memory health, skill/tool usage, reasoning state, identity); supports "why did you do that?" via routing audit trail and autonomous action history
- **`associate_skill.py`** — Spreading activation through semantic graph (<500ms)
- **`scheduler_skill.py`** — Create/list/cancel reminders and scheduled tasks (<100ms)
- **`autobiography_skill.py`** — Retrieve synthesized user narrative with optional section extraction (<500ms)
- **`list_skill.py`** — Deterministic list management: add/remove/check items, view, history (<50ms)
- **`focus_skill.py`** — Focus session management: set, check, clear with distraction detection (<50ms)
- **`persistent_task_skill.py`** — Multi-session background task management: create (with plan decomposition), pause, resume, cancel, check status, show plan, set priority (<100ms; create ~2-5s with LLM decomposition)
- **`document_skill.py`** — Document search and management via ACT loop: search (hybrid semantic via sqlite-vec + FTS5 + keyword retrieval), list, view, delete, restore; documents are reference material retrieved via skill, not context assembly; search results include `[Source: document_id=...]` markers for frontal cortex citation
- **`read_skill.py`** — Fetch and read web page content for information gathering and research
- **`reflect_skill.py`** — On-demand experiential synthesis via lightweight LLM call; retrieves ACT loop outcomes, episodes, concepts, and strategy patterns, then synthesizes into actionable insight (what worked, what didn't, patterns noticed, connections formed); optionally stores as gist
- **`find_tools_skill.py`** — Discover registered tools via semantic search against tool capability profiles
- **`notes_skill.py`** — Search past conversation transcript for on-demand retrieval of older context (renamed to `transcript` skill; `notes` alias preserved for backward compat)

## Worker Processes (`backend/workers/`)

### Queue Workers (Daemon Threads)
- **Digest Worker** — Core pipeline: classify → unified generate → enqueue memory job
- **Episodic Memory Worker** — Builds episodes from sequences of exchanges
- **Semantic Consolidation Worker** — Extracts concepts + relationships from episodes

### Services/Daemons (Daemon Threads)
- **REST API + WebSocket** — Flask app with flask-sock on port 8081
- **Reasoning Loop** — Signal-driven continuous reasoning (see service listing above); attention-gated
- **Ambient Inference Service** — Deterministic inference of place, attention, energy, mobility, tempo from browser telemetry (<1ms, zero LLM)
- **Place Learning Service** — Accumulates place fingerprints in SQLite; learned patterns override heuristics after 20+ observations
- **Decay Engine** — Periodic memory decay cycle; applies reliability multipliers so unreliable memories decay faster
- **Routing Stability Regulator** — Single authority for router weight mutation
- **Experience Assimilation** — Tool results → episodic memory (60s poll)
- **Thread Expiry Service** — Expires stale threads (5min cycle)
- **Scheduler Service** — Fires due reminders/tasks (60s poll)
- **Autobiography Synthesis** — Synthesizes user narrative (6h cycle)
- **Triage Calibration** — Triage correctness scoring (24h cycle); wires user corrections to tool preferences; learns usage scenarios from clarification→tool resolution chains
- **Profile Enrichment** — Tool profile enrichment (6h cycle, 3 tools/cycle); preference decay; usage-triggered full profile rebuilds (15 successes or reliability < 50%)
- **Curiosity Pursuit** — Explores curiosity threads via ACT loop (6h cycle)
- **Moment Enrichment** — Enriches pinned moments with gists + LLM summary, seals after 4hrs (5min poll)
- **Temporal Pattern Service** — Mines behavioral patterns from interaction timestamps (24h cycle, 5min warmup); detects hour-of-day peaks, day-of-week peaks, topic-time clusters; stores as `behavioral_pattern` user traits
- **Persistent Task Worker** — Runs eligible multi-session background tasks via bounded ACT loop (30min cycle with ±30% jitter); plan-aware execution follows step DAG when present (up to 3 steps/cycle with per-step fatigue budgets), falls back to flat loop otherwise; adaptive user surfacing at coverage milestones
- **Document Worker** — PromptQueue worker for document processing: text extraction → metadata extraction → adaptive chunking → batch embedding → storage; 10min timeout per document
- **Document Purge Service** — Hard-deletes documents past their 30-day soft-delete window (6h cycle)
- **VaultService** — AES-256-GCM envelope encryption; PBKDF2-derived KEK wraps a random DEK stored in ``vault_config``; unlocked post-login; migrates legacy Fernet data on first unlock

## Data Flow Pipeline

### User Input → Response Pipeline
```
[User Input]
  → [run.py] → [PromptQueue] → [Digest Worker]
    ├─ Classification (embedding-based, two-signal boundary detection)
    ├─ Context Assembly (transcript + compaction + semantic memories)
    ├─ Unified LLM Generation (skills + tools discoverable inline)
    │  └─ ACT loop runs inline when LLM invokes skills/tools
    └─ Enqueue Memory Job
      → [Episodic Memory Queue] → [Episodic Memory Worker]
        → SQLite Episodes Table
        → [Semantic Consolidation Queue] → [Semantic Consolidation Worker]
          → SQLite Concepts Table
```

### Background Processes
```
[Routing Stability Regulator] ← 24h cycle
    → adjusts configs/generated/mode_router_config.json

[Decay Engine] → runs every 1800s (30min)
    ├─ Episodic decay (salience-weighted)
    ├─ Semantic decay (strength-weighted)
    └─ User trait decay (category-specific)

[Cognitive Drift Engine] → during worker idle
    ├─ Seed selection (weighted random)
    ├─ Spreading activation (depth 2, decay 0.7/level)
    └─ LLM synthesis → stores as drift gist
```

## Key Architectural Decisions

### Deterministic Mode Router
- **Decoupled**: Mode selection (mathematical, ~5ms) separate from response generation (LLM, ~2-15s)
- **Signals**: ~17 observable signals from context + NLP (context warmth, question marks, greeting patterns, etc.)
- **Scores**: Each mode gets weighted composite score; highest wins
- **Tie-breaker**: ONNX classifier for ambiguous cases
- **Self-leveling**: Router naturally shifts toward UNIFIED as memory accumulates

### Single Authority for Weight Mutation
- **Routing Stability Regulator** is the only service that modifies router weights
- Other services log "pressure signals" but don't mutate state
- Updates bounded: max ±0.02/day, 48h cooldown per parameter
- **Closed-loop control**: Verifies adjustments work before persisting

### Mode-Specific Prompts
- Each mode (UNIFIED, ACT, IGNORE) has its own focused prompt template
- Replaces old approach: single combined prompt with mode selection embedded
- Focused scope prevents elaboration and improves consistency

### Memory Hierarchy
- **Topic Transcript** (SQLite + sqlite-vec) — Persistent, append-only conversation record per topic; budget-aware filling replaces fixed turn limits
- **Compaction** (SQLite) — Incremental LLM summarization of older transcript entries; preserves facts/decisions/preferences, discards conversation flow
- **Working Memory** (MemoryStore, legacy fallback) — FIFO buffer used only when no transcript data exists yet
- **Gists** (MemoryStore, 30min TTL) — Compressed exchange summaries
- **Facts** (MemoryStore, 24h TTL) — Atomic key-value assertions
- **Episodes** (SQLite + sqlite-vec) — Narrative units with decay
- **Concepts** (SQLite + sqlite-vec) — Knowledge nodes and relationships
- **Procedural Memory** (SQLite) — Learned action reliability; surfaced in context assembly as reliability hints (≥8 attempts, top 3 skills)
- **User Traits** (SQLite) — Personal facts with category-specific decay (includes behavioral patterns from temporal mining)
- **Lists** (SQLite) — Deterministic ground-truth state (shopping, to-do, chores); perfect recall, no decay, full event history

Each layer optimized for its timescale; all integrated via context assembly. Context assembly reads compaction summary + budget-constrained recent transcript entries for working memory context. Lists are injected into all prompts as `{{active_lists}}` for passive awareness; the ACT loop uses the `list` skill for mutations.

### Configuration Precedence
```
Environment variables > .env file > JSON config files > hardcoded defaults
```

See `docs/02-PROVIDERS-SETUP.md` for provider configuration.

### Thread-Safe Worker State
- All workers run as daemon threads within a single Python process
- Shared state managed via thread-safe data structures (locks, queues)
- No multiprocessing overhead — lightweight, in-process coordination

### Adaptive Topic Boundary Detection
- Replaces static 0.65 cosine similarity threshold with a 3-layer self-calibrating detector
- **Two-Signal Detection**: consecutive similarity (cos to previous message) AND window similarity (cos to centroid of last 5 messages) must both drop below self-calibrating thresholds (mean - K*std, K=1.6) for a boundary to fire
- **Discourse Markers**: 16 regex patterns for explicit topic switch phrases ("by the way", "speaking of", etc.) provide a high-precision fast path bypassing the two-signal gate
- All thresholds derived from running conversation statistics; stats only updated on non-boundary messages to keep baseline clean
- State persisted in MemoryStore (`two_signal_boundary:{thread_id}`, 24h TTL); cold-start mode (markers only) during first 6 messages

### Topic Confidence Reinforcement
- Topic confidence updated via bounded reinforcement formula
- `new = current + (new_confidence - current) * 0.5`
- Ensures gradual adaptation without oscillation

### Error Resilience
- All workers catch JSON decode errors from LLM responses
- Log meaningful messages instead of crashing
- Return status strings for graceful degradation

## Safety & Constraints

### Hard Boundaries
- **Prompt hierarchy** immutable (marked as "authoritative and final")
- **Skill registry** fixed at startup (no runtime skill registration)
- **Data scope** parameterized by topic (no cross-topic leakage)
- **Speaker confidence** gates trait storage (unknown speakers = 0.3 penalty)

### Operational Limits
- **ACT loop**: 60s cumulative timeout, 30 max iterations; post-action critic verification
- **Persistent tasks**: 5 active max, 3 cycles/hr rate limit, 14-day auto-expiry; plan-decomposed tasks: 3–8 steps per plan, up to 3 steps executed per cycle, 3 ACT iterations per step
- **Fatigue budget**: 2.5 activation units per 30min
- **Per-concept cooldown**: 60min (prevents circular rumination)
- **Delegation rate**: 1 per topic per 30min

### Anti-Manipulation
- **Identity isolation**: 6 vectors with coherence constraints
- **No vulnerability simulation**: Explicitly forbidden
- **Exponential backoff**: System retreats on silence (opposite of dependency)
- **No flattery optimization**: Soul axiom: "Never optimize by misleading"

## Configuration Files

### Primary Configuration
- **`configs/connections.json`** — SQLite path and MemoryStore settings
- **`configs/agents/*.json`** — LLM settings (model, temperature, timeout)
- **`configs/generated/mode_router_config.json`** — Learned router weights (generated)

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
- **`tasks`** — Active persistent tasks, curiosity threads, triage calibration
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

- **Mode Router**: Deterministic mathematical function selecting engagement mode from observable signals
- **Tie-Breaker**: ONNX classifier consulted when top 2 modes are within effective margin
- **Routing Signals**: Observable features collected from MemoryStore and NLP analysis (~5ms)
- **Router Confidence**: Normalized gap between top 2 scores — measures routing certainty
- **Pressure Signal**: Metric logged by monitors, consumed by the single regulator
- **Context Warmth**: Signal (0.0-1.0) measuring how much context is available for current topic
- **Drift Gist**: Spontaneous thought stored during idle periods (DMN)
- **Episode**: Narrative memory unit with intent, context, action, emotion, outcome, salience
- **Concept**: Knowledge node with strength decay and spreading activation
- **Salience**: Computed importance metric (0.1-1.0) based on novelty, emotion, commitment
