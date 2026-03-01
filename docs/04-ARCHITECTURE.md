# System Architecture

## Overview

Chalie is a human-in-the-loop cognitive assistant that combines memory consolidation, semantic reasoning, and proactive assistance. The system processes user prompts through a chain of workers and services, enriching conversations with memory chunks and generating episodic memories for future use.

## Core Architecture

### System Type
- **Synthetic cognitive brain** using LLMs to replicate human brain functions
- **Tech Stack**: Python backend, PostgreSQL + pgvector, Redis, Ollama (configurable LLMs), Vanilla JavaScript frontend (Radiant design system)
- **Core Pattern**: Worker-based architecture with Redis queue, service-oriented design

### Communication Pattern
1. User sends message → POST to `/chat` with text
2. Backend processes: Mode router selects mode → mode-specific LLM generates response
3. Response delivered: Via SSE stream (status → message → done events)
4. Authentication: Session cookie-based authentication (`@require_session` decorator)

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
├── tools/             # Skill implementations
├── tests/             # Test suite
└── consumer.py        # Main supervisor process
```

Frontend applications located separately:
```
frontend/
├── interface/         # Main chat UI (HTML/CSS/JS, Radiant design system)
├── brain/             # Admin/cognitive dashboard
└── on-boarding/       # Account setup wizard
```

**IMPORTANT**: UI code must exist under `/interface/`, `/brain/`, or `/on-boarding/` only.

## Key Services

### Core Services (`backend/services/`)

#### Routing & Decision Making
- **`mode_router_service.py`** — Deterministic mode routing (~5ms) with signal collection + tie-breaker
- **`routing_decision_service.py`** — Routing decision audit trail (PostgreSQL)
- **`routing_stability_regulator_service.py`** — Single authority for router weight mutation (24h cycle, ±0.02/day max)
- **`routing_reflection_service.py`** — Idle-time peer review of routing decisions via strong LLM
- **`cognitive_triage_service.py`** — LLM-based 4-step triage (social filter → LLM → self-eval → dispatch); routes to RESPOND/ACT/CLARIFY/ACKNOWLEDGE; defers tool selection to ACT loop when tools exist but none named
- **`cognitive_reflex_service.py`** — Learned fast path via semantic abstraction; heuristic pre-screen (~1ms) + pgvector cluster lookup (~5-20ms) bypasses full pipeline for self-contained queries; rolling-average centroids generalize from observed examples; self-correcting per cluster via user corrections and shadow validation

#### Response Generation
- **`frontal_cortex_service.py`** — LLM response generation using mode-specific prompts
- **`voice_mapper_service.py`** — Translates identity vectors to tone instructions

#### Memory System
- **`context_assembly_service.py`** — Unified retrieval from 6 memory layers (working memory, moments, facts, gists, episodes, procedural, concepts) with weighted budget allocation; procedural hints surface learned action reliability (≥8 attempts, top 3, confidence labels)
- **`episodic_retrieval_service.py`** — Hybrid vector + FTS search for episodes
- **`semantic_retrieval_service.py`** — Vector similarity + spreading activation for concepts
- **`user_trait_service.py`** — Per-user trait management with category-specific decay (core, relationship, physical, preference, communication_style, micro_preference, behavioral_pattern)
- **`temporal_pattern_service.py`** — Mines hour-of-day and day-of-week distributions from `interaction_log` for behavioral pattern detection; stores discoveries as `behavioral_pattern` user traits with generalized labels; 24h background worker cycle
- **`episodic_storage_service.py`** — PostgreSQL CRUD for episodic memories
- **`semantic_storage_service.py`** — PostgreSQL CRUD for semantic concepts
- **`gist_storage_service.py`** — Redis-backed short-term memory with deduplication
- **`list_service.py`** — Deterministic list management (shopping, to-do, chores); perfect recall with full history via `lists`, `list_items`, `list_events` tables
- **`moment_service.py`** — Pinned message bookmarks with LLM-enriched context, pgvector semantic search, and salience boosting; stores user-pinned Chalie responses as permanent, searchable moments via `moments` table
- **`moment_enrichment_service.py`** — Background worker (5min poll): collects gists from ±4hr interaction window, generates LLM summaries, seals moments after 4hrs; boosts related episode salience on seal
- **`moment_card_service.py`** — Inline HTML card emission for moment display in the conversation spine

#### Autonomous Behavior
- **`cognitive_drift_engine.py`** — Default Mode Network (DMN) for spontaneous thoughts during idle; attention-gated (skips when user in deep focus)
- **`autonomous_actions/`** — Decision routing (priority 10→6): CommunicateAction, SuggestAction (skill-matched proactive suggestions), NurtureAction (gentle phase-appropriate presence), PlanAction (proactive plan proposals from recurring topics, 7-gate eligibility with signal persistence), ReflectAction, SeedThreadAction
- **`spark_state_service.py`** — Tracks relationship phase progression (first_contact → surface → exploratory → connected → graduated)
- **`spark_welcome_service.py`** — First-contact welcome message triggered on first SSE connection; runs once per lifecycle
- **`curiosity_thread_service.py`** — Self-directed exploration threads (learning and behavioral) seeded from cognitive drift
- **`curiosity_pursuit_service.py`** — Background worker exploring active threads via ACT loop
- **`decay_engine_service.py`** — Periodic decay (episodic 0.05/hr, semantic 0.03/hr)

#### Ambient Awareness
- **`ambient_inference_service.py`** — Deterministic inference engine (<1ms, zero LLM): place, attention, energy, mobility, tempo, device_context from browser telemetry + behavioral signals; thresholds loaded from `configs/agents/ambient-inference.json`; emits transition events (place, attention, energy) to event bridge when `emit_events=True`
- **`place_learning_service.py`** — Accumulates place fingerprints (geohash ~1km, never raw coords) in `place_fingerprints` table; learned patterns override heuristics after 20+ observations
- **`client_context_service.py`** — Rich client context with location history ring buffer (12 entries), place transition detection, session re-entry detection (>30min absence), demographic trait seeding from locale, and circadian hourly interaction counts; emits session_start/session_resume events to event bridge
- **`event_bridge_service.py`** — Connects ambient context changes (place, attention, energy, session) to autonomous actions; enforces stabilization windows (90s), per-event cooldowns, confidence gating, aggregation (60s bundle window), and focus gates; config in `configs/agents/event-bridge.json`

#### ACT Loop & Critic
- **`act_orchestrator_service.py`** — Unified, parameterized ACT loop runner. Single implementation replaces per-worker loop copies. Configurable: `critic_enabled`, `smart_repetition` (embedding-based), `escalation_hints` (budget warnings), `persistent_task_exit`, `deferred_card_context`. Caller-specific behavior via `on_iteration_complete` callback. Config flag `act_use_unified_orchestrator` for gradual rollout.
- **`act_loop_service.py`** — Fatigue-based cognitive iteration manager with action execution, history tracking, and telemetry. Constructor-injected critic and dispatcher (no monkey-patching). Generic scalar output chaining between sequential actions.
- **`act_dispatcher_service.py`** — Routes actions to skill handlers with timeout enforcement; returns structured results with confidence and contextual notes
- **`critic_service.py`** — Post-action verification: evaluates each action result for correctness via lightweight LLM (reuses `cognitive-triage` agent config); safe actions get silent correction, consequential actions pause; EMA-based confidence calibration
- **`act_completion_service.py`** — Detects when expected tools were not invoked; injects `[NO_ACTION_TAKEN]` signal
- **`act_reflection_service.py`** — Enqueues tool outputs for background experience assimilation
- **`persistent_task_service.py`** — Multi-session background task management with state machine (PROPOSED → ACCEPTED → IN_PROGRESS → COMPLETED/PAUSED/CANCELLED/EXPIRED); duplicate detection via Jaccard similarity; rate limiting (3 cycles/hr, 5 active tasks max)
- **`plan_decomposition_service.py`** — LLM-powered goal → step DAG decomposition; validates DAG (Kahn's cycle detection), step quality (4–30 word descriptions, Jaccard dedup), and cost classification (cheap/expensive); plans stored in `persistent_tasks.progress` JSONB; ready-step ordering (shallowest depth, cheapest first)

#### Constants & Registries
- **`services/innate_skills/registry.py`** — Authoritative frozenset definitions for all skill membership sets (`ALL_SKILL_NAMES`, `PLANNING_SKILLS`, `COGNITIVE_PRIMITIVES`, `CONTEXTUAL_SKILLS`, `TRIAGE_VALID_SKILLS`, etc.). Single source of truth — all consumers import from here.
- **`services/act_action_categories.py`** — Authoritative frozenset definitions for action behavior categories (`READ_ACTIONS`, `DETERMINISTIC_ACTIONS`, `SAFE_ACTIONS`, `CRITIC_SKIP_READS`, `ACTION_FATIGUE_COSTS`).
- **`services/act_redis_keys.py`** — Centralized Redis key patterns for the ACT system (deferred cards, tool caches, heartbeat, SSE, reflection queue).

#### Tool Integration
- **`tool_registry_service.py`** — Tool discovery, metadata management via `run_interactive` (bidirectional stdin/stdout dialog protocol)
- **`tool_output_utils.py`** — Shared tool output formatting (`format_tool_result`), sanitization (`sanitize_tool_output`), and telemetry building (`build_tool_telemetry`); used by both ToolRegistryService and CronToolWorker
- **`tool_card_enqueue_service.py`** — Post-loop tool card rendering and delivery with dedup guards (B7/B8 fixes)
- **`tool_container_service.py`** — Container lifecycle; `run()` for single-shot, `run_interactive()` for bidirectional tool↔Chalie dialog (JSON-lines stdout, Chalie responses via stdin)
- **`tool_config_service.py`** — Tool configuration persistence; webhook key generation (HMAC-SHA256 + replay protection via X-Chalie-Signature/X-Chalie-Timestamp)
- **`tool_performance_service.py`** — Performance metrics tracking; correctness-biased ranking (50% success_rate, 15% speed, 15% reliability, 10% cost, 10% preference); post-triage tool reranking; user correction propagation; 30-day preference decay
- **`tool_profile_service.py`** — LLM-generated tool capability profiles with `triage_triggers` (short action verbs injected into triage prompt for vocabulary bridging), `short_summary`, `full_profile`, and `usage_scenarios`; Redis-cached triage summaries (5min TTL)
- **Webhook endpoint** (`/api/tools/webhook/<name>`) — External tool triggers with HMAC-SHA256 or simple token auth, 30 req/min rate limit, 512KB payload cap

#### Identity & Learning
- **`identity_service.py`** — 6-dimensional identity vector system with coherence constraints
- **`identity_state_service.py`** — Tracks identity state changes and evolution
- **`user_trait_service.py`** — User trait management with category-specific decay

#### Infrastructure
- **`database_service.py`** — PostgreSQL connection pool and migrations
- **`redis_client.py`** — Redis connection handling
- **`config_service.py`** — Environment and JSON file config (precedence: env > .env > json)
- **`output_service.py`** — Output queue management for responses
- **`event_bus_service.py`** — Pub/sub event routing
- **`card_renderer_service.py`** — Card system rendering engine

#### Topic Classification
- **`topic_classifier_service.py`** — Embedding-based deterministic topic classification with adaptive boundary detection
- **`adaptive_boundary_detector.py`** — 3-layer self-calibrating topic boundary detector (NEWMA + Transient Surprise + Leaky Accumulator); persists per-thread state in Redis; degrades gracefully to static threshold when Redis is unavailable
- **`topic_stability_regulator_service.py`** — 24h adaptive tuning of topic classification and boundary detector parameters

#### Session & Conversation
- **`thread_conversation_service.py`** — Redis-backed conversation thread persistence
- **`thread_service.py`** — Manages conversation threads with expiry
- **`session_service.py`** — Tracks user sessions and topic changes

#### Documents & File Management
- **`document_service.py`** — Document CRUD, chunk storage, hybrid search (semantic + FTS + keyword boost via Reciprocal Rank Fusion), soft delete with 30-day purge window, dual-layer duplicate detection (SHA-256 hash + cosine similarity on summary embeddings)
- **`document_processing_service.py`** — Full extraction pipeline: text extraction (pdfplumber, python-docx, python-pptx, trafilatura), regex-based metadata extraction (dates, companies, monetary values, reference numbers, document type heuristic), adaptive chunk sizing by document type, SimHash fingerprinting, language detection (langdetect)
- **`camera_ocr_service.py`** — Vision LLM-based text extraction from camera-captured images; multi-provider (Anthropic, OpenAI, Gemini, Ollama); 10MB image limit
- **`document_card_service.py`** — Inline HTML card emission for document search results (source attribution with type badges, confidence indicators), upload confirmations, document previews, and lifecycle events; cyan `#00F0FF` accent

### Innate Skills (`backend/services/innate_skills/` and `backend/skills/`)

11 built-in cognitive skills for the ACT loop:
- **`recall_skill.py`** — Unified retrieval across ALL memory layers including user traits (<500ms); supports "what do you know about me?" via `user_traits` layer with broad/specific query modes and confidence labels
- **`memorize_skill.py`** — Store gists and facts (<50ms)
- **`introspect_skill.py`** — Self-examination (context warmth, FOK signal, stats, decision explanations, recent autonomous actions) (<100ms); supports "why did you do that?" via routing audit trail and autonomous action history
- **`associate_skill.py`** — Spreading activation through semantic graph (<500ms)
- **`scheduler_skill.py`** — Create/list/cancel reminders and scheduled tasks (<100ms)
- **`autobiography_skill.py`** — Retrieve synthesized user narrative with optional section extraction (<500ms)
- **`list_skill.py`** — Deterministic list management: add/remove/check items, view, history (<50ms)
- **`focus_skill.py`** — Focus session management: set, check, clear with distraction detection (<50ms)
- **`moment_skill.py`** — Natural language moment recall ("Do you remember...") and listing via pgvector search
- **`persistent_task_skill.py`** — Multi-session background task management: create (with plan decomposition), pause, resume, cancel, check status, show plan, set priority (<100ms; create ~2-5s with LLM decomposition)
- **`document_skill.py`** — Document search and management via ACT loop: search (hybrid semantic+FTS+keyword retrieval), list, view, delete, restore; documents are reference material retrieved via skill, not context assembly; search results include `[Source: document_id=...]` markers for frontal cortex citation

## Worker Processes (`backend/workers/`)

### Queue Workers
- **Digest Worker** — Core pipeline: classify → route → generate response → enqueue memory job; fast-path ACT loop via `ACTOrchestrator` (config-flagged)
- **Tool Worker** — Background ACT loop execution (RQ queue); post-action critic verification; deferred card context injection; uses `ACTOrchestrator` (config-flagged)
- **Cron Tool Worker** (`workers/cron_tool_worker.py`) — Scheduled tool execution as background service; extracted from ToolRegistryService for SRP
- **Memory Chunker Worker** — Enriches exchanges with memory chunks via LLM
- **Episodic Memory Worker** — Builds episodes from sequences of exchanges
- **Semantic Consolidation Worker** — Extracts concepts + relationships from episodes

### Services/Daemons
- **REST API Worker** — Flask REST API on port 8080
- **Cognitive Drift Engine** — Generates spontaneous thoughts during worker idle (attention-gated: skips when user in deep focus)
- **Ambient Inference Service** — Deterministic inference of place, attention, energy, mobility, tempo from browser telemetry (<1ms, zero LLM)
- **Place Learning Service** — Accumulates place fingerprints in PostgreSQL; learned patterns override heuristics after 20+ observations
- **Decay Engine** — Periodic memory decay cycle
- **Routing Stability Regulator** — Single authority for router weight mutation
- **Routing Reflection** — Idle-time peer review of routing decisions
- **Topic Stability Regulator** — Adaptive tuning of topic classification parameters
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
- **Document Worker** — RQ queue worker (`document-queue`) for document processing: text extraction → metadata extraction → adaptive chunking → batch embedding → storage; 10min timeout per document
- **Document Purge Service** — Hard-deletes documents past their 30-day soft-delete window (6h cycle)

## Data Flow Pipeline

### User Input → Response Pipeline
```
[User Input]
  → [Consumer] → [Prompt Queue] → [Digest Worker]
    ├─ Classification (embedding-based, adaptive boundary detection)
    ├─ Context Assembly (retrieve from all 5 memory layers)
    ├─ Mode Routing (deterministic ~5ms mathematical router)
    ├─ Mode-Specific LLM Generation
    │  └─ If ACT: action loop → re-route → terminal response
    └─ Enqueue Memory Chunking Job
      → [Memory Chunker Queue] → [Memory Chunker Worker]
        → [Conversation JSON] (enriched)
      → [Episodic Memory Queue] → [Episodic Memory Worker]
        → PostgreSQL Episodes Table
        → [Semantic Consolidation Queue] → [Semantic Consolidation Worker]
          → PostgreSQL Concepts Table
```

### Background Processes
```
[Routing Stability Regulator] ← reads routing_decisions (24h cycle)
    → adjusts configs/generated/mode_router_config.json

[Routing Reflection Service] ← reads reflection-queue (idle-time)
    → writes routing_decisions.reflection → feeds pressure to regulator

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
- **Tie-breaker**: Small LLM (qwen3:4b) for ambiguous cases
- **Self-leveling**: Router naturally shifts toward RESPOND as memory accumulates

### Single Authority for Weight Mutation
- **Routing Stability Regulator** is the only service that modifies router weights
- Other services log "pressure signals" but don't mutate state
- Updates bounded: max ±0.02/day, 48h cooldown per parameter
- **Closed-loop control**: Verifies adjustments work before persisting

### Mode-Specific Prompts
- Each mode (RESPOND, CLARIFY, ACKNOWLEDGE, ACT) has its own focused prompt template
- Replaces old approach: single combined prompt with mode selection embedded
- Focused scope prevents elaboration and improves consistency

### Memory Hierarchy
- **Working Memory** (Redis, 4 turns, 24h TTL) — Current conversation
- **Gists** (Redis, 30min TTL) — Compressed exchange summaries
- **Facts** (Redis, 24h TTL) — Atomic key-value assertions
- **Episodes** (PostgreSQL + pgvector) — Narrative units with decay
- **Concepts** (PostgreSQL + pgvector) — Knowledge nodes and relationships
- **Procedural Memory** (PostgreSQL) — Learned action reliability; surfaced in context assembly as reliability hints (≥8 attempts, top 3 skills)
- **User Traits** (PostgreSQL) — Personal facts with category-specific decay (includes behavioral patterns from temporal mining)
- **Lists** (PostgreSQL) — Deterministic ground-truth state (shopping, to-do, chores); perfect recall, no decay, full event history

Each layer optimized for its timescale; all integrated via context assembly. Lists are injected into all prompts as `{{active_lists}}` for passive awareness; the ACT loop uses the `list` skill for mutations.

### Configuration Precedence
```
Environment variables > .env file > JSON config files > hardcoded defaults
```

See `docs/02-PROVIDERS-SETUP.md` for provider configuration.

### Thread-Safe Worker State
- `WorkerManager` maintains shared dictionary via `multiprocessing.Manager()`
- Workers use `WorkerBase._update_shared_state` to merge per-worker metrics
- Avoids global locks, keeps worker pool lightweight

### Adaptive Topic Boundary Detection
- Replaces static 0.65 cosine similarity threshold with a 3-layer self-calibrating detector
- **NEWMA** (fast/slow EWMA divergence) detects gradual semantic drift
- **Transient Surprise** (z-score of similarity drop) catches sharp topic shifts
- **Leaky Accumulator** provides hysteresis — single-message outliers don't create false topics
- All thresholds derived from running conversation statistics; no manual tuning
- State persisted in Redis (`adaptive_boundary:{thread_id}`, 24h TTL); cold-start fallback (0.55 threshold) when Redis unavailable or < 5 messages
- Base parameters (`accumulator_boundary_base`, `accumulator_leak_rate`, NEWMA windows) are the slow outer loop controlled by Topic Stability Regulator

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
- **ACT loop**: 60s cumulative timeout, ~7 max iterations; post-action critic verification (0.3 fatigue cost per evaluation)
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
- **`configs/connections.json`** — Redis & PostgreSQL endpoints
- **`configs/agents/*.json`** — LLM settings (model, temperature, timeout)
- **`configs/generated/mode_router_config.json`** — Learned router weights (generated)

### Provider Configuration
- Stored in PostgreSQL `providers` table (not JSON files)
- Runtime configurable via REST API (`/api/providers`)
- Supports: Ollama, Anthropic, OpenAI, Google Gemini

See `docs/02-PROVIDERS-SETUP.md` for detailed setup instructions.

## REST API

### Available Blueprints
- **`user_auth`** — Account creation, login, API key management
- **`conversation`** — Chat endpoint (SSE streaming), conversation list/retrieval
- **`memory`** — Memory search, fact management
- **`proactive`** — Outreach/notifications, upcoming tasks
- **`privacy`** — Data deletion, export
- **`system`** — Health, version, settings, observability (routing, memory, tools, identity, tasks, autobiography, traits)
- **`tools`** — Tool execution, configuration
- **`providers`** — LLM provider configuration
- **`push`** — Push notification subscription
- **`scheduler`** — Reminders and scheduled tasks
- **`lists`** — List management
- **`stubs`** — Placeholder endpoints (calendar, notifications, integrations, voice, permissions) returning 501

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
- `@pytest.mark.integration` — Requires PostgreSQL/Redis (slower)

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

### Local Development (without Docker)
```bash
# Terminal 1: PostgreSQL + Redis
# (ensure postgres + redis running locally)

# Terminal 2: Consumer (all workers)
python consumer.py

# Terminal 3: Test/debug
python -c "from api import create_app; app = create_app(); app.run()"
```

### Docker Development
```bash
docker-compose build
docker-compose up -d
docker-compose logs -f backend
```

## Deployment Notes

- **No Telemetry**: Zero external calls except to configured LLM/voice providers
- **Local First**: All data stored locally unless external providers configured
- **Encryption**: API keys and provider credentials encrypted in PostgreSQL
- **CORS**: Defaults to localhost, restrict before production
- **Default Password**: PostgreSQL password is `chalie` — **change in production**

## Future Roadmap

### Completed
- **User memory transparency API**: Observability endpoints for autobiography, traits, memory, routing, identity, tools, and tasks — with user trait deletion for correction

### Planned (Priority 1)

### Planned (Priority 2)
- **Cross-topic pattern mining**: Beyond temporal — behavioral prediction, sequence rules across topics
- **Active error detection**: Pre-delivery validation against known facts
- **Negative memory mechanism**: Store "X is FALSE" assertions

### Planned (Priority 3)
- **Formal hypothesis testing**: A/B evaluation of alternatives
- **Sandboxed computation**: Math evaluation and code execution skill
- **Memory versioning**: Track how beliefs change over time

## Glossary

- **Mode Router**: Deterministic mathematical function selecting engagement mode from observable signals
- **Tie-Breaker**: Small LLM consulted when top 2 modes are within effective margin
- **Routing Signals**: Observable features collected from Redis and NLP analysis (~5ms)
- **Router Confidence**: Normalized gap between top 2 scores — measures routing certainty
- **Pressure Signal**: Metric logged by monitors, consumed by the single regulator
- **Context Warmth**: Signal (0.0-1.0) measuring how much context is available for current topic
- **Drift Gist**: Spontaneous thought stored during idle periods (DMN)
- **Episode**: Narrative memory unit with intent, context, action, emotion, outcome, salience
- **Concept**: Knowledge node with strength decay and spreading activation
- **Salience**: Computed importance metric (0.1-1.0) based on novelty, emotion, commitment
