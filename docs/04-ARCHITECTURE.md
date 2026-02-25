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

#### Response Generation
- **`frontal_cortex_service.py`** — LLM response generation using mode-specific prompts
- **`voice_mapper_service.py`** — Translates identity vectors to tone instructions

#### Memory System
- **`context_assembly_service.py`** — Unified retrieval from all 5 memory layers with weighted budget allocation
- **`episodic_retrieval_service.py`** — Hybrid vector + FTS search for episodes
- **`semantic_retrieval_service.py`** — Vector similarity + spreading activation for concepts
- **`user_trait_service.py`** — Per-user trait management with category-specific decay
- **`episodic_storage_service.py`** — PostgreSQL CRUD for episodic memories
- **`semantic_storage_service.py`** — PostgreSQL CRUD for semantic concepts
- **`gist_storage_service.py`** — Redis-backed short-term memory with deduplication
- **`list_service.py`** — Deterministic list management (shopping, to-do, chores); perfect recall with full history via `lists`, `list_items`, `list_events` tables
- **`moment_service.py`** — Pinned message bookmarks with LLM-enriched context, pgvector semantic search, and salience boosting; stores user-pinned Chalie responses as permanent, searchable moments via `moments` table
- **`moment_enrichment_service.py`** — Background worker (5min poll): collects gists from ±4hr interaction window, generates LLM summaries, seals moments after 4hrs; boosts related episode salience on seal
- **`moment_card_service.py`** — Inline HTML card emission for moment display in the conversation spine

#### Autonomous Behavior
- **`cognitive_drift_engine.py`** — Default Mode Network (DMN) for spontaneous thoughts during idle; attention-gated (skips when user in deep focus)
- **`autonomous_actions/`** — Decision routing (priority 10→6): CommunicateAction, SuggestAction (skill-matched proactive suggestions), NurtureAction (gentle phase-appropriate presence), ReflectAction, SeedThreadAction
- **`spark_state_service.py`** — Tracks relationship phase progression (first_contact → surface → exploratory → connected → graduated)
- **`spark_welcome_service.py`** — First-contact welcome message triggered on first SSE connection; runs once per lifecycle
- **`curiosity_thread_service.py`** — Self-directed exploration threads (learning and behavioral) seeded from cognitive drift
- **`curiosity_pursuit_service.py`** — Background worker exploring active threads via ACT loop
- **`decay_engine_service.py`** — Periodic decay (episodic 0.05/hr, semantic 0.03/hr)

#### Ambient Awareness
- **`ambient_inference_service.py`** — Deterministic inference engine (<1ms, zero LLM): place, attention, energy, mobility, tempo, device_context from browser telemetry + behavioral signals; thresholds loaded from `configs/agents/ambient-inference.json`
- **`place_learning_service.py`** — Accumulates place fingerprints (geohash ~1km, never raw coords) in `place_fingerprints` table; learned patterns override heuristics after 20+ observations
- **`client_context_service.py`** — Rich client context with location history ring buffer (12 entries), place transition detection, session re-entry detection (>30min absence), demographic trait seeding from locale, and circadian hourly interaction counts

#### Tool Integration
- **`act_loop_service.py`** — Iterative action execution with safety limits (60s timeout)
- **`act_dispatcher_service.py`** — Routes actions to skill handlers with timeout enforcement
- **`tool_registry_service.py`** — Tool discovery, metadata management, and cron execution via `run_interactive` (bidirectional stdin/stdout dialog protocol)
- **`tool_container_service.py`** — Container lifecycle; `run()` for single-shot, `run_interactive()` for bidirectional tool↔Chalie dialog (JSON-lines stdout, Chalie responses via stdin)
- **`tool_config_service.py`** — Tool configuration persistence; webhook key generation (HMAC-SHA256 + replay protection via X-Chalie-Signature/X-Chalie-Timestamp)
- **`tool_performance_service.py`** — Performance metrics tracking
- **`tool_profile_service.py`** — Tool capability caching and profiles
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

### Innate Skills (`backend/services/innate_skills/` and `backend/skills/`)

9 built-in cognitive skills for the ACT loop:
- **`recall_skill.py`** — Unified retrieval across ALL memory layers (<500ms)
- **`memorize_skill.py`** — Store gists and facts (<50ms)
- **`introspect_skill.py`** — Self-examination (context warmth, FOK signal, stats) (<100ms)
- **`associate_skill.py`** — Spreading activation through semantic graph (<500ms)
- **`scheduler_skill.py`** — Create/list/cancel reminders and scheduled tasks (<100ms)
- **`autobiography_skill.py`** — Retrieve synthesized user narrative with optional section extraction (<500ms)
- **`list_skill.py`** — Deterministic list management: add/remove/check items, view, history (<50ms)
- **`focus_skill.py`** — Focus session management: set, check, clear with distraction detection (<50ms)
- **`moment_skill.py`** — Natural language moment recall ("Do you remember...") and listing via pgvector search

## Worker Processes (`backend/workers/`)

### Queue Workers
- **Digest Worker** — Core pipeline: classify → route → generate response → enqueue memory job
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
- **Triage Calibration** — Triage correctness scoring (24h cycle)
- **Profile Enrichment** — Tool profile enrichment (6h cycle)
- **Curiosity Pursuit** — Explores curiosity threads via ACT loop (6h cycle)
- **Moment Enrichment** — Enriches pinned moments with gists + LLM summary, seals after 4hrs (5min poll)

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
- **User Traits** (PostgreSQL) — Personal facts with category-specific decay
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
- **ACT loop**: 60s cumulative timeout, ~7 max iterations
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
- **`system`** — Health, version, settings
- **`tools`** — Tool execution, configuration
- **`providers`** — LLM provider configuration
- **`push`** — Push notification subscription
- **`scheduler`** — Reminders and scheduled tasks
- **`lists`** — List management
- **`stubs`** — Placeholder endpoints (calendar, notifications, integrations, voice, permissions) returning 501

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

### Planned (Priority 1)
- **Strategic multi-session planning**: Goal stack + cross-session task tracking
- **User memory transparency API**: Direct REST endpoints for memory inspection

### Planned (Priority 2)
- **Cross-topic pattern mining**: Behavioral prediction, sequence rules
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
