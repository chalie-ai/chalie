# Chalie Documentation Index

Welcome to the Chalie documentation. This is your guide to understanding, deploying, and developing the cognitive assistant system.

## Getting Started

**New to Chalie?** Start here:
1. **[01-QUICK-START.md](01-QUICK-START.md)** — Quick start guide, prerequisites, deployment instructions

## Setup & Configuration

**Setting up Chalie for the first time?** Follow these guides in order:
1. **[02-PROVIDERS-SETUP.md](02-PROVIDERS-SETUP.md)** — Configure LLM providers (Ollama, Anthropic, OpenAI, Gemini)

**Running multiple instances on the same host?**
- **[11-MULTI-INSTANCE-SETUP.md](11-MULTI-INSTANCE-SETUP.md)** — Configure PORT and COMPOSE_PROJECT_NAME for parallel stacks

## Understanding the System

**Want to understand how Chalie works?** Read these in order:
1. **[04-ARCHITECTURE.md](04-ARCHITECTURE.md)** — Complete system architecture, services, workers, data flow
2. **[05-WORKFLOW.md](05-WORKFLOW.md)** — Detailed step-by-step flow of prompt processing
3. **[07-COGNITIVE-ARCHITECTURE.md](07-COGNITIVE-ARCHITECTURE.md)** — Deterministic mode router and decision flow
4. **[06-WORKERS.md](06-WORKERS.md)** — Worker processes and services overview
5. **[08-DATA-SCHEMAS.md](08-DATA-SCHEMAS.md)** — Data schemas for Redis and PostgreSQL

## If You're a Developer Exploring the Codebase

Recommended reading order for engineers:
1. **[05-WORKFLOW.md](05-WORKFLOW.md)** — The full request pipeline in 15 steps; fastest way to build a mental map
2. **[04-ARCHITECTURE.md](04-ARCHITECTURE.md)** — All services, workers, and data flow in one place
3. **[07-COGNITIVE-ARCHITECTURE.md](07-COGNITIVE-ARCHITECTURE.md)** — The deterministic mode router and decision logic
4. **[09-TOOLS.md](09-TOOLS.md)** — How to extend Chalie with sandboxed tools
5. **[10-CONTEXT-RELEVANCE.md](10-CONTEXT-RELEVANCE.md)** — Token optimization and selective context injection

## Tools & Extensions

**Building tools to extend Chalie's capabilities?**
- **[09-TOOLS.md](09-TOOLS.md)** — Tools architecture, creating tools, sandbox constraints, examples

## Performance & Optimization

**Optimizing Chalie's performance?**
- **[10-CONTEXT-RELEVANCE.md](10-CONTEXT-RELEVANCE.md)** — Context relevance pre-parser, selective context injection, configuration tuning

## User Interface

**Building or modifying the web interface?**
- **[03-WEB-INTERFACE.md](03-WEB-INTERFACE.md)** — Web UI requirements, layout, functionality

## Quick Reference

### File Organization
```
docs/
├── INDEX.md                          ← You are here
├── 01-QUICK-START.md                 ← Getting started
├── 02-PROVIDERS-SETUP.md             ← LLM provider configuration
├── 03-WEB-INTERFACE.md               ← Web UI specification
├── 04-ARCHITECTURE.md                ← System architecture
├── 05-WORKFLOW.md                    ← Request processing pipeline
├── 06-WORKERS.md                     ← Worker processes overview
├── 07-COGNITIVE-ARCHITECTURE.md      ← Mode router & cognition
├── 08-DATA-SCHEMAS.md                ← Data structures
├── 09-TOOLS.md                       ← Tools system & creation guide
├── 10-CONTEXT-RELEVANCE.md           ← Context relevance pre-parser & optimization
└── 11-MULTI-INSTANCE-SETUP.md        ← Running multiple instances on one host
```

### Important Project Files (Not in docs/)
- **`CLAUDE.md`** — Project instructions for Claude Code (development guidance)
- **`README.md`** — Root-level project overview (mirrors 01-QUICK-START.md)
- **`docker-compose.yml`** — Service definitions and port mappings (supports PORT variable for multi-instance)
- **`.env.example`** — Configuration template with defaults (PORT, POSTGRES_PASSWORD, SESSION_SECRET_KEY, COOKIE_SECURE)

### Key Directories
- **`backend/`** — Python backend (services, workers, API, configs, migrations)
- **`frontend/interface/`** — Main chat web UI (HTML, CSS, JavaScript)
- **`frontend/brain/`** — Admin/cognitive dashboard
- **`frontend/on-boarding/`** — Account setup wizard
- **`backend/prompts/`** — LLM prompt templates (mode-specific)
- **`backend/configs/`** — Configuration files and schemas
- **`backend/migrations/`** — Database migration scripts

## Common Tasks

### Deploying Chalie
1. Clone repository
2. Copy `.env.example` to `.env`
3. Run `docker-compose build && docker-compose up -d`
4. Configure providers via REST API (see 02-PROVIDERS-SETUP.md)
5. Open http://localhost:8081/ in browser

### Understanding a Specific Component
- **Memory system?** → See 04-ARCHITECTURE.md "Memory Hierarchy"
- **How routing works?** → See 07-COGNITIVE-ARCHITECTURE.md
- **Data flow?** → See 05-WORKFLOW.md or 04-ARCHITECTURE.md "Data Flow Pipeline"
- **Worker responsibilities?** → See 06-WORKERS.md
- **Tools & extensions?** → See 09-TOOLS.md

### Configuring a New LLM Provider
1. Read 02-PROVIDERS-SETUP.md
2. Use REST API (`POST /providers`) to register provider
3. Optionally assign to specific jobs (`PUT /providers/jobs/{job_name}`)

### Building/Modifying the Web UI
1. Read 03-WEB-INTERFACE.md for requirements
2. All code goes in `frontend/interface/` (HTML, CSS, JS)
3. UI communicates with backend via REST API at `/chat`

### Adding New Services or Workers
1. Create file in `backend/services/` or `backend/workers/`
2. Register in `backend/consumer.py`
3. Document in 06-WORKERS.md
4. Add tests in `backend/tests/`

### Creating a New Tool
1. Read 09-TOOLS.md for architecture and requirements
2. Create `backend/tools/tool_name/` directory
3. Add `manifest.json` (metadata, parameters, trigger type)
4. Add `Dockerfile` (container image definition)
5. Implement tool logic in your language of choice
6. Configure via REST API (`PUT /tools/<name>/config`)

## Architecture Quick Facts

- **Language**: Python 3.9+
- **Databases**: PostgreSQL (+ pgvector extension), Redis
- **Frontend**: Vanilla JavaScript (Radiant design system)
- **LLM Support**: Ollama, Anthropic, OpenAI, Google Gemini
- **Port**: Backend API on 8080, Frontend on 8081
- **Configuration**: env vars > .env file > JSON files > hardcoded defaults
- **Worker Pattern**: Queue-based (Redis) with multiple worker types
- **Safety**: Deterministic routing, single authority for learning, bounded parameter updates

## Key Concepts

### Mode Routing
Chalie selects one of 5 engagement modes for each user message:
- **RESPOND** — Give a substantive answer
- **CLARIFY** — Ask a clarifying question
- **ACKNOWLEDGE** — Brief social response
- **ACT** — Execute internal actions (memory, reasoning)
- **IGNORE** — No response needed

Mode is selected by a fast mathematical router (~5ms) based on observable signals, then the LLM generates a response in that mode-specific style.

### Memory Hierarchy
Information flows through multiple layers with different timescales:
1. **Working Memory** (4 turns, 24h) → Current conversation
2. **Gists** (30min) → Compressed exchange summaries
3. **Facts** (24h) → Atomic assertions
4. **Episodes** (permanent, decaying) → Narrative memories
5. **Concepts** (permanent, decaying) → Knowledge graph
6. **Lists** (permanent, no decay) → Deterministic ground-truth state (shopping, to-do, chores)

### Autonomy
Chalie can generate spontaneous thoughts during idle periods via the Cognitive Drift Engine (Default Mode Network). Thoughts go through three gates (quality, timing, engagement) before being sent to users.

### Safety Boundaries
- Prompts are immutable (marked as "authoritative")
- Skills are fixed at startup (no runtime registration)
- Data scoped by topic (no cross-topic leakage)
- Hard timeouts on all operations
- All external actions are async and audited

## Cognitive Terminology Glossary

| Term | Meaning in Chalie |
|---|---|
| **Episodic memory** | A stored narrative unit representing a past interaction — what happened, what was felt, what was decided |
| **Semantic memory** | Knowledge nodes (concepts) and their relationships — facts abstracted away from specific events |
| **Decay** | Natural fading of memory strength over time; prevents noise accumulation without explicit deletion |
| **Salience** | How contextually relevant a memory is to the current moment — gates retrieval priority |
| **Gist** | A compressed summary of a conversation exchange, bridging working memory and episodic storage |
| **Spreading activation** | When one concept is retrieved, related concepts are activated with lower priority — mimics associative recall |
| **Mode router** | The deterministic component that selects how Chalie should respond before any LLM is called |
| **Deterministic routing** | Mode selection via scored signals (~5ms), not via LLM inference — auditable and fast |
| **ACT loop** | The autonomous task execution cycle: plan → act → observe → continue-or-stop |
| **Cognitive drift (DMN)** | Spontaneous thought generation during idle periods, inspired by the Default Mode Network |

## What Chalie Is Not

- **Not AGI** — it does not plan or act autonomously without human instruction
- **Not a surveillance system** — memory decays by design; old facts fade unless reinforced
- **Not a productivity robot** — it is a thinking aid, not a task manager
- **Not a cloud service** — every byte stays local unless you configure an external LLM provider
- **Not a general automation platform** — tools are sandboxed, audited, and bounded by hard limits

## Support & Development

- **Issues**: Check GitHub issues or project backlog
- **Contributing**: Create feature branch, add tests, follow existing patterns
- **Questions**: Review relevant documentation section, check `docs/04-ARCHITECTURE.md` for recent work

## Document Status

**Last Updated**: 2026-02-21

All documentation reflects the current state of the codebase as of this date. See `CLAUDE.md` for recent changes and current development focus.

**Recent Additions**:
- Curiosity Threads: Replaced user-facing goals with self-directed exploration threads (learning and behavioral) seeded from cognitive drift
- New services: `curiosity_thread_service.py`, `curiosity_pursuit_service.py`, `seed_thread_action.py`
- Updated 07-COGNITIVE-ARCHITECTURE.md: 8 innate skills (removed goal)
- Updated 08-DATA-SCHEMAS.md: Replaced goals table with curiosity_threads table
- Updated 04-ARCHITECTURE.md: Added curiosity thread and pursuit services
- 09-TOOLS.md: Comprehensive tools system documentation
