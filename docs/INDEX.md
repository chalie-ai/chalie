# Chalie Documentation: Guides, Architecture, and API Reference

Welcome to the Chalie documentation. Explore guides, architecture deep-dives, and API references. See also [Quick Start](01-QUICK-START.md).

## Popular Pages
- [Architecture](04-ARCHITECTURE.md) — How Chalie works under the hood
- [Cognitive Router](07-COGNITIVE-ARCHITECTURE.md) — 5ms deterministic mode selection
- [Tools System](09-TOOLS.md) — Build and extend with custom tools
- [Quick Start](01-QUICK-START.md) — Get running in 2 minutes

## Vision & Philosophy

**Understand what Chalie is and why it exists:**
- **[00-VISION.md](00-VISION.md)** — Product vision, design principles, delegation boundary, feature decision filter

## Getting Started

**New to Chalie?** Start here:
1. **[01-QUICK-START.md](01-QUICK-START.md)** — Quick start guide, prerequisites, deployment instructions
2. **[FAQ](15-FAQ.md) — Common questions answered**

## Setup & Configuration

**Setting up Chalie for the first time?** Follow these guides in order:
1. **[02-PROVIDERS-SETUP.md](02-PROVIDERS-SETUP.md)** — Configure [LLM providers](02-PROVIDERS-SETUP.md) (Ollama, Anthropic, OpenAI, Gemini)

## Understanding the System

**Want to understand how Chalie works?** Read these in order:
1. **[04-ARCHITECTURE.md](04-ARCHITECTURE.md)** — Complete [system architecture](04-ARCHITECTURE.md), services, workers, data flow
2. **[13-MESSAGE-FLOW.md](13-MESSAGE-FLOW.md)** — [Visual flow diagrams](13-MESSAGE-FLOW.md): every path, every MemoryStore/DB hit, every LLM call
3. **[05-WORKFLOW.md](05-WORKFLOW.md)** — Detailed step-by-step flow of [prompt processing](05-WORKFLOW.md)
4. **[07-COGNITIVE-ARCHITECTURE.md](07-COGNITIVE-ARCHITECTURE.md)** — [Deterministic mode](07-COGNITIVE-ARCHITECTURE.md) router and [decision flow](07-COGNITIVE-ARCHITECTURE.md)
5. **[06-WORKERS.md](06-WORKERS.md)** — [Worker processes](06-WORKERS.md) and services overview
6. **[08-DATA-SCHEMAS.md](08-DATA-SCHEMAS.md)** — [Data schemas](08-DATA-SCHEMAS.md) for MemoryStore and SQLite

## If You're a Developer Exploring the Codebase

Recommended reading order for engineers:
0. **[00-VISION.md](00-VISION.md)** — Start with why: product vision, design principles, and feature decision filter
1. **[13-MESSAGE-FLOW.md](13-MESSAGE-FLOW.md)** — Visual map of every path, storage hit, and LLM call; fastest way to build a mental model

2. **[05-WORKFLOW.md](05-WORKFLOW.md)** — The full [request pipeline](05-WORKFLOW.md) in 15 steps; narrative explanation
3. **[04-ARCHITECTURE.md](04-ARCHITECTURE.md)** — All services, workers, and data flow in one place
4. **[07-COGNITIVE-ARCHITECTURE.md](07-COGNITIVE-ARCHITECTURE.md)** — The [deterministic mode](07-COGNITIVE-ARCHITECTURE.md) router and decision logic
5. **[09-TOOLS.md](09-TOOLS.md)** — How to extend Chalie with [sandboxed tools](09-TOOLS.md)
6. **[10-CONTEXT-RELEVANCE.md](10-CONTEXT-RELEVANCE.md)** — [Token optimization](10-CONTEXT-RELEVANCE.md) and [selective context injection](10-CONTEXT-RELEVANCE.md)

## Tools & Extensions

**Building tools to extend Chalie's capabilities?**
- **[09-TOOLS.md](09-TOOLS.md)** — Tools architecture, creating tools, sandbox constraints, examples
- **[14-DEFAULT-TOOLS.md](14-DEFAULT-TOOLS.md)** — First-party [default tools](14-DEFAULT-TOOLS.md) installed on first startup, auto-install behavior, `--disable-default-tools`

## Performance & Optimization

**Optimizing Chalie's performance?**
- **[10-CONTEXT-RELEVANCE.md](10-CONTEXT-RELEVANCE.md)** — [Context relevance](10-CONTEXT-RELEVANCE.md) pre-parser, [selective context injection](10-CONTEXT-RELEVANCE.md), configuration tuning

## User Interface

**Building or modifying the [web interface](03-WEB-INTERFACE.md)?**
- **[03-WEB-INTERFACE.md](03-WEB-INTERFACE.md)** — Web UI requirements, layout, functionality

## Quick Reference

### File Organization
```
docs/
├── INDEX.md                          ← You are here
├── 00-VISION.md                      ← Product vision & design compass
├── 01-QUICK-START.md                 ← Getting started
├── 02-PROVIDERS-SETUP.md             ← LLM provider configuration
├── 03-WEB-INTERFACE.md               ← Web UI specification
├── 04-ARCHITECTURE.md                ← [System architecture](04-ARCHITECTURE.md)
├── 05-WORKFLOW.md                    ← Request processing pipeline
├── 06-WORKERS.md                     ← [Worker processes](06-WORKERS.md) overview
├── 07-COGNITIVE-ARCHITECTURE.md      ← Mode router & cognition
├── 08-DATA-SCHEMAS.md                ← Data structures
├── 09-TOOLS.md                       ← Tools system & creation guide
├── 10-CONTEXT-RELEVANCE.md           ← [Context relevance](10-CONTEXT-RELEVANCE.md) pre-parser & optimization
├── 12-TESTING.md                     ← Test conventions, fixtures, mock strategies
├── 13-MESSAGE-FLOW.md                ← [Visual flow diagrams](13-MESSAGE-FLOW.md): all paths, MemoryStore/DB, LLM calls
└── 14-DEFAULT-TOOLS.md               ← [Default tools](14-DEFAULT-TOOLS.md) installed on first startup
```

### Important Project Files (Not in docs/)
- **`CLAUDE.md`** — Project instructions for Claude Code (development guidance)
- **`README.md`** — Root-level project overview
- **`installer/install.sh`** — One-line installer (published at https://chalie.ai/install)
- **`.env.example`** — Configuration template (PORT — all secrets auto-generate)

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
1. `curl -fsSL https://chalie.ai/install | bash`
2. `chalie` — opens at http://localhost:8081
3. Complete onboarding to configure your LLM provider (see 02-PROVIDERS-SETUP.md)

### Understanding a Specific Component
- **Product philosophy?** → See 00-VISION.md — core principles, delegation boundary, behavioral guidelines
- **Should we build this feature?** → See 00-VISION.md "Decision Filter" — 7 yes/no questions
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
2. Register in `backend/run.py`
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
- **Databases**: SQLite (WAL mode + sqlite-vec + FTS5), MemoryStore (in-memory)
- **Frontend**: Vanilla JavaScript (Radiant design system)
- **LLM Support**: Ollama, Anthropic, OpenAI, Google Gemini
- **Port**: 8081 (configurable via `--port=N`)
- **Configuration**: env vars > .env file > JSON files > hardcoded defaults
- **Worker Pattern**: Thread-based (PromptQueue) with daemon worker threads
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

## Testing

**Writing or reviewing tests?**
- **[12-TESTING.md](12-TESTING.md)** — Test conventions, fixture catalog, mock strategies, how to add tests

## Document Status

**Last Updated**: 2026-02-26

All documentation reflects the current state of the codebase as of this date. See `CLAUDE.md` for recent changes and current development focus.

**Recent Additions**:
- Testing guide: `12-TESTING.md` — conventions, fixtures, mock strategies
- Observability endpoints: `/system/observability/*` for cognitive legibility
- Moments API: Pin, list, search, and forget meaningful exchanges
- Task strip: Persistent background tasks visible in the UI
- Understanding tab: Brain dashboard cognitive transparency
- Curiosity Threads: Self-directed exploration threads seeded from cognitive drift
- 09-TOOLS.md: Comprehensive tools system documentation
- Document skill: Upload, extract, chunk, embed, and hybrid-search documents (warranties, contracts, manuals) via ACT loop innate skill

## Related Documentation
- [Vision & Philosophy](00-VISION.md)
- [Quick Start Guide](01-QUICK-START.md)
- [LLM Providers Setup](02-PROVIDERS-SETUP.md)
- [Web Interface](03-WEB-INTERFACE.md)
- [System Architecture](04-ARCHITECTURE.md)
- [Workflow Guide](05-WORKFLOW.md)
- [Workers Overview](06-WORKERS.md)
- [Cognitive Architecture](07-COGNITIVE-ARCHITECTURE.md)
- [Data Schemas](08-DATA-SCHEMAS.md)
- [Tools & Extensions](09-TOOLS.md)
- [Context Relevance](10-CONTEXT-RELEVANCE.md)
- [Testing Guide](12-TESTING.md)
- [Message Flow Diagrams](13-MESSAGE-FLOW.md)
- [Default Tools](14-DEFAULT-TOOLS.md)