# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Vision & Principles

See **[docs/00-VISION.md](docs/00-VISION.md)** for the Life OS vision, core principles, design guidelines, and decision filter. That document is the source of truth — what follows here are codebase-specific rules for agents working in this repo.

**Tech Stack:**
- Backend: Python (Flask + flask-sock WebSocket, SQLite + sqlite-vec + FTS5)
- In-Memory Store: Thread-safe MemoryStore (replaces Redis — same API, zero infrastructure)
- Frontend: Vanilla JavaScript served directly by Flask (no nginx)
- LLM Integration: Ollama, OpenAI, Anthropic, Google Gemini APIs
- Architecture: Single-process, multi-threaded (all workers run as daemon threads)
- Voice: Native Flask blueprint (Moonshine Voice STT + Kokoro TTS, both ONNX)
- Docker: Used only for deploying Chalie itself (Dockerfile + docker-compose)

## Mandatory Standards

### Datetime

**ALL datetimes must be timezone-aware UTC. No exceptions.**

- Use `utc_now()` from `services.time_utils` instead of `datetime.now()` or `datetime.utcnow()`
- Use `parse_utc(value)` from `services.time_utils` whenever reading a datetime from SQLite, JSON, config files, or any external source
- Never call `datetime.now()`, `datetime.utcnow()`, or `datetime.fromisoformat()` directly
- When saving timestamps, always use `utc_now().isoformat()` (produces `+00:00` suffix)

**Why:** SQLite's `datetime('now')` returns naive strings. The `time_utils` module is the single chokepoint that prevents `TypeError` from mixing naive and aware datetimes.

### Memory Storage

**All memory operations MUST use BOTH the MCP memory server AND file-based memory.**

- Always call `store_memory` (MCP) alongside writing to the file-based memory system
- Always call `fetch_memory` (MCP) alongside reading file-based memory when recalling context
- **Proactive recording — do NOT wait to be asked:** After every commit, plan closure, architectural decision, or feature discussion, IMMEDIATELY call `store_memory` + write file-based memory with full detail

### GitHub Issues

**NEVER close GitHub issues without explicit user permission.** Investigate, analyse, present findings — then wait for the user to decide.

### Debugging Discipline

**NEVER describe a problem without concrete proof.** No speculation, no "probably", no "likely because." Read the actual error/logs/data, query actual state, THEN speak. If you can't get evidence, say "I don't know yet" and go get it.

## Architecture Rules

**Tool-agnostic infrastructure.** No tool-specific logic in triage, dispatcher, worker, frontend, or card rendering. Tools self-declare via manifests. Innate skills are the exception — they are core cognitive capabilities with dedicated services.

**External tools are ALWAYS lazy-loaded via `find_tools`.** NEVER pre-inject external tool schemas into the native tools list, system prompt, or any pre-loaded context. The `find_tools` innate skill performs semantic discovery at runtime — that is the only path for external tool access. Pre-injection bloats context, creates staleness bugs, and breaks the tool-agnostic architecture. If tool routing accuracy needs improving, improve `find_tools` itself — do not bypass it.

**Model-agnostic.** Different cognitive functions may use different LLM providers. Never hardcode which model powers which function.

**Persistent runtime.** Continuous cognitive loop with long-lived worker threads. Every message flows through the memory pipeline (Topic Context → Episode → Concept → Abstraction). Do not bypass or shortcut it.

**Clean removal. No hollow passthroughs.** When removing a service, class, model, or function: rip it out completely. Delete the file, delete all imports, delete all callers, delete tests. Never leave wrapper functions that just call another function without doing any mutation, logic, or gating — that is bloat. The only backwards compatibility we care about is the database schema (and minimally the interface). No re-exports, no "deprecated" shims, no passthrough functions.

## Git Workflow

**Merging to main and releasing must always be done manually — never automated.**

### Branch Strategy
```
Day-to-day work:  commit directly on rc-<version>
Bigger features:  feature/<name> → merge into rc-<version>
Release:          rc-<version> → merge into main → tag v*.*.*
```

- **`rc-*`**: Active development branch. Most work happens here directly.
- **`feature/*`**: For larger features that need isolation. Merge into `rc-*` when ready.
- **`main`**: Stable. Only receives merges from `rc-*` when a release goes live.
- **Release tags** (`v*.*.*`): Cut from main after `rc-*` is merged in.

### Pre-Merge Checks

Before merging any branch into `rc-*` or `main`, run:
```bash
cd backend && pytest -m unit -q        # all unit tests must pass
```

If tests fail, do not merge. Fix the issue or send back for rework.

### CI Behavior
- **Build log**: Synced to website on every push.
- **Docker publish + installer**: Only on `v*.*.*` tags.
- **Docs sync**: Only on pushes to `main` that touch `docs/`.

## Development Commands

### Running Chalie
```bash
./run.sh                              # sets up venv + deps, then launches
./run.sh --port=9000                  # custom port
./run.sh --no-voice                   # skip voice dep sync (faster cold start)
python backend/run.py                 # direct launch (for debugging)
```

### Testing
```bash
cd backend
pytest                                # all tests
pytest -m unit                        # unit only (fast, in-memory SQLite)
pytest -m integration                 # integration (requires services)
pytest tests/test_file.py::TestClass::test_method  # single test
```

### Setup
```bash
cd backend && pip install -r requirements.txt
# Onboarding: http://localhost:8081/on-boarding/
```

## Architecture

Single entry point: `python backend/run.py --port=8081 --host=0.0.0.0`. Auto-creates SQLite DB, runs migrations, starts all worker/service daemon threads, monitors thread health, handles graceful shutdown.

### High-Level Data Flow
```
User Input (WebSocket)
  ├─ Persist to topic transcript (SQLite, append-only)
  ├─ Retrieve context: compaction summary + recent transcript + semantic memories
  ├─ Tool loop in MessageProcessor.send():
  │    LLM call → if tool_calls → dispatch via ActDispatcherService
  │    → store results (ephemeral) → append tool_result messages
  │    → per-iteration synthesis to WebSocket → repeat until done or cap hit
  │    (30 iterations / 15 min max)
  ├─ Final response committed to transcript
  └─ Background threads: episode extraction (rolling transcript trigger), decay, consolidation
```

User messages go through `MessageProcessor.send()` which runs a standard tool-calling loop. No routing gate — the LLM decides whether to respond directly or invoke skills/tools. `ModeRouterService` remains active for non-user flows (drift, proactive, fallback). All tool invocations are audited in the `tool_calls` table; ephemeral records (loop results, steers) are excluded from Previous Turns context.

### Memory Hierarchy
- **Topic Transcript** — persistent, append-only per topic; budget-aware filling
- **Compaction** — incremental LLM-powered summarization of older transcript entries
- **Episodes** — transcript-linked narrative units with power-law decay (storage_strength + retrieval_weight); created by rolling transcript trigger, consolidate into super episodes
- **Knowledge** — unified store for traits, facts, procedures, preferences, rules, metrics (RRF hybrid search: exact + FTS5 + vector KNN)

### Key Subsystems
For the full service catalog, worker list, and database schema, see **[docs/04-ARCHITECTURE.md](docs/04-ARCHITECTURE.md)** and **`backend/schema.sql`**.

Key subsystems worth knowing about:
- **DMN Service**: Default Mode Network — timer-based proactive intelligence (60min idle → recent context, 6h cadence → salience context); calls unified_generate with proactive=True
- **Tool Loop**: Standard tool-calling while loop inside `MessageProcessor.send()` — LLM calls, dispatches via `ActDispatcherService`, stores ephemeral results via `ToolCallService`, appends tool_result messages, repeats; safety cap 30 iterations / 15 min
- **ToolCallService**: Unified API for all `tool_calls` table operations; `ephemeral` flag controls what appears in Previous Turns context
- **Goal Pursuit**: `GoalPursuitProcessor` daemon thread (subclass of `MessageProcessor`) spawned by the `goal_pursuit` innate skill; single `goal` string parameter; 50 iterations / 2h timeout; no state machine, no plan phase; surfaces via `OutputService.enqueue_proactive()`
- **Document Service**: Hybrid search (semantic + FTS + keyword boost via RRF), soft delete, duplicate detection
- **Event Bridge**: Connects ambient context changes to goal signal routing with stabilization windows and focus gates
- **Interface Layer**: External apps pair via bluetooth-style protocol, expose tool capabilities, health-monitored (30s interval, 3-failure threshold)

### Innate Skills (`backend/services/innate_skills/`)
Built-in cognitive skills always available to the LLM:
`memory`, `introspect`, `associate`, `schedule`, `autobiography`, `list`, `goal_pursuit`, `document`, `read`, `reflect`, `find_tools`, `notes`, `goals`, `rich_render`, `review_tool_calls`

## Code Organization

- **Frontend**: Three independent apps — `frontend/interface/` (main chat UI), `frontend/brain/` (admin dashboard), `frontend/on-boarding/` (setup wizard)
- **Frontend modules**: `app.js` is a thin orchestrator. Domain logic in focused ES6 modules. Modules communicate via constructor injection + callbacks + custom DOM events. `app.js` wires all connections.
- **Asset versioning**: `VERSION` file at project root is the single source of truth. Flask injects importmap with `?v=VERSION`.
- **Block protocol**: ALL content uses universal block format (JSON arrays of typed block objects). `BlocksRenderService` (backend) → `blocks.js` (frontend). No HTML over the wire.
- **Configuration**: CLI args for runtime (port, host); JSON files in `configs/` for agent configs; LLM providers in database via REST API
- **Mode-specific prompts**: `backend/prompts/` by mode name
- **Tool isolation**: Tools in `backend/tools/` are in-process Python modules. NEVER modify the tool framework to accommodate a specific tool.

## Design Philosophy: Radiant

Cinematic dark UI — near-black canvas, precision accent glows (violet, magenta, cyan), atmospheric depth via drifting orbs. Restraint is the guiding principle: when something glows, it matters. Full spec: **[docs/03-WEB-INTERFACE.md](docs/03-WEB-INTERFACE.md)**.

## Testing Strategy

- `@pytest.mark.unit` — No external dependencies (fast, in-memory SQLite)
- `@pytest.mark.integration` — End-to-end (slower, requires services)
- Unit tests use in-memory SQLite (`:memory:`) and MemoryStore directly (no mocking needed — it IS production)
- **Nightly scenarios** — Black-box YAML scenarios at `/Volumes/llm/chalie-nightly-test/scenarios/`. When writing or reviewing scenarios, read `.claude/commands/nightly-tests.md` first for the full test criteria and template.

## Important Notes

- **No Telemetry**: Zero external calls except to configured LLM/voice providers
- **Local First**: All data in SQLite unless external LLM providers configured
- **Encryption**: AES-256-GCM envelope encryption via VaultService (password-derived KEK wraps a random DEK)
- **Single Process**: Everything in one Python process — no external databases
- **CORS**: Defaults to localhost, restrict before production
- **Reverse Proxy Safe**: Works behind nginx/caddy/etc. via ProxyFix middleware

## Long-Term Trajectory

Do not build toward these prematurely, but do not make decisions that would prevent them.

- **Agent-to-Agent Communication**: Instances may eventually exchange specialized knowledge
- **Interface Agnosticism**: The frontend is disposable; the backend cognitive runtime is the product
- **Specialization Through Constraint**: Memory compression forces trade-offs, encouraging natural specialization

## Plans

Plans live **outside the repo** at `/Volumes/llm/chalie-plans/`, organized by release version:

```
/Volumes/llm/chalie-plans/
  v0.3.0/          ← current release plans
  v0.4.0/          ← next release (when created)
```

When creating or referencing plans, always use the version subfolder matching the current release branch (`rc-X.Y.Z` → `vX.Y.Z/`).

## Useful Files

- `docs/INDEX.md` — Documentation hub
- `docs/00-VISION.md` — Product vision and decision filter
- `docs/03-WEB-INTERFACE.md` — Radiant design system spec
- `docs/04-ARCHITECTURE.md` — Full architecture, services, schema, workers
- `backend/schema.sql` — Database schema
- `backend/run.py` — Entry point
