# Changelog

All notable changes to Chalie are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### In Progress
- **Uncertainty Engine** — Contradiction detection and resolution across the memory hierarchy. Adds `reliability` field to traits, episodes, and concepts; new `uncertainties` table; `UncertaintyService` and `ContradictionClassifierService`; drift RECONCILE action. See `docs/15-UNCERTAINTY-ENGINE.md`.

---

## Recent

### DB-Backed Context Window Management (2026-04-09)
- Eliminated in-memory message accumulation in the LLM tool-calling loop; OOM kills in Docker containers no longer occur regardless of tool loop depth
- New `context_window_service.py` always reconstructs the messages array from the database on every LLM call — nothing accumulates in memory
- Compaction triggers at 80% of the provider context limit; overflow handling triggers compaction before storing a tool result that would exceed 100% of the limit
- After overflow compaction, the tool result is stored to transcript (id > watermark) and appears naturally in the next `build_messages()` call
- Tool-triggered compaction adds a "Current Task State" section to the compaction summary so the model can continue mid-task without context loss
- Compaction always uses the same provider job as the conversation — LLMs are never mixed within a turn
- `MessageProcessor.send()` now tracks only transcript IDs in memory; removed `_prune_messages()`, `_resolve_token_budget()`, `_estimate_tokens()`, `MAX_RESULT_CHARS`, `_TOKEN_BUDGET_RATIO`, `_TOKEN_BUDGET_CAP`
- `UserPromptAssemblyService.build()` no longer injects conversation history; history is handled entirely by `context_window_service.build_messages()`
- `ToolCallService.store()` and `store_batch()` now accept `tool_call_id` (the LLM-generated call ID); new `get_by_transcript_ids()` method for efficient batch loading during context reconstruction
- Schema: `tool_calls.tool_call_id TEXT` (migration 038), `compactions.overflow_content TEXT` (migration 039)
- 44 integration tests in `test_context_window_service.py` using real SQLite

### Cognitive Reflex Service
- Learned fast-path that bypasses the full triage pipeline for self-contained queries
- Heuristic pre-screen (~1ms) + sqlite-vec cosine cluster lookup (~5-20ms)
- Rolling-average centroids generalize from examples; self-correcting via user corrections and shadow validation

### User Trait Follow-Up
- Chalie asks a natural follow-up question when a user volunteers personal information
- Detects volunteer signals (unprompted self-disclosure) and generates contextually appropriate questions

### Triage Calibration Events
- Fixed: calibration events failing to insert due to missing `id` column

### Input Dock Polish
- Fixed: icon buttons misaligned on desktop
- Fixed: right padding matching left for breathing room around send icon
- Send arrow uses primary accent color

---

## Feature History

### Persistent Tasks & Plan Decomposition
- Multi-session background tasks with state machine (PROPOSED → ACCEPTED → IN_PROGRESS → COMPLETED/PAUSED/CANCELLED/EXPIRED)
- LLM-powered goal → step DAG decomposition via `PlanDecompositionService`
- Plan-aware execution: follows step DAG (up to 3 steps/cycle), falls back to flat loop

### Document System
- Upload documents (PDF, DOCX, PPTX, HTML, plain text) via REST API
- Hybrid search: semantic (sqlite-vec) + full-text (FTS5) + keyword boost via Reciprocal Rank Fusion
- Soft delete with 30-day purge window; duplicate detection (SHA-256 hash + cosine similarity)
- Background processing worker with adaptive chunking and SimHash fingerprinting

### Ambient Awareness System
- Deterministic ambient inference: place, attention, energy, mobility, tempo, device context (<1ms, zero LLM)
- Place Learning Service: accumulates fingerprints, overrides heuristics after 20+ observations
- Event Bridge: stabilization windows, cooldowns, confidence gating, focus gates
- See `docs/16-AMBIENT-AWARENESS.md`

### Cognitive Drift Engine (DMN)
- Default Mode Network: spontaneous thoughts during idle periods
- Attention-gated: suppressed during deep focus
- Autonomous actions: COMMUNICATE, REFLECT, PLAN, RECONCILE (uncertainty)

### Memory Observability
- Brain dashboard observability tab: autobiography, traits, routing, memory, tools, identity, tasks
- `GET /system/observability/*` endpoints for all cognitive dimensions
- `DELETE /system/observability/traits/<key>` for user-driven trait correction

### Moments
- Pin meaningful Chalie responses as permanent searchable memories
- LLM-enriched context with gist collection from ±4hr window
- sqlite-vec semantic search for natural language recall ("Do you remember...")

### Autobiography Service
- 6h synthesis cycle producing a running narrative of who the user is
- Delta tracking: changed vs unchanged sections surfaced in observability

### Adaptive Boundary Detection
- 3-layer self-calibrating topic boundary detector (NEWMA + Transient Surprise + Leaky Accumulator)
- Replaces static cosine similarity threshold
- State persisted per-thread in MemoryStore; outer loop tuned by Topic Stability Regulator

### Deterministic Mode Router
- Mathematical scoring function over ~17 observable signals (~5ms)
- Separated from response generation (previously a single ~15s LLM call)
- Single authority for weight mutation: Routing Stability Regulator (24h cycle, ±0.02/day max)
- Full audit trail in `routing_decisions` SQLite table

### Voice (Native)
- Native STT (faster-whisper) and TTS (KittenTTS) — no Docker required
- Auto-detects dependencies on startup; returns 503 gracefully if unavailable

### Single-Process Architecture
- All workers run as daemon threads in one Python process
- MemoryStore replaces Redis — same API, zero infrastructure
- Docker optional: only needed for sandboxed tool execution
