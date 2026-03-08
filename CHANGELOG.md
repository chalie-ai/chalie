# Changelog

All notable changes to Chalie are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### In Progress
- **Uncertainty Engine** — Contradiction detection and resolution across the memory hierarchy. Adds `reliability` field to traits, episodes, and concepts; new `uncertainties` table; `UncertaintyService` and `ContradictionClassifierService`; drift RECONCILE action. See `docs/15-UNCERTAINTY-ENGINE.md`.

---

## Recent

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
- Upload documents (PDF, DOCX, PPTX, HTML, plain text) via REST API or camera OCR
- Hybrid search: semantic (sqlite-vec) + full-text (FTS5) + keyword boost via Reciprocal Rank Fusion
- Soft delete with 30-day purge window; duplicate detection (SHA-256 hash + cosine similarity)
- Background processing worker with adaptive chunking and SimHash fingerprinting

### Ambient Awareness System
- Deterministic ambient inference: place, attention, energy, mobility, tempo, device context (<1ms, zero LLM)
- Place Learning Service: accumulates fingerprints, overrides heuristics after 20+ observations
- Event Bridge: stabilization windows, cooldowns, confidence gating, focus gates
- See `docs/16-AMBIENT-AWARENESS.md`

### Curiosity System
- Self-directed curiosity threads (learning and behavioral types)
- Seeded from cognitive drift; pursued via bounded ACT loop on 6h cycle
- Findings enter normal memory pipeline; surface naturally in future conversations
- See `docs/17-CURIOSITY-SYSTEM.md`

### Cognitive Drift Engine (DMN)
- Default Mode Network: spontaneous thoughts during idle periods
- Attention-gated: suppressed during deep focus
- Autonomous actions: COMMUNICATE, REFLECT, PLAN, SeedThreadAction, RECONCILE (uncertainty)

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
