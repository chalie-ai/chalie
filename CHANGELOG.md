# Changelog

All notable changes to Chalie are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### In Progress
- **Uncertainty Engine** — Contradiction detection and resolution across the memory hierarchy. Adds `reliability` field to traits, episodes, and concepts; new `uncertainties` table; `UncertaintyService`; drift RECONCILE action. See `docs/15-UNCERTAINTY-ENGINE.md`.

### Changed
- Pattern matcher rewrite — `pattern_extractor.py` (694 LOC, 6-vertical × 4-class flow) replaced by `PatternMatchProcessor`, a single-pass LLM matcher that runs every ≥50 new transcripts. The model emits `save_pattern` / `save_graph` tool calls in parallel (`MAX_ITERATIONS=30`). Decay (−0.005 per pass, soft-delete at 0) moved from `DecayEngine` to `PatternMatchProcessor.postTurn()`.
- New API endpoint `POST /system/subconscious/tick` — forces one subconscious worker tick bypassing both gates. Used by nightly tests; auth required.
- `SubconsciousWorker.run_once(force: bool = False)` — gate check moved to `run_once`; pass `force=True` to bypass the idle and already-fired gates.
- `ToolRenderAndRecordService._record()` — short-circuits with DEBUG log when `transcript_id is None` (prevents IntegrityError for `SKIP_TRANSCRIPT_WRITE=True` processors that dispatch tools).

### Removed
- `services/pattern_extractor.py` and `tests/test_pattern_extractor.py`.
- `DecayEngineService._decay_behavioral_patterns()` — pattern decay is now owned by `PatternMatchProcessor.postTurn()`.
- `_step_patterns` / `patterns` step name on `SubconsciousWorker`; replaced by `_step_pattern_match` / `pattern_match`.
- Old `behavioral_pattern` content fields: `vertical / class / slots / recurrence / sigma_confidence / status / decay_days`. New shape: `name / frequency / time_anchor / summary / confidence / last_seen_at / evidence_transcript_ids`.

---

## rc-0.4.0

### v0.5.0 §5 SubconsciousWorker — idle-gated 5-minute cognition tick (Phase 2 final)

- **`SubconsciousWorker`** (new — `backend/services/subconscious_worker.py`) — single daemon thread `subconscious-worker` that owns latent cognition. Tick interval `SUBCONSCIOUS_TICK_SEC` (default 300). Two gates: user-active (`last_user_message_at` within `SUBCONSCIOUS_IDLE_WINDOW_SEC`, default 1800), already-fired (`subconscious_last_fired_at > last_user_message_at`). When both pass, runs four steps in order, each isolated in its own `try/except`: (1) consolidate apex episodes per channel via `SuperEpisodeEncoderProcessor`, (2) `DecayEngineService.run_once()`, (3) `PatternExtractor.run_once()`, (4) `UserSummaryProcessor.send()`.
- Backpressure: when `bg_llm:queue` depth ≥ `SUBCONSCIOUS_BG_QUEUE_THRESHOLD` (default 20, headroom under `MAX_QUEUE_DEPTH=25`), steps 3 + 4 (LLM-heavy) are skipped; steps 1 + 2 still run.
- Re-entrancy: a non-blocking `threading.Lock` guards against overlapping ticks; concurrent `run_once()` returns `skipped='re_entrant'` without doing work.
- State persistence (`subconscious_last_fired_at`): MemoryStore key `subconscious:last_fired_at` for fast read; mirrored to `data_graph` (`kind='system'`, `key='subconscious_last_fired_at'`) for durability across restarts. Hydrated into the cached property on construction.
- Wiring: `WorkerManager.register_service('subconscious-worker', subconscious_worker)` in `backend/run.py`, between `background-llm-worker` and the optional services block.
- Tests (`backend/tests/test_subconscious_worker.py`, 12 unit, real `WorldState`, monkeypatched step methods): cold-boot fires; user-active gate skips; already-fired gate skips; fresh user message resets gate; one step raising does not block the others; bg-LLM saturated → only steps 1+2 run; concurrent `run_once()` returns `skipped='re_entrant'`; hydrate from durable store loads cached value across simulated restart; persist bumps timestamp to ~now; `_is_bg_llm_saturated` defaults to `False` on a poisoned `MemoryClientService`; synthesis self-skip detail propagates to `step.detail`.

### v0.5.0 §5 SubconsciousWorker — arbiter pass refinements

- **Hydrated already-fired gate** — when `_cached_last_fired` is loaded from durable storage but `WorldState.last_user_message_at` is still `None` (cold boot, no traffic since restart), `_check_gates()` now returns `"already_fired"` instead of falling through. Previously the worker would fire every tick after a process restart until the first user message arrived.
- **Backpressure import canary** — top-level `from services.background_llm_queue import QUEUE_KEY` added so a future module move surfaces at import time. Inside `_is_bg_llm_saturated` an `ImportError` now logs at `WARNING` (silent swallow would permanently bypass backpressure); other connection errors stay at `DEBUG`.
- **Decay engine cached on the worker instance** — `DecayEngineService` is built once on first decay step and reused. Previously a fresh instance was created every 5-minute tick, which reread `episodic-memory` config every cycle.
- **Persist log levels** — `_persist_last_fired` now logs `WARNING` on either MemoryStore or data_graph failure (was asymmetric: data_graph at `DEBUG`, MemoryStore at `WARNING`). Split-brain divergence is real state to surface.
- **Monotonic cadence** — `next_tick = monotonic() + interval` is now anchored after `run_once()` returns, not before. Long ticks (> 5 min) used to land `next_tick` in the past and immediately re-fire.
- **Gate-skip log** — bumped from `DEBUG` to `INFO` for operator visibility.
- **Dead-code cleanup tied to §4.4 daemon rip** — `DecayEngineService(decay_interval=…)` parameter and attribute removed (only consumer was the deleted daemon's sleep cadence); `decay_interval_seconds` orphan key dropped from `configs/agents/episodic-memory.json`; "Follows IdleConsolidationService pattern" docstring rewritten (class no longer exists); two dead constructor tests removed from `test_decay_engine_service.py`. Production prose updates in `user_summary_processor.py`, `user_message_processor.py`, `tests/test_super_episode_pipeline.py`, and `tests/test_episodic_redesign.py` route references to the live SubconsciousWorker step path.

### GPU-aware ORT install + runtime EP fallback

- `installer/install.sh` now detects the host GPU (NVIDIA via `nvidia-smi`; AMD via `/dev/kfd` + `amdgpu` module) and installs the matching `onnxruntime` wheel (`onnxruntime-gpu`, `onnxruntime-rocm`, or CPU) after venv setup. The GPU wheel is only swapped in after `pip install --dry-run` confirms it is reachable — CPU wheel always remains as fallback. ORT version pinned at `1.20.1` as a single source of truth in the installer. `ROCM_PIP_INDEX` env var overrides the AMD package index for air-gapped installs.
- `backend/requirements.txt`: `onnxruntime==1.20.1` pin dropped; `rapidocr_onnxruntime` transitively pulls the CPU wheel for dev workflows that bypass the installer.
- `backend/services/onnx_session.py` (new): single chokepoint for all `ort.InferenceSession` construction in the process. `choose_providers(model_path)` returns the ordered provider list and strips `CoreMLExecutionProvider` when any initializer dimension exceeds the Metal 16384 2D-texture ceiling. `build_session(path, opts, providers, log_prefix)` retries with CPU-only on construction failure. Metal texture-limit check previously lived in `embedding_service.py`; it now lives here.
- `EmbeddingService`, `VoiceService` (`api/voice.py`), and `Doc2QueryService` all route through `build_session` — no service constructs `ort.InferenceSession` directly.

---

## Recent

### Brain Memory Sub-Panel Rework (rc-0.3.3)
- Replaced `GET /system/observability/memory` (which double-counted `data_graph` rows under multiple labels and referenced MemoryStore keys that were never written in production) with `GET /system/observability/records?source=<episodes|user|system>&q=<str>&offset=<int>`
- New endpoint returns 250-row pages of `{created, last_accessed, key, value}` ordered `last_accessed_at IS NULL, last_accessed_at DESC, created DESC`
- Brain Memory sub-panel now renders a source switcher (Episodes / User / System), a search input, a paginated table, and a Load-more button
- Removed `/system/observability/traits` and `/system/observability/traits/<key>` endpoints (already ripped in 5c9b8c2 alongside the Understanding sub-tab)

### LUT Canonicalization Engine + Forget Action (2026-04-17)
- Replaced the ONNX contradiction classifier (`ContradictionClassifierService`, 5-way MLP head) with a deterministic 27-concept LUT engine inside `data_graph_service.store()`
- LUT lives at `backend/services/data_graph/assets/concept_lut.yaml` (27 hand-curated concepts: 3 immutable, 7 temporal, 17 coexist) with pre-shipped sqlite-vec embeddings at `concept_lut.sqlite` (~490 KB, 768-dim gte-modernbert)
- `store()` for `user_specific` writes: key embedding → KNN against `concept_lut.sqlite` (cosine threshold 0.80) → rule dispatch: `temporal` supersedes old value + creates supersedes edge; `coexist` stores additively; `immutable` blocks write and returns conflict dict
- `store()` for `system` kind: cosine match against existing key embeddings (threshold 0.80) → uniform temporal supersession; below threshold → additive insert
- LUT misses logged to new `concept_lut_misses` observability table (kind, key, value preview, count, first/last seen)
- `forget()` action added to `data_graph_service`: hard-delete with rule-aware semantics — temporal removes full version chain; coexist removes single value by exact match; immutable deletes the single row
- `memory_skill.handle_memory()` now handles `forget` action; `store` returns one of 12 structured response templates so the LLM can self-correct on canonicalization surprises
- `memory_skill.TOOL_SCHEMA` description injects all 27 canonical keys + 5 store rules + niche-fact fallback so the LLM canonicalizes at extraction time, not only at write time
- `pending_contradictions` table and `PendingContradictionService` deleted (no call sites outside the cleanup loop; immutable conflicts surface via the returned conflict dict)
- `("contradiction", "contradiction")` removed from `onnx_inference_service.MODEL_REGISTRY`; contradiction `.npz` head no longer loaded or downloaded; deliberation-score head unaffected
- Generator script `backend/utils/generate_concept_lut.py`: reads YAML, embeds canonical keys via gte-modernbert, writes `lut_concepts` + `lut_embeddings` vec0 virtual table into `concept_lut.sqlite`; run with `python -m utils.generate_concept_lut` after YAML changes
- One-shot migration `backend/utils/migrate_canonicalize_user_keys.py`: backfills existing `user_specific` rows to canonical keys; idempotent
- Cosine formula: `cos = max(0.0, 1.0 - distance ** 2 / 2.0)` via `_l2_dist_to_cosine()`
- Constants: `_CONCEPT_LUT_THRESHOLD = 0.80`, `_SYSTEM_KEY_THRESHOLD = 0.80`, `_RECALL_COSINE_FLOOR = 0.42`
- New nightly scenarios: `096-lut-canonicalize-residence-temporal.yaml`, `097-lut-coexist-favorite-foods.yaml`, `098-job-title-temporal-supersedes.yaml`, `099-lut-miss-recorded.yaml`, `100-contradiction-classifier-removed.yaml`

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
