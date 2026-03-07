# Chalie Glossary — Project-Specific Terms and Definitions

This glossary defines key terms, concepts, and architecture-specific vocabulary used throughout the Chalie project documentation. Whether you're a new user exploring cognitive assistant capabilities or a contributor diving into the codebase, these definitions provide clarity on how Chalie's systems work together to create an intelligent, memory-rich conversational experience.

---

## A

**ACT Loop** — The action execution pipeline triggered when ACT mode is selected. Runs innate skills (recall, memorize, introspect, etc.) in sequence before re-routing through the Mode Router to a terminal response mode. Uses `qwen3:8b` model and does not include soul.md personality layer.

**Adaptive Boundary Detector** — A dynamic topic boundary detection system using NEWMA (Neural Exponential Weighted Moving Average) combined with Transient Surprise signals, feeding into a Leaky Accumulator to determine when conversation topics shift. Cold start uses static 0.55 threshold; active conversations use learned thresholds.

**Ambient Inference Service** — Background service that continuously analyzes interaction patterns and environmental context to inform autonomous behavior decisions without explicit user triggers.

**Autonomous Actions** — Spontaneous behaviors generated during idle periods by the Cognitive Drift Engine. Includes six action types ranked by priority: CommunicateAction (10), SuggestAction, NurtureAction, PlanAction, ReflectAction, SeedThreadAction. Each follows specific eligibility gates and relationship phase awareness.

---

## C

**Cognitive Drift Engine** — The Default Mode Network implementation that generates spontaneous thoughts during idle periods when all queues are empty. Uses spreading activation from semantic memory to seed LLM synthesis, storing results as drift gists that surface in conversation context. Attention-gated to skip generation when user is in deep focus mode.

**Cognitive Reflex Service** — Learned fast-path routing via semantic abstraction. Performs heuristic pre-screen (~1ms) followed by sqlite-vec cosine search (~5-20ms) to bypass the full pipeline for self-contained queries. Uses rolling-average centroids that generalize from observed examples and self-correct per cluster via user corrections and shadow validation.

**Context Assembly Service** — Unified retrieval service pulling context from all six memory layers (working memory, moments, facts, gists, episodes, procedural/concepts) with weighted budget allocation. Procedural hints surface learned action reliability for tools requiring ≥8 attempts, showing top 3 results with confidence labels.

**Context Warmth** — A floating-point score (0.0–1.0) indicating how established a conversation topic is based on accumulated memory. Cold context (<0.3) favors CLARIFY mode; warm context (>0.7) favors RESPOND mode. Calculated from presence of facts, episodes, and semantic concepts in the current topic scope.

---

## D

**Default Mode Network (DMN)** — Borrowed from neuroscience, this refers to Chalie's idle-state cognitive system that generates spontaneous thoughts, maintains relationship threads, and pursues curiosity-driven exploration when not actively responding to user input. Implemented via Cognitive Drift Engine and Curiosity Thread Service.

**Deterministic Mode Router** — A mathematical scoring function (~5ms latency) that selects engagement modes using observable conversation signals without LLM involvement. Decouples mode selection from response generation for predictability, auditability, and speed. Routing decisions are logged to SQLite for inspection.

---

## E

**Episodic Memory** — Long-term memories of specific interactions stored in SQLite with rich metadata including intent, context, action, emotion, outcome, gist summary, salience score, freshness decay factor, 768-dim embedding vector, and open loops. Retrieved via hybrid vector + FTS search for conversationally relevant recall.

---

## G

**Gist Storage Service** — MemoryStore-backed short-term memory layer providing fast deduplicated context retrieval. Stores concise summaries of recent interactions with automatic TTL-based expiration. Serves as the primary working memory for active conversations before consolidation into episodic storage.

---

## I

**Innate Skills** — Eight non-LLM cognitive operations that execute in sub-second timeframes (<50ms to <500ms): `recall` (unified retrieval), `memorize` (gist/fact storage), `introspect` (self-examination of context and world state), `associate` (spreading activation), `schedule` (reminder management), `autobiography` (narrative synthesis), `list` (deterministic list operations), `focus` (attention session control).

**Innate Skills Registry** — Centralized service (`services/innate_skills/registry.py`) that discovers, validates, and dispatches innate skill invocations. Provides uniform interface for all skills regardless of underlying implementation complexity.

---

## L

**Leaky Accumulator** — A decay-based state tracker used in the Adaptive Boundary Detector to smooth topic boundary signals over time. New evidence increases the accumulator; idle periods cause gradual leakage (decay). When threshold is exceeded, a new topic boundary is declared.

---

## M

**Memory Layers** — The six-tier hierarchical memory architecture: (1) Working Memory for immediate context, (2) Moments for pinned bookmarks, (3) Facts for structured knowledge, (4) Gists for short-term summaries, (5) Episodes for rich interaction memories, and (6) Procedural/Concepts for learned patterns and semantic graphs.

**MemoryStore** — In-memory, thread-safe key-value store backed by SQLite persistence. Manages queue topics (`prompt-queue`, `memory-chunker-queue`, etc.), conversation threads with 24h TTL, and runtime state without global locks. Core data structure for the single-process architecture.

**Mode Router Service** — The core routing service implementing deterministic mode selection via mathematical scoring over observable signals (context warmth, topic confidence, skill availability). Scores all five modes (ACT, RESPOND, CLARIFY, ACKNOWLEDGE, IGNORE) and selects highest; uses LLM tiebreaker only when top-2 scores are within margin.

**Moments Service** — Pinned message bookmark system allowing users to permanently save Chalie responses with LLM-enriched context summaries. Stores in SQLite `moments` table with semantic search via sqlite-vec, salience boosting for related episodes, and inline HTML card rendering in conversation spine. Background enrichment worker runs on 5-minute poll cycle.

---

## P

**Procedural Memory Service** — Learned action reliability tracking system that records tool invocation outcomes (success/failure, execution time) to build confidence profiles. Surfaces top-3 reliable tools for similar intents after ≥8 attempts with labeled confidence scores in context assembly.

**Prompt Queue** — Thread-based job queue (`services/prompt_queue.py`) holding incoming prompt jobs consumed by the Digest Worker. Implements producer-consumer pattern within single-process architecture, avoiding inter-process communication overhead while maintaining thread safety.

---

## R

**Radiant Design System** — The vanilla JavaScript frontend UI framework powering Chalie's web interface. Provides consistent styling, components, and interaction patterns across chat interface (`/interface/`), cognitive dashboard (`/brain/`), and onboarding wizard (`/on-boarding/`). No external CSS frameworks or build tools required.

**Routing Stability Regulator Service** — Single authority for mode router weight mutation running on 24-hour cycle. Reads pressure signals from multiple monitors (routing reflection, triage calibration) but is the only entity that mutates weights directly. Applies bounded corrections (max ±0.02/day per parameter) with 48-hour cooldown to prevent oscillation.

**Routing Reflection Service** — Idle-time peer review system using a strong LLM to analyze routing decision audit trails from SQLite. Performs dimensional analysis on past decisions and feeds pressure signals into the Routing Stability Regulator for gradual weight adjustments.

---

## S

**Salience Factors** — LLM-computed scores (0–3 scale) determining episode importance: novelty, emotional intensity, commitment level, and unresolved status. Combined via weighted formula `Base = 0.4·novelty + 0.4·emotional + 0.2·commitment`, then multiplied by 1.25 if unresolved loops exist. Retrieved episodes receive additional 0.2 boost (reconsolidation).

**Semantic Consolidation Service** — Background process converting episodic memories into semantic concepts via spreading activation and pattern extraction. Runs on configurable cycle to build the concept graph used for long-term knowledge retrieval and associative reasoning.

**Spreading Activation** — Semantic graph traversal algorithm starting from seed concepts and following weighted edges to related nodes. Used by Cognitive Drift Engine for spontaneous thought generation, Associate Skill for semantic queries, and Context Assembly Service for enriched retrieval beyond direct vector similarity.

---

## T

**Thread Conversation Service** — MemoryStore-backed conversation thread management with 24h TTL and confidence tracking via bounded reinforcement formula `new = current + (new_confidence - current) * 0.5`. Handles topic switching, exchange storage, and metadata persistence to SQLite for durability beyond in-memory expiry.

**Trusted Tools** — Python subprocess tools that run without Docker isolation, requiring no container runtime. Must be explicitly marked `"trust": "trusted"` in `embodiment_library.json` catalog (authors cannot self-declare). Include `runner.py` entry point instead of `Dockerfile`. Default tools are trusted for instant startup.

---

## U

**User Traits Service** — Per-user characteristic management across six categories: core identity, relationship dynamics, physical attributes, preferences, communication style, micro-preferences, and behavioral patterns. Supports category-specific decay rates to model trait stability over time. Behavioral patterns discovered by Temporal Pattern Service from interaction log mining.

---

## Related Documentation

- **[01-QUICK-START.md](01-QUICK-START.md)** — Installation and basic usage instructions
- **[04-ARCHITECTURE.md](04-ARCHITECTURE.md)** — System architecture reference for understanding data flow and components
- **[05-WORKFLOW.md](05-WORKFLOW.md)** — Detailed pipeline explanation from prompt to response
- **[07-COGNITIVE-ARCHITECTURE.md](07-COGNITIVE-ARCHITECTURE.md)** — Mode routing, decision flow, and innate skills deep dive
- **[08-DATA-SCHEMAS.md](08-DATA-SCHEMAS.md)** — Database schemas for episodes, threads, lists, and providers
- **[09-TOOLS.md](09-TOOLS.md)** — Tools system documentation including sandbox requirements and contracts

---

*Last updated: 2026-03-07 | Part of Chalie Documentation SEO Overhaul Phase 3*
