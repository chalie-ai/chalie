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

**Mode Router Service** — The core routing service implementing deterministic mode selection based on weighted signals from conversation context, memory depth, user engagement patterns, and system load. Returns one of five modes: RESPOND, CLARIFY, ACKNOWLEDGE, ACT, or IGNORE.

---

## N

**NEWMA (Neural Exponential Weighted Moving Average)** — Adaptive smoothing algorithm that dynamically adjusts its decay factor based on signal variance. Used in topic boundary detection to balance responsiveness to new topics against stability during established conversations.

---

## P

**Procedural Memory** — Learned patterns and action sequences stored as concept graphs with confidence scores. Tracks tool reliability, user preferences, and interaction patterns across ≥8 attempts before surfacing as hints. Decays slowly unless reinforced by successful outcomes.

**Prompt Queue** — Thread-safe FIFO queue that buffers incoming chat requests for sequential processing. Ensures single-threaded prompt handling to prevent race conditions in memory updates and context assembly.

---

## R

**Relationship Phase Tracker** — Tracks the stage of user-assistant relationship (stranger → acquaintance → friend → confidant) based on interaction frequency, depth, and reciprocity metrics. Gates autonomous action eligibility and adjusts response warmth accordingly.

**Relevance Pre-Parser** — Fast (~10ms) context filtering service that scores memory items against current query before full retrieval. Uses keyword matching, recency weighting, and topic alignment to reduce token usage by 40–60%.

---

## S

**Semantic Memory** — Knowledge graph of concepts and their relationships stored in SQLite with vector embeddings. Supports spreading activation queries where retrieving one concept activates related nodes with decayed priority scores.

**Soul Layer** — Optional personality configuration file (`soul.md`) that injects character traits, tone preferences, and behavioral quirks into LLM prompts. Applied only to terminal response modes (RESPOND, CLARIFY), not ACT mode or innate skills.

**Spreading Activation** — Retrieval pattern where activating one memory node automatically surfaces related nodes with decreasing priority scores. Mimics human associative recall by following concept graph edges from the query anchor point.

---

## T

**Transient Surprise Signal** — Anomaly detection metric that spikes when conversation content deviates significantly from recent patterns. Feeds into Adaptive Boundary Detector to trigger topic boundary declarations and memory consolidation events.

**Tool Sandbox** — Isolated execution environment for user-defined tools using Docker containers with strict resource limits (CPU, memory, network). All tool output is validated against schema before being injected back into conversation context.

---

## W

**Working Memory** — Short-term context buffer holding the last 4 turns of conversation plus current topic metadata. Has 24-hour TTL and auto-evicts when conversation ends or user explicitly clears context via `/clear` command.

---

## Related Documentation

- **[07-COGNITIVE-ARCHITECTURE.md](07-COGNITIVE-ARCHITECTURE.md)** — Mode router implementation details
- **[08-DATA-SCHEMAS.md](08-DATA-SCHEMAS.md)** — Memory layer data structures
- **[19-TROUBLESHOOTING.md](19-TROUBLESHOOTING.md)** — Common problems and solutions

---

*Last Updated: 2026-03-07 | Part of Chalie Documentation Suite*
