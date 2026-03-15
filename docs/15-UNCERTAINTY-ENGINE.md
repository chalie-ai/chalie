# Uncertainty Engine - Contradiction Detection & Resolution

## Overview

The Uncertainty Engine models epistemic confidence across Chalie's memory systems. It detects contradictions between memories, classifies them, and resolves or surfaces them according to a tiered strategy that protects user attention.

**Core insight:** Contradictions are not bugs — they are signals. Some indicate temporal change ("switched jobs"), some reveal genuine uncertainty ("likes Honda but buying Toyota"), and some expose stale or incorrect beliefs. The engine mirrors how humans handle internal conflict: notice, sit with the discomfort, gather evidence, and resolve when context allows.

**Design principle:** Background-first. The engine runs primarily during cognitive drift and ingestion. It only interrupts the active conversation when a contradiction involves a belief or trait that directly affects the current exchange.

---

## Memory Reliability Model

Every durable memory (trait, episode, concept, relationship) carries a **reliability state**:

| State | Meaning | Decay Multiplier |
|-------|---------|-----------------|
| `reliable` | Default. Normal confidence. | 1.0x (category-specific) |
| `uncertain` | Inferred/researched but unverified by user. | 1.5x |
| `contradicted` | Actively conflicts with another memory. | 2.0x |
| `superseded` | Temporally replaced. True in past tense only. | 3.0x (fast fade, preserved for "used to" queries) |

The `reliability` field is added to three durable stores:
- `user_traits` (already has `last_conflict_at` — this replaces and extends it)
- `episodes`
- `semantic_concepts`

Context assembly uses a **weighted formula** that prevents over-suppression of important contradicted memories:

```
assembly_weight = base_type_weight * importance * max(reliability_floor, reliability_factor)
```

Where `reliability_factor` is: `reliable=1.0`, `uncertain=0.6`, `contradicted=0.4`, `superseded=0.2`.
And `reliability_floor=0.3` ensures high-importance contradicted memories still surface (they carry the contradiction flag for the response pipeline to act on).

The decay engine reads the reliability field and applies the decay multiplier.

---

## Uncertainty as First-Class Citizen

Not all uncertainties are contradictions. The engine tracks four types:

| Type | Description | Example |
|------|-------------|---------|
| `contradiction` | Two memories that cannot both be true simultaneously | "likes Honda" vs "buying Toyota" |
| `unverified` | System discovered/inferred something but user hasn't confirmed | "Found a Docker feature in beta — would user want this?" |
| `stale` | Memory may be outdated based on temporal signals | "Mom lives in Canada" (3 years old, no reinforcement) |
| `ambiguous` | Memory is context-dependent or figurative | "Hates mornings" (literal? or just prefers afternoons?) |

Solo uncertainties (no second memory) are valid — the Docker research example has no conflicting memory, it's simply unverified knowledge that should be surfaced when relevant.

---

## Detection Contexts

### Context 1: Active Conversation (Ingestion)

Detection runs as a **parallel subprocess** after intent detection, not in the critical path:

```
User sends message
  -> Intent detection fires (main pipeline continues)
  -> Parallel: Uncertainty subprocess fires
     |
     +-> Embed user message
     +-> Vector search against traits + concepts (fast, <50ms)
     +-> Filter: high similarity + value divergence?
     |   +-> No matches: subprocess exits, zero cost
     |   +-> Matches found: pass to contradiction classifier
     |
     +-> Contradiction classifier (lightweight LLM):
         Input: new statement, matched memories, temporal context,
                memory metadata (created_at, reinforcement_count,
                last_reinforced_at, source, confidence)
         Output: classification + resolution recommendation
         Core question: "Can these both be true simultaneously?"
         (compatible co-existence is NOT a contradiction —
          "likes Honda" + "likes Toyota" = not contradictory,
          "likes Honda" + "hates Honda" = contradictory,
          "lives in Canada" + "lives in China" = contradictory)
         |
         +-> Temporal change detected (job switch, moved city):
         |   Auto-resolve silently. Supersede old memory.
         |   Old: "works at X" -> superseded, value -> "used to work at X"
         |   New: "works at Y" -> reliable
         |
         +-> Contradicts semantic/trait (non-temporal):
         |   Flag to response pipeline. Weave into RESPOND naturally:
         |   "You mentioned you like Toyota — interesting, I remember
         |    you being a Honda fan. What changed?"
         |
         +-> Contradicts episode only:
             Likely temporal. Background-resolve, don't interrupt.
```

**Key behaviors:**
- Not all messages trigger the full pipeline — only those where vector search finds high-similarity matches with divergent values
- The subprocess races the response pipeline. If contradiction is found before response generation completes, it gets woven into the response. If response already sent, contradiction is queued for drift
- Contradictions against semantics or traits (beliefs/facts) warrant weaving into response. Episode contradictions are background-resolved

### Context 2: Cognitive Drift (Background)

The bulk of contradiction work happens during idle periods. This integrates with the existing `CognitiveDriftEngine`:

```
Drift cycle fires
  -> Sample N memories (weighted by recency, access, strength)
  -> For each sampled memory:
     +-> Vector search for similar memories across stores
     |   (traits, concepts, episodes — cross-store)
     |
     +-> Pre-filter: topic + embedding similarity
     |   (vector search handles even old memories — the mom/Canada
     |    vs mom/China case gets caught because embeddings are similar
     |    regardless of age)
     |
     +-> For each candidate pair with high similarity + value divergence:
         +-> Deterministic resolution possible?
         |   (temporal signal, large confidence gap, one decayed)
         |   -> Resolve silently, log resolution in uncertainties table
         |
         +-> Both high-confidence, genuinely contradictory?
         |   -> Mark BOTH as "contradicted" (reliability field)
         |   -> Create uncertainty record
         |   -> Generate surface_context: description of when to surface
         |   -> Wait for natural conversational context
         |
         +-> One inferred, one explicit?
             -> Temporal preference to explicit
             -> Mark inferred as "uncertain"
             -> New reinforcing evidence can rehabilitate either
```

**Sampling strategy:** Don't check everything against everything (O(n^2)). Instead:
- Prioritize recently created/modified memories
- Prioritize memories with high access counts (frequently used = higher impact if wrong)
- **Topic domain rotation**: Cycle through domains on a schedule (e.g., cycle 1: work, cycle 2: personal, cycle 3: relationships, cycle 4: hobbies). This prevents cross-domain noise and ensures full coverage over time without expensive all-at-once sweeps
- Budget: ~10-20 memories per drift cycle, with vector search fan-out
- As memory grows (50k+), topic partitioning keeps cost constant per cycle — only the rotation period grows

### Context 3: Pre-Action (ACT Loop)

Before the ACT loop executes an action based on a memory:

```
ACT loop selects action based on memory X
  -> Check: is X.reliability in ('uncertain', 'contradicted')?
  -> If yes:
     Route through CLARIFY before executing.
     "I was going to [action], but I'm not certain about [memory].
      Should I proceed?"
  -> If no: execute normally
```

This prevents the system from acting on unreliable information.

### Context 4: Semantic Consolidation (Post-Processing)

When new concepts/relationships are extracted from episodes:

```
New concept extracted from episode
  -> Before storing: vector search for similar existing concepts
  -> If near-duplicate with different definition/value:
     -> Create uncertainty record (type: 'contradiction')
     -> Store new concept with reliability='contradicted'
     -> Mark existing concept as 'contradicted'
  -> If novel: store normally as 'reliable'
```

This catches contradictions at the point where episodic memory crystallizes into semantic knowledge.

---

## Severity Classification (Heuristic)

Severity is rule-based, not LLM-classified:

| Memory Type | Severity | Rationale |
|-------------|----------|-----------|
| Trait vs Trait | `critical` | Traits are beliefs/facts about the user. Getting these wrong breaks trust. |
| Concept vs Trait | `high` | Semantic knowledge contradicting a personal fact. |
| Concept vs Concept | `high` | Conflicting beliefs in the knowledge graph. |
| Episode vs Trait | `medium` | Episode may be contextual; trait is established. |
| Episode vs Concept | `medium` | Episode is narrative; concept is distilled knowledge. |
| Episode vs Episode | `low` | Episodes are fragile and temporal by nature. |
| Solo (unverified) | `low` | No conflict, just unconfirmed knowledge. |

The heuristic: **anything that made it to semantics or traits is a "belief" — contradicting a belief is serious. Episodes are observations — contradicting an observation is expected.**

---

## Resolution Strategies

### Temporal Supersession (Auto)
- **Trigger:** Clear temporal signal (job change, relocation, preference evolution)
- **Action:** Old memory reliability -> `superseded`, value annotated with "used to". New memory stored as `reliable`.
- **Feedback:** Semantic relationship created (type: `temporal_evolution`, e.g., Honda -> Toyota)

### Evidence Resolution (Auto)
- **Trigger:** New information reinforces one side of an existing uncertainty
- **Action:** Reinforced memory reliability -> `reliable`, conflicting memory -> `superseded` or decayed
- **Feedback:** Uncertainty record resolved, reinforcement count updated

### Confidence Dominance (Auto)
- **Trigger:** Large confidence gap (>2x) with clear source hierarchy (explicit > inferred)
- **Action:** Higher-confidence memory wins, lower one marked `uncertain`
- **Feedback:** Logged but not surfaced

### Natural Decay (Passive)
- **Trigger:** Contradicted/uncertain memory decays below floor
- **Action:** Memory deleted by decay engine as normal
- **Feedback:** Uncertainty record state -> `decayed`

### Conversational Clarification (Interactive)
- **Trigger:** High-severity contradiction surfaces in relevant conversational context
- **Action:** Woven into RESPOND naturally (not a separate clarification prompt)
- **Feedback:** User's response feeds back through ingestion pipeline:
  - Winner: reliability -> `reliable`, reinforcement boosted
  - Loser: reliability -> `superseded` (if temporal) or deleted (if wrong)
  - Related concepts strengthened/weakened
  - Semantic relationship created capturing the resolution
  - Episode created capturing the clarification exchange

---

## Surfacing Strategy

Uncertainties are surfaced when conversational context is relevant, not eagerly:

1. **At detection time:** The contradiction classifier generates a `surface_context` description (e.g., "user discusses cars or purchasing decisions")
2. **At surfacing time:** When context assembly retrieves a memory marked `contradicted`, check the linked uncertainty record's `surface_context` against the current topic
3. **Weave, don't interrupt:** Contradictions are woven into the response naturally, not presented as system alerts
4. **No re-surfacing:** If the user doesn't engage with a contradiction, it remains uncertain and decays naturally. Contradictions are only surfaced once — at detection time — woven into the current response
5. **Real-time generation:** The actual clarification phrasing is generated fresh at surfacing time (not pre-generated) to match current conversational tone and context

---

## Uncertainty Tolerance (Identity Trait)

Chalie's willingness to tolerate uncertainty is an **identity trait that self-tunes**, not a user setting:

- Starts at a neutral baseline
- Adjusts based on observed user behavior:
  - User frequently corrects Chalie -> lower tolerance (surface earlier, clarify more)
  - User rarely corrects, engages with clarifications -> maintain current level
  - User dismisses clarifications repeatedly -> higher tolerance (resolve more silently)
- Stored as a Chalie identity dimension, tuned by the same mechanisms as other identity traits
- Affects: surfacing threshold, auto-resolution aggressiveness, drift contradiction budget

---

## Database Schema

### New Table: `uncertainties`

```sql
CREATE TABLE IF NOT EXISTS uncertainties (
    id TEXT PRIMARY KEY,

    -- What's uncertain
    memory_a_type TEXT NOT NULL,        -- 'trait', 'episode', 'concept', 'relationship'
    memory_a_id TEXT NOT NULL,
    memory_b_type TEXT,                 -- NULL for solo uncertainties (unverified research, etc.)
    memory_b_id TEXT,

    -- Classification
    uncertainty_type TEXT NOT NULL,     -- 'contradiction', 'unverified', 'stale', 'ambiguous'
    severity TEXT NOT NULL,             -- 'low', 'medium', 'high', 'critical'

    -- Context
    detection_context TEXT NOT NULL,    -- 'ingestion', 'drift', 'pre_action', 'consolidation'
    reasoning TEXT,                     -- LLM explanation of why these conflict
    temporal_signal INTEGER DEFAULT 0,  -- 1 if temporal change pattern detected
    surface_context TEXT,               -- what conversational context should trigger surfacing

    -- Resolution
    state TEXT NOT NULL DEFAULT 'open', -- 'open' | 'resolved'
    resolution_strategy TEXT,           -- 'temporal_supersede', 'user_clarified',
                                        -- 'evidence_resolved', 'confidence_dominance', 'decayed'
    resolution_detail TEXT,             -- what was decided and why
    resolved_at DATETIME,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_uncertainties_state ON uncertainties(state);
CREATE INDEX IF NOT EXISTS idx_uncertainties_memory_a ON uncertainties(memory_a_type, memory_a_id);
CREATE INDEX IF NOT EXISTS idx_uncertainties_memory_b ON uncertainties(memory_b_type, memory_b_id);
CREATE INDEX IF NOT EXISTS idx_uncertainties_severity ON uncertainties(severity, state);
```

### Schema Modifications

```sql
-- Add reliability column to durable memory stores
ALTER TABLE user_traits ADD COLUMN reliability TEXT DEFAULT 'reliable';
ALTER TABLE episodes ADD COLUMN reliability TEXT DEFAULT 'reliable';
ALTER TABLE semantic_concepts ADD COLUMN reliability TEXT DEFAULT 'reliable';
```

---

## Service Architecture

### New Services

#### `UncertaintyService` (`backend/services/uncertainty_service.py`)
Core service. CRUD for uncertainty records, severity classification, resolution execution.
- `detect_contradiction(new_memory, memory_type)` -> scans for conflicts via vector search
- `create_uncertainty(memory_a, memory_b, type, context)` -> stores record, updates reliability fields
- `resolve_uncertainty(id, strategy, detail)` -> resolves record, updates memory reliability, creates feedback
- `get_active_uncertainties(severity_filter, topic_filter)` -> for drift and context assembly
- `check_memory_reliability(memory_type, memory_id)` -> pre-action safety check

#### `ContradictionClassifierService` (`backend/services/contradiction_classifier_service.py`)
LLM-based classification of detected conflicts.
- **Core question:** "Can these both be true simultaneously?"
- Input: two memories + temporal context + user traits + memory metadata (`created_at`, `reinforcement_count`, `last_reinforced_at`, `source`, `confidence`)
- Output: classification (temporal_change | true_contradiction | context_dependent | figurative | compatible), confidence, recommended resolution
- `compatible` classification = not a contradiction, memories can co-exist (e.g., "likes Honda" + "likes Toyota")
- Temporal evidence (memory age, reinforcement recency, event timestamps) strengthens auto-supersession decisions
- Uses lightweight model (same tier as triage: fast, structured output)

### Modified Services

#### `CognitiveDriftEngine`
- New drift action: `RECONCILE` — sample memories, cross-reference, detect contradictions
- Budget: 10-20 memory comparisons per drift cycle
- Interleaves with existing REFLECT/COMMUNICATE/PLAN actions

#### `SemanticConsolidationService`
- Post-extraction: check new concepts against existing for contradictions before storing
- If contradiction found: create uncertainty record, mark both concepts

#### `UserTraitService`
- Replace `last_conflict_at` logic with uncertainty creation
- On conflict: create uncertainty record instead of silent timestamp update
- On explicit correction: resolve any linked uncertainties

#### `DecayEngineWorker`
- Read `reliability` field, apply decay multiplier per the reliability model table
- On decay deletion: resolve linked uncertainty records as `decayed`

#### `ContextAssemblyService`
- Apply weighted reliability formula: `base_weight * importance * max(0.3, reliability_factor)` — deprioritizes unreliable memories without fully suppressing high-importance ones
- When assembling context for a topic: check for open uncertainties matching that topic
- If high-severity uncertainty exists for current topic: flag for response pipeline with both conflicting memories attached

#### `FrontalCortexService` (Response Generation)
- When contradiction flag is present in assembled context:
  - Include both conflicting memories in prompt
  - Instruct LLM to weave the contradiction naturally into the response
  - Not a separate CLARIFY mode — it's woven into whatever mode was selected (usually RESPOND)

#### `ACTDispatcherService`
- Pre-action reliability check: if action depends on unreliable memory, route through CLARIFY first

### New Worker

#### `UncertaintyWorker` (optional, or integrated into drift)
- Could be a dedicated worker thread for heavy contradiction detection sweeps
- Or: integrated as a drift action type (`RECONCILE`), which keeps the architecture simpler
- Recommendation: **start as a drift action**, graduate to dedicated worker if load demands it

---

## Pipeline Integration Points

```
                    INGESTION (active conversation)
                    ===============================
User message
  |
  +---> Intent detection (main path continues)
  |
  +---> [Parallel subprocess]
        Embed message -> vector search traits+concepts
        -> contradiction classifier (if matches found)
        -> result fed to response pipeline OR queued for drift
  |
  v
Response generated (with contradiction woven in, if flagged)


                    COGNITIVE DRIFT (background)
                    ============================
Drift cycle
  |
  +---> Existing: COMMUNICATE / REFLECT / PLAN / NOTHING
  |
  +---> New: RECONCILE
        Sample memories -> cross-store vector search
        -> deterministic pre-filter -> LLM classification
        -> auto-resolve OR create uncertainty record


                    CONSOLIDATION (background)
                    ==========================
Episode -> Concept extraction
  |
  +---> New concept -> vector search existing concepts
        -> if conflict: create uncertainty, mark both
        -> if novel: store as reliable


                    ACT LOOP (action execution)
                    ===========================
Action selected
  |
  +---> Check: is source memory reliable?
        -> if unreliable: CLARIFY before executing
        -> if reliable: proceed


                    DECAY ENGINE (background)
                    =========================
Decay cycle
  |
  +---> Read reliability field -> apply multiplier
        -> if deleted: resolve linked uncertainties as 'decayed'


                    RESOLUTION FEEDBACK
                    ====================
User clarifies (or new evidence arrives)
  |
  +---> Uncertainty resolved
        -> Winner: reliability='reliable', reinforcement++
        -> Loser: reliability='superseded' or deleted
        -> Semantic relationship created (type: resolution_type)
        -> Episode created capturing the exchange
```

---

## Implementation Status

All four phases are **complete and live**.

### Phase 1: Foundation ✅
1. Schema: `reliability` columns on `user_traits`, `episodes`, `semantic_concepts`; `uncertainties` table
2. `UncertaintyService`: CRUD, severity heuristic, rank-guard on `_set_reliability()`
3. Decay engine multiplier: `contradicted` ×0.5, `uncertain` ×0.75
4. Context assembly deprioritization: lower weight for non-reliable memories

### Phase 2: Detection ✅
5. Ingestion detection: `ContradictionClassifierService.check_ingestion()` — 600ms time-box, vector pre-screen, LLM classify in `digest_worker.generate_for_mode()`
6. Consolidation detection: `SemanticConsolidationService.consolidate_concept()` hook via `check_concept_conflict()`
7. Drift RECONCILE: `ReconcileAction` (priority 4, 30min cooldown) registered in `CognitiveDriftEngine`
8. Trait conflict path: `UserTraitService.store_trait()` creates uncertainty records on same-key conflicts

### Phase 3: Resolution ✅
9. Temporal auto-resolution: `_auto_supersede()` in `ReconcileAction`; `temporal_change` classification auto-resolves silently
10. Response weaving: `{{contradiction_context}}` in `frontal-cortex-respond.md`; `FrontalCortexService._inject_parameters()` builds hint block
11. ACT loop pre-action check: `ActDispatcherService._check_source_reliability()` — reduces confidence 40%, annotates `reliability_warning`
12. Resolution feedback loop: `UserTraitService.correct_trait()` resolves linked uncertainties with `strategy='user_clarified'`

### Phase 4: Self-Tuning ✅
13. Uncertainty tolerance identity vector: seeded in `schema.sql` (`baseline=0.5, min=0.2, max=0.8`)
14. Evidence-based resolution: `UncertaintyService.resolve_by_reinforcement()` — auto-resolves when reinforced confidence > 2× opposing
15. Tolerance nudge: `_nudge_uncertainty_tolerance()` called on `correct_trait()` (−0.03) and trait reinforcement (+0.01)
