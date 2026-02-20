# Cognitive Architecture - Deterministic Mode Router & Decision Flow

## Overview

This document defines the cognitive architecture for mode routing and response generation. User input flows through classification, **deterministic mode routing** (~5ms), and mode-specific LLM generation.

Mode selection is decoupled from response generation. A mathematical router selects the engagement mode using observable signals, then a mode-specific prompt drives the LLM to generate the response. A small LLM tie-breaker handles ambiguous cases.

---

## Core Principles

### 1. Routing Is Deterministic, Generation Is Creative

**Routing (deterministic):** Which engagement mode to use — decided by a mathematical scoring function over observable signals (~5ms).

**Generation (creative):** What to say in that mode — decided by the LLM using a mode-specific prompt (~2-15s depending on mode).

This separation eliminates:
- Malformed JSON from conflating mode selection with response generation
- The fragile decision gate that overrode the LLM's mode choice
- Fatigue fallbacks on simple greetings
- ~15s latency for trivial interactions (ACKNOWLEDGE now uses qwen3:4b, ~2s)

### 2. Single Authority for Weight Mutation

Multiple monitors observe routing quality but **none modify weights directly**. They log pressure signals. A single `RoutingStabilityRegulator` (24h cycle) is the only entity that mutates router weights, with bounded corrections (max ±0.02/day) and 48h cooldown per parameter.

### 3. Self-Leveling via Context Warmth

The router naturally shifts behavior as memory accumulates:
- Cold context (new topic, no facts) → favors CLARIFY
- Warm context (established topic, facts present) → favors RESPOND
- This happens through signal-weighted scoring, not explicit rules

---

## Mode Types

### Primary Modes

#### ACT (Gather Information)
- **Type:** Continuation mode (triggers ACT loop, then re-routes)
- **Purpose:** Execute internal actions (memory queries, reasoning) before responding
- **Prompt:** `frontal-cortex-act.md` (no soul.md — pure action planning)
- **LLM Model:** qwen3:8b
- **After completion:** Re-routes through router (excluding ACT) → terminal mode

#### RESPOND (Give Answer)
- **Type:** Terminal mode
- **Purpose:** Provide substantive answer to user
- **Prompt:** `frontal-cortex-respond.md` + `soul.md`
- **LLM Model:** qwen3:8b

#### CLARIFY (Ask Question)
- **Type:** Terminal mode
- **Purpose:** Ask clarifying question when information is insufficient
- **Prompt:** `frontal-cortex-clarify.md` + `soul.md`
- **LLM Model:** qwen3:8b

#### ACKNOWLEDGE (Brief Acknowledgment)
- **Type:** Terminal mode
- **Purpose:** Brief social response (greetings, thanks, confirmations)
- **Prompt:** `frontal-cortex-acknowledge.md` (no soul.md — lightweight)
- **LLM Model:** qwen3:4b (~2s latency)

#### IGNORE (No Response)
- **Type:** Terminal mode
- **Purpose:** Empty/nonsense input
- **Behavior:** No LLM call, returns empty response immediately (0ms)

---

## Innate Skills (Action Types)

The ACT loop uses 5 innate cognitive skills. All are non-LLM operations (fast, sub-cortical).

| Skill | Category | Speed | Purpose |
|---|---|---|---|
| `recall` | memory | <500ms | Unified retrieval across ALL memory layers (working memory, gists, facts, episodes, concepts) |
| `memorize` | memory | <50ms | Store gists (short-term) and/or facts (medium-term) |
| `introspect` | perception | <100ms | Self-examination: context_warmth, FOK signal, recall_failure_rate, skill stats, world state |
| `associate` | cognition | <500ms | Spreading activation from seed concepts through semantic graph |
| `delegate` | delegation | variable | External specialist consultation (Phase 1: empty registry) |

### Backward Compatibility Aliases

| Old Name | Maps To |
|---|---|
| `memory_query` | `recall` |
| `memory_write` | `memorize` |
| `world_state_read` | `introspect` |
| `internal_reasoning` | `recall` |
| `semantic_query` | `recall` |

---

## Decision Flow

### Step 1: Classification (same as before)
```
User Input → Topic Classifier (embedding-based, deterministic)
  → {topic, confidence, similar_topic}
```

### Step 2: Context Assembly (same as before)
```
Classification Result → Load Context:
  - Gists, facts, working memory, world state
  - Episodes + concepts (vector similarity)
  - Calculate context_warmth (0.0-1.0)
```

### Step 3: Deterministic Mode Routing (~5ms)
```
Routing Signals → ModeRouterService.route()
  → Score all modes → Select highest
  → If ambiguous: LLM tie-breaker (qwen3:4b, ~2s)
  → {selected_mode, confidence, scores, tiebreaker_used}
```

### Step 4: Mode-Specific Generation
```
If IGNORE → return empty (no LLM call)
If ACT    → generate_with_act_loop() → re-route → generate_for_mode()
Otherwise → generate_for_mode(selected_mode)
  → Mode-specific prompt + context → LLM → response
```

---

## Deterministic Mode Router

### Signal Collection

The router collects signals from existing services (all Redis reads, ~5ms total) plus NLP regex patterns (<1ms):

**Context Signals (from Redis):**
- `context_warmth` (float 0-1)
- `working_memory_turns` (int 0-4)
- `gist_count` (int, excluding cold_start type)
- `fact_count` (int 0-50), `fact_keys` (list)
- `world_state_present` (bool)
- `topic_confidence`, `is_new_topic` (from classifier)
- `session_exchange_count` (int)

**NLP Signals (from raw text, regex):**
- `prompt_token_count`, `has_question_mark`, `interrogative_words`
- `greeting_pattern` (hey/hi/hello/yo/sup/etc.)
- `explicit_feedback` ('positive'/'negative'/None)
- `information_density` (unique tokens / total tokens)
- `implicit_reference` ("you remember", "we discussed", "last time")

### Scoring Formula

Each mode gets a weighted composite score:

| Mode | Base | Primary Boosters | Primary Penalties |
|------|------|-----------------|-------------------|
| RESPOND | 0.50 | context_warmth, fact_density, gist_density, question+context | cold start |
| CLARIFY | 0.30 | cold context, question+no_facts, new_topic+question | warm context (>0.6) |
| ACT | 0.20 | question+moderate_context, interrogative+gap_in_facts, implicit_reference | very cold, very warm+facts |
| ACKNOWLEDGE | 0.10 | greeting_pattern (+0.60), positive_feedback (+0.40) | has_question (-0.30) |
| IGNORE | -0.50 | empty_input only (+1.0) | everything else |

### Anti-Oscillation Guards

Per-request ephemeral adjustments (NOT weight mutations):
- If `previous_mode == 'ACT'` and ACT was unproductive → `act_score -= 0.15`
- If `previous_mode == 'CLARIFY'` → `respond_score += 0.05` (user just answered a question)

### Short-Term Hysteresis

Tracks `router_confidence` for last 3 exchanges on same topic. If all 3 were below 0.15 (low confidence streak), widens tie-breaker margin by +0.05 for that topic. Resets when confidence recovers.

### Tie-Breaker

When top 2 modes are within effective margin, invokes small LLM (qwen3:4b, ~2s):

```
effective_margin = base(0.20) - (base - min(0.08)) × warmth + semantic_uncertainty
```

Semantic uncertainty widens margin for:
- `implicit_reference` (+0.05)
- Low `information_density` (+0.03)
- `interrogative_words` without question mark (+0.03)

The tie-breaker prompt presents only the top 2 candidates with context. Falls back to higher-scoring mode on failure.

### Router Confidence

```
router_confidence = (top_score - second_score) / max(abs(top_score), 0.001)
```

Used for: offline tuning, detecting unstable routing regions, hysteresis trigger.

---

## ACT Loop (Simplified)

The ACT loop executes internal actions with safety limits. No decision gate or net value evaluation — the router already decided this is an ACT situation.

### Flow
1. Router selects ACT mode
2. LLM generates actions via `frontal-cortex-act.md` (action planning only)
3. Execute actions, append results to history
4. Check continuation: timeout or max_iterations (default 5) → stop
5. Otherwise loop (LLM re-plans with action results in context)
6. After loop ends → **re-route** through router (excluding ACT) → terminal mode
7. Generate terminal response via `generate_for_mode()`

### Continuation Check (Simplified)
```python
def can_continue(self):
    if elapsed >= cumulative_timeout: return False, 'timeout'      # 60s default
    if iteration_number >= max_iterations: return False, 'max_iterations'  # 5 default
    return True, None
```

### Termination Reasons
- `timeout` — cumulative timeout reached (safety limit)
- `max_iterations` — iteration cap reached

---

## Routing Feedback & Learning

### Post-Routing Feedback

After generation, detect router misclassification using user behavior signals from the NEXT exchange:

| Signal | Indicates | Logged As |
|--------|-----------|-----------|
| User immediately clarifies/repeats | RESPOND was wrong → should be CLARIFY | misroute (missed_clarify) |
| User asks memory-related follow-up | RESPOND was wrong → should be ACT | misroute (missed_act) |
| Negative reward after ACKNOWLEDGE | Should have been RESPOND | misroute (under_engagement) |
| Positive reward after any mode | Routing was correct | correct_route |

Feedback is stored in `routing_decisions.feedback` (JSONB).

### Routing Stability Regulator (24h Cycle)

Single authority for weight mutation. Follows `TopicStabilityRegulatorService` pattern:

1. Reads pressure signals from `routing_decisions` table (last 24h)
2. Computes: tie-breaker rate, mode entropy, misroute rate, ACT opportunity miss rate, reflection disagreement
3. Selects worst pressure, maps to single parameter adjustment
4. Max ±0.02 per day, 48h cooldown per parameter, hard bounds on all weights
5. **Closed-loop control**: Evaluates whether previous adjustments improved metrics. Reverts if no improvement or degradation detected.
6. Persists to `configs/generated/mode_router_config.json`

### Routing Reflection (Idle-Time Peer Review)

Strong LLM (qwen3:14b) reviews past routing decisions as a **consultant, not authority**:

1. During idle periods (all queues empty), dequeues from `reflection-queue`
2. Analyzes ambiguity dimensions (memory_availability, intent_clarity, tone_ambiguity, etc.)
3. Produces structured insight about WHERE ambiguity exists, not just WHAT to change
4. Stratified sampling: 50% low-confidence, 20% high-confidence, 30% tie-breaker decisions

**Anti-authority safeguards:**
- Confidence gate: only count disagreements with LLM confidence > 0.70
- User override: trust positive user feedback over LLM disagreement
- Sustained pattern required: >25% disagreement rate over 7 days to generate pressure
- Dimensional causality check: flagged dimensions must correlate with signal patterns

---

## Mode Entropy Monitoring

Healthy mode distribution ranges:

| Mode | Healthy Range | Red Flag |
|------|--------------|----------|
| RESPOND | 50-75% | >85% (overconfident) or <40% (under-committing) |
| CLARIFY | 8-20% | >30% (over-questioning) or <3% (never clarifying) |
| ACT | 5-15% | <2% (ACT death) or >25% (over-processing) |
| ACKNOWLEDGE | 3-12% | <1% (ignoring social cues) or >20% (trivializing) |
| IGNORE | <2% | >5% (dropping messages) |

---

## Logging & Observability

### Routing Decision Audit Trail

Every routing decision is logged to `routing_decisions` table:

```sql
CREATE TABLE routing_decisions (
    id UUID PRIMARY KEY,
    topic TEXT NOT NULL,
    exchange_id TEXT,
    selected_mode TEXT NOT NULL,
    router_confidence FLOAT,
    scores JSONB NOT NULL,          -- all mode scores
    tiebreaker_used BOOLEAN,
    tiebreaker_candidates JSONB,
    margin FLOAT,
    effective_margin FLOAT,
    signal_snapshot JSONB NOT NULL,  -- full signal vector
    weight_snapshot JSONB,
    routing_time_ms FLOAT,
    feedback JSONB,                 -- filled post-exchange
    reflection JSONB,               -- filled during idle
    previous_mode TEXT,
    created_at TIMESTAMP
);
```

### ACT Loop Iteration Logging

ACT loop iterations continue to log to `cortex_iterations` table for backward compatibility. Simplified fields (decision gate columns use zero-value placeholders).

### Log Prefixes

```
[ROUTER] Mode selected: RESPOND (confidence: 0.85, 2.3ms)
[ROUTER] Tie-breaker invoked: RESPOND vs CLARIFY → RESPOND
[MODE:ACT] [ACT LOOP] Iteration 0: executing 2 actions
[MODE:RESPOND] Generating response via frontal-cortex-respond.md
```

---

## Default Mode Network (Cognitive Drift Engine)

The cognitive drift engine models the brain's Default Mode Network — generating spontaneous internal thoughts during idle periods. These thoughts emerge from residual activation in the semantic memory network and are grounded by episodic experience.

### Drift Cycle

```
All queues idle? ──no──→ skip
      │yes
Recent episodes? ──no──→ skip (nothing to think about)
      │yes
Fatigued? ──yes──→ skip (budget exhausted)
      │no
Select seed concept (weighted random)
      │
Spreading activation (depth 2)
      │
Activation energy > 0.4? ──no──→ skip (weak associations)
      │yes
Retrieve grounding episode
      │
LLM synthesis → reflection | question | hypothesis
      │
Store as drift gist (surfaces in frontal cortex context)
```

### Seed Selection Strategies

| Strategy | Weight | Source |
|---|---|---|
| Decaying | 40% | Concepts with fading strength (0.2 < strength < 2.0), ordered by weakest first |
| Recent | 30% | Concepts linked to the most recent episode |
| Salient | 20% | Concepts related to the highest-salience episode in the last 7 days |
| Random | 10% | Any active concept with confidence >= 0.4 |

### Safeguards

- **Per-concept cooldown** (60min): Prevents circular rumination on the same concept
- **Fatigue budget** (2.5 per 30min): Stronger activations consume more budget, throttling drift naturally
- **Stochastic jitter** (±30%): Check interval varies between 210-390s (base 300s)
- **Long gap probability** (10%): Occasional extended silence (1.8-2.5x interval) for realism
- **Activation energy threshold** (0.4): Weak spreading activations don't produce thoughts
- **Decaying reinforcement**: Only decaying seeds get a +0.1 strength bump, and only on successful drift

---

## Future Enhancements

### Goal-Oriented Autonomous Thought
The system currently produces reactive responses (user-prompted) and associative drift thoughts (DMN). The next step is goal-oriented thought — forming intentions and pursuing them across time without user prompting.

**Prerequisites:**
- **Skills system**: Registry of capabilities the system can invoke autonomously
- **Discovery mechanism**: How the system discovers available skills and understands preconditions/effects

### Per-Message Encoding
Shift from complete-turn encoding to per-message encoding where each message triggers its own independent memory cycle.

---

## Glossary

- **Mode Router:** Deterministic mathematical function that selects engagement mode from observable signals
- **Tie-Breaker:** Small LLM (qwen3:4b) consulted when top 2 modes are within effective margin
- **Routing Signals:** Observable features collected from Redis and NLP analysis (~5ms)
- **Effective Margin:** Dynamic threshold for tie-breaker invocation (narrows with context warmth)
- **Router Confidence:** Normalized gap between top 2 scores — measures routing certainty
- **Pressure Signal:** Metric logged by monitors, consumed by the single regulator
- **Terminal Mode:** Mode that produces a user-facing response (RESPOND, CLARIFY, ACKNOWLEDGE, IGNORE)
- **Continuation Mode:** Mode that triggers internal actions before re-routing (ACT only)
- **Context Warmth:** Signal (0.0-1.0) measuring how much context is available for the current topic
- **Anti-Oscillation Guard:** Per-request ephemeral score adjustment to prevent mode flip-flopping
- **Hysteresis:** Stabilization mechanism that widens tie-breaker margin on low-confidence streaks
