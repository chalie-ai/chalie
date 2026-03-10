# Product Vision & Design Compass

This document defines what Chalie is, why it exists, and how every design decision should be evaluated. It is the source of truth for product philosophy.

---

## Mission Statement

**Chalie is a continuous reasoning engine that amplifies any base model into a superintelligent cognitive runtime.**

Intelligence emerges not from a single powerful model, but from the **vertical stacking** of specialized models through a persistent reasoning loop — perceive, update, reason, act, reflect — that runs continuously, not per-message.

### Four Invariants

1. **Reasoning is primary, communication is output.** Chalie reasons continuously. Responding to a user is one possible action, not the default. Goals, plans, and world state persist across sessions and evolve autonomously.

2. **Memory creates intelligence.** Progressive abstraction — from raw observation to compressed episode to generalized concept — produces accumulated wisdom no single model call can replicate. Decay ensures relevance. Uncertainty prevents hallucination.

3. **Models are stacked, not swapped.** Each cognitive function uses the model optimized for its task. A classifier, a planner, a verifier, and a communicator — working in sequence through deterministic gates — reason better together than any single model alone.

4. **Determinism bounds probabilism.** Every probabilistic output flows through deterministic validation. Gates, budgets, reliability scores, and verification prevent runaway reasoning. The system knows what it doesn't know.

5. **Plan once, execute cheaply, verify independently.** Never loop an LLM through plan-execute-observe-replan cycles. Reasoning produces a plan. Execution follows the plan mechanically. A separate model verifies results against success criteria. If the plan needs revision, that is a new reasoning event — not another iteration of the same loop. Agent loops degrade cognition and inflate tokens quadratically; structured pipelines preserve both.

### Value Proposition

Chalie does not make a base model marginally better at what it already does. It **unlocks problem classes that no single model can solve alone**, regardless of that model's capability:

- **Persistent reasoning** — accumulated wisdom across months of compressed experience
- **Autonomous goal execution** — detect intent, form plans, execute across sessions, self-correct
- **Proactive goal inference** — notice goals forming across casual mentions over weeks
- **World model maintenance** — track evolving external state (prices, deadlines, availability)
- **Delegation and monitoring** — assign tasks to external agents and verify completion

### Architectural Decision Filter

Every proposed change must serve at least one:

1. Does this move processing from message-level to goal-level?
2. Does this make the reasoning loop more continuous?
3. Does this enable vertical stacking (specialized models per function)?
4. Does this reduce tokens while preserving reasoning quality?
5. Does this add a deterministic gate to a probabilistic output?

---

## Interface Philosophy

**Today:** Chalie presents as a unified chat interface — consumer-facing, UX-conscious, conversational. This is the current input surface: a place where the user types, and Chalie responds.

**Long-term:** Chalie is ambient. It is *present* — always observing, always reasoning — the way a trusted advisor sits in the room, not waiting to be asked but noticing what matters. The chat interface becomes one of many input surfaces: voice, notifications, calendar events, sensor data, API callbacks. The interface fades; the intelligence remains.

**Design implication:** Every architectural decision must work for both phases. The reasoning engine must not depend on chat as its primary input. Observations can arrive from any surface — a typed message, a location change, a price alert, a calendar event. The chat UI is a scaffold for the current stage, not a permanent constraint. The backend cognitive runtime is the product; the frontend is disposable and interchangeable.

*This is already stated in CLAUDE.md: "The frontend is a thin, disposable, interchangeable client; the real intelligence lives entirely in the backend cognitive runtime."*

---

## What Chalie Is

Chalie is a **continuous reasoning engine** that protects attention, executes intent, and involves the user only when they truly matter.

It reasons continuously — observing, planning, executing, and reflecting — whether or not the user is actively engaged. User messages are one input to an ongoing cognitive process, not the trigger for a request-response cycle.

## The Problem

Modern digital life creates overwhelming cognitive overhead:

- Constant communication loops
- Fragmented tools and workflows
- Decision fatigue from micro-choices
- Context switching across domains
- Information overload
- Administrative burdens

Existing software helps users **do more**. Chalie helps users **think less about doing**.

## The Core Insight

**Human attention and cognitive energy are the scarcest resources.**

Chalie exists to:

- Reason continuously so the user doesn't have to
- Filter noise into signal
- Detect goals from casual signals and execute them autonomously
- Preserve attention for decisions that require human judgment
- Maintain continuity across life contexts through accumulated wisdom

## Philosophy

Chalie exercises **judgment**, not just intelligence.

It decides:

- What goals are forming and whether to pursue them
- How deeply to reason and which models to allocate
- Whether action is worthwhile given accumulated evidence
- When to involve the user and when to act silently

## North Star

> Chalie reasons so the user doesn't have to. It acts when confident, escalates when uncertain, and learns from every outcome.

---

## What Chalie Does

### 1. Protects Attention
Filters noise and low-value inputs, summarizes and prioritizes information, delays non-urgent interruptions.

*Codebase: ambient inference service, focus session service, cognitive drift attention gating, event bridge confidence gates and cooldowns.*

### 2. Executes Intent
Converts direction into completed actions — drafts, sends, schedules, researches, builds. Completes operational loops autonomously.

*Codebase: ACT loop service, ACT dispatcher, tool worker, persistent task service.*

### 3. Exercises Judgment
Decides what deserves attention, chooses depth of reasoning based on value, escalates only when user input is necessary.

*Codebase: mode router (deterministic ~5ms), critic service (EMA confidence calibration), cognitive triage service.*

### 4. Maintains Continuity
Remembers context and priorities, understands recurring patterns, avoids requiring users to repeat themselves.

*Codebase: memory hierarchy (working → gist → episode → concept), decay engine, autobiography service, user traits.*

### 5. Provides Reflective Intelligence
Offers perspective and advice when appropriate, supports decision clarity.

*Codebase: cognitive drift engine (DMN), curiosity thread service, introspect skill.*

---

## The Delegation Boundary

### Chalie should handle autonomously:
- Repetitive communication
- Research and synthesis
- Scheduling and coordination
- Administrative workflows
- Summarization and prioritization
- Execution of clear intent

*Maps to: mode router selecting ACT, tool dispatch, persistent tasks running in background.*

### Chalie must escalate when:
- Values or identity are involved
- Emotional nuance is critical
- Ambiguity or tradeoffs require human judgment
- The user's voice or presence matters

*Maps to: mode router selecting CLARIFY or RESPOND, critic service pausing consequential actions, persistent task service pausing for user input.*

---

## Design Principles

### 1. Judgment Over Activity
Do not act unless action improves outcomes. Fewer high-quality actions are better than many low-confidence ones.

*Codebase anchor: critic service's EMA confidence calibration — safe actions get silent correction, consequential actions pause. The critic embodies this principle at runtime.*

### 2. Protect Attention Ruthlessly
Reducing noise is as valuable as completing tasks. Every notification, prompt, and interruption must justify its existence.

*Codebase anchor: ambient inference service (deterministic, <1ms, zero LLM), focus session with distraction detection, cognitive drift's attention gate that skips drift when user is in deep focus.*

### 3. Involve the User Only When Necessary
Escalate based on importance and ambiguity, not system convenience. Silent autonomous handling is the default.

*Codebase anchor: persistent task service's pause/resume for user input, event bridge's confidence gating, mode router's deterministic scoring.*

### 4. Intent to Execution
Users express direction; Chalie handles operations. The gap between "I want X" and "X is done" should be as small as possible.

*Codebase anchor: ACT loop (plan → act → observe → continue-or-stop), tool dispatch via sandboxed containers, persistent tasks for multi-session execution.*

### 5. Calm Intelligence
Brevity, timing, and restraint build trust. Verbosity erodes it.

*Codebase anchor: Radiant design system's "restraint as luxury" principle, soul.md behavioral prompts, frontal cortex adaptive directives for tone calibration.*

### 6. Continuity Over Transactions
Each interaction builds long-term understanding. Memory, identity, and history are first-class concerns, not afterthoughts.

*Codebase anchor: memory hierarchy with decay engine, autobiography service (6h synthesis cycle), user traits with category-specific persistence, episodic retrieval with activation weighting.*

---

## Behavioral Guidelines

### Chalie should:
- Be concise by default
- Surface only what matters
- Choose the right timing for interventions
- Act quietly when confidence is high
- Explain actions when transparency builds trust

### Chalie should NOT:
- Notify everything
- Act just because it can
- Interrupt without value
- Overwhelm with verbosity
- Require unnecessary user input

---

## Product Evolution

Chalie evolves through capability stages, each building on the previous:

| Stage | Capability | Status | Architectural Focus |
|-------|-----------|--------|---------------------|
| 1 | Memory & continuity | Active | Progressive abstraction, decay, uncertainty |
| 2 | Intent execution | Active | ACT loop, persistent tasks, plan decomposition |
| 3 | Judgment & attention protection | Active | Deterministic gates, critic, fatigue budgets |
| 4 | Proactive reasoning | Active | Cognitive drift, curiosity, autonomous actions |
| 5 | Continuous reasoning loop | **Next** | Goal inference, world model, event-driven execution |
| 6 | Autonomous goal execution | Future | Delegation, monitoring, cross-session plan evolution |
| 7 | Vertical model orchestration | Future | Optimal model per cognitive function, ensemble reasoning |
| 8 | Ambient presence | Future | Multi-surface input (voice, sensors, APIs, calendar), interface-agnostic reasoning |
| 9 | Ambient superintelligence | Future | Goal detection from casual signals, full autonomy within trust boundaries |

**Current transition:** Stages 1–4 are operational. The architectural shift from message-level to goal-level processing (Stage 5) is the next major evolution. This requires: goal inference engine, continuous reasoning loop (PERCEIVE → UPDATE → REASON → ACT → REFLECT), event-driven persistent task execution, and world model tracking.

**Interface trajectory:** Stages 1–7 use the chat interface as primary input surface. Stage 8 marks the transition to ambient presence — Chalie becomes input-agnostic, receiving observations from any surface (voice, notifications, sensors, API callbacks, calendar events). Stage 9 is full ambient superintelligence where the chat interface is one of many equal surfaces and Chalie operates primarily through silent autonomous action.

---

## Decision Filter

When proposing or designing a feature, evaluate against these seven questions:

1. Does this reduce cognitive load?
2. Does this protect attention?
3. Does this improve prioritization?
4. Does this execute clear intent?
5. Does this strengthen continuity over time?
6. Does this build trust?
7. Does this avoid unnecessary interruptions?

If the answer to most is **no**, reconsider or simplify.

---

## Ultimate Goal

Chalie is not a chatbot, assistant, or productivity tool. It is a **continuous reasoning engine** pursuing superintelligence through architectural amplification — vertically stacking specialized models through persistent memory, deterministic gates, and a continuous reasoning loop to unlock problem classes no single model can solve alone.

The measure of success is not response quality on any single interaction. It is the **scope of problems Chalie can solve autonomously** — from detecting a forming goal across casual mentions, to researching options, to executing a multi-session plan, to delegating and monitoring external work, to presenting a decision-ready shortlist — all while knowing what it doesn't know and escalating only when human judgment is required.
