# Product Vision & Design Compass

This document defines what Chalie is, why it exists, and how every design decision should be evaluated. It is the source of truth for product philosophy.

---

## What Chalie Is

Chalie is a **personal intelligence layer** that protects attention, executes intent, and involves the user only when they truly matter.

It stands in for the user where their involvement adds little value, and involves them where judgment, identity, or values are required.

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

- Filter noise into signal
- Preserve attention for high-value thinking
- Execute clear intent without supervision
- Maintain continuity across life contexts

## Philosophy

Chalie exercises **judgment**, not just intelligence.

It decides:

- What deserves attention
- How deeply to think
- Whether action is worthwhile
- When to involve the user

## North Star

> Chalie handles what doesn't require the user and involves them when it truly matters.

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

*Codebase anchor: mode router's ACKNOWLEDGE mode (brief social response, minimal interruption), persistent task service's pause/resume for user input, event bridge's confidence gating.*

### 4. Intent to Execution
Users express direction; Chalie handles operations. The gap between "I want X" and "X is done" should be as small as possible.

*Codebase anchor: ACT loop (plan → act → observe → continue-or-stop), tool dispatch via sandboxed containers, persistent tasks for multi-session execution.*

### 5. Calm Intelligence
Brevity, timing, and restraint build trust. Verbosity erodes it.

*Codebase anchor: Radiant design system's "restraint as luxury" principle, soul.md behavioral prompts, mode router's ACKNOWLEDGE for moments that need presence rather than substance.*

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

Chalie evolves through trust-building stages:

| Stage | Capability | Status |
|-------|-----------|--------|
| 1 | Noise reduction and summarization | Active |
| 2 | Intent execution and task completion | Active |
| 3 | Context memory and continuity | Active |
| 4 | Priority intelligence and timing judgment | Active |
| 5 | Proactive guidance and insights | Active |
| 6 | Trusted selective autonomy | Future |
| 7 | Personal operating layer and ecosystem | Future |
| 8 | Ambient intelligence | Future |

**Reconciliation note:** The runtime uses a 5-phase trust model in `spark_state_service.py` that maps to stages 1–5 above. Stages 6–8 represent the long-term product vision and do not yet have runtime counterparts. Features should support this progression rather than skip stages prematurely.

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

Chalie is not a chatbot or productivity tool. It is a **cognitive layer** that restores human attention as a protected resource and enables people to operate with clarity, focus, and calm control.
