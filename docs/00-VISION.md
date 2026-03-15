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

*Codebase: ambient inference service, focus session service, reasoning loop attention gating, event bridge confidence gates and cooldowns.*

### 2. Executes Intent
Converts direction into completed actions — drafts, sends, schedules, researches, builds. Completes operational loops autonomously.

*Codebase: ACT loop service, ACT dispatcher, tool worker, persistent task service.*

### 3. Exercises Judgment
Decides what deserves attention, chooses depth of reasoning based on value, escalates only when user input is necessary.

*Codebase: message gate service (deterministic ONNX mode gate, ~5ms), mode router (deterministic scorer for non-user flows), critic service (post-execution learning signal with EMA confidence calibration), autonomous execution gate (consequence tier + domain confidence).*

### 4. Maintains Continuity
Remembers context and priorities, understands recurring patterns, avoids requiring users to repeat themselves.

*Codebase: memory hierarchy (working → gist → episode → concept), decay engine, autobiography service, user traits.*

### 5. Provides Reflective Intelligence
Offers perspective and advice when appropriate, supports decision clarity.

*Codebase: reasoning loop service (event-driven continuous reasoning, replaces timer-based DMN), curiosity thread service, introspect skill, reflect skill (on-demand + automatic post-loop), ACT orchestrator auto-reflection trigger.*

---

## The Delegation Boundary

### Chalie should handle autonomously:
- Repetitive communication
- Research and synthesis
- Scheduling and coordination
- Administrative workflows
- Summarization and prioritization
- Execution of clear intent

*Maps to: message gate routing to ACT pipeline, tool dispatch, persistent tasks running in background.*

### Chalie must escalate when:
- Values or identity are involved
- Emotional nuance is critical
- Ambiguity or tradeoffs require human judgment
- The user's voice or presence matters

*Maps to: mode router selecting RESPOND, ACT loop requesting clarification as part of its reasoning, persistent task service pausing for user input.*

---

## Design Principles

### 1. Judgment Over Activity
Do not act unless action improves outcomes. Fewer high-quality actions are better than many low-confidence ones.

*Codebase anchor: critic service's post-execution learning signal — EMA confidence calibration tracks action quality over time. The message gate's deterministic ONNX scoring and the ACT loop's iteration budget embody this principle at runtime.*

### 2. Protect Attention Ruthlessly
Reducing noise is as valuable as completing tasks. Every notification, prompt, and interruption must justify its existence.

*Codebase anchor: ambient inference service (deterministic, <1ms, zero LLM), focus session with distraction detection, reasoning loop's attention gate that skips reasoning when user is in deep focus.*

### 3. Involve the User Only When Necessary
Escalate based on importance and ambiguity, not system convenience. Silent autonomous handling is the default.

*Codebase anchor: persistent task service's pause/resume for user input, event bridge's confidence gating, mode router's deterministic scoring.*

### 4. Intent to Execution
Users express direction; Chalie handles operations. The gap between "I want X" and "X is done" should be as small as possible.

*Codebase anchor: ACT loop (plan → act → observe → continue-or-stop), tool dispatch via sandboxed containers, persistent tasks for multi-session execution, autonomous execution gate for goal auto-acceptance.*

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

| Stage | Capability | Status | What it delivers |
|-------|-----------|--------|-----------------|
| 1 | Memory & continuity | **Complete** | Progressive abstraction, decay, uncertainty |
| 2 | Intent execution | **Complete** | ACT loop, persistent tasks, plan decomposition |
| 3 | Judgment & attention protection | **Complete** | Deterministic gates (message gate, ONNX classifiers), post-execution critic |
| 4 | Proactive reasoning | **Complete** | Reasoning loop, curiosity, autonomous actions |
| 5 | Continuous reasoning loop | **Complete** | Goal inference, world model, event-driven signal processing |
| 6 | True autonomy | **Complete** | Consequence classifier, domain confidence from memory, autonomous execution gate |
| 7 | Situational intelligence | **Next** | Chalie understands *what's happening right now* — not just what you said |
| 8 | Expanded perception | Future | Chalie sees beyond the chat window — calendar, email, files, APIs, webhooks |
| 9 | Cognitive OS | Future | Chalie becomes the shared cognitive layer for external agents |

### Stage 7: Situational Intelligence

Chalie has all the raw signals — place, energy, attention, focus, topics, identity, engagement, world state. But nothing integrates them. The drift engine checks focus directly. Autonomous actions each check spark phase directly. Context assembly formats ambient as flat text. Every consumer makes its own isolated judgment from the same fragmented data.

Stage 7 builds the **binding layer**: a continuously updated situation model that computes composite scores (interruptibility, receptiveness, cognitive load, proactivity budget) from existing signals. Every downstream consumer reads from one place instead of making its own assessment.

**What changes for the user:**
- Responses adapt to context — shorter when you're low energy, deeper when you're engaged
- Proactive thoughts arrive at the right moment, not on a fixed timer
- Conversation phase awareness — Chalie doesn't introduce new topics when you're wrapping up
- Memories surface because of where/when you are, not just what you said

**What it requires:** SituationModelService (deterministic, <1ms), conversation phase tracking, behavioral adaptation rules, situation-aware prompt injection. Optional: ONNX emotion/engagement/tone models for richer signals. No new external data sources — this stage works with signals Chalie already has.

**Plan:** `plans/situational-awareness-7-10.md` (Waves 1-4)

### Stage 8: Expanded Perception

Everything before Stage 8, Chalie is still a chat app that thinks in the background. Its only window into the user's world is what the user types and what the browser reports. Stage 8 gives Chalie eyes.

**New input surfaces:**
- Calendar integration (knows you have a meeting in 30 minutes, surfaces prep notes)
- Email/notification ingestion (sees incoming signals without the user forwarding them)
- File system watchers (notices when documents change)
- API webhooks (price alerts, deployment notifications, external event triggers)
- Sensor data (wearables, IoT, location beacons)

**What changes for the user:**
- "You have a meeting with Sarah in 20 minutes — last time you discussed the Q3 roadmap, here's where you left off"
- Price alert fires → Chalie researches options and presents a recommendation when you're available
- Document updated in shared drive → Chalie reads the diff and surfaces what matters to you

**What it requires:** Input adapter framework (calendar, email, webhook, file watcher), signal normalization into the reasoning loop, privacy controls (what Chalie can and cannot see), and the situation model from Stage 7 to know *when* to surface information.

### Stage 9: Cognitive OS

Chalie stops being an application and becomes a **cognitive operating system**. External agents (coding tools, research agents, communication agents) plug into Chalie instead of building their own memory, judgment, and user models.

**Agent API (four operations):**

| Operation | What it does | Maps to |
|-----------|-------------|---------|
| **Query** | "What does the user prefer about X?" | Context assembly + episodic retrieval |
| **Observe** | "I noticed Y while doing Z" | Observation stream → memory pipeline |
| **Judge** | "Should I proceed with X or escalate?" | Reasoning loop + uncertainty engine |
| **Report** | "Task complete, here's what happened" | World model update + episode creation |

**What changes for the user:**
- A coding agent in an IDE asks Chalie "what does the user care about in auth?" and gets reliability-weighted context accumulated over months
- Multiple specialized agents coordinate through Chalie's shared cognitive layer
- The user's preferences, history, and judgment are available to every tool they use, without re-explaining

**What it requires:** REST API for agent operations, agent registration, delegation and monitoring, cross-agent knowledge sharing, and the full memory + autonomy stack from Stages 1-8 as the foundation.

**Current transition:** Stages 1–6 are complete. Stage 7 — situational intelligence — is next. It deepens *processing quality* with signals Chalie already has. Stage 8 then expands *what Chalie can perceive*. Stage 9 opens Chalie as a platform for external agents. Each stage builds on the previous — situational intelligence makes perception useful (knowing *when* to surface information), and perception makes the Cognitive OS valuable (agents need a world-aware cognitive layer, not just memory retrieval).

### The Cognitive OS Endgame (Stage 9)

At Stage 9, Chalie is not an application — it is a **cognitive operating system**. The same way a traditional OS provides memory management, process scheduling, and inter-process communication to applications, Chalie provides persistent memory, reasoning, judgment, and user understanding to specialized agents. No agent needs its own memory system or user model — Chalie is the shared cognitive layer.

The path to Stage 9 is sequential and deliberate: situational intelligence (Stage 7) gives Chalie contextual awareness with existing signals. Expanded perception (Stage 8) gives it eyes beyond the chat window. The Cognitive OS (Stage 9) opens that intelligence as a platform. Each layer makes the next one valuable — without situational awareness, perception is noise; without perception, the agent API has nothing to share.

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
