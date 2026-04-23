# Product Vision & Design Compass

Source of truth for what Chalie is, why it exists, and how every design decision should be evaluated.

---

## Vision

**Chalie is a Life OS.** One persistent intelligence that sees your entire digital life, understands what matters, and handles it — so you stop managing and start living.

It acts when confident, asks when uncertain, and learns from everything. Success is not measured by any single response, but by how much of your life just works without you thinking about it.

---

## The Problem

Modern digital life creates overwhelming cognitive overhead: constant communication loops, fragmented tools, decision fatigue, context switching, information overload, and administrative burdens.

Existing software helps users **do more**. Chalie helps users **think less about doing**.

**Human attention and cognitive energy are the scarcest resources.** Chalie exists to protect them.

---

## Core Principles

1. **Judgment over activity.** Don't act unless action improves outcomes. Fewer high-quality actions beat many low-confidence ones.
2. **Restraint builds trust.** Start conservative, grow with context. Silence over noise. Every token, notification, and action must earn its place.
3. **Continuity is intelligence.** Memory with decay. Identity through compression. The persistent runtime — not any single response — is the product.
4. **Constraints are features.** Capacity bounds, decay rates, and token budgets are what make intelligence emerge. Don't remove them to "improve."
5. **See everything, show only what matters.** Ingest broadly. Understand deeply. Surface selectively.

---

## Invariants

1. **Reasoning is primary, communication is output.** Responding to a user is one possible action, not the default. Goals, plans, and world state persist across sessions and evolve autonomously.

2. **Memory creates intelligence.** Progressive abstraction — raw observation → compressed episode → generalized concept — produces accumulated wisdom no single model call can replicate. Decay ensures relevance. Uncertainty prevents hallucination.

3. **Models are stacked, not swapped.** A classifier, a planner, a verifier, and a communicator — working in sequence through deterministic gates — reason better together than any single model alone.

4. **Determinism bounds probabilism.** Every probabilistic output flows through deterministic validation. Gates, budgets, reliability scores, and verification prevent runaway reasoning. The system knows what it doesn't know.

5. **Plan once, execute cheaply, verify independently.** Never loop an LLM through plan-execute-observe-replan cycles. Reasoning produces a plan. Execution follows mechanically. A separate model verifies results. If the plan needs revision, that is a new reasoning event. Agent loops degrade cognition and inflate tokens quadratically; structured pipelines preserve both.

6. **Every token earns its place.** Context is finite and expensive. Duplicate information across injection points is a bug. Prose wrappers around structured data waste tokens — use the most compact representation that preserves meaning. When in doubt, cut.

---

## Value Proposition

Chalie unlocks problem classes that no single model can solve alone, regardless of that model's capability:

| Capability | What it means |
|---|---|
| Persistent reasoning | Accumulated wisdom across months of compressed experience |
| Autonomous goal execution | Detect intent, form plans, execute across sessions, self-correct |
| Proactive goal inference | Notice goals forming across casual mentions over weeks |
| World model maintenance | Track evolving external state — prices, deadlines, availability |
| Delegation and monitoring | Assign tasks to external agents and verify completion |

---

## Delegation Boundary

**Chalie handles autonomously:**
- Repetitive communication
- Research and synthesis
- Scheduling and coordination
- Administrative workflows
- Summarization and prioritization
- Execution of clear intent

**Chalie escalates when:**
- Values or identity are involved
- Emotional nuance is critical
- Ambiguity or trade-offs require human judgment
- The user's voice or presence matters

---

## Interface Philosophy

**Today:** Chalie presents as a unified chat interface. This is the current input surface.

**Long-term:** Chalie is ambient — always observing, always reasoning. The chat interface becomes one of many input surfaces: voice, notifications, calendar events, sensor data, API callbacks. The interface fades; the intelligence remains.

**Design implication:** The backend cognitive runtime is the product. The frontend is disposable and interchangeable. Every architectural decision must work for both phases.

---

## Behavioral Guidelines

**Chalie should:**
- Be concise by default
- Surface only what matters
- Choose the right timing for interventions
- Act quietly when confidence is high
- Explain actions when transparency builds trust

**Chalie should not:**
- Notify everything
- Act just because it can
- Interrupt without value
- Overwhelm with verbosity
- Require unnecessary user input

---

## Decision Filter

When proposing or designing a feature, evaluate against these questions:

**Architectural filter** — does this change:
1. Move processing from message-level to goal-level?
2. Make the reasoning loop more continuous?
3. Enable vertical stacking (specialized models per function)?
4. Reduce tokens while preserving reasoning quality?
5. Add a deterministic gate to a probabilistic output?

**Product filter** — does this change:
1. Reduce cognitive load?
2. Protect attention?
3. Improve prioritization?
4. Execute clear intent?
5. Strengthen continuity over time?
6. Build trust?
7. Avoid unnecessary interruptions?

If the answer to most is **no**, reconsider or simplify.

---

## Product Evolution

| Stage | Capability | Status |
|---|---|---|
| 1 | Memory & continuity | Complete |
| 2 | Intent execution | Complete |
| 3 | Judgment & attention protection | Complete |
| 4 | Proactive reasoning | Complete |
| 5 | Continuous reasoning loop | Complete |
| 6 | True autonomy | Complete |
| 7 | Situational intelligence | Complete |
| 8 | Expanded perception | **Next** |
| 9 | Cognitive OS | Future |

**Stage 8 — Expanded Perception:** Gives Chalie eyes beyond the chat window. New input surfaces: calendar, email/notifications, file watchers, API webhooks, sensor data. Requires an input adapter framework, signal normalization into the reasoning loop, and privacy controls. The situation model from Stage 7 governs *when* to surface information.

**Stage 9 — Cognitive OS:** Chalie stops being an application and becomes a shared cognitive layer. External agents plug into Chalie for memory, judgment, and user models instead of building their own. Four agent operations: Query, Observe, Judge, Report.

---

## Ultimate Goal

Chalie is not a chatbot, assistant, or productivity tool. It is a **persistent cognitive runtime** — vertically stacking specialized models through persistent memory, deterministic gates, and a continuous reasoning loop to unlock problem classes no single model can solve alone.

The measure of success is not response quality on any single interaction. It is the **scope of problems Chalie can solve autonomously** — detecting a forming goal across casual mentions, researching options, executing a multi-session plan, delegating and monitoring external work, presenting a decision-ready shortlist — all while knowing what it doesn't know and escalating only when human judgment is required.

---

*Related: [docs/04-ARCHITECTURE.md](04-ARCHITECTURE.md) — system design and service map*
