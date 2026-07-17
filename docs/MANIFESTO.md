# The Chalie Manifesto

The one mandatory read before contributing — human or agent. Everything else is loaded when the work needs it ([index](index.md)).

This document exists so work is judged the same way whether the maintainer is in the room or not. Principles are numbered so reviews cite them — "violates P8" — instead of re-arguing them.

## I · The vision filter

Chalie is a **Life OS**: one conversational surface that understands your life well enough that you stop opening the apps behind it. *Life, handled.* Born from one frustration — systems that forget what you tell them — so the product is **accumulated understanding**: what Chalie knows on day 300 that no fresh install can. That is the moat, and why the bar is absolute: trust in a system that runs your life is earned over months and lost in one silent failure.

Every decision passes three questions:

1. **Does it move a user closer to never opening another app?** If not, it waits.
2. **Does it serve the vision and audience — not any one person's setup?** The maintainer is the builder, not the sole target user.
3. **Does it amplify the model, or contain it?** Natural exits over artificial budgets, learning signals over gatekeepers, immediate execution over delays.

Never: a chatbot with plugins · a weight-vest harness built from fear · a demo that lies in production.

## II · Principles

Nearly every correction on this project traces to one pattern: a shortcut that hid the real state of the system. Speculation hides the unknown; truncation hides lost data; shims hide unfinished migrations; swallowed exceptions hide failures. **Never hide the state of the system** — every principle below is a special case of it.

### Truth

- **P1 — Reality over narrative.** Every claim names its evidence — the file, log, or output that proves it; otherwise the answer is "I don't know yet," and you go find out. *Ask: can I point at the thing that proves this?*
- **P2 — Thoroughness is cheap; regressions are expensive.** Investigation costs minutes; a silent regression costs months, later. Trace the behavioral path, not the import graph. *Ask: will this surprise anyone in three months?*
- **P3 — Look outside before building.** How has everyone else solved this — what was tried and abandoned, and why? Diverge only with a named reason. *Ask: what does the rest of the world do here, and why do we differ?*

### Design

- **P4 — Fix the root, never the symptom.** Misbehavior lives in the wiring, not the last visible layer (in an LLM system: not the prompt). Each patch on a wrong foundation is compound interest (Case I). *Ask: am I removing the problem, or wrapping it?*
- **P5 — The smallest change that works.** Net-negative LOC is a feature; a refactor that grows the code has failed by that fact alone. One exception: pull the higher-level lever that removes a whole class of problems. *Ask: could this be smaller — or is there a lever one level up?*
- **P6 — One path; steps self-no-op.** No "if needed" guards — make the step harmless and run it unconditionally: `s.replace(":)", ":))")` needs no `if`. Branches multiply paths, tests, and lines. *Ask: am I guarding a step that is already harmless?*
- **P7 — Nothing "just in case."** No column, flag, hook, or abstraction without a concrete need today. *Ask: what breaks this week if I leave it out?*
- **P8 — Zero residue.** "Remove" means the codebase looks like it never existed — file, imports, callers, tests, same commit. Git history is the reference. *Ask: would a new reader know the old thing was here?*

### Craft

- **P9 — Errors are loud, always.** Catch only to add context and re-raise; an error that needs archaeology equals no error. No timeouts, truncations, or silent data loss without the maintainer's explicit sign-off. *Ask: when this fails at 3 a.m., where does it show?*
- **P10 — One source of truth.** State has one owner, usually the durable store; a second in-memory copy creates drift, then apparatus, then a shipped bug (Case III). *Ask: which query rebuilds this instead?*
- **P11 — Tests prove what a user would feel.** Real entry point, real stack, zero mocks. A test that breaks without behavior changing asserts implementation — delete it. *Ask: if this failed, what would a user notice?*

### Conduct

- **P12 — Decisions are constraints.** A committed decision binds until explicitly revisited: work within it or surface the conflict — never silently re-decide, and never delete a deliberate feature to fix a metric (Case II). *Ask: did someone choose this on purpose — and have I checked?*

## III · The bar

**Would you ship this to someone who runs their life on Chalie?** "Mostly works" is not a shippable state; the smallest defect you knowingly ship is a statement of your standard. Every submission is walked against P1–P12 — every time, regardless of how good the idea is — and failing any one sends it back. Same contract for human, agent, and maintainer.

Judgment calls: the vision is owned. A spec contradiction, or any choice that isn't unambiguously best-in-class design — ask the owner. Implementation-level choices are yours; design-level ones are not.

## IV · Case law

Full rulings live in [CASE-LAW.md](CASE-LAW.md). The three cited above, in one line each:

- **Case I** — four band-aids in one day couldn't fix dead-air TTS; deleting 350 lines of scaffolding and calling the library's public API did (net −659 LOC, bug gone).
- **Case II** — an automated cycle deleted a deliberately built feature to fix one score; restored. Optimizers may only remove what an optimizer added.
- **Case III** — state carried across processor instances drifted and shipped a decay bug; every value was derivable from two queries, so the whole carry apparatus was deleted with the bug.

## V · How this grows

1. Pushback that matches no principle → a written ruling in [CASE-LAW.md](CASE-LAW.md).
2. A ruling cited three times → drafted as a principle.
3. A principle uncited for months → deleted.
4. Reviews cite by number — "violates P8" is one sentence, impersonal, and teaches.

The loop, not the page, encodes judgment. The page just makes the loop cheap.
