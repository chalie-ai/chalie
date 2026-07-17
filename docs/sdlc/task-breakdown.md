---
name: task-breakdown
description: Use when turning an approved approach into executable work — slicing a feature into tasks, ordering a migration, planning parallel workstreams, or writing tickets.
---

# Task Breakdown

## Overview

A breakdown is good when every task leaves the system working and every task's done-condition fits in one sentence. Breakdown quality decides everything downstream: review size, rollback cost, and whether parallel workers collide.

## Slicing Rules

1. **Vertical slices only.** Each task ships observable behavior through the whole stack — schema to UI if that's the path. Horizontal slicing ("all models, then all endpoints, then all UI") produces weeks of nothing-works and one unreviewable merge.
2. **One-sentence done-condition.** If you can't state what "done" looks like — observable, checkable — in one sentence, the task is too big or too vague. Split it.
3. **One ticket per task.** The ticket carries context (why), acceptance criteria (what), and evidence-of-done (proof). A future stranger — human or agent — must be able to execute it without the conversation that spawned it.
4. **Multi-part features repeat the full pipeline per part.** Brainstorm → build → review → QA for part one, *then* part two. Never batch parts through one pass; batching multiplies review surface and hides which part broke what. The pipeline is sized to each part's risk — a trivial part gets a proportionally light pass. Batching is the sin, not lightness.
5. **Riskiest first.** Unknowns, integrations, and third-party contact surfaces go first — they're where the plan dies, and you want it to die cheap. Polish goes last.
6. **Every task ends green.** Sequence so the system builds, boots, and passes after each task. A plan with a "broken in the middle" phase is a plan to get interrupted in the middle.
7. **Architecture decisions precede scoped tasks.** If work spans a process/service/interface boundary, the boundary contract (protocol, daemon model, data flow) is decided and written *before* tasks are cut. Teams here have watched four of five implementations come back wrong because the boundary was left implicit.

## Code Migration Discipline

Code migrations (moving/rewriting live functionality — distinct from schema migrations, see `scalability-stability.md`) have their own iron sequence, per unit:

**copy → wire → verify → delete** — one function or unit at a time.

- Never leave a hollow shell (a stub returning `''`, a passthrough wrapper). One discovered shell makes the *entire* migration untrustworthy — every other "done" now needs re-verification.
- Never park on peripheral work while the critical path is a stub. Finish the spine first.
- The delete happens in the same task as the wire-up. "Remove old path later" is how two sources of truth are born.

## Parallel Work

- Partition by file ownership. Two workers on one file is a collision scheduled in advance.
- Shared-file changes get sequenced, not parallelized.
- Workers proceed on best judgment and **batch questions for the end** — blocking a parallel run on a mid-flight question kills the parallelism that justified it. Only a true fork (owner-level decision) blocks.
- Each worker commits only its own files. Foreign changes in the tree are someone else's live work — untouchable.

## Anti-Patterns

| Anti-pattern | Why it's fatal |
|---|---|
| Horizontal slicing | Nothing works until everything works; review and rollback become all-or-nothing. |
| Mega-tickets / "misc fixes" | Unreviewable, unrevertable, and hides scope creep by design. |
| Scope smuggling | Work not in any ticket is work nobody approved and nobody will remember. |
| Hollow-shell migration | The empty stub lies about progress; trust in the whole effort collapses with it. |
| Batching parts "for efficiency" | One giant pass through the pipeline is slower than N small ones — review cost is superlinear in diff size. |
| Plan frozen against reality | When implementation contradicts the plan, stop and flag — don't silently rewrite either the code or the plan to match the other. |
