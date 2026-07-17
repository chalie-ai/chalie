---
name: software-development
description: Use when designing, writing, or modifying code — any feature, fix, refactor, or removal — from the first line of design to the last line of test.
---

# Software Development — Design, Build, Test

## Overview

Code is a liability that occasionally pays rent. The craft is threefold: understand before touching, make correctness structural, and leave zero residue. This handbook works with the project's conventions ([CONTRIBUTING.md](../CONTRIBUTING.md)) and is judged by the manifesto's principles ([MANIFESTO.md](../MANIFESTO.md)); it governs how you carry them through a change.

## Understand Before You Code

No edit before you can answer these five questions with evidence:

1. **What is the current runtime behavior?** Trace the behavioral path — what executes, in what order, with what data. Import graphs and greps are not behavior.
2. **What exercises this path?** Which tests, which flows, which users. If nothing does, that's a finding.
3. **What breaks for old data and upgrade paths?** Code meets databases written by every previous version of itself.
4. **What surprises a maintainer three months out?** Adjacent services, implicit contracts, timing assumptions.
5. **What are the silent side effects?** The write you didn't know fired, the cache you didn't know existed.

Also read the graveyard: has this approach been tried and deliberately removed? Deleted code died for a reason; know it before resurrecting it.

## Design

- **Climb the ladder; stop at the first rung that holds.** Does it need to exist? → stdlib? → platform/framework feature? → already-installed dependency? → one line? → only then, minimum code.
- **Public API only.** Never build on a library's private internals — every future fix becomes a band-aid on a wound you inflicted. If the public API can't do it, that's a design signal, not a challenge.
- **Minimum effective dose, with the leverage exception.** Smallest change wins — unless a change one level up eliminates the entire problem class. Then take the top; one change at the top beats N patches at the leaves. Either way: propose scope changes, never assume them.
- **Variance into data, not control flow.** One common path whose steps self-no-op when not needed. The second `if X: do A` next to `if Y: do A'` is a chokepoint announcing itself — build it.
- **Contracts before tests.** A typed signature, an abstract method, a single return type, a DB constraint — each makes a bug class unrepresentable, which beats any test asserting the bug didn't happen.
- **No state outside the database.** In-process state carried across calls or instances is a second source of truth that drifts. Rebuild from the store; any process may die at any line.

## Build

- **Zero residue.** Rename = every caller updated in the same commit, no alias. Remove = file, imports, callers, tests, config, docs — all gone, same commit. Import changes land in the same edit as the code that made them stale. No commented-out "for reference" code.
- **No adapters, shims, polyfills, or forwarding wrappers — ever.** A function that only calls another function is bloat wearing a compatibility costume. Fix the callers.
- **Loud errors.** Catch to add context and re-raise, or log with enough context to diagnose — one of the two, always. Never return a sentinel where an exception belongs. Never let an HTTP error body leak internals — log server-side, return generic.
- **The failure path is a feature.** Whatever consumes your output — user, service, or model — must be told when a step failed. A silently skipped step poisons every decision made downstream of it; nothing can reason about what it can't see.
- **Names are inherited, not invented.** Use the exact existing names — the owner's, the codebase's, the tool's. Inventing a new umbrella name for an existing thing creates a translation layer everyone pays forever.

## Test

- **Feature tests on the real hot path, zero mocks.** Regressions live *between* steps; mocks amputate exactly the seams where bugs breed. Mock only what you genuinely cannot run.
- **Proportionality.** Major rework earns feature tests; a one-line relocation doesn't. Test-line inflation that offsets production-line reduction is bloat with a green badge.
- Full testing doctrine — threshold branches, schema sources, which tests to delete — lives in `qa-principles.md`.

## Definition of Done

Committed tree (not working tree) verified: runs end-to-end, feature-tested, both branch sides covered, zero residue, migrations + seeders shipped, docs updated, reviewed. Anything less is "in progress."

## Red Flags — Stop and Retrace

- You're writing a wrapper to avoid updating callers.
- You're adding a parameter/flag/field "in case."
- Your diff is net-positive LOC and you can't defend every added line.
- You're editing the spec or the test to match the code.
- The current edit doesn't serve the original ask in one sentence — that's drift; back out to the last decision that did.
