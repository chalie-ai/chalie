---
name: scalability-stability
description: Use when designing or reviewing anything that holds state, crosses a process boundary, handles failure, schedules work, or must survive restarts, retries, upgrades, and growth.
---

# Scalability & Stability Principles

## Overview

Stable systems share one property: **their state is always derivable and their failures are always visible.** Scale is not the hard part — most systems die of drift and silence long before they die of load.

## State

- **The database is the only truth.** Any process may die between any two lines. In-memory state carried across calls, instances, or restarts is a second source of truth that *will* drift ([Case III](../CASE-LAW.md)) — rebuild from the store on every cycle. If rebuilding feels expensive, that's a schema/query problem, not a license to cache truth in RAM.
- **One source of truth per concept.** Every mirror, cache, and copy is a reconciliation bug on a timer. Caches must be disposable: deletable at any moment with zero correctness impact.
- **Constraints live in the database; contracts live in types.** A NOT NULL, a UNIQUE, a foreign key, a typed signature, a single return type — each makes a bug class unrepresentable. Application-level discipline is what you use when you've failed to encode the rule structurally.
- **All time is timezone-aware UTC** through one shared utility. Naive datetimes silently corrupt data.

## Failure

- **Loud or it didn't happen.** Every failure either bubbles up or is logged with diagnostic context — never both suppressed. An error requiring archaeology equals no error.
- **Crash beats corrupt.** A process that dies is restartable; a process that continues on bad state spreads it to everything it touches.
- **Gates fail closed.** Anything protective — permission checks, approval gates, policy blocks — must deny when the gate itself faults. A gate that fails open is a hole with paperwork.
- **No silent data loss, ever.** Timeouts, truncations, drop-oldest, "best effort" — all forbidden without explicit owner approval. Legitimate work can run for hours; killing it on a timer is destroying user data on a schedule.
- **Retries require idempotency.** Every mutation must be safe to repeat before anything is allowed to repeat it. Idempotent, self-no-op steps are what make the one-common-path design safe under failure.
- **No fallback parsing of malformed input.** Accepting garbage via a lenient fallback masks the producer's bug today and creates a new silent failure mode tomorrow. Reject loudly; fix the producer.
- **Internals never leak outward.** Errors log rich server-side, return generic client-side. Stack traces in responses are a security finding, not a debugging convenience.

## Lifecycle

- **Boot honestly.** Deploy infrastructure publishes your port the instant the container starts; if you bind late, every early request is a connection reset. Bind first, serve an honest "starting" state, flip a readiness signal when true.
- **Schema migrations ship complete:** schema change + data backfill + seeder update, together. A new column without a backfill shows users a hole where their data should be. One-shot migrations get removed after they run — write-only-forever code is residue.
- **Data has a lifecycle.** Tables and directories that only ever grow are leaks. Deletion must actually delete — including shadow copies, indexes, and reclaimable space. "Deleted" data that remains readable is a broken promise.
- **Upgrades are a feature.** New code meets old data on every user's machine. The upgrade path is tested like a feature, because it is one.
- **Before any destructive environment operation** — container recreate, volume prune, re-init, "cleanup" — verify the irreplaceable files (keys, secrets, databases) live on storage that survives it. This lesson was paid for twice; there is no budget for a third.

## Simplicity Under Load

- **Boring scales.** A table, a queue, a cron — proven parts, in that order of preference. Complex systems that work evolve from simple systems that worked.
- **Name the ceiling instead of building past it.** Solve today's load; document the known ceiling and the upgrade path (`# lean: global lock; per-account locks if throughput matters`). Speculative scale-engineering is bloat that also happens to be untested.
- **No artificial throttles, delays, or budget caps** as complexity management. They mask the real constraint and punish legitimate use. Natural exit conditions only.
- **Continuous work runs continuously.** If a background process needs a cooldown to be tolerable, the process is wrong, not the schedule.

## Anti-Patterns

| Anti-pattern | Why it's fatal |
|---|---|
| Cross-instance in-memory state | Works until the second instance, the first crash, or the first deploy — whichever comes first. |
| Liveness by proxy (object exists ≠ process alive) | A valid reference to a dead thing passes every check and delivers to nowhere. |
| Swallow-and-DEBUG-log | Functionally identical to `except: pass` for anyone operating the system. |
| Drop-oldest at the transport layer | Data loss placed where nobody will ever look for it. |
| Cleanup by deletion of unknown files | Keys, secrets, and databases die in "cleanup." If you didn't create it, you don't delete it. |
| Fail-open protective checks | The one day the gate faults is the one day it mattered. |
