# Case Law

Rulings from real reviews on this project. Judgment transfers through decided cases, not abstract rules — each entry records what was submitted, why it came back, and what right looks like, so the next contributor doesn't relearn it the expensive way.

This log grows by the loop in [MANIFESTO.md](MANIFESTO.md) Part V: a review pushback that matches no existing principle becomes an entry here; an entry cited three times gets drafted as a principle.

Entry format: **Teaches** (principles) · **What happened** · **The ruling**.

---

## Case I — Five patches, or one deletion (2026-05)

**Teaches P4 · P5.**

**What happened.** Voice synthesis produced 26 seconds of dead air mid-reply. Four fixes landed in one day: a lock to serialize a race, a per-chunk retry, a silence trimmer, and a saturation detector with sentence-level rescue. The bug survived all four. The question that ended it: *the library's own docs say this is three to five lines — why do we have hundreds?* The code had bypassed the library's public API, called its private internals directly, and re-implemented — worse — every step the public entry point already performed.

**The ruling.** The scaffolding was deleted and the public API called directly: net −659 lines including tests, and the bug never reproduced. When a dependency misbehaves, re-read its public API before patching around its internals. Two patches on one symptom means the design is wrong — deleting a subsystem is almost always cheaper than fixing it on top of itself.

## Case II — The metric that ate a feature (2026-05)

**Teaches P12.**

**What happened.** An automated improvement cycle deleted a deliberately designed enrichment — built to inject a full capability index into tool discovery — because one test score regressed. Measured only by its own metric, the deletion looked like a win.

**The ruling.** Restored, design intact. A regression that traces to a deliberate feature is a conflict to surface to the feature's owner — never a license to remove the feature. Fix the regression while preserving the design. Optimizers, automated or human, may only remove what an optimizer added.

## Case III — The state that should never have been born (2026-06)

**Teaches P10 · P6.**

**What happened.** A processing chain carried in-memory state from instance to instance — touched-record ids, counters, snapshots — held up by a config hook on every class in the chain, a copy loop, and a regression test to police it all. The state drifted anyway and shipped a subtle decay bug.

**The ruling.** Every carried value was already derivable from two queries against the durable store. The hook, the loop, the policing test, and the bug were deleted together. When a continuation "loses" state, the question is *which query rebuilds it* — never *how do we carry it*.
