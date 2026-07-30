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

## Case IV — The honest answer that killed every turn (2026-07)

**Teaches P4 · P10.**

**What happened.** Sizing a model's context window returned `None` when the client could not name a figure, on the reasoning that admitting ignorance beats fabricating a number. Four of the five provider clients could reach that branch — a model slug newer than a local cache, a hosted endpoint that serves completions but publishes no window, a transient SDK error. The caller in front of every request raises when the window is unknown, so each of those branches was not a missing measurement: it was a provider that could not answer a single message. A test asserted the `None` and locked the behaviour in as intended.

**The ruling.** `None` was conflating two different states — *I could not reach it* and *it answered but named no size* — and only the first deserves it. A provider that answers is now always sized: its own reported figure where it has one, otherwise a default read from the model family, chosen by one owner module rather than copied into five clients. A provider that cannot be reached still returns `None`, so a host that is briefly down never stamps a guessed window onto its record. Before adding a return value that means "I don't know", find out what the caller does with it — a truthful sentinel that reliably breaks the feature is a bug wearing honesty as a costume. When one value carries two failure modes, split the value, not the caller.

**The same door, reopened.** The first fix still caught every exception from the sizing request alike, so a host answering with an HTTP status — a routine `400` from a reasoning model rejecting a parameter, a `401`, a `404` — was treated as a host answering nothing, and the bug survived behind a different door. A refusal is an answer: it proves the host is up. Sizing it lets the real fault surface where the message names it, instead of masking a wrong key or an absent model as "cannot determine the context window". When a fix rests on a distinction, audit every place the code makes that distinction — one corrected branch does not make the others correct.

## Case V — The empty prompt that crashed every turn (2026-07)

**Teaches P9 · P11 · P4.**

**What happened.** A delegate shipped complete — config, system prompt, pinned toolset, ability, tests — except for one arm in the prompt dispatcher. The dispatcher's final branch logged a warning and returned an empty string, so the delegate's user message was empty; because that channel suppresses history, the empty string was not a missing *section*, it was the entire request. The task text had been sitting in the turn's raw input the whole time, never read. One provider tolerated the empty content and answered plausible nonsense. The provider actually selected validates its request parameters, so it rejected the call with a `400`, three byte-identical retries, and a crashed turn — on every single invocation. The delegate's own tests stayed green throughout: they asserted the answer mentioned any of several expected words, and a hallucinated answer to no question still satisfied that.

**The ruling.** A missing dispatch arm is a wiring error, and `return ""` disguises it as output the caller cannot tell apart from the real thing — the same costume as Case IV's `None`, and it fails the same way: either silent garbage or a stack trace that names the vendor instead of the omission. Three things came out of it. The channel got its own builder rather than being remapped onto the user channel's, because a delegated task is a hand-off, not a user utterance, and borrowing that assembly would have injected the user-identity synthesis and a continuation banner naming a tool the delegate does not hold. The builder carries World State deliberately: it holds the only date anchor any channel receives, and an agent that cannot date its own work writes wrong dates into the files it creates. And the guard is not a test for this one channel — it enumerates every config, drives the real dispatcher, and fails if *any* channel reaches the fallthrough, so it catches the next omission instead of this one twice. Two habits to carry: tolerance for a malformed request is per-provider, so never infer the failure mode from an observation on a different model; and an assertion that the answer contains any of several plausible words cannot detect an agent that was never told the question.

## Case VI — The tests that could not fail (2026-07)

**Teaches P11 · P7 · P9.**

**What happened.** A release branch came in several thousand lines heavier than the features it added and removed could account for, and two thirds of the growth was test code. Four kinds of test made up the excess, and none of them could go red for any change a contributor would plausibly make. One walked a static table of default values and asserted each entry equalled the value written in the table. One asserted a name was present in a literal set defined three lines away in the module it imported. One asserted the *absence* of fields a refactor had already dropped — on a result whose keys the assertion above it already pinned to an exact set, so the loop had nothing left to catch. One stood up a third-party speech model and asserted its inference quality: a probability floor on silence, a frames-above-threshold ratio on synthesised speech, determinism across two fresh instances, and its own argument validation. And one existed, by the confession in its own module docstring, to buy back a guarantee the code had stopped making when a table replaced a convention — with its second half admitting in a docstring that the framework already raises at boot for the case it checked.

**The ruling.** All of it deleted, net −238 lines, suite green. A test earns its place by failing when a real regression lands; if you cannot name the change that turns it red, it is documentation with a run cost, and worse than documentation because green output implies coverage that does not exist. Restating a constant, asserting a vendor's library works, and asserting a removed feature stays removed all share one defect: the assertion's subject is not this project's behaviour. The last kind is the one to watch, because it looks the most responsible — a test written to cover an enforcement gap. That is a design decision smuggled in as coverage. Contracts belong in code that refuses to run when they break, not in a test that reports the breakage after the fact; where a gap is real, close it at the boot path or accept it explicitly and in the open. What survived the sweep is the shape to imitate: the tests that stand up a real host on a real socket and prove a provider that answers gets sized while one that cannot be reached does not — those can fail, and once did.

**Applied to the whole suite.** The ruling is a law, not a verdict on one branch, so it went across all of them: a syntactic pass flagged every test whose entire assertion set was one of the three shapes, and each candidate was then read, because the decisive distinction is invisible to a matcher. Asserting an absence from a *rendered output* is behaviour — a sanitizer that strips a script tag, a channel that refuses to inject a non-discoverable tool, a request payload that omits a field the model does not support. Asserting an absence from a *module or a literal constant* is structure, and structure is the code's job. Reading also reversed the machine twice in the other direction. Four file-operation tests looked like assertions that the standard library works, and their sources turned out to hold a path guard, an atomic existence check, an octal parser and a directory-versus-file branch — so the on-disk assertion is what proves the operation happened rather than being echoed back, and they stayed. A factory test that asserted only the returned type stayed too: an unknown platform already raises, but a *reordered* branch returning the wrong client for a valid one fails silently, and nothing else catches it. Screen mechanically, decide by reading; a pattern names a suspect, never a verdict.
