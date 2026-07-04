---
name: brainstorming
description: Use when starting any creative or structural work — a new feature, capability, redesign, or significant change — before any spec or code exists, while ideas are still cheap to kill.
---

# Brainstorming

## Overview

Brainstorming exists to kill bad ideas while they cost nothing. The best feature is often the one not built; the best change is often upstream of where the request points. Building for the sake of building is the most expensive habit in software — **when in doubt, ask; asking costs a sentence, unwanted work costs a review cycle plus a revert.**

## The Sequence

1. **Start from the problem, not the solution.** Who hurts, when, doing what? If the request arrives as a solution, walk it back to the problem it serves — the stated solution is one candidate, not the spec.
2. **Apply the product filter first.** Every idea must pass the vision filter — the three questions in [MANIFESTO.md](../MANIFESTO.md) Part I. If the idea fails the filter, stop here. One sharp filter beats a feature checklist.
3. **Respect knowledge asymmetry — in both directions.** The product owner lives in this product; they hold context no document captures. If their spec contradicts your technical findings, *surface the conflict explicitly and ask* — never silently bend the spec toward your code, and never silently comply with what you believe is wrong. Conversely: your technical findings are evidence they don't have — state them plainly.
4. **Check the graveyard.** Search decisions, memory, and history for approaches already tried and removed *and why*. Features get deliberately removed for quality reasons; re-proposing one without addressing why it died is a fast way to burn trust. Chesterton's fence applies to absences too.
5. **Explore deletion and upstream moves before addition.** Can the problem vanish by removing something? Can one change at a higher abstraction eliminate the whole class instead of patching this instance? One change at the top beats N patches at the leaves — this is the only sanctioned reason to expand scope, and it still gets proposed, not assumed.
6. **Collapse convergent shapes into data, not branches.** When two features have evolved into the same shape, unify them — but a good collapse pushes the variance into *data* (a list, a config, a table); a bad collapse is two functions stapled together with if-branches. Test: fewer concepts, no case-switching.
7. **Research how others solved it.** Harvest their failure modes and scaling ceilings before committing. You are not the first to face this.
8. **Produce real alternatives.** At least two genuinely different approaches, each priced in code, complexity, and maintenance. Recommend one. A single option isn't a decision, it's a fait accompli.

## Output Contract

One page, no more:

- **Problem** — one sentence, in user terms.
- **Chosen approach** — and the why, as an evidence chain, not adjectives.
- **Rejected alternatives** — each with the reason it lost.
- **Non-goals** — what is explicitly NOT being built. Scope is a feature.
- **Open questions** — batched for the owner; only true forks block.

Hard gate: on structural or architectural work, this page gets explicit owner approval **before any implementation code exists.**

## Anti-Patterns

| Anti-pattern | Why it's fatal |
|---|---|
| Solutioneering | Committing to the first idea and reverse-justifying it. The point is to generate rivals and let them fight. |
| "While we're at it" | Scope smuggled in without consent burns the owner's scarce review budget. Propose; don't expand. |
| Future-proofing without a present need | No field, column, or abstraction "in case." Test: needed today, for a real use? |
| Designing for imagined scale | Solve today's load; name the ceiling and the upgrade path instead of building it. |
| Novelty-driven choices | The boring, proven option wins by default. New tech must beat it on evidence. |
| Judging by your own setup | Judge by the product vision and the whole audience, not by what you personally run. |
| Deciding the owner's forks | "I assumed you'd want…" on a real fork is taking decisions that aren't yours. Ask. |
