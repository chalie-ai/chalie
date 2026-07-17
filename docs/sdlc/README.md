# The SDLC Handbook

## Overview

This is the master map. Nine handbooks in this directory each govern one phase or discipline; this file defines the laws that bind all of them and the handoffs that connect them.

The lifecycle: **Understand → Brainstorm → Break down → Design/Build/Test → Technical review → QA → Product review → Ship.** Debugging enters whenever reality disagrees with expectation. Two codes cut across every phase: stability principles and UI/UX principles.

One meta-defect underlies almost every failure this project has recorded: **a shortcut that hid the real state of the system.** Speculation hides the unknown. A silent catch hides the error. A hollow wrapper hides an unfinished migration. Scope creep hides that the real ask was smaller. A band-aid hides a wrong design. Everything below exists to keep system state visible.

## Using This Directory

Route from the [handbook index](../index.md): read the file named in the lifecycle table when entering that phase, not all at once. This directory is the authoritative process spec for work on this repository — it evolves by PR like everything else here.

Precedence, highest first: the maintainer's explicit instructions → the project's conventions ([CONTRIBUTING.md](../CONTRIBUTING.md)) → the manifesto and these handbooks → anything else. Product-specific details inside are worked examples, not scope limits.

## The Laws

The laws that bind every phase are the manifesto's principles — **[MANIFESTO.md](../MANIFESTO.md) P1–P12**. Reviews cite those numbers. Two conduct laws sit alongside them for anyone — human or agent — operating in a shared tree:

- **Agency is sacred.** Irreversible, destructive, or out-of-scope actions require explicit consent — truncating data, deleting files you didn't create, closing tickets, silencing scanners, touching auth. When blocked by friction (a failed login, a stubborn gate), STOP and report. Never destroy the obstacle to complete the task.
- **Never touch work that isn't yours.** Unexpected changes in the tree belong to someone else. Commit only your own files. Conflict on a shared file → stop and ask.

## The Owner

Every handbook says "ask the owner" at forks. The owner is whoever holds product authority over the surface being changed — the person who designs it, decides for it, and answers for it. Asking means: state the fork, your evidence, and your recommendation — then wait if the fork blocks, batch if it doesn't. Owner unreachable? Irreversible forks wait; reversible ones take the conservative path, loudly flagged.

## The Lifecycle

| Phase | Handbook | Exit artifact — what the next phase receives |
|---|---|---|
| Explore the problem | `brainstorming.md` | Decision page: problem, chosen approach + why, rejected alternatives + why, non-goals, batched questions |
| Slice the work | `task-breakdown.md` | Ordered vertical slices, each with a one-sentence done-condition |
| Design, build, test | `software-development.md` | Working increment: zero residue, contracts enforced, feature-tested on the real path |
| Reality disagrees | `debugging-techniques.md` | Root cause with an evidence chain, then back to build |
| Inspect the change | `technical-reviews.md` | Verified findings logged as tickets; explicit verdict |
| Verify the system | `qa-principles.md` | Committed tree proven in a fresh environment |
| Live the product | `product-reviews.md` | First-30-minutes verdict through customer eyes |
| Cross-cutting | `scalability-stability.md`, `ui-ux-principles.md` | Applied in every phase above |

Phases repeat per part on multi-part features. Never batch parts through one pipeline pass.

## The Drift Protocol

The most seductive anti-pattern: you start with a goal, hit friction, and gradually the goal becomes "get this thing to pass." You are now solving a different problem — usually by hiding state.

**Drift signals — any one means STOP:**
- You are editing production code to satisfy a test or tool.
- You are adding a flag, sleep, retry, or wrapper to get past a failure you don't understand.
- You are editing the spec to match what you built.
- You are on your second workaround for the same symptom.
- You can't state, in one sentence, how the current edit serves the original ask.
- You are about to bypass something protecting the system (auth, a gate, a scanner, a failing check).

**Recovery:** Write down the original goal. List the decisions since. Find the first one made to serve "getting it done" instead of the goal. Back out to that point — deleting work you did is cheap; shipping drift is not.

**Drift vs. discovery:** friction sometimes exposes a real defect. The test: would the change still be justified if your ticket didn't exist? Yes — it's a finding: ticket it and ask whether it preempts your task. No — it exists only to get your artifact to pass: back it out.

## Universal Rationalizations

| Excuse | Reality |
|---|---|
| "It's a small change, skip the process" | Small changes ship time bombs too. The process is sized to risk, not skipped. |
| "I'll clean it up later" | Later never comes. Residue lands in the same commit or the work isn't done. |
| "This wrapper keeps it backwards compatible" | A hollow passthrough is bloat hiding an unfinished migration. Fix the callers. |
| "I'll just quickly work around it" | Workarounds are how one bug becomes a system of bugs. Stop, trace, fix upstream. |
| "Asking will slow things down" | Wrong work is the slowest thing there is. Ask at forks; batch the rest. |
