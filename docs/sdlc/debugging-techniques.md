---
name: debugging-techniques
description: Use when behavior contradicts expectation — a bug, crash, regression, flaky test, wrong output, silent failure, or performance mystery — before proposing or writing any fix.
---

# Debugging Techniques

## Overview

Debugging is the discipline of replacing narrative with evidence. You are done when you can explain the failure mechanism end-to-end and prove it — not when the symptom stops.

## The Method

1. **Reproduce first.** No reproduction, no fix. A fix you can't watch fail is a guess. If it only fails "sometimes," the reproduction rate is your first measurement.
2. **Read the actual artifact.** The real error text, the real log lines, the real rows, the real HTTP exchange. Never reason from what the error "probably" says. Never trust a version file, a status page, or a comment over the running system.
3. **Measure, don't eyeball.** "It looks fine" has been wrong here before; the user's stopwatch was right. Count the requests. Time the boot. Probe the port. A number ends arguments that impressions start.
4. **Trace the behavioral path, not the import path.** What actually executes at runtime, in what order, with what data? "Nothing imports this" is not evidence nothing reaches it.
5. **One written hypothesis at a time.** State it, state the observation that would falsify it, run the check. If the fix works, you must be able to say *why* — a fix that works for unknown reasons is an unexploded bug.
6. **Bisect.** Halve the search space: git bisect across commits, comment-out bisect across code, subset bisect across data, environment bisect (fresh install vs warm dev). Fresh environments expose what warm ones hide — a system can be flawless on your machine and dead on first boot.
7. **Exhaust the symptom, not the first cause.** Two independent root causes can produce one symptom — it has happened here (a slow port bind AND a client redirect loop behind one blank screen). Finding a real bug doesn't close the investigation until the symptom is *fully* accounted for.
8. **Suspect your own code before the platform.** When a mature library misbehaves, re-read its public API before patching around it. Five band-aids on one subsystem here turned out to be wounds from calling private internals — the public call replaced 659 lines and the bug vanished ([Case I](../CASE-LAW.md)).

## The Band-Aid Counter

Keep a running count of fixes applied to one symptom.

- **1 fix** that you can explain mechanistically: fine.
- **2 fixes on the same symptom**: the design is wrong. Stop patching and rip the subsystem back to its root — that is almost always cheaper than a third fix stacked on the second.

## Exit Criteria

- Mechanism explained end-to-end, each link backed by an observation. This bar does not scale with diff size — a one-line fix needs its mechanism explained same as a rewrite. Proportionality governs how much you build and test, never how much you understand.
- Original reproduction now passes; both sides of every branch you touched exercised.
- All instrumentation removed — zero residue.
- The fix is at the root. If your fix is downstream of the cause, you've written a shim.

## Anti-Patterns

| Anti-pattern | Why it's fatal |
|---|---|
| Shotgun debugging (change several things, rerun) | When it passes you've learned nothing; the bug retreats, it doesn't die. |
| Rerun-and-hope | Flakiness is a bug with a probability attached. Reruns hide it until production rolls the dice. |
| Silencing the signal (skip the test, ack the scanner, catch-and-pass) | You didn't fix the bug, you blinded the instrument that found it. Requires explicit permission, always. |
| Editing prod code to appease test infra | Fix the infra, not the product — the harness rule lives in `qa-principles.md`. |
| Blaming infrastructure first | Prove your application innocent with evidence before pointing outward. |
| Timeout/truncate as a "fix" | That's silent data loss with a trigger condition. Legitimate work can run for hours. |
| Declaring victory on symptom disappearance | If you can't say why it stopped, it hasn't. |

## Red Flags — Stop and Retrace

- "Let me just add a retry/sleep/flag and see."
- "It works now, moving on" (without knowing why).
- "The error is probably harmless."
- You are debugging the test instead of the behavior it caught.
- You are about to reset credentials, state, or an environment to get past a failure — that is destruction, not debugging. Stop and report.
