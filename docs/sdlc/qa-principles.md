---
name: qa-principles
description: Use when deciding what to test, how to test it, building test environments or harnesses, interpreting test results, or verifying work before claiming it done.
---

# QA Principles & Techniques

## Overview

QA answers one question with evidence: **does the shipped thing work for a real user?** Automated suites are instruments, not answers — a green suite is a precondition for confidence, never the source of it. No unit test, however elegant, substitutes for exercising the product.

## The Three Tiers — Never Blurred

| Tier | Nature | Question it answers |
|---|---|---|
| **Feature tests** | In-repo, deterministic, real stack, zero mocks | Does this mechanism work end-to-end? |
| **Scenario observation** | Automated product runs mimicking real users; pass/fail | Does the product behave when a human uses it naturally? |
| **Benchmarks** | Scored quality runs | How *well* does it perform, trending over time? |

Each tier answers a question the others cannot. A feature test asserting quality, a scenario asserting string formats, a benchmark used as a gate — all category errors. Keep the walls.

**The harness wraps the product — never the other way around.** Test infrastructure adapts to the product; production code is never modified to accommodate a test (no skip-flags, no test-only endpoints, no backdoors). An endpoint no real consumer uses cannot exist: it's dead weight plus attack surface, built to flatter a suite.

## Feature Testing

- **Real hot path, zero mocks.** Regressions live *between* components; mocks amputate exactly those seams. Mock only what you genuinely cannot run. Test count and coverage percentages are not quality metrics — path reality is.
- **Assert downstream effects,** not return values: the row written, the message emitted, the file produced, the state visible to the next component.
- **Reproduce bugs as tests before fixing.** The failing test is the proof your fix fixes.
- **Both sides of every threshold.** The branch your test data never triggers is where the production 500 is waiting. Test data must be shaped to cross the boundary, not just approach it.
- **Build test state from production sources** — the real schema file, the real seeders. Hand-copied DDL and fixtures rot silently and then test a database that no longer exists.
- **Portability.** No hardcoded local paths, no environment assumptions. A test that only passes on its author's machine is a rumor.
- **Proportionality.** Major work earns feature tests; trivial mechanical changes don't. A 97-line test for a one-line relocation is bloat. Deleted dead code needs a grep, not a suite run.
- **Delete bad tests.** Gatekeeper tests (asserting a constant contains a substring), brittle tests (break on every touch, no behavior change), tests of generated artifacts the user never receives — all negative value. Deleting them is QA work.

## Scenario Observation

- **Mimic a real user, or measure nothing.** Implicit natural prompts, real workflows, no steering the system toward the mechanism under test — a user can't do that, so the test mustn't. The value is observing *natural* behavior.
- **Judge behavior, not strings.** Pass/fail on what happened — latency, wrong action, wasted steps, off-topic drift — never on literal output formats a real user wouldn't notice.
- **Respect asynchrony: poll, don't snapshot.** Asserting on async results with a one-shot read is the single most common source of false failures. Poll with a deadline.
- **Instrument before you run.** If the environment tears down after the run, evidence not captured live is gone forever. Monitoring starts before the pipeline does.
- **Weight critical steps.** An average across steps lets one catastrophic failure hide inside a passing score. Gate-worthy steps fail the whole scenario alone.
- **A failed run gets a diagnosis, not a rerun.** Rerun-until-green launders bugs into noise.

## Test Environments — Hard Rules

- **A live instance is never a test target.** Test against disposable environments you provisioned for the purpose, torn down after. Production-adjacent instances holding real data are off-limits, unconditionally.
- **Every automated test agent gets explicit DO-NOT constraints:** no credential resets, no auth/vault mutation, no destructive shell operations, no deleting files it didn't create. Agents optimize for task completion; an unstated boundary is a boundary that will be crossed.
- **Blocked ≠ licensed.** A failed login or a locked resource is a result to report, not an obstacle to defeat. Destroying an obstacle to complete a test is the single worst failure recorded in this team's history.

## Verifying "Done"

- **Verify the committed tree, not the working tree.** Local gates have passed here on a dirty tree while the actual commit crash-looped in deploy. The strongest evidence: a fresh environment built from the commit, reaching healthy, exercising the change.
- **Claiming done requires having watched it work** — the command output, the response, the row, the pixel. "Should work" is not a QA verdict.
- **Deadlines don't change the epistemics.** Under time pressure your deliverable is a fast, honest risk statement: what is known, what isn't, with evidence for both. Shipping with a known unknown is the owner's call — never make it silently on their behalf.

## Anti-Patterns

| Anti-pattern | Why it's fatal |
|---|---|
| Mock theater | Green tests over amputated seams verify a product that doesn't exist. |
| Coverage worship | 100% coverage of the wrong assertions is 0% confidence, at full maintenance cost. |
| Flaky tolerance | A flaky test is a real bug with a probability attached, being trained into background noise. |
| Test-only backdoors in prod | The suite passes; the product carries a scar and an attack surface. |
| Skip-flags to appease infra | Papering over a real ordering/dependency bug and shipping the paper. |
| Rerun-and-hope | Turns your instrument into a slot machine. |
