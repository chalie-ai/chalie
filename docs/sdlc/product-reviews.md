---
name: product-reviews
description: Use when evaluating the product as a whole — before a release, after a feature lands, or on a recurring cadence — to judge what a customer will actually experience.
---

# Product Reviews

## Overview

A product review answers one question: **what does a customer actually experience?** Not what the code does, not what the tests assert — what a person sees, waits for, and feels. There is no substitute for using the product; every automated gate is a precondition to this review, never a replacement for it.

Customer trust is the asset. It is built slowly through honesty and quality, and spent instantly — a bad day one kills retention permanently.

## The Method

1. **Become a customer, not the author.** Fresh environment, fresh account, empty data, default config, modest hardware. The author's warm dev setup hides an entire class of failure — a system here was flawless in dev and a dead blank screen on every fresh install, because only fresh installs paid the cold-boot cost.
2. **Live the first 30 minutes.** Install → first boot → first meaningful success. Time it. Every stumble in this window is weighted 10x — it's the only window every customer is guaranteed to experience.
3. **Walk the golden paths end-to-end, then the ugly ones.** Empty states, error states, the slow machine, the flaky network, the impatient double-click, walking away mid-operation and coming back. The demo path is rehearsed; customers improvise.
4. **Judge by the vision and the whole audience.** Not by your own setup, your own providers, your own habits. Users run configurations you don't; the product filter question (defined in `brainstorming.md`) is the yardstick, not personal taste.
5. **Audit the words.** Error messages must say what happened and what to do next, in product vocabulary. Internal jargon, raw exception text, or invented names leaking into the UI is a defect, same severity as a logic bug.
6. **Verify the release story.** For each shipped change: state in one sentence what changed *for the user*. If no such sentence exists, why did it ship? If the sentence overpromises, the release notes are lying — release notes are developer-honest, not marketing fluff.
7. **Check the waits.** Every long operation: does the product tell the truth about what it's doing? A holding screen with honest status beats a frozen or blank one; an operation that blocks must *visibly* block, not pretend to be done.

## Verdict Contract

The review ends in a written verdict:

- **Ship / don't ship**, with the blocking findings if any.
- Findings filed as tickets immediately — a finding in your head is a finding lost.
- Each finding described as the customer experiences it ("uploading a file lets me keep chatting, then answers without the file's content"), not as the developer explains it.

## Anti-Patterns

| Anti-pattern | Why it's fatal |
|---|---|
| Developer-eyes testing | You know where not to click. Customers don't. |
| Works-on-my-machine sign-off | Your machine is warm, fast, seeded, and configured. Theirs is none of those. |
| Demo-path-only review | The rehearsed path is the one path guaranteed to work; it carries no information. |
| Green-suite complacency | Suites prove the code you tested; the customer runs the product you shipped. |
| Polish as optional | To a customer, a rough edge and a bug are the same thing: evidence of carelessness. |
| Reviewing with seeded/ideal data | Empty and messy data are the customer's default, not the edge case. |
