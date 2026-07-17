---
name: ui-ux-principles
description: Use when building or changing anything a user sees or touches — views, components, states, styling, copy, client-side data flow, or frontend architecture.
---

# UI/UX Principles

## Overview

The UI is where all backend honesty either survives or dies. Two commitments govern everything: **the interface never lies about system state**, and **the client never owns the truth.**

## Data Flow — the Render Doctrine

- **The API is the only source of data.** Push channels (websockets, events) carry *triggers*, never authoritative payloads: on receiving one, refetch from the API or flip a transient visual flag. This structurally kills client-side drift — a buffer that doesn't exist can't diverge from the server.
- **Templates render; logic lives elsewhere.** Views/components hold markup and bindings. State, API calls, and business rules live in composables/stores/services. A component with a fetch in it is a lint error in spirit.
- **The backend formats; the frontend displays.** Timestamps, numbers, locale — formatted server-side by the single authority. Client-side parsing of server data is a second implementation waiting to disagree.
- **No client-side mirrors of server state.** Every locally-cached copy of the truth is a drift generator. Refetch is cheap; reconciliation bugs are not.

## State Honesty

- **Every view designs all four states:** loading, empty, error, success. "Empty" and "error" are the states new customers meet first — they're the front door, not edge cases.
- **Never a dead screen.** From first paint, the user sees the truth: a boot/holding screen with honest status until the system is actually ready. A blank page during startup reads as "broken," and to the user, is.
- **Blocking operations block visibly.** If the system cannot proceed until work completes (a file still processing, a save in flight), the UI says so and prevents the conflicting action. Letting the user continue and silently producing wrong results is the worst UI bug there is.
- **Errors reach the user in their language.** What happened, what to do next — product vocabulary, never raw exception text, never internal jargon, and never silence.
- **Feedback is immediate.** Every action acknowledges instantly; long work shows honest progress; destructive actions confirm; disabled controls explain why.

## Consistency & Craft

- **One pattern per problem.** The same interaction solved two ways is a bug in the design system. Reuse the established pattern or explicitly replace it everywhere — never fork it.
- **Vocabulary is fixed.** UI copy uses the product's exact established names. Never invent a synonym for an existing concept; every invented name is a translation the user pays for.
- **Theme discipline.** All colors through design tokens/CSS variables — never hardcoded values. Every change verified in both light and dark themes before it's done.
- **Hierarchy over decoration.** The most important element is visually primary; progressive disclosure over wall-of-controls. Imagery must demonstrate or inform — decorative noise dilutes the signal.
- **Guard the redirect/navigation logic like auth code** — because it usually is. Navigation guards comparing the wrong state (current vs target) have produced infinite redirect loops and request floods here. Navigation logic gets the same review rigor as backend logic.

## Anti-Patterns

| Anti-pattern | Why it's fatal |
|---|---|
| Authoritative data over the push channel | Client and server now disagree about the truth; every reconnect is a coin flip. |
| Optimistic UI without rollback | You showed a success that may not have happened. That's lying with extra steps. |
| Silent UI failure | A failed action with no feedback teaches the user the product is haunted. |
| Spinner-forever | An unbounded wait with no status is a dead screen with a costume. |
| Hardcoded colors/spacing | Breaks theming today, guarantees inconsistency forever. |
| Inventing UI names for existing concepts | The docs, the UI, and the code now speak three languages. |
