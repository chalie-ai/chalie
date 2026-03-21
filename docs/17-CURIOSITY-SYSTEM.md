# Curiosity System

## Overview

Chalie generates its own questions. During idle periods, the Reasoning Loop Service can seed a **curiosity thread** — a self-directed line of exploration that Chalie pursues in the background via its ACT loop. This is not task execution driven by user requests; it is exploratory cognition initiated by Chalie itself, shaped by its accumulated memory and identity.

Curiosity threads are one of the mechanisms through which Chalie's knowledge and identity evolve independently. Two instances of Chalie with different conversation histories will form different curiosity threads, explore different topics, and diverge over time.

---

## Two Thread Types

| Type | What It Explores | Example |
|------|-----------------|---------|
| `learning` | A topic or concept Chalie encountered but doesn't fully understand | "User mentioned reinforcement learning — I have surface-level knowledge, I should understand it better" |
| `behavioral` | A pattern in the user's behavior Chalie wants to understand more deeply | "User consistently disengages on Monday mornings — I wonder why" |

Both types follow the same lifecycle but differ in their seeding source: learning threads come from knowledge gaps encountered during conversations; behavioral threads come from recurring capability gaps surfaced by the Self Model Service (`SelfModelService.get_frequent_gaps`) during each pursuit cycle.

---

## Services

### Curiosity Thread Service (`services/curiosity_thread_service.py`)

CRUD and lifecycle management for curiosity threads. Threads are stored in the `curiosity_threads` SQLite table with state tracking (`active`, `dormant`, `abandoned`).

Core operations:
- **Seed** — Creates a new thread in `active` state from a drift observation or knowledge gap signal; deduplicated by `seed_topic`
- **Dormant** — Thread transitions to `dormant` when engagement score drops below 0.2 or the thread goes unexplored for 45 days
- **Abandon** — Thread transitions to `abandoned` when dormant for 60+ days with engagement still below 0.2

---

### Curiosity Pursuit Service (`services/curiosity_pursuit_service.py`)

The background worker that actually explores active threads. Runs on a **6-hour cycle**, picking up threads in `active` state and executing them via the ACT loop.

Pursuit looks like any other ACT loop execution:
- Tools can be invoked (web search, recall, introspect)
- Results are evaluated by the Critic Service
- Findings feed back into episodic memory and semantic consolidation
- If a finding is significant, it may surface as a drift thought in a future conversation

**Budget constraints:** The pursuit service uses a bounded ACT loop governed by an iteration cap. The fatigue budget parameter is accepted for call-site compatibility but is ignored at runtime — only the iteration cap constrains execution. The service also self-regulates: it skips cycles when memory richness is below 0.15 or when user-facing worker queues are non-empty.

---

## Lifecycle

```
[ReasoningLoopService] — idle period
  |
  +-- Drift action: SeedThreadAction
      |
      Identifies a knowledge gap or behavioral pattern worth exploring
      Creates curiosity_thread record (state: 'active')

[Curiosity Pursuit Service] — 6h cycle (±30% jitter)
  |
  Also seeds behavioral threads from SelfModelService.get_frequent_gaps()
  |
  Picks up active threads eligible for exploration
  |
  +-- Runs bounded ACT loop (iteration-capped):
      ├── recall (what do I already know?)
      ├── tool invocation (web search, etc.)
      ├── critic evaluation
      └── findings → learning note → episodic memory → semantic consolidation
  |
  Thread state: active → dormant (engagement drops below 0.2, or 45 days unexplored)
               dormant → abandoned (dormant 60+ days AND engagement still < 0.2)

[Findings]
  |
  +-- New semantic concepts stored in SQLite
  +-- User trait updates (if behavioral thread)
  +-- May surface as drift thought in future conversation
```

---

## Seeding: Where Curiosity Comes From

Threads are seeded by the Reasoning Loop Service during idle periods. The reasoning loop has a dedicated autonomous action — `SeedThreadAction` — that fires when the loop detects a candidate worthy of deeper exploration. Additionally, the Curiosity Pursuit Service directly seeds behavioral threads each cycle from recurring capability gaps tracked by `SelfModelService`. Candidates are evaluated against:

1. **Relevance** — Is this connected to topics the user cares about?
2. **Knowledge gap** — Does Chalie have shallow or uncertain knowledge here?
3. **Recency** — Was this topic encountered recently (high recency = higher seed priority)?
4. **Expected value** — Would resolving this genuinely help future interactions?

If most answers are yes, a thread is seeded. If not, the drift cycle continues to other actions (COMMUNICATE, REFLECT, PLAN, RECONCILE).

---

## Surfacing Findings

Curiosity findings are not pushed to the user unprompted. They enter the normal memory pipeline:
- Findings become semantic concepts (decaying knowledge nodes)
- Related episodes get salience boosts
- In a future conversation where the topic surfaces naturally, the assembled context will include the finding
- If significant, the drift engine may generate a `COMMUNICATE` thought: "I looked into X and found something interesting..."

This follows the core design principle: involve the user only when it matters. Curiosity findings surface when relevant, not when completed.

---

## Observability

Active curiosity threads are visible via the Brain dashboard and the observability API:

```
GET /system/observability/tasks
```

Returns active persistent tasks, curiosity threads (with state, type, topic, and progress), and triage calibration data.

---

## Related

- **`04-ARCHITECTURE.md`** — full service listing under "Autonomous Behavior"
- **`09-TOOLS.md`** — how tools are invoked during curiosity pursuit
