# System Architecture

## What Chalie Is

Chalie is a persistent cognitive runtime — a single Python process that keeps thinking between conversations. It is not a request-response wrapper around an LLM. Every message flows through a multi-layer memory pipeline, background workers maintain and decay knowledge while the user is idle, and the system forms a continuously evolving model of the person it talks to. Intelligence accumulates over time; it does not reset per session.

The stack: Flask + flask-sock for HTTP and WebSocket, SQLite (WAL mode, sqlite-vec for vector search, FTS5 for keyword search), an in-process thread-safe MemoryStore (no Redis), and a pluggable LLM provider layer supporting Ollama, Anthropic, OpenAI, and Google Gemini. Everything runs in one process.

---

## How a Message Flows

A user message arrives over WebSocket. The handler spawns a daemon thread, constructs a `UserMessageProcessor` for this turn, and calls `send()`. Nothing else touches the message.

Inside `send()`:

1. **Memory seed** — recent episodes are retrieved and attached to the turn context.
2. **Thinking gate** — a lightweight ONNX classifier reads the message and assigns a deliberation depth: low (conversational), medium, or high. High depth triggers a one-shot pre-reasoning pass before the tool loop begins.
3. **ACT loop** — the processor assembles a single user message containing the literal conversation history, world state, memory seed, and the current input, then calls the LLM. If the LLM invokes a tool, the result is appended to the trail and the loop continues. This repeats until the LLM returns a plain text response or hits the iteration cap.
4. **Atomic write** — one SQLite transaction commits the user turn, every tool call from the loop, and the assistant response. Nothing is written to the database mid-loop.
5. **Post-turn fan-out** — services that react to a completed turn (conversation phase update, adaptive signals, DMN timer reset, metrics) run after the atomic write. The response is already on its way to the client before fan-out begins.

```
WebSocket frame
  └─ daemon thread
       └─ UserMessageProcessor.send()
            ├─ memory seed
            ├─ thinking gate  (classify → optional exploration pass)
            ├─ ACT loop ──────────────────────────────────────────┐
            │    assemble prompt (history + world state + seed)   │
            │    → LLM call                                       │
            │    → tool calls → tool results → back to LLM  ─────┘
            ├─ atomic write (transcript + tool_calls, one tx)
            └─ post-turn fan-out → response → client
```

See `docs/13-MESSAGE-FLOW.md` for the full turn lifecycle.

---

## Message Processors

`MessageProcessor` is the abstract base for every LLM turn in the system. The architectural rules are simple:

- **One class per channel.** User messages, DMN thoughts, goal pursuit, scheduled prompts, and internal encoders each have their own subclass. There is no shared dispatcher or central router.
- **One instance per turn.** All turn state lives on the instance. No singletons, no shared instances.
- **Subclasses hardcode their channel and role.** A processor knows what it is. Context scoping flows from that identity.
- **Atomic store at the end.** `store()` commits everything in one transaction when the ACT loop finishes.
- **`handleTool()` is the single dispatch chokepoint.** Tool errors return structured strings to the LLM; they never surface to the user or crash the loop.
- **`postTurn()` is where channel-specific fan-out lives.** Shared plumbing goes in the base; subclass-specific services go in the subclass.

History reaches the LLM as a literal `## Previous Messages` text block inside the user message body. The provider always receives a single-element `messages[]` array — not a multi-turn array. This is an intentional design choice: it gives the system full control over what context the model sees on each turn.

Internal processors (episode encoders, the user summary synthesiser) set a flag that suppresses transcript writes — they run the ACT loop without polluting the conversation record.

---

## Memory Hierarchy

Four layers, each optimised for a different timescale and purpose:

| Layer | What it stores | Decays? |
|-------|---------------|---------|
| **Transcript** | Append-only conversation record, channel-scoped | No (pruned after 90 days) |
| **Compaction** | LLM-generated summary of history beyond context limit | No (one per channel) |
| **Episodes** | Narrative units extracted from transcript windows | Yes — power-law retrieval weight decay |
| **Data Graph** | Structured knowledge (facts, preferences, moments) | Yes — per-kind decay policy |

**Transcript** is the raw record. `getPreviousMessages()` renders everything above the compaction watermark as a literal text block. When that block approaches the provider's context limit, compaction fires and summarises the older portion — the summary becomes the new floor.

**Episodes** are extracted automatically by a rolling trigger: when enough new transcript lines have accumulated for a channel, a background processor encodes that window into narrative snapshots with emotional valence, arousal, and salience scores. Similar episodes consolidate into super-episodes over time. Retrieval uses hybrid vector + FTS5 search, adaptive radius, and apex traversal (following consolidation links upward).

**Data Graph** is the knowledge layer. Writes for user-specific facts go through a canonicalisation engine: the key is compared against a set of high-level concepts, and a rule (temporal supersede / coexist additive / immutable block) is applied. This prevents duplicate or contradictory facts from accumulating. The database shape lives in `backend/schema.sql`.

**Query expansion.** Every knowledge and data-graph write is enqueued to the `SearchExpanderService` — a single boot-time FIFO daemon. It generates paraphrased variants via doc2query, embeds each, and writes them to `expanded_semantic` + `expanded_semantic_vec` keyed back to the source rowid. Recall adds a KNN signal against the variant index so paraphrased questions hit the right facts even when the literal surface form does not match. The daemon is event-driven (not busy-loop) and self-heals on boot by rescanning rows with `search_queries IS NULL`.

---

## Background Reasoning

Chalie keeps thinking when you are not typing. Background workers run as daemon threads in the same process:

- **Decay engine** — periodically applies power-law decay to episode retrieval weights and data-graph entries, consolidates similar episodes into super-episodes, and purges old transcript entries below the compaction watermark.
- **DMN (Default Mode Network)** — after a period of idle time, Chalie initiates a proactive thought using recent or high-salience episodes as context. Uses its own `MessageProcessor` subclass; exits silently when nothing warrants a response.
- **Goal pursuit** — long-running background tasks spawned by the `goal_pursuit` innate skill. Each runs its own processor with a high iteration cap and surfaces its result as a proactive message when complete.
- **Scheduled prompts** — the scheduler fires due reminders and timed tasks via their own processor subclass.
- **Supporting workers** — user summary synthesis, world awareness (weather, news), moment context enrichment, document purge, folder watcher, interface health monitor, self-model health signals, optional profile enrichment, and the `SearchExpanderService` (single FIFO consumer that generates + embeds query variants for every new knowledge/data-graph row).

No worker shares its processor instance with another. Each channel is fully isolated.

---

## Tools and Skills

Two tiers:

**Innate skills** are always available to the LLM. They cover memory (store, recall, reflect, forget), scheduling, list management, goal pursuit, document search, web reading, introspection, rich rendering, and tool discovery. These are core cognitive capabilities wired directly into every turn.

**External tools** are never pre-injected. The `find_tools` innate skill performs semantic search against tool capability profiles at runtime. When the LLM invokes `find_tools`, the matching tools become available for the remainder of that ACT loop. This is not a convenience optimisation — pre-injecting external tool schemas into every turn would bloat context, create staleness bugs, and break tool-agnostic routing.

Tool results flow through a single render-and-record path that formats the output and writes it to the `tool_calls` table. Tool infrastructure has no knowledge of specific tools; tools have no knowledge of infrastructure. See `docs/09-TOOLS.md` and `docs/15-INTERFACES.md`.

---

## Ambient Awareness

`WorldState` maintains a lightweight in-process snapshot of four typed facts: last user message timestamp, last heartbeat timestamp, current device class, and client-reported local time. The snapshot is updated via `world_state.absorb(Signal(...))` — called from the WebSocket handler on every user message and from POST `/health` on every client heartbeat.

`absorb()` and `snapshot()` operate on an in-process dict with no database writes and no LLM calls. Unknown signal kinds are silently ignored. All four snapshot fields default to `None` when no signal of that kind has been received since boot.

See `docs/16-AMBIENT-AWARENESS.md` for the full `Signal` dataclass contract and call sites.

---

## Frontend

Four independent single-page applications: the main chat interface, the brain admin dashboard, the onboarding wizard, and the login form. A shared auth-gate module enforces redirect rules uniformly.

The chat interface is built from focused ES6 modules wired together by a thin orchestrator (`app.js`). Modules communicate through constructor injection, callback registration, and custom DOM events. No module references another directly.

**Block protocol:** all LLM-to-client content is JSON arrays of typed block objects. No HTML travels over the wire. The backend renderer produces blocks; `blocks.js` in the frontend renders them to DOM.

**Asset versioning:** every static asset reference in served HTML has the version string injected into its filename at response time (e.g. `app.js` becomes `app-0.3.3.js`). Static routes strip the version suffix before the disk lookup, so nothing is renamed on disk. Versioned filenames are used instead of query strings because some service workers and proxies ignore query strings when keying caches. HTML responses themselves are never cached.

See `docs/03-WEB-INTERFACE.md` for the full Radiant design system spec.

---

## REST API

The REST API covers conversation, memory, privacy, providers, tools, scheduling, lists, and observability. Endpoints live in `backend/api/`. See individual blueprints for the full surface.

---

## Key Architectural Rules

These are invariants, not conventions. Violating them creates systemic problems.

- **Atomic per-turn persistence.** All database writes for a turn happen in a single transaction at the end of the ACT loop. No mid-loop writes.
- **Literal-text history.** Previous messages are rendered as a text block, not a multi-turn `messages[]` array. The provider always sees one user message.
- **Flat channels.** A channel is a stable string identifier. Additional routing context (goal ID, scheduled item ID) lives in metadata, not in the channel string.
- **Channel-scoped data.** Transcript, compaction, and episode data are keyed by channel. No cross-channel leakage.
- **Tool agnosticism.** No tool-specific logic in triage, dispatch, or frontend rendering. Tools self-declare via manifests. Innate skills are the exception.
- **No external tool pre-injection.** External tools are discovered at runtime via `find_tools` only. Never add tool schemas to the native tools list, system prompt, or pre-loaded context.
- **Clean removal.** When a service, class, or function is removed, delete it completely — file, imports, callers, tests. No hollow passthroughs, no deprecated shims, no re-exports.
- **Model-agnostic.** Different cognitive functions may use different LLM providers. Nothing is hardcoded to a specific model.

---

## Glossary

| Term | Meaning |
|------|---------|
| **MessageProcessor** | Abstract base for all LLM turns. One instance per turn, one subclass per channel. |
| **Channel** | Stable string scoping transcript and compaction data (e.g. `user`, `dmn`, `goal_pursuit`). |
| **Block Protocol** | Content format: JSON arrays of typed block objects, backend → frontend. No HTML over the wire. |
| **DMN** | Default Mode Network — timer-based proactive intelligence that fires during idle periods. |
| **Episode** | Narrative memory unit extracted from transcript windows. Has salience score and decaying retrieval weight. |
| **Data Graph** | Structured knowledge store with canonicalisation, typed edges, and per-kind decay. |
| **Salience** | Computed importance score [1–10] based on emotional arousal, valence, open loops, and novelty. |

---

Testing: `docs/12-TESTING.md`. Development setup: `docs/01-QUICK-START.md`. Interface pairing: `docs/15-INTERFACES.md`.
