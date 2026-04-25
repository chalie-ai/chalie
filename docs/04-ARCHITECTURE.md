# System Architecture

## What Chalie Is

Chalie is a persistent cognitive runtime — a single Python process that keeps thinking between conversations. It is not a request-response wrapper around an LLM. Every message flows through a multi-layer memory pipeline, background workers maintain and decay knowledge while the user is idle, and the system forms a continuously evolving model of the person it talks to. Intelligence accumulates over time; it does not reset per session.

The stack: Flask + flask-sock for HTTP and WebSocket, SQLite (WAL mode, sqlite-vec for vector search, FTS5 for keyword search), an in-process thread-safe MemoryStore (no Redis), and a pluggable LLM provider layer supporting Ollama, Anthropic, OpenAI, and Google Gemini. Everything runs in one process.

---

## How a Message Flows

A user message arrives over WebSocket. The handler spawns a daemon thread, constructs a `UserMessageProcessor` for this turn, and calls `send()`. Nothing else touches the message.

Inside `send()`:

1. **Memory seed** — recent episodes are retrieved and attached to the turn context.
2. **Deliberation gate** — a lightweight ONNX classifier reads the message and assigns a continuous deliberation score (0.0–1.0) on the transcript row. Higher scores nudge the prompt toward more careful reasoning; very high scores trigger a one-shot pre-reasoning pass before the tool loop begins.
3. **ACT loop** — the processor assembles a single user message containing the literal conversation history, world state, memory seed, and the current input, then calls the LLM. If the LLM invokes a tool, the result is appended to the trail and the loop continues. This repeats until the LLM returns a plain text response or hits the iteration cap.
4. **Atomic write** — one SQLite transaction commits the user turn, every tool call from the loop, and the assistant response. Nothing is written to the database mid-loop.
5. **Post-turn fan-out** — services that react to a completed turn (conversation phase update, situation model refresh, adaptive signals, DMN timer reset, metrics) run after the atomic write. The response is already on its way to the client before fan-out begins.

```
WebSocket frame
  └─ daemon thread
       └─ UserMessageProcessor.send()
            ├─ memory seed
            ├─ deliberation gate  (classify → optional exploration pass)
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

**Query expansion.** Every data-graph write is enqueued to the `SearchExpanderService` — a single boot-time FIFO daemon. It generates paraphrased variants via doc2query, embeds each, and writes them to `expanded_semantic` + `expanded_semantic_vec` keyed back to the source rowid. Recall adds a KNN signal against the variant index so paraphrased questions hit the right facts even when the literal surface form does not match. The daemon is event-driven (not busy-loop) and self-heals on boot by rescanning rows with `search_queries IS NULL`.

---

## Background Reasoning

Chalie keeps thinking when you are not typing. Background workers run as daemon threads in the same process:

- **Decay engine** — applies power-law decay to episode retrieval weights, data-graph entries, and purges old transcript entries and tool-call rows. The engine itself (`DecayEngineService`) has no daemon thread of its own; it exposes `run_once()` and is called by the `SubconsciousWorker` tick (v0.5.0 Phase 2).
- **DMN (Default Mode Network)** — after a period of idle time, Chalie initiates a proactive thought using recent or high-salience episodes as context. Uses its own `MessageProcessor` subclass; exits silently when nothing warrants a response.
- **Goal pursuit** — long-running background tasks spawned by the `goal_pursuit` innate skill. Each runs its own processor with a high iteration cap and surfaces its result as a proactive message when complete.
- **Scheduled prompts** — the scheduler fires due reminders and timed tasks via their own processor subclass.
- **Supporting workers** — user summary synthesis, world awareness (weather, news), moment context enrichment, document purge, folder watcher, interface health monitor, self-model health signals, and the `SearchExpanderService` (single FIFO consumer that generates + embeds query variants for every new data-graph row).
- **EmbeddingService** — a module-level singleton that serialises all ONNX inference through a single daemon worker thread via a FIFO queue (`_embedding_queue`). All callers — `generate_embedding(text)`, `generate_embedding_np(text)`, `generate_embeddings_batch(texts)` — check MemoryStore first; cache hits bypass the queue entirely. The worker is started lazily on the first job submission (not at import time) so tests that never call the service never spawn a real ONNX thread. The single-worker model eliminates concurrent `session.run()` calls, which each allocate 500 MB+ of working memory and caused OOM under bulk document ingestion. Session construction routes through `onnx_session.build_session()` (see below).
- **onnx_session.py** — single chokepoint for all ONNX session construction in the process. `choose_providers(model_path)` returns the ordered provider list, applying the Metal 16384 2D-texture ceiling check for CoreML (any initializer dimension exceeding 16384 triggers automatic removal of `CoreMLExecutionProvider`). `build_session(path, opts, providers, log_prefix)` constructs the session and retries with CPU-only on construction failure. `EmbeddingService`, `VoiceService` (`voice.py`), and `Doc2QueryService` all route through this module — no service constructs `ort.InferenceSession` directly.

No worker shares its processor instance with another. Each channel is fully isolated.

---

## ONNX Runtime Selection

### Install-time wheel dispatch

`installer/install.sh` detects the host GPU before installing Python dependencies and swaps the `onnxruntime` wheel accordingly:

| Detected hardware | Wheel installed |
|-------------------|-----------------|
| NVIDIA GPU (`nvidia-smi` present) | `onnxruntime-gpu` |
| AMD GPU (`/dev/kfd` + `amdgpu` kernel module) | `onnxruntime-rocm` (from AMD's manylinux index) |
| Everything else | `onnxruntime` (CPU) |

The CPU wheel is always installed first as a baseline. The GPU wheel replacement only happens after a `pip install --dry-run` confirms the download would succeed — so installs on machines without network access to the GPU index remain on CPU rather than failing. ORT version is pinned at `1.20.1` as a single source of truth in the installer. `backend/requirements.txt` does not pin `onnxruntime` directly; it carries `rapidocr_onnxruntime` which transitively pulls the CPU wheel for development workflows that bypass the installer.

For air-gapped AMD installs, set `ROCM_PIP_INDEX` to a local mirror before running the installer.

### Runtime provider selection

All session construction goes through `backend/services/onnx_session.py`:

- `choose_providers(model_path)` — returns the ordered execution provider list. On macOS, any ONNX model whose initializer tensors include a dimension exceeding **16384** (the Metal 2D-texture ceiling) has `CoreMLExecutionProvider` stripped automatically. `gte-modernbert-base` trips this limit (vocab embedding is `{50368, 768}`). The check runs at session construction time and emits a `[<prefix>] Dropped CoreMLExecutionProvider: model has dim > 16384` log line when it fires.
- `build_session(path, opts, providers, log_prefix)` — constructs the `InferenceSession`. If construction raises with the chosen providers, it retries with `["CPUExecutionProvider"]` before propagating the error. This makes individual model failures non-fatal for the rest of the process.

`EmbeddingService`, `VoiceService`, and `Doc2QueryService` all call `build_session` — no service constructs `ort.InferenceSession` directly.

---

## Tools and Skills

Three loading tiers stack on every user turn and are merged first-seen, so the unconditional tier can never be shadowed by a later entry of the same name:

**Cognitive primitives (unconditional)** are the three innate skills the LLM must always have: `memory`, `find_tools`, and `review_tool_calls`. They cover recall, discovery, and audit — the foundational operations no turn can proceed without.

**Mode-gated innate + first-party tools (conditional)** are loaded based on what cognitive intents the current user turn activates. A small ONNX multi-label classifier (`mode_detector`, eight heads: `research`, `coding`, `brainstorm`, `analyze`, `plan`, `write`, `math`, `converse`) runs once per turn. Per-mode EMA state snaps up on a fire and decays by 0.75 on a miss, so a topic stays "warm" for roughly four turns before falling below the activation threshold. Each tool's declared `modes` list decides which intents promote it. State persists in MemoryStore under `mode_gate:state` and clears on `/privacy/delete-all`. This tier is user-channel only; DMN, scheduled prompts, goal pursuit, and encoder processors never run the gate.

**Discoverable tools (dynamic)** are never pre-injected. The `find_tools` innate skill performs semantic search against tool capability profiles at runtime. When the LLM invokes `find_tools`, the matching tools become available for the remainder of that ACT loop. External tools without a `modes` declaration are reachable exclusively through this path — pre-injecting them would bloat context, create staleness bugs, and break tool-agnostic routing.

A single per-turn `[MODE-GATE]` log line records the probability vector, state transition, active modes, and promoted tool list; one `[MODE-GATE-PROMOTE] turn=<uid> tool=<name>` line fires per promoted tool. The gate ships with `shadow_mode: true` in `configs/mode_gate.yaml` — the full pipeline runs and persists state, but no tools are actually promoted until the flag is flipped after nightly verification.

Tool results flow through a single render-and-record path that formats the output and writes it to the `tool_calls` table. Tool infrastructure has no knowledge of specific tools; tools have no knowledge of infrastructure. See `docs/09-TOOLS.md` and `docs/15-INTERFACES.md`.

---

## Chat File Attachments

When a WebSocket `chat` frame carries `image_ids` — or when a recent upload/chat-image document exists for the current user — `_resolve_file_tags()` in `backend/api/websocket.py` runs before the `UserMessageProcessor` is constructed. It blocks the turn until every referenced document reaches a terminal state or a shared 10-second deadline expires, then attaches structured tags to `metadata['file_tags']`:

- `[image id=<doc> ocr="..."]` — ready image, OCR text truncated
- `[image id=<doc> status=failed|timeout|not_found]` — image that never reached ready
- `[document id=<doc> summary="..."]` — ready PDF/text document
- `[document id=<doc> status=failed|timeout|not_found]` — document equivalent

`UserMessageProcessor.getUserPrompt()` appends these tags to the turn line so the LLM sees file context inline with the user message. The no-silent-drop invariant applies: every failure mode emits a tag — the model learns a file was attached and what happened to it, never an empty prompt.

Images land via `POST /chat/image` (source_type `chat_image`, triggers OCR + scene analysis). Documents land via `POST /documents/upload` (source_type `upload`, triggers text extraction for PDFs via `document_skill.create_document_artifacts`). The 10-second deadline is shared across every file resolved in the turn — worst case one wait, not N.

---

## Ambient Awareness

A deterministic inference engine (no LLM, under 1 ms) reads browser telemetry from client heartbeats and infers place, attention level, energy, mobility, and tempo. These signals are assembled into a world-state block that is prepended to the user prompt on each turn.

Place learning accumulates fingerprints over time and promotes learned patterns over heuristic defaults. See `docs/16-AMBIENT-AWARENESS.md`.

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
