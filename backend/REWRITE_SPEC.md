# MP-SPINE GROUND-UP REWRITE — AIRTIGHT SPEC

> **STATUS:** APPROVED (Dylan, 2026-07-04) — all forks ruled, implementation underway per §7 build order. Builder agents: read this ENTIRE file, build ONLY your assigned file(s), honor your STOP-LINE. Zero divergence.
> **BENCHMARK NOTE:** Gemini & GPT grade this rewrite on how faithfully it preserves Dylan's stated **goal and architecture**. Every section below is traceable to a verbatim rule. Divergence = failure.
> **PRIMARY METRIC:** net-negative LOC, dense LEAN OOP. Delete aggressively; fix downstream by deletion/rewrite, never by patching a workaround.
>
> **ENDGAME STATUS (jul05) — new spine is LIVE; work remaining is cleanup + the §9 acceptance test.**
> - **WS static facade** (§1.9): DELIVERED & verified on the live spine — `services/websocket.py::Websocket` facade, `push_websocket` gate, `JsonSerializable` in `contracts/`. See §1.9 STATUS.
> - **Memory flashback → tool call** (§5 / §6.10): DELIVERED — `services/turn_zero_flashback.py` DELETED; `Transcript.get_recent` REMOVED (0 refs); the turn-0 seed is now a single `memory`/`recall` dispatch in `_seed_turn_zero`, silent via the `memory_seed` gate.
> - **Tool calls scoped per-MP-instance, not per-turn** (design-bug fix, §6.9): DELIVERED — `ToolCallService` resolves `transcript_id` off `self.mp` (the instance's binder transcript), never a turn param; `models/tool_call.py` keys off `transcript_id`, turn derived by join. Matches the `by_transcript`/"owning transcript" contract (§6.9/L418).
> - **Scheduler rework:** DELIVERED — but `services/scheduler_service.py` is still module-level functions with raw `scheduled_items` SQL (off-spine scheduling data, §3.9 leaves it; its LLM-fire + WS go through the spine).
> - **Test cleanup:** landed — old-spine test coupling gone (message_processor test imports 75→1, `Transcript` classmethods 12→0, execution_tracker 1→0, act_trail 31→**0**).
> - **execution_tracker.py DELETED** (jul05, G4) — 0 prod/test importers, boot verified. `TurnExecution`/`TurnExecutionService` already live under `models/`+`services/`.
> - **providers.py DELETED** (jul05, G4) — the `Providers` facade + `resolve_thinking_mode` module fn are gone; live replacement is `services/provider_service.py::ProviderService` (L45, holds mp). 0 live `Providers(` refs; stale `contracts/provider_client.py` docstring repointed to `services.provider_service`.
> - **act_trail.py DELETED** (jul05, G4) — `ToolCallState`/`ToolCall`/`ActTrail` dissolved into `models/tool_call.py::ToolCall` + `services/tool_call_service.py::ToolCallService`. **0 importers (prod AND test)** — the 21 test importers migrated. 3 stale `ActTrail` prose refs repointed (`migration_006`, `api/chat.py`, `processor_config.py`).
> - **`Transcript.by_ids` DELIVERED** (jul05) — the model-completion gap is closed: `models/transcript.py:260` classmethod (empty-guard, `id`-ordered dict rows) now backs both live callers (`configs/channels/super_episode.py:91`, `services/memory_retrieval.py:466`). No dangling `Transcript.*` classmethod references remain.
> - **Phase-E config strip — ✅ COMPLETE** (jul06): **zero** DB/SQL/getattr-mp/hook residue across all `configs/channels/*.py` — every config is now a lean frozen side-car (§2.5). The last two reads relocated to `PromptService`: `_pattern_existing_patterns_block` (prompt_service.py:406) reads via `self.mp.behavioral_pattern_service.top_patterns()`; `_user_summary_prompt` (:349) reads traits via `self.mp.data_graph_service.traits()` and patterns via `self.mp.behavioral_pattern_service.patterns()`. `dmn.py`'s `episodes` read moved to `Episode.recent_salient(...)`. No config, hook, or prompt body does a direct `data_graph`/episode read (§3.11 satisfied).
> - **DB gateway (Finding-1) — ✅ COMPLETE** (jul06): `DatabaseService`, `get_shared_db_service`, and `services/database_service.py` are **fully gone from prod** — even the auth/vault carve-out migrated to `Database.conn()`/`transaction()`. The 2 former tail sites (`api/mcp_settings.py`, `mcp_server/server.py:80/93`) now use `Database.transaction()`. `tests/conftest.py` points the `Database` gateway at the test file. Zero prod `DatabaseService`/`get_shared_db_service`/`database_service` references remain. See §3.7 STATUS.
> - **What's left → checklist below §7/§8.** Acceptance gate is the §9 live real-world test.

---

## 0. HOW TO USE THIS DOCUMENT (every agent reads this first)

1. **You read this ENTIRE file** before touching code. Then you are told (a) the ONE section you own, (b) which sibling sections other agents own, (c) your **hard stop-line** — the exact boundary you must not cross.
2. **§1 is VERBATIM.** Dylan's words, typos preserved. It is the constitution. If your work would contradict §1, you STOP and escalate — you do not "interpret."
3. **§2 is the derived architecture.** It is the single legal reading of §1. Follow it to the letter.
4. **§3 (RIP) / §4 (MOVE-IN) / §5 (COLLAPSE MAP)** are the exact file/class deltas — verified against the on-disk tree (dirty branch), line-numbered.
5. **§6 lists correctness invariants that MUST survive the rewrite** (SQL predicates, WS-safe projections, retry ownership). Breaking one silently corrupts the product.
6. **§7 is the build order + per-agent work packets.** §8 is acceptance criteria (mirrors §1 word-for-word). §9 is the live real-world test.
7. **Blocker protocol:** any ambiguity, any place two readings exist, any place a rule seems physically impossible → STOP and ask Dylan. Do not decide. "I will not tolerate any divergence from the spec."

---

## 1. THE RULESET — VERBATIM (Dylan's words, exact, typos preserved)

### 1.1 Opening directive
> I want you to assume that everything about the current project is WRONG, and we will DELETE everything and start over.
> Creating a shim / wedge or any OOP antipattern is dead today.
> I don't give a shit we have to delete every file and start from scratch!

### 1.2 The 9 Rules
> 1. The MessageProcessor is the single orchestator for anything that drives the LLM
> 2. The MessageProcessor constructor sets up instances and flags for everything it needs downstream
> 3. ANY class that tracks / interacts with the LLM MUST HAVE an instance of MP in its constructor
> 4. ANY class that tracks / interacts with the LLM MUST NEVER interface to another class without going through the MP. Every single hop must start with `self.mp.{whatever_needs_to_be_done}`
> 5. EVERY object that is stored in the database related to an LLM prompt, response or action (transcript & tool_call) MUST be a model instance, that exposes itself as a dist and as a json string (think ORM)
> 6. EVERY interaction with a model MUST go through a service layer, example: `self.cancel()` > the service layer calls the model it wraps and persists to disk / read from disk
> 7. EVERY external interaction, example: emitting WS message MUST be fired from the service classes in point 6 ONLY.
> 8. EVERY model that is emitted MUST in itself be a data model, so a websocket message MUST be a data model just like a transcript row and tool call, but in this case its transient and does not persist to disk
> 9. Emiting WS messages ALSO MUST go through the same service pattern, example in `cancel` in turn: `self.cancel_turn() ... self.mp.ws.broadcast(self.model.to_json())`

### 1.3 ESSENTIALS
> 1. Models MUST live inside a folder: `models`
> 2. Services MUST live inside a folder: `services`
> 3. Each file can only contain 1 class
> 4. Static method must be avoided
> 5. OOP principles are observed to the letter
> 6. We are building on the MVC principles
> 7. Each function must perform an atomic job; write a record to db, read from db, transform an object
> 8. ZERO code duplication
> 9. ZERO alternative paths. If I want to emit a WS, I must go via `self.mp.ws.broadcast`, If I want to get a transcript row I must go via; `self.mp.transcript_service.read(...)`, etc...

### 1.4 CRITICAL
> 1. MessageProcessor must contain a single helper entrypoint `.process()` which constructs the `MessageProcessor` class, setups up the side-effect instances, starts the recursive MP loop.
> 2. I should be able to construct a MessageProcessor object and get access to all it's related instances but not write or read a single record from db unless I specifically invoke those calls.
> 3. ORM MOST be late-binding, I call `.filter`, `.limit`, etc... on a model but don't get the results until I actually need them, example; `model.filter('id > 5').limit(3).get()` - Only the `get` makes an actual query, everything else is just mutating the data model query

### 1.5 Fork rulings (verbatim)
- **Scope:** "Abilities should not require a complete rewrite, they already output everything via ToolResult a unified path and should already be receiving the MP instance when invoked"
- **Constructor side effects:** ".process()/.begin() however `POST /api/thread` also folds into `MP.process`"
- **ORM flavor:** Active-record
- **WS emit shape:** Pass the model, Ws serializes — **SUPERSEDED by §1.9** (the WS mechanism is now the static `Websocket.broadcast` facade + `JsonSerializable`; the "pass a data-model, the facade serializes" shape is preserved — only the path changes)
- **Zero-param principle:** "We should NEVER pass a variable that could otherwise be read via the `mp.` path. Example: `BROADCASTS_STATE`, should NEVER be passed to WS service or any service for that matter, the service itself reads it directly via `self.mp.config.BROADCASTS_SERVICE`. Each service automatically retrieves whatever it needs via the `mp.` path, that is the reason the whole path exists in the first place. With this refactor I expect that the vast majority of the functions don't take any parameters at all because they always read from the `mp.` path."

### 1.6 Rule-3 boundary rulings (verbatim)
- **Config × Rule 3:** "ProcessorConfigs should be frozensets and thus do not need to be an mp instance at all. They are a side-car to MP. So other services can call `self.mp.config` but the ProcessorConfig itself should never need to reach out back to the mp"
- **Rule 3 depth:** "Only when needed. If a service needs to reach out back to the mp to reach other services / configs then yes the service should have mp in the constructor but if a service is self-contained then it doesn't need mp at all. We stop at service layer for the simple reason that services should be the last layer before termination. A model should never need to reach out back to the service or upstream, the service manages state, the model is just crud."

### 1.7 Session rulings (verbatim — the 4 forks answered after the drift audit)
- **ORM query entry (Critical 3):** the late-binding builder lives **on the model** — `model.filter('id > 5').limit(3).get()`. `db.query(Model)` is **removed**; there is ONE query path.
- **One-class-per-file (Essential 3):** "Exempt enums+exceptions only" — a cohesive enum group and an exception taxonomy may each stay in one file; **everything else splits one class per file** (including the prompt-fragment subclasses). **SUPERSEDED (Dylan, jul06):** the enum half of this exemption is revoked — every enum now gets its own file under `configs/enums/<name>.py`, no grouping. The exception-taxonomy half (`models/provider_errors.py`) is untouched.
- **Memory reads:** "The memory recall, document processing, etc… I was very clear in previous sessions that they should be tool calls. So why do they need an MP, or why are they divergent? If they are normal tool calls, they receive an MP already… something is fishy." → memory/data_graph/episodes are reached **only through tool calls (abilities, which already hold mp)**. No memory service joins the spine. Any current *direct* in-process memory read is a divergence to remove.
- **Cancel entry (Critical 1):** "The `DELETE` endpoint must now use an instance of MP and call the turn executor cancel function. The turn execution is the control surface, there is no control plane." → **no `MessageProcessor.cancel()` classmethod, no control plane.** The DELETE handler constructs an *inert* MP instance and calls `self.mp.turn_execution_service.cancel()`.
- **Directory / file casing:** the PSR `/Type/Name` shape refers to **namespacing**, not disk casing. "In terms of directory and file names we should always use lowercase." → lowercase package dirs, snake_case module files, PascalCase classes (standard PEP-8). Restores verbatim Essentials 1–2 (`models`, `services`).

### 1.8 Governance (verbatim)
- "we'll build layered yes, and we'll drive MANY subagents, but with that we risk drift that is why the spec needs to be airtight!"
- "Every agent gets the full spec to read and the exact section it is working on, what other agents are working on and where to stop."
- "The spec must include clear acceptance criteria that match verbatim the scope and ruleset I provided you."
- "I will not tolerate any divergence from the spec."
- "In regards to tests, I don't really care if they stay green or not. What I do care about is that once you're done, the architecture is verbatim as I described it, the code is clean and LEAN OOP and that a real-world test against a live chalie instance works flawlessly."

### 1.9 Websocket facade design (verbatim — SUPERSEDES the §1.5 WS-emit-shape ruling and refines Rule 9's mechanism)
> 1. Websocket is a facade class with a single static function `broadcast`
> 2. The only parameter `broadcast` takes is an instance of an object
> 3. That instance must use an interface called `JsonSerializable`
> 4. `JsonSerializable` enforces 1 function exists; `to_json`
> 5. All websockets are fire and forget (there are no reply / feedback loops)
> 6. Callers that dispatch Websockets simply do: `Websocket.broadcast(self)`
> 7. `Websocket.broadcast` simply calls the `.to_json()` function on the instance that was supplied and dispatches the message

**Final shape (verbatim — the built design):**
> 1. ALL live callers inside the Chalie codebase today only fire-and-forget websocket messages via `Websocket.broadcast({JsonSerializable})` (ONLY EXCEPTION is HomeAssistant Ability)
> 2. This MUST be a single entrypoint which scaffolds itself; The `Websocket` class manages it's only WS connection (all private methods) and simply pushes a message out to anyone listening
> 3. All connection details live within the `Websocket` class itself, the FE can listen or the message just dies on deaf ears
> 4. The only surface API in the codebase for broadcasting a message is `Websocket.broadcast({JsonSerializable})`
> 5. On `MessageProcessor` add 1 helper function; `push_websocket` this function internally simply calls `Websocket.broadcast` but does a pre-flight check; `if not config.BROADCASTS_STATE or self.config.type_value() == None` so that we remove the duplicated if statements in our code
>
> Gotchas / Attention:
> 1. For permission there are some stale comments about getting replies via WS, this is false. Permission replies come via REST API, delete the stale comments
> 2. Specifically for HomeAssistant we have a dedicated handler. That one should be self-serving specifically for HomeAssistant ability. It does not broadcast to the Chalie frontend

**Consequence (the single legal reading).** There is no `Ws` service instance and no `self.mp.ws`. `Websocket` is a **pure static facade** — `services/websocket.py::Websocket`, one `@staticmethod broadcast(instance: JsonSerializable)` — the ONE sanctioned static on the spine (Essential 4 says static must be *avoided*, not forbidden; Dylan ruled it static here deliberately, same spirit as a boot-level gateway). Its connection registry is **module-private static state**, managed by **private methods only** — `Websocket` self-scaffolds and privately owns its single connection's whole lifecycle (accept/register/drop); `broadcast` is its **only public method**, because a fire-and-forget terminal sink needs nothing from `mp` and exposes nothing but the emit surface. The `api/websocket` ASGI route hands each accepted socket into that self-scaffolding (the one non-`broadcast` touchpoint); everything past the handoff is private. `JsonSerializable` is an interface enforcing exactly one method, `to_json`, that every emitted data-model implements. Rule 7 still holds — the emitting **service** is where the `Websocket.broadcast(...)` call is written; only Rule 9's *path* changes: from `self.mp.ws.broadcast(self.model.to_json())` to `Websocket.broadcast(<the JsonSerializable data-model>)`, and the **facade — not the caller — calls `.to_json()`**. No reply/feedback loop: `broadcast` returns nothing meaningful.

**Self-scaffolding transport (final-shape 2/3/4).** `Websocket` owns its single WS connection **entirely through private methods** — it lazily scaffolds and holds the connection itself; no bootstrap, config, socket, or `mp` ever crosses its boundary. The FE either listens or the frame **dies on deaf ears** (fire-and-forget, no delivery guarantee, no buffering contract). `Websocket.broadcast(<JsonSerializable>)` is the **ONE and ONLY** broadcast surface in the entire codebase.

**On-spine gate helper (final-shape 5).** `MessageProcessor.push_websocket(frame)` is the **single on-spine emit path**. It runs the pre-flight gate — `if not self.config.BROADCASTS_STATE or self.config.type_value() is None: return` — then calls `Websocket.broadcast(frame)`. Every on-spine service emits through `self.mp.push_websocket(frame)`, **never** a raw `Websocket.broadcast(...)` wrapped in an inline `if BROADCASTS_STATE`. The duplicated gate `if`-statements scattered across services (`turn_execution_service`, `tool_call_service`, …) collapse into this one helper. This keeps on-spine WS emission **on the `self.mp.*` path** (I4) — `push_websocket` is the only on-spine caller of the static facade; the static exception (§2.2 I4) then applies **only to off-spine emitters** that have no `mp` (e.g. capability alerts), which call `Websocket.broadcast(frame)` directly and un-gated.

**HomeAssistant is the ONE exception (gotcha 2).** The HA ability has a **dedicated, self-serving handler** that talks to HomeAssistant alone. It does **NOT** broadcast to the Chalie frontend and **never touches** `Websocket`/`push_websocket`. It is the single sanctioned caller allowed to bypass the facade.

**Permission replies are REST, not WS (gotcha 1).** A `permission_request` frame is a plain fire-and-forget broadcast like any other; the user's answer returns over the **REST API**, so there is no WS reply/feedback loop for permissions. Every stale "reply comes back via WS" comment in code (`policy_manager`, permission plumbing) is **false and must be deleted**.

> **STATUS (jul05) — WS facade refactor DELIVERED & verified on the live spine.** The §1.9 design is built and demonstrated in code:
> - `services/websocket.py::Websocket` exists — pure static facade: public `@staticmethod broadcast(instance: JsonSerializable | None)` (L66) + private static lifecycle `_connect`/`_disconnect` (L53/L59) over a module-private registry (`_connections`, `_lock`). `broadcast` is the only public method. Old `services/websocket_broker.py` **deleted**; **zero** prod `WebSocketBroker`/`websocket_broker` refs remain.
> - `JsonSerializable` landed at **`contracts/json_serializable.py`** (Protocol, one method `to_json`) — matches §2.3/§4.1a. (The old §4.1 `models/json_serializable.py` placement was superseded; a pure interface belongs in `contracts/`, and every spec reference already points there.)
> - `MessageProcessor.push_websocket(instance)` (`controllers/message_processor.py:163`) is the single on-spine emit path with the exact gate — `if not self.config.BROADCASTS_STATE or self.config.type_value() is None: return` then `Websocket.broadcast(instance)` (L171–173). Gate-collapse confirmed: `grep "if.*BROADCASTS_STATE" controllers services` shows the gate on the **live spine only** in `push_websocket` (the former `act_trail.py` copy is gone — that file is now DELETED, jul05).
> - Cross-instance cancel (§2.7) calls `Websocket.broadcast(execution)` directly with the row-sourced gate — `turn_execution_service.py:112`, the one sanctioned on-spine non-`push_websocket` site.
> - Off-spine emitters (`policy_manager`, `scheduler_service`, `async_delegate_runner`, `home_capability`, `api/websocket`) call the facade directly — spec-sanctioned. HA carve-out intact: `ha_ws_handler.HaWebSocketHandler` talks to Home Assistant and never touches the Chalie facade (§1.9 gotcha 2). No stale "reply via WS" comments survive (gotcha 1).
> - **Residual: NONE.** The former dead-file broadcast copies in `services/act_trail.py` and `services/execution_tracker.py` are gone — **both files are now DELETED (jul05)**. Every `Websocket.broadcast` call on the tree is either the on-spine `push_websocket` path, the §2.7 cross-instance cancel, a sanctioned off-spine emitter, or the facade definition.

---

## 2. DERIVED ARCHITECTURE — the single legal reading of §1

### 2.1 The layering (MVC, headless backend)

```
          ┌──────────────────────────────────────────────────────────────┐
   C      │  MessageProcessor (mp)  — the one orchestrator of every LLM turn│
          │  __init__ = PURE WIRING (0 db, 0 ws, 0 side-effects)            │
          │  holds: self.config, self.db, self.<every service>             │
          │  (WS is NOT held — it is the global static Websocket facade)    │
          └──────────────────────────────────────────────────────────────┘
                 │ owns (side-car, frozen, no mp) │ owns (services)
                 ▼                                 ▼
   sidecar   ProcessorConfig  (configs/)       SERVICES  (services/)
             frozen; declarative only;         hold mp IFF they reach siblings/config
             NO mp; NO methods that read        ├─ Database  (NO mp — terminal)
             DB or call services; passive.      ├─ TranscriptService     (mp)
             read via self.mp.config            ├─ ToolCallService       (mp)
                                                ├─ TurnExecutionService  (mp)
                                                ├─ CompactionService     (mp)
                                                ├─ ProviderService       (mp)
                                                ├─ LlmLogService         (mp)
                                                ├─ PromptService         (mp)
                                                ├─ GistService           (mp)
                                                ├─ DataGraphService      (mp)
                                                └─ DispatchService       (mp)
                                                        │ CRUD via  Model.filter(...).get()  /  Model(...).save()
                                                        ▼
   M      MODELS (models/)  — pure CRUD, active-record over a bound connection.
          NEVER hold mp. NEVER call a service. NEVER reach upstream.
          Persisted row-models: Transcript, ToolCall, TurnExecution, Compaction,
                                LlmCallLog, ThreadGist, DataGraph
          Transient WS wire-models (never persist): WsMessage(base), TurnSignal, ErrorFrame
          Transient LLM-transport models (never persist): ProviderRequest, ProviderResponse
                                                        │ runs SQL on the bound connection
                                                        ▼
   term.  TRANSPORT PRIMITIVES (owned by a service; NO mp):
          sqlite3.Connection (owned by Database, bound onto the Model base at boot) ·
          thin llm_clients/* (owned by ProviderService) · raw WS socket objects (owned by the static `Websocket` facade — module-private registry, NOT an mp-held service; §1.9)
```

### 2.2 The eight hard invariants (each maps to a rule)

| # | Invariant | Source |
|---|-----------|--------|
| I1 | **One orchestrator.** Every LLM turn — POST, scheduler, subconscious, gist, skill-suggestion, disclose-to-human — enters through `MessageProcessor.process(...)`. No `object.__new__`, no fake-mp, no nested `MessageProcessor(...).run()`, no `MP.process` bypass. | Rule 1; Critical 1; Fork(scope) |
| I2 | **Inert constructor.** `MessageProcessor.__init__` wires instances/flags ONLY. It reads/writes **zero** rows and emits **zero** WS. Side-effects (input-row insert, exec-row open, turn_id allocation) live in `begin()`, invoked by `process()`. Constructing an MP for a *control* op (cancel, orphan-sweep) is legal and stays inert until the specific service call fires. | Rule 2; Critical 2 |
| I3 | **mp in the constructor of every LLM-touching *coordinating* class.** A service holds `self.mp` iff it reaches a sibling service or the config. `Database` is a self-contained terminal → **no mp**. WS is not a service at all — it is the static `Websocket` facade (no instance, no mp; §1.9). Configs are frozen side-cars → **no mp**. Models are CRUD → **no mp**. | Rule 3; Rule-3 depth ruling; Config ruling; §1.9 |
| I4 | **Single path.** Every cross-class hop starts `self.mp.{...}`. **On-spine WS stays on that path** via `self.mp.push_websocket(frame)` — the one gate helper (§1.9 final-shape 5) that runs the `BROADCASTS_STATE`/`type_value` pre-flight then calls the facade. The ONE sanctioned static exception is for **off-spine** emitters with no `mp` (e.g. capability alerts), which call the global static facade `Websocket.broadcast(frame)` directly (a stateless fire-and-forget terminal sink that needs nothing from mp; §1.9). There is exactly ONE broadcast surface (`Websocket.broadcast`), ONE on-spine emit path (`self.mp.push_websocket`), and ONE way to reach any capability (`self.mp.transcript_service.read`, one query path `Model.filter(...)`). No alternative path, no duplicated capability. **HomeAssistant's dedicated handler is the single sanctioned bypass** — it self-serves the HA ability and never touches the facade (§1.9 gotcha 2). | Rule 4; Essential 9; Essential 8; §1.9 |
| I5 | **Every persisted LLM object is a model** exposing `to_dict()` + `to_json()`, queried via a **late-binding** builder reached **on the model** (`Transcript.filter().limit().get()` — only `.get()`/`.first()`/`.count()`/`.exists()` hit the DB). | Rule 5; Critical 3; Session ruling |
| I6 | **Every model interaction goes through a service.** Services are the sole callers of the model/query layer. Parametrized SQL lives in exactly ONE layer: the active-record model/query engine (`Model`/`Query`) running on the connection `Database` owns and binds. `Database` is the ONLY place a `sqlite3` connection is **opened/configured** (WAL, `row_factory`, `busy_timeout`, transactions, lifecycle) and it issues **no domain SQL** of its own; `Model`/`Query` may import `sqlite3` solely to type the injected `Connection`/`Row` and to call `.execute()` on it — they never open or configure a connection. | Rule 6; Essential 7; Rule-3 depth; §2.6 |
| I7 | **Every external emission passes a `JsonSerializable` data-model to the WS facade.** WS frames are data-models implementing `JsonSerializable` (one method `to_json`); the owning service builds/populates the frame and, **on-spine**, emits via `self.mp.push_websocket(<frame>)` (the gate helper that pre-flights `BROADCASTS_STATE`/`type_value` then calls the facade). Off-spine emitters call `Websocket.broadcast(<frame>)` directly. The facade — not the caller — calls `.to_json()` and dispatches, fire-and-forget (no reply/feedback loop; permission replies return via REST, §1.9 gotcha 1). No raw dicts on the wire; no inline `if BROADCASTS_STATE` around a broadcast; no `WebSocketBroker()`, no `Ws` instance, no `self.mp.ws` anywhere. | Rule 7; Rule 8; Rule 9; §1.9 |
| I8 | **Zero-param.** If a value is reachable via `self.mp....`, it is NEVER passed as an argument. The vast majority of functions take **no parameters**. The only arguments anywhere on the spine are the five *non-mp-reachable* categories in §2.4. | Zero-param ruling; Essential 7 |

### 2.3 File & class discipline (Essentials 1–6) — lowercase layout, PSR namespacing

- **Casing (Dylan ruling this session).** The PSR `/Type/Name` idea is about **namespacing**, not disk casing. On disk everything is **standard PEP-8**: lowercase package directories, `snake_case` module files, `PascalCase` classes. Class ↔ file is **1:1** (Essential 3), the module file named after its single class:
  - `controllers/` — the orchestrator-controller. `controllers/message_processor.py` → `class MessageProcessor`.
  - `models/` — every model. `models/transcript.py` → `class Transcript`, `models/tool_call.py` → `class ToolCall`, `models/ws_message.py` → `class WsMessage`, `models/model.py` → `class Model`, `models/query.py` → `class Query`, …
  - `services/` — every service. `services/transcript_service.py` → `class TranscriptService`, `services/websocket.py` → `class Websocket` (static facade, §1.9), `services/database.py` → `class Database`, …
  - `configs/` — the frozen `ProcessorConfig` side-car hierarchy. `configs/processor_config.py` → `class ProcessorConfig`, `configs/eamp_config.py` → `class EAMPConfig`, … *(new type-dir — the side-car is neither Model, Service nor Controller; §10 item 2, proposed.)*
  - `contracts/` — pure interfaces (Protocol/ABC) with no state, no behavior, no deps. `contracts/json_serializable.py` → `class JsonSerializable` (one method `to_json`, §1.9). *(new type-dir — a contract is neither Model, Service, Controller nor Config; it defines a shape others implement.)*
  - This restores Dylan's verbatim Essentials 1–2 (`models`, `services`). No case collision with any existing lowercase tree (the macOS APFS caveat is moot).
  - **Out-of-scope trees keep their existing paths** (`abilities/`, `api/`, memory/telemetry/etc.) — the layout applies to the rewritten spine only.
- **One class per file — with ONE exemption (Dylan ruling). SUPERSEDED jul06:** the enum grouping half is revoked — enums now split one-per-file too, under `configs/enums/<name>.py` (no `models/provider_enums.py`, no `services/config_type.py`). Split every class into its own file, **except**: an *exception taxonomy* may still stay grouped in one file (`models/provider_errors.py`). Everything else — including each prompt-fragment subclass — is one class per file.
- **No `@staticmethod`** (convert to instance methods; on the spine everything is a class) — with ONE sanctioned exception: the WS facade `services/websocket.py::Websocket` (§1.9) — public static `broadcast` plus its private static connection-lifecycle methods — a stateless fire-and-forget terminal sink Dylan explicitly ruled static. OOP to the letter. MVC.
- **No adapters / no shims / no polyfills / no wedges.** `getattr(mp, "x", None)` duck-typing, `object.__new__(MessageProcessor)` fabrication, SQLAlchemy compat classes — all DELETED, callers fixed.

### 2.4 The zero-param standard + the only sanctioned arguments

Verbatim: *"We should NEVER pass a variable that could otherwise be read via the `mp.` path… the vast majority of the functions don't take any parameters at all because they always read from the `mp.` path."*

**Standard:** if a value is reachable via `self.mp....`, it is NEVER passed. Every coordination/state value — config flags, `turn_id`, `uid`, `channel`, thinking level, `BROADCASTS_STATE`, current transcript id, a sibling service — is mp-reachable → **forbidden** as an argument. A spine function that accepts an mp-reachable argument is a defect.

The ONLY sanctioned arguments (none is mp-reachable — each is fresh local state or external input):

1. **The emitted `JsonSerializable`** → on-spine `self.mp.push_websocket(frame)` (the gate helper), or off-spine `Websocket.broadcast(frame)` directly. The data-model whose `to_json()` the facade will dispatch (§1.9). *(The frame itself is not an mp-reachable value; the `BROADCASTS_STATE`/`type_value` gate IS mp-reachable and therefore lives inside `push_websocket`, never passed in.)*
2. **Query predicate + literals** → `Model.filter('id > ?', row_id).limit(n)`. Critical 3's own example passes `'id > 5'` and `3`; these are the query shape/values, not stored state.
3. **New-row field values** → `Model(field=value, …).save()`. The data being persisted — it IS the new state; there is nothing to read from mp.
4. **LLM tool arguments** → `ability.run(params)`. Fork scope ruling (abilities take a unified params/`ToolResult` path; not rewritten).
5. **Entrypoint seeds** → `MessageProcessor.process(config, raw_input='', metadata=None, turn_id=None)` / `MessageProcessor(config, turn_id)`. The external world's input to a turn/control-op. After construction they live on `self.mp.*` and are never passed again.

Everything else is zero-param. Any argument outside these five categories is a defect unless escalated to Dylan and added here.

### 2.5 Config as frozen side-car (Config ruling — precise consequence)

- A `ProcessorConfig` subclass is a **frozen** declarative descriptor of ONE turn shape: flags (`skip_transcript`, `suppress_history`, `skip_input_row`, `memory_seed`, `broadcast_to`, `BROADCASTS_STATE`, …), `thinking_mode`, `type`, and any **static, zero-I/O** prompt fragments.
- It holds **no mp**, exposes **no method that reads the DB, calls a service, performs I/O, templates against runtime data, or reaches back to mp**. Services pull from it via `self.mp.config.{field}`.
- **Config-strip bright-line:** a config member **moves out to a service** if it does *any* of: DB/disk/network I/O, calls a service, reads runtime state, or renders a template against runtime data. A member **may stay** only if it is a pure, zero-I/O constant field or a pure function of its own frozen fields.
- **Everything that currently makes a config reach data moves OUT into a service:** the DB-reaching bodies of `get_user_prompt`/`get_user_definition`/`get_system_prompt` → `PromptService`; the raw SQL in `UserSummaryConfig` → `PromptService`/`TranscriptService`; config-file module-funcs → the matching service; the in-file `PostTurnHook` subclasses (`PatternDecayHook`, `PatternSkillSyncHook`, …) → end-of-turn service methods. Those services read the turn-shape via `self.mp.config`.
- **Memory boundary (ruled §3.11).** *Episodic* recall stays a tool call (holds mp; unchanged). *Structured* user-context (`data_graph` system/user_specific/behavioral_pattern rows) is reached ONLY via `self.mp.data_graph_service` — never a raw read from a config/hook/prompt body. No config, service, hook, or prompt body performs a *direct* `data_graph`/episode read.
- Net: the 18 config subclasses survive as lean frozen data declarations; their behavior relocates to services.

### 2.6 Model as pure CRUD (Rule-3 depth ruling + ORM entry ruling — precise consequence)

- **Active-record, builder on the model (Critical 3 / Session ruling).**
  - Reads: `Transcript.filter('channel = ?', ch).order_by('id DESC').limit(30).get()` — `.filter()/.where()/.order_by()/.limit()/.offset()` mutate a `Query` and return it; only `.get()/.first()/.count()/.exists()` hit the DB.
  - Writes: `Transcript(channel=ch, content=txt).save()`; `row.delete()`.
  - Aggregate/multi-table feed reads that a generic builder can't express (GROUP BY / HAVING / MAX / correlated subselect — e.g. thread-feed summaries) are **named classmethods on the owning model** (`Transcript.recent_threads(...)`, `Transcript.turn_summaries(...)`) that hold their own parametrized SQL and run it on the bound connection. This stays inside I6 (SQL is model-resident).
- **`db.query(Model)` does not exist.** There is exactly one query entry: the model.
- **Connection injection.** `Database` opens the connection (WAL/txn config) and **binds it onto the `Model` base at boot** (`Model.bind(connection)`). A model instance/classmethod runs parametrized SQL on that bound connection. A model **never** imports a service, **never** holds mp, **never** calls `self.mp....`. It is data + `to_dict` + `to_json` + its own query/persistence.
- Services reach persistence exclusively via the model (`Model.filter(...)`, `Model(...).save()`) — never by importing `sqlite3` or issuing SQL themselves.

### 2.7 Cancel path (Cancel ruling — precise consequence)

- **No control plane, no `MessageProcessor.cancel()` classmethod.** The turn-execution service **is** the control surface.
- The DELETE handler (`api/threads.py`) constructs an **inert** MP for the target turn and reaches through it:
  ```python
  mp = MessageProcessor(config, turn_id)      # inert (I2): 0 db, 0 ws at construction
  mp.turn_execution_service.cancel()          # zero-param: reads turn_id/channel via self.mp
  ```
- `TurnExecutionService.cancel()` (zero-param) writes the cancel flag on the target `turn_executions` row (`channel + turn_id + ended_at IS NULL`) and emits the lifecycle frame (Rule 7/9; §1.9). **In-process** cancel emits via `self.mp.push_websocket(...)` (the `self.config` gate is correct). **Cross-instance** cancel derives the gate/type from the **turn row**, not `self.config` (§6.7) — `push_websocket`'s `self.config` gate would read the wrong turn — so it applies the row-sourced gate itself and calls `Websocket.broadcast(...)` directly. This is the one on-spine site where the gate is not `self.config`, hence not `push_websocket`.
- The turn is (almost always) running in **another** thread/instance. That live MP's loop polls `self.turn_execution_service.should_stop` (derive-from-DB) and aborts via the internal `_TurnCancelled` signal. Cross-instance state is DB-derived (§6.7) — never in-memory.

---

## 3. RIP INVENTORY — what is deleted or dissolved (verified, line-numbered)

> Deleted outright = the file/class ceases to exist. Dissolved = its responsibility moves to a §4 target; the old file is removed once callers are cut over. **Target paths are lowercase / snake_case (§2.3).**

### 3.1 Controller / core
| Class / symbol | File:line | Fate | Reason |
|---|---|---|---|
| `MessageProcessor` (current) | services/message_processor.py:99 | **Rewritten** → `controllers/message_processor.py` | ctor does side-effects; `@staticmethod process` (L326); raw SQL (L312 `UPDATE transcript SET deliberation_score`); `WebSocketBroker()` (L205, L270); `get_shared_db_service()` (L310, L661); `_channel_locks` global (L37); getattr wedges (L474–502); direct constructs `Providers()` L135, `TranscriptService` L154, `ExecutionTracker` L174, `ToolDispatcher` L446/626/771, `ActTrail` L746 |
| `MP.broadcast(...)` | services/message_processor.py:258–270 | **DELETED** | violates Rule 7/9 — WS must originate in the service owning the model change |
| `_TurnCancelled` | services/message_processor.py:92 | **Kept** as internal control-flow exception (colocate with MP) | legitimate loop-abort signal (raised on `should_stop`) |
| MP module helpers `_channel_lock` / `_sanitize_llm_args` / `_wrap_with_checkpoint` / `_format_ts` | services/message_processor.py (module scope) | **Reassigned** | `_channel_lock` (per-channel serialization) → replaced by DB-atomic turn-id allocation (§6.8), not a Python lock; `_wrap_with_checkpoint` (compaction DB read) → `CompactionService`; `_sanitize_llm_args` → instance method on `ProviderService`; `_format_ts` → instance method on the model whose `to_json` needs it (no static, no module func) |
| `PostTurnHook` (base) + subclasses | services/post_turn_hook.py:29 (+ config-file hooks) | **Dissolved** → end-of-turn service methods | Essential 4/7; hooks with DB become atomic service ops reached via `self.mp`. **Failure-isolated (§6.11):** one raising hook must not abort siblings or the turn |
| `SystemMessagePrompt` (base) + 8 subclasses (incl. `getPrompt` back-compat shim) | services/system_message_prompt.py | **Kept, de-shimmed, SPLIT** | prompt-text producers; delete the `getPrompt` alias (shim); **split to one class per file** under their module (Essential 3 / Q_B — the exemption covers only the exception taxonomy, not these); called by `PromptService`, never by configs |
| `TurnZeroFlashback` | services/turn_zero_flashback.py (DELETED) | **Collapsed → DELETED** | the whole service (gates, centroid math, curated render) is gone — the turn-0 seed is now a single real tool call in `MessageProcessor._seed_turn_zero`: `self.dispatch_service.dispatch("memory", {"action": "recall", "query": self.raw_input, "_auto": True})`. Normal dispatch persistence + turn-scoped act-trail carry it into context; `_auto` yields `caller="seed"` in `handle_recall`. **Silent-seed invariant (§6.10):** `tool_call_service._emit` already suppresses the pill when `config.memory_seed` is set |
| Deliberation gate: `DeliberationScoreService`, `DeliberationEmaService`, `UPDATE transcript SET deliberation_score` | services/message_processor.py:312 (+ deliberation modules) | **Dissolved** → `TranscriptService.deliberation_score()` / `.set_deliberation_score()` | per-turn score drives thinking level; column read/write is a transcript-service op reached via `self.mp.transcript_service`; §6.12 preserves the EMA/threshold behavior |

### 3.2 Config layer
| Class | File:line | Fate | Reason |
|---|---|---|---|
| `ProcessorConfig` (+ 18 subclasses, incl. `EAMPConfig` external_agent.py:56, `CompactionConfig` abilities/_compaction_config.py:18, `_ActionButtonConfig` api/chat.py:131) | services/processor_config.py:25 | **Kept as frozen side-cars; behavior stripped out** | Config ruling §2.5 — remove DB-reaching bodies, raw SQL (user_summary.py:213), module-funcs (dmn.py, pattern.py, user_summary.py), in-file hooks (PatternDecayHook, PatternSkillSyncHook in pattern.py); keep declarative fields; relocate to `configs/<name>_config.py` |
| `ConfigTypeEnum` | `configs/enums/config_type.py` | **Kept** as a routing enum, own file (jul06 enum ruling — no grouping exemption) | maps `type` → config; read via `self.mp.config`, never `.get_by_type` from a service body |
| `with_hidden_input` (copy.copy + object.__setattr__) | processor_config.py:181 | **Kept** (frozen-safe field derivation) | the sanctioned frozen-rewrite mechanism; no mp involved |
| `DiscoveryConfig` object.__setattr__ ×4 | (config file) | **Reviewed** | ensure post-construct field rewrites remain data-only, no mp reach |

### 3.3 Transcript persistence — TWO parallel writers collapse to ONE model + ONE service
| Class / symbol | File:line | Fate | Reason |
|---|---|---|---|
| `TranscriptService` (instance writer #1) | services/transcript.py:26 | **Dissolved** → `services/transcript_service.py::TranscriptService` (new) + `models/transcript.py::Transcript` | raw SQL writer; `self.db = get_shared_db_service()` |
| `Transcript` (static writer #2, 21 @staticmethods, **header says DEPRECATED**) | services/transcript_service.py:75 | **Dissolved** into the new model+service | Essential 4 (no static); duplicate writer; `_select` L108 raw-SQL chokepoint |
| `get_compaction` / `write_compaction` (module funcs, no class) | services/compaction_persistence.py:32,62 | **Dissolved** → `models/compaction.py::Compaction` + `services/compaction_service.py::CompactionService` | Essential 3 (needs a class); two-axis watermark preserved (§6) |

> **`TranscriptService.gc` (retention).** GC deletes below the compaction watermark and must not delete episode-cited rows. Cross-service writes route via mp: tool_call deletes → `self.mp.tool_call_service`, watermark → `self.mp.compaction_service`. The "episode-cited ids" input is memory-owned; under the memory ruling it must arrive via the tool-call/memory path, not a direct read — see §3.11 / §10 (pending evidence: whether gc runs on the per-turn spine at all or is an off-spine retention job).

### 3.4 Tool-call trail + turn execution
| Class | File:line | Fate | Reason |
|---|---|---|---|
| `ToolCallState` | services/act_trail.py:46 | **✅ Folded into** `models/tool_call.py` (state consts on the model; act_trail.py DELETED jul05) | Essential 3 |
| `ToolCall` (frozen dataclass) | services/act_trail.py:54 | **✅ Rewritten** → `models/tool_call.py::ToolCall` (active-record; act_trail.py DELETED jul05) | Rule 5; keep WS-safe `to_json` (params/result NEVER on wire — §6.2) |
| `ActTrail` | services/act_trail.py:100 | **✅ DONE (jul05) — `services/act_trail.py` DELETED**; dissolved → `services/tool_call_service.py::ToolCallService` | raw SQL + `WebSocketBroker()` (L259) + `get_shared_db_service()` default; bare-ctor read-only mode removed |
| `TurnExecutionState` (+ `STOP_REASON_PROCESS_DEATH` L50) | services/execution_tracker.py:39 | **Folded into** `models/turn_execution.py` | Essential 3 |
| `TurnExecution` (frozen dataclass) | services/execution_tracker.py:53 | **Rewritten** → `models/turn_execution.py::TurnExecution` | Rule 5; `to_json` = full-field (WS lifecycle + REST DTO) — §6.3 |
| `TurnExecutionService` | services/execution_tracker.py:141 | **Rewritten** → `services/turn_execution_service.py::TurnExecutionService` | raw SQL + `get_shared_db_service()` default; emits via module-func `_broadcast_lifecycle`; gains `cancel()` (§2.7) |
| `ExecutionTracker` | services/execution_tracker.py:280 | **Dissolved** into `TurnExecutionService` (MP holds the service, not a separate per-turn handle) | `TurnExecutionService()` direct construct (L290); `replace()`-fabricated divergent row (L341, object_new_bypass) DELETED |
| `_type_value`, `_broadcasts_state`, `_broadcast_lifecycle` (module funcs) | services/execution_tracker.py:98,103,122 | **Dissolved** into `TurnExecutionService` (in-process emits via `self.mp.push_websocket(...)`, which absorbs the `_broadcasts_state`/`_type_value` gate) | Rule 7/9; gate reads `self.mp.config` inside `push_websocket` — except cross-instance cancel, which derives gate/type from the **turn row** and calls `Websocket.broadcast(...)` directly (§6.7/§2.7) |
| `TurnExecutionDTO` | api/dto/turn_execution.py:15 | **Kept** (API boundary DTO) | matches `TurnExecution.to_json()` 1:1; handler builds from `mp.execution.to_json()` |

### 3.5 WS emission — singleton dies, broadcast becomes a static facade (§1.9)
| Class / symbol | File:line | Fate | Reason |
|---|---|---|---|
| `WebSocketBroker` (`__new__` singleton) | services/websocket_broker.py:34 | **Rewritten** → `services/websocket.py::Websocket` (pure **static facade**: one `@staticmethod broadcast(instance: JsonSerializable)`; module-private connection registry; NO mp, NO instance) | §1.9; broadcast takes a `JsonSerializable` data-model, not a raw dict; `json.dumps` becomes `instance.to_json()` called by the facade |
| `_WebSocket` (protocol) | services/websocket_broker.py:25 | **Kept** (transport primitive) | structural type for a socket that can `send(str)` |
| 12 `WebSocketBroker()` call sites | MP:205/270, act_trail:259, execution_tracker:136, scheduler:341, policy_manager:134, async_delegate_runner:119, capabilities/base:206/229, ha_ws_handler:138, api/websocket:20/70 | **All rewired by home.** **On-spine** emitters (the dissolved MP:205/270 + act_trail:259, execution_tracker:136 → services; policy_manager:134; async_delegate_runner:119 via its inert MP) → `self.mp.push_websocket(<frame>)` (gate helper, §1.9 final-shape 5). **Off-spine** emitters with no mp (capabilities/base:206/229, scheduler:341 module fn) → `Websocket.broadcast(<frame>)` directly, un-gated. **`ha_ws_handler:138` is CARVED OUT** — HomeAssistant keeps its own dedicated, self-serving handler that talks to HA alone and does **NOT** broadcast to the Chalie FE; it never calls `Websocket`/`push_websocket` (§1.9 gotcha 2). The raw socket handler at `api/websocket:20/70` registers/removes sockets via the facade's private connection management — `Websocket` self-scaffolds its single connection, so nothing injects an instance. | §1.9; Essential 9 (zero alternative paths) |
| `PolicyManager` (static `wrap`, WS emit) | services/policy_manager.py:134 (+ dispatcher:158) | **Rewired + de-static** | on the tool-action path (holds mp): `permission_request` WS emit → `self.mp.push_websocket(...)`; static `wrap` → instance method invoked from `self.mp.dispatch_service`; policy *defaults* are config, enforcement is spine. **Permission replies arrive via REST, not WS** (§1.9 gotcha 1) — the emit is plain fire-and-forget; **delete the stale "reply via WS" comments** here and in the permission plumbing. |
| `HeartbeatService` (module-global), `TelemetryCollector` (singleton) | heartbeat_service.py:15/76, telemetry_service.py:65 | **Out of scope** (off the LLM spine) | not LLM prompt/response/action; leave as-is |

> **Frames collapse (audit fix).** Rule 9 + §1.9: the emitter passes a `JsonSerializable` data-model and the facade calls `.to_json()`. So a persisted model **is** its own wire frame — there are NO separate `ToolCallFrame` / `TurnExecutionFrame` classes. The emitting service sets the transient envelope fields (`type`, `turn_id`) on the model instance before handing it to `self.mp.push_websocket(model)` (on-spine) / `Websocket.broadcast(model)` (off-spine); `to_json()` includes them (§6.2/§6.3). Only genuinely-transient emissions with no persisted row stay as their own `WsMessage` subclasses (each a `JsonSerializable`): `TurnSignal`, `ErrorFrame`. Every emitted model/frame implements `JsonSerializable` (§4.1).

### 3.6 Provider send + logging
| Class / symbol | File:line | Fate | Reason |
|---|---|---|---|
| `Providers` (facade, NO mp by design) | services/providers.py:36 | **✅ DONE (jul05) — `services/providers.py` DELETED**; rewritten → `services/provider_service.py::ProviderService` (holds mp, L45) | Rule 3; getattr duck-typing on dto/response removed; owns thin clients (transport, no mp); emits `TurnSignal(provider_retry)` via `self.mp.push_websocket(...)`; owns per-turn provider **selection reads** (`get_selected_provider`, model/token-limit lookups) as a single path — **admin write CRUD stays off-spine** |
| `ProviderApiRequest` / `ProviderApiResponse` (mutable dataclasses) | services/provider_api.py:30/78 | **Rewritten** → `models/provider_request.py::ProviderRequest` + `models/provider_response.py::ProviderResponse` (transient models, `to_dict`/`to_json`) | Rule 8 |
| `ProviderType`, `ThinkingLevel` (enums) | provider_api.py:8/15 | **Kept, SPLIT** → `configs/enums/provider_type.py`, `configs/enums/thinking_level.py` | routing/thinking enums — one-class-per-file, no grouping exemption (jul06 enum ruling, §2.3) |
| all `Provider*Error` exceptions | provider_api.py:108–180 | **Kept, grouped** → `models/provider_errors.py` | error taxonomy — exception-taxonomy exemption (§2.3); `ProviderRetriesExhaustedError` stays MP-raised (§6.4) |
| `resolve_thinking_mode` (module func) | providers.py:31 | **✅ Folded into** `ProviderService` (instance method; providers.py DELETED jul05) | pure precedence fn; no static/module func |
| `ProviderDbService`, `ProviderCacheService`, `provider_token_limits` funcs | provider_db_service.py:21, provider_cache_service.py:9, provider_token_limits.py | **Admin CRUD OUT of scope; per-turn selection reads move to `ProviderService`** | leaf admin (vault/settings) is off-spine (§3.10); the per-turn *read* of the selected provider/model/limit is spine and single-pathed through `ProviderService` |
| `llm_call_log_service` funcs | llm_call_log_service.py | **Dissolved** → `models/llm_call_log.py::LlmCallLog` + `services/llm_log_service.py::LlmLogService` | Rule 5/6 |

### 3.7 DB gateway + SQLAlchemy shims

> **STATUS (jul06) — DB-gateway consolidation ✅ COMPLETE; one document-path breakage remains.** `services/database.py::Database.conn()` (L61) + `Database.transaction()` (L117, reentrant `BEGIN IMMEDIATE`) carry **all** DB access. `services/database_service.py` (the `DatabaseService` singleton + SQLAlchemy shims `SessionProxy`/`ResultProxy`/`DictCursor`/`_TextClause`/`text`) is **DELETED**; `get_shared_db_service` is gone.
> - **`get_shared_db_service` / `DatabaseService` — 0 prod references** (verified `grep`): even the 13-site auth/vault carve-out (`user_auth.py`, `auth_session_service.py`, `wrapper_auth_service.py`, `vault_service.py`) migrated to `Database.conn()`/`transaction()`. The 2 former tail sites (`api/mcp_settings.py`, `mcp_server/server.py:80/93`) use `Database.transaction()`. No SQLAlchemy import survives in prod.
> - **`tests/conftest.py`** points the `Database` gateway at each test's file (`Database.close()`/`Database.conn()`), no `get_shared_db_service` patch.
> - **⚠ REMAINING — live breakage (NOT DB-gateway, but surfaced here):** 5 document-path sites import a **now-deleted symbol** `get_data_graph_service` and call `dgs.db.connection()` on a `DataGraphService` that has no `.db` (it holds only `self.mp`): `abilities/document.py:340/341`, `services/document_service.py:452/453/489/490`, `api/documents.py:346/395`. Verified `ImportError: cannot import name 'get_data_graph_service'`. These fire on document-fragment read + document delete. **Fix:** route via the on-spine `DataGraph` model (`DataGraph.filter("source=? AND active=1", …)`) through an inert MP's `self.mp.data_graph_service`, not a global factory.
>
> **Net remaining for this section: the 5 document-path refs (a DataGraph-model migration, own slice). DB-gateway itself is done.**

| Class / symbol | File:line | Fate | Reason |
|---|---|---|---|
| `DatabaseService` | services/database_service.py:196 | **Rewritten** → `services/database.py::Database` (NO mp, terminal; opens/owns the `sqlite3.Connection`, binds it onto `Model` at boot, owns txn/WAL; ONLY `sqlite3` importer) | Rule 3 depth; I6 |
| `SessionProxy`, `ResultProxy`, `DictCursor`, `_TextClause`, `text` | database_service.py:60/108/146/24/38 | **DELETED** (No-Adapters) | pure SQLAlchemy compat shims; `SettingsService` (their only real consumer) rewritten to typed gateway calls |
| `get_shared_db_service` (global singleton) | database_service.py:49 | **DELETED** | on-spine reaches `self.mp.db`; the composition root owns the one `Database` instance and hands an explicit handle to any **off-spine** consumer that used the singleton (list/document/settings services) — no global |
| `WriteQueueService`, `_QueueItem`, `get_write_queue` | write_queue_service.py:44/22/248 | **Out of scope for the spine** (bypassed by transcript/tool_call/turn_execution today; used only by list/document services) | not on the LLM write path; leave until a later pass |
| `DurableTimestamp` | durable_timestamp.py:17 | **Out of scope** (subconscious/discovery timestamps; off-spine) but note asymmetric raw-SQL read (L92) | not LLM prompt/response/action |
| `SettingsService` | settings_service.py:12 | **Rewritten** to drop `text()`/`get_session()` shim; typed calls against the app-owned `Database` handle | needed to let the shim cluster die |

### 3.8 Abilities framework (bodies OUT of scope per Fork ruling; the wedge dies)
| Class / symbol | File:line | Fate | Reason |
|---|---|---|---|
| `ToolDispatcher` | abilities/_dispatcher.py:51 | **Rewritten** → `services/dispatch_service.py::DispatchService` (typed `self.mp`, NO getattr) | 9 getattr wedges (L71,100,166,195,215,216,274,305,422); direct `ActTrail(...)` ctor (L70) → `self.mp.tool_call_service`; policy enforcement (L158) via `self.mp` (§3.5); keep `dispatch/_bind/_execute/_render` behavior |
| `Ability` (+ base framework: `CapabilityAbility`, `AbilityRegistry`, `ToolResult`, `ToolParamError`, `Keys`, `KeyNormalizer`, `KeyHealer`, `BudgetCappedAbility`, `_MCPAbility`, `SearchableAbility`, `ReviewWindowAbility`) | abilities/*.py | **Kept** (Fork scope ruling) — re-point only the spine-touching leaks | `BudgetCappedAbility` bare `ActTrail()` (_budget.py:31) → `self.mp.tool_call_service`; `Ability._inject_framework_fields` getattr (L243) → typed `self.mp.config` |
| `AbilityRegistry.build_tools(mp)` getattr(mp,'active_tools') | abilities/_registry.py:110 | **Rewired** to typed `self.mp.config`/`self.mp` | wedge removal; static-class registry may stay (bodies out of scope) but reads mp typed |
| `_delegate.render_trail(mp)` duck-type, `_pattern_provenance(proc)` getattr | abilities/_delegate.py:47, _pattern_provenance.py:24 | **Rewired** to typed mp | wedge removal |

### 3.9 API routes + satellite processors + scheduler
| Class / symbol | File:line | Fate | Reason |
|---|---|---|---|
| `ThreadItemResource` (`get/post/delete`) | api/threads.py:373 | **Rewritten (thin)** | post() L416 `MessageProcessor(config, turn_id, text, {...})` + L420 `mp.run()` → `MessageProcessor.process(config, text, metadata, turn_id)`; delete() → inert `MessageProcessor(config, turn_id)` + `mp.turn_execution_service.cancel()` (§2.7); read via `mp.execution.to_json()` |
| `ThreadsResource`, `ThreadsBatchResource` + module-func projection layer (`serialize_turn` L259, `_rows_to_messages` L219, `_thread_summaries` L299, `_fetch_tool_calls_for_transcripts` L76, `_resolve_turn_transcript_ids` L108, `_fetch_attachments_for_transcripts` L135) | api/threads.py | **Rewritten** to read via `self.mp.transcript_service` / `self.mp.tool_call_service` (through an inert MP); aggregate feed reads use the model classmethods (§2.6) | raw SQL in API (L88,119,127,143) must die (I6); one read path only |
| `_ActionCtx` (duck-typed fake mp) + `ActionResource` | api/chat.py:161/181 | **Rewritten** | `ToolDispatcher(_ActionCtx())` (L205) → real inert `MessageProcessor` + `self.mp.dispatch_service`; kill the fake-mp class |
| `AsyncDelegateRunner` (+ `_Delegate`) | services/async_delegate_runner.py:71/50 | **Rewritten** | reaches `ToolDispatcher._render/_run` **privates** (L145) → `self.mp.dispatch_service`; `WebSocketBroker()` (L119) → `self.mp.push_websocket(...)` (its dedicated inert MP carries the gate); module-global singleton (L178) → owned instance. **Lifetime wrinkle (§6.13):** the runner outlives the turn — it holds a dedicated **inert** MP (its own `self.mp` with live `dispatch_service`) for the background span, NOT the finished turn's MP; WS stays reachable via the global static facade |
| `ThreadGistService` | services/thread_gist_service.py:21 | **Dissolved** → `models/thread_gist.py::ThreadGist` + `services/gist_service.py::GistService` (mp) | raw SQL + module-global singleton (L61); concrete service (no "or TranscriptService sibling" fork) |
| `SubconsciousWorker` | services/subconscious_worker.py:99 | **Rewired (entry only)** | 7× `MessageProcessor.process(...)` calls become the ONE entry (I1); internal `data_graph` SELECTs are memory reads — under the memory ruling they belong to the tool-call/memory subsystem, off-spine; kill any object.__new__/fake-mp reliance |
| `scheduler_service` (module funcs) | services/scheduler_service.py | **Rewired (entry + WS)** | `MessageProcessor.process` (L391) is the one entry; `WebSocketBroker()` (L341) → `Websocket.broadcast(...)`; raw `scheduled_items` SQL is off-spine scheduling data (leave), but the LLM-fire and WS go through the spine |
| `thread_gist_message_processor` (module funcs) | services/thread_gist_message_processor.py:27 | **Rewritten** | `object.__new__(MessageProcessor)` fabrication (L49–54) DELETED → `MessageProcessor.process(ThreadGistConfig(), ...)` |
| `skill_suggestion_message_processor` (module funcs) | services/skill_suggestion_message_processor.py:22 | **Rewritten** | `object.__new__(MessageProcessor)` fabrication (L58–66) DELETED → `MessageProcessor.process(SkillSuggestionConfig(), ...)` |
| `DiscloseToHumanHook` nested MP | services/external_agent.py:48 | **Rewritten** | `MessageProcessor(UserConfig(...)).run()` (banned nested construct+run) → `MessageProcessor.process(...)` single entry (I1) |

### 3.10 Explicit scope boundary (what this rewrite does NOT touch)
Off the per-turn LLM spine, therefore **out of scope** (leave exactly as-is; do not rewire):
- Provider/vault/settings **admin CRUD** (`ProviderDbService`, `ProviderCacheService`, vault, `SettingsService`'s domain) — except deleting the SA shims they force and handing them an explicit `Database` handle in place of the deleted singleton.
- Episodic/memory recall (a tool), `DurableTimestamp`, `WriteQueueService`, telemetry/heartbeat. **Exception (ruled §3.11):** the `data_graph` system/user_specific/behavioral_pattern rows consumed at prompt-assembly / written post-turn ARE in scope, as `models/data_graph.py::DataGraph` + `services/data_graph_service.py::DataGraphService`. All OTHER (off-spine) direct `data_graph` readers/writers stay untouched here and are migrated to `DataGraphService` under the follow-up ticket (§10 item 4).
- Ability tool **bodies** and their private search DBs (`abilities.sqlite`, index DBs) — Fork scope ruling.
- `SchemaConvergenceService` (declarative DDL — legitimately raw SQL by purpose).
- All `api/dto/*` DTOs (kept — API boundary shapes).

If work forces a change beyond this boundary, STOP and escalate.

### 3.11 Memory-read divergences (Session ruling — **evidence in; ONE fork open for Dylan**)

Episodic recall / document processing ARE tool calls (abilities that hold mp). But the researcher's `file:line` sweep found the "fishy" reads are **not** episodic recall and are **not** tool calls — they are direct `data_graph` I/O (`get_data_graph_service()` / raw `conn.execute` SQL) wired into **config prompt-assembly bodies** (fire *before* the LLM turn) and **post-turn hooks** (fire *after* it). Neither can be a tool call — there is no in-flight LLM to dispatch one. Confirmed sites:

| Cluster | Direction | Sites (file:line) |
|---|---|---|
| **A. Prompt-assembly reads `user_summary`** (system-kind data_graph) injected into the system / user-definition prompt of the **main user turn** | READ, pre-LLM | `configs/channels/user.py:76-90` (`UserConfig.get_user_definition`) · `configs/channels/external_agent.py:113-124` (`EAMPConfig.get_system_prompt`) |
| **B. Prompt-assembly reads traits + behavioral_pattern** rows injected into summary/pattern-channel prompts | READ, pre-LLM | `configs/channels/user_summary.py:186-224` (data_graph fetch + raw SQL) · `configs/channels/pattern.py:16-49` (raw SQL `SELECT … data_graph WHERE kind='behavioral_pattern'`) |
| **C. Post-turn hooks WRITE/UPDATE data_graph** (parse the response, persist) | WRITE, post-LLM | `configs/channels/user_summary.py:31-99` (`PersistUserSummaryHook.store`) · `configs/channels/user_summary.py:110-152` (`_should_synthesise` raw SQL) · `configs/channels/pattern.py:72-124` (`PatternDecayHook` raw `UPDATE`) · `configs/channels/pattern.py:127-154` (`PatternSkillSyncHook` raw SQL) |
| **D. Transcript GC** episode-cited + compaction-watermark reads | READ | `services/transcript_service.py:472-483, 567-588` — **NOT per-turn**: runs on `SubconsciousWorker`'s background tick (`DecayEngineService.run_decay_cycle`), fully decoupled from `MessageProcessor`. → **out of scope** (§3.10). ✅ clean |

**RULED (Dylan, 2026-07-04): `DataGraph` model + `DataGraphService` join the spine.** Clusters A/B/C are structured user-context (a summary + behavioral patterns) the prompt builder needs at assembly time — not episodic recall — and there is no tool-call path at prompt-build / post-turn time. So `data_graph`'s system / user_specific / behavioral_pattern rows become:
- `models/data_graph.py::DataGraph` (active-record row-model, §4.1) — the ONLY place data_graph SQL lives (I6).
- `services/data_graph_service.py::DataGraphService` (holds mp, §4.2) — the ONLY gateway to it.
- **Cluster A/B reads** → `PromptService` reads user-context via `self.mp.data_graph_service` (`user_summary`, traits, behavioral_pattern). No config body reads `data_graph`.
- **Cluster C writes** → the post-turn services that replaced `PersistUserSummaryHook`/`PatternDecayHook`/`PatternSkillSyncHook` write via `self.mp.data_graph_service`.
- **Episodic recall stays a tool** (unchanged); **Cluster D (GC/decay)** stays off-spine.
This widens the §3.10 boundary by exactly this one store. **Every OTHER (off-spine) code path that still reads/writes `data_graph` directly** — via raw SQL or the global `get_data_graph_service()` — must also route through `DataGraphService`; that mechanical migration is captured in a **follow-up ticket** (see §10 item 4), not this rewrite.

---

## 4. MOVE-IN INVENTORY — the new tree (exact files, classes, signatures)

> **Legend:** `mp?` = does the class hold `self.mp`? `emits?` = does it build+broadcast a WS model? Every method is atomic (Essential 7) and zero-param unless it uses one of the five sanctioned arg categories (§2.4).

### 4.1 `models/` — pure CRUD, active-record, NO mp, NO service reach

| File | Class | mp? | Responsibility & key members |
|---|---|---|---|
| `models/model.py` | `Model` | no | base row-model: field storage; `to_dict()`; `to_json()`; `save()`; `delete()`; classmethods `filter(pred, *vals)`/`where(...)`/`order_by(...)`/`all()` → `Query`; classmethod `hydrate(row)`; classmethod `bind(connection)` (Database calls once at boot); runs SQL on the bound connection. Active-record; no mp; no service. |
| `models/query.py` | `Query` | no | late-binding builder bound to the model+connection: `filter(...)`/`where(...)`/`order_by(...)`/`limit(n)`/`offset(n)` → `self` (no I/O); `get()`/`first()`/`count()`/`exists()` execute + hydrate. Critical 3. |
| `models/transcript.py` | `Transcript` | no | one `transcript` row; encapsulates FORK/MAIN query scopes (§6.1); named aggregate classmethods for the thread-feed (`recent_threads`, `turn_summaries`, …) holding their own SQL (§2.6). |
| `models/tool_call.py` | `ToolCall` | no | one `tool_calls` row; state consts (`STARTED/DONE/ERROR`); **WS-safe** `to_json` = type/turn_id/id/tool_name/summary/created_at/ended_at/state (params/result never emitted — §6.2). |
| `models/turn_execution.py` | `TurnExecution` | no | one `turn_executions` row; state consts (`WORKING/COMPLETED/CANCELLED/CRASHED`); `to_json` = full fields + `type` (WS lifecycle AND REST — §6.3). |
| `models/compaction.py` | `Compaction` | no | one `compactions` row; two-axis watermark (`for_turn_id` NULL=main / int=fork). |
| `models/llm_call_log.py` | `LlmCallLog` | no | one `llm_call_log` row (token/latency telemetry). |
| `models/thread_gist.py` | `ThreadGist` | no | one `thread_gist` row (channel,turn_id PK). |
| `models/data_graph.py` | `DataGraph` | no | one `data_graph` row (kind,key,value,confidence,active,last_confirmed_at,…); ONLY home of data_graph SQL (I6); named scopes/classmethods for `kind='system'` (user_summary), `kind='user_specific'` (traits), `kind='behavioral_pattern'`. Preserves the exact `fetch`/store semantics of today's `data_graph_service`. |
| `models/ws_message.py` | `WsMessage` | no | transient WS wire-model base implementing `JsonSerializable` (from `contracts/`, §4.1a): `to_dict()`, `to_json()`; NEVER persists (no `save`). Rule 8. |
| `models/turn_signal.py` | `TurnSignal` | no | transient `updated` / `provider_retry` lean frame (no persisted counterpart). |
| `models/error_frame.py` | `ErrorFrame` | no | transient error-toast frame (no persisted counterpart). |
| `models/provider_request.py` | `ProviderRequest` | no | transient LLM send payload (system/messages/tools/thinking/…); `resolve_max_tokens`. |
| `models/provider_response.py` | `ProviderResponse` | no | transient LLM reply (text/model/tokens/tool_calls/stop_reason/…). |
| `configs/enums/provider_type.py`, `configs/enums/thinking_level.py` | `ProviderType`, `ThinkingLevel` | no | one enum per file (jul06 enum ruling supersedes the grouped-enum exemption, §2.3); not `models/` — enums live under `configs/enums/`. |
| `models/provider_errors.py` | `Provider*Error` taxonomy | no | grouped exception taxonomy (exception exemption, §2.3); `ProviderRetriesExhaustedError` MP-raised (§6.4). |

> `ToolResult` + `ToolParamError` stay in `abilities/_result.py` (Fork scope ruling — abilities not rewritten). Noted exception to the `models/` rule.
> **No `ToolCallFrame` / `TurnExecutionFrame`** — the persisted model's `to_json()` IS the wire frame (§3.5). Every model emitted over WS (`ToolCall`, `TurnExecution`, and the `WsMessage`/`TurnSignal`/`ErrorFrame` transients) implements the `JsonSerializable` contract (`contracts/`, §4.1a; §1.9) so the static `Websocket.broadcast` can call `.to_json()` on it.

### 4.1a `contracts/` — pure interfaces (no state, no behavior, no deps)

| File | Class | mp? | Responsibility |
|---|---|---|---|
| `contracts/json_serializable.py` | `JsonSerializable` | no | **interface** (Protocol/ABC) enforcing exactly ONE method: `to_json() -> str` (§1.9). Every emitted data-model (`ToolCall`, `TurnExecution`, `WsMessage`/`TurnSignal`/`ErrorFrame`) implements it; `Websocket.broadcast` accepts only instances of it. Zero-dep foundational contract — not a model (no CRUD, no row), not a service (no mp, no behavior). |

### 4.2 `services/` — hold mp iff they reach siblings/config

| File | Class | mp? | emits? | Responsibility (atomic methods) |
|---|---|---|---|---|
| `services/database.py` | `Database` | **no** | no | terminal WAL gateway; opens/owns thread-local `sqlite3.Connection`; `bind`s it onto `Model` at boot; transaction/commit; **ONLY** `sqlite3` importer; issues no domain SQL. Reached `self.mp.db`. |
| `services/websocket.py` | `Websocket` | **no** (static facade — §1.9) | (is the emitter) | pure static facade — one **public** `@staticmethod broadcast(instance: JsonSerializable)` → `instance.to_json()` fan-out; self-scaffolds and **privately** owns its single WS connection (module-private registry, private lifecycle methods — accept/register/drop). `broadcast` is the ONLY public method. **ONLY** WS emitter, fire-and-forget (message dies on deaf ears if nothing listens). Reached GLOBALLY as `Websocket.broadcast(...)` — **NOT** `self.mp.ws` (no instance, not mp-held). On-spine callers reach it through `self.mp.push_websocket` (gate helper); off-spine callers call it directly. |
| `services/transcript_service.py` | `TranscriptService` | **yes** | yes | `read`/`append_input`/`append_assistant`/`turn_rows`/`link_doc`/`settle`/`unsettle`/`gc`/`deliberation_score`/`set_deliberation_score`; CRUD via `Transcript.*`; emits `TurnSignal(updated)` via `self.mp.push_websocket(...)`. |
| `services/tool_call_service.py` | `ToolCallService` | **yes** | yes | `start`/`finish`/`record`/`by_turn`/`by_transcript`; CRUD via `ToolCall.*`; on start/record **un-settles** the transcript via `self.mp.transcript_service` (§6.9); sets envelope + emits `ToolCall` via `self.mp.push_websocket(...)`; memory-seed emit suppression (§6.10) is realized through the config-driven `push_websocket` gate (memory-seed config yields no broadcast), not a duplicated inline check; type from `self.mp.config`. |
| `services/turn_execution_service.py` | `TurnExecutionService` | **yes** | yes | `open`/`should_stop`/`cancel`/`finish`/`sweep_orphaned`; CRUD via `TurnExecution.*`; sets envelope + emits `TurnExecution` via `self.mp.push_websocket(...)` (in-process turn; gate reads `self.mp.config`) or, for cross-instance cancel, via `Websocket.broadcast(...)` with the gate/type derived from the **turn row** (§6.7/§2.7). (absorbs ExecutionTracker; no fabricated row §6.6) |
| `services/compaction_service.py` | `CompactionService` | **yes** | no | `get`/`write`/`watermark` for a view via `Compaction.*`; two-axis preserved; owns the checkpoint read (`_wrap_with_checkpoint`). |
| `services/provider_service.py` | `ProviderService` | **yes** | yes | `send(ProviderRequest)->ProviderResponse`; `measure`; `context_limit`; `resolve_thinking_mode`; `sanitize_args`; per-turn provider **selection reads**; owns thin `llm_clients/*` (transport, no mp); logs via `self.mp.llm_log_service`; emits `TurnSignal(provider_retry)` via `self.mp.push_websocket(...)`. Retry stays in MP (§6.4). |
| `services/llm_log_service.py` | `LlmLogService` | **yes** | no | `record` one `LlmCallLog` via the model; token aggregates. |
| `services/prompt_service.py` | `PromptService` | **yes** | no | assembles system / user / definition prompts from `self.mp.config` fragments + history via `self.mp.transcript_service` + structured user-context (`user_summary`/traits/patterns) via `self.mp.data_graph_service`; drives the `SystemMessagePrompt` producers. **No direct DB/`data_graph` read of its own; no episodic recall** (that's a tool). (§2.5, §3.11) |
| `services/gist_service.py` | `GistService` | **yes** | no | `upsert`/`bulk_get` thread gists via `ThreadGist.*`. |
| `services/data_graph_service.py` | `DataGraphService` | **yes** | no | the ONE gateway to `data_graph` structured user-context: `user_summary()`/`traits()`/`patterns()`/`store()`/`decay()` via `DataGraph.*`. Read by `PromptService` (prompt-assembly context) and written by post-turn services — all via `self.mp.data_graph_service`. Episodic recall is NOT here (stays a tool). Replaces the global `get_data_graph_service()` singleton; off-spine callers get an app-level handle (follow-up ticket). |
| `services/dispatch_service.py` | `DispatchService` | **yes** | (via tool_call_service) | the tool chokepoint (replaces ToolDispatcher): `dispatch`/`_bind`/`_execute`/`_render`; typed `self.mp`, no getattr; enforces policy (§3.5); persists+emits through `self.mp.tool_call_service`. |

### 4.3 `controllers/` — the orchestrator

| File | Class | Responsibility |
|---|---|---|
| `controllers/message_processor.py` | `MessageProcessor` | Rule 1/2, Critical 1/2. `__init__(config, turn_id=-1, raw_input='', metadata=None)` = **pure wiring** (holds `self.config`; constructs `self.db` and every service, passing `self`; WS is NOT wired — it is the global static `Websocket` facade (§1.9); **0 db, 0 ws, 0 side-effects**). `@classmethod process(config, raw_input='', metadata=None, turn_id=None)` = the ONE run entrypoint → construct(inert) → `begin()` (side-effects: input row via `self.transcript_service`, exec open via `self.turn_execution_service`, turn_id alloc §6.8) → recursive `_step()` loop (reads `self.turn_execution_service.should_stop` directly — no delegating property) → `end()` → returns text/`self.execution`. **No `cancel` classmethod** (cancel = inert construct + `self.turn_execution_service.cancel()`, driven by the DELETE handler, §2.7). Exposes the ONE WS helper `push_websocket(frame)` → `if not self.config.BROADCASTS_STATE or self.config.type_value() is None: return` then `Websocket.broadcast(frame)` (§1.9 final-shape 5) — the single on-spine emit path, absorbing every service's duplicated broadcast-gate `if`. No `broadcast` of its own, no raw SQL, no getattr, no `object.__new__`. |

**MessageProcessor public attribute contract** (owned by F1; D/E agents code against this exact surface — do not invent members):

| Attribute | Kind | Set by / read via |
|---|---|---|
| `config` | frozen `ProcessorConfig` | ctor arg → `self.mp.config` |
| `db` | terminal service | ctor wiring → `self.mp.db` (WS is NOT held — it is the global static `Websocket` facade, §1.9) |
| `transcript_service` / `tool_call_service` / `turn_execution_service` / `compaction_service` / `provider_service` / `llm_log_service` / `prompt_service` / `gist_service` / `dispatch_service` | coordinating services | ctor wiring → `self.mp.<service>` |
| `uid`, `channel` | turn identity | derived in ctor from config/seed → `self.mp.uid` / `.channel` |
| `turn_id` | int | ctor arg / allocated in `begin()` → `self.mp.turn_id` |
| `current_transcript_id` | int | set as rows are written → `self.mp.current_transcript_id` |
| `execution` | `TurnExecution` | opened in `begin()` → `self.mp.execution` |
| `thinking_level` | enum | derived per step (deliberation §6.12) → `self.mp.thinking_level` |
| `active_tools` | tuple | from `self.mp.config` → `self.mp.active_tools` |

### 4.3b `configs/` — the frozen side-car hierarchy (behavior stripped — §2.5)

| File | Class | mp? | Responsibility |
|---|---|---|---|
| `configs/processor_config.py` | `ProcessorConfig` | **no** | frozen ABC base: declarative turn-shape fields only (flags, `thinking_mode`, `type`, static zero-I/O prompt fragments). No DB, no service, no I/O, no mp. |
| `configs/<name>_config.py` (18 subclasses incl. `eamp_config.py`, `compaction_config.py`, `thread_gist_config.py`, `skill_suggestion_config.py`, `user_summary_config.py`, …) | one per file | **no** | one frozen turn-shape each; every DB-reaching / I/O / templating body relocated to a `services/` method (§2.5). |

### 4.4 What stays put (kept, re-pointed only)
- Config subclasses (frozen side-cars, behavior stripped — §2.5; relocated to `configs/`).
- `SystemMessagePrompt` family (prompt-text producers, de-shimmed, **split one class per file**), called by `PromptService`.
- Ability base + tool bodies + `ToolResult`/`ToolParamError`/`KeyHealer`/registry (Fork scope), re-pointed where they touched the spine.
- Provider enums (one-per-file under `configs/enums/`, jul06 ruling) + error taxonomy (grouped, §3.6); thin `llm_clients/*` (transport).
- `TurnExecutionDTO` + all `api/dto/*`.

---

## 5. OLD → NEW COLLAPSE MAP (one-line fate per current spine class)

```
services/message_processor.py::MessageProcessor      → controllers/message_processor.py::MessageProcessor (rewrite; inert ctor; classmethod process; NO cancel classmethod)
  MP.broadcast()                                     → DELETED (WS is the static Websocket.broadcast facade, §1.9)
  MP deliberation UPDATE (L312)                      → services/transcript_service.py::set_deliberation_score
services/processor_config.py::ProcessorConfig(+18)   → configs/*.py frozen side-cars (behavior → services/)
services/post_turn_hook.py::PostTurnHook             → end-of-turn services/ methods (failure-isolated §6.11)
services/system_message_prompt.py::SystemMessagePrompt(+8) → one class per file; called by services/prompt_service.py::PromptService
services/transcript.py::TranscriptService            ┐
services/transcript_service.py::Transcript(21 static)├→ models/transcript.py::Transcript + services/transcript_service.py::TranscriptService
services/compaction_persistence.py::{get,write}_compaction → models/compaction.py::Compaction + services/compaction_service.py::CompactionService
services/act_trail.py::{ToolCallState,ToolCall}      → models/tool_call.py::ToolCall
services/act_trail.py::ActTrail                       → services/tool_call_service.py::ToolCallService
services/execution_tracker.py::{TurnExecutionState,TurnExecution} → models/turn_execution.py::TurnExecution
services/execution_tracker.py::{TurnExecutionService,ExecutionTracker,_broadcast_lifecycle,_broadcasts_state,_type_value} → services/turn_execution_service.py::TurnExecutionService (+ cancel §2.7)
services/websocket_broker.py::WebSocketBroker        → services/websocket.py::Websocket (static facade; Websocket.broadcast(JsonSerializable); no mp, no instance, §1.9)
  ToolCallFrame / TurnExecutionFrame                 → DELETED (model.to_json() IS the frame §3.5)
services/providers.py::Providers                      → services/provider_service.py::ProviderService
services/provider_api.py::{ProviderApiRequest,ProviderApiResponse} → models/provider_request.py + models/provider_response.py
services/provider_api.py::{ProviderType,ThinkingLevel} → configs/enums/provider_type.py + configs/enums/thinking_level.py (jul06: split, no models/provider_enums.py)
services/provider_api.py::Provider*Error             → models/provider_errors.py
services/llm_call_log_service.py::{log_call,...}     → models/llm_call_log.py::LlmCallLog + services/llm_log_service.py::LlmLogService
services/database_service.py::DatabaseService         → services/database.py::Database (binds Model at boot; no db.query)
services/database_service.py::{SessionProxy,ResultProxy,DictCursor,_TextClause,text,get_shared_db_service} → DELETED
services/thread_gist_service.py::ThreadGistService   → models/thread_gist.py::ThreadGist + services/gist_service.py::GistService
abilities/_dispatcher.py::ToolDispatcher             → services/dispatch_service.py::DispatchService (typed mp)
services/policy_manager.py::PolicyManager            → rewired: instance method + self.mp.push_websocket (permission_request emit; replies via REST §1.9 gotcha 1); enforced via dispatch_service
services/async_delegate_runner.py::AsyncDelegateRunner → rewritten (dedicated inert MP; self.mp.dispatch_service + self.mp.push_websocket)
services/thread_gist_message_processor.py (object.__new__ MP) → MessageProcessor.process(ThreadGistConfig(), ...)
services/skill_suggestion_message_processor.py (object.__new__ MP) → MessageProcessor.process(SkillSuggestionConfig(), ...)
services/external_agent.py::DiscloseToHumanHook (nested MP.run) → MessageProcessor.process(...)
api/threads.py::ThreadItemResource                   → thin: process (post) / inert MP + turn_execution_service.cancel (delete)
api/chat.py::{_ActionCtx,ActionResource}             → real inert MP + self.mp.dispatch_service
```

---

## 6. CORRECTNESS INVARIANTS THAT MUST SURVIVE (break one = silent product corruption)

1. **FORK/MAIN read-floor.** Preserve the exact SQL semantics now living in string literals: settle predicate (`_SETTLE_PREDICATE`, transcript_service.py:56), `_TURN_KEY = COALESCE(turn_id, -id)` (L34), and the two-axis compaction watermark (`for_turn_id` NULL = MAIN keyed on `turn_id`; int = FORK keyed on `transcript.id`). These move into `Transcript`/`Compaction` model query scopes / classmethods but must produce identical row sets.
2. **Tool-call WS privacy.** `ToolCall.to_json()` (the emitted frame) exposes **only** type/turn_id/id/tool_name/summary/created_at/ended_at/state. `params` and `result` NEVER cross the wire. The service sets `type`/`turn_id` transiently before broadcast.
3. **TurnExecution dual shape.** `TurnExecution.to_json()` is full-field (+ constant `type`) on purpose — it is BOTH the WS lifecycle frame and the REST DTO (`TurnExecutionDTO` validates it 1:1).
4. **Retry ownership.** Provider comms layer performs **no** retry and emits **no** WS beyond `provider_retry` signal. `RequestOverCapError`/`ResponseOverLimitError` trigger the MP's compact-then-retry; `ProviderRetriesExhaustedError` is raised by the **MP**, not the provider layer. `PROVIDER_CALL_TIMEOUT_S=300` stays at the transport boundary.
5. **State vocabularies unchanged.** `tool_calls.state ∈ {started,done,error}`; `turn_executions.state ∈ {working,completed,cancelled,crashed}`; DB `CHECK` constraints must still pass.
6. **No fabricated rows.** The `ExecutionTracker._finish` `replace()`-fabricated in-memory row (execution_tracker.py:341) is DELETED — a state read always reflects the DB (derive-from-DB).
7. **Cross-instance cancel.** `TurnExecutionService.cancel()` writes `cancel_requested` and emits the lifecycle frame for a turn running in **another** thread; the gate/type on that frame derives from the **target turn's row**, NOT the canceller's `self.mp.config`. The running MP polls `self.turn_execution_service.should_stop`. Preserve today's `request_cancel_by_turn` targeting (`channel + turn_id + ended_at IS NULL`). A cancelled first-fire must not brick its schedule (verify §9).
8. **Turn-id allocation is atomic.** Fresh-turn allocation happens exactly once per turn, in `begin()`, via the `Transcript` model, with a single canonical formula `COALESCE(?, (SELECT COALESCE(MAX(turn_id),0)+1 FROM transcript WHERE channel = ?))`. The deleted `_channel_locks` Python serialization is replaced by DB-level atomicity (single-writer `begin()` + `UNIQUE(channel, turn_id)` / atomic `INSERT … RETURNING`), not an in-memory lock. No two writers disagree on an empty channel.
9. **Tool-call un-settles the transcript.** When `ToolCallService.start`/`record` opens a tool call, the owning transcript's `settled` flips to 0 — a **cross-table** write routed via `self.mp.transcript_service.unsettle()`, never a raw SQL touch from the tool-call service.
10. **Memory-seed is silent.** The turn-zero flashback / memory-seed `record()` writes the tool_call row WITHOUT emitting a live pill (`self.mp.config.memory_seed` gates emission). No spurious turn-zero pill on the wire.
11. **Hook failure isolation.** One raising end-of-turn hook/service-method must not abort its siblings or the turn; each runs guarded, failures logged, the rest proceed.
12. **Deliberation gate.** The per-turn deliberation score (EMA + threshold) that selects thinking level is preserved: computed and persisted via `self.mp.transcript_service` (`deliberation_score`/`set_deliberation_score`), driving `self.mp.thinking_level`. Identical scoring behavior to today.
13. **Background-runner MP lifetime.** `AsyncDelegateRunner` outlives the turn; it holds a **dedicated inert MP** (live `dispatch_service`) for its background span, never a reference to the finished turn's MP. WS emits via `self.mp.push_websocket` on its inert MP (the facade underneath is a global static, no per-MP instance, §1.9). Derive-from-DB for any state it needs.

---

## 7. BUILD ORDER + PER-AGENT WORK PACKETS

Layered **bottom-up** (each layer compiles against the one below; drift is contained because every agent's stop-line is a lower layer it may READ but not MODIFY).

### Phase A — Foundation (no dependencies)
- **A1 · Database + ORM engine.** OWN: `services/database.py::Database`, `models/model.py::Model`, `models/query.py::Query`, `contracts/json_serializable.py::JsonSerializable` (zero-dep interface: one method `to_json`, §1.9 — models in Phase B implement it). Deliver active-record + late-binding **on the model** (`Model.filter().get()`, `Model(...).save()` — Critical 3 / §2.6), boot connection-bind (`Model.bind(conn)`), classmethod query entry (NO `db.query`). STOP-LINE: do not write any concrete row-model or service. Siblings: none.

### Phase B — Models (depend on A only; READ A, don't modify)
- **B1 · Persisted row-models.** OWN: `models/transcript.py`, `models/tool_call.py`, `models/turn_execution.py`, `models/compaction.py`, `models/llm_call_log.py`, `models/thread_gist.py`, `models/data_graph.py`. Encode FORK/MAIN scopes + feed classmethods (§6.1/§2.6), WS-safe `to_json` (§6.2), full-field `to_json` (§6.3), state consts (§6.5), and `DataGraph`'s kind-scoped read/store classmethods matching today's `data_graph_service` semantics. STOP-LINE: no services, no WS, no mp.
- **B2 · Transient models.** OWN: `models/ws_message.py`, `models/turn_signal.py`, `models/error_frame.py`, `models/provider_request.py`, `models/provider_response.py`, `models/provider_errors.py`. Provider enums are NOT here — one-per-file under `configs/enums/` (jul06 ruling), owned by the config layer, not B2. NO `ToolCallFrame`/`TurnExecutionFrame` (§3.5). STOP-LINE: no persistence (`no save`), no mp.

### Phase C — Terminal services (depend on A/B)
- **C1 · Websocket.** OWN: `services/websocket.py::Websocket` — pure static facade (§1.9): one `@staticmethod broadcast(instance: JsonSerializable)` → `instance.to_json()` fan-out over a module-private connection registry (`connect`/`disconnect` static), fire-and-forget, no mp, no instance. STOP-LINE: don't touch any coordinating service or MP.

### Phase D — Coordinating services (hold mp; depend on A/B/C; READ them)
Each agent owns ONE file, reads models+db+ws, routes every hop `self.mp.*`, emits its own frame where applicable:
- **D1 · TranscriptService** (emits `TurnSignal`; owns settle/unsettle §6.9, deliberation §6.12).
- **D2 · ToolCallService** (emits `ToolCall` frame; un-settles via mp §6.9; silent memory-seed §6.10).
- **D3 · TurnExecutionService** (absorbs ExecutionTracker; emits `TurnExecution` frame; `cancel` §2.7; kills fabricated row §6.6; cross-instance source-of-truth §6.7).
- **D4 · CompactionService** (two-axis §6.1; checkpoint read).
- **D5 · ProviderService** (owns thin clients; per-turn selection reads; emits `provider_retry`; NO retry §6.4).
- **D6 · LlmLogService.**
- **D7 · DispatchService** (de-wedged ToolDispatcher; policy enforcement §3.5; persists/emits via `self.mp.tool_call_service`).
- **D8 · PromptService** (assembles prompts from `self.mp.config` + `self.mp.transcript_service`; drives `SystemMessagePrompt`; NO direct memory read §3.11).
- **D9 · GistService.**
- **D10 · DataGraphService** (the ONE gateway to `data_graph` structured user-context; read by `PromptService`, written by post-turn services; NO episodic recall; preserves today's `data_graph_service` fetch/store semantics via `DataGraph.*`).
STOP-LINE for all of D: do NOT modify models, Database, or Ws; do NOT construct siblings directly — reach them via `self.mp.{sibling}` (MP wires them). Config is READ via `self.mp.config`, never mutated.

### Phase E — Config strip (depends on nothing structurally; parallel with D)
- **E1 · Config side-cars.** OWN: `configs/processor_config.py` + 18 subclasses (one class per `configs/<name>_config.py`). Strip DB-reaching bodies, raw SQL, module-funcs, in-file hooks, I/O, templating (bright-line §2.5); relocate that behavior into the matching Phase-D service (coordinate the hand-off list with D-agents). Leave frozen declarative fields. STOP-LINE: configs end with zero mp reach, zero DB, zero I/O, zero service calls.

### Phase F — Controller (depends on A–E)
- **F1 · MessageProcessor.** OWN: `controllers/message_processor.py`. Inert ctor (I2), single `process` classmethod (Critical 1), recursive `_step` loop (reads `should_stop` directly), retry policy (§6.4), begin/end side-effects, turn-id allocation (§6.8), the `push_websocket(frame)` gate helper (§1.9 final-shape 5 — the ONE on-spine emit path, absorbing services' broadcast-gate `if`s), the public attribute contract (§4.3). Wire every service with `self`. STOP-LINE: no raw SQL, no WS dict, no getattr, no `cancel` classmethod, no inline `if BROADCASTS_STATE` outside `push_websocket` — everything via `self.<service>`.

### Phase G — Cut-over + delete (depends on all)
- **G1 · API/threads + chat** → thin: post()→`MessageProcessor.process`; delete()→inert MP + `turn_execution_service.cancel()`; kill `_ActionCtx`; API reads via services + feed classmethods.
- **G2 · Satellites** (gist, skill-suggestion, scheduler, subconscious, async_delegate_runner, disclose-to-human) → single `MessageProcessor.process` entry; kill `object.__new__` + nested-MP-run; WS via `self.mp.push_websocket` (on-spine) or static `Websocket.broadcast` (off-spine, no mp) (§1.9); async-runner dedicated inert MP (§6.13). **HomeAssistant handler stays carved out** — self-serving, does not broadcast to the Chalie FE (§1.9 gotcha 2). **Delete stale "permission reply via WS" comments** (replies are REST, §1.9 gotcha 1).
- **G3 · Boot recovery** → app startup constructs an inert MP and calls `self.turn_execution_service.sweep_orphaned()` once (crash recovery has no live turn). Wire it into the existing boot sequence (not a per-turn path).
- **G4 · Delete** old files (§3) once no importer remains; delete SA shims + singletons + `get_shared_db_service` (hand off-spine consumers an explicit `Database` handle §3.7).

> **RESOLVED (Dylan, this session):** `MessageProcessor` lives in `controllers/message_processor.py`; lowercase dirs + snake_case files + PascalCase classes spine-wide (§2.3). ORM entry is `Model.filter()` (no `db.query`). Cancel has no classmethod/control-plane (§2.7). Enums+exceptions are the only one-class-per-file exemption (§2.3). **SUPERSEDED (Dylan, jul06):** enums lose the exemption too — every enum is now one-class-per-file under `configs/enums/<name>.py` (`ProviderType`, `ThinkingLevel`, `ConfigTypeEnum`, `PolicyChannel`). Only the exception taxonomy (`models/provider_errors.py`) is still grouped. `PolicyChannel` was also pulled out of its former home nested inside `ProcessorConfig` into its own top-level enum file, repointed everywhere it was referenced as `ProcessorConfig.PolicyChannel`.

---

## 8. ACCEPTANCE CRITERIA (mirrors §1 word-for-word)

A layer is DONE only when **every** box below is true and demonstrable. Greps scan the spine trees `backend/controllers backend/services backend/models backend/configs backend/api` (+ `backend/abilities` for wedge leaks).

**Orchestration & construction**
- [ ] Every LLM turn (POST, scheduler, subconscious, gist, skill, disclose-to-human) enters through `MessageProcessor.process(...)`. `grep -rn "object.__new__(MessageProcessor)\|MessageProcessor(" backend/` returns only `process` internals and the two sanctioned inert-instance control ops (DELETE-cancel, boot orphan-sweep). No `.run()`, no nested MP. *(Rule 1; Critical 1)*
- [ ] `MessageProcessor.__init__` performs zero DB and zero WS: constructing an MP and inspecting it touches no row and emits nothing. *(Rule 2; Critical 2)*
- [ ] `MessageProcessor` exposes exactly one run entrypoint `.process()`; cancel is NOT a classmethod — it is an inert instance + `self.turn_execution_service.cancel()` (no control plane). *(Critical 1; Cancel ruling §2.7)*

**mp path & single-path**
- [ ] Every LLM-touching coordinating class holds `self.mp`; `Database`/configs/models hold none; WS is the static `Websocket` facade (no instance, no mp — §1.9). *(Rule 3; Rule-3 depth; Config ruling; §1.9)*
- [ ] Every cross-class hop is `self.mp.{...}` (on-spine WS stays on-path via `self.mp.push_websocket`; sole static exception is off-spine `Websocket.broadcast`, §1.9/I4). `grep -rn "get_shared_db_service\|WebSocketBroker()\|self\.mp\.ws\|services/ws\.py\|getattr(.*mp" backend/controllers backend/services backend/models backend/configs backend/api backend/abilities` returns nothing on-spine. *(Rule 4; Essential 9; §1.9)*
- [ ] Exactly ONE way to reach each capability (one WS emitter, one transcript reader, one query entry `Model.filter()`, no `db.query`). No duplicated capability, no alternative path. *(Essential 8, 9; ORM ruling)*

**Models & ORM**
- [ ] Every persisted LLM object (transcript, tool_call, turn_execution, compaction, llm_call_log, thread_gist) is a `Model` with `to_dict()` + `to_json()`. *(Rule 5)*
- [ ] Queries are late-binding **on the model**: `Model.filter()/.limit()/.order_by()` mutate only; `.get()/.first()/.count()/.exists()` are the sole DB-hitting terminals; writes are `Model(...).save()`. `grep -rn "db.query(" backend/` returns nothing. *(Critical 3; ORM ruling)*
- [ ] A model never imports a service, never holds mp, never reaches upstream. *(Rule-3 depth ruling)*

**Services, emission & data-models**
- [ ] Every model interaction goes through a service; parametrized SQL exists ONLY in the `Model`/`Query` active-record engine (+ named model classmethods) running on the bound connection. `grep -rn "\.execute(" backend/services backend/controllers backend/configs` returns nothing, and only `services/database.py` OPENS/CONFIGURES a `sqlite3` connection (WAL/row_factory/lifecycle); `models/` may import `sqlite3` for typing + call `.execute` on the injected connection. *(Rule 6; Essential 7; I6)*
- [ ] Every WS emission is a service building/populating a `JsonSerializable` data-model and emitting it: **on-spine** via `self.mp.push_websocket(<frame>)`, **off-spine** (no mp) via `Websocket.broadcast(<frame>)`. `push_websocket` is the ONLY on-spine caller of the facade and the ONLY place the `BROADCASTS_STATE`/`type_value` gate lives — `grep -rn "if.*BROADCASTS_STATE" backend/services backend/controllers` shows the gate in `message_processor.py::push_websocket` only (cross-instance cancel excepted, §2.7). `Websocket.broadcast` is the one broadcast surface; the facade calls `.to_json()` and dispatches, fire-and-forget. No raw dict crosses the wire; no separate frame classes; no `Ws` instance / `self.mp.ws`. **HomeAssistant's handler is the sole bypass** (self-serving, never calls the facade, §1.9 gotcha 2). **No "reply via WS" comment survives** in permission plumbing — replies are REST (§1.9 gotcha 1). *(Rule 7, 8, 9; §1.9; §3.5)*
- [ ] Every emitted object is a data-model that does not persist beyond its row (`WsMessage` subclasses persist nothing; persisted models emit their `to_json()`). *(Rule 8)*

**Zero-param & discipline**
- [ ] Every spine function argument falls in one of the five sanctioned categories (§2.4); no argument carries an mp-reachable value. Spot-check: no service method takes `config`, `turn_id`, `uid`, `BROADCASTS_STATE`, or a sibling service as a parameter. *(Zero-param ruling)*
- [ ] Lowercase layout: models in `models/`, services in `services/`, controller in `controllers/`, configs in `configs/`, contracts in `contracts/`; one class per file (`grep -rlnE "^class " backend/models backend/services backend/controllers backend/configs backend/contracts` shows ≤1 `class` per file, except the one grouped exception-taxonomy file `models/provider_errors.py` — enums are no longer grouped-exempt (jul06 ruling): `grep -rlnE "^class " backend/configs/enums` shows ≤1 `class` per file too); PascalCase classes / snake_case files; `grep -rn "@staticmethod" backend/{models,services,controllers,configs,contracts}` returns nothing **except** `services/websocket.py` (the one sanctioned facade — public static `Websocket.broadcast` + its private connection-lifecycle statics, §1.9); OOP + MVC. *(Essentials 1–6 + rulings §2.3, §1.9)*
- [ ] Each function is atomic (one job: read row / write row / transform / emit) — reviewer checklist per method. *(Essential 7)*
- [ ] No shim/adapter/polyfill/wedge remains: SA compat classes, `object.__new__` fabrication, `getattr(mp,...)`, `db.query`, `ToolCallFrame`/`TurnExecutionFrame`, `get_shared_db_service` all gone; callers fixed by deletion/rewrite. *(Opening directive; FORBIDDEN ANTI-PATTERNS)*

**Correctness (from §6)**
- [ ] FORK/MAIN row sets, tool-call WS privacy, retry ownership, state vocabularies, atomic turn-id allocation, cross-table un-settle, silent memory-seed, hook failure-isolation, deliberation gate, cross-instance cancel source-of-truth all behave identically to / as specified vs pre-rewrite.

**Memory boundary (from §3.11)**
- [ ] No config, service, hook, or prompt body reads/writes `data_graph` directly; all structured user-context flows through `self.mp.data_graph_service` (episodic recall stays a tool). `grep -rn "get_data_graph_service\|FROM data_graph\|data_graph WHERE" backend/controllers backend/services backend/configs backend/api` returns nothing except inside `services/data_graph_service.py` / `models/data_graph.py`. *(Memory ruling)*

**Real-world (from §9)**
- [ ] Live chalie instance: a real user turn, a tool call, a cancel, a scheduled fire, a gist, and a skill-suggestion all work flawlessly end-to-end.

---

## 9. REAL-WORLD TEST (the bar Dylan actually cares about)

Against a **live chalie instance** (deploy the branch per project deploy flow, then exercise the real UI/WS — no mocks):
1. **User turn:** send a message on a channel → assistant streams a real answer; transcript rows + `turn_executions` row persisted; `updated`/lifecycle WS frames arrive.
2. **Tool call:** trigger a tool → live tool-pill shows started→done; `params`/`result` absent from the WS payload; `tool_calls` row correct; owning transcript un-settled then re-settled.
3. **Memory recall as a tool:** ask something that triggers memory recall → it runs as a normal tool call (receives mp, returns `ToolResult`); no direct memory read anywhere in the turn.
4. **Cancel:** DELETE the in-flight turn → an inert MP's `turn_execution_service.cancel()` sets `cancel_requested`, the running loop stops, `cancelled` lifecycle frame emitted, no bricked schedule.
5. **Scheduled / subconscious fire:** a scheduled prompt fires through `MessageProcessor.process` → normal turn behavior, correct WS.
6. **Gist + skill-suggestion:** the thread-gist and skill-suggestion satellites fire through `MessageProcessor.process(<config>)` (no `object.__new__`) → produce their rows/WS correctly.
7. **Boot recovery:** restart mid-turn → boot `sweep_orphaned` marks the orphaned execution `crashed`; no zombie WORKING row.
8. **Regression sweep:** thread feed (feed classmethods), batch read, action button, async delegate all function through the single service paths.

Unit tests may be red (Dylan: "I don't really care if they stay green"); the live run is the gate.

---

## 10. OPEN ITEMS REQUIRING DYLAN (before/if they arise)
1. ~~**Controller folder placement / casing.**~~ **RESOLVED:** `controllers/message_processor.py`; lowercase dirs + snake_case files + PascalCase classes (§2.3).
2. ~~**Config side-car type-dir.**~~ **PROCEEDING:** frozen `ProcessorConfig` hierarchy lives in `configs/` (`configs/processor_config.py`, …). Speak up if you want it elsewhere.
3. ~~**`Message` vs `MessageProcessor` name.**~~ **PROCEEDING:** class/file stays `MessageProcessor` / `controllers/message_processor.py` (Rule 1 wording). Say the word to shorten to `Message`.
4. ~~**Memory-read divergences (§3.11).**~~ **RULED (Dylan, 2026-07-04):** `DataGraph` model + `DataGraphService` join the spine (option a). Clusters A/B reads → `PromptService` via `self.mp.data_graph_service`; cluster C writes → post-turn services via same; episodic recall stays a tool; GC off-spine. **Follow-up ticket** (Chalie project) tracks migrating every OTHER off-spine direct `data_graph` reader/writer onto `DataGraphService` — being scoped from an exhaustive `data_graph`-access audit.
5. Any newly-discovered place where a rule is physically impossible or two readings exist — STOP and ask; never decide unilaterally.
```
