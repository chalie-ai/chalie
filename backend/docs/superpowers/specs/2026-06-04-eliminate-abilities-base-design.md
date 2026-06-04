# Design: Eliminate `abilities/_base.py`

**Date:** 2026-06-04
**Branch:** rc-0.9.0
**Status:** Design — awaiting review
**Author:** Chalie (with Dylan)

---

## 1. Problem

`abilities/_base.py` (633 lines) is a single `Ability` ABC that has absorbed **seven distinct
responsibilities**, plus a circular-import / test-patchability hack. It is the canonical SRP/OOP
violation in the abilities layer: the class that is supposed to *describe and run one tool* also
orchestrates dispatch, persists the act-trail to SQL, manages a daemon-thread delegate registry,
loads client telemetry, broadcasts WebSocket events, proxies MCP calls, and hard-codes a channel
routing rule. A pile of module-level procedural functions (`_run_ability`, `_normalise_run_result`,
`_emit`, `_dispatch_mcp`, `_supports_async_delivery`, `_run_async_delegate`, `_load_tool_telemetry`,
`_populate_module_aliases`) sits beside it.

**Goal:** dissolve `_base.py` entirely into single-purpose, well-named modules with a clear
inheritance hierarchy, abstract methods, value objects, and zero free-floating procedural functions.

### Responsibility inventory (evidence)

| Responsibility | Current symbols (`abilities/_base.py`) | Lines |
|---|---|---|
| Tool abstraction | `Ability` ABC: ClassVars, `__init_subclass__`, `run` (abstract), `get_description`, `get_input_schema`, `enrich_rich_payload` | ~234–285, 477–582 |
| Dispatch orchestration | `use`, `match`, `_bind`, `execute`, `_run_ability`, `_normalise_run_result` | 86–141, 289–365, 504–550 |
| Trail persistence | `record`, `fetch_by_transcript_id`, `render` (raw SQL on `tool_calls`) | 370–451 |
| Async-delegate lifecycle | `_active_delegates`, `_run_async_delegate`, `cancel_delegate`, `get_active_delegates`, `_supports_async_delivery` | 36–46, 177–218, 455–473 |
| Result routing | async-delivery decision in `execute`; the `post_turn` callable-field on `ProcessorConfig` | 525–544 |
| Client context | `_load_tool_telemetry` | 48–83 |
| WS emission | `_emit` | 162–175 |
| MCP proxy | `_dispatch_mcp`, `_MCPAbility` | 143–160, 585–601 |
| Import/patch hack | `AbilityRegistry/PolicyManager/WebSocketBroker = None`, `_populate_module_aliases`, `if X is None` guards | 21–37, 299–300, 334–335, 354–355, 169–170, 614–633 |

---

## 2. Goals / Non-goals

**Goals**
- `abilities/_base.py` no longer exists after this work.
- `Ability` becomes a purely declarative ABC + `run()` + schema hooks.
- Every other responsibility becomes its own class with one owner.
- Delete the import/patchability hack (`_populate_module_aliases` + module aliases).
- Behaviour-preserving: every existing unit + nightly test passes unchanged in semantics
  (the two intentional behavioural deltas are called out explicitly in §4.0 and §4.4).
- **Async becomes a per-call model decision, not a per-ability trait.** The `ASYNC_CAPABLE`
  ClassVar is deleted; the base `get_input_schema` injects a framework `async` boolean into every
  tool **on channels whose config sets `SUPPORTS_ASYNC`** (only `UserConfig` today, §4.8d); the
  dispatcher pops it; when true the call runs in the background. The async-vs-sync split governs only
  *when* a result arrives, never *where* — routing is `mp`-driven (§4.0).
- **`mp` is the only context handle.** Tool dispatch always receives the originating
  `MessageProcessor`. `_supports_async_delivery`'s hard-coded `channel == "user"` rule and the
  `channel`-string plumbing are deleted; delivery is config-driven re-entry (§4.0, §4.4).
- `post_turn` callable-field → a set of independent `PostTurnHook` objects on `ProcessorConfig`
  (order-independent, failure-isolated, future-parallelisable) — these are also the cross-channel
  router (§4.8).
- `_load_tool_telemetry` → `ClientContext` value object.
- **Delete confirmed-dead code** (evidence §3): `POLICY_CATEGORY` / `POLICY_LABELS` (the ABC
  ClassVars + their definitions on 28 abilities) and `AbilityRegistry.policy_visible()`.

**Non-goals (explicitly out of scope)**
- The `mcp_server.talk_to_chalie` return-value contract stays a **function return**, not a push
  (per "additive router" decision). `process()` still returns the primary text.
- No collapse of the trail SQL into other repositories this pass (flagged as the candidate for a
  *separate* net-negative follow-up, §10).
- No change to `abilities/_delegate.py` (delegate-tools-as-abilities; unrelated).

---

## 3. Key evidence (anchors)

- `_registry.py` globs **every non-underscore `.py` in `abilities/`** as a tool module
  ([`_registry.py:48`](../../../abilities/_registry.py)) → new infra files **must** be
  underscore-prefixed.
- `_registry.py:6` `from abilities._base import Ability` is the **only** reason the import hack
  exists. Move `Ability` to its own light module → cycle dies.
- `Ability.use` callers (6 prod, 2 test): `message_processor.py:652,662,709`; `memory.py:348,349`;
  `api/chat.py:448`; tests `test_dispatch_alias_binding_regression.py:46,58`.
- `Ability.record` callers: `message_processor.py:550,948,997`. `fetch_by_transcript_id`:
  `message_processor.py:894,909`. `render`: `message_processor.py:910`.
- `Ability.cancel_delegate`: `api/chat.py:370`. `get_active_delegates`: `api/chat.py:349`.
- `_emit` external caller: `message_processor.py:996,1006` (act narration).
- `self.telemetry` consumers (8): `weather.py:105`, `contacts.py:113`, `calendar.py:142`,
  `email.py:176`, `home.py:113`, `ubiquiti.py:164`, `place.py:89`, `news.py:94`.
- **Async today:** `ASYNC_CAPABLE = True` is set on **exactly two** abilities —
  `web_search.py:106` and `web_browse.py:101` (the delegate tools). The other 34 default to
  `False`. The gate at `_base.py:525` is `self.ASYNC_CAPABLE and _supports_async_delivery(channel)`;
  `_supports_async_delivery` returns `True` only for `channel == "user"` (`_base.py:39–45`).
- **Delivery hardwires the user channel:** `_run_async_delegate` calls
  `dispatch_message(result_text, channel=channel, hidden_input=True)` (`_base.py:209`), and
  `dispatch_message` → `_start_turn` **always builds `UserConfig`** (`api/chat.py:249`); its
  `channel` param is vestigial ("currently always `user`", `api/chat.py:203`). Config-driven
  re-entry (§4.4) is what lets an async result return to the processor that spawned it.
- **Dead code (zero readers/callers):** `POLICY_CATEGORY` / `POLICY_LABELS` are defined on 28
  abilities and the ABC, with **no readers** anywhere (`utils/`, `api/`, `services/`,
  `build_ability_db`, frontend `api/policies.py` — none serialize them).
  `AbilityRegistry.policy_visible()` has **no call sites** (only comment references).
- `ProcessorConfig.post_turn` field at `processor_config.py:121`; 5 non-None impls
  (UserConfig, EAMPConfig, PatternConfig, GeoConfig, UserSummaryConfig).
- 36 ability subclasses `import Ability` from `abilities._base`.

---

## 4. Target module map

`abilities/_base.py` → deleted, replaced by the modules below.

### 4.0 Two orthogonal levers — `async` (blocking) and hooks (routing)

These are different concerns; the spec keeps them strictly separate. Conflating them was the
original sin of `_base.py`.

**Lever 1 — `async`: does this tool call block the current ACT iteration? Nothing else.** The
tool's internal `run()` is *identical* either way, and the ACT loop *always* waits for a result to
come back from a dispatch. `async` is "non-blocking" only because the dispatcher **short-circuits
the return** with a placeholder result (`"<NAME> dispatched (id: …)"`) the instant it hands the real
work to a background thread. The loop receives that placeholder and proceeds. `async` is NOT a
routing concept and never touches a channel. The `async` option is only *present* on loops whose
config sets `SUPPORTS_ASYNC` (§4.8d) — only a push channel with a durable session can honour a
deferred result; elsewhere the schema omits it and every tool is synchronous.

**Lever 2 — routing/delivery: always via the originating `mp`.** `mp = MessageProcessor(config)`
starts the loop and *is* the root context; `mp.config` (frozen) carries the channel identity *and*
the declarative post-turn hooks. Dispatch always passes `mp` — to the ability
(`ability.MessageProcessor = mp`) and, on the async path, to the runner (`spawn(ability, params,
mp)`). A background thread **captures the `mp` object itself**, so it can send responses and
`WS.emit` on the channel it spawned from **even after the ACT loop has completed and closed** — `mp`
simply lives on in the thread's memory.

On completion (of a turn *or* a backgrounded tool) delivery is, every time:
- **Hot path** → emit/respond on the channel `mp` spawned from (`mp.config` channel /
  `broadcast_to`). Always the origin channel.
- **Hooks** → run `mp.config.post_turn_hooks`. A hook MAY spawn a *secondary* `MessageProcessor`
  for a *different* channel — this is how multi-channel broadcast works (e.g. `ExternalAgent`:
  hot-path emit back to the external channel **and** a hook spawning a `UserConfig` processor for
  the human-in-the-loop disclosure).

Same-channel surfacing reuses the captured `mp`; cross-channel surfacing is a hook that spawns a new
processor for the target config. The two levers compose freely: a tool may be sync or async (lever
1) and a config may broadcast to one or many channels (lever 2), independently. No bare `channel`
string crosses a boundary again; `_supports_async_delivery(channel)` and the `channel=` parameters
die.

### 4.1 `abilities/_ability.py` — the `Ability` ABC (one job)

Keeps only what *describes and runs a tool*:

```python
class Ability(ABC):
    NAME: ClassVar[str]
    SUMMARY: ClassVar[str]
    EXAMPLES: ClassVar[list[str]]
    INPUT_SCHEMA: ClassVar[dict]
    SEARCH_TOOLTIP: ClassVar[str] = ""

    MessageProcessor: object | None = None   # per-call binding (the originating mp)
    telemetry: dict | None = None            # per-call binding (ClientContext.as_dict() or None)

    def __init_subclass__(cls, **kw): ...     # unchanged validation
    @abstractmethod
    def run(self, params: dict) -> dict | str: ...
    def get_description(self) -> str: ...
    def get_input_schema(self, mp=None) -> dict:
        """Base returns INPUT_SCHEMA with the framework `async` property injected.
        Overrides that enrich the schema MUST start from super().get_input_schema(mp)."""
    @classmethod
    def enrich_rich_payload(cls, payload, row) -> dict: ...
```

- `ASYNC_CAPABLE`, `POLICY_CATEGORY`, `POLICY_LABELS` are **removed** (per-call async + dead code).
- **`async` injection:** the base `get_input_schema` deep-copies `INPUT_SCHEMA` and adds a
  framework property:
  ```python
  "async": {"type": "boolean", "default": False,
            "description": "Run in the background instead of blocking this step. You get an "
                           "immediate acknowledgement and the result is delivered on this channel "
                           "when it completes. Use for long-running calls."}
  ```
  This is the *only* place async is declared, and it is **conditional**: inject `iff
  mp is not None and mp.config.SUPPORTS_ASYNC` (§4.8d). On channels without the capability the
  property is omitted entirely, so the model cannot pick async and every tool is de-facto
  synchronous. When `mp` is unknown (None), omit it — synchronous is the safe default. The handful
  of subclasses that override `get_input_schema` (e.g. `find_tools`) must be audited to start from
  `super().get_input_schema(mp)` so the gate applies uniformly (§5).
- `async` is a JSON-schema property key and a dict key only — read as `params.pop("async", False)`,
  never bound as a Python identifier, so the reserved-word collision never arises.

No `use`/`execute`/`match`/`_bind`/`record`/threading/telemetry-loading. 36 subclasses change one
import line: `from abilities._base import Ability` → `from abilities._ability import Ability`.

### 4.2 `abilities/_dispatcher.py` — `ToolDispatcher`, bound to the invoking `mp`

The single chokepoint, replacing `Ability.use(self, …)`:

```python
class ToolDispatcher:
    def __init__(self, mp: object) -> None:
        self._mp = mp

    def dispatch(self, tool_name: str, params: dict) -> str:
        """match → bind → gate(PolicyManager.wrap) → execute → record → return str."""

    # internals (were Ability statics / module fns):
    def _match(self, tool_name) -> Ability | None        # native registry | _MCPAbility | None
    def _bind(self, tool_name) -> Ability | None          # fresh per-call instance, .MessageProcessor=self._mp
    def _execute(self, ability, params, act_summary) -> str   # emit → async-decision → run → emit
    @staticmethod
    def _run(ability, params) -> dict                     # was _run_ability (+ VaultLockedError)
    @staticmethod
    def _normalise(raw) -> dict                            # was _normalise_run_result
```

- **Async decision (per-call):** `_execute` pops the framework `async` flag exactly as it pops
  `act_summary` (framework keys, never passed to `run()`):
  ```python
  run_async = bool(params.pop("async", False))
  ...
  if run_async:
      result_text = async_delegate_runner.spawn(ability, params, self._mp)   # always mp
  else:
      result = self._run(ability, params)
      result_text = str(result.get("result", ""))
  ```
  No `ASYNC_CAPABLE`, no `supports_async_delivery`, no `channel`. The model's flag is the sole gate;
  the originating `mp` is the only context the runner needs.
- Loads client context: `ability.telemetry = ctx.as_dict() if (ctx := ClientContext.current()) else None`
  immediately before `_run` (dict form — zero consumer change, see §4.5).
- Records via `ActTrail` (§4.3), not a static on `Ability`.
- Imports `AbilityRegistry`, `PolicyManager`, `WebSocketBroker` **normally** (it is not imported by
  any of them) — no alias hack.

**Migration:** `Ability.use(mp, name, params)` → `ToolDispatcher(mp).dispatch(name, params)` at all
8 call sites.

### 4.3 `services/act_trail.py` — `ActTrail` repository

`record` / `fetch_by_transcript_id` / `render` are raw SQL over `tool_calls` — a repository, not an
Ability concern.

```python
class ActTrail:
    def record(self, *, tool_name, params, result, transcript_id, ephemeral=True) -> None
    def fetch_by_transcript_id(self, transcript_id: int) -> list[dict]
    @staticmethod
    def render(row: dict) -> str          # "[tool_name] params → result"
```

Constructed with the shared DB service (default `get_shared_db_service()`), so it is a real object
with a dependency, not a static dumping ground. **Migration:** `Ability.record/...` →
`ActTrail().record/...` at the 6 `message_processor` sites + inside `ToolDispatcher.dispatch`.

### 4.4 `services/async_delegate_runner.py` — `AsyncDelegateRunner`

Owns the daemon-thread lifecycle for backgrounded tool calls. **Module-level singleton instance**,
matching the codebase's `heartbeat_service = HeartbeatService()` convention — one shared registry,
real instance methods, no classmethod/static dumping ground:

```python
class AsyncDelegateRunner:
    def __init__(self) -> None:
        self._active: dict[str, threading.Event] = {}

    def spawn(self, ability, params, mp) -> str:     # returns "<NAME> dispatched (id: …)"
    def cancel(self, delegate_id: str) -> bool
    def active_ids(self) -> list[str]

async_delegate_runner = AsyncDelegateRunner()        # the shared singleton
```

- `spawn` takes — and the daemon body **captures** — the originating **`mp` object** (never a
  `channel`). It returns the placeholder string (`"<NAME> dispatched (id: …)"`) immediately so the
  ACT iteration is never blocked (§4.0 lever 1). The thread copies contextvars exactly as today,
  runs the ability through the **shared sync-run primitive**, then **delivers via the captured `mp`**
  using the pinned completion mechanism below (spawn a fresh synthesis `MessageProcessor` → emit via
  hooks). Because the thread holds the live `mp`, this works **even after the originating ACT loop has
  closed**. The capture-`mp` mechanism is channel-general by construction, but async is *exposed* only
  where `SUPPORTS_ASYNC` is set — `UserConfig` alone today (§4.8d) — so in practice every background
  completion surfaces on the user channel. (A future push channel that sets `SUPPORTS_ASYNC` would
  surface on *its own* channel, since the captured `mp` carries that config — never force-routed to the
  user channel as today's `dispatch_message` hardcode does.)
- **Behavioural delta (intended):** `web_search` / `web_browse`, which were *always* async, now
  default to **inline** and the model opts into background execution per-call via `async: true`.
  This is the new, more-flexible model Dylan asked for — confirmed in §11.
- **Delivery seam (replaces the `UserConfig` hardcode):** today the daemon delivers via
  `dispatch_message(result_text, channel=channel, hidden_input=True)`, and `dispatch_message` →
  `_start_turn` always rebuilds `UserConfig` (`api/chat.py:249`) — i.e. every async result is
  force-routed to the user channel *and* shoved through the foreground `/chat` turn machinery. The new
  path delivers through the captured `mp` instead. **Completion mechanism (pinned):** on completion the
  runner spawns a **fresh synthesis `MessageProcessor`** (clone of the captured `mp`'s config, `input =
  tool result`, `hidden_input = True`) so the LLM synthesises the result, then **emits the assistant
  turn via the hooks / hot path** (§4.0 lever 2) — it never emits raw text.
- **Each `MessageProcessor` owns its own ACT-loop lifecycle; background synthesis is decoupled from
  `/chat` (resolves ④).** Every MP is fully self-contained — any number can run in parallel without
  interfering, because none of them shares turn state with another. `_active_ump` is **not** a turn
  slot or a concurrency lever: it is simply the in-memory handle that keeps the *foreground user turn*
  reachable from the `/chat` endpoints, which is the **only** reason that turn can be interrupted
  (`/chat/interrupt` → `_active_ump.cancel()`). A background synthesis MP is deliberately **not
  registered** there: it does NOT go through `dispatch_message` / `_start_turn`, does NOT claim
  `_active_ump`, and does NOT cancel/combine the user's in-flight turn. It runs its own loop to
  completion and **emits its assistant result via the hooks** — nothing more. Consequences, all
  intended: it is not user-interruptible via `/chat` (it's a background task), and the user may be
  mid-turn on something else or see several background results land at once — each just appends another
  assistant turn. No queue, registry, or parallel turn manager is added; self-containment is the whole
  mechanism, and delivery is the captured `mp` + hooks, never the `/chat` chokepoint.
- **Sync-run primitive sourcing (phasing):** the run+normalise primitive lives in `_base.py` as
  `_run_ability`/`_normalise_run_result` until P7, where it moves to `ToolDispatcher._run`/`._normalise`.
  At P6 the runner imports the in-place primitive; at P7 that import is repointed to the dispatcher.
  (At P4 the async-delivery change still lives in `_base._run_async_delegate`, which calls the
  in-place primitive directly — the runner does not yet exist.) No forward dependency — the function
  is real and present at every phase.
- **Migration:** `api/chat.py:349` → `async_delegate_runner.active_ids()`; `api/chat.py:370` →
  `async_delegate_runner.cancel(...)`.

> The single shared instance preserves today's cross-call semantics: a `cancel` from `api/chat`
> sees delegates spawned by any processor, exactly as the module-level `_active_delegates` dict did.

### 4.5 `services/client_context.py` — `ClientContext` value object

```python
@dataclass(frozen=True)
class ClientContext:
    lat: float | None; lon: float | None
    location_name: str; city: str; country: str
    time: str; timezone: str; locale: str; language: str; currency: str

    @classmethod
    def current(cls) -> "ClientContext | None":
        """Flatten locale_service (backed by heartbeat_service singleton); None on miss."""

    def as_dict(self) -> dict:   # back-compat shape for capability handlers
```

- `_load_tool_telemetry` dies. Dispatcher sets `ability.telemetry = ClientContext.current().as_dict()`.
- The 8 consumer abilities keep reading `self.telemetry` as a dict. **Decision:** ship `as_dict()`
  and store the **dict** form on `ability.telemetry`, so the 8 consumers are byte-for-byte unchanged
  this pass (typed-attribute migration is a separate, optional cleanup). Blast radius for consumers
  stays zero while the procedural loader is replaced by a value object.

### 4.6 `abilities/_mcp_ability.py` — `_MCPAbility` + MCP dispatch

`_MCPAbility` (synthetic `Ability` subclass, `_SYNTHETIC=True`) + the `_dispatch_mcp` body move here
as a private method. `ToolDispatcher._match` returns `_MCPAbility(tool_name)` for `_mcp_*` names.

### 4.7 `abilities/_event_emitter.py` — `ActEventEmitter` (WS emission)

Replaces the procedural `_emit(config, event)`:

```python
class ActEventEmitter:
    def __init__(self, config: object) -> None: self._config = config
    def emit(self, event: dict) -> None:
        """Broadcast iff config.broadcast_to is non-None; swallow broker errors."""
```

Encapsulates the `broadcast_to is None` gate **once** (today duplicated between `execute` and the
loop). **Migration:** dispatcher constructs `ActEventEmitter(mp.config)` for start/end events;
`message_processor.py:1006` narration uses the same emitter.

> **Decision (open item #8):** emitter object chosen over a `config.broadcasts` property, because it
> removes the duplicated gate and keeps transport out of `ProcessorConfig`. Revisitable.

### 4.8 `ProcessorConfig` — `PostTurnHook` set (composition + the cross-channel router)

The `post_turn` callable-field becomes a **set of independent hook objects** the processor invokes
after the assistant row is persisted. These hooks ARE the "surface on a separate channel" mechanism
from §4.0: each is a single-responsibility unit read off `mp.config`; configs compose the ones they
need; cross-cutting behaviours (metrics, decay, disclosure) become reusable units.

**(a) New `services/post_turn_hook.py` — the hook abstraction.**

```python
class PostTurnHook(ABC):
    """One independent, self-contained unit of after-turn work.

    INDEPENDENCE CONTRACT (load-bearing — do not weaken):
      - Hooks are mutually INDEPENDENT. A hook MUST NOT depend on, observe, or
        order itself relative to any other hook. There is NO defined execution
        order, and none may ever be assumed.
      - The framework MAY run hooks in any order, and MAY in the future run them
        concurrently / fully async from one another. A hook must therefore be
        self-contained: it owns its own context (copy contextvars if it touches
        request-scoped state), holds no reference to sibling hooks, and shares no
        mutable state with them.
      - FAILURE ISOLATION: a hook that raises MUST NOT affect any sibling hook.
        The invoker isolates each call (log + continue). One hook failing is a
        non-event for the others.
    """
    @abstractmethod
    def run(self, mp: "MessageProcessor", result_text: str) -> None: ...
```

**(b) `ProcessorConfig` field** (replaces `post_turn: Callable | None` at line 121):

```python
post_turn_hooks: tuple[PostTurnHook, ...] = ()      # tuple — config is frozen; empty = no-op
```

**(c) The 5 current callables become hook classes** (each beside its config):

| Today (callable) | Hook class | Home |
|---|---|---|
| `_ump_post_turn` | `ProactiveSuggestionHook` | `configs/channels/user.py` |
| `_make_eamp_post_turn(name, project, loop_in_human)` | `DiscloseToHumanHook(agent_name, project)` (composed only when `loop_in_human`) | `configs/channels/external_agent.py` |
| `_pattern_post_turn` | `PatternDecayHook` | `configs/channels/pattern.py` |
| `_geo_pattern_post_turn` | `GeoCounterHook` | `configs/channels/geo_pattern.py` |
| `_user_summary_post_turn` | `PersistUserSummaryHook` | `configs/channels/user_summary.py` |

EAMP's closure-over-constructor-args becomes honest object fields:
`post_turn_hooks = (DiscloseToHumanHook(agent_name, project),) if loop_in_human else ()`. The hook's
body spawns a `UserConfig` `MessageProcessor` with hidden input — the §4.0 cross-channel surface, and
the *same* category of background, emit-only processor as the async-completion synthesis MP (§4.4).
It therefore spawns the MP **directly and emits via hooks**, **not** through the `/chat` chokepoint:
like every background MP it is invisible to the foreground turn machinery (never claims `_active_ump`,
never cancels/combines the user's in-flight turn). The user-visible disclosure (a hidden-input
`UserConfig` turn appearing in the chat) is unchanged; the one intended delta is that it no longer
routes through `dispatch_message` / `_start_turn`, so it can no longer disturb the active-turn slot.

**(d) Async capability — `SUPPORTS_ASYNC` ClassVar (replaces `_supports_async_delivery`, different
job).** The old `_supports_async_delivery(channel)` was a *routing* gate — that job is gone (routing
is the captured `mp` + hooks, §4.0/§4.4). But a distinct question remains: **may the model even
choose async on this loop?** Only a push channel with a durable session can honour a deferred
result; every other loop consumes `process()`'s synchronous return (`talk_to_chalie`
[mcp_server/server.py:179], delegate sub-loops, compaction, subconscious), so a deferred result
would arrive after the caller has gone.

```python
class ProcessorConfig(ABC):
    SUPPORTS_ASYNC: ClassVar[bool] = False     # push channel + durable session only

class UserConfig(ProcessorConfig):
    SUPPORTS_ASYNC = True                       # the only async-capable channel today
```

`Ability.get_input_schema(mp)` injects the `async` property **iff** `mp is not None and
mp.config.SUPPORTS_ASYNC` (§4.1). Everywhere else the property is omitted and all tools are
synchronous. This is a *schema-exposure* gate, not a routing gate — which is exactly why deleting
the old routing property was still correct. (`SUPPORTS_ASYNC` is a `ClassVar`, not a dataclass
field, so the frozen `ProcessorConfig` stays a pure value object.)

**(e) Invocation contract in `MessageProcessor._record`** (replaces the single
`self.config.post_turn(self, response_text)` call):

```python
for hook in self.config.post_turn_hooks:
    try:
        hook.run(self, response_text)
    except Exception as exc:                       # noqa: BLE001 — failure isolation
        logger.warning("[post_turn] hook %s failed (isolated): %s",
                       type(hook).__name__, exc)
```

Sequential-with-isolation is the *current* implementation; the *contract* (a) promises only
independence + isolation, so a later switch to concurrent/async fan-out (one
`contextvars.copy_context()` thread per hook, mirroring `AsyncDelegateRunner`) is a drop-in change
that breaks no caller. The spec commits to the contract, not the sequential loop.

### 4.9 Delete the import/patch hack

Once `Ability` lives alone in `_ability.py`, `_registry` imports it without pulling registry/policy/
broker. Remove:
- `AbilityRegistry/PolicyManager/WebSocketBroker = None` module aliases.
- `_populate_module_aliases()` and every `if X is None: _populate_module_aliases()` guard.

The one regression test that patches `abilities._base.X`
(`test_dispatch_alias_binding_regression.py`) is repointed at the real collaborators
(`abilities._dispatcher` / `abilities._registry` / `services.policy_manager`).

### 4.10 Delete confirmed-dead code

In scope (Dylan: "if it's dead, remove it") — evidence §3:
- **`POLICY_CATEGORY` / `POLICY_LABELS`** — remove the ABC ClassVars and every definition across the
  28 abilities. Zero readers anywhere; nothing serializes them.
- **`AbilityRegistry.policy_visible()`** (`_registry.py`) — remove; no call sites (comment refs
  only). Removing it makes `MessageProcessor.ALWAYS_AVAILABLE` a removal *candidate* — verify its
  other readers at build time before touching it (do not assume it is dead).

A final zero-caller grep precedes each deletion (cheap insurance against a dynamic `getattr`).

---

## 5. Call-site migration table

| Old | New | Sites |
|---|---|---|
| `Ability.use(mp, n, p)` | `ToolDispatcher(mp).dispatch(n, p)` | mp:652,662,709; memory:348,349; chat:448; +2 tests |
| `Ability.record(...)` | `ActTrail().record(...)` | mp:550,948,997 + dispatcher |
| `Ability.fetch_by_transcript_id(t)` | `ActTrail().fetch_by_transcript_id(t)` | mp:894,909 |
| `Ability.render(row)` | `ActTrail.render(row)` | mp:910 |
| `Ability.get_active_delegates()` | `async_delegate_runner.active_ids()` | chat:349 |
| `Ability.cancel_delegate(id)` | `async_delegate_runner.cancel(id)` | chat:370 |
| `_emit(cfg, ev)` | `ActEventEmitter(cfg).emit(ev)` | mp:996,1006 + dispatcher |
| `ASYNC_CAPABLE = True` | *(deleted)* — model sets `async: true` per call | web_search:106, web_browse:101 |
| `get_input_schema` overrides returning `INPUT_SCHEMA` directly | start from `super().get_input_schema(mp)` so `async` is injected | audit (e.g. find_tools) |
| `_supports_async_delivery(channel)` / `channel=` plumbing | *(deleted)* — `mp.config`-driven re-entry | _base:39,193,209; chat dispatch |
| `from abilities._base import Ability` | `from abilities._ability import Ability` | 36 abilities + registry + mp + chat + tests |
| `import abilities._base as base` (patch) | repoint to real modules | test_dispatch_alias_binding_regression.py |
| `POLICY_CATEGORY` / `POLICY_LABELS` defs | *(deleted)* | ABC + 28 abilities |
| `AbilityRegistry.policy_visible()` | *(deleted)* | _registry.py |

---

## 6. Build phasing (each phase independently committable + `pytest -m unit -q` green)

Ordered easy-wins-first, the contract-touching changes last.

- **P0 — dead-code removal.** Delete `POLICY_CATEGORY` / `POLICY_LABELS` (ABC + 28 abilities) and
  `AbilityRegistry.policy_visible()` after a final zero-caller grep. Pure deletion, isolated, no
  behaviour. *(lowest risk)*
- **P1 — `ClientContext` value object.** Add `services/client_context.py`; dispatcher not yet
  touched, so temporarily `_load_tool_telemetry` delegates to `ClientContext.current().as_dict()`.
  Zero consumer change. *(low risk)*
- **P2 — `_MCPAbility` extraction.** Move proxy + `_dispatch_mcp` to `abilities/_mcp_ability.py`;
  `_base` imports it back for now. *(low risk)*
- **P3 — `ActEventEmitter`.** Add `abilities/_event_emitter.py`; rewire `_base.execute` and
  `message_processor` narration to use it; delete `_emit`. *(low risk)*
- **P4 — per-call async + captured-`mp` delivery.** Delete `ASYNC_CAPABLE` (2 abilities + ABC); add
  `SUPPORTS_ASYNC: ClassVar = False` on `ProcessorConfig` and `= True` on `UserConfig`; inject
  `async` in base `get_input_schema` **gated on `mp.config.SUPPORTS_ASYNC`** and audit overrides to
  call `super()`; change the decision at `_base.execute` to pop `params["async"]` (placeholder
  returned immediately, never blocking); change `_run_async_delegate` to capture and deliver through
  the originating `mp` (hot-path emit + hooks) instead of the `UserConfig`-hardwired
  `dispatch_message(channel=…)`; delete `_supports_async_delivery`. This is the intended behavioural
  delta (delegates default inline; async offered only on the user channel; async results no longer
  force-routed). *(medium — behavioural)*
- **P5 — `ActTrail` repository.** Add `services/act_trail.py`; migrate the 6 `message_processor`
  sites + the `record` call inside `use`. Delete `record/fetch/render` from `Ability`. *(medium)*
- **P6 — `AsyncDelegateRunner`.** Add `services/async_delegate_runner.py` (`spawn(ability, params,
  mp)`); migrate `api/chat.py` (2 sites); `_base.execute` calls `async_delegate_runner.spawn`.
  Delete `_active_delegates`, `_run_async_delegate`, `cancel_delegate`, `get_active_delegates`.
  *(medium)*
- **P7 — `ToolDispatcher` + `Ability` ABC split.** Create `abilities/_ability.py` (clean ABC) and
  `abilities/_dispatcher.py`; move `use/match/_bind/execute/_run_ability/_normalise_run_result`;
  migrate all 8 `use` call sites + 36 ability imports + registry import; delete the import hack;
  repoint the regression test. **Delete `abilities/_base.py`.** *(highest — this is the cutover)*
- **P8 — `post_turn` callable → `PostTurnHook` set.** Add `services/post_turn_hook.py`
  (`PostTurnHook` ABC + independence/isolation contract); convert the 5 callables to hook classes
  beside their configs; replace the `post_turn` field with `post_turn_hooks: tuple[...] = ()`;
  rewrite `_record` to the isolated invocation loop (e). *(touches every config — do last, isolated)*

> P0–P6 each leave a still-present-but-shrinking `_base.py`. P7 is the deletion. P8 is the
> `ProcessorConfig` OOP cleanup, kept separate so a routing regression can't be confused with a
> dispatch regression.

---

## 7. LOC estimate

Net **≈ neutral to slightly negative** now that dead code is in scope. Relocation is roughly
zero-sum (delete `_base.py` −633, relocate bodies ≈ +424, 6 new file scaffolds ≈ +60,
`ClientContext` ≈ +25, `PostTurnHook` ABC + 5 hooks ≈ +40, `async` injection ≈ +10), and the
deletions pull it back down: the import hack (~45), `_supports_async_delivery`, `ASYNC_CAPABLE`
(2 lines), `POLICY_*` across the ABC + 28 abilities (≈ −100 to −150), `policy_visible()` (≈ −15).
36 ability import-line swaps net 0. The value is cohesion/SRP + composability, not line count. The
trail-SQL collapse (§10) remains the candidate for a further net-negative follow-up.

---

## 8. Testing (per `writing-feature-tests` — zero mocks, real hot path)

- Every phase ends with `cd backend && pytest -m unit -q` green (CLAUDE.md rule 3).
- Behaviour-preserving phases assert *observable* output is byte-identical where the spec claims so
  (act-trail render string, telemetry dict shape, dispatched-delegate message text).
- **Per-call async (P4) gets explicit feature tests:** (1) the `async` property is **present** in a
  tool's schema under `UserConfig` and **absent** under a non-async config (`SUPPORTS_ASYNC` gate,
  §4.8d); (2) a real tool dispatched with `async: false` (default) returns its result **inline**;
  (3) the same tool with `async: true` returns the "dispatched (id: …)" text and later delivers a
  real follow-up turn via the captured `mp` on the **user** channel (guards the §4.0/§4.4 delta).
- New seams get feature tests exercising the real path: `ToolDispatcher(mp).dispatch(...)` end-to-end
  through `PolicyManager.wrap` and a real `Ability.run`; `ActTrail` round-trips a real `tool_calls`
  row; `AsyncDelegateRunner.spawn(..., mp)` delivers via real `mp.config` re-entry into a real fresh
  turn; `ClientContext.current()` reads the real `heartbeat_service`-backed `locale_service`.
- `test_ability_base.py` splits to follow the new modules (`test_tool_dispatcher.py`,
  `test_act_trail.py`, `test_async_delegate_runner.py`, `test_client_context.py`,
  `test_post_turn_hooks.py`).
- **Hook isolation is feature-tested explicitly**: a config composed with `(ThrowingHook(),
  PersistUserSummaryHook())` must still persist the summary on a real turn — i.e. the throwing
  sibling does not block the working one. This guards the load-bearing isolation contract, not a
  mock.
- Nightly scenarios are the regression net for the cutover (P7) and routing (P8).

---

## 9. Risks

- **P7 cutover is wide** (36 imports + 8 call sites + hack removal in one commit). Mitigation:
  P0–P6 shrink `_base.py` first so P7 is mostly mechanical import/rename; the regression test for
  alias binding is the canary.
- **P4 is a real behavioural change**, not a pure refactor: delegates that were always-async become
  inline-by-default, and async delivery moves off the hardwired user channel onto the captured
  `mp`'s channel. Mitigation: the explicit P4 feature tests above + nightly delegate scenarios. If a
  delegate scenario depended on fire-and-forget timing or user-channel delivery, it surfaces here.
- **`async` injection gating**: the `async` property is injected only when
  `mp.config.SUPPORTS_ASYNC` (user channel today). Two failure modes: (a) an override of
  `get_input_schema` that does not call `super()` drops the option on the user channel; (b) a
  forgotten gate-check would leak `async` onto a synchronous `process()`-return channel and break it
  (§4.8d). Mitigation: the override audit row in §5 plus a feature test asserting `async` is
  **present** on a `UserConfig` tool schema and **absent** on a non-async config's schema.
- **Captured-`mp` delivery seam**: today's daemon delivers via `dispatch_message` → `_start_turn`
  which hardcodes `UserConfig` (`api/chat.py:249`). Routing through the captured `mp` instead is the
  load-bearing P4/P6 detail; the user-channel mid-turn-cancel semantics must be preserved for
  non-hidden input. The completion mechanism is pinned to re-injecting a fresh synthesis turn (§4.4).
- **Concurrency of async completions (④) — self-contained MPs, nothing shared.** Every
  `MessageProcessor` owns its own ACT-loop lifecycle; any number run in parallel without interfering.
  `_active_ump` is **only** the interrupt handle for the foreground user turn (so `/chat/interrupt`
  can reach it) — not a shared slot, so there is nothing to contend for. Background synthesis MPs are
  simply not registered there: they never claim `_active_ump`, never cancel/combine the user's turn,
  and only emit via hooks. Risk to watch at build time: the runner must spawn the synthesis MP
  **directly** (NOT via `dispatch_message` / `_start_turn`); a feature test asserts a background
  completion emits an assistant turn **without** cancelling a concurrently-active user turn.
- **`telemetry` shape**: consumers read a dict today. Storing a dict (`as_dict()`) on
  `ability.telemetry` keeps them unchanged; do **not** switch them to typed attrs this pass.
- **`post_turn` field→hook-set (P8)**: a frozen-dataclass `Callable | None` field becoming a
  `tuple[PostTurnHook, ...]` is a real shape change for anything introspecting `config.post_turn`.
  Blast scan shows only `_record` reads it — verify no test asserts on the old field/callable.
- **Hook independence is a real constraint, not a nicety**: a hook author MUST NOT rely on order or
  sibling state, because the invoker is licensed to parallelise. If a future hook needs request-scoped
  context (locale/timezone), it copies contextvars itself.
- **`AsyncDelegateRunner` shared state**: must preserve the single shared registry semantics
  (cancellation from `api/chat` must see delegates spawned anywhere). The module-level singleton
  instance (`async_delegate_runner`) preserves it; do not construct per-call instances.
- **`ALWAYS_AVAILABLE` cascade (P0)**: removing `policy_visible()` may orphan
  `MessageProcessor.ALWAYS_AVAILABLE`. Verify its other readers before removing it; if still read,
  leave it.

---

## 10. Flagged follow-ups (NOT in this scope)

1. **`abilities/_delegate.py`** holds 2 procedural helpers (`delegate_goal`, `render_trail`) — same
   OOP smell, separate cleanup.
2. **Trail SQL collapse** — `ActTrail` raw SQL likely overlaps other repositories; the offsetting
   net-negative opportunity.

---

## 11. Open decisions — resolved

| # | Decision | Choice |
|---|---|---|
| Scope | This session | Full plan, **phased build** |
| Routing | What the config owns | **Additive** — side-deliveries are post-turn hooks on `mp.config`; `process()` return unchanged |
| Context handle | What dispatch passes | **Always `mp`** — never a `channel` string; same-channel reuses the captured `mp`, cross-channel = hooks on `mp.config` |
| `async` lever | What it controls | **Blocking only** — does this call block the ACT iteration? `run()` is identical; loop always waits for a result; async just short-circuits the return with a placeholder while a thread runs the real work. NOT routing |
| Async model | Per-ability vs per-call | **Per-call** — `ASYNC_CAPABLE` deleted; base `get_input_schema` injects an `async` boolean; **model decides** per call; default inline |
| Async exposure | Which loops offer `async` | **`SUPPORTS_ASYNC` ClassVar on `ProcessorConfig`** — `True` only on `UserConfig`; gates whether `async` is injected into tool schemas; elsewhere omitted ⇒ all tools synchronous (push channel + durable session is the requirement) |
| Async delivery | Target | **Captured `mp`** — thread holds the live `mp`, delivers via its hot path + hooks even after the loop closes; not the hardwired `UserConfig`/user channel |
| Async concurrency (④) | background vs foreground turns | **Self-contained MPs** — each owns its full ACT-loop lifecycle; parallel MPs never interfere. `_active_ump` is only the foreground turn's **interrupt handle** (the reason `/chat/interrupt` can reach it), not a shared slot. Background MPs aren't registered there: spawn directly, **emit-only via hooks**, never cancel/combine the user's turn; concurrent completions are additive assistant turns |
| post_turn | shape | **Set of `PostTurnHook` objects** (not a method, not bare callables) — composition + the cross-channel router |
| hook contract | order / failure | **Order NEVER matters** (future-parallelisable); **failures isolated** per hook |
| #8 | WS emission | **`ActEventEmitter` object** (not a `config.broadcasts` property) |
| telemetry | consumer shape | store **dict** (`as_dict()`) on `ability.telemetry` — zero consumer change |
| POLICY_* / policy_visible | dead code | **Remove** (in scope, §4.10) — zero readers/callers |
