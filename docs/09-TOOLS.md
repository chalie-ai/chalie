# Tools System

Every dispatchable capability in Chalie is an `Ability` subclass under `backend/abilities/`. Tool scope — which abilities are pre-injected and which are discoverable — is carried per-turn on the `ProcessorConfig` (`always_available`, `discoverable`, `blocked`). There are no `MessageProcessor` subclasses: each channel supplies a config built from the shared defaults in `backend/configs/channels.py` (`DEFAULT_ALWAYS_AVAILABLE`, `DEFAULT_DISCOVERABLE`).

**Innate abilities (`always_available`)** are pre-injected on every ACT iteration for the current turn. The shared default is `["find_skills", "find_tools", "memory"]`. Most configs use this directly — the user, DMN, and external-agent channels all use the default. The pattern-match configs override to `["save_pattern", "save_graph"]`. `find_skills` is innate (not discoverable) because returning procedural playbooks is infrastructure — the same rationale that makes `find_tools` and `memory` innate.

**Discoverable abilities (`discoverable`)** are never pre-injected. `DEFAULT_DISCOVERABLE` lists the first-party abilities surfaceable at runtime: `bash`, `browser`, `calendar`, `chalie_docs`, `code_eval`, `contacts`, `document`, `email`, `file_permissions`, `file_write`, `home`, `list`, `mcp_manager`, `news`, `place`, `programming_docs_search`, `read`, `review_tool_calls`, `review_transcript`, `schedule`, `search`, `search_files`, `skill_builder`, `timer`, `ubiquiti`, `vision`, `weather`, `web_browse`, `web_download`, `web_search`. The `find_tools` ability activates tools into the running ACT turn. It supports two input modes — `select` (array of exact tool names, case-insensitive match against the effective allow-list = DISCOVERABLE ∪ online MCP tool names) and `query` (natural-language hybrid vec+FTS RRF semantic search). Both are optional; at least one must be supplied. When both are given, `select` wins and `query` is ignored. The `query` path applies a `MIN_RRF_SCORE = 0.075` relevance floor (scores below are single-signal rank artifacts) and caps results at 5; names that pass the floor are appended to `active_tools`. The `select` path appends all matched names immediately, and reports any unresolved names as `Tools not found or unavailable: <names>`. Both paths return the same result format: a lead phrase containing `added the following tools` followed by a JSON array of `{"name": <tool>, "input_schema": <schema>}` entries. When the LLM invokes `find_tools`, the matched abilities become available for the remainder of that ACT loop. All first-party and MCP abilities are reachable exclusively through this path — pre-injecting them would bloat context, create staleness bugs, and break tool-agnostic routing.

**Blocked abilities (`blocked`)** is a per-config `frozenset` (empty by default). A config narrows the discoverable scope by listing tool names to exclude from both `discoverable` and the `find_tools` index. The three discovery-capable loops — user, DMN, and external-agent — each carry a `blocked` set built from shared constants in `configs/channels/_common.py`: all three block `PATTERN_WRITE_TOOLS` (`save_pattern`/`save_graph`, exclusive to the pattern passes) and `DELEGATE_INTERNAL_TOOLS` (`browser`/`search`, the raw web tools reachable only via the `web_search`/`web_browse` delegates); `DMN_CONFIG` additionally blocks `DELEGATE_TOOLS` (`web_search`/`web_browse`/`vision`), preventing the background reflection loop from spawning delegate work (the delegate tools replaced the retired single `subagent` tool). The `vision` delegate — an image-reading subagent on the brain's vision provider — follows the same rule: surfaceable on the user-facing loops (UserConfig, external-agent) but blocked on background loops; its policy defaults are `allow` on chat and external_agent, `deny` on subconscious.

**`get_search_tooltip()`** is a required zero-arg getter (`@abstractmethod`) on every non-INTERNAL `Ability` subclass, returning a non-empty 2–5 word description used to build the `find_tools` index — enforced at import time by `__init_subclass__` (which probes a throwaway `mp=None` instance). It is one of the five metadata getters every concrete ability implements: `get_name`, `get_summary`, `get_examples`, `get_search_tooltip`, `get_parameters` (see "Adding a first-party ability" below).

## How tools are loaded on a turn

Two loading tiers stack on each ACT iteration and are de-duplicated first-seen:

1. **Unconditional** — every ability in the processor's `ALWAYS_AVAILABLE` list. Always present.
2. **Dynamic (discoverable)** — abilities surfaced this turn via the `find_tools` ability's semantic search, gated to the processor's `DISCOVERABLE` list. Every non-innate first-party ability is reachable exclusively through this path.

`ModeGateService` is a built-but-currently-dormant prompt-steering layer; it **does not gate tool availability**. The design: it classifies the turn along eight independent cognitive intents (`research`, `coding`, `brainstorm`, `analyze`, `plan`, `write`, `math`, `converse`) using a small ONNX multi-label head; per-mode state follows an asymmetric EMA (fire snaps up, miss decays by 0.75 per turn), persists across turns in MemoryStore under `mode_gate:state` (cleared by `/privacy/delete-all`), and once a mode's EMA crosses `STEER_THRESHOLD = 0.6` its steering directives (long-summary swap on `converse`; brainstorm/research/analyze suffixes) are appended to the system prompt. The user config reads this via `get_system_prompt_additions()` when `mp._mode_gate_cached` is set — but the flat ACT-loop refactor dropped the wiring that ran the gate per turn (the old `UserMessageProcessor._get_mode_state()`), so `_mode_gate_cached` is never assigned, the classifier never runs on a real turn, and no steering ever fires (pending a decision to re-wire it in `MessageProcessor._setup()` or remove the layer).

Observability: when live, the gate emits one `[MODE-GATE]` INFO log line per user turn with the full probability vector, state before/after, and the active mode set — the grep-friendly anchor for nightly scenarios. While dormant (see above) this line is never emitted.

## Tool status

Three status values appear in the tools list:

| Status | Meaning |
|---|---|
| `system` | Built-in, no configuration required |
| `available` | Discovered but not yet configured (missing required secrets) |
| `connected` | Fully configured and ready to use |

## Adding a first-party ability

A first-party ability is a Python module under `backend/abilities/` that subclasses `Ability`, implements the five zero-arg metadata getters, and a single dispatch method `run(self, params)`:

```python
from abilities._ability import Ability

class WeatherAbility(Ability):
    def get_name(self) -> str:
        return "weather"

    def get_summary(self) -> str:
        return "Live weather lookup by coordinates or city name."

    def get_examples(self) -> list[str]:
        return [
            "what is the weather in Valletta",
            "is it raining in San Francisco",
            # 6 to 8 entries total — enforced by __init_subclass__
        ]

    def get_search_tooltip(self) -> str:
        return "weather lookup"

    def get_parameters(self) -> dict:
        return {"type": "object", "properties": {...}}

    def run(self, params):
        ...
```

The metadata getters replaced the old `NAME` / `SUMMARY` / `EXAMPLES` / `SEARCH_TOOLTIP` / `INPUT_SCHEMA` ClassVars (TKT-837). They are zero-arg and read `self.mp` (the invoking MessageProcessor, constructor-injected) when a value depends on the live request; at `self.mp is None` (introspection / `build_ability_db`) they MUST return deterministic base text so the embedded index stays machine-independent. The full LLM-facing descriptor — `{name, description, input_schema}` plus the framework fields `act_summary` (always, required) and `async` (iff `config.SUPPORTS_ASYNC`) — is assembled in ONE place, the `@typing.final Ability.get_input_schema()`; overriding it (or `_inject_framework_fields`) is an import-time error. There is no `description` field — `get_summary()` is both the model-facing description and the search-corpus base text.

`params` are the LLM-extracted arguments validated against `get_parameters()` (the framework `act_summary` / `async` keys are stripped before `run()` sees them). The conversation channel and flattened client telemetry (location, time, locale — fields may be null) are reached via `self.mp.config.channel` and `self.telemetry` (set by the dispatch spine immediately before `run()`). Abilities run to completion — there is no framework execution timeout. The return value is dispatched through `ToolRenderAndRecordService` and tag-formatted by `services/innate_skills/_tag.py`. `get_summary()` + `get_examples()` drive the semantic search row that `find_tools` matches on; `get_summary()` is the most important — it determines whether `find_tools` surfaces this ability. `get_search_tooltip()` provides a compact label for the `find_tools` index. Both `find_tools` and `find_skills` inherit from `SearchableAbility` (`abilities/_search.py`) which provides the shared vec+FTS5 RRF fusion search infrastructure.

After registering the ability, wire it into the appropriate config(s). For a discoverable ability, add its name (the string `get_name()` returns) to `DEFAULT_DISCOVERABLE` in `backend/configs/channels.py`. For an always-available ability, add it to `DEFAULT_ALWAYS_AVAILABLE` instead. If a specific channel should not see the ability, add the name to that config's `blocked` frozenset. Then regenerate `abilities.sqlite` via `python -m utils.build_ability_db` so the embedded index is up to date; CI's drift check fails the build if `abilities_sha.json` does not match.

## Configuration

Tools that require API keys or custom endpoints declare their required config keys in their metadata. Configure them through the Brain UI (Settings > Tools) or via the REST API — see the API reference for endpoints. Stored secrets are masked in all API responses.

## Ability output format

Every ability returns its result as a canonical tag block defined in `backend/services/innate_skills/_tag.py`. `_tag.py` is the single source of truth — no ability constructs its own format string. (The `services/innate_skills/` directory holds only this formatter after the Phase 4 cutover; every dispatchable ability lives under `backend/abilities/`.)

```
[<ability_name>(k1=v1, k2=v2)]
<body>
[end:<ability_name>]
```

If the body is empty (error path with no content), the body line is omitted:

```
[memory(action=recall, error=no-query)]
[end:memory]
```

Errors are just arguments — `error=<slug>` in the opener, not a separate response format. Multi-line bodies (e.g. memory recall results, rich render reference) appear verbatim between opener and terminator.

The `memory` ability preserves its inner per-row marker format inside the body so downstream services that parse `[id:X,relevance:Y]` continue to work:

```
[memory(query=Malta, results=3)]
[id:residence,relevance:high] Valletta
[id:partner,relevance:medium] Sarah
[id:food_and_drink,relevance:low] pastizzi
[end:memory]
```

`find_tools` returns a plain string (the lead phrase + JSON array described above). The `_discovered_tools` side-channel dict shape was removed in v2 — `find_tools` has no dict return path.

## Safety constraints

- Tool invocations time out. Exceeded timeouts are logged as failures.
- Output is sanitized before it enters LLM context: action-like patterns are stripped.
- Every invocation is written to an audit trail with the topic, outcome, and execution time.
- A global kill switch can disable all tools if needed.
