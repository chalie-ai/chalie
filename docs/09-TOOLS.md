# Tools System

Every dispatchable capability in Chalie is an `Ability` subclass under `backend/abilities/`. Tool scope — which abilities are pre-injected and which are discoverable — is declared on the `MessageProcessor` base class and overridden only where needed by subclasses.

**Innate abilities (ALWAYS_AVAILABLE)** are pre-injected on every ACT iteration for the current processor. The base class default is `["find_tools", "memory"]`. Most processors inherit this directly — `UserMessageProcessor`, `DMNMessageProcessor`, and `ExternalAgentMessageProcessor` all use the base default. `PatternMatchProcessor` overrides to `["save_pattern", "save_graph"]`. `SubagentProcessor` sets `ALWAYS_AVAILABLE` per-instance from the type registry (`SUBAGENT_TYPES[agent_type]['native_tools']`).

**Discoverable abilities (DISCOVERABLE)** are never pre-injected. The base class default lists all 20 first-party abilities: `browser`, `calendar`, `code_eval`, `contacts`, `document`, `email`, `home`, `list`, `news`, `place`, `programming_docs_search`, `read`, `review_tool_calls`, `review_transcript`, `schedule`, `search`, `subagent`, `timer`, `ubiquiti`, `weather`. The `find_tools` ability performs semantic search against `abilities.sqlite` at runtime, restricted to the calling processor's `DISCOVERABLE` list minus its `_BLOCKED` set. When the LLM invokes `find_tools`, the matching abilities become available for the remainder of that ACT loop. All external (first-party + interface) abilities are reachable exclusively through this path — pre-injecting them would bloat context, create staleness bugs, and break tool-agnostic routing.

**Blocked abilities (_BLOCKED)** is a `frozenset` on the base class (empty by default). Subclasses override it to exclude specific tools from both `DISCOVERABLE` and the `find_tools` index without redeclaring the full list. `DMNMessageProcessor` and `SubagentProcessor` both set `_BLOCKED = frozenset({"subagent"})` — preventing background processes from spawning further background work.

**SEARCH_TOOLTIP** is a required `ClassVar[str]` on every non-INTERNAL `Ability` subclass (enforced at import time by `__init_subclass__`). It provides a 2–5 word description used to build the `find_tools` index. When `get_tools()` assembles the tool list, `_enrich_find_tools_schema()` injects a compact index of all discoverable tools (name + tooltip) into the `find_tools` query field description — giving the LLM visibility into what `find_tools` can surface before calling it.

**Interface tools** are capabilities exposed by external applications that have paired with Chalie via the interface protocol. They are projected into the abilities surface at boot via the same RRF discovery path. See [15-INTERFACES.md](15-INTERFACES.md).

## How tools are loaded on a turn

Two loading tiers stack on each ACT iteration and are de-duplicated first-seen:

1. **Unconditional** — every ability in the processor's `ALWAYS_AVAILABLE` list. Always present.
2. **Dynamic (discoverable)** — abilities surfaced this turn via the `find_tools` ability's semantic search, gated to the processor's `DISCOVERABLE` list. Every non-innate ability (first-party + interface) is reachable exclusively through this path.

`ModeGateService` runs once per user turn but **does not gate tool availability**. It classifies the turn along eight independent cognitive intents (`research`, `coding`, `brainstorm`, `analyze`, `plan`, `write`, `math`, `converse`) using a small ONNX multi-label head; per-mode state follows an asymmetric EMA (fire snaps up, miss decays by 0.75 per turn). State persists across turns in MemoryStore under `mode_gate:state` and is cleared by `/privacy/delete-all`. The active mode set powers prompt-steering directives in `UserMessageProcessor` (long-summary swap on `converse`, brainstorm/research/analyze suffixes appended to the system prompt) and is reserved for future mode-driven features.

Observability: every user turn emits one `[MODE-GATE]` INFO log line with the full probability vector, state before/after, and the active mode set. This is the grep-friendly anchor for nightly scenarios.

## Tool status

Three status values appear in the tools list:

| Status | Meaning |
|---|---|
| `system` | Built-in, no configuration required |
| `available` | Discovered but not yet configured (missing required secrets) |
| `connected` | Fully configured and ready to use |

## Adding a first-party ability

A first-party ability is a Python module under `backend/abilities/` that subclasses `Ability` and implements a single dispatch method:

```python
from abilities._base import Ability

class WeatherAbility(Ability):
    NAME = "weather"
    SUMMARY = "Live weather lookup by coordinates or city name."
    EXAMPLES = [
        "what is the weather in Valletta",
        "is it raining in San Francisco",
        # 6 to 8 entries total — enforced by __init_subclass__
    ]
    INPUT_SCHEMA = {"type": "object", "properties": {...}}
    TIMEOUT = 10  # optional; default is 10

    def execute(self, channel, params, telemetry):
        ...
```

`channel` is the current conversation channel, `params` are the LLM-extracted arguments validated against `INPUT_SCHEMA`, and `telemetry` carries flattened client context (location, time, locale — fields may be null). The return value is dispatched through `ToolRenderAndRecordService` and tag-formatted by `services/innate_skills/_tag.py`. SUMMARY + EXAMPLES drive the semantic search row that `find_tools` matches on; SUMMARY is the most important field — it determines whether `find_tools` surfaces this ability. SEARCH_TOOLTIP provides a compact label for the `find_tools` index (injected into the tool's query description at runtime).

After registering the ability, wire it into the appropriate processor(s). For a discoverable ability, add its `NAME` to `DISCOVERABLE` on `MessageProcessor` (the base class default). For an always-available ability, add it to `ALWAYS_AVAILABLE` instead. If a specific processor should not see the ability, add the name to that processor's `_BLOCKED` frozenset. Then regenerate `abilities.sqlite` via `python -m utils.build_ability_db` so the embedded index is up to date; CI's drift check fails the build if `abilities_sha.json` does not match.

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

`find_tools` and `review_tool_calls` return dicts (the orchestrator reads `_discovered_tools` as a side channel). Their `text` field is wrapped in a tag block; side-channel keys are untouched.

## Safety constraints

- Tool invocations time out. Exceeded timeouts are logged as failures.
- Output is sanitized before it enters LLM context: action-like patterns are stripped.
- Every invocation is written to an audit trail with the topic, outcome, and execution time.
- A global kill switch can disable all tools if needed.
