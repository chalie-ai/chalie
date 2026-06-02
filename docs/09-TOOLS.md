# Tools System

Every dispatchable capability in Chalie is an `Ability` subclass under `backend/abilities/`. Tool scope — which abilities are pre-injected and which are discoverable — is carried per-turn on the `ProcessorConfig` (`always_available`, `discoverable`, `blocked`). There are no `MessageProcessor` subclasses: each channel supplies a config built from the shared defaults in `backend/configs/channels.py` (`DEFAULT_ALWAYS_AVAILABLE`, `DEFAULT_DISCOVERABLE`).

**Innate abilities (`always_available`)** are pre-injected on every ACT iteration for the current turn. The shared default is `["find_skills", "find_tools", "memory"]`. Most configs use this directly — the user, DMN, and external-agent channels all use the default. The pattern-match configs override to `["save_pattern", "save_graph"]`. `find_skills` is innate (not discoverable) because returning procedural playbooks is infrastructure — the same rationale that makes `find_tools` and `memory` innate.

**Discoverable abilities (`discoverable`)** are never pre-injected. `DEFAULT_DISCOVERABLE` lists the first-party abilities surfaceable at runtime: `bash`, `browser`, `calendar`, `chalie_docs`, `code_eval`, `contacts`, `document`, `email`, `file_permissions`, `file_write`, `home`, `list`, `mcp_manager`, `news`, `place`, `programming_docs_search`, `read`, `research`, `review_tool_calls`, `review_transcript`, `schedule`, `search`, `search_files`, `skill_builder`, `summariser`, `timer`, `ubiquiti`, `weather`, `web_browse`, `web_download`, `web_search`. The `find_tools` ability performs semantic search against `abilities.sqlite` at runtime, restricted to the calling config's `discoverable` list minus its `blocked` set. When the LLM invokes `find_tools`, the matching abilities become available for the remainder of that ACT loop. All external (first-party + interface) abilities are reachable exclusively through this path — pre-injecting them would bloat context, create staleness bugs, and break tool-agnostic routing.

**Blocked abilities (`blocked`)** is a per-config `frozenset` (empty by default). A config narrows the discoverable scope by listing tool names to exclude from both `discoverable` and the `find_tools` index. `DMN_CONFIG` sets `blocked = frozenset({"web_search", "research", "web_browse", "summariser"})` — preventing the background reflection loop from spawning delegate work (the four delegate tools replaced the retired single `subagent` tool).

**SEARCH_TOOLTIP** is a required `ClassVar[str]` on every non-INTERNAL `Ability` subclass (enforced at import time by `__init_subclass__`). It provides a 2–5 word description used to build the `find_tools` index.

## How tools are loaded on a turn

Two loading tiers stack on each ACT iteration and are de-duplicated first-seen:

1. **Unconditional** — every ability in the processor's `ALWAYS_AVAILABLE` list. Always present.
2. **Dynamic (discoverable)** — abilities surfaced this turn via the `find_tools` ability's semantic search, gated to the processor's `DISCOVERABLE` list. Every non-innate first-party ability is reachable exclusively through this path.

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

`channel` is the current conversation channel, `params` are the LLM-extracted arguments validated against `INPUT_SCHEMA`, and `telemetry` carries flattened client context (location, time, locale — fields may be null). The return value is dispatched through `ToolRenderAndRecordService` and tag-formatted by `services/innate_skills/_tag.py`. SUMMARY + EXAMPLES drive the semantic search row that `find_tools` matches on; SUMMARY is the most important field — it determines whether `find_tools` surfaces this ability. SEARCH_TOOLTIP provides a compact label for the `find_tools` index. Both `find_tools` and `find_skills` inherit from `SearchableAbility` (`abilities/_search.py`) which provides the shared vec+FTS5 RRF fusion search infrastructure.

After registering the ability, wire it into the appropriate config(s). For a discoverable ability, add its `NAME` to `DEFAULT_DISCOVERABLE` in `backend/configs/channels.py`. For an always-available ability, add it to `DEFAULT_ALWAYS_AVAILABLE` instead. If a specific channel should not see the ability, add the name to that config's `blocked` frozenset. Then regenerate `abilities.sqlite` via `python -m utils.build_ability_db` so the embedded index is up to date; CI's drift check fails the build if `abilities_sha.json` does not match.

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
