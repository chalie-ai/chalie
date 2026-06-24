# Tools System

Every tool the LLM can call is an **Ability** — a Python class under `backend/abilities/`. The full catalogue is in [14-DEFAULT-TOOLS.md](14-DEFAULT-TOOLS.md).

## Tool Visibility

Which tools a turn can see is governed by a single binary — the `Ability.DISCOVERABLE` flag — plus each channel config's `always_available` list (default in `backend/configs/channels/_common.py`). A tool reaches the model exactly one of two ways:

- **Always available** — pre-injected on every iteration, listed in the channel's `always_available`. The default is just three meta-tools: `find_skills`, `find_tools`, `memory`.
- **Discoverable** — every ability with `DISCOVERABLE = True` (the base default). Never pre-injected; the model activates one mid-turn by calling `find_tools` (by name, or by natural-language search). Discovery is **global** — `find_tools` searches one roster, `AbilityRegistry.discoverable_names()`, not a per-channel slice. This keeps the request small and works with any number of installed tools.

An ability that sets `DISCOVERABLE = False` is absent from that roster, so `find_tools` can never surface it on any channel. It reaches the model **only** by being pinned into a processor's `always_available`. That is how the raw `browser` / `search` / `news` tools stay exclusive to their delegate configs (`WebBrowseConfig` / `WebSearchConfig`), and the pattern-write tools (`save_pattern` / `save_graph`) stay exclusive to the pattern-match processor. Channel isolation is therefore just two facts: whether a tool is `DISCOVERABLE`, and whether the invoking processor carries `find_tools`.

`find_tools` returns a structured result the model can act on:

```
[find_tools(status=success, injected=2, query=email)]
{"injected": [{"name": "email", "summary": "search, send and manage emails"},
              {"name": "contacts", "summary": "look up contacts"}],
 "not_found": []}
[end:find_tools]
```

Search is hybrid (vector KNN + FTS5, reciprocal-rank fusion) over a pre-built index — `backend/abilities/assets/abilities.sqlite` for first-party tools, `data/mcp_tools.sqlite` for connected MCP servers — with a relevance floor and a cap of 5 results per query.

## Anatomy of an Ability

```python
from abilities._ability import Ability
from abilities._result import ToolResult

class WeatherAbility(Ability):
    def get_name(self) -> str:          # the string the LLM calls
        return "weather"

    def get_summary(self) -> str:       # model-facing description AND search text
        return "Live weather lookup by coordinates or city name."

    def get_examples(self) -> list[str]:    # 6-8 phrases; drives semantic search
        return ["what is the weather in Valletta", "is it raining in San Francisco", ...]

    def get_search_tooltip(self) -> str:    # short label in find_tools results
        return "weather lookup"

    def get_parameters(self) -> dict:       # plain JSON Schema for run()'s params
        return {"type": "object", "properties": {...}, "required": [...]}

    def run(self, params: dict) -> ToolResult:
        ...
        return ToolResult.ok(result_dict)
```

- **Registration is automatic.** The registry imports every non-underscore module in `backend/abilities/` and collects `Ability` subclasses — there is no decorator, manifest, or registration call.
- `self.mp` is the invoking MessageProcessor (channel, config); `self.telemetry` carries client context (location, time, locale) and is set just before `run()`.
- Optionally declare `ACTION_REQUIRED: ClassVar[dict]` mapping actions to their required params — the dispatcher rejects incomplete calls *before* the permission gate, with a `missing-params` error the model can self-correct from.
- Metadata getters must return deterministic text when `self.mp is None` (that's how the search index is built offline).

## The Result Contract

`run()` returns a `ToolResult`, built only via two constructors:

```python
ToolResult.ok(body, *, rich=None, **meta)
# body: str (shown verbatim) or dict/list (rendered as compact JSON)
# rich: optional payload for a rich-media card in the chat UI
# meta: flat scalars shown in the envelope's opening tag

ToolResult.err(message, *, code, hint=None, valid=(), **meta)
# code:  stable kebab-case machine code (required)
# hint:  one-line recovery step for the model
# valid: acceptable values, when the model passed an invalid one
```

The ability never formats output — the dispatcher renders the single wire envelope the model sees:

```
[weather(status=success)]
{"location": "Valletta, MT", "condition": "Clear", "temperature_c": 24.1, ...}
[end:weather]

[memory(status=error, code=no-query-or-location, action=recall)]
recall requires either a query or a location.
hint: pass query= to search by topic, or location= to filter by place.
[end:memory]
```

Errors are data for the model, not exceptions: they are returned into the loop so the model can correct itself, and they never crash the turn.

## Dispatch & Permissions

Every call — model-issued, framework seed, or background pass — goes through `ToolDispatcher(mp).dispatch(name, params)`:

1. resolve the ability (an `_mcp_`-prefixed name resolves to the MCP proxy),
2. pre-validate required params (`ACTION_REQUIRED`),
3. classify the action's risk and check it against the **policy gate** — every action is allow / ask / deny per context (Chat, Background, External Agent), editable in Brain → Policies,
4. execute (inline, or on a background thread when the model sets the `async` flag — user channel only),
5. render the envelope and record the call in the `tool_calls` audit trail (retained as long as its transcript turn is — reaped together with the turn by the transcript GC).

## Adding a Tool — Checklist

1. Create `backend/abilities/<name>.py` with an `Ability` subclass: five metadata getters + `run()` returning a `ToolResult`.
2. Leave `DISCOVERABLE = True` (the base default) so the tool joins the global `find_tools` roster — or set `DISCOVERABLE = False` and pin its name into the `always_available` of every channel that should reach it (this is how delegate-only and pattern-write tools are scoped).
3. Rebuild the search index: `python -m utils.build_ability_db` (CI fails on a stale index). Only `DISCOVERABLE = True` abilities are indexed.
4. If the tool needs credentials, register its config keys with `ToolConfigService` so they're configurable from the Brain UI and stored encrypted.

## MCP Tools

Chalie is both an MCP server and an MCP client:

- **Inbound** — `backend/mcp_server/` exposes a `talk_to_chalie` tool so external agents (Claude Code, Cursor, …) can converse with Chalie. See [for_agents/MCP_SETUP.md](for_agents/MCP_SETUP.md).
- **Outbound** — connect remote MCP servers via the `mcp_manager` tool or Brain → MCP. Remote tools appear as `_mcp_<server>_<tool>`, are embedded into the same search index at add time, and dispatch through the exact same pipeline (policy gate, audit trail, result contract) as first-party tools. Connection failures surface as stable codes: `mcp-unreachable`, `mcp-unknown-tool`, `mcp-tool-error`.
