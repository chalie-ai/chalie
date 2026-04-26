# Tools System

Three capability tiers exist in Chalie, each with a different scope and lifecycle.

**Innate skills** are core cognitive capabilities — memory, schedule, list, goal_pursuit, document, read, find_tools, rich_render, and review_tool_calls. They have direct access to Chalie's services and memory. **All innate skills are always loaded on every user turn** — there is no mode-gated tool tier.

**First-party tools** are shipped with Chalie. Each is a simple Python module invoked directly in-process. They handle things the LLM cannot do alone: search, news, live weather, sandboxed code execution, and more. See [14-DEFAULT-TOOLS.md](14-DEFAULT-TOOLS.md) for the current set.

**Interface tools** are capabilities exposed by external applications that have paired with Chalie via the interface protocol. They extend what Chalie can act on without being committed to this repo. See [15-INTERFACES.md](15-INTERFACES.md).

## How tools are loaded on a user turn

Two loading tiers stack on each ACT iteration and are de-duplicated first-seen:

1. **Unconditional** — every innate skill. Always present.
2. **Dynamic (discoverable)** — tools surfaced this turn via the `find_tools` innate skill's semantic search. All external (first-party + interface) tools are reachable exclusively through this path.

`ModeGateService` runs once per user turn but **does not gate tool availability**. It classifies the turn along eight independent cognitive intents (`research`, `coding`, `brainstorm`, `analyze`, `plan`, `write`, `math`, `converse`) using a small ONNX multi-label head; per-mode state follows an asymmetric EMA (fire snaps up, miss decays by 0.75 per turn). State persists across turns in MemoryStore under `mode_gate:state` and is cleared by `/privacy/delete-all`. The active mode set powers prompt-steering directives in `UserMessageProcessor` (long-summary swap on `converse`, brainstorm/research/analyze suffixes appended to the system prompt) and is reserved for future mode-driven features.

Observability: every user turn emits one `[MODE-GATE]` INFO log line with the full probability vector, state before/after, and the active mode set. This is the grep-friendly anchor for nightly scenarios.

## Tool status

Three status values appear in the tools list:

| Status | Meaning |
|---|---|
| `system` | Built-in, no configuration required |
| `available` | Discovered but not yet configured (missing required secrets) |
| `connected` | Fully configured and ready to use |

## Adding a first-party tool

A first-party tool is a Python module that exposes a single function:

```python
def execute(topic: str, params: dict, config: dict = None, telemetry: dict = None) -> dict
```

`topic` is the current conversation topic, `params` are the LLM-extracted arguments, `config` contains any stored secrets or endpoints, and `telemetry` carries flattened client context (location, time, locale — fields may be null). The return dict can include a `text` key for a plain-text result, an `html` key for a UI card fragment, and an `error` key that signals failure and suppresses the other fields.

Alongside the module, declare the tool's metadata: a description that the semantic search will embed, a parameter schema the LLM uses to extract arguments, and any constraints. The description is the most important field — it determines when `find_tools` surfaces this tool.

## Configuration

Tools that require API keys or custom endpoints declare their required config keys in their metadata. Configure them through the Brain UI (Settings > Tools) or via the REST API — see the API reference for endpoints. Stored secrets are masked in all API responses.

## Safety constraints

- Tool invocations time out. Exceeded timeouts are logged as failures.
- Output is sanitized before it enters LLM context: action-like patterns are stripped and the result is truncated.
- Every invocation is written to an audit trail with the topic, outcome, and execution time.
- A global kill switch can disable all tools if needed.
