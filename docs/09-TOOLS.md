# Tools System

Three capability tiers exist in Chalie, each with a different scope and lifecycle.

**Innate skills** are core cognitive capabilities — memory, introspect, schedule, list, goal_pursuit, document, read, find_tools, goals, rich_render, and review_tool_calls. They have direct access to Chalie's services and memory. Three of them (`memory`, `find_tools`, `review_tool_calls`) are **cognitive primitives** and are always loaded on every user turn. The remaining eight are **mode-gated** (see below) and only appear in context when the user turn activates a cognitive mode they serve.

**First-party tools** are shipped with Chalie. Each is a simple Python module invoked directly in-process. They handle things the LLM cannot do alone: search, news, live weather, sandboxed code execution, and more. See [14-DEFAULT-TOOLS.md](14-DEFAULT-TOOLS.md) for the current set.

**Interface tools** are capabilities exposed by external applications that have paired with Chalie via the interface protocol. They extend what Chalie can act on without being committed to this repo. See [15-INTERFACES.md](15-INTERFACES.md).

## How tools are loaded on a user turn

Three loading tiers stack on each ACT iteration and are de-duplicated first-seen:

1. **Unconditional** — the three cognitive primitives. Always present.
2. **Conditional (mode-gated)** — tools whose declared `modes` overlap with the user turn's active cognitive intents. Resolved once per turn by `ModeGateService` and cached on the processor instance; subsequent ACT iterations reuse the cached set. Only the `user` channel opts in — DMN, scheduled prompts, goal pursuit, and encoder processors never run the gate.
3. **Dynamic (discoverable)** — tools surfaced this turn via the `find_tools` innate skill's semantic search. External tools that never declare `modes` stay reachable exclusively through this path.

The gate classifies each user turn along eight independent cognitive intents (`research`, `coding`, `brainstorm`, `analyze`, `plan`, `write`, `math`, `converse`) using a small ONNX multi-label head. Per-mode state follows an asymmetric EMA: a fire snaps state up to the classifier probability, a miss decays by 0.75 per turn (a mode stays "warm" for roughly four subsequent turns before dropping below the activation threshold). State persists across turns in MemoryStore under `mode_gate:state` and is cleared by `/privacy/delete-all`.

Which innate skills serve which modes is declared at the top of `backend/services/innate_skills/registry.py` (`SKILL_MODES`). External tools declare the same field on their `TOOL_METADATA` entry — the field is stripped before any schema is handed to the LLM. Tools without a `modes` declaration are zero-regression: they behave exactly as they always did (find-tools-only for externals).

Observability: every user turn emits one `[MODE-GATE]` INFO log line with the full probability vector, state before/after, active modes, and the promoted tool list, plus one `[MODE-GATE-PROMOTE] turn=<uid> tool=<name>` line per promoted tool. These are grep-friendly anchors for nightly scenarios.

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
