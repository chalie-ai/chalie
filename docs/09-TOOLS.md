# Tools System

Three capability tiers exist in Chalie, each with a different scope and lifecycle.

**Innate skills** are core cognitive capabilities — recall, memorize, schedule, find_tools, reflect, and a handful of others. They are always loaded into LLM context and have direct access to Chalie's services and memory.

**First-party tools** are shipped with Chalie. Each is a simple Python module invoked directly in-process. They handle things the LLM cannot do alone: live weather, web search, sandboxed code execution. See [14-DEFAULT-TOOLS.md](14-DEFAULT-TOOLS.md) for the current set.

**Interface tools** are capabilities exposed by external applications that have paired with Chalie via the interface protocol. They extend what Chalie can act on without being committed to this repo. See [15-INTERFACES.md](15-INTERFACES.md).

## How tools are used

The LLM never has the full tool list in context. Instead, when it needs a capability it invokes the `find_tools` innate skill, which runs a semantic search over tool capability profiles and returns the closest matches. The LLM then decides whether to invoke one. The result comes back into context as structured output and the conversation continues. This keeps context lean and makes tool discovery robust to naming variation — matching is by meaning, not keyword.

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
