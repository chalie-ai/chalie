## `find_tools` — External Tool Discovery

Search for external capabilities (tools and interface actions) when you need to interact with the outside world.

### Actions

**search** — Find tools by describing what you need:
```json
{"type": "find_tools", "action": "search", "query": "send an email"}
```
Parameters:
- `query` (required): Natural language description of the capability you need
- `limit` (optional): Max results (default 5, max 10)

**details** — Get full invocation guide for a specific tool:
```json
{"type": "find_tools", "action": "details", "tool_name": "weather"}
```
Parameters:
- `tool_name` (required): Name of the tool to inspect

### When to use
- You need to do something external (web search, send message, check calendar, etc.)
- You're unsure which tool handles a specific task
- You need parameter details before invoking a tool

### Workflow
1. `search` with what you need → get matching tools with relevance scores
2. Optionally `details` on the best match → get parameters and invocation format
3. Invoke the tool directly: `{"type": "<tool_name>", ...params}`
