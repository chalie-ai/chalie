# Tools System

Tools are one of three capability tiers in Chalie:

1. **Innate Skills** (`backend/services/innate_skills/`) — Built-in cognitive capabilities with direct access to Chalie's services, database, and memory. Always injected into the LLM context. Examples: `recall`, `memorize`, `schedule`, `list`, `document`, `find_tools`, `reflect`.

2. **Tools** (`backend/tools/`) — First-party capabilities committed to the repo. Simple callable Python modules invoked directly in-process. All metadata declared in `ToolLibraryService`. This document covers these.

3. **Interfaces** (`frontend/_interfaces/`) — External third-party integrations that pair with Chalie via the interface protocol. Can expose capabilities and update world state. See [15-INTERFACES.md](15-INTERFACES.md).

## Overview

The tools system provides:
- **Direct invocation**: First-party tools are called in-process (no subprocess, no IPC)
- **Interface tools**: External applications can pair with Chalie and expose tool capabilities via the interface protocol
- **Configuration Management**: Per-tool secrets and credentials stored in SQLite (encrypted)
- **Semantic Matching**: Tool relevance determined via embedding-based similarity, not regex patterns
- **Audit Trail**: All tool invocations logged to procedural memory with success/failure and execution time

## Architecture

### Components

**Tool Library Service** (`backend/services/tool_library_service.py`)
- Single source of truth for first-party tool metadata and handlers
- All tool descriptions, parameters, and constraints declared in Python (like innate skills)
- Maps tool names to handler functions imported from `backend/tools/*.py`

**Tool Registry Service** (`backend/services/tool_registry_service.py`)
- Singleton that loads tools from ToolLibraryService at startup
- Invokes first-party tools directly in-process
- Routes interface tools via HTTP to paired interfaces
- Logs outcomes for feedback/learning

**Tool Config Service**
- SQLite backend for per-tool configuration
- Stores API keys, credentials, and parameters as key-value pairs
- Secrets are masked in API responses (shows `***` instead of actual value)

**Tool Relevance Service**
- Embedding-based semantic matching between user intent and available tools
- Caches embeddings for performance (disk-persisted)
- Threshold-based filtering (default: 0.35 relevance minimum)

**REST API** (`backend/api/tools.py`)
- List tools with status and config schema
- Get/set/delete tool configuration
- Test tool configuration completeness

## Tool Interface

All first-party tools expose a single function:

```python
def execute(topic: str, params: dict, config: dict = None, telemetry: dict = None) -> dict
```

**Input parameters:**

| Arg | Contents |
|-----|----------|
| `topic` | Current conversation topic |
| `params` | LLM-extracted parameters matching tool schema |
| `config` | Per-tool config from database (API keys, endpoints) |
| `telemetry` | Flattened client context (location, time, locale — fields may be null) |

**Return dict:**

| Key | Description |
|-----|-------------|
| `text` | Plain text result (optional). If `output.synthesize: true`, rewritten in Chalie's voice |
| `html` | HTML fragment for UI card (optional). Inline CSS only, no JS, no dangerous tags |
| `title` | Dynamic card title override (optional) |
| `error` | Error message — if present, triggers fallback and skips text/html |

## Using Tools

### Configure Tool via REST API

1. **List available tools:**
   ```bash
   curl http://localhost:8080/tools \
     -H "Authorization: Bearer YOUR_API_KEY"
   ```

2. **Set configuration (API keys, endpoints):**
   ```bash
   curl -X PUT http://localhost:8080/tools/my_tool/config \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"api_key": "sk-...", "endpoint": "https://..."}'
   ```

3. **Test configuration:**
   ```bash
   curl -X POST http://localhost:8080/tools/my_tool/test \
     -H "Authorization: Bearer YOUR_API_KEY"
   ```
   Returns `{"ok": true, "message": "Configuration looks complete"}` if all required keys are set.

4. **Get configuration (secrets masked):**
   ```bash
   curl http://localhost:8080/tools/my_tool/config \
     -H "Authorization: Bearer YOUR_API_KEY"
   ```

5. **Delete a config key:**
   ```bash
   curl -X DELETE http://localhost:8080/tools/my_tool/config/api_key \
     -H "Authorization: Bearer YOUR_API_KEY"
   ```

### Tool Execution Flow

When user sends a message that matches ACT mode:

1. **Semantic Matching** — Tool Relevance Service embeds user intent, scores against all available tools
2. **Tool Selection** — Mode router picks most relevant tools with relevance > threshold
3. **Parameter Extraction** — LLM extracts parameters from conversation context
4. **Configuration Injection** — ToolConfigService fetches stored API keys/endpoints
5. **Direct Invocation** — Handler called in-process via `ToolLibraryService.get_handler()`
6. **Output Sanitization** — Result stripped of action-like patterns, truncated to 3000 chars
7. **Memory Logging** — Outcome (success/failure, execution time) logged to procedural memory
8. **Integration** — Tool output wrapped in `[TOOL:name]...[/TOOL]` markers and included in LLM context

### Tool Status

Tools have three status values (from API `/tools` endpoint):

- **"system"** — Built-in tool with no configuration required
- **"available"** — Tool discovered but not yet configured (missing required secrets)
- **"connected"** — Tool fully configured and ready to use

## Safety & Constraints

### Timeouts

- **Default timeout**: 9 seconds
- **Configurable** per tool in `constraints.timeout_seconds`
- Exceeded timeouts logged as failures in procedural memory

### Cost Budgets

Optional per-tool budget tracking (if tool returns `budget_remaining` field):
- Budget info included in tool output metadata
- Useful for API-based tools (e.g., search engines with rate limits)

### Output Sanitization

Tool output is sanitized before integration:
- Removes action-like patterns: `{...}`, function calls, ACTION: keywords
- Prevents tool output from instructing Chalie to take unintended actions
- Truncated to 3000 characters max


## Troubleshooting

### Tool Not Appearing in List

1. Check the tool has an entry in `TOOL_HANDLERS` and `TOOL_METADATA` in `backend/services/tool_library_service.py`
2. Check the tool module exists: `backend/tools/tool_name.py`
3. Check the module exposes `execute(topic, params, config, telemetry) -> dict`
4. View logs in the `python backend/run.py` console output

### "Tool not found" Error

Tool name in `TOOL_HANDLERS` must match the name used in `TOOL_METADATA` exactly.

### Configuration Not Being Used

1. Verify config is set: `curl http://localhost:8080/tools/my_tool/config`
2. Test configuration: `curl -X POST http://localhost:8080/tools/my_tool/test`
3. Check required keys are present (marked with `"required": true`)

### Tool Timeout

1. Increase `timeout_seconds` in `TOOL_METADATA` constraints: `"constraints": {"timeout_seconds": 30}`
2. Optimize tool code (database queries, API calls, etc.)

## Safety Guardrails

- **Kill Switch**: Set `tools_enabled: false` in config to disable all tools
- **Declared Library**: Tools are declared in `ToolLibraryService`, not discovered by scanning
- **Single Authority**: Procedural memory (reward signal) is single authority for tool retraining
- **Data Scope**: All tool invocations scoped to topic (no cross-topic leakage)
- **Audit Trail**: Every invocation logged with topic, success/failure, execution time

