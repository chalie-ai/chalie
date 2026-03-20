# Tools System

Tools are one of three capability tiers in Chalie:

1. **Innate Skills** (`backend/services/innate_skills/`) — Built-in cognitive capabilities with direct access to Chalie's services, database, and memory. Always injected into the LLM context. Examples: `recall`, `memorize`, `schedule`, `list`, `document`, `find_tools`, `reflect`.

2. **Tools** (`backend/tools/`) — First-party capabilities committed to the repo. Run as trusted subprocesses with zero access to Chalie internals. Declared in `TOOL_LIBRARY` and loaded at startup. This document covers these.

3. **Interfaces** (`frontend/_interfaces/`) — External third-party integrations that pair with Chalie via the interface protocol. Can expose capabilities and update world state. See [15-INTERFACES.md](15-INTERFACES.md).

## Overview

The tools system provides:
- **Subprocess execution**: All tools run as Python subprocesses (same IPC contract: base64 JSON in, JSON out)
- **Interface tools**: External applications can pair with Chalie and expose tool capabilities via the interface protocol
- **Configuration Management**: Per-tool secrets and credentials stored in SQLite (encrypted)
- **Semantic Matching**: Tool relevance determined via embedding-based similarity, not regex patterns
- **Safety Limits**: Timeouts (default 9s), no privilege escalation
- **Audit Trail**: All tool invocations logged to procedural memory with success/failure and execution time

## Architecture

### Components

**Tool Registry Service**
- Singleton that loads tools declared in `TOOL_LIBRARY` (`backend/services/tool_registry_service.py`)
- Validates manifest.json at startup
- Dispatches invocations via `ToolSubprocessService`
- Logs outcomes for feedback/learning

**Tool Subprocess Service**
- Runs tools as Python subprocesses (same OS user as Chalie)
- IPC contract: base64-encoded JSON in (CMD arg) → JSON out (stdout)
- Supports `run()` for single-shot and `run_interactive()` for bidirectional dialog

**Tool Config Service**
- SQLite backend for per-tool configuration
- Stores API keys, credentials, and parameters as key-value pairs
- Secrets are masked in API responses (shows `***` instead of actual value)

**Tool Relevance Service**
- Embedding-based semantic matching between user intent and available tools
- Caches embeddings for performance (disk-persisted)
- Replaces regex-based tool hints with cosine similarity scoring
- Threshold-based filtering (default: 0.35 relevance minimum)

**REST API** (`backend/api/tools.py`)
- List tools with status and config schema
- Get/set/delete tool configuration
- Test tool configuration completeness

## IPC Contract

All tools implement a unified contract: **base64-encoded JSON in → JSON out**.

**Input** (from framework → tool subprocess as base64 CMD arg):

| Key | Contents |
|-----|----------|
| `params` | LLM-extracted parameters matching manifest schema |
| `settings` | Per-tool config from database (API keys, endpoints) |
| `telemetry` | Flattened client context (location, time, locale — fields may be null) |

**Output** (tool → stdout as JSON):

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
5. **Subprocess Execution** — ToolSubprocessService runs the tool with timeout
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

1. Check the tool is declared in `TOOL_LIBRARY` in `backend/services/tool_registry_service.py`
2. Check tool directory exists: `backend/tools/tool_name/`
3. Check manifest.json is valid JSON: `python -m json.tool manifest.json`
4. Check runner.py or runner.sh exists: `ls backend/tools/tool_name/`
5. View logs in the `python backend/run.py` console output

### "Tool not found" Error

Tool name in manifest must match directory name exactly (case-sensitive).

### Configuration Not Being Used

1. Verify config is set: `curl http://localhost:8080/tools/my_tool/config`
2. Test configuration: `curl -X POST http://localhost:8080/tools/my_tool/test`
3. Check required keys are present (marked with `"required": true`)

### Tool Timeout

1. Increase timeout in manifest: `"constraints": {"timeout_seconds": 30}`
2. Optimize tool code (database queries, API calls, etc.)

### "Timed out after 9s" Error

Tool exceeded timeout. Options:
1. Increase `timeout_seconds` in manifest
2. Optimize tool code
3. Add caching if tool does expensive computation

## Safety Guardrails

- **Kill Switch**: Set `tools_enabled: false` in config to disable all tools
- **Declared Library**: Tools are declared in `TOOL_LIBRARY`, not discovered by scanning
- **Single Authority**: Procedural memory (reward signal) is single authority for tool retraining
- **Data Scope**: All tool invocations scoped to topic (no cross-topic leakage)
- **Audit Trail**: Every invocation logged with topic, success/failure, execution time

