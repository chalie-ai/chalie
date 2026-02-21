# Tools System

Tools extend Chalie's capabilities by allowing sandboxed execution of external code. Tools are containerized, versioned, and can be triggered on-demand or on a schedule.

## Overview

The tools system provides:
- **Sandboxing**: Each tool runs in an isolated Docker container with resource constraints
- **Configuration Management**: Per-tool secrets and credentials stored in PostgreSQL (encrypted)
- **Semantic Matching**: Tool relevance determined via embedding-based similarity, not regex patterns
- **Safety Limits**: Timeouts (default 9s), memory limits (256MB), network isolation, no privilege escalation
- **Audit Trail**: All tool invocations logged to procedural memory with success/failure and execution time

## Architecture

### Components

**Tool Registry Service**
- Singleton that discovers and validates tools from `backend/tools/` directory
- Loads manifest.json and builds Docker images at startup
- Dispatches tool invocations via ToolContainerService
- Logs outcomes for feedback/learning

**Tool Container Service**
- Manages Docker lifecycle: building images, running containers
- Enforces sandbox constraints: memory limits, network isolation, read-only filesystem (by default)
- Handles timeouts, captures stdout/stderr, parses JSON output

**Tool Config Service**
- PostgreSQL backend for per-tool configuration
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

## Creating a Tool

Each tool is a subdirectory in `backend/tools/` with three required files:

### Tool Contract (Formalized JSON Interface)

All tools implement a unified contract: **base64-encoded JSON in → JSON out**.

#### Input Payload (from framework)

The framework sends this to the tool container as a base64-encoded JSON string (CMD arg):

```json
{
  "params": {
    "query": "user's search query",
    "location": "optional param value"
  },
  "settings": {
    "api_key": "abc123",
    "endpoint": "https://api.example.com"
  },
  "telemetry": {
    "lat": 35.8762,
    "lon": 14.5366,
    "city": "Valletta",
    "country": "Malta",
    "time": "2026-02-20T16:54:01Z",
    "locale": "en-MT",
    "language": "en-US"
  }
}
```

- **`params`**: LLM-extracted parameters from the user's intent (matches manifest `parameters` schema)
- **`settings`**: Tool-specific config from the database (API keys, endpoints, etc.)
- **`telemetry`**: Flattened client context (always present, fields may be null)

#### Output Format (from tool)

The tool **must** write this JSON to stdout (and only stdout):

```json
{
  "text": "Human-readable result text. Optional.",
  "html": "<div style=\"...\">Inline HTML card. Optional.</div>",
  "title": "Dynamic card title (optional, overrides manifest title)",
  "error": "Error message string (if operation failed)"
}
```

- **`text`**: Plain text result. If manifest `output.synthesize: true`, Chalie rewrites this in its own voice.
- **`html`**: HTML fragment for UI card display. **Rules:**
  - **Inline CSS only** — use `style="..."` attributes, no `<style>` blocks or external CSS
  - **No JavaScript** — no `<script>` tags, no event handlers (`onclick`, `onerror`, etc.), no `javascript:` URIs
  - **Fragment only** — no `<html>`, `<head>`, `<body>` tags. Must be self-contained.
  - **No dangerous tags** — no `<iframe>`, `<form>`, `<input>`, `<object>`, `<embed>`, `<base>`
  - Backend enforces strict sanitization before sending to frontend.
- **`title`**: Optional dynamic title to override the manifest card title
- **`error`**: If present, triggers fallback behavior and skips text/html processing

### 1. manifest.json

Required fields:
```json
{
  "name": "tool_name",
  "description": "Human-readable description for Chalie to understand what this tool does",
  "version": "1.0.0",
  "category": "search|calculation|memory|integration|utility|context|research|communication",

  "trigger": {
    "type": "on_demand|cron|webhook"
  },

  "parameters": {
    "param_name": {
      "type": "string|integer|float|boolean",
      "description": "What this parameter does",
      "required": true,
      "default": null
    }
  },

  "returns": {
    "text": { "type": "string" },
    "html": { "type": "string" }
  },

  "output": {
    "synthesize": true,
    "ephemeral": false,
    "card": {
      "enabled": true,
      "title": "Card Title",
      "accent_color": "#4a90d4",
      "background_color": "rgba(74, 144, 212, 0.10)"
    }
  }
}
```

**Trigger Types:**

- `"on_demand"` — Called when relevant during ACT mode (default)
- `"cron"` — Runs on schedule, results enqueued as prompts
  - Requires `"schedule"` (simple cron: `*/30` = every 30 minutes)
  - Requires `"prompt"` (template string, tool output appended)
- `"webhook"` — Not currently implemented

**Output Configuration:**

The `output` section controls how the tool's result is displayed:

```json
{
  "output": {
    "synthesize": true,
    "ephemeral": false,
    "card": {
      "enabled": true,
      "title": "Weather in {{location}}",
      "accent_color": "#4a90d4",
      "background_color": "rgba(74, 144, 212, 0.10)"
    }
  }
}
```

- **`synthesize`**: If `true`, the framework rewrites `text` in Chalie's voice. If `false`, `text` is hidden.
- **`ephemeral`**: If `true`, the tool's output is never assimilated into episodic memory and is excluded from action-completion verification. Use for tools whose output is transient by nature (e.g., current weather). Defaults to `false`.
- **`card.enabled`**: If `true`, the framework renders the `html` field as a UI card.
- **`card.title`**: Default card title (can be overridden by tool's `title` in output JSON)
- **`card.accent_color`**: Accent color for the card (CSS color string)
- **`card.background_color`**: Background color for the card (CSS color string)

**Optional Fields:**

```json
{
  "icon": "fa-star",
  "config_schema": {
    "api_key": {
      "description": "Your API key",
      "secret": true,
      "required": true
    },
    "endpoint": {
      "description": "API endpoint URL",
      "secret": false,
      "default": "https://api.example.com"
    }
  },
  "constraints": {
    "timeout_seconds": 9,
    "cost_budget": 1000
  },
  "sandbox": {
    "memory": "512m",
    "network": "bridge|none|host",
    "writable": false
  },
  "notification": {
    "default_enabled": false
  }
}
```

### 2. Dockerfile

Must be a valid Dockerfile that:
- Accepts base64-encoded JSON as command argument
- Outputs JSON to stdout on success (containing `text`, `html`, `title`, and/or `error` fields)
- Exits non-zero with error text on stderr for failures

Example (Python):
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install -q requests

ENTRYPOINT ["python", "-u", "runner.py"]
```

Example (Bash):
```dockerfile
FROM alpine:3.19
RUN apk add --no-cache bash jq
WORKDIR /tool
COPY runner.sh .
RUN chmod +x runner.sh
ENTRYPOINT ["bash", "runner.sh"]
```

### 3. runner.py or runner.sh

The tool script receives the formalized payload and must return the formalized output.

**Python example:**
```python
#!/usr/bin/env python3
import json
import base64
import sys

# Decode base64 payload from command arg
payload = json.loads(base64.b64decode(sys.argv[1]).decode())

params = payload.get("params", {})    # user-provided parameters
settings = payload.get("settings", {}) # stored tool config (API keys, etc.)
telemetry = payload.get("telemetry", {}) # client context (lat, lon, city, etc.)

try:
    # Your tool logic here
    result_data = fetch_data(params, settings, telemetry)

    # Format output with text and optional HTML
    output = {
        "text": f"Weather: {result_data['temp']}°C and {result_data['condition']}",
        "html": f'<div style="padding:16px"><div style="font-size:2rem">{result_data["temp"]}°C</div></div>'
    }
    print(json.dumps(output))
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
```

**Bash example (canonical reference):**
```bash
#!/usr/bin/env bash
set -euo pipefail

# Decode base64 payload
PAYLOAD=$(echo "$1" | base64 -d)

# Extract fields using jq
NAME=$(echo "$PAYLOAD" | jq -r '.params.name // "World"')
CITY=$(echo "$PAYLOAD" | jq -r '.telemetry.city // ""')

# Compose text and HTML
TEXT="Hello, $NAME!"
HTML="<div style=\"padding:16px;font-family:sans-serif\"><div style=\"font-size:1.4rem\">Hello, $NAME!</div></div>"

# Output formalized contract JSON
jq -n \
  --arg text "$TEXT" \
  --arg html "$HTML" \
  '{"text": $text, "html": $html}'
```

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
5. **Sandbox Execution** — ToolContainerService runs Docker container with timeout
6. **Output Sanitization** — Result stripped of action-like patterns, truncated to 3000 chars
7. **Memory Logging** — Outcome (success/failure, execution time) logged to procedural memory
8. **Integration** — Tool output wrapped in `[TOOL:name]...[/TOOL]` markers and included in LLM context

### Tool Status

Tools have three status values (from API `/tools` endpoint):

- **"system"** — Built-in tool with no configuration required
- **"available"** — Tool discovered but not yet configured (missing required secrets)
- **"connected"** — Tool fully configured and ready to use

## Safety & Constraints

### Sandboxing

Every tool container runs with:
- **Memory limit** (default: 256MB, configurable in manifest `sandbox.memory`)
- **CPU shares** (fair scheduling)
- **Network mode** (default: bridge, can be isolated with `none`)
- **Capabilities dropped** (no CAP_SYS_ADMIN, etc.)
- **No privilege escalation** (no-new-privileges flag)
- **PID limit** (max 64 processes)
- **Read-only filesystem** (by default, unless `sandbox.writable: true`)

### Timeouts

- **Default timeout**: 9 seconds
- **Configurable** per tool in `constraints.timeout_seconds`
- Exceeded timeouts logged as failures with `-0.2` reward in procedural memory

### Cost Budgets

Optional per-tool budget tracking (if tool returns `budget_remaining` field):
- Budget info included in tool output metadata
- Useful for API-based tools (e.g., search engines with rate limits)

### Output Sanitization

Tool output is sanitized before integration:
- Removes action-like patterns: `{...}`, function calls, ACTION: keywords
- Prevents tool output from instructing Chalie to take unintended actions
- Truncated to 3000 characters max

## Tool Development Checklist

When creating a new tool, ensure:

- [ ] **Tool directory exists**: `backend/tools/tool_name/`
- [ ] **manifest.json is valid**:
  - [ ] Required fields: `name`, `description`, `version`, `trigger`, `parameters`, `returns`
  - [ ] `output` section with `synthesize` and `card` config
  - [ ] `trigger.type` is one of: `on_demand`, `cron`, `webhook`
  - [ ] Run `python -m json.tool manifest.json` to validate JSON syntax
- [ ] **Dockerfile exists and builds**:
  - [ ] `docker build -t test-tool .` succeeds
  - [ ] Entrypoint is correct (e.g., `["bash", "runner.sh"]` or `["python", "runner.py"]`)
- [ ] **runner.py/runner.sh implements the contract**:
  - [ ] Decodes base64 payload from `sys.argv[1]`
  - [ ] Extracts `params`, `settings`, `telemetry` from payload
  - [ ] Returns JSON with `text`, `html`, `title`, and/or `error` fields
  - [ ] Uses `jq` or JSON library to avoid shell injection
  - [ ] HTML uses only **inline styles** (no `<style>` blocks, no `<script>` tags)
- [ ] **Test locally**:
  ```bash
  PAYLOAD='{"params":{"name":"Test"},"settings":{},"telemetry":{"city":"Malta","country":"Malta"}}'
  ENCODED=$(echo $PAYLOAD | base64)
  docker run --rm <image> "$ENCODED"
  ```
  - Output should be valid JSON with `text` and/or `html` fields
- [ ] **No external dependencies on framework internals**: Tool should work standalone
- [ ] **Error handling**: Tool exits with `{"error": "reason"}` on failure
- [ ] **HTML is safe**: No JavaScript, no external stylesheets, no form inputs

## Tool Output Formats

All tools follow the **formalized contract** defined above:

### Success Response

```json
{
  "text": "Human-readable plain text result (optional)",
  "html": "<div style=\"...\">Inline HTML fragment (optional)</div>",
  "title": "Card title override (optional)"
}
```

### Error Response

```json
{
  "error": "Human-readable error message"
}
```

The framework then:
- If `synthesize: true`: Rewrites `text` in Chalie's voice and includes it in chat
- If `synthesize: false`: Hides `text` and shows only the card (if `card.enabled: true`)
- If `card.enabled: true`: Renders `html` as a UI card with metadata from `card` config

## Example: Weather Tool

Complete example demonstrating the formalized contract.

Directory structure:
```
backend/tools/tool_example/
├── manifest.json
├── Dockerfile
└── runner.sh
```

**manifest.json:**
```json
{
  "name": "tool_example",
  "description": "Example tool demonstrating the formalized tool contract",
  "version": "1.0",
  "category": "utility",
  "icon": "fa-star",
  "trigger": { "type": "on_demand" },
  "parameters": {
    "name": {
      "type": "string",
      "required": true,
      "description": "Name to greet"
    }
  },
  "returns": {
    "text": { "type": "string" },
    "html": { "type": "string" }
  },
  "output": {
    "synthesize": false,
    "card": {
      "enabled": true,
      "title": "Hello",
      "accent_color": "#6c63ff",
      "background_color": "rgba(108, 99, 255, 0.10)"
    }
  },
  "constraints": { "timeout_seconds": 10 },
  "sandbox": { "network": "none", "memory": "64m" }
}
```

**Dockerfile:**
```dockerfile
FROM alpine:3.19
RUN apk add --no-cache bash jq
WORKDIR /tool
COPY runner.sh .
RUN chmod +x runner.sh
ENTRYPOINT ["bash", "runner.sh"]
```

**runner.sh:**
```bash
#!/usr/bin/env bash
set -euo pipefail

# Decode base64 payload from command arg
PAYLOAD=$(echo "$1" | base64 -d)

# Extract fields using jq
NAME=$(echo "$PAYLOAD" | jq -r '.params.name // "World"')
CITY=$(echo "$PAYLOAD" | jq -r '.telemetry.city // ""')
TIME=$(echo "$PAYLOAD" | jq -r '.telemetry.time // ""')

# Compose text output
TEXT="Hello, $NAME!"
[ -n "$CITY" ] && TEXT="$TEXT You're in $CITY."

# Compose HTML card — inline CSS only, no scripts, fragment only
HTML=$(jq -rn \
  --arg name "$NAME" \
  --arg city "$CITY" \
  --arg time "$TIME" \
  '"<div style=\"padding:16px;font-family:sans-serif\">
     <div style=\"font-size:1.4rem;font-weight:600;margin-bottom:8px\">Hello, \($name)!</div>
     " + (if $city != "" then "<div style=\"color:#666;margin-bottom:4px\">📍 \($city)</div>" else "" end) + "
     " + (if $time != "" then "<div style=\"color:#999;font-size:0.85rem\">\($time)</div>" else "" end) + "
   </div>"'
)

# Output formalized contract JSON to stdout
jq -n \
  --arg text "$TEXT" \
  --arg html "$HTML" \
  '{"text": $text, "html": $html}'
```

**Test:**
```bash
PAYLOAD='{"params":{"name":"Dylan"},"settings":{},"telemetry":{"city":"Valletta","country":"Malta","time":"2026-02-20T17:00:00Z","lat":35.8762,"lon":14.5366}}'
ENCODED=$(echo "$PAYLOAD" | base64)
docker run --rm chalie-tool-tool_example:1.0 "$ENCODED"
# Output: {"text": "Hello, Dylan! You're in Valletta.", "html": "<div style=\"...\">...</div>"}
```

## Troubleshooting

### Tool Not Appearing in List

1. Check Docker is running: `docker ps`
2. Check tool directory exists: `backend/tools/tool_name/`
3. Check manifest.json is valid JSON: `python -m json.tool manifest.json`
4. Check Dockerfile exists: `ls backend/tools/tool_name/Dockerfile`
5. View logs: `docker-compose logs -f backend | grep TOOL`

### "Tool not found" Error

Tool name in manifest must match directory name exactly (case-sensitive).

### Configuration Not Being Used

1. Verify config is set: `curl http://localhost:8080/tools/my_tool/config`
2. Test configuration: `curl -X POST http://localhost:8080/tools/my_tool/test`
3. Check required keys are present (marked with `"required": true`)

### Tool Timeout

1. Increase timeout in manifest: `"constraints": {"timeout_seconds": 30}`
2. Optimize tool code (database queries, API calls, etc.)
3. Check Docker resource limits (especially memory)

### "Timed out after 9s" Error

Container exceeded timeout. Options:
1. Increase `timeout_seconds` in manifest
2. Optimize tool code
3. Add caching if tool does expensive computation

## Advanced

### Custom Sandbox Configuration

In manifest.json:
```json
{
  "sandbox": {
    "memory": "1g",
    "network": "none",
    "writable": true
  }
}
```

- `memory`: Docker memory limit (e.g., `256m`, `1g`)
- `network`: `bridge` (default), `none` (isolated), `host` (access host network)
- `writable`: `false` (read-only) or `true` (can write to `/tmp`)

### Cron-Triggered Tools

For scheduled tools (e.g., reminder checks, daily digests):

**manifest.json:**
```json
{
  "trigger": {
    "type": "cron",
    "schedule": "*/30",
    "prompt": "Check for any overdue reminders and summarize them:"
  }
}
```

- `schedule`: Simple cron expression. `*/30` = every 30 minutes
- `prompt`: Template string. Tool output is appended before enqueueing on prompt-queue

### Webhook Tools (Not Yet Implemented)

Placeholder for future external event triggers (e.g., email received, calendar event).

## Safety Guardrails

- **Kill Switch**: Set `tools_enabled: false` in config to disable all tools
- **Single Authority**: Procedural memory (reward signal) is single authority for tool retraining
- **No Skill Registration**: Tools fixed at startup (no runtime registration)
- **Data Scope**: All tool invocations scoped to topic (no cross-topic leakage)
- **Audit Trail**: Every invocation logged with topic, success/failure, execution time

