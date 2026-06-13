# Default Tools

Chalie ships **33 LLM-callable tools**: 3 always available on every turn, and 30 discoverable via `find_tools`. A further 5 abilities are framework-internal (listed at the bottom). How the tiers work is covered in [09-TOOLS.md](09-TOOLS.md).

## Always Available

| Tool | What it does |
|---|---|
| `find_tools` | Activates discoverable tools for the current turn, by name or natural-language search |
| `find_skills` | Surfaces curated and user-created step-by-step playbooks for complex tasks |
| `memory` | Stores, recalls, reflects on, and forgets facts about the user |

## Discoverable

| Tool | What it does |
|---|---|
| `bash` | Runs shell commands, with risk classification and destructive-pattern blocking |
| `calendar` | Lists, creates, updates, and deletes calendar events (CalDAV via the mail connection) |
| `chalie_docs` | Fetches Chalie's own documentation (basics, tools, releases, codebase) |
| `code_eval` | Executes Python in a restricted sandbox (no filesystem, no network, 10-min cap) |
| `contacts` | Searches and looks up contacts from the local people index |
| `document` | Searches, lists, views, creates, deletes, and restores ingested documents |
| `email` | Searches, reads, drafts, sends, replies to, and forwards email (IMAP/SMTP) |
| `file_permissions` | Changes permissions on a file or directory (octal, symbolic, or intent keywords) |
| `file_write` | Writes content to an absolute path (overwrites require a prior `read`) |
| `home` | Controls smart-home devices through Home Assistant |
| `list` | Manages persistent checklists (create, add, check, remove, rename, …) |
| `mcp_manager` | Connects and manages external MCP tool servers |
| `news` | Searches news across Google News and curated RSS categories |
| `place` | Saves and looks up named locations (home, work, gym, …) |
| `programming_docs_search` | Searches official docs for 12 languages and 11 frameworks |
| `read` | Reads a URL or local file and returns the extracted text |
| `review_tool_calls` | Replays raw tool-call records around a given timestamp |
| `review_transcript` | Reads back conversation transcript rows around a given timestamp |
| `schedule` | Creates, lists, searches, and cancels reminders and scheduled tasks |
| `search_files` | Cross-platform file search — `glob` by filename, `grep` by content |
| `skill_builder` | Creates, edits, and deletes user-defined skill playbooks |
| `timer` | Starts an ephemeral countdown card in the chat UI (client-side only) |
| `ubiquiti` | Monitors and controls a UniFi network |
| `vision` | Reads image contents via the vision provider (OCR fallback) — delegate, user-facing channels only |
| `weather` | Current conditions and tomorrow's forecast (Open-Meteo / wttr.in, no API key) |
| `web_browse` | Delegates an interactive browsing task to a focused browser agent |
| `web_download` | Streams a file from a URL to a temp path (100 MB cap) |
| `web_search` | Delegates a research task to a focused search agent |
| `browser` | Step-by-step page driving (Playwright) — *only reachable inside the `web_browse` delegate* |
| `search` | Multi-provider search: Wikipedia, GitHub, Reddit, arXiv, Stack Exchange, … — *only reachable inside the `web_search` delegate* |

Tools that need an external connection (`email`, `calendar`, `contacts`, `home`, `ubiquiti`) report a `not-connected` error with setup guidance until their capability is configured in Brain.

## Sample Outputs

What the model actually receives — the dispatcher wraps every result in the same envelope:

**`weather`** (also renders a rich card in the chat UI):

```
[weather(status=success)]
{"location": "Valletta, MT", "condition": "Partly cloudy", "temperature_c": 24.1,
 "feels_like_c": 25.0, "humidity_pct": 64, "wind_kmh": 19.0,
 "forecast_tomorrow_condition": "Light rain", "forecast_tomorrow_max_c": 23.0,
 "sunrise": "2026-06-12T05:46", "sunset": "2026-06-12T20:21",
 "hourly": [{"hour": 10, "temp_c": 24, "code": 2}, ...]}
[end:weather]
```

**`memory` recall:**

```
[memory(status=success, action=recall, results=3)]
{"results": [
  {"id": "residence", "content": "Lives in Valletta", "score": 0.91, "kind": "user_specific"},
  {"id": "partner", "content": "Partner's name is Sarah", "score": 0.74, "kind": "user_specific"},
  {"id": "food_and_drink", "content": "Loves pastizzi", "score": 0.55, "kind": "user_specific"}]}
[end:memory]
```

**`search_files` error** (errors carry a stable code plus recovery signals):

```
[search_files(status=error, code=invalid-regex)]
Invalid regex 'foo(': missing ), unterminated subpattern at position 3
hint: escape the special characters or pass a valid Python regex.
[end:search_files]
```

## Internal Abilities

These exist as abilities so they ride the same dispatch, audit, and rendering pipeline, but the model can never discover or call them directly:

| Ability | Fired by |
|---|---|
| `thinking` | The framework, at turn zero when the deliberation gate scores *high* |
| `chat_history_compactor` | The ACT loop, when the request outgrows the context window |
| `tool_chain_compactor` | Same — compacts the current turn's tool trail |
| `save_pattern` / `save_graph` | The background pattern-match and geo passes only (budget-capped per turn) |

(`skill_manager` is a system-channel variant of `skill_builder` used by the background skill-suggestion pass.)
