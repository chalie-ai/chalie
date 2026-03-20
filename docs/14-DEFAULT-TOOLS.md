# First-Party Tools

First-party tools are Chalie's built-in capabilities that run as trusted subprocesses. They live in `backend/tools/` and are committed to the main repo. They have zero access to Chalie internals (SQLite, MemoryStore, services) — they receive input via base64 JSON and return JSON results.

Tools are discoverable via the `find_tools` innate skill (semantic search over tool profiles) and available on-demand through the ACT loop.

## Current Tools

### Weather

| | |
|---|---|
| **Path** | `backend/tools/weather/` |
| **Trigger** | On-demand |

Fetches current weather conditions and tomorrow's forecast using [Open-Meteo](https://open-meteo.com/) and [wttr.in](https://wttr.in/). No API key required.

### Web Search

| | |
|---|---|
| **Path** | `backend/tools/web_search/` |
| **Trigger** | On-demand |

Searches the web via DuckDuckGo. Privacy-focused, no API key required.

### Code Eval

| | |
|---|---|
| **Path** | `backend/tools/code_eval/` |
| **Trigger** | On-demand |

Executes Python snippets in a restricted sandbox to verify formulas, test algorithms, or compute results precisely.

### Programming Docs Search

| | |
|---|---|
| **Path** | `backend/tools/programming_docs_search/` |
| **Trigger** | On-demand |

Searches and reads official documentation for 12 languages and 11 major frameworks. Languages: PHP, Python, JavaScript/TypeScript, Go, Rust, Java, Ruby, C#, Dart, C/C++, Bash, SQL. Frameworks: Django, Flask, NumPy, Pandas, Laravel, Node.js, React, Vue, Spring, Rails, Flutter. No API key required.

---

## Adding a Tool

Create a directory under `backend/tools/` with:

- `runner.py` — Subprocess entry point (base64 JSON in, JSON out)
- `handler.py` — Core execution logic (`execute()` function)
- `manifest.json` — Capability metadata (name, description, trigger, parameters, returns)
- `requirements.txt` — Python dependencies (optional)

The tool will be auto-discovered by `ToolRegistryService` on startup.
