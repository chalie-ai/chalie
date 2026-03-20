# First-Party Tools

First-party tools are simple callable Python modules in `backend/tools/`. Each exposes a single `execute(topic, params, config, telemetry) -> dict` function and is invoked directly in-process by `ToolRegistryService`.

All tool metadata (descriptions, parameters, constraints) is declared in `backend/services/tool_library_service.py`. Tools are discoverable via the `find_tools` innate skill (semantic search over tool profiles) and available on-demand through the ACT loop.

## Current Tools

### Weather

| | |
|---|---|
| **Module** | `backend/tools/weather.py` |
| **Trigger** | On-demand |

Fetches current weather conditions and tomorrow's forecast using [Open-Meteo](https://open-meteo.com/) and [wttr.in](https://wttr.in/). No API key required.

### Web Search

| | |
|---|---|
| **Module** | `backend/tools/web_search.py` |
| **Trigger** | On-demand |

Searches the web via DuckDuckGo. Privacy-focused, no API key required.

### Code Eval

| | |
|---|---|
| **Module** | `backend/tools/code_eval.py` |
| **Trigger** | On-demand |

Executes Python snippets in a restricted sandbox to verify formulas, test algorithms, or compute results precisely.

### Programming Docs Search

| | |
|---|---|
| **Module** | `backend/tools/programming_docs_search.py` |
| **Trigger** | On-demand |

Searches and reads official documentation for 12 languages and 11 major frameworks. Languages: PHP, Python, JavaScript/TypeScript, Go, Rust, Java, Ruby, C#, Dart, C/C++, Bash, SQL. Frameworks: Django, Flask, NumPy, Pandas, Laravel, Node.js, React, Vue, Spring, Rails, Flutter. No API key required.
