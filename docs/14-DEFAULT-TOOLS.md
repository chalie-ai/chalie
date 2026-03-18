# Embodiments

Embodiments are Chalie's built-in capabilities — always installed, always enabled, internally managed. They are not user-configurable tools; they are core abilities that Chalie uses autonomously through the ACT loop.

Embodiments live in separate repositories but are bundled with every Chalie instance. On startup, Chalie downloads or upgrades each embodiment to the version pinned in `backend/configs/embodiment_library.json`.

## Current Embodiments

### Weather

| | |
|---|---|
| **Repo** | [chalie-ai/chalie-tool-weather](https://github.com/chalie-ai/chalie-tool-weather) |
| **Trigger** | On-demand |

Fetches current weather conditions and tomorrow's forecast using [Open-Meteo](https://open-meteo.com/) and [wttr.in](https://wttr.in/). No API key required.

### Reddit

| | |
|---|---|
| **Repo** | [chalie-ai/reddit-tool](https://github.com/chalie-ai/reddit-tool) |
| **Trigger** | On-demand |

Searches Reddit for community discussions, opinions, recommendations, and troubleshooting threads. No API key required.

### Web Search

| | |
|---|---|
| **Repo** | [chalie-ai/chalie-tool-web-search](https://github.com/chalie-ai/chalie-tool-web-search) |
| **Trigger** | On-demand |

Searches the web via DuckDuckGo. Privacy-focused, no API key required.

### Code Eval

| | |
|---|---|
| **Trigger** | On-demand |

Executes Python snippets in a restricted sandbox to verify formulas, test algorithms, or compute results precisely. Built-in (no external repo).

### World Clock

| | |
|---|---|
| **Repo** | [chalie-ai/world-clock-interface](https://github.com/chalie-ai/world-clock-interface) |
| **Trigger** | Cron (15-minute interval) |

Populates world state with current time-of-day phase, sunrise, sunset, solar noon, and day length derived from client telemetry. No API key required.

---

## Versioning

Each embodiment is pinned to a specific version (git tag) in `backend/configs/embodiment_library.json`. On startup, Chalie compares the installed version against the pinned version and downloads the correct release if they differ.

Versions are updated in `embodiment_library.json` as part of Chalie releases — users do not manage embodiment versions manually.

## Adding an Embodiment

Add an entry to `backend/configs/embodiment_library.json`:

```json
{
  "name": "my_tool",
  "title": "My Tool",
  "icon": "fa-star",
  "repo": "chalie-ai/chalie-tool-my-tool",
  "summary": "One-line description",
  "trigger": "on_demand",
  "version": "v1.0.0"
}
```

Requirements:
- Must have a `runner.py` (subprocess entry point)
- Must have a `manifest.json` with valid `name`, `description`, `trigger`, `parameters`, `returns`
- Must be hosted on GitHub with a tagged release matching the `version` field
