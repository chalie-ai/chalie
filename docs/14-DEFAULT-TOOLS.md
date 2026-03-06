# Chalie Default [Tools](09-TOOLS.md) Reference - Auto-install Behavior & Built-in Toolset

This comprehensive guide covers default [tools](09-TOOLS.md), built-in toolset, auto-install behavior, providing essential information for developers and users. For related topics, see: [[Tools](09-TOOLS.md) System Documentation](../docs/09-[TOOLS](09-TOOLS.md).md) | [Documentation Index](../docs/INDEX.md) | [[LLM Providers](02-PROVIDERS-SETUP.md) Configuration](../docs/02-PROVIDERS-SETUP.md)


This comprehensive guide covers Chalie documentation, technical guide, providing essential information for developers and users. For related topics, see: 


Chalie ships with a curated set of **default [tools](09-TOOLS.md)** that are installed automatically on first startup. These [tools](09-TOOLS.md) are listed in `backend/configs/embodiment_library.json` with `"installs_by_default": true`.

Default [tools](09-TOOLS.md) are **trusted** — they run as subprocesses in Chalie's Python environment rather than inside Docker containers. This means they start instantly and don't require Docker to be installed.

## Auto-Install Behavior

On first startup, if a default tool is not present in `backend/[tools](09-TOOLS.md)/`, Chalie fetches the latest release tarball from the tool's GitHub repository and installs it automatically (background thread, non-blocking). Subsequent startups skip [tools](09-TOOLS.md) that are already present.

To opt out of this behavior at install time:

```bash
curl -fsSL https://chalie.ai/install | bash -s -- --disable-default-[tools](09-TOOLS.md)
```

This writes a `backend/data/.no-default-[tools](09-TOOLS.md)` marker file. Chalie will not auto-install default [tools](09-TOOLS.md) as long as this file exists.

> **Note:** Disabling default [tools](09-TOOLS.md) does not prevent you from installing them manually later via the [Tools](09-TOOLS.md) → Catalog UI or the `POST /api/[tools](09-TOOLS.md)/install` endpoint.

---

## Installed by Default

### Weather

| | |
|---|---|
| **Repo** | [chalie-ai/chalie-tool-weather](https://github.com/chalie-ai/chalie-tool-weather) |
| **Category** | Context |
| **Trigger** | On-demand |
| **Trust** | Trusted (subprocess) |
| **Requires API key** | No |

Fetches current weather conditions and tomorrow's forecast using [Open-Meteo](https://open-meteo.com/) (coordinates-based, primary) and [wttr.in](https://wttr.in/) (city name fallback). Both sources are free with no API key required.

**When Chalie uses it:** Any time weather context is relevant — explicit questions ("what's the weather like?"), implied context ("should I bring a jacket?"), or when outdoor conditions matter to an activity ("I'm going for a run").

**Returns:**

| Field | Description |
|---|---|
| `temperature_c` / `temperature_f` | Current temperature |
| `feels_like_c` | Apparent temperature |
| `condition` | Human-readable description (e.g. "Partly cloudy") |
| `humidity_pct` | Relative humidity % |
| `wind_kmh` / `wind_direction` | Wind speed and compass direction |
| `visibility_km` | Visibility |
| `uv_index` | UV index |
| `precip_mm` | Precipitation |
| `is_raining` / `is_hot` / `is_cold` / `is_windy` / `is_clear` / `is_daylight` | Boolean condition flags |
| `forecast_tomorrow_condition` | Tomorrow's condition description |
| `forecast_tomorrow_max_c` / `_min_c` | Tomorrow's temperature range |
| `forecast_tomorrow_precip_chance_pct` | Tomorrow's rain probability |

Results are cached per location for 10 minutes. Location is detected automatically from client telemetry — only pass an explicit `location` parameter when asking about a different place.

---

## Adding More Default [Tools](09-TOOLS.md)

To mark a new tool as a default, add an entry to `backend/configs/embodiment_library.json`:

```json
{
  "name": "my_tool",
  "title": "My Tool",
  "icon": "fa-star",
  "repo": "chalie-ai/chalie-tool-my-tool",
  "summary": "One-line description shown in the catalog",
  "category": "utility",
  "trigger": "on_demand",
  "trust": "trusted",
  "installs_by_default": true
}
```

Requirements for trusted default [tools](09-TOOLS.md):
- Must have a `runner.py` (subprocess entry point) — no `Dockerfile` needed
- Must have a `manifest.json` with valid `name`, `description`, `trigger`, `parameters`, `returns`
- Must be hosted on GitHub with at least one tagged release
- Must not bundle any secrets or environment-specific configuration

## Related Documentation
- [Vision & Philosophy](00-VISION.md)
- [Quick Start Guide](01-QUICK-START.md)
- [LLM Providers Setup](02-PROVIDERS-SETUP.md)
- [Web Interface](03-WEB-INTERFACE.md)
- [System Architecture](04-ARCHITECTURE.md)
- [Workflow Guide](05-WORKFLOW.md)
- [Workers Overview](06-WORKERS.md)
- [Cognitive Architecture](07-COGNITIVE-ARCHITECTURE.md)
- [Data Schemas](08-DATA-SCHEMAS.md)
- [Tools & Extensions](09-TOOLS.md)
- [Context Relevance](10-CONTEXT-RELEVANCE.md)
- [Testing Guide](12-TESTING.md)
- [Message Flow Diagrams](13-MESSAGE-FLOW.md)