# chalie-tool-weather

Weather tool for [Chalie](https://github.com/chalie-ai/chalie) — current conditions and tomorrow's forecast.

**Sources:** [Open-Meteo](https://open-meteo.com/) (primary, uses coordinates) · [wttr.in](https://wttr.in/) (fallback, city names)
**No API key required.**

## What it returns

| Field | Type | Description |
|-------|------|-------------|
| `location` | string | Resolved place name |
| `condition` | string | Weather description |
| `temperature_c` / `temperature_f` | number | Current temperature |
| `feels_like_c` | number | Apparent temperature |
| `humidity_pct` | integer | Relative humidity % |
| `wind_kmh` | number | Wind speed |
| `wind_direction` | string | Compass direction (N, NNE, …) |
| `visibility_km` | number | Visibility |
| `uv_index` | integer | UV index |
| `precip_mm` | number | Precipitation |
| `observation_time` | string | Local observation timestamp |
| `is_raining` | boolean | |
| `is_daylight` | boolean | |
| `is_hot` | boolean | feels_like ≥ 30°C |
| `is_cold` | boolean | feels_like ≤ 10°C |
| `is_windy` | boolean | wind ≥ 30 km/h |
| `is_clear` | boolean | |
| `forecast_tomorrow_condition` | string | Tomorrow's condition |
| `forecast_tomorrow_max_c` / `_min_c` | number | Tomorrow's temp range |
| `forecast_tomorrow_precip_chance_pct` | integer | Tomorrow's rain chance |

Results are cached per location for 10 minutes.

## Installation

Install via the Chalie Brain UI → Tools → Catalog, or from the CLI:

```
POST /api/tools/install  {"git_url": "https://github.com/chalie-ai/chalie-tool-weather"}
```

This tool ships as a **trusted** tool and runs as a subprocess (no Docker required).

## Requirements

- Python 3.9+
- `requests>=2.32.5`
