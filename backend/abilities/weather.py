"""
WeatherAbility — Open-Meteo (primary) with wttr.in fallback.

Open-Meteo: free, no key, returns only requested fields (~1KB vs ~30KB for j1).
wttr.in: fallback for city-name lookups and when Open-Meteo fails.

Routing:
  - Coords available (from telemetry): Open-Meteo -> wttr.in fallback
  - City name in params: wttr.in directly (Open-Meteo needs coordinates)

Class-level cache per location key, 10-min TTL.

Rich-media rendering:
  A successful weather lookup returns ``ToolResult.ok(payload, rich=payload)`` —
  the weather dict is both the structured body the model reads AND the card
  payload. The dispatcher (``ToolDispatcher._render``) owns ordinal assignment +
  the span-tag instruction, and injects the card ONLY when the invoking channel
  broadcasts to the user. Error lookups return ``ToolResult.err`` with no card.
"""

import logging
import time
from typing import ClassVar

import requests

from abilities._ability import Ability
from abilities._result import ToolResult
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

# WMO weather interpretation codes (Open-Meteo)
_WMO = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}

_RAIN_WORDS = ("rain", "drizzle", "shower")
_CLEAR_WORDS = ("clear", "sunny", "mainly clear")

# Provider base URLs as module constants (not hardcoded inside the fetchers) so
# the request endpoints are a single, overridable point of configuration. Offline
# tests neutralise the module's OWN config by pointing these at a dead host —
# forcing deterministic provider failure without mocking the requests collaborator.
_OPEN_METEO_BASE = "https://api.open-meteo.com"
_WTTR_BASE = "https://wttr.in"


class WeatherAbility(Ability):
    def get_name(self) -> str:
        return "weather"

    def get_summary(self) -> str:
        return "Get current weather and tomorrow's forecast for a city or device coordinates."

    def get_examples(self) -> list[str]:
        return [
            "what's the weather like today",
            "will it rain tomorrow",
            "do I need a jacket",
            "is it sunny outside",
            "temperature in London",
            "weather forecast for tomorrow",
            "should I bring an umbrella",
        ]

    def get_search_tooltip(self) -> str:
        return "weather forecasts"

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": (
                    "City or place name (e.g. 'Malta', 'London'). Omit to "
                    "use the user's current location automatically."
                ),
            },
        },
        "required": [],
    }

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    _cache: ClassVar[dict] = {}
    _CACHE_TTL: ClassVar[int] = 600  # 10 minutes

    def run(self, params: dict) -> ToolResult:
        """Get current weather for a location.

        Reads the optional ``location`` param (omit → client coordinates from
        ``self.telemetry``). Returns ``ToolResult.ok(payload, rich=payload)`` on
        success — the dispatcher pairs the weather card when this turn broadcasts
        to the user — or ``ToolResult.err`` when every source is unavailable.
        """
        location_param = params.get("location", "").strip()
        lat, lon, location_name = _extract_location(self.telemetry)

        # Guardrail: with no device coordinates AND no location param there is no
        # place to look up. The fallback chain would contact NO provider yet still
        # report code=provider-unreachable — a lie. Reject loudly before any cache
        # lookup or fetch so the model names a city instead of retrying a dead end.
        if not location_param and (lat is None or lon is None):
            return ToolResult.err(
                "No location available: no device coordinates and no location parameter.",
                code="missing-location",
                hint="name a city, e.g. location=London",
            )

        cache_key = _build_cache_key(location_param, lat, lon, location_name)

        cached = _get_fresh_cache(cache_key)
        if cached is not None:
            return ToolResult.ok(cached, rich=cached)

        result, open_meteo_err, wttr_err = _fetch_with_fallback(location_param, lat, lon, location_name, cache_key)

        if result is not None:
            WeatherAbility._cache[cache_key] = (result, time.time())
            return ToolResult.ok(result, rich=result)

        payload = _stale_or_error(cache_key, open_meteo_err, wttr_err)
        if "error" in payload:
            return ToolResult.err(
                payload["error"],
                code="provider-unreachable",
                hint="try again shortly or name a specific city.",
                details=payload.get("details", ""),
            )
        # Stale-but-real cached data — still a usable card, but mark it loudly so
        # the model can distinguish degraded data from a fresh observation
        # (loudness precedent: search fallback=ddg, programming_docs_search
        # degraded=true). The fresh-cache hit above is NOT a degradation — unmarked.
        return ToolResult.ok(payload, rich=payload, stale=True)


def _extract_location(telemetry: dict | None) -> tuple:
    """Pull (lat, lon, location_name) from telemetry; all default to None."""
    if not telemetry:
        return None, None, None
    lat = telemetry.get("lat")
    lon = telemetry.get("lon")
    city = telemetry.get("city", "")
    country = telemetry.get("country", "")
    location_name = f"{city}, {country}" if city and country else city or country or None
    return lat, lon, location_name


def _build_cache_key(location_param: str, lat, lon, location_name: str | None) -> str:
    """Prefer resolved name, fall back to coords, then to the literal param ('auto' if empty)."""
    if location_name and not location_param:
        return location_name.lower()
    if lat is not None and lon is not None and not location_param:
        return f"{lat:.4f},{lon:.4f}"
    return location_param.lower() if location_param else "auto"


def _get_fresh_cache(cache_key: str):
    """Return the cached result if still within TTL, else None."""
    if cache_key not in WeatherAbility._cache:
        return None
    cached_result, cached_ts = WeatherAbility._cache[cache_key]
    if (time.time() - cached_ts) < WeatherAbility._CACHE_TTL:
        return cached_result
    return None


def _fetch_with_fallback(location_param: str, lat, lon, location_name, cache_key: str) -> tuple:
    """Try Open-Meteo (coords) → wttr.in (city) → 5s retry of either. Returns (result, om_err, wttr_err).

    Routing:
      - No location_param: prefer Open-Meteo with telemetry coords.
      - location_param matches telemetry city/country: prefer Open-Meteo with
        telemetry coords (LLM is just naming the user's home turf).
      - location_param is a different place: use wttr.in directly (Open-Meteo
        needs coords we don't have for foreign queries).

    wttr.in is also used as a fallback when Open-Meteo fails. In all wttr.in
    paths, when we have a clean Nominatim location_name AND the query refers
    to that same place, we override wttr's mojibake-prone location field.
    """
    result = None
    open_meteo_err = ""
    wttr_err = ""
    have_coords = lat is not None and lon is not None
    use_telemetry_coords = have_coords and (not location_param or _matches_telemetry(location_param, location_name))

    if use_telemetry_coords:
        result, open_meteo_err = _fetch_open_meteo(lat, lon, location_name or f"{lat:.4f}, {lon:.4f}")

    if result is None and location_param:
        result, wttr_err = _fetch_wttr(location_param)
        if result is not None and use_telemetry_coords and location_name:
            result["location"] = location_name

    if result is None:
        logger.warning(f"[WEATHER] Both sources failed, retrying with 5s timeout for '{cache_key}'")
        if use_telemetry_coords:
            result, _ = _fetch_open_meteo(lat, lon, location_name or f"{lat:.4f}, {lon:.4f}", timeout=5)
        if result is None and location_param:
            result, _ = _fetch_wttr(location_param, timeout=5)
            if result is not None and use_telemetry_coords and location_name:
                result["location"] = location_name

    return result, open_meteo_err, wttr_err


def _matches_telemetry(location_param: str, location_name: str | None) -> bool:
    """True when the LLM-supplied location names the same place as telemetry.

    Case-insensitive substring match in either direction so 'Żabbar' matches
    'Iż-Żabbar, Malta' and 'malta' matches the country half. Diacritics are
    not normalised — a partial English/Maltese collision is acceptable.
    """
    if not location_param or not location_name:
        return False
    p = location_param.strip().lower()
    n = location_name.strip().lower()
    return p in n or n in p


def _stale_or_error(cache_key: str, open_meteo_err: str, wttr_err: str) -> dict:
    """Return stale-cached result if any, else build a structured error response."""
    if cache_key in WeatherAbility._cache:
        logger.warning(f"[WEATHER] All sources unavailable, returning stale cache for '{cache_key}'")
        return WeatherAbility._cache[cache_key][0]

    failures = []
    if open_meteo_err:
        failures.append(f"Open-Meteo: {open_meteo_err}")
    if wttr_err:
        failures.append(f"wttr.in: {wttr_err}")
    logger.error(f"[WEATHER] All weather sources unavailable for '{cache_key}': {'; '.join(failures)}")
    return {
        "error": "All weather sources unavailable",
        "details": "; ".join(failures) if failures else "Unknown error",
    }


def _fetch_open_meteo(lat: float, lon: float, location_name: str, timeout: int = 15) -> tuple:
    """Fetch current weather from Open-Meteo. Returns (result, error_str)."""
    try:
        url = (
            f"{_OPEN_METEO_BASE}/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,apparent_temperature,relative_humidity_2m,"
            "precipitation,weather_code,wind_speed_10m,wind_direction_10m,is_day"
            "&hourly=temperature_2m,weather_code"
            "&daily=weathercode,precipitation_probability_max,precipitation_sum,"
            "temperature_2m_max,temperature_2m_min,sunrise,sunset"
            "&wind_speed_unit=kmh"
            "&timezone=auto"
            "&forecast_days=2"
        )
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Chalie/1.0 cognitive-agent"})
        resp.raise_for_status()
        data = resp.json()

        cc = data.get("current")
        if not cc:
            return None, "No current data in response"

        weather_code = int(cc.get("weather_code", 0))
        condition = _WMO.get(weather_code, f"Code {weather_code}")
        condition_lower = condition.lower()

        temp_c = float(cc.get("temperature_2m", 0))
        feels_like_c = float(cc.get("apparent_temperature", temp_c))

        # Extract tomorrow's forecast from daily data (index 1 = tomorrow)
        daily = data.get("daily", {})
        tomorrow_condition = None
        tomorrow_precip_chance = None
        tomorrow_precip_mm = None
        tomorrow_max_c = None
        tomorrow_min_c = None
        sunrise = None
        sunset = None
        if daily and len(daily.get("time", [])) > 1:
            raw_code = daily.get("weathercode", [None, None])[1]
            if raw_code is not None:
                tomorrow_condition = _WMO.get(int(raw_code), f"Code {raw_code}")
            tomorrow_precip_chance = daily.get("precipitation_probability_max", [None, None])[1]
            tomorrow_precip_mm = daily.get("precipitation_sum", [None, None])[1]
            tomorrow_max_c = daily.get("temperature_2m_max", [None, None])[1]
            tomorrow_min_c = daily.get("temperature_2m_min", [None, None])[1]
            sunrise_arr = daily.get("sunrise", [])
            sunset_arr = daily.get("sunset", [])
            sunrise = sunrise_arr[0] if sunrise_arr else None
            sunset = sunset_arr[0] if sunset_arr else None

        hourly_strip = _extract_hourly_strip(data.get("hourly", {}), cc.get("time", ""))

        return {
            "location": location_name,
            "condition": condition,
            "temperature_c": temp_c,
            "temperature_f": round(temp_c * 9 / 5 + 32, 1),
            "feels_like_c": feels_like_c,
            "humidity_pct": int(cc.get("relative_humidity_2m", 0)),
            "wind_kmh": float(cc.get("wind_speed_10m", 0)),
            "wind_direction": _degrees_to_compass(float(cc.get("wind_direction_10m", 0))),
            "visibility_km": None,
            "uv_index": None,
            "precip_mm": float(cc.get("precipitation", 0)),
            "observation_time": cc.get("time", ""),
            "is_raining": any(w in condition_lower for w in _RAIN_WORDS),
            "is_daylight": bool(cc.get("is_day", 1)),
            "is_hot": feels_like_c >= 30,
            "is_cold": feels_like_c <= 10,
            "is_windy": float(cc.get("wind_speed_10m", 0)) >= 30,
            "is_clear": weather_code in (0, 1),
            "forecast_tomorrow_condition": tomorrow_condition,
            "forecast_tomorrow_max_c": tomorrow_max_c,
            "forecast_tomorrow_min_c": tomorrow_min_c,
            "forecast_tomorrow_precip_chance_pct": tomorrow_precip_chance,
            "forecast_tomorrow_precip_mm": tomorrow_precip_mm,
            "sunrise": sunrise,
            "sunset": sunset,
            "hourly": hourly_strip,
        }, ""
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as e:
        # The degradation set: transport failures (RequestException) and parse /
        # shape surprises (ValueError covers JSON decode + float/int casts;
        # KeyError/IndexError/TypeError cover unexpected provider-JSON shapes).
        # These route to the fallback chain; a real bug (e.g. AttributeError) is
        # NOT caught here and propagates so it is never misreported as a provider error.
        logger.warning(f"[WEATHER] Open-Meteo failed for ({lat},{lon}): {e}")
        return None, str(e)


def _fetch_wttr(location: str, timeout: int = 15) -> tuple:
    """Fetch current weather from wttr.in j1. Returns (result, error_str)."""
    try:
        url = f"{_WTTR_BASE}/{location}?format=j1"
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Chalie/1.0 cognitive-agent"})
        resp.raise_for_status()
        # wttr.in serves JSON without a charset header; requests guesses Latin-1
        # and mojibakes UTF-8 place names ('Żabbar' → 'Il-AÅ¦Ofra'). Force UTF-8.
        resp.encoding = "utf-8"
        data = resp.json()

        current_conditions = data.get("current_condition")
        if not current_conditions or not isinstance(current_conditions, list):
            return None, "No current_condition in response"
        cc = current_conditions[0]
        if not cc or not isinstance(cc, dict):
            return None, "Invalid current_condition format"

        # Resolve location name from nearest_area
        nearest = data.get("nearest_area", [])
        if nearest and isinstance(nearest, list):
            area = nearest[0]
            area_name = (area.get("areaName") or [{}])[0].get("value", location)
            country = (area.get("country") or [{}])[0].get("value", "")
            location_name = f"{area_name}, {country}" if country else area_name
        else:
            location_name = location

        condition = (cc.get("weatherDesc") or [{}])[0].get("value", "")
        condition_lower = condition.lower()
        temp_c = float(cc.get("temp_C", 0))
        feels_like_c = float(cc.get("FeelsLikeC", temp_c))
        obs_time = cc.get("localObsDateTime", "")

        # Extract tomorrow's forecast from weather array (index 1 = tomorrow)
        weather_days = data.get("weather", [])
        tomorrow_condition = None
        tomorrow_precip_chance = None
        tomorrow_precip_mm = None
        tomorrow_max_c = None
        tomorrow_min_c = None
        if len(weather_days) > 1:
            tmrw = weather_days[1]
            tmrw_hourly = tmrw.get("hourly", [])
            if tmrw_hourly:
                tomorrow_precip_chance = max(int(h.get("chanceofrain", 0)) for h in tmrw_hourly)
                tomorrow_precip_mm = round(sum(float(h.get("precipMM", 0)) for h in tmrw_hourly), 1)
                # Use midday slot (index 4 = noon) for condition description
                noon = tmrw_hourly[4] if len(tmrw_hourly) > 4 else tmrw_hourly[-1]
                tomorrow_condition = (noon.get("weatherDesc") or [{}])[0].get("value", "") or None
            tomorrow_max_c = float(tmrw.get("maxTempC", 0)) if tmrw.get("maxTempC") is not None else None
            tomorrow_min_c = float(tmrw.get("minTempC", 0)) if tmrw.get("minTempC") is not None else None

        return {
            "location": location_name,
            "condition": condition,
            "temperature_c": temp_c,
            "temperature_f": float(cc.get("temp_F", 0)),
            "feels_like_c": feels_like_c,
            "humidity_pct": int(cc.get("humidity", 0)),
            "wind_kmh": float(cc.get("windspeedKmph", 0)),
            "wind_direction": cc.get("winddir16Point", ""),
            "visibility_km": float(cc.get("visibility", 0)),
            "uv_index": int(cc.get("uvIndex", 0)),
            "precip_mm": float(cc.get("precipMM", 0)),
            "observation_time": obs_time,
            "is_raining": any(w in condition_lower for w in _RAIN_WORDS),
            "is_daylight": _estimate_daylight(obs_time),
            "is_hot": feels_like_c >= 30,
            "is_cold": feels_like_c <= 10,
            "is_windy": float(cc.get("windspeedKmph", 0)) >= 30,
            "is_clear": any(w in condition_lower for w in _CLEAR_WORDS),
            "forecast_tomorrow_condition": tomorrow_condition,
            "forecast_tomorrow_max_c": tomorrow_max_c,
            "forecast_tomorrow_min_c": tomorrow_min_c,
            "forecast_tomorrow_precip_chance_pct": tomorrow_precip_chance,
            "forecast_tomorrow_precip_mm": tomorrow_precip_mm,
            "sunrise": None,
            "sunset": None,
            "hourly": [],
        }, ""
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as e:
        # Same degradation set as Open-Meteo: transport + parse/shape failures route
        # to the fallback; programming bugs propagate rather than masquerade as a
        # provider outage.
        logger.warning(f"[WEATHER] wttr.in failed for '{location}': {e}")
        return None, str(e)


def _extract_hourly_strip(hourly: dict, current_iso: str) -> list:
    """Pick the next 8 hourly readings starting from the slot containing current_iso.

    Open-Meteo returns ``hourly.time`` as ``["2026-05-03T09:00", ...]`` aligned
    with the location's local timezone (``timezone=auto``).  The current time
    is also a local ISO string with no offset, so simple prefix-match against
    the ``YYYY-MM-DDTHH`` portion locates the first slot — no parsing needed.
    """
    if not hourly:
        return []
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    codes = hourly.get("weather_code", [])
    if not times or not temps:
        return []

    start_idx = 0
    if current_iso:
        prefix = current_iso[:13]  # "YYYY-MM-DDTHH"
        for i, t in enumerate(times):
            if t.startswith(prefix):
                start_idx = i
                break

    out = []
    end_idx = min(start_idx + 8, len(times))
    for i in range(start_idx, end_idx):
        try:
            hour = int(times[i][11:13])
        except (ValueError, IndexError):
            hour = None
        try:
            temp = round(float(temps[i]))
        except (TypeError, ValueError, IndexError):
            temp = None
        code = None
        if i < len(codes):
            try:
                code = int(codes[i])
            except (TypeError, ValueError):
                code = None
        if hour is not None and temp is not None:
            out.append({"hour": hour, "temp_c": temp, "code": code})
    return out


def _degrees_to_compass(degrees: float) -> str:
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return directions[round(degrees / 22.5) % 16]


def _estimate_daylight(obs_time: str) -> bool:
    """Estimate daylight from wttr.in observation time string."""
    try:
        if obs_time:
            parts = obs_time.strip().split(" ")
            if len(parts) >= 2:
                hour = int(parts[1].split(":")[0])
                if len(parts) >= 3:
                    if parts[2].upper() == "PM" and hour != 12:
                        hour += 12
                    elif parts[2].upper() == "AM" and hour == 12:
                        hour = 0
                return 6 <= hour <= 20
    except (ValueError, IndexError):
        # Unparseable time string → fall through to the UTC-hour heuristic; a real
        # bug is not swallowed.
        pass
    return 6 <= utc_now().hour <= 20
