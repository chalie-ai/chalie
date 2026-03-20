"""
Weather Tool Handler — Open-Meteo (primary) with wttr.in fallback.

Open-Meteo: free, no key, returns only requested fields (~1KB vs ~30KB for j1).
wttr.in: fallback for city-name lookups and when Open-Meteo fails.

Routing:
  - Coords available (from telemetry): Open-Meteo → wttr.in fallback
  - City name in params: wttr.in directly (Open-Meteo needs coordinates)

Module-level cache per location key, 10min TTL.
Note: Cache does not persist across container runs (Docker sandbox).
"""

import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

_cache: dict = {}
_CACHE_TTL = 600  # 10 minutes

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


def execute(topic: str, params: dict, config: dict = None, telemetry: dict = None) -> dict:
    """
    Get current weather for a location.

    Args:
        topic: Conversation topic (passed by framework)
        params: {"location": str (optional city name; omit to use client coordinates)}
        config: {} (no credentials needed)
        telemetry: Client telemetry with location coords and resolved name

    Returns:
        location, condition, temperature_c/f, feels_like_c, humidity_pct,
        wind_kmh, wind_direction, visibility_km, uv_index, precip_mm,
        observation_time, is_raining, is_daylight, is_hot, is_cold, is_windy, is_clear
    """
    location_param = params.get("location", "").strip()

    # Extract lat/lon and location name from telemetry (now flattened)
    lat = lon = None
    location_name = None
    if telemetry:
        # New flattened telemetry format: lat, lon, city, country directly at top level
        lat = telemetry.get("lat")
        lon = telemetry.get("lon")
        city = telemetry.get("city", "")
        country = telemetry.get("country", "")
        location_name = f"{city}, {country}" if city and country else city or country or None

    # Build cache key — prefer resolved name, fall back to coords or param
    if location_name and not location_param:
        cache_key = location_name.lower()
    elif lat is not None and lon is not None and not location_param:
        cache_key = f"{lat:.4f},{lon:.4f}"
    else:
        cache_key = location_param.lower() if location_param else "auto"

    now = time.time()
    if cache_key in _cache:
        cached_result, cached_ts = _cache[cache_key]
        if (now - cached_ts) < _CACHE_TTL:
            return cached_result

    result = None
    open_meteo_err = ""
    wttr_err = ""

    # Use Open-Meteo when we have coordinates and no explicit city name
    if lat is not None and lon is not None and not location_param:
        result, open_meteo_err = _fetch_open_meteo(lat, lon, location_name or f"{lat:.4f}, {lon:.4f}")

    # Fall back to wttr.in only when we have an explicit city name param
    # (raw coordinates passed to wttr.in fail silently — skip if coords-only)
    if result is None and location_param:
        result, wttr_err = _fetch_wttr(location_param)

    # Final retry with shorter timeout before giving up
    if result is None:
        logger.warning(f"[WEATHER] Both sources failed, retrying with 5s timeout for '{cache_key}'")
        if lat is not None and lon is not None and not location_param:
            result, _ = _fetch_open_meteo(lat, lon, location_name or f"{lat:.4f}, {lon:.4f}", timeout=5)
        if result is None and location_param:
            result, _ = _fetch_wttr(location_param, timeout=5)

    if result is not None:
        _cache[cache_key] = (result, now)
        return result

    # Return stale cache if available
    if cache_key in _cache:
        logger.warning(f"[WEATHER] All sources unavailable, returning stale cache for '{cache_key}'")
        return _cache[cache_key][0]

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
    import requests
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,apparent_temperature,relative_humidity_2m,"
            "precipitation,weather_code,wind_speed_10m,wind_direction_10m,is_day"
            "&daily=weathercode,precipitation_probability_max,precipitation_sum,"
            "temperature_2m_max,temperature_2m_min"
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
        if daily and len(daily.get("time", [])) > 1:
            raw_code = daily.get("weathercode", [None, None])[1]
            if raw_code is not None:
                tomorrow_condition = _WMO.get(int(raw_code), f"Code {raw_code}")
            tomorrow_precip_chance = daily.get("precipitation_probability_max", [None, None])[1]
            tomorrow_precip_mm = daily.get("precipitation_sum", [None, None])[1]
            tomorrow_max_c = daily.get("temperature_2m_max", [None, None])[1]
            tomorrow_min_c = daily.get("temperature_2m_min", [None, None])[1]

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
        }, ""
    except Exception as e:
        logger.warning(f"[WEATHER] Open-Meteo failed for ({lat},{lon}): {e}")
        return None, str(e)


def _fetch_wttr(location: str, timeout: int = 15) -> tuple:
    """Fetch current weather from wttr.in j1. Returns (result, error_str)."""
    import requests
    try:
        url = f"https://wttr.in/{location}?format=j1"
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Chalie/1.0 cognitive-agent"})
        resp.raise_for_status()
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
        }, ""
    except Exception as e:
        logger.warning(f"[WEATHER] wttr.in failed for '{location}': {e}")
        return None, str(e)


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
    except Exception:
        pass
    return 6 <= datetime.now().hour <= 20
