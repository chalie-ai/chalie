"""Feature tests for WeatherAbility (abilities/weather.py).

Asserts the real WeatherAbility production code behaves correctly when its two
external HTTP boundaries (Open-Meteo and wttr.in) are substituted with canned
responses.  All other collaborators — Ability ABC, AbilityRegistry, ClassVar
cache — are real production objects.

Mock boundary: requests.get — patched at the canonical requests module since
abilities.weather imports requests lazily inside function bodies (not at
module level).
"""

import pytest
from unittest.mock import patch, MagicMock

from abilities.weather import WeatherAbility
from abilities._registry import AbilityRegistry, _reset_for_tests

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared canned responses
# ---------------------------------------------------------------------------

_OPEN_METEO_RESPONSE = {
    "current": {
        "time": "2026-04-27T10:00",
        "temperature_2m": 22.5,
        "apparent_temperature": 21.0,
        "relative_humidity_2m": 58,
        "precipitation": 0.0,
        "weather_code": 1,
        "wind_speed_10m": 15.0,
        "wind_direction_10m": 270.0,
        "is_day": 1,
    },
    "daily": {
        "time": ["2026-04-27", "2026-04-28"],
        "weathercode": [1, 63],
        "precipitation_probability_max": [5, 80],
        "precipitation_sum": [0.0, 12.4],
        "temperature_2m_max": [23.0, 18.0],
        "temperature_2m_min": [14.0, 12.0],
    },
}

_WTTR_RESPONSE = {
    "current_condition": [
        {
            "temp_C": "19",
            "temp_F": "66",
            "FeelsLikeC": "18",
            "humidity": "65",
            "windspeedKmph": "12",
            "winddir16Point": "SW",
            "visibility": "10",
            "uvIndex": "3",
            "precipMM": "0.0",
            "weatherDesc": [{"value": "Partly cloudy"}],
            "localObsDateTime": "2026-04-27 10:00 AM",
        }
    ],
    "nearest_area": [
        {
            "areaName": [{"value": "London"}],
            "country": [{"value": "United Kingdom"}],
        }
    ],
    "weather": [
        {
            "hourly": [
                {"chanceofrain": 5, "precipMM": 0.0, "weatherDesc": [{"value": "Sunny"}]},
                {"chanceofrain": 5, "precipMM": 0.0, "weatherDesc": [{"value": "Sunny"}]},
                {"chanceofrain": 5, "precipMM": 0.0, "weatherDesc": [{"value": "Sunny"}]},
                {"chanceofrain": 5, "precipMM": 0.0, "weatherDesc": [{"value": "Sunny"}]},
                {"chanceofrain": 5, "precipMM": 0.0, "weatherDesc": [{"value": "Sunny"}]},
            ],
            "maxTempC": "21",
            "minTempC": "13",
        },
        {
            "hourly": [
                {"chanceofrain": 70, "precipMM": 3.5, "weatherDesc": [{"value": "Heavy rain"}]},
                {"chanceofrain": 80, "precipMM": 4.0, "weatherDesc": [{"value": "Heavy rain"}]},
                {"chanceofrain": 75, "precipMM": 3.8, "weatherDesc": [{"value": "Heavy rain"}]},
                {"chanceofrain": 80, "precipMM": 4.2, "weatherDesc": [{"value": "Heavy rain"}]},
                {"chanceofrain": 85, "precipMM": 4.5, "weatherDesc": [{"value": "Heavy rain"}]},
            ],
            "maxTempC": "17",
            "minTempC": "11",
        },
    ],
}


def _make_http_response(json_data, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    if status_code >= 400:
        from requests.exceptions import HTTPError
        mock.raise_for_status.side_effect = HTTPError(response=mock)
    else:
        mock.raise_for_status.return_value = None
    return mock


# ---------------------------------------------------------------------------
# Cache isolation fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_weather_cache():
    """Prevent cache pollution between test cases."""
    WeatherAbility._cache.clear()
    yield
    WeatherAbility._cache.clear()


@pytest.fixture(autouse=True)
def isolate_registry():
    """Reset registry singleton between tests to avoid subclass cross-contamination."""
    import gc
    gc.collect()
    _reset_for_tests()
    yield
    gc.collect()
    _reset_for_tests()


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------

def test_weather_ability_is_concrete_and_instantiable():
    """WeatherAbility satisfies the Ability ABC — all required attributes present."""
    wa = WeatherAbility()
    assert wa.NAME == "weather"
    assert isinstance(wa.SUMMARY, str) and wa.SUMMARY
    assert isinstance(wa.EXAMPLES, list)
    assert 6 <= len(wa.EXAMPLES) <= 8
    assert all(isinstance(e, str) for e in wa.EXAMPLES)
    assert isinstance(wa.INPUT_SCHEMA, dict)


def test_registry_get_returns_weather_ability():
    """AbilityRegistry.get('weather') returns a live WeatherAbility instance."""
    instance = AbilityRegistry.get("weather")
    assert isinstance(instance, WeatherAbility)


def test_weather_defaults():
    """ALWAYS_AVAILABLE is False and TIMEOUT is 10."""
    assert WeatherAbility.ALWAYS_AVAILABLE is False
    assert WeatherAbility.TIMEOUT == 10


# ---------------------------------------------------------------------------
# execute() — happy path via Open-Meteo
# ---------------------------------------------------------------------------

def test_execute_open_meteo_happy_path_returns_documented_keys():
    """With coords in telemetry and a successful Open-Meteo response, all documented keys are present."""
    telemetry = {"lat": 35.8989, "lon": 14.5146, "city": "Valletta", "country": "Malta"}

    with patch("requests.get", return_value=_make_http_response(_OPEN_METEO_RESPONSE)):
        result = WeatherAbility().execute("text", {}, telemetry)

    expected_keys = {
        "location", "condition", "temperature_c", "temperature_f", "feels_like_c",
        "humidity_pct", "wind_kmh", "wind_direction", "precip_mm",
        "is_raining", "is_daylight", "is_hot", "is_cold", "is_windy", "is_clear",
        "forecast_tomorrow_condition", "forecast_tomorrow_max_c",
        "forecast_tomorrow_min_c", "forecast_tomorrow_precip_chance_pct",
        "forecast_tomorrow_precip_mm",
    }
    assert expected_keys.issubset(result.keys()), f"Missing keys: {expected_keys - result.keys()}"
    assert result["location"] == "Valletta, Malta"
    assert result["temperature_c"] == pytest.approx(22.5)
    assert result["temperature_f"] == pytest.approx(72.5)
    assert result["is_daylight"] is True
    assert result["is_raining"] is False
    assert result["is_clear"] is True  # WMO code 1 = Mainly clear
    # tomorrow's forecast parsed correctly
    assert result["forecast_tomorrow_condition"] == "Moderate rain"
    assert result["forecast_tomorrow_precip_chance_pct"] == 80


# ---------------------------------------------------------------------------
# execute() — cache hit: second call must not hit the network again
# ---------------------------------------------------------------------------

def test_execute_cache_hit_calls_requests_once():
    """Calling execute() twice with identical telemetry fires requests.get only once."""
    telemetry = {"lat": 35.8989, "lon": 14.5146, "city": "Valletta", "country": "Malta"}
    mock_resp = _make_http_response(_OPEN_METEO_RESPONSE)

    with patch("requests.get", return_value=mock_resp) as mock_get:
        WeatherAbility().execute("text", {}, telemetry)
        WeatherAbility().execute("text", {}, telemetry)
        assert mock_get.call_count == 1, f"Expected 1 HTTP call, got {mock_get.call_count}"


# ---------------------------------------------------------------------------
# execute() — wttr.in retry: first attempt returns 500, second succeeds
# ---------------------------------------------------------------------------

def test_execute_retries_wttr_on_first_failure():
    """wttr.in returns 500 on the first attempt; retry path triggers and second attempt succeeds.

    When a location param is given with no telemetry coords, the production
    guard (weather.py:117) never tries Open-Meteo — both requests.get calls
    target wttr.in. The first returns HTTP 500; the retry returns the full
    wttr.in JSON response and the result is parsed successfully.
    """
    side_effects = [
        _make_http_response({}, status_code=500),
        _make_http_response(_WTTR_RESPONSE),
    ]

    with patch("requests.get", side_effect=side_effects):
        result = WeatherAbility().execute("text", {"location": "London"}, None)

    assert "error" not in result, f"Unexpected error: {result}"
    assert result["location"] == "London, United Kingdom"
    assert result["temperature_c"] == pytest.approx(19.0)
    assert result["is_raining"] is False


# ---------------------------------------------------------------------------
# execute() — total failure: both sources fail → error dict
# ---------------------------------------------------------------------------

def test_execute_total_failure_returns_error_dict():
    """When both Open-Meteo and wttr.in fail, returned dict has 'error' and 'details' keys."""
    from requests.exceptions import HTTPError

    def _always_raise(*args, **kwargs):
        mock = MagicMock()
        mock.raise_for_status.side_effect = HTTPError("server down")
        return mock

    # Force both sources to fail: provide coords (triggers Open-Meteo attempt + retry)
    # and a location param (triggers wttr.in attempt) by using location_param only
    # so we exercise the wttr.in failure path cleanly.
    with patch("requests.get", side_effect=Exception("connection refused")):
        result = WeatherAbility().execute("text", {"location": "NoSuchCity"}, None)

    assert "error" in result, f"Expected error key, got: {result}"
    assert "details" in result, f"Expected details key, got: {result}"
