# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Weather-ability-specific business-logic tests migrated from the per-ability
conformance file removed in TKT-975.

Pins TKT-914's contracts: the missing-location guardrail, the provider-unreachable
path when all providers are dead, fresh vs stale cache rendering, and the frozen
schema (exactly one optional ``location`` property).
"""

import json

import pytest

from abilities import weather as weather_mod
from abilities._dispatcher import ToolDispatcher
from abilities._registry import AbilityRegistry
from abilities.weather import WeatherAbility
from configs.channels import UserConfig
from tests._tool_result_harness import MP, body, head, seed_transcript

pytestmark = pytest.mark.unit

_DEAD_HOST = "http://127.0.0.1:9"


# ── Fixtures / helpers ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_weather_cache():
    """``WeatherAbility._cache`` is CLASS-level — clear it before AND after each
    test so a primed payload from one test never leaks into another."""
    WeatherAbility._cache.clear()
    yield
    WeatherAbility._cache.clear()


@pytest.fixture
def dead_providers(monkeypatch):
    """Point BOTH provider URL constants at a dead host so any real fetch fails
    fast and deterministically offline (neutralise the module's own config —
    TKT-892 precedent)."""
    monkeypatch.setattr(weather_mod, "_OPEN_METEO_BASE", _DEAD_HOST)
    monkeypatch.setattr(weather_mod, "_WTTR_BASE", _DEAD_HOST)


@pytest.fixture
def no_telemetry(monkeypatch):
    """Force ``ClientContext.current()`` to None so the dispatcher binds
    ``ability.telemetry = None`` — no device coordinates, fully deterministic."""
    monkeypatch.setattr(
        "services.client_context.ClientContext.current", classmethod(lambda cls: None)
    )


@pytest.fixture
def chat_mp(db):
    """A real chat-channel mp bound to the test database, with a seeded transcript
    anchor. Weather seeds ``allow`` on chat in the db template, so the real gate
    passes through to the production run()."""
    return MP(seed_transcript(db, "chat", "what's the weather"), UserConfig({}))


def _head(rendered: str) -> str:
    return head(rendered, "weather")


def _body(rendered: str) -> str:
    """Extract the body between the open tag and ``[end:weather]``. On a
    user-broadcasting channel a rich card is paired (``<json>\\n\\n<span ...>``);
    the ``rich=True`` harness extractor splits on the blank line and returns the
    JSON head."""
    return body(rendered, "weather", rich=True)


def _sample_payload(location: str = "Valletta, Malta") -> dict:
    """A real-shaped weather payload carrying the FULL key set the Open-Meteo
    fetcher emits — so the cache-hit tests round-trip a genuine card payload."""
    return {
        "location": location,
        "condition": "Partly cloudy",
        "temperature_c": 21.4,
        "temperature_f": 70.5,
        "feels_like_c": 21.0,
        "humidity_pct": 60,
        "wind_kmh": 12.0,
        "wind_direction": "NW",
        "visibility_km": None,
        "uv_index": None,
        "precip_mm": 0.0,
        "observation_time": "2026-06-11T09:00",
        "is_raining": False,
        "is_daylight": True,
        "is_hot": False,
        "is_cold": False,
        "is_windy": False,
        "is_clear": False,
        "forecast_tomorrow_condition": "Slight rain",
        "forecast_tomorrow_max_c": 24.0,
        "forecast_tomorrow_min_c": 18.0,
        "forecast_tomorrow_precip_chance_pct": 40,
        "forecast_tomorrow_precip_mm": 1.2,
        "sunrise": "2026-06-11T05:55",
        "sunset": "2026-06-11T20:18",
        "hourly": [{"hour": 9, "temp_c": 21, "code": 2}],
    }


# ── 1. missing-location: no coords AND no location param → loud guardrail ────────


def test_no_location_is_missing_location(db, chat_mp, no_telemetry, dead_providers):
    """No device coordinates AND no ``location`` param short-circuits to a stable
    ``code=missing-location`` BEFORE any provider is contacted — never the
    ``provider-unreachable`` lie that claims a source was tried."""
    out = ToolDispatcher(chat_mp).dispatch("weather", {"act_summary": "x"})

    assert "[weather(status=error, code=missing-location" in out
    assert "code=error]" not in out
    # provider-unreachable would be the lie — assert we did NOT fall into it.
    assert "provider-unreachable" not in out
    assert "hint:" in out
    assert "[end:weather]" in out


# ── 2. provider-unreachable: a real location, every provider dead ───────────────


def test_named_location_all_providers_dead_is_provider_unreachable(
    db, chat_mp, no_telemetry, dead_providers
):
    """A named location with every provider pointed at a dead host genuinely tried
    and failed → ``code=provider-unreachable`` with a hint and ``details`` meta in
    the head (the failure strings), never the ``code=error`` placeholder."""
    out = ToolDispatcher(chat_mp).dispatch(
        "weather", {"location": "Valletta", "act_summary": "x"}
    )

    h = _head(out)
    assert "status=error" in h
    assert "code=provider-unreachable" in h
    assert "details=" in h
    assert "code=error]" not in out
    assert "hint:" in out
    # This path genuinely contacted a provider — it is NOT the missing-location guard.
    assert "missing-location" not in out


# ── 3. fresh cache hit → success, body round-trips, NO stale marker ─────────────


def test_fresh_cache_hit_is_plain_success(db, chat_mp, no_telemetry, dead_providers):
    """A fresh cached payload is returned as a plain ``ok`` success: the body JSON
    round-trips to the primed payload and the head carries NO ``stale=true`` —
    fresh data must not be marked degraded."""
    import time

    payload = _sample_payload()
    WeatherAbility._cache["valletta"] = (payload, time.time())

    out = ToolDispatcher(chat_mp).dispatch(
        "weather", {"location": "Valletta", "act_summary": "x"}
    )

    h = _head(out)
    assert "status=success" in h
    assert "stale=true" not in h
    assert json.loads(_body(out)) == payload


# ── 4. stale cache → success BUT loud stale=true marker ──────────────────────────


def test_stale_cache_is_marked_stale(db, chat_mp, no_telemetry, dead_providers):
    """A stale-but-real cached payload (timestamp older than the TTL) returned when
    every live provider is dead carries ``stale=true`` in the head so the model can
    tell degraded data from fresh — the body still round-trips to the payload."""
    import time

    payload = _sample_payload()
    stale_ts = time.time() - (WeatherAbility._CACHE_TTL + 60)
    WeatherAbility._cache["valletta"] = (payload, stale_ts)

    out = ToolDispatcher(chat_mp).dispatch(
        "weather", {"location": "Valletta", "act_summary": "x"}
    )

    h = _head(out)
    assert "status=success" in h
    assert "stale=true" in h
    assert json.loads(_body(out)) == payload


# ── 5. schema frozen (exemplar guard): exactly one optional `location` ───────────


def test_schema_is_frozen_one_optional_location():
    """The exemplar's schema is pinned: exactly one property ``location`` and an
    empty ``required`` list. Others copy this schema — it must not grow params."""
    schema = AbilityRegistry.get("weather").get_parameters()
    props = schema["properties"]
    assert list(props.keys()) == ["location"]
    assert schema["required"] == []
    assert props["location"]["type"] == "string"
