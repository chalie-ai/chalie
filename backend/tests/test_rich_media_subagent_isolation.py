"""Round-3 regression sentinels — subagent boundary isolation for rich-media.

The contract being asserted:

  1. ``ActDispatcherService.dispatch_action`` injects ``_rich_media_ordinal``
     ONLY when called with ``channel == 'user'``. Any other channel — most
     importantly ``'subagent'`` — never gets the ordinal, so tools that
     synthesise an instruction trailer from the ordinal (e.g. weather) return
     their plain dict result instead.

  2. ``rich_media_parser.strip_spans`` removes every ``<span id='name_N'>…</span>``
     wrapper while keeping inner text. The subagent ability calls this on the
     processor's returned text in both ``_run_sync`` and ``_run_async`` so even
     a rogue span (memorised, hallucinated, or leaked from a future trailer)
     never reaches the parent agent verbatim.

If either of these guards regresses, scenarios 058 and 059 silently lose their
rich-media cards by reverting to the round-432 failure mode (subagent
paraphrasing the span away). These tests are the canary.
"""

from __future__ import annotations

import json
import pytest

from services.act_dispatcher_service import ActDispatcherService
from services.rich_media_parser import strip_spans

pytestmark = pytest.mark.integration


class TestDispatcherChannelGate:
    """Ordinal injection is gated on channel — user only, not subagent."""

    def test_user_channel_gets_ordinal_injected(self):
        captured: dict = {}

        def _capture(channel, action):
            captured.update(action)
            return "ok"

        d = ActDispatcherService()
        d.handlers["weather"] = _capture
        d.dispatch_action("user", {"type": "weather"})

        assert captured.get("_rich_media_ordinal") == 1, (
            "User-channel dispatch must inject ordinal — without it the trailer "
            "never reaches the LLM and rich-media cards never render."
        )

    def test_subagent_channel_never_gets_ordinal(self):
        captured: dict = {}

        def _capture(channel, action):
            captured.update(action)
            return "ok"

        d = ActDispatcherService()
        d.handlers["weather"] = _capture
        d.dispatch_action("subagent", {"type": "weather"})

        assert "_rich_media_ordinal" not in captured, (
            "Subagent-channel dispatch must NOT inject the ordinal. If this "
            "fails, the rich-media trailer can reach a subagent — which then "
            "either paraphrases away the span (silent card loss) or leaks "
            "raw markup back to the parent."
        )

    def test_subagent_channel_does_not_advance_user_counter(self):
        """Mixed-channel dispatches must not inflate the user-channel ordinal.

        The dispatcher state is shared across channels (one instance per
        MessageProcessor turn), so a subagent dispatch that bumped the
        counter would silently shift subsequent user dispatches off-by-one.
        """
        captured: list[dict] = []

        def _capture(channel, action):
            captured.append(dict(action))
            return "ok"

        d = ActDispatcherService()
        d.handlers["weather"] = _capture

        d.dispatch_action("subagent", {"type": "weather"})  # no ordinal
        d.dispatch_action("subagent", {"type": "weather"})  # no ordinal
        d.dispatch_action("user", {"type": "weather"})       # ordinal 1
        d.dispatch_action("user", {"type": "weather"})       # ordinal 2

        ordinals = [c.get("_rich_media_ordinal") for c in captured]
        assert ordinals == [None, None, 1, 2], (
            f"Subagent dispatches must not advance the user counter. "
            f"Got ordinals {ordinals}, expected [None, None, 1, 2]."
        )


class TestWeatherWithoutOrdinalReturnsDict:
    """When dispatched without ordinal injection, weather must return a dict
    (no instruction trailer string)."""

    def test_direct_call_with_no_ordinal_yields_dict(self, monkeypatch):
        from abilities import weather as _weather_mod

        # Stub the network call so the test is hermetic — we only care about
        # the serialisation branch, not the upstream API.
        sample_payload = {
            "location": "Test City",
            "condition": "Clear sky",
            "temperature_c": 20.0,
            "temperature_f": 68.0,
            "feels_like_c": 19.0,
            "humidity_pct": 50,
            "wind_kmh": 5.0,
            "wind_direction": "N",
            "visibility_km": None,
            "uv_index": None,
            "precip_mm": 0.0,
            "observation_time": "",
            "is_raining": False,
            "is_daylight": True,
            "is_hot": False,
            "is_cold": False,
            "is_windy": False,
            "is_clear": True,
            "forecast_tomorrow_condition": None,
            "forecast_tomorrow_max_c": None,
            "forecast_tomorrow_min_c": None,
            "forecast_tomorrow_precip_chance_pct": None,
            "forecast_tomorrow_precip_mm": None,
        }
        monkeypatch.setattr(
            _weather_mod,
            "_fetch_with_fallback",
            lambda *a, **kw: (sample_payload, "", ""),
        )

        ability = _weather_mod.WeatherAbility()
        result = ability.execute(
            channel="subagent",
            params={"location": "Test City"},  # NO _rich_media_ordinal
            telemetry=None,
        )

        assert isinstance(result, dict), (
            "Weather called without ordinal must return a dict — receiving a "
            "string means the trailer is being emitted on the subagent path."
        )
        assert "This tool supports rich-media rendering" not in json.dumps(result), (
            "Weather dict result must not contain the instruction trailer."
        )


class TestStripSpans:
    """The defensive scrub used at the subagent → parent boundary."""

    def test_single_span_is_unwrapped(self):
        text = "Before <span id='weather_1'>It is sunny.</span> after."
        assert strip_spans(text) == "Before It is sunny. after."

    def test_multiple_spans_all_unwrapped(self):
        text = (
            "<span id='weather_1'>Paris is 18°C.</span> "
            "Meanwhile <span id='weather_2'>Tokyo is 22°C.</span>"
        )
        assert strip_spans(text) == "Paris is 18°C. Meanwhile Tokyo is 22°C."

    def test_double_quotes_also_stripped(self):
        text = '<span id="weather_1">A.</span>'
        assert strip_spans(text) == "A."

    def test_multiline_synthesis_inside_span_preserved(self):
        text = "<span id='weather_1'>Line 1.\nLine 2.\nLine 3.</span>"
        assert strip_spans(text) == "Line 1.\nLine 2.\nLine 3."

    def test_no_spans_returned_unchanged(self):
        text = "Plain prose with <p>html</p> but no rich-media spans."
        assert strip_spans(text) == text

    def test_empty_input_returned_as_is(self):
        assert strip_spans("") == ""
        assert strip_spans(None) is None  # type: ignore[arg-type]

    def test_idempotent(self):
        text = "<span id='weather_1'>X.</span> middle <span id='weather_2'>Y.</span>"
        once = strip_spans(text)
        twice = strip_spans(once)
        assert once == twice

    def test_unrelated_html_spans_not_touched(self):
        """Only ``id='name_N'`` spans are stripped — other span tags pass through."""
        text = '<span class="hl">highlighted</span> normal text'
        assert strip_spans(text) == text
