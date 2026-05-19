"""Feature tests for TimerAbility (abilities/timer.py).

Real production object end-to-end. The ability is pure (no I/O collaborators),
so these are deterministic feature tests asserting the dispatch contract:
payload shape, ordinal-driven trailer behaviour, input validation, and the
explicit non-leak rule that ``started_at`` must NEVER appear in any LLM-visible
surface (the wall-clock anchor is injected server-side at parse time from the
tool_calls row's created_at — see services/rich_media_parser.py).
"""

import json

import pytest

from abilities.timer import TimerAbility

pytestmark = pytest.mark.unit


def test_no_ordinal_returns_dict_without_trailer():
    """When ordinal is absent the ability returns a plain dict — required by
    the subagent / action-button / direct-call paths that must not embed a
    rich-media instruction trailer. ``started_at`` is intentionally absent
    from this dict too — only the rich-media render path needs a wall-clock
    anchor and that is injected by the parser."""
    ability = TimerAbility()
    result = ability.execute(
        channel="subagent",
        params={"title": "Focus block", "duration_seconds": 1500},
        telemetry=None,
    )
    assert isinstance(result, dict)
    assert result["title"] == "Focus block"
    assert result["duration_seconds"] == 1500
    assert "started_at" not in result


def test_ordinal_returns_string_with_trailer():
    """When ordinal is supplied (user-channel dispatch) the return is a
    `<JSON>\\n\\n<instruction>` string. The instruction must reference the
    correct ordinal-derived span tag. The JSON body must NOT contain
    ``started_at`` — keeping the LLM ignorant of the wall-clock is a hard
    invariant of the timer contract."""
    ability = TimerAbility()
    result = ability.execute(
        channel="user",
        params={"title": "Pasta", "duration_seconds": 600, "_rich_media_ordinal": 2},
        telemetry=None,
    )
    assert isinstance(result, str)
    body, _, instruction = result.partition("\n\n")
    payload = json.loads(body)
    assert payload["title"] == "Pasta"
    assert payload["duration_seconds"] == 600
    assert "started_at" not in payload, "started_at must never be exposed to the LLM"
    assert "started_at" not in result, "started_at must not appear anywhere in tool result"
    assert "timer_2" in instruction
    assert "<span id='timer_2'>" in instruction


def test_missing_title_returns_error_dict():
    ability = TimerAbility()
    result = ability.execute(
        channel="user",
        params={"duration_seconds": 60, "_rich_media_ordinal": 1},
        telemetry=None,
    )
    assert isinstance(result, dict)
    assert "error" in result


def test_invalid_duration_returns_error_dict():
    ability = TimerAbility()
    for bad in (0, -5, 86401, "30", None):
        result = ability.execute(
            channel="user",
            params={"title": "Bad", "duration_seconds": bad, "_rich_media_ordinal": 1},
            telemetry=None,
        )
        assert isinstance(result, dict), f"expected error dict for {bad!r}"
        assert "error" in result, f"expected error key for {bad!r}"


def test_long_title_truncated_to_80_chars():
    ability = TimerAbility()
    long_title = "x" * 200
    result = ability.execute(
        channel="subagent",
        params={"title": long_title, "duration_seconds": 60},
        telemetry=None,
    )
    assert len(result["title"]) == 80


def test_parser_skips_injection_when_created_at_missing():
    """If the tool_calls row has no ``created_at`` key, the parser must not
    fabricate a ``started_at`` — the FE will hit its "Invalid timer payload"
    guard rather than silently rendering an undefined countdown.
    """
    from services.rich_media_parser import parse

    raw = TimerAbility().execute(
        channel="user",
        params={"title": "Pasta", "duration_seconds": 600, "_rich_media_ordinal": 1},
        telemetry=None,
    )
    tool_calls = [{
        "tool_name": "timer",
        "params": "{}",
        "result": raw,
        "ephemeral": 1,
        # created_at deliberately absent
    }]
    segments = parse("<span id='timer_1'>Started.</span>", tool_calls)
    assert segments[0]["payload"].get("started_at") is None


def test_parser_rejects_unparseable_created_at_sentinel():
    """``parse_utc`` returns ``datetime.min`` (year 0001) on garbage rather
    than raising. Without an explicit sentinel-rejection, that would propagate
    as a truthy ISO string and the FE would treat the timer as instantly
    expired and fire the alarm. The parser must reject the sentinel and
    return None so the FE falls through cleanly.
    """
    from services.rich_media_parser import parse

    raw = TimerAbility().execute(
        channel="user",
        params={"title": "Pasta", "duration_seconds": 600, "_rich_media_ordinal": 1},
        telemetry=None,
    )
    tool_calls = [{
        "tool_name": "timer",
        "params": "{}",
        "result": raw,
        "ephemeral": 1,
        "created_at": "this is not a date",
    }]
    segments = parse("<span id='timer_1'>Started.</span>", tool_calls)
    assert "started_at" not in segments[0]["payload"]


def test_parser_does_not_inject_started_at_for_non_timer_tools():
    """The ``tool_name == "timer"`` gate ensures other rich-media tools (weather,
    list, …) never get a fabricated ``started_at`` even though their rows also
    carry a ``created_at`` column. Verifying the gate prevents future drift."""
    from services.rich_media_parser import parse

    weather_result = (
        '{"location_name": "Valletta", "temperature_c": 22}\n\n'
        "Tool supports rich-media. <span id='weather_1'>Sunny.</span>"
    )
    tool_calls = [{
        "tool_name": "weather",
        "params": "{}",
        "result": weather_result,
        "ephemeral": 1,
        "created_at": "2026-05-03 14:30:00",
    }]
    segments = parse("<span id='weather_1'>Sunny.</span>", tool_calls)
    assert "started_at" not in segments[0]["payload"]


def test_parser_injects_started_at_from_tool_calls_created_at():
    """The wall-clock anchor lives in the tool_calls row's ``created_at`` and
    is grafted onto the timer payload by ``rich_media_parser.parse``. This is
    the only place ``started_at`` exists — verify the FE-bound segment carries
    it and the LLM-bound result string does not."""
    from services.rich_media_parser import parse

    ability = TimerAbility()
    raw = ability.execute(
        channel="user",
        params={"title": "Pasta", "duration_seconds": 600, "_rich_media_ordinal": 1},
        telemetry=None,
    )
    assert "started_at" not in raw

    tool_calls = [{
        "tool_name": "timer",
        "params": "{}",
        "result": raw,
        "ephemeral": 1,
        "created_at": "2026-05-03 14:30:00",
    }]
    content = "<span id='timer_1'>Started a 10-minute timer for the pasta.</span>"

    segments = parse(content, tool_calls)
    assert len(segments) == 1
    seg = segments[0]
    assert seg["type"] == "rich"
    assert seg["tag"] == "timer_1"
    assert seg["payload"]["title"] == "Pasta"
    assert seg["payload"]["duration_seconds"] == 600
    # ISO 8601 with explicit UTC offset — timer.js parses this with Date.parse().
    assert seg["payload"]["started_at"] == "2026-05-03T14:30:00+00:00"


def test_max_duration_accepted():
    """24-hour upper bound is the documented edge — exercise it explicitly."""
    ability = TimerAbility()
    result = ability.execute(
        channel="subagent",
        params={"title": "Long", "duration_seconds": 86400},
        telemetry=None,
    )
    assert result["duration_seconds"] == 86400


def test_registered_in_ability_registry():
    """The registry walks backend/abilities/*.py and instantiates every
    concrete Ability subclass. timer.py must surface at NAME='timer'."""
    from abilities._registry import AbilityRegistry, _reset_for_tests

    _reset_for_tests()
    try:
        ability = AbilityRegistry.get("timer")
        assert isinstance(ability, TimerAbility)
    finally:
        _reset_for_tests()
