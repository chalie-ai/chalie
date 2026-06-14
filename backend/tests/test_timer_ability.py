# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Timer-ability-specific business-logic tests migrated from the per-ability
conformance file removed in TKT-975.

Covers the invalid-duration / missing-params error codes, title truncation, the
max-duration edge, and the full started_at injection chain via the real
``rich_media_parser.parse`` → ``TimerAbility.enrich_rich_payload`` path.
"""

import json

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.timer import TimerAbility
from configs.channels import DmnConfig, UserConfig
from tests._tool_result_harness import MP, body, seed_transcript

pytestmark = pytest.mark.unit


@pytest.fixture
def user_mp(db):
    """A real user-broadcast mp (``broadcast_to == 'user'``). On this channel the
    dispatcher assigns a rich-media ordinal and renders the timer card trailer."""
    return MP(seed_transcript(db, "chat", "start a timer"), UserConfig({}))


@pytest.fixture
def dmn_mp(db):
    """A real non-user-broadcast mp (``broadcast_to is None``). On this channel the
    dispatcher drops the rich card, so the rendered body is the raw model-facing
    payload dict with no span/instruction trailer."""
    return MP(seed_transcript(db, "subconscious", "start a timer"), DmnConfig())


def _body(rendered: str, tool: str = "timer") -> str:
    """Return the text between the open tag and ``[end:<tool>]``."""
    return body(rendered, tool)


def _render_rich(title: str, duration: int, ordinal: int = 1) -> str:
    """Render a timer result through the ONE production formatter with an ordinal
    — the exact string production persists to ``tool_calls.result`` on a
    user-broadcast turn (rich present → card trailer)."""
    tr = TimerAbility().run({"title": title, "duration_seconds": duration})
    return ToolDispatcher._render("timer", tr, ordinal)


# ── Contract: invalid durations are run()-side invalid-duration errors ───────────


def test_invalid_durations_are_invalid_duration_code(db, user_mp):
    """Out-of-range / wrong-type durations reach run() (title present, so the
    pre-gate passes) and return ``code=invalid-duration`` — not the banned
    ``code=error``. Note: ``duration_seconds=0`` is falsy → caught by the pre-gate
    as ``missing-params`` (documented fleet behaviour, asserted separately)."""
    for bad in (-5, 86401, "30"):
        out = ToolDispatcher(user_mp).dispatch(
            "timer", {"title": "Bad", "duration_seconds": bad, "act_summary": "x"}
        )
        assert "[timer(status=error" in out, (bad, out)
        assert "code=invalid-duration" in out, (bad, out)
        assert "code=error" not in out, (bad, out)


def test_zero_duration_hits_pre_gate_as_missing_params(db, user_mp):
    """``duration_seconds=0`` is falsy, so the pre-gate's ``not params.get(p)``
    check treats it as missing — a documented consequence of the fleet pre-gate,
    surfacing as ``missing-params`` rather than ``invalid-duration``."""
    out = ToolDispatcher(user_mp).dispatch(
        "timer", {"title": "Zero", "duration_seconds": 0, "act_summary": "x"}
    )
    assert "[timer(status=error" in out, out
    assert "code=missing-params" in out, out


# ── Contract: title truncation + max-duration edge ───────────────────────────────


def test_long_title_truncated_to_80_chars(db, dmn_mp):
    """A title longer than 80 chars is truncated in the body payload dict."""
    out = ToolDispatcher(dmn_mp).dispatch(
        "timer", {"title": "x" * 200, "duration_seconds": 60, "act_summary": "x"}
    )
    payload = json.loads(_body(out))
    assert len(payload["title"]) == 80, payload


def test_max_duration_accepted(db, dmn_mp):
    """The 24-hour upper bound (86400) is the documented edge — accepted."""
    out = ToolDispatcher(dmn_mp).dispatch(
        "timer", {"title": "Long", "duration_seconds": 86400, "act_summary": "x"}
    )
    assert "[timer(status=success" in out, out
    payload = json.loads(_body(out))
    assert payload["duration_seconds"] == 86400, payload


# ── started_at injection: the real renderer → real parser chain ──────────────────
#
# These render the ability's ToolResult through the production formatter with a
# rich ordinal (the user-broadcast shape) so the persisted ``tool_calls.result``
# carries the card trailer, then feed it through the real rich_media_parser.parse
# exactly as the conversation-render path does.


def test_parser_injects_started_at_from_created_at():
    """The wall-clock anchor lives in the tool_calls row's ``created_at`` and is
    grafted onto the timer card payload by ``rich_media_parser.parse`` →
    ``TimerAbility.enrich_rich_payload``. It surfaces on the FE-bound segment as an
    ISO 8601 string with explicit UTC offset, and NEVER on the LLM-bound result."""
    from services.rich_media_parser import parse

    raw = _render_rich("Pasta", 600)
    assert "started_at" not in raw

    tool_calls = [{
        "tool_name": "timer",
        "params": "{}",
        "result": raw,
        "ephemeral": 1,
        "created_at": "2026-05-03 14:30:00",
    }]
    segments = parse("<span id='timer_1'>Started a 10-minute timer for the pasta.</span>", tool_calls)
    assert len(segments) == 1
    seg = segments[0]
    assert seg["type"] == "rich"
    assert seg["tag"] == "timer_1"
    assert seg["payload"]["title"] == "Pasta"
    assert seg["payload"]["duration_seconds"] == 600
    assert seg["payload"]["started_at"] == "2026-05-03T14:30:00+00:00"


def test_parser_rejects_unparseable_created_at_sentinel():
    """``parse_utc`` returns ``datetime.min`` (year 0001) on garbage rather than
    raising. The enrich hook must reject that sentinel so the FE falls through to
    its "Invalid timer payload" guard instead of firing the alarm at year 0001."""
    from services.rich_media_parser import parse

    raw = _render_rich("Pasta", 600)
    tool_calls = [{
        "tool_name": "timer",
        "params": "{}",
        "result": raw,
        "ephemeral": 1,
        "created_at": "this is not a date",
    }]
    segments = parse("<span id='timer_1'>Started.</span>", tool_calls)
    assert "started_at" not in segments[0]["payload"]
    assert "started_at" not in raw


def test_parser_skips_injection_when_created_at_missing():
    """No ``created_at`` on the row → no fabricated ``started_at``; the FE hits its
    guard rather than rendering an undefined countdown."""
    from services.rich_media_parser import parse

    raw = _render_rich("Pasta", 600)
    tool_calls = [{
        "tool_name": "timer",
        "params": "{}",
        "result": raw,
        "ephemeral": 1,
        # created_at deliberately absent
    }]
    segments = parse("<span id='timer_1'>Started.</span>", tool_calls)
    assert segments[0]["payload"].get("started_at") is None
    assert "started_at" not in raw


def test_parser_does_not_inject_started_at_for_non_timer_tools():
    """The ``tool_name == "timer"`` gate ensures other rich-media tools (weather,
    list, …) never get a fabricated ``started_at`` even though their rows also
    carry a ``created_at`` column."""
    from services.rich_media_parser import parse

    weather_result = (
        '{"location": "Valletta", "temperature_c": 22}\n\n'
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
