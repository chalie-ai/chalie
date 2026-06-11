# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the timer tool's ToolResult contract (TKT-903).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint against a real ``mp``-shaped
context, the real ``AbilityRegistry`` resolution of the production
``TimerAbility``, the real ``ToolDispatcher._prevalidate`` ACTION_REQUIRED
pre-gate, and the real ``PolicyManager.wrap`` gate reading the real ``policy``
table the ``db`` fixture seeds (``timer`` is seeded ``internal`` — always
allowed, no human prompt to hang on). The act-trail write is real too.

The contract this ticket migrates timer onto:
  - run() returns ``ToolResult.ok(payload, rich=payload)`` on every success;
    body and rich are the SAME ``{title, duration_seconds}`` dict.
  - The rich card path is via ``ToolResult(rich=…)`` — the dispatcher owns the
    ordinal + span instruction and injects them ONLY on a user-broadcast turn.
  - Errors carry stable kebab-case codes (``invalid-duration`` from run(),
    ``missing-params`` from the pre-gate) — NEVER ``code="error"``.
  - ``started_at`` (the wall-clock anchor) must NEVER appear in any LLM-visible
    surface; it is grafted server-side at parse time from the tool_calls row's
    ``created_at`` by ``TimerAbility.enrich_rich_payload`` (via
    ``services.rich_media_parser.parse``).

The parser tests feed the value production actually persists in
``tool_calls.result``: the dispatcher-rendered envelope produced by the ONE
production formatter (``ToolDispatcher._render``). Rendering the real ability's
``ToolResult`` through that formatter keeps the chain ability → renderer →
parser entirely on the production hot path.
"""

import json

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.timer import TimerAbility
from configs.channels import DmnConfig, UserConfig
from services.act_trail import ActTrail
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


# ── Contract: happy path, no card on a non-broadcast turn ────────────────────────


def test_non_broadcast_renders_compact_dict_body_no_trailer(db, dmn_mp):
    """On a non-user-broadcast turn the dispatcher drops the card: the body is the
    compact ``{"title":...,"duration_seconds":...}`` JSON with NO span/instruction
    trailer, and ``started_at`` appears nowhere in the rendered string."""
    out = ToolDispatcher(dmn_mp).dispatch(
        "timer", {"title": "Focus block", "duration_seconds": 1500, "act_summary": "x"}
    )

    assert "[timer(status=success" in out, out
    body = _body(out)
    assert "\n\n" not in body, "non-broadcast turn must not embed a rich trailer"
    assert "span id=" not in out and "You MUST present" not in out, out
    payload = json.loads(body)
    assert payload == {"title": "Focus block", "duration_seconds": 1500}, payload
    assert "started_at" not in out, "started_at must never reach the model"


# ── Contract: rich card path on a user-broadcast turn ────────────────────────────


def test_user_broadcast_renders_rich_trailer_with_span(db, user_mp):
    """On a user-broadcast turn the dispatcher pairs the card: the body becomes the
    rich payload JSON + blank line + the span instruction (``<span id='timer_1'>``).
    The card payload carries title + duration_seconds and NO ``started_at`` — the
    wall-clock anchor is grafted later by the parser."""
    out = ToolDispatcher(user_mp).dispatch(
        "timer", {"title": "Pasta", "duration_seconds": 600, "act_summary": "x"}
    )

    assert "[timer(status=success" in out, out
    assert "<span id='timer_1'>" in out, out
    body = _body(out)
    payload_json, _, instruction = body.partition("\n\n")
    payload = json.loads(payload_json)
    assert payload == {"title": "Pasta", "duration_seconds": 600}, payload
    assert "timer_1" in instruction, instruction
    assert "started_at" not in out, "started_at must never reach the model"


# ── Contract: missing title is a pre-gate missing-params error ───────────────────


def test_missing_title_pre_gate_missing_params(db, user_mp):
    """A call with no ``title`` is caught by the ACTION_REQUIRED pre-gate BEFORE
    run() — a single ``code=missing-params`` error naming the missing param. The
    old run()-level guard returned the banned ``code=error``."""
    out = ToolDispatcher(user_mp).dispatch(
        "timer", {"duration_seconds": 60, "act_summary": "x"}
    )
    assert "[timer(status=error" in out, out
    assert "code=missing-params" in out, out
    assert "title" in out, out

    # The act-trail recorded the same error envelope against the transcript.
    trail = ActTrail().fetch_by_transcript_id(user_mp.uid)
    assert any("code=missing-params" in row["result"] for row in trail), trail


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


def _render_rich(title: str, duration: int, ordinal: int = 1) -> str:
    """Render a timer result through the ONE production formatter with an ordinal
    — the exact string production persists to ``tool_calls.result`` on a
    user-broadcast turn (rich present → card trailer)."""
    tr = TimerAbility().run({"title": title, "duration_seconds": duration})
    return ToolDispatcher._render("timer", tr, ordinal)


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


# ── Registry ─────────────────────────────────────────────────────────────────────


def test_registered_in_ability_registry():
    """The registry walks backend/abilities/*.py and instantiates every concrete
    Ability subclass. timer.py must surface at NAME='timer'."""
    from abilities._registry import AbilityRegistry, _reset_for_tests

    _reset_for_tests()
    try:
        ability = AbilityRegistry.get("timer")
        assert isinstance(ability, TimerAbility)
    finally:
        _reset_for_tests()
