# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Timer-ability business-logic tests migrated from the per-ability conformance file ().

from typing import TYPE_CHECKING, cast

import pytest

from abilities.timer import TimerAbility
from contracts.params.timer_params_bag import TimerParamsBag
from services.dispatch_service import DispatchService
from tests._tool_result_harness import built

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

pytestmark = pytest.mark.unit


# (The dispatch-level contract tests that lived here —
# ``test_invalid_durations_are_invalid_duration_code``,
# ``test_long_title_truncated_to_80_chars``, ``test_max_duration_accepted`` —
# were removed during the old-spine ``ToolDispatcher`` cleanup: they drove
# ``tests._tool_result_harness.MP`` (a bare ``{uid, config}`` fake) through
# ``ToolDispatcher(mp).dispatch(...)``, which the new ``DispatchService.dispatch``
# does not support without a fully wired ``MessageProcessor`` (Rule 3/4 service
# coupling — sanitize_args, tool_call_service, policy gate, etc.). See the
# dead-code cleanup report for the systemic gap this exposed.)


def _render_rich(title: str, duration: int, ordinal: int = 1) -> str:
    tr = TimerAbility().run(built(TimerParamsBag.from_params({"title": title, "duration_seconds": duration})))
    return DispatchService(mp=cast("MessageProcessor", None))._render("timer", tr, ordinal)


# ── The countdown anchor: the real renderer → real parser chain ──────────────────
#
# These render the ability's ToolResult through the production formatter with a
# rich ordinal (the user-broadcast shape) so the persisted ``tool_calls.result``
# carries the card trailer, then feed it through the real rich_media_parser.parse
# exactly as the conversation-render path does.
#
# The card's wall-clock anchor rides the SEGMENT (``segment["created_at"]``), not
# the payload. That separation is the point: ``DispatchService._render_rich``
# REPLACES the model-visible tool body with the payload, so a timestamp in the
# payload is a timestamp the model reads and may try to reason about.


def test_anchor_rides_the_segment_and_never_the_model_visible_payload() -> None:
    from services.rich_media_parser import RichMediaParser

    raw = _render_rich("Pasta", 600)

    tool_calls = [{
        "tool_name": "timer",
        "params": "{}",
        "result": raw,
        "ephemeral": 1,
        "created_at": "2026-05-03 14:30:00",
    }]
    segments = RichMediaParser.parse("<span id='timer_1'>Started a 10-minute timer for the pasta.</span>", tool_calls)
    assert len(segments) == 1
    seg = segments[0]
    assert seg["type"] == "rich"
    assert seg["tag"] == "timer_1"
    assert cast(dict[str, object], seg["payload"])["title"] == "Pasta"
    assert cast(dict[str, object], seg["payload"])["duration_seconds"] == 600

    # The SQLite-format row timestamp is normalised to ISO-8601 UTC for the card.
    assert seg["created_at"] == "2026-05-03T14:30:00+00:00"

    # The model's view — the rendered body AND the payload — carries no timestamp.
    assert "2026-05-03" not in raw
    assert set(cast(dict[str, object], seg["payload"])) == {"title", "duration_seconds"}


def test_unparseable_row_timestamp_yields_no_anchor() -> None:
    """``parse_utc`` returns ``datetime.min`` (year 0001) on garbage rather than
    raising. The parser rejects that sentinel so the card falls through to its
    invalid-payload guard instead of counting down from year 0001."""
    from services.rich_media_parser import RichMediaParser

    tool_calls = [{
        "tool_name": "timer",
        "params": "{}",
        "result": _render_rich("Pasta", 600),
        "ephemeral": 1,
        "created_at": "this is not a date",
    }]
    segments = RichMediaParser.parse("<span id='timer_1'>Started.</span>", tool_calls)
    assert segments[0]["created_at"] is None


def test_missing_row_timestamp_yields_no_anchor() -> None:
    from services.rich_media_parser import RichMediaParser

    tool_calls = [{
        "tool_name": "timer",
        "params": "{}",
        "result": _render_rich("Pasta", 600),
        "ephemeral": 1,
        # created_at deliberately absent
    }]
    segments = RichMediaParser.parse("<span id='timer_1'>Started.</span>", tool_calls)
    assert segments[0]["created_at"] is None


def test_anchor_is_generic_segment_metadata_not_a_timer_special_case() -> None:
    """Every rich segment carries the anchor — it is "when this tool call ran",
    not a timer field. Other cards simply ignore it, and no payload is mutated to
    carry it."""
    from services.rich_media_parser import RichMediaParser

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
    segments = RichMediaParser.parse("<span id='weather_1'>Sunny.</span>", tool_calls)
    assert segments[0]["created_at"] == "2026-05-03T14:30:00+00:00"
    assert set(cast(dict[str, object], segments[0]["payload"])) == {"location", "temperature_c"}
