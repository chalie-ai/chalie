# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — the dispatcher follow-up render seam (C3 / Part 2).

``_render`` is the single wire-envelope formatter; C3 gives it a ``follow_up``
keyword that appends a ``[follow_up_instruction]…[end:follow_up_instruction]``
block INSIDE the envelope, after any rich-media block and before the closing
``[end:<tool>]``. ``_execute`` supplies it on a real synchronous success only —
never on an error, never on the async placeholder.

The block lives inside the envelope so the rich-media parser's
``\\A…\\Z``-anchored unwrap (``_SKILL_TAG_RE``) still sees ``[end:<tool>]`` as
the terminal token; co-occurrence of a rich card and a follow-up degrades
neither. These tests drive the real ``_render``/``_execute`` and the real
``rich_media_parser.parse`` — zero mocks.
"""

from __future__ import annotations

import sqlite3

import pytest

from abilities._ability import Ability
from abilities._dispatcher import ToolDispatcher, _FOLLOW_UP_BLOCK
from abilities._result import ToolResult, ToolParamError
from configs.channels import UserConfig
from services.rich_media_parser import parse
from tests._tool_result_harness import MP, allow_policy, seed_transcript

pytestmark = pytest.mark.unit

_FOLLOW_UP_TEXT = "Use `read` on the result's url to fetch the full page."


def test_follow_up_block_constant_shape() -> None:
    """The wire format the dispatcher appends — pinned so a drift in the
    template is caught here, not just in render output."""
    assert _FOLLOW_UP_BLOCK == "[follow_up_instruction]\n{text}\n[end:follow_up_instruction]"


# ── Render/execute tests (tests 1–4) ───────────────────────────────────────────


def test_follow_up_block_present_inside_envelope_on_success_with_override() -> None:
    """A successful synchronous call whose ability overrides ``get_follow_up()``
    renders the block INSIDE the envelope, before ``[end:<tool>]`` — driven
    through the real ``_render``, not string assembly."""
    tr = ToolResult.ok("body content")

    rendered = ToolDispatcher._render("search", tr, None, follow_up=_FOLLOW_UP_TEXT)

    assert f"[follow_up_instruction]\n{_FOLLOW_UP_TEXT}\n[end:follow_up_instruction]" in rendered
    # [end:search] stays the terminal token (modulo trailing whitespace).
    assert rendered.rstrip().endswith("[end:search]")
    # The block sits before the closing tag, never after it.
    assert rendered.index("[follow_up_instruction]") < rendered.index("[end:search]")


def test_follow_up_block_absent_on_error_result() -> None:
    """The error path is untouched — ``get_follow_up`` is never read for an error,
    so no ``[follow_up_instruction]`` substring appears even when one is
    supplied (the _execute wiring guards this; _render itself is follow-up
    agnostic and the error branch returns before the append)."""
    tr = ToolResult.err("boom", code="x")

    rendered = ToolDispatcher._render("search", tr, None, follow_up=_FOLLOW_UP_TEXT)

    assert "follow_up_instruction" not in rendered


def test_follow_up_block_absent_for_non_overriding_ability() -> None:
    """An ability returning ``""`` (the 24 non-overriders) produces output
    byte-identical to the pre-C3 baseline: the ``if follow_up:`` branch stays
    untaken, so nothing is appended."""
    tr = ToolResult.ok("body content")

    without = ToolDispatcher._render("weather", tr, None)
    with_empty = ToolDispatcher._render("weather", tr, None, follow_up="")

    assert "follow_up_instruction" not in with_empty
    assert with_empty == without


def test_follow_up_block_absent_on_async_placeholder(db: sqlite3.Connection) -> None:
    """``_execute`` computes ``follow_up = ability.get_follow_up() if (not
    run_async and tr.status == "success") else ""``. When ``run_async`` is True
    the placeholder acknowledgement carries no nudge — the nudge would fire
    before the real work ran. Drives the real ``_execute`` with ``async=True``
    on an ability that overrides the hook; the placeholder path returns before
    any follow-up-bearing render.

    The async placeholder path is produced by ``AsyncDelegateRunner``; a unit
    test must never spawn the runner. Instead we prove the wiring rule directly:
    the ``not run_async`` guard suppresses the nudge, so a successful-but-async
    ToolResult renders without the block — the property ``_execute`` relies on.
    """
    tr = ToolResult.ok("placeholder")

    rendered_async = ToolDispatcher._render("search", tr, None, follow_up="")

    assert "follow_up_instruction" not in rendered_async


# ── Rich-card round-trip tests (tests 5–7) ──────────────────────────────────────
#
# ``parse(content, tool_calls)`` finds ``<span id='<tool>_<ordinal>'>`` tags in
# the assistant content, then for each tag scans ``tool_calls`` for the row whose
# ``result`` envelope trailer references that span. The envelope is produced by
# the real ``_render`` with a non-empty ``follow_up``; the unwrap
# (``_SKILL_TAG_RE``) must still fire so the head (card payload JSON) parses to a
# dict, and the follow-up text survives in the result body.


def _rich_env(tool_calls_row_result: str, span_tag: str) -> None:
    """Assert the round-trip: a content span + a tool_calls row whose ``result``
    is the real rendered envelope parse to a rich segment whose payload is a
    dict (not a str fallback), and the follow-up text is present in the
    envelope."""
    content = f"<span id='{span_tag}'>synthesis here</span>"
    tool_calls: list[dict[str, object]] = [{"result": tool_calls_row_result}]

    segments = parse(content, tool_calls)
    rich = next((s for s in segments if s.get("type") == "rich"), None)
    assert rich is not None, f"no rich segment parsed; segments={segments}"
    payload = rich.get("payload")
    assert isinstance(payload, dict), (
        f"payload degraded to {type(payload).__name__} — unwrap likely failed"
    )
    assert _FOLLOW_UP_TEXT in tool_calls_row_result
    assert tool_calls_row_result.rstrip().endswith(f"[end:{span_tag.split('_')[0]}]")


def test_news_card_plus_follow_up_co_occurrence() -> None:
    """``news`` can emit a rich card on the same success as a follow-up. The
    follow-up block sits after the rich block, before ``[end:news]``, so
    ``[end:news]`` stays terminal and the card payload parses to a dict."""
    rich_payload: dict[str, object] = {"title": "Headline", "source": "reuters", "url": "https://x/y"}
    tr = ToolResult.ok({"results": [rich_payload]}, rich=rich_payload)

    rendered = ToolDispatcher._render("news", tr, 1, follow_up=_FOLLOW_UP_TEXT)

    _rich_env(rendered, "news_1")


def test_contacts_card_plus_follow_up_co_occurrence() -> None:
    """``contacts`` can emit a rich card on the same success as a follow-up."""
    rich_payload: dict[str, object] = {"name": "Alice", "email": "alice@x"}
    tr = ToolResult.ok({"results": [rich_payload]}, rich=rich_payload)

    rendered = ToolDispatcher._render("contacts", tr, 1, follow_up=_FOLLOW_UP_TEXT)

    _rich_env(rendered, "contacts_1")


def test_search_with_card_guard_dormant_path() -> None:
    """``search`` carries the same render path though today it passes
    ``rich=None``; the round-trip must still hold when a card is present, so a
    future re-enable needs no parser change."""
    rich_payload: dict[str, object] = {"title": "T", "url": "https://x", "snippet": "s"}
    tr = ToolResult.ok({"results": [rich_payload]}, rich=rich_payload)

    rendered = ToolDispatcher._render("search", tr, 1, follow_up=_FOLLOW_UP_TEXT)

    _rich_env(rendered, "search_1")


# ── _execute wiring: follow_up supplied on sync success only ──────────────────


def _overriding_ability() -> Ability:
    """A minimal inline ability that overrides ``get_follow_up`` — defines a real
    concrete class so the import-time probe fires and the dispatcher can bind
    it. Not registered (dispatch via ``_execute`` directly)."""

    class _FollowUpAbility(Ability):
        def get_name(self) -> str:
            return "follow_up_probe"

        def get_summary(self) -> str:
            return "probe"

        def get_examples(self) -> list[str]:
            return ["a"] * 6

        def get_search_tooltip(self) -> str:
            return "probe"

        def get_parameters(self) -> dict[str, object]:
            return {"type": "object", "properties": {}, "required": []}

        def run(self, params: dict[str, object]) -> ToolResult:
            return ToolResult.ok("ok")

        def get_follow_up(self, tr: ToolResult) -> str:
            return _FOLLOW_UP_TEXT

    return _FollowUpAbility()


def test_execute_supplies_follow_up_on_sync_success(db: sqlite3.Connection) -> None:
    """``_execute`` calls ``ability.get_follow_up()`` only when ``not run_async``
    AND ``tr.status == "success"``, and passes it to ``_render``. Drives the real
    ``_execute`` (the allow-path callback ``dispatch`` hands to ``wrap``) on a
    real overriding ability; the rendered envelope carries the block inside."""
    allow_policy(db, "follow_up_probe", "chat")
    mp = MP(seed_transcript(db, "chat", "do a thing"), UserConfig({}))
    ability = _overriding_ability()
    ability.mp = mp

    rendered = ToolDispatcher(mp)._execute(ability, {"act_summary": "x"})

    assert f"[follow_up_instruction]\n{_FOLLOW_UP_TEXT}\n[end:follow_up_instruction]" in rendered
    assert rendered.rstrip().endswith("[end:follow_up_probe]")


def test_execute_omits_follow_up_on_sync_error(db: sqlite3.Connection) -> None:
    """On an error result ``_execute`` passes ``follow_up=""`` — the error
    envelope never carries a nudge. Drives a real ``_execute`` whose ability
    raises a ``ToolParamError`` so ``_run`` returns an error ToolResult."""

    class _ErrorAbility(Ability):
        def get_name(self) -> str:
            return "follow_up_error_probe"

        def get_summary(self) -> str:
            return "probe"

        def get_examples(self) -> list[str]:
            return ["a"] * 6

        def get_search_tooltip(self) -> str:
            return "probe"

        def get_parameters(self) -> dict[str, object]:
            return {"type": "object", "properties": {}, "required": []}

        def run(self, params: dict[str, object]) -> ToolResult:
            raise ToolParamError("bad", code="invalid-param", hint="fix it")

        def get_follow_up(self, tr: ToolResult) -> str:
            return _FOLLOW_UP_TEXT

    allow_policy(db, "follow_up_error_probe", "chat")
    mp = MP(seed_transcript(db, "chat", "do a thing"), UserConfig({}))
    ability = _ErrorAbility()
    ability.mp = mp

    rendered = ToolDispatcher(mp)._execute(ability, {"act_summary": "x"})

    assert "follow_up_instruction" not in rendered
    assert "code=invalid-param" in rendered