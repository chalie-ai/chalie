# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — the ``Ability.get_follow_up(tr)`` contract at import and call time.

A tool whose success routinely leaves the model one obvious, loop-safe step short
of a complete answer overrides ``get_follow_up(tr)`` to return a short next-step
instruction; the dispatcher wraps it in a
``[follow_up_instruction]…[end:follow_up_instruction]`` block appended INSIDE the
envelope on a successful synchronous call. The hook is RESULT-AWARE: it receives
the success :class:`ToolResult` being rendered, so an override can interpolate
live values straight from the result the model already sees (the downloaded
``path``, the activated tool ``name``, the anchor ``date_time``) — present data
lifts compliance over a generic nudge.

Two override shapes ship:
  * STATIC — same text on every success, ignores ``tr`` (document, email, search,
    news, web_browse).
  * DYNAMIC — interpolates a live value off ``tr``; degrades to ``""`` when the
    result lacks the data it needs (find_tools, mcp_manager, web_download,
    review_tool_calls, programming_docs_search).

Every override MUST return ``""`` on a shapeless ``ToolResult.ok("")`` — that is
the exact probe ``Ability.__init_subclass__`` fires at import, so a dynamic
override that assumes a body shape would crash the registry. These feature tests
run the real hot path — zero mocks, real registry, real
``Ability.__init_subclass__``. The discovery walk mirrors
``test_ability_returns_tool_result`` (real ``AbilityRegistry.all()``); the
shipping-only filter mirrors ``test_param_key_resilience._shipping_abilities``
(test abilities leak into ``all()`` during a full pytest run).
"""

from __future__ import annotations

from typing import cast

import pytest

from abilities._ability import Ability
from abilities._registry import AbilityRegistry
from abilities._result import ToolResult

pytestmark = pytest.mark.unit

_SHIPPING = sorted(
    (a for a in AbilityRegistry.all() if type(a).__module__.startswith("abilities.")),
    key=lambda a: a.get_name(),
)
_SHIPPING_NAMES = [a.get_name() for a in _SHIPPING]

# STATIC overriders — verbatim text returned on every success regardless of tr.
_STATIC_FOLLOW_UPS: dict[str, str] = {
    "document": (
        "search and list only return document ids, names and snippets, not full text. "
        "If you need the actual content of a result, call document again with "
        "action=view and that id."
    ),
    "email": (
        "search results are snippets only, not the full message. If you need the "
        "complete body of an email before acting on it, call email again with "
        "action=read and that uid."
    ),
    "search": (
        "For the pages that are aligned with your query, use the `read(url=…)` tool "
        "with that result's url to get the full content of the page before quoting it "
        "or stating its claims as fact."
    ),
    "news": (
        "If the results don't match the context you're looking for, pivot to using "
        "the `search` tool to get data from other sources online."
    ),
    "web_browse": "To get a full textual version of this page call the `read` tool.",
}

# DYNAMIC overriders — name → (representative success ToolResult, expected text the
# override interpolates from it). Each MUST also collapse to "" on ToolResult.ok("").
_DYNAMIC_FOLLOW_UPS: dict[str, tuple[ToolResult, str]] = {
    "find_tools": (
        ToolResult.ok({"injected": [{"name": "calendar"}], "not_found": []}),
        "`calendar` is now available and can be called.",
    ),
    "mcp_manager": (
        ToolResult.ok({"id": "x", "name": "weather", "url": "h", "status": "online"}),
        "`weather` is now available. Call `find_tools` with `weather` and the action "
        "you want to perform to get its tools in context.",
    ),
    "web_download": (
        ToolResult.ok({"path": "/tmp/x/file.pdf", "bytes": 1, "content_type": "application/pdf"}),
        "File downloaded. Use the `read(/tmp/x/file.pdf)` tool to fetch its content.",
    ),
    "review_tool_calls": (
        ToolResult.ok([{"iter": 1}], anchor="2026-04-07T14:30:00+00:00"),
        "Params are clipped to ~120 chars and full results are omitted. For the exact "
        "wording a user or assistant used in that window, call "
        "`review_transcript(date_time=2026-04-07T14:30:00+00:00)` to read the "
        "untruncated messages.",
    ),
    "programming_docs_search": (
        ToolResult.ok([{"source": "Py", "title": "t", "url": "https://docs.python.org/x", "excerpt": "e"}]),
        "The excerpt is capped. Use the `read(https://docs.python.org/x)` tool to "
        "fetch the full document contents.",
    ),
}

_OVERRIDE_NAMES = sorted(set(_STATIC_FOLLOW_UPS) | set(_DYNAMIC_FOLLOW_UPS))
_NON_OVERRIDE_NAMES = [n for n in _SHIPPING_NAMES if n not in _OVERRIDE_NAMES]


def _by_name(name: str) -> Ability:
    return next(a for a in _SHIPPING if a.get_name() == name)


@pytest.mark.parametrize("name", _SHIPPING_NAMES)
def test_get_follow_up_returns_str_on_empty_result(name: str) -> None:
    """Every shipping ability's ``get_follow_up(tr)`` returns a ``str`` on the
    shapeless probe result — the exact call ``Ability.__init_subclass__`` makes at
    import. The sweep pins it at call time too so a future ability that dodges the
    probe (synthetic / ABC skip paths) is still caught here."""
    assert isinstance(_by_name(name).get_follow_up(ToolResult.ok("")), str), (
        f"{name}.get_follow_up() is not str"
    )


@pytest.mark.parametrize("name", sorted(_STATIC_FOLLOW_UPS))
def test_static_override_returns_exact_verbatim_text(name: str) -> None:
    """Each static overrider returns its exact spec verbatim text (byte-for-byte)
    on any success — pinned so a future edit cannot silently change a tool's nudge.
    A representative non-empty body proves it ignores ``tr`` rather than reacting
    to it."""
    ability = _by_name(name)
    assert ability.get_follow_up(ToolResult.ok("anything")) == _STATIC_FOLLOW_UPS[name]
    # A static override is result-agnostic: identical text on the empty probe too.
    assert ability.get_follow_up(ToolResult.ok("")) == _STATIC_FOLLOW_UPS[name]


@pytest.mark.parametrize("name", sorted(_DYNAMIC_FOLLOW_UPS))
def test_dynamic_override_interpolates_live_value(name: str) -> None:
    """Each dynamic overrider interpolates the live value off a representative
    success ``ToolResult`` — proving the present-in-context data reaches the nudge
    verbatim."""
    representative, expected = _DYNAMIC_FOLLOW_UPS[name]
    assert _by_name(name).get_follow_up(representative) == expected


@pytest.mark.parametrize("name", sorted(_DYNAMIC_FOLLOW_UPS))
def test_dynamic_override_degrades_to_empty_without_data(name: str) -> None:
    """Each dynamic overrider returns ``""`` when the result lacks the data it
    interpolates — the import probe fires it on ``ToolResult.ok("")``, so a shape
    assumption would crash the registry. This is the contract that lets a dynamic
    override ship at all."""
    assert _by_name(name).get_follow_up(ToolResult.ok("")) == ""


@pytest.mark.parametrize("name", _NON_OVERRIDE_NAMES)
def test_non_overriding_ability_returns_empty_follow_up(name: str) -> None:
    """Every non-overriding ability returns ``""`` for any success — so no ability
    can silently grow an un-reviewed nudge."""
    ability = _by_name(name)
    assert ability.get_follow_up(ToolResult.ok("")) == "", f"{name} unexpectedly non-empty"
    assert ability.get_follow_up(ToolResult.ok("body")) == "", f"{name} unexpectedly non-empty"


def test_follow_up_split_is_complete() -> None:
    """The overrides + the non-overriding remainder partition the shipping
    registry — a new ability that ships without a deliberate override-or-not
    decision lands in neither list and trips this guard, forcing the author to
    classify it."""
    assert set(_OVERRIDE_NAMES).isdisjoint(_NON_OVERRIDE_NAMES)
    assert sorted(set(_SHIPPING_NAMES)) == sorted(set(_OVERRIDE_NAMES) | set(_NON_OVERRIDE_NAMES))


def test_subclass_returning_non_str_raises_typeerror_at_import() -> None:
    """``Ability.__init_subclass__`` probes ``get_follow_up(ToolResult.ok(""))`` on
    a throwaway mp=None instance at class-body evaluation time; a subclass returning
    a non-str raises ``TypeError`` at import, not at call. Drives the real
    ``__init_subclass__`` path — the factory's class-body statement evaluates it.

    A concrete (non-synthetic) class is required: ``_SYNTHETIC`` skips ALL metadata
    probes (the early return in ``__init_subclass__``), so the follow_up probe would
    never fire. The class provides full valid metadata so the probes before it
    pass, isolating the follow_up check."""
    with pytest.raises(TypeError, match="get_follow_up.*must return str"):
        _bad_follow_up_cls()


def _bad_follow_up_cls() -> type[Ability]:
    """Build and return a concrete ``Ability`` subclass whose ``get_follow_up``
    returns an ``int`` — defining the class body fires ``__init_subclass__``, which
    raises ``TypeError`` before the class is ever created. The factory makes the
    class-body evaluation an explicit call so static analysis sees the class as
    used (it is: the raise IS the assertion)."""
    class _BadFollowUp(Ability):
        def get_name(self) -> str:
            return "bad_follow_up"

        def get_summary(self) -> str:
            return "probe"

        def get_examples(self) -> list[str]:
            return ["a"] * 6

        def get_search_tooltip(self) -> str:
            return "probe"

        def get_parameters(self) -> dict[str, object]:
            return {"type": "object", "properties": {}, "required": []}

        def run(self, params: dict[str, object]) -> ToolResult:
            return ToolResult.ok("x")

        def get_follow_up(self, tr: ToolResult) -> str:
            return cast(str, 123)  # mypy accepts the cast; the runtime probe catches the int

    return _BadFollowUp


def test_subclass_with_default_follow_up_is_safe() -> None:
    """A subclass that does NOT override ``get_follow_up`` inherits the base ``""``
    default and imports cleanly — the import-time probe accepts it."""

    class _GoodFollowUp(Ability):
        def get_name(self) -> str:
            return "good_follow_up"

        def get_summary(self) -> str:
            return "probe"

        def get_examples(self) -> list[str]:
            return ["a"] * 6

        def get_search_tooltip(self) -> str:
            return "probe"

        def get_parameters(self) -> dict[str, object]:
            return {"type": "object", "properties": {}, "required": []}

        def run(self, params: dict[str, object]) -> ToolResult:
            return ToolResult.ok("x")

    assert _GoodFollowUp().get_follow_up(ToolResult.ok("")) == ""
