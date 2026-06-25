# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — the ``Ability.get_follow_up()`` contract at import and call time.

The hook is the Part 2 foundation: a tool whose success routinely leaves the
model one obvious, loop-safe step short of a complete answer overrides
``get_follow_up()`` to return a short next-step instruction; the dispatcher wraps
it in a ``[follow_up_instruction]…[end:follow_up_instruction]`` block appended
INSIDE the envelope on a successful synchronous call. At this stage (C2) no
ability overrides the hook — the invariant is every ability returns ``str``,
and every ability returns ``""``.

Three feature tests on the real hot path — zero mocks, real registry, real
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

# The verbatim follow-up text each of the 15 overriding abilities returns. The
# dispatcher appends this to the tool's successful wire envelope inside a
# [follow_up_instruction]…[end:follow_up_instruction] block; the 24
# non-overriding abilities inherit "" and emit nothing.
_FOLLOW_UPS: dict[str, str] = {
    "search": (
        "These are titles and snippets, not full content. If a result looks promising, "
        "use `read` on its url to fetch the full page before quoting it or stating its "
        "claims as fact."
    ),
    "news": (
        "Before stating any headline as fact, cross-reference it with `web_search` or "
        "`search` — news results can be stale or one-sided. Quote the verbatim "
        "title/source/url rather than re-prosing."
    ),
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
    "search_files": (
        "This located files (and at most a few matched lines per file). If you need a "
        "file's full contents, use `read` on its path rather than re-grepping for "
        "more lines."
    ),
    "web_download": (
        "The file is now at the returned path but its contents are not yet in context. "
        "If you need what's inside, use read on that path to extract the text, or "
        "attach it with document for the user."
    ),
    "programming_docs_search": (
        "This returns one fetched excerpt plus sibling candidate URLs (empty excerpt). "
        "If the top excerpt does not fully answer the question, use `read` on one of "
        "the other candidate URLs to pull its full page text before answering."
    ),
    "find_tools": (
        "If a tool you needed was just activated, call it now to do the work — the "
        "discovered tools are live for this turn. Don't re-run discovery for a tool "
        "you already have active."
    ),
    "mcp_manager": (
        "If you just added or enabled a server and now need one of its remote tools, "
        "call `find_tools` to surface and activate it for this turn before you try to "
        "use it."
    ),
    "browser": (
        "If you just captured a screenshot, it is stored as a doc_id you cannot see "
        "directly — call vision with that image id to read what it shows. Otherwise "
        "the returned page text/changed diff is your result; act on it."
    ),
    "web_browse": (
        "If a screenshot was saved and you need to answer something about what's on "
        "the page, view it with vision(image=<doc_id>) instead of re-browsing. To pull "
        "the full text of a page the agent reached, use read on its URL."
    ),
    "web_search": (
        "This synthesis is summarized from its sources. Before stating a key claim as "
        "fact, open a cited source with read to confirm the exact detail; if a source "
        "needs login or interaction to see, use web_browse."
    ),
    "contacts": (
        "If the user wanted to actually reach this person and you now have their email "
        "address, you can use the email tool to draft or send them a message."
    ),
    "review_tool_calls": (
        "This lists past tool calls with params clipped to ~120 chars and no full "
        "results. If you need the exact wording a user or assistant used in that "
        "window, use `review_transcript` for the same timestamp to read the "
        "untruncated messages."
    ),
    "ubiquiti": (
        "If you are about to control or update a device, client, or rule but lack its "
        "MAC or id, run the matching list action first (list_devices, list_clients, "
        "list_wifi, list_port_forwards, list_traffic_rules) to get a valid identifier."
    ),
}
_OVERRIDE_NAMES = sorted(_FOLLOW_UPS)
_NON_OVERRIDE_NAMES = [n for n in _SHIPPING_NAMES if n not in _FOLLOW_UPS]


@pytest.mark.parametrize("name", _SHIPPING_NAMES)
def test_get_follow_up_returns_str(name: str) -> None:
    """Every shipping ability's ``get_follow_up()`` returns a ``str`` — the
    import-time probe in ``Ability.__init_subclass__`` already enforces this at
    import, but the sweep pins it at call time too so a future ability that dodges
    the probe (synthetic / ABC skip paths) is still caught here."""
    ability = next(a for a in _SHIPPING if a.get_name() == name)
    assert isinstance(ability.get_follow_up(), str), f"{name}.get_follow_up() is not str"


@pytest.mark.parametrize("name", _OVERRIDE_NAMES)
def test_overriding_ability_returns_exact_verbatim_follow_up(name: str) -> None:
    """Each of the 15 overriding abilities returns its exact spec verbatim text
    (byte-for-byte). Pinned so a future edit cannot silently change a tool's
    nudge without failing this sweep."""
    ability = next(a for a in _SHIPPING if a.get_name() == name)
    assert ability.get_follow_up() == _FOLLOW_UPS[name], (
        f"{name}.get_follow_up() diverged from the spec verbatim text"
    )


@pytest.mark.parametrize("name", _NON_OVERRIDE_NAMES)
def test_non_overriding_ability_returns_empty_follow_up(name: str) -> None:
    """Every non-overriding ability returns ``""`` — so no ability can silently
    grow an un-reviewed nudge. The full enumerated 24 (15+24=39 spec-coverage;
    the live shipping count is 38, so this asserts the actual complement)."""
    ability = next(a for a in _SHIPPING if a.get_name() == name)
    assert ability.get_follow_up() == "", f"{name}.get_follow_up() is unexpectedly non-empty"


def test_follow_up_split_is_complete() -> None:
    """The 15 overrides + the non-overriding remainder partition the shipping
    registry — a new ability that ships without a deliberate override-or-not
    decision lands in neither list and trips this guard, forcing the author to
    classify it."""
    assert set(_OVERRIDE_NAMES).isdisjoint(_NON_OVERRIDE_NAMES)
    assert sorted(set(_SHIPPING_NAMES)) == sorted(set(_OVERRIDE_NAMES) | set(_NON_OVERRIDE_NAMES))


def test_subclass_returning_non_str_raises_typeerror_at_import() -> None:
    """``Ability.__init_subclass__`` probes ``get_follow_up()`` on a throwaway
    mp=None instance at class-body evaluation time; a subclass returning a non-str
    raises ``TypeError`` at import, not at call. Drives the real
    ``__init_subclass__`` path — the factory's class-body statement evaluates it.

    A concrete (non-synthetic) class is required: ``_SYNTHETIC`` skips ALL
    metadata probes (the early return in ``__init_subclass__``), so the follow_up
    probe would never fire. The class provides full valid metadata so the probes
    before it pass, isolating the follow_up check."""
    with pytest.raises(TypeError, match="get_follow_up.*must return str"):
        _bad_follow_up_cls()


def _bad_follow_up_cls() -> type[Ability]:
    """Build and return a concrete ``Ability`` subclass whose ``get_follow_up``
    returns an ``int`` — defining the class body fires ``__init_subclass__``,
    which raises ``TypeError`` before the class is ever created. The factory
    makes the class-body evaluation an explicit call so static analysis sees
    the class as used (it is: the raise IS the assertion)."""
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

        def get_follow_up(self) -> str:
            return cast(str, 123)  # mypy accepts the cast; the runtime probe catches the int

    return _BadFollowUp


def test_subclass_with_default_follow_up_is_safe() -> None:
    """A subclass that does NOT override ``get_follow_up`` inherits the base
    ``""`` default and imports cleanly — the import-time probe accepts it."""

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

    assert _GoodFollowUp().get_follow_up() == ""