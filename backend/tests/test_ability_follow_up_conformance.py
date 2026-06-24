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


@pytest.mark.parametrize("name", _SHIPPING_NAMES)
def test_get_follow_up_returns_str(name: str) -> None:
    """Every shipping ability's ``get_follow_up()`` returns a ``str`` — the
    import-time probe in ``Ability.__init_subclass__`` already enforces this at
    import, but the sweep pins it at call time too so a future ability that dodges
    the probe (synthetic / ABC skip paths) is still caught here."""
    ability = next(a for a in _SHIPPING if a.get_name() == name)
    assert isinstance(ability.get_follow_up(), str), f"{name}.get_follow_up() is not str"


@pytest.mark.parametrize("name", _SHIPPING_NAMES)
def test_get_follow_up_is_empty_at_c2_baseline(name: str) -> None:
    """At C2 (no overrides exist yet) every shipping ability returns ``""``. This
    is the baseline; later stages refine it to assert the exact-text overrides
    alongside the empty-string non-overriders."""
    ability = next(a for a in _SHIPPING if a.get_name() == name)
    assert ability.get_follow_up() == "", f"{name}.get_follow_up() is non-empty at C2"


def test_subclass_returning_non_str_raises_typeerror_at_import() -> None:
    """``Ability.__init_subclass__`` probes ``get_follow_up()`` on a throwaway
    mp=None instance at class-body evaluation time; a subclass returning a non-str
    raises ``TypeError`` at import, not at call. Drives the real
    ``__init_subclass__`` path — defining the class evaluates it.

    A concrete (non-synthetic) class is required: ``_SYNTHETIC`` skips ALL
    metadata probes (the early return in ``__init_subclass__``), so the follow_up
    probe would never fire. The class provides full valid metadata so the probes
    before it pass, isolating the follow_up check."""

    with pytest.raises(TypeError, match="get_follow_up.*must return str"):

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