# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Regression guards for two boot-order defects found during the rc-0.9.0
ACT-loop e2e verification (NOT part of the §9a blind-spec set).

1. Lazy module aliases left ``None`` by the _base <-> _registry circular import.
   ``Ability.use`` must self-heal by re-binding the aliases (incl. PolicyManager)
   on first use, or the Turn-0 memory seed crashes the chat.
2. ``AbilityRegistry.get`` raises ``KeyError`` on an unknown tool; ``match`` must
   turn that into ``None`` so ``use`` returns a graceful ``Unknown tool`` string.
"""

import abilities._base as base
import pytest
from abilities._base import Ability, _populate_module_aliases

pytestmark = pytest.mark.unit


class _Cfg:
    channel = "user"
    policy_channel = None        # match() short-circuits before the gate for an unknown tool
    broadcast_to = None


class _MP:
    config = _Cfg()
    uid = None
    cancel_event = None


def test_use_rebinds_none_aliases():
    """use must re-populate aliases when the circular import left them None."""
    saved = (base.AbilityRegistry, base.PolicyManager)
    try:
        base.AbilityRegistry = None
        base.PolicyManager = None

        out = Ability.use(_MP(), "definitely_not_a_real_tool", {"action": "x"})

        assert "Unknown tool" in out
        assert base.AbilityRegistry is not None
        assert base.PolicyManager is not None
    finally:
        base.AbilityRegistry, base.PolicyManager = saved


def test_use_unknown_tool_is_graceful_not_keyerror():
    """An unknown tool name returns an error string, never an uncaught KeyError."""
    _populate_module_aliases()
    out = Ability.use(_MP(), "no_such_ability_xyz", {"action": "noop"})
    assert "Unknown tool" in out


def test_real_tool_resolves_after_rebind():
    """A genuinely registered ability is found once aliases are bound."""
    _populate_module_aliases()
    names = {a.NAME for a in base.AbilityRegistry.all()}
    assert "memory" in names
    assert "find_tools" in names
