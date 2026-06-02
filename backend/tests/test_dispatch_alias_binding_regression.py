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
   When ``abilities._registry`` is imported before ``abilities._base`` (the real
   application boot order), ``_populate_module_aliases()`` runs mid circular
   import, the ``from abilities._registry import AbilityRegistry`` raises and is
   silently swallowed, so the module-level ``AbilityRegistry`` / ``PolicyService``
   stay ``None``.  The first tool call then crashed every chat at turn zero with
   ``'NoneType' object has no attribute 'get'`` (the Turn-0 memory seed).
   ``Ability.dispatch`` must self-heal by re-binding the aliases on first use.

2. ``AbilityRegistry.get`` raises ``KeyError`` on an unknown tool, but
   ``dispatch`` was written to expect ``None``.  An unknown / hallucinated tool
   name must yield a graceful ``Unknown tool`` error, never an uncaught
   ``KeyError`` that kills the turn.
"""

import abilities._base as base
import pytest
from abilities._base import Ability, _populate_module_aliases

pytestmark = pytest.mark.unit


class _Cfg:
    channel = "user"
    broadcast_to = None


class _MP:
    config = _Cfg()
    cancel_event = None


def test_dispatch_rebinds_none_aliases():
    """dispatch must re-populate aliases when the circular import left them None."""
    saved = (base.AbilityRegistry, base.PolicyService, base.WebSocketBroker)
    try:
        base.AbilityRegistry = None
        base.PolicyService = None
        base.WebSocketBroker = None

        # Must not raise AttributeError('NoneType' ...); must resolve gracefully.
        out = Ability.dispatch(_MP(), "definitely_not_a_real_tool", {"action": "x"})

        assert "Unknown tool" in out
        assert base.AbilityRegistry is not None
        assert base.PolicyService is not None
    finally:
        base.AbilityRegistry, base.PolicyService, base.WebSocketBroker = saved


def test_dispatch_unknown_tool_is_graceful_not_keyerror():
    """An unknown tool name returns an error string, never an uncaught KeyError."""
    _populate_module_aliases()  # ensure bound regardless of import order
    out = Ability.dispatch(_MP(), "no_such_ability_xyz", {"action": "noop"})
    assert "Unknown tool" in out


def test_real_tool_resolves_after_rebind():
    """A genuinely registered ability is found once aliases are bound."""
    _populate_module_aliases()
    names = {a.NAME for a in base.AbilityRegistry.all()}
    assert "memory" in names
    assert "find_tools" in names
