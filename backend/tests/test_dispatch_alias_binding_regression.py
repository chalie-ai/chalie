# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Regression guard for unknown-tool dispatch resolution (rc-0.9.0 ACT-loop e2e).

The lazy-module-alias self-heal that this file used to guard was deleted in P7:
``ToolDispatcher`` (``abilities/_dispatcher.py``) now imports ``AbilityRegistry``
and ``PolicyManager`` normally, so the circular-import boot defect can no longer
arise. What remains worth guarding is the graceful path: ``_match`` /
``_bind`` must turn ``AbilityRegistry.get``'s ``KeyError`` into a ``None`` so
``dispatch`` returns an ``Unknown tool`` string instead of crashing a turn.

Spec: eliminate-_base §4.9 (repoint the regression test at the real collaborators).
"""

import pytest
from abilities._dispatcher import ToolDispatcher
from abilities._registry import AbilityRegistry

pytestmark = pytest.mark.unit


class _Cfg:
    channel = "user"
    policy_channel = None        # _match short-circuits before the gate for an unknown tool
    broadcast_to = None


class _MP:
    config = _Cfg()
    uid = None
    cancel_event = None


def test_dispatch_unknown_tool_is_graceful_not_keyerror():
    """An unknown tool name returns an error string, never an uncaught KeyError."""
    out = ToolDispatcher(_MP()).dispatch("no_such_ability_xyz", {"action": "noop"})
    assert "Unknown tool" in out


def test_real_tool_resolves_from_registry():
    """A genuinely registered ability is found through the real registry."""
    names = {a.NAME for a in AbilityRegistry.all()}
    assert "memory" in names
    assert "find_tools" in names
