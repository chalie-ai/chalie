# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

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


def test_dispatch_unknown_tool_is_graceful_not_keyerror() -> None:
    out = ToolDispatcher(_MP()).dispatch("no_such_ability_xyz", {"action": "noop"})
    assert "Unknown tool" in out


def test_real_tool_resolves_from_registry() -> None:
    names = {a.get_name() for a in AbilityRegistry.all()}
    assert "memory" in names
    assert "find_tools" in names
