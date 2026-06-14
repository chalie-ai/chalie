# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Registry-wide ToolResult conformance sweep.

Every tool the model can ever call funnels through ``ToolDispatcher``, which
hard-fails any ability whose ``run()`` returns a non-:class:`ToolResult`
(``code=non-canonical-result``). The per-ability envelope details are pinned once
in :mod:`tests.test_tool_result_contract`; this module pins the *boundary* that
makes that single contract sufficient — that the boundary holds for **every
registered ability**, not just the handful the contract test exercises with
throwaway abilities.

Two complementary guards, both auto-collected from :class:`AbilityRegistry` so a
newly added ability is swept the moment it registers (no per-ability edit):

* :func:`test_run_returns_toolresult` drives each ability's real ``run()`` through
  the dispatcher's ``_run`` seam with empty args on a no-human channel (the
  cheapest, fully offline edge/error path) and asserts the return is a
  ``ToolResult`` that did NOT trip the non-canonical hard-fail. This catches an
  ability that returns a legacy string/dict on its error path — a leak a
  happy-path-only check would miss.
* :func:`test_run_is_annotated_toolresult` asserts the *declared* return type of
  every ``run`` resolves to ``ToolResult``, so the contract is enforced statically
  at the source as well as at runtime.
"""

import threading
import typing

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities._registry import AbilityRegistry
from abilities._result import ToolResult
from configs.channels import DmnConfig
from tests._tool_result_harness import seed_transcript

pytestmark = pytest.mark.unit


class _MP:
    """Minimal MessageProcessor stand-in carrying only what ``_bind``/``_run`` read.

    A no-human (subconscious) channel keeps every ability's edge path fully
    offline: the policy gate never parks for human consent, so ``run()`` is reached
    with empty args and resolves to its own validation/error branch without any
    network call.
    """

    def __init__(self, uid, config):
        self.config = config
        self.uid = uid
        self._uid = uid
        self.DISCOVERABLE = []
        self.active_tools = []
        self.cancel_event = threading.Event()


_ABILITIES = sorted(AbilityRegistry.all(), key=lambda a: a.get_name())
_NAMES = [a.get_name() for a in _ABILITIES]


@pytest.mark.parametrize("name", _NAMES)
def test_run_returns_toolresult(db, name):
    """Every registered ability's run() returns a ToolResult on the edge path."""
    mp = _MP(seed_transcript(db, "dmn"), DmnConfig())
    ability = ToolDispatcher(mp)._bind(name)
    tr = ToolDispatcher._run(ability, {})
    assert isinstance(tr, ToolResult), f"{name}.run() returned {type(tr).__name__}"
    assert tr.code != "non-canonical-result", f"{name}: {tr.body}"


@pytest.mark.parametrize("name", _NAMES)
def test_run_is_annotated_toolresult(name):
    """Every registered ability declares run() -> ToolResult at the source."""
    ability = next(a for a in _ABILITIES if a.get_name() == name)
    hints = typing.get_type_hints(type(ability).run)
    assert hints.get("return") is ToolResult, (
        f"{name}.run() is annotated {hints.get('return')!r}, expected ToolResult"
    )
