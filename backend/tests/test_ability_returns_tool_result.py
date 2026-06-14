# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Registry-wide ToolResult return-type contract — statically enforced.

Every tool the model can call funnels through ``ToolDispatcher``, which at runtime
hard-fails any ability whose ``run()`` returns a non-:class:`ToolResult`
(``code=non-canonical-result``). That runtime guard only ever fires on the single
code path a given call happens to take. This module enforces the same contract
*statically, across every path*, by type-checking the abilities package: a
``run()`` that could return a non-``ToolResult`` on any branch is a build-time
error here, not a latent runtime surprise that surfaces only when that branch is
hit in production.

Two guards, together forming the static contract:

* :func:`test_run_is_annotated_toolresult` pins the ``-> ToolResult`` annotation on
  every registered ability. This is load-bearing: a type checker only validates a
  function body when the function is annotated, so this annotation is precisely
  what makes the return-type check below apply to ``run()`` at all.
* :func:`test_ability_return_types_are_statically_honoured` runs mypy over the real
  abilities package and asserts no declared return type is violated on any path
  (the ``return-value`` / ``return`` error family). With every ``run()`` annotated
  above, this proves each one yields a ``ToolResult`` on every branch — the
  all-paths static superset of the old single-path runtime conformance sweep.

The per-ability envelope *shape* (status/body/meta/code invariants) is pinned once
in :mod:`tests.test_tool_result_contract`; this module pins the *boundary* that
makes that single contract sufficient for **every registered ability**.
"""

import importlib.util
import subprocess
import sys
import typing

import pytest

from abilities._registry import AbilityRegistry
from abilities._result import ToolResult
from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.unit

_ABILITIES = sorted(AbilityRegistry.all(), key=lambda a: a.get_name())
_NAMES = [a.get_name() for a in _ABILITIES]

# mypy error codes that mean "a declared return type was violated on some path".
# This family is never noise — unlike arg-type/attr-defined, a return-type error
# is always a genuine breach of the contract this module guards.
_RETURN_CONTRACT_CODES = ("[return-value]", "[return]")


@pytest.mark.parametrize("name", _NAMES)
def test_run_is_annotated_toolresult(name):
    """Every registered ability declares ``run() -> ToolResult`` at the source.

    This is the precondition that makes the static return check below bite: mypy
    only validates a function body once the function carries an annotation.
    """
    ability = next(a for a in _ABILITIES if a.get_name() == name)
    hints = typing.get_type_hints(type(ability).run)
    assert hints.get("return") is ToolResult, (
        f"{name}.run() is annotated {hints.get('return')!r}, expected ToolResult"
    )


def test_ability_return_types_are_statically_honoured():
    """mypy proves no ability violates its declared return type on ANY path.

    Combined with the per-ability annotation guard above, this guarantees every
    ``run()`` yields a ``ToolResult`` on every branch — caught at build time,
    before it could ever reach the dispatcher's runtime ``non-canonical-result``
    rescue. It is the all-paths static superset of a runtime conformance sweep
    (which can only exercise one input, hence one path, per ability).
    """
    assert importlib.util.find_spec("mypy") is not None, (
        "mypy must be installed (it is a pyproject dependency) for the static "
        "ToolResult return-type gate to run"
    )

    proc = subprocess.run(
        [sys.executable, "-m", "mypy", str(FileMapperService.get_abilities_path())],
        cwd=str(FileMapperService.get_backend_path()),
        capture_output=True,
        text=True,
    )
    # returncode 0 = clean, 1 = type errors found (expected: the package carries
    # unrelated arg-type/attr-defined errors we do NOT gate on). Anything else is
    # a mypy crash or usage error — fail loudly with its diagnostics.
    assert proc.returncode in (0, 1), (
        f"mypy failed to run (returncode {proc.returncode}):\n{proc.stderr}\n{proc.stdout}"
    )

    offenders = [
        line for line in proc.stdout.splitlines()
        if any(code in line for code in _RETURN_CONTRACT_CODES)
    ]
    assert not offenders, (
        "an ability violates its declared return type on some path — every "
        "Ability.run() must return a ToolResult on every branch:\n"
        + "\n".join(offenders)
    )
