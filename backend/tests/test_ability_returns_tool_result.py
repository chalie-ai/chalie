# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Enforce ToolResult as the static return-type boundary for every registered ability.

This module handles two concerns: (1) asserting each ``Ability.run()`` carries the
``-> ToolResult`` annotation at its source, and (2) running mypy over the real abilities
package to verify no path violates that declared type. The per-ability envelope *shape*
(status/body/meta/code invariants) is pinned once in :mod:`tests.test_tool_result_contract`.
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
def test_run_is_annotated_toolresult(name: str) -> None:
    ability = next(a for a in _ABILITIES if a.get_name() == name)
    hints = typing.get_type_hints(type(ability).run)
    assert hints.get("return") is ToolResult, (
        f"{name}.run() is annotated {hints.get('return')!r}, expected ToolResult"
    )


def test_ability_return_types_are_statically_honoured() -> None:
    """Assert mypy sees no return-type violations for the abilities package.

    The subprocess stderr/stdout is also asserted to confirm mypy itself ran
    successfully (returncode ∈ {0, 1}) rather than crashing or misconfiguring.
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
