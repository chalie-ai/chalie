# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Import-ban sweep (TKT-882): the wire envelope is formatted in EXACTLY one
place — ``ToolDispatcher._render``.  No module under ``backend/abilities/`` may
import ``services.innate_skills._tag`` (the legacy skill-tag formatter), because
an ability that hand-formats ``[tool(...)]`` envelopes forks the contract.

AST-based so a string match in a comment/docstring does not trip it: we only
flag genuine ``import`` statements that pull in ``services.innate_skills._tag``.
"""

import ast

import pytest

from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.unit

_BANNED_MODULE = "services.innate_skills._tag"


def _imports_banned(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _BANNED_MODULE:
                    return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == _BANNED_MODULE:
                return True
            # `from services.innate_skills import _tag`
            if module == "services.innate_skills" and any(
                a.name == "_tag" for a in node.names
            ):
                return True
    return False


def test_no_ability_module_imports_skill_tag():
    abilities_dir = FileMapperService.get_abilities_path()
    offenders = []
    for path in sorted(abilities_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _imports_banned(tree):
            offenders.append(path.name)

    assert not offenders, (
        f"These ability modules still import {_BANNED_MODULE!r} — the wire "
        f"envelope must be rendered ONLY by ToolDispatcher._render: {offenders}"
    )
