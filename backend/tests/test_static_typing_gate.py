# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Whole-source mypy --strict gate for all first-party backend code.

This test enforces that the entire first-party source tree — every package
listed in ``_PACKAGES`` plus the top-level modules in ``_TOP_LEVEL_MODULES``,
including the test suite itself — passes ``mypy --strict`` with zero errors.

There is no relax/override block in ``pyproject.toml``: the whole backend is
held to strict from the start, so any new untyped or unsound code fails this
gate immediately. See ``docs/typing-ratchet.md`` for how the codebase was
migrated to this state and for the supported narrowing patterns.

The per-ability ToolResult return-type assertion lives in
:mod:`tests.test_ability_returns_tool_result` as a special case of this
general gate — it remains independently so that return-type violations are
reported with targeted ability-level context rather than buried in bulk
mypy output.
"""

import importlib.util
import subprocess
import sys

import pytest

from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.unit

# First-party packages and top-level modules to check.
# This list must mirror the gate verification step in docs/typing-ratchet.md.
_PACKAGES = [
    "abilities",
    "api",
    "capabilities",
    "configs",
    "mcp_server",
    "services",
    "tools",
    "utils",
    "workers",
    "migrations",
    "scripts",
    "tests",
]
_TOP_LEVEL_MODULES = [
    "consumer.py",
    "run.py",
    "runtime_config.py",
    "migrate_transcript_rebuild.py",
]


def test_first_party_source_is_strict_clean() -> None:
    """Assert the whole first-party source tree passes mypy --strict.

    The migration to full strict is complete: there is no relax override in
    pyproject.toml, so this gate requires a clean ``mypy --strict`` run over
    every package and top-level module. Any error — strict or default
    correctness — fails the build.

    See docs/typing-ratchet.md for the supported narrowing patterns when a
    strict error needs to be resolved without weakening a type.
    """
    assert importlib.util.find_spec("mypy") is not None, (
        "mypy must be installed (it is a pyproject dependency) for the "
        "strict typing gate to run"
    )

    backend_path = FileMapperService.get_backend_path()

    proc = subprocess.run(
        [sys.executable, "-m", "mypy", *_PACKAGES, *_TOP_LEVEL_MODULES],
        cwd=str(backend_path),
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, (
        "mypy --strict reported errors over the first-party source tree.\n"
        "Fix every error before committing — the whole backend must stay "
        "strict-clean (there is no relax override).\n"
        "See docs/typing-ratchet.md for the supported narrowing patterns.\n\n"
        f"mypy output:\n{proc.stdout}{proc.stderr}"
    )
