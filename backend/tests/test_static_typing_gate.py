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

This test enforces that the entire first-party source tree passes
``mypy --strict`` without errors. Unmigrated packages are intentionally
relaxed via the ``[[tool.mypy.overrides]]`` relax block in
``pyproject.toml`` — those packages appear in the ``module`` list there
with every strict sub-flag turned off plus ``disable_error_code`` covering
the default correctness codes that still fire.

The gate tightens automatically as each package is migrated: remove a
package's glob from the ``module`` list in the relax override, run this
test, fix every surfaced strict and correctness error, then commit. When
all globs are removed the relax block (and its ``disable_error_code``
entry) can be deleted entirely in the final-flip ticket.

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
# This list must mirror the packages passed to mypy in pyproject.toml and
# the gate verification step in docs/typing-ratchet.md.
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

    The relax override in pyproject.toml keeps unmigrated packages green
    automatically. As each package is migrated and its glob is removed from
    that override, this test tightens without any change to this file.
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
        "mypy --strict found errors in the first-party source tree.\n"
        "To migrate a package, remove its glob from the [[tool.mypy.overrides]] "
        "module list in pyproject.toml, then fix every surfaced error.\n"
        "See docs/typing-ratchet.md for the full migration workflow.\n\n"
        f"mypy output:\n{proc.stdout}"
    )
