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

import ast
import importlib.util
import os
import re
import subprocess
import sys

import pytest

from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.unit

# Directories that never hold first-party source we type-check: build/cache
# artifacts and vendored trees. Pruned from every walk below.
_SKIP_DIRS = frozenset(
    {
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".git",
    }
)

# Top-level backend directories that legitimately contain zero .py files
# (model/data artifacts). The completeness gate asserts this set exactly so a
# new source package can never appear outside the type-checked roster unnoticed.
_PY_FREE_DIRS = frozenset({"pre-trained", "vision"})

# Matches a mypy type-suppression comment in any spacing/error-code variant.
# The raw pattern text below is not itself a match (a backslash, not
# whitespace, follows the hash), so this file does not trip its own gate.
_TYPE_IGNORE_RE = re.compile(r"#\s*type:\s*ignore")

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


def _iter_first_party_py_files() -> "list[str]":
    """Yield every first-party ``.py`` file path under the backend root.

    Walks from ``FileMapperService.get_backend_path()`` and prunes the build /
    cache / vendored directories in ``_SKIP_DIRS``. Returns absolute paths.
    """
    backend_path = str(FileMapperService.get_backend_path())
    out: list[str] = []
    for root, dirs, files in os.walk(backend_path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            if name.endswith(".py"):
                out.append(os.path.join(root, name))
    return out


def _annotation_nodes(tree: ast.AST) -> "list[ast.expr]":
    """Return every annotation expression in ``tree``.

    Covers the three positions ``Any`` can hide in a type: function parameter
    annotations, function return annotations, and annotated assignments
    (``x: T`` / ``x: T = v``). This mirrors ruff's ``ANN401`` scope.
    """
    anns: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            anns.append(node.annotation)
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.returns is not None
        ):
            anns.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            anns.append(node.annotation)
    return anns


def test_no_type_ignore_comments_in_first_party_source() -> None:
    """Forbid every mypy type-suppression comment in first-party source.

    ``mypy --strict`` only flags *unused* suppressions; a *used* one silently
    punches a hole in the type wall. The standing typing bar bans them outright
    — a real type error must be fixed with a narrowing pattern (see
    docs/typing-ratchet.md), never silenced. This gate keeps the count at zero.
    """
    offenders: list[str] = []
    for path in _iter_first_party_py_files():
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if _TYPE_IGNORE_RE.search(line):
                    rel = os.path.relpath(path, str(FileMapperService.get_backend_path()))
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Type-suppression comments are banned in first-party source — resolve "
        "the underlying type error with a sanctioned narrowing pattern instead "
        "(see docs/typing-ratchet.md).\n\nOffending lines:\n"
        + "\n".join(offenders)
    )


def test_no_explicit_any_in_annotations() -> None:
    """Forbid the literal ``Any`` token in any first-party annotation.

    ``mypy --strict`` permits explicit ``Any`` — it is the single largest hole
    left in a strict build. The standing bar requires the most-primitive
    concrete type instead (``object`` / ``dict[str, object]`` / …), so this
    gate fails the build if ``Any`` ever reappears in a parameter, return, or
    variable annotation. (AST-based, so it is precise: ``Any`` in a docstring,
    a comment, or an identifier name is not a match.)
    """
    offenders: list[str] = []
    backend_path = str(FileMapperService.get_backend_path())
    for path in _iter_first_party_py_files():
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source, filename=path)
        for ann in _annotation_nodes(tree):
            for sub in ast.walk(ann):
                if isinstance(sub, ast.Name) and sub.id == "Any":
                    rel = os.path.relpath(path, backend_path)
                    offenders.append(f"{rel}:{sub.lineno}")

    assert not offenders, (
        "Explicit `Any` is banned in first-party annotations — use the "
        "most-primitive concrete type (object / dict[str, object] / …) and "
        "narrow with an inline cast at the read site (see "
        "docs/typing-ratchet.md).\n\nOffending annotations:\n"
        + "\n".join(offenders)
    )


def _dir_contains_py(path: str) -> bool:
    """True if ``path`` (or any non-skipped subdirectory) holds a ``.py`` file."""
    for _root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        if any(name.endswith(".py") for name in files):
            return True
    return False


def _tracked_entries(backend_path: str) -> set[str]:
    """Top-level entry names git tracks under ``backend_path``. The roster gate
    covers first-party *source*, so untracked dirs (locally-staged docs) and
    git-ignored build artifacts (``*.egg-info``) — which a clean checkout never
    has — must not be measured against it."""
    proc = subprocess.run(
        ["git", "-C", backend_path, "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {path.split("/", 1)[0] for path in proc.stdout.split("\0") if path}


def test_typing_gate_covers_every_first_party_package_and_module() -> None:
    """Guarantee the strict gate's roster is exhaustive — no source escapes it.

    The strict gate runs mypy over a hand-maintained ``_PACKAGES`` /
    ``_TOP_LEVEL_MODULES`` list; a brand-new top-level package would otherwise
    be silently unchecked. This test fails the moment the on-disk layout
    diverges from that roster, forcing the list to be updated in lockstep.
    """
    backend_path = str(FileMapperService.get_backend_path())

    py_bearing_dirs: set[str] = set()
    py_free_dirs: set[str] = set()
    top_level_modules: set[str] = set()

    tracked = _tracked_entries(backend_path)

    for entry in tracked:
        full = os.path.join(backend_path, entry)
        if os.path.isdir(full):
            if entry in _SKIP_DIRS:
                continue
            target = py_bearing_dirs if _dir_contains_py(full) else py_free_dirs
            target.add(entry)
        elif os.path.isfile(full) and entry.endswith(".py"):
            top_level_modules.add(entry)

    assert py_bearing_dirs == set(_PACKAGES), (
        "The set of first-party source packages on disk no longer matches the "
        "strict gate's _PACKAGES roster. Update _PACKAGES (and "
        "docs/typing-ratchet.md) so every package is type-checked.\n"
        f"  on disk : {sorted(py_bearing_dirs)}\n"
        f"  _PACKAGES: {sorted(_PACKAGES)}"
    )
    assert py_free_dirs == set(_PY_FREE_DIRS), (
        "A top-level backend directory's .py status changed. If a new source "
        "package appeared, add it to _PACKAGES; if an artifact dir appeared, "
        "add it to _PY_FREE_DIRS.\n"
        f"  py-free on disk: {sorted(py_free_dirs)}\n"
        f"  _PY_FREE_DIRS  : {sorted(_PY_FREE_DIRS)}"
    )
    assert top_level_modules == set(_TOP_LEVEL_MODULES), (
        "The set of top-level .py modules on disk no longer matches the strict "
        "gate's _TOP_LEVEL_MODULES roster. Update it (and "
        "docs/typing-ratchet.md) so every module is type-checked.\n"
        f"  on disk           : {sorted(top_level_modules)}\n"
        f"  _TOP_LEVEL_MODULES: {sorted(_TOP_LEVEL_MODULES)}"
    )
