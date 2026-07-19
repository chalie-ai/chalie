# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ReplaceAllAbility — replace every occurrence of a literal string in a file or across a directory tree.

One of the inner tools of the ``code_agent`` toolkit
(``abilities/code_agent.py``). code-agent-delegate-exclusive; pinned on
CodeAgentConfig only — never reaches any other channel.

Two modes in one tool:

* Point ``path`` at a single file (absolute) and every occurrence in that file
  is replaced in one shot — no try/except, no silent skip: a binary or
  unreadable target raises loudly.
* Point ``path`` at a directory (or omit it, defaulting to the code_agent
  workspace) and ``os.walk`` sweeps the tree — same glob filter, same
  ``IGNORED_DIRS`` prune, same symlink-escape and unreadable-file guards as
  before. Per-file counts are always reported.

The search string must match the raw file text exactly.
"""

from __future__ import annotations

import os
from fnmatch import fnmatch
from pathlib import Path
from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from services.file_mapper_service import FileMapperService


class ReplaceAllAbility(Ability):
    DISCOVERABLE: ClassVar[bool] = False  # code-agent-delegate-exclusive; pinned on CodeAgentConfig only

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.search, Keys.replace_)}

    #: Directory names never worth walking when a tool sweeps a tree:
    #: version-control internals, Python bytecode caches, and vendored or
    #: virtual-env trees.
    IGNORED_DIRS: ClassVar[frozenset[str]] = frozenset(
        {".git", "__pycache__", "node_modules", ".venv", "venv"}
    )

    def get_name(self) -> str:
        return "replace_all"

    def get_summary(self) -> str:
        return (
            "Replace every occurrence of a literal string in a single file or "
            "across a directory tree, reporting per-file replacement counts. "
            "Point ``path`` at an absolute file to edit that file in one shot "
            "(binary or unreadable targets raise loudly, never silently skip); "
            "or at an absolute directory (or omit it to sweep the code_agent "
            "workspace) to touch every matching file in the tree. When a glob "
            "pattern is provided, only file names matching it are touched. "
            "The search string must match the raw file text exactly."
        )

    def get_examples(self) -> list[str]:
        return [
            "replace all occurrences of the old API URL across the project",
            "rename this variable everywhere it's used in the codebase",
            "update the import path in every TypeScript file",
            "replace the old package name with the new one across the tree",
            "change this string across every file that uses it",
            "swap out the deprecated function call everywhere",
            "update the version number in all matching files",
            "replace the old namespace with the new one project-wide",
        ]

    def get_search_tooltip(self) -> str:
        return "replace text in a file or across a directory tree"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.search: {
                "type": "string",
                "description": "The exact literal text to find.",
            },
            Keys.replace_: {
                "type": "string",
                "description": "The replacement text.",
            },
            Keys.glob: {
                "type": "string",
                "description": (
                    "A filename pattern such as '*.ts' to limit which files are touched. "
                    "Matched against each file name; when omitted all text files are candidates."
                ),
            },
            Keys.path: {
                "type": "string",
                "description": (
                    "Absolute path to a single file or to a directory to scan. "
                    "Defaults to the code_agent workspace directory when omitted."
                ),
            },
        },
        "required": [Keys.search, Keys.replace_],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        search = cast(str, self.param(params, Keys.search, required=True))
        replace = cast(str, self.param(params, Keys.replace_, required=True))
        glob_pattern = cast("str | None", self.param(params, Keys.glob))
        path_raw = cast("str | None", self.param(params, Keys.path))

        if path_raw is not None:
            return self._run_explicit(search, replace, glob_pattern, path_raw)
        return self._run_workspace(search, replace, glob_pattern)

    def _run_explicit(
        self,
        search: str,
        replace: str,
        glob_pattern: str | None,
        raw_path: str,
    ) -> ToolResult:
        # Absolute-path guard — must come before resolve (which would succeed on
        # a relative path).
        if not Path(raw_path).is_absolute():
            return ToolResult.err(
                f"path must be an absolute path, got {raw_path!r}.",
                code="invalid-path",
            )

        target = Path(raw_path)
        if not target.exists():
            return ToolResult.err(
                f"{raw_path} does not exist.",
                code="not-found",
                hint="check the path with the read tool.",
            )

        if target.is_file():
            return self._replace_in_file(search, replace, glob_pattern, target, raw_path)
        # Directory (or symlink to a directory): walk it.
        return self._walk(target, search, replace, glob_pattern)

    def _replace_in_file(
        self,
        search: str,
        replace: str,
        glob_pattern: str | None,
        target: Path,
        raw_path: str,
    ) -> ToolResult:
        # Glob filter applies to the file name; a non-matching file is zero
        # candidates, not an error.
        if glob_pattern is not None and not fnmatch(target.name, glob_pattern):
            return ToolResult.ok(
                "No occurrences were found.", files_changed=0, total_replacements=0
            )

        # Read WITHOUT a try/except — a binary or unreadable direct target must
        # raise loudly, never silently skip.
        content = target.read_text(encoding="utf-8")

        count = content.count(search)
        if count == 0:
            return ToolResult.ok(
                "No occurrences were found.", files_changed=0, total_replacements=0
            )

        new_content = content.replace(search, replace, count)
        target.write_text(new_content, encoding="utf-8")

        return ToolResult.ok(
            f"{raw_path}: {count}",
            files_changed=1,
            total_replacements=count,
        )

    def _run_workspace(
        self,
        search: str,
        replace: str,
        glob_pattern: str | None,
    ) -> ToolResult:
        root = FileMapperService.get_code_agent_workspace_path()
        return self._walk(root, search, replace, glob_pattern)

    def _walk(
        self,
        root: Path,
        search: str,
        replace: str,
        glob_pattern: str | None,
    ) -> ToolResult:
        real_root = root.resolve()

        changed_files: list[tuple[str, int]] = []

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in self.IGNORED_DIRS]

            for filename in filenames:
                if glob_pattern is not None and not fnmatch(filename, glob_pattern):
                    continue

                real_file = Path(os.path.join(dirpath, filename)).resolve()
                if real_file != real_root and not real_file.is_relative_to(real_root):
                    continue

                rel_path = os.path.relpath(os.path.join(dirpath, filename), root)

                try:
                    content = real_file.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue

                count = content.count(search)
                if count == 0:
                    continue

                new_content = content.replace(search, replace, count)
                real_file.write_text(new_content, encoding="utf-8")
                changed_files.append((rel_path, count))

        if not changed_files:
            return ToolResult.ok(
                "No occurrences were found.", files_changed=0, total_replacements=0
            )

        total = sum(c for _, c in changed_files)
        body_lines = [f"{path}: {cnt}" for path, cnt in changed_files]

        return ToolResult.ok(
            "\n".join(body_lines),
            files_changed=len(changed_files),
            total_replacements=total,
        )
