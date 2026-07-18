# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ListFilesAbility — gitignore-aware indented tree listing of the code_agent workspace.

One of the 12 inner tools of the ``code_agent`` toolkit
(``abilities/code_agent.py``). code-agent-delegate-exclusive; pinned on
CodeAgentConfig only — never reaches any other channel.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._result import ToolResult
from abilities._workspace import get_workspace_root, load_gitignore_patterns, resolve_in_root, should_skip
from configs.enums.param_key import Keys

_INDENT = "  "


class ListFilesAbility(Ability):
    DISCOVERABLE: ClassVar[bool] = False  # code-agent-delegate-exclusive; pinned on CodeAgentConfig only

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": ()}

    def get_name(self) -> str:
        return "list_files"

    def get_summary(self) -> str:
        return (
            "List the files and directories in the code_agent workspace as an "
            "indented tree, starting from the workspace root or a given "
            "subdirectory. Entries ignored by the workspace's .gitignore (plus "
            "always-ignored directories like .git and node_modules) are skipped."
        )

    def get_examples(self) -> list[str]:
        return [
            "show me the files in the workspace",
            "list everything in the src folder",
            "what files are in this project",
            "give me a tree of the workspace",
            "list the contents of the tests directory",
            "show the project structure",
            "what's in the current workspace",
            "list files under the utils folder",
        ]

    def get_search_tooltip(self) -> str:
        return "list files in the code workspace"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.path: {
                "type": "string",
                "description": (
                    "Subdirectory relative to the workspace root to list. "
                    "Omit to list the whole workspace from the root."
                ),
            },
        },
        "required": [],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        raw_path = cast("str | None", self.param(params, Keys.path))

        if raw_path:
            try:
                start = resolve_in_root(raw_path)
            except ValueError as exc:
                return ToolResult.err(str(exc), code="path-escapes-root")
            if not start.exists():
                return ToolResult.err(f"{raw_path} does not exist.", code="not-found")
            if not start.is_dir():
                return ToolResult.err(f"{raw_path} is not a directory.", code="not-a-directory")
        else:
            start = get_workspace_root()

        patterns = load_gitignore_patterns(get_workspace_root())

        lines: list[str] = [f"{start.name or '.'}/"]
        entry_count = self._walk(start, patterns, 1, lines)

        return ToolResult.ok("\n".join(lines), entry_count=entry_count)

    def _walk(self, directory: Path, patterns: list[str], depth: int, lines: list[str]) -> int:
        try:
            children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return 0

        count = 0
        for child in children:
            is_dir = child.is_dir()
            if should_skip(child.name, is_dir, patterns):
                continue

            count += 1
            suffix = "/" if is_dir else ""
            lines.append(f"{_INDENT * depth}{child.name}{suffix}")

            if is_dir:
                count += self._walk(child, patterns, depth + 1, lines)

        return count
