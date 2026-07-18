# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ReplaceAllAbility — replace every occurrence of a literal string across the workspace.

One of the 12 inner tools of the ``code_agent`` toolkit
(``abilities/code_agent.py``). code-agent-delegate-exclusive; pinned on
CodeAgentConfig only — never reaches any other channel.
"""

from __future__ import annotations

import os
from fnmatch import fnmatch
from pathlib import Path
from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._result import ToolResult
from abilities._workspace import IGNORED_DIRS, get_workspace_root, looks_line_numbered
from configs.enums.param_key import Keys


class ReplaceAllAbility(Ability):
    DISCOVERABLE: ClassVar[bool] = False  # code-agent-delegate-exclusive; pinned on CodeAgentConfig only

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.search, Keys.replace_)}

    def get_name(self) -> str:
        return "replace_all"

    def get_summary(self) -> str:
        return (
            "Replace every occurrence of a literal string across the code_agent "
            "workspace and report per-file replacement counts. When a glob pattern "
            "is provided, only file names matching it are touched; otherwise every "
            "text file in the tree is a candidate. The search string must be the "
            "raw file text, not read_file's display-only line-number prefixes."
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
        return "replace text across every file in the workspace"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.search: {
                "type": "string",
                "description": "The exact literal text to find in every file.",
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
        },
        "required": [Keys.search, Keys.replace_],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        search = cast(str, self.param(params, Keys.search, required=True))
        replace = cast(str, self.param(params, Keys.replace_, required=True))
        glob_pattern = cast("str | None", self.param(params, Keys.glob))

        root = get_workspace_root()
        real_root = root.resolve()

        changed_files: list[tuple[str, int]] = []

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]

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
            body = "No occurrences were found."
            if looks_line_numbered(search):
                body += (
                    " Your search text includes read_file's line-number prefixes "
                    "— those are display-only. Strip them so the search matches "
                    "the real file content."
                )
            return ToolResult.ok(body, files_changed=0, total_replacements=0)

        total = sum(c for _, c in changed_files)
        body_lines = [f"{path}: {cnt}" for path, cnt in changed_files]

        return ToolResult.ok(
            "\n".join(body_lines),
            files_changed=len(changed_files),
            total_replacements=total,
        )
