# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CreateFolderAbility — create a directory inside the code_agent workspace.

One of the 12 inner tools of the ``code_agent`` toolkit
(``abilities/code_agent.py``). code-agent-delegate-exclusive; pinned on
CodeAgentConfig only — never reaches any other channel.
"""

from __future__ import annotations

from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._result import ToolResult
from abilities._workspace import resolve_in_root
from configs.enums.param_key import Keys


class CreateFolderAbility(Ability):
    DISCOVERABLE: ClassVar[bool] = False  # code-agent-delegate-exclusive; pinned on CodeAgentConfig only

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.path,)}

    def get_name(self) -> str:
        return "create_folder"

    def get_summary(self) -> str:
        return (
            "Create a directory (and any missing parent directories) inside the "
            "code_agent workspace. Idempotent — reports success without changes "
            "when the directory already exists."
        )

    def get_examples(self) -> list[str]:
        return [
            "create a folder called src",
            "make a new directory for the tests",
            "create the folder structure src/utils",
            "make a directory called output",
            "create a components folder",
            "make a new subdirectory for the assets",
            "create a folder to hold the generated files",
            "set up a directory called scripts",
        ]

    def get_search_tooltip(self) -> str:
        return "create a directory in the code workspace"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.path: {
                "type": "string",
                "description": "Path relative to the workspace root where the directory should be created.",
            },
        },
        "required": [Keys.path],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        raw_path = cast(str, self.param(params, Keys.path, required=True))

        try:
            resolved = resolve_in_root(raw_path)
        except ValueError as exc:
            return ToolResult.err(str(exc), code="path-escapes-root")

        if resolved.is_dir():
            return ToolResult.ok(
                f"Directory already exists: {raw_path}",
                path=raw_path,
                created=False,
            )

        if resolved.exists():
            return ToolResult.err(
                f"{raw_path} exists but is not a directory.",
                code="not-a-directory",
            )

        resolved.mkdir(parents=True)

        return ToolResult.ok(
            f"Directory created: {raw_path}",
            path=raw_path,
            created=True,
        )
