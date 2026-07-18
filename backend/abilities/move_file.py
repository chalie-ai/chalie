# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MoveFileAbility — move or rename a file/directory inside the code_agent workspace.

One of the 12 inner tools of the ``code_agent`` toolkit
(``abilities/code_agent.py``). code-agent-delegate-exclusive; pinned on
CodeAgentConfig only — never reaches any other channel.

Plain filesystem move only — no language-server import-fixing hook (Chalie
does not run an LSP manager for the code_agent workspace, so that concern
from the source toolkit this was ported from does not apply here).
"""

from __future__ import annotations

import shutil
from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._result import ToolResult
from abilities._workspace import resolve_in_root
from configs.enums.param_key import Keys


class MoveFileAbility(Ability):
    DISCOVERABLE: ClassVar[bool] = False  # code-agent-delegate-exclusive; pinned on CodeAgentConfig only

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.path, Keys.destination)}

    def get_name(self) -> str:
        return "move_file"

    def get_summary(self) -> str:
        return (
            "Move or rename a file or directory inside the code_agent workspace "
            "(plain filesystem move). The source must exist and the destination "
            "must not already exist."
        )

    def get_examples(self) -> list[str]:
        return [
            "rename main.ts to index.ts",
            "move utils.ts into the src folder",
            "rename the draft file to final.ts",
            "move the test file into the tests directory",
            "rename config.json to settings.json",
            "move output.txt to the results folder",
            "rename this folder to archive",
            "move the script into a subdirectory",
        ]

    def get_search_tooltip(self) -> str:
        return "move or rename a file in the code workspace"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.path: {
                "type": "string",
                "description": "Current location of the file or directory relative to the workspace root.",
            },
            Keys.destination: {
                "type": "string",
                "description": "New location for the file or directory relative to the workspace root.",
            },
        },
        "required": [Keys.path, Keys.destination],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        raw_source = cast(str, self.param(params, Keys.path, required=True))
        raw_destination = cast(str, self.param(params, Keys.destination, required=True))

        try:
            resolved_source = resolve_in_root(raw_source)
        except ValueError as exc:
            return ToolResult.err(str(exc), code="path-escapes-root")

        try:
            resolved_destination = resolve_in_root(raw_destination)
        except ValueError as exc:
            return ToolResult.err(str(exc), code="path-escapes-root")

        if not resolved_source.exists():
            return ToolResult.err(f"{raw_source} does not exist.", code="not-found")

        if resolved_destination.exists():
            return ToolResult.err(
                f"{raw_destination} already exists.",
                code="already-exists",
                hint="Delete it first or choose another name.",
            )

        resolved_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(resolved_source), str(resolved_destination))

        return ToolResult.ok(f"Moved {raw_source} -> {raw_destination}", path=raw_destination)
