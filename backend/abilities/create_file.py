# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CreateFileAbility — create a new file inside the code_agent workspace.

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


class CreateFileAbility(Ability):
    DISCOVERABLE: ClassVar[bool] = False  # code-agent-delegate-exclusive; pinned on CodeAgentConfig only

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.path,)}

    def get_name(self) -> str:
        return "create_file"

    def get_summary(self) -> str:
        return (
            "Create a new file with the given content inside the code_agent workspace. "
            "Rejects the call if the path already exists — use update_file to overwrite."
        )

    def get_examples(self) -> list[str]:
        return [
            "create a file called main.ts with this content",
            "write a new script that fetches data from an API",
            "create an empty file called notes.txt",
            "make a new TypeScript file with a fibonacci function",
            "create config.json with these settings",
            "write a new file containing this class definition",
            "create a README for this workspace",
            "make a new file with the code you just wrote",
        ]

    def get_search_tooltip(self) -> str:
        return "create a file in the code workspace"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.path: {
                "type": "string",
                "description": "Path relative to the workspace root where the file should be created.",
            },
            Keys.content: {
                "type": "string",
                "description": "Content to write to the file. Defaults to an empty string.",
            },
        },
        "required": [Keys.path],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        raw_path = cast(str, self.param(params, Keys.path, required=True))
        content = cast(str, self.param(params, Keys.content, default=""))

        try:
            resolved = resolve_in_root(raw_path)
        except ValueError as exc:
            return ToolResult.err(str(exc), code="path-escapes-root")

        if resolved.exists():
            return ToolResult.err(
                f"{raw_path} already exists.",
                code="already-exists",
                hint="Use update_file to overwrite an existing file.",
            )

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

        return ToolResult.ok(
            f"File created: {raw_path}",
            path=raw_path,
            bytes_written=len(content.encode("utf-8")),
        )
