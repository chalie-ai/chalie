# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DeleteFileAbility — delete a file or empty directory inside the code_agent workspace.

One of the 12 inner tools of the ``code_agent`` toolkit
(``abilities/code_agent.py``). code-agent-delegate-exclusive; pinned on
CodeAgentConfig only — never reaches any other channel.
"""

from __future__ import annotations

from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._result import ToolResult
from abilities._workspace import get_workspace_root, resolve_in_root
from configs.enums.param_key import Keys


class DeleteFileAbility(Ability):
    DISCOVERABLE: ClassVar[bool] = False  # code-agent-delegate-exclusive; pinned on CodeAgentConfig only

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.path,)}

    def get_name(self) -> str:
        return "delete_file"

    def get_summary(self) -> str:
        return (
            "Delete a file or an empty directory inside the code_agent workspace. "
            "The workspace root itself is never deleted. Non-empty directories are "
            "rejected to prevent accidental data loss."
        )

    def get_examples(self) -> list[str]:
        return [
            "delete the temp.ts file",
            "remove the old test file",
            "delete that empty output folder",
            "get rid of scratch.js",
            "delete the file I no longer need",
            "remove the unused config file",
            "delete the draft version of the script",
            "clean up by deleting the old build artifact",
        ]

    def get_search_tooltip(self) -> str:
        return "delete a file or empty directory in the code workspace"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.path: {
                "type": "string",
                "description": "Path relative to the workspace root of the file or empty directory to delete.",
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

        if resolved == get_workspace_root().resolve():
            return ToolResult.err(
                "Cannot delete the workspace root.",
                code="cannot-delete-root",
            )

        if not resolved.exists():
            return ToolResult.err(
                f"{raw_path} does not exist.",
                code="not-found",
            )

        if resolved.is_dir() and any(resolved.iterdir()):
            return ToolResult.err(
                f"{raw_path} is a non-empty directory.",
                code="directory-not-empty",
                hint="Only empty directories can be removed.",
            )

        if not resolved.is_file() and not resolved.is_dir():
            return ToolResult.err(
                f"{raw_path} is neither a regular file nor a directory and cannot be deleted.",
                code="unsupported-file-type",
            )

        if resolved.is_file():
            resolved.unlink()
        else:
            resolved.rmdir()

        return ToolResult.ok(f"Deleted: {raw_path}", path=raw_path)
