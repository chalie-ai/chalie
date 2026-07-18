# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""UpdateFileAbility — overwrite an existing file inside the code_agent workspace.

One of the 12 inner tools of the ``code_agent`` toolkit
(``abilities/code_agent.py``). code-agent-delegate-exclusive; pinned on
CodeAgentConfig only — never reaches any other channel.
"""

from __future__ import annotations

from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._result import ToolResult
from abilities._workspace import resolve_existing_file
from configs.enums.param_key import Keys

# Guard against a destructive partial-edit: a model sometimes calls
# update_file (a full overwrite) with only a code SNIPPET when it means to
# make a small edit, silently destroying a large working file. Refuse when
# the new content is a tiny fraction of a substantial existing file and steer
# to the targeted-edit tools instead. Deliberately conservative (<5% of a
# >=4 KB file) so legitimate rewrites are unaffected; an intentional full
# shortening can still use delete_file + create_file.
_DESTRUCTIVE_MIN_EXISTING_BYTES = 4000
_DESTRUCTIVE_SHRINK_FACTOR = 20


class UpdateFileAbility(Ability):
    DISCOVERABLE: ClassVar[bool] = False  # code-agent-delegate-exclusive; pinned on CodeAgentConfig only

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.path, Keys.content)}

    def get_name(self) -> str:
        return "update_file"

    def get_summary(self) -> str:
        return (
            "Overwrite an existing file with entirely new content (full overwrite, "
            "not a patch) — use for a full rewrite of a small file. For a targeted "
            "change to part of a file prefer replace_one or replace_all; to make a "
            "brand-new file use create_file."
        )

    def get_examples(self) -> list[str]:
        return [
            "rewrite main.ts with this new version",
            "replace the entire content of config.json",
            "overwrite the script with the corrected code",
            "update the file with the fixed implementation",
            "rewrite server.ts from scratch",
            "replace the contents of utils.ts entirely",
            "overwrite the test file with the new test suite",
            "update index.ts with the latest version",
        ]

    def get_search_tooltip(self) -> str:
        return "overwrite a file in the code workspace"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.path: {
                "type": "string",
                "description": "Path relative to the workspace root of the file to overwrite.",
            },
            Keys.content: {
                "type": "string",
                "description": "New content to write to the file, replacing all existing content.",
            },
        },
        "required": [Keys.path, Keys.content],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        raw_path = cast(str, self.param(params, Keys.path, required=True))
        content = cast(str, self.param(params, Keys.content, required=True))

        resolved, error = resolve_existing_file(raw_path)
        if error is not None:
            return error
        assert resolved is not None

        try:
            existing_size = resolved.stat().st_size
        except OSError:
            existing_size = 0
        new_size = len(content.encode("utf-8"))
        if (
            existing_size >= _DESTRUCTIVE_MIN_EXISTING_BYTES
            and new_size * _DESTRUCTIVE_SHRINK_FACTOR < existing_size
        ):
            return ToolResult.err(
                f"{raw_path}: refusing to overwrite a {existing_size}-byte file "
                f"with {new_size} bytes — this looks like a partial edit (a "
                f"snippet), not a full rewrite, and would destroy the existing "
                f"content. Use replace_one / replace_all for a targeted edit, "
                f"or delete_file + create_file if you genuinely mean to replace "
                f"the whole file.",
                code="destructive-partial-overwrite",
                hint="Use replace_one / replace_all for small edits.",
            )

        resolved.write_text(content, encoding="utf-8")

        return ToolResult.ok(
            f"File updated: {raw_path}",
            path=raw_path,
            bytes_written=new_size,
        )
