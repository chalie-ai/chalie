# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ReplaceOneAbility — replace one unique occurrence of a literal string in a file.

One of the inner tools of the ``code_agent`` toolkit
(``abilities/code_agent.py``). code-agent-delegate-exclusive; pinned on
CodeAgentConfig only — never reaches any other channel.
"""

from __future__ import annotations

from typing import ClassVar, cast

from abilities._result import ToolResult
from abilities._replace import ReplaceAbilityBase
from configs.enums.param_key import Keys


class ReplaceOneAbility(ReplaceAbilityBase):
    DISCOVERABLE: ClassVar[bool] = False  # code-agent-delegate-exclusive; pinned on CodeAgentConfig only

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.path, Keys.search, Keys.replace_)}

    def get_name(self) -> str:
        return "replace_one"

    def get_summary(self) -> str:
        return (
            "Replace exactly one occurrence of a literal string in a single file. "
            "Files conventionally live in the code_agent workspace but any absolute "
            "path is accepted. Refuses when the match is not unique in the file — "
            "include more surrounding context in the search string. The search "
            "string must match the raw file text exactly."
        )

    def get_examples(self) -> list[str]:
        return [
            "replace the old function name with the new one in main.ts",
            "change this variable name to something clearer",
            "fix the typo in this line of code",
            "swap out this hardcoded value for the constant",
            "update the import statement in this file",
            "replace this comment with a more accurate one",
            "change the return type of this function",
            "fix the off-by-one error in this loop condition",
        ]

    def get_search_tooltip(self) -> str:
        return "replace one occurrence of text in a file"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.path: {
                "type": "string",
                "description": "Absolute path to the file to edit.",
            },
            Keys.search: {
                "type": "string",
                "description": "The exact literal text to find in the file.",
            },
            Keys.replace_: {
                "type": "string",
                "description": "The replacement text.",
            },
        },
        "required": [Keys.path, Keys.search, Keys.replace_],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        raw_path = cast(str, self.param(params, Keys.path, required=True))
        search = cast(str, self.param(params, Keys.search, required=True))
        replace = cast(str, self.param(params, Keys.replace_, required=True))

        resolved, error = self._validate_file_path(raw_path)
        if error is not None:
            return error
        assert resolved is not None

        content = resolved.read_text(encoding="utf-8")
        count = content.count(search)

        if count == 0:
            return ToolResult.err(
                f"The search string was not found in {raw_path}.",
                code="no-match",
                hint="Check the exact text with the read tool.",
            )

        if count > 1:
            return ToolResult.err(
                f"Found {count} occurrences of the search string in {raw_path}; "
                f"replacement was refused because the match must be unique.",
                code="ambiguous-match",
                hint="Include more surrounding context in the search string.",
            )

        new_content = content.replace(search, replace, 1)
        resolved.write_text(new_content, encoding="utf-8")

        return ToolResult.ok(f"Replaced 1 occurrence in {raw_path}.", occurrences=1)
