# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""EditFileAbility — replace a single occurrence of literal text in one file.

The anchored-edit primitive: discoverable on every channel via ``find_tools``
and additionally pinned on the ``code_agent`` delegate. Replaces whole-file
rewrites for targeted changes — every line outside the replaced span stays
byte-identical.

Single-file, single-occurrence: ``path`` MUST point to an existing file, and
the ``search`` string must match the file content exactly once — zero matches
and ambiguous (2+) matches are both loud errors, never silent no-ops. The
search is literal and whitespace-significant — no regex, no glob, no
directory walk.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.edit_file_params_bag import EditFileParamsBag
from contracts.params.param_bag import ParamBag


class EditFileAbility(Ability[EditFileParamsBag]):
    DISCOVERABLE: ClassVar[bool] = True

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.search, Keys.replace_, Keys.path)}

    # The typed input contract: the dispatch seam builds the bag via
    # EditFileParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = EditFileParamsBag

    def get_name(self) -> str:
        return "edit_file"

    def get_summary(self) -> str:
        return (
            "Replace a single occurrence of literal text in a single file. "
            "Point ``path`` at an absolute file path, ``search`` for the exact "
            "literal text to find (whitespace-significant), and ``replace`` "
            "for the replacement text. The search string must appear exactly "
            "once in the file — if it appears zero times, you get a not-found "
            "error; if it appears more than once, you must include more "
            "surrounding context to make it unique. Every line outside the "
            "replaced span is left byte-identical."
        )

    def get_examples(self) -> list[str]:
        return [
            "replace the API URL in this config file",
            "update the import path in this single file",
            "swap the function name in this source file",
            "change the version string in this package file",
            "fix the typo in this documentation file",
            "update the environment variable name in this file",
            "replace the deprecated method call in this file",
            "change the configuration key in this settings file",
        ]

    def get_search_tooltip(self) -> str:
        return "replace literal text in a single file (must be unique)"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.search: {
                "type": "string",
                "description": "The exact literal text to find. Whitespace-significant; must appear exactly once in the file.",
            },
            Keys.replace_: {
                "type": "string",
                "description": "The replacement text. Whitespace-significant.",
            },
            Keys.path: {
                "type": "string",
                "description": "Absolute path to the single file to edit. Required.",
            },
        },
        "required": [Keys.search, Keys.replace_, Keys.path],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: EditFileParamsBag) -> ToolResult:
        if not Path(params.path).is_absolute():
            return ToolResult.err(
                f"path must be an absolute path, got {params.path!r}.",
                code="invalid-path",
            )

        target = Path(params.path)
        if not target.exists():
            return ToolResult.err(
                f"{params.path} does not exist.",
                code="not-found",
                hint="check the path with the read tool.",
            )

        if target.is_dir():
            return ToolResult.err(
                f"{params.path} is a directory, not a file.",
                code="invalid-path",
                hint="point path at a single file.",
            )

        content = target.read_text(encoding="utf-8")
        count = content.count(params.search)

        if count == 0:
            return ToolResult.err(
                "The search text was not found in the file.",
                code="not-found",
                hint="re-read the file — `search` must match the current content exactly, including whitespace.",
            )

        if count > 1:
            return ToolResult.err(
                "You can only replace 1 occurrence at a time, include more of the surrounding text to make the `search` unique.",
                code="not-unique",
                hint=f"found {count} occurrences; include more context to make it unique.",
            )

        new_content = content.replace(params.search, params.replace_, 1)
        target.write_text(new_content, encoding="utf-8")

        return ToolResult.ok(
            f"Replaced 1 occurrence in {params.path}.",
            files_changed=1,
            total_replacements=1,
        )
