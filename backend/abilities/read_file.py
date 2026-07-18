# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ReadFileAbility — read a file inside the code_agent workspace with optional line paging.

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

# Enforce the size cap when no paging parameters are supplied, so a huge file
# does not blow the context window on a single unpaged read.
_MAX_CHARACTERS = 50000


def _number_lines(lines: list[str], first_lineno: int) -> str:
    """Render *lines* cat -n style: ``     N\\t<line>`` starting at *first_lineno*.

    The line number is right-aligned in a width-6 column so numbers line up,
    and reflects the true file line number (not a page-relative index) so a
    paged read still shows the real line numbers.
    """
    return "\n".join(
        f"{first_lineno + offset:6d}\t{line}" for offset, line in enumerate(lines)
    )


class ReadFileAbility(Ability):
    DISCOVERABLE: ClassVar[bool] = False  # code-agent-delegate-exclusive; pinned on CodeAgentConfig only

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.path,)}

    def get_name(self) -> str:
        return "read_file"

    def get_summary(self) -> str:
        return (
            "Read a file's contents inside the code_agent workspace, with optional "
            "start_line/end_line paging for large files. Output is numbered cat -n "
            "style — the numbers are display-only, never write them into a file or "
            "a replace_one/replace_all search string."
        )

    def get_examples(self) -> list[str]:
        return [
            "read the contents of main.ts",
            "show me lines 40 to 80 of server.ts",
            "open the config file and show me what's in it",
            "read the file I just created",
            "what does the first 100 lines of index.ts look like",
            "show me the rest of utils.ts starting from line 200",
            "read package.json",
            "let me see the current content of the script",
        ]

    def get_search_tooltip(self) -> str:
        return "read a file in the code workspace"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.path: {
                "type": "string",
                "description": "Path relative to the workspace root.",
            },
            Keys.start_line: {
                "type": "integer",
                "description": "1-based first line to include (inclusive).",
            },
            Keys.end_line: {
                "type": "integer",
                "description": "1-based last line to include (inclusive).",
            },
        },
        "required": [Keys.path],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        raw_path = cast(str, self.param(params, Keys.path, required=True))
        start_line = cast("int | None", params.get(Keys.start_line))
        end_line = cast("int | None", params.get(Keys.end_line))

        try:
            resolved = resolve_in_root(raw_path)
        except ValueError as exc:
            return ToolResult.err(str(exc), code="path-escapes-root")

        if not resolved.exists() or not resolved.is_file():
            return ToolResult.err(
                f"{raw_path} does not exist or is not a regular file.",
                code="not-a-file",
            )

        content = resolved.read_text(encoding="utf-8", errors="replace")

        if start_line is None and end_line is None:
            return self._read_whole(content, raw_path)
        return self._read_range(content, raw_path, start_line, end_line)

    def _read_whole(self, content: str, raw_path: str) -> ToolResult:
        if len(content) > _MAX_CHARACTERS:
            lines_count = len(content.splitlines())
            return ToolResult.err(
                f"{raw_path} is too large ({len(content)} characters, {lines_count} lines).",
                code="file-too-large",
                hint="Use start_line/end_line to read a smaller range.",
            )
        file_lines = content.splitlines()
        return ToolResult.ok(_number_lines(file_lines, 1), total_lines=len(file_lines))

    def _read_range(
        self, content: str, raw_path: str, start_line: "int | None", end_line: "int | None"
    ) -> ToolResult:
        lines = content.splitlines()
        total_lines = len(lines)

        if start_line is not None and end_line is not None:
            if (
                not isinstance(start_line, int) or isinstance(start_line, bool)
                or not isinstance(end_line, int) or isinstance(end_line, bool)
                or start_line < 1 or end_line < 1 or start_line > end_line
            ):
                return ToolResult.err(
                    f"Invalid line range: start_line={start_line}, end_line={end_line}. "
                    "Both must be positive integers with start_line <= end_line.",
                    code="bad-range",
                )
        elif start_line is not None:
            if not isinstance(start_line, int) or isinstance(start_line, bool) or start_line < 1:
                return ToolResult.err(
                    f"Invalid line range: start_line={start_line}. "
                    "start_line must be a positive integer.",
                    code="bad-range",
                )
            end_line = total_lines
        else:
            if not isinstance(end_line, int) or isinstance(end_line, bool) or end_line < 1:
                return ToolResult.err(
                    f"Invalid line range: end_line={end_line}. "
                    "end_line must be a positive integer.",
                    code="bad-range",
                )
            if end_line > total_lines:
                return ToolResult.err(
                    f"Invalid line range: end_line={end_line}, file has {total_lines} lines. "
                    "end_line must not exceed the file length.",
                    code="bad-range",
                )
            start_line = 1

        assert start_line is not None and end_line is not None
        sliced = lines[start_line - 1: end_line]
        returned = "\n".join(sliced)
        if len(returned) > _MAX_CHARACTERS:
            return ToolResult.err(
                f"{raw_path} is too large ({len(returned)} characters for the requested range).",
                code="file-too-large",
                hint="Request a narrower start_line/end_line range.",
            )
        return ToolResult.ok(
            _number_lines(sliced, start_line), total_lines=total_lines, returned_lines=len(sliced)
        )
