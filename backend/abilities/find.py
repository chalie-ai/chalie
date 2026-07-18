# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""FindAbility — search file contents inside the code_agent workspace.

One of the 12 inner tools of the ``code_agent`` toolkit
(``abilities/code_agent.py``). code-agent-delegate-exclusive; pinned on
CodeAgentConfig only — never reaches any other channel.

Prefers ripgrep when it is on PATH; Chalie's installer does not bundle it, so
the pure-Python walk below is the tool's NORMAL path in the shipped product,
not a defensive fallback.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._result import ToolResult
from abilities._workspace import get_workspace_root, load_gitignore_patterns, locate_rg, resolve_in_root, should_skip
from configs.enums.param_key import Keys

_MAX_RESULTS = 200
_MAX_LINE_LENGTH = 300
_SEARCH_TIMEOUT_SECONDS = 30


class FindAbility(Ability):
    DISCOVERABLE: ClassVar[bool] = False  # code-agent-delegate-exclusive; pinned on CodeAgentConfig only

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.query,)}

    def get_name(self) -> str:
        return "find"

    def get_summary(self) -> str:
        return (
            "Search file contents in the code_agent workspace for a query and "
            "return matching lines as 'path:line:content'. Exact by default; set "
            "fuzzy to match the query's words in order with anything in between, "
            "case-insensitively. Optionally scope the search to a subdirectory."
        )

    def get_examples(self) -> list[str]:
        return [
            "find every place that calls this function",
            "search the workspace for TODO comments",
            "where is this variable defined",
            "find all references to the old API endpoint",
            "search for the word 'deprecated' in the codebase",
            "find lines mentioning error handling in the utils folder",
            "search for something like 'connect to database' in the code",
            "find where this constant is used",
        ]

    def get_search_tooltip(self) -> str:
        return "search file contents in the code workspace"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.query: {
                "type": "string",
                "description": "Text to search for in file contents.",
            },
            Keys.path: {
                "type": "string",
                "description": "Subdirectory relative to the workspace root to scope the search to.",
            },
            Keys.fuzzy: {
                "type": "boolean",
                "description": (
                    "When true, match the query's words in order with anything "
                    "in between, case-insensitively, instead of an exact substring."
                ),
            },
        },
        "required": [Keys.query],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        query = cast(str, self.param(params, Keys.query, required=True))
        raw_path = cast("str | None", self.param(params, Keys.path))
        fuzzy = bool(self.param(params, Keys.fuzzy) or False)

        if raw_path:
            try:
                scope = resolve_in_root(raw_path)
            except ValueError as exc:
                return ToolResult.err(str(exc), code="path-escapes-root")
            if not scope.exists():
                return ToolResult.err(f"{raw_path} does not exist.", code="not-found")
        else:
            scope = get_workspace_root()

        rg_path = locate_rg()
        if rg_path is not None:
            return self._search_with_rg(rg_path, query, scope, fuzzy)
        return self._search_pure_python(query, scope, fuzzy)

    def _search_with_rg(self, rg_path: str, query: str, scope: Path, fuzzy: bool) -> ToolResult:
        args = [rg_path, "--line-number", "--no-heading", "--max-count", str(_MAX_RESULTS)]
        if fuzzy:
            args += ["--ignore-case", self._fuzzy_pattern(query)]
        else:
            args += ["--fixed-strings", query]
        args.append(str(scope))

        try:
            proc = subprocess.run(  # noqa: S603
                args,
                capture_output=True,
                text=True,
                timeout=_SEARCH_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.err("The search timed out.", code="timeout")

        if proc.returncode not in (0, 1):
            return ToolResult.err(
                f"ripgrep failed: {proc.stderr.strip()}",
                code="search-failed",
            )

        lines = [line for line in proc.stdout.splitlines() if line][:_MAX_RESULTS]
        formatted = [self._format_rg_line(line) for line in lines]
        return self._respond(formatted)

    def _format_rg_line(self, rg_line: str) -> str:
        try:
            abs_path, lineno, content = rg_line.split(":", 2)
        except ValueError:
            return rg_line
        try:
            rel_path = Path(abs_path).resolve().relative_to(get_workspace_root().resolve())
        except ValueError:
            rel_path = Path(abs_path)
        return f"{rel_path}:{lineno}:{content[:_MAX_LINE_LENGTH]}"

    def _fuzzy_pattern(self, query: str) -> str:
        words = query.split()
        return ".*".join(re.escape(word) for word in words) if words else re.escape(query)

    def _search_pure_python(self, query: str, scope: Path, fuzzy: bool) -> ToolResult:
        root = get_workspace_root()
        patterns = load_gitignore_patterns(root)

        matcher = self._build_matcher(query, fuzzy)
        results: list[str] = []

        for path in self._iter_files(scope, patterns):
            if len(results) >= _MAX_RESULTS:
                break
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            rel_path = path.relative_to(root)
            for lineno, line in enumerate(text.split("\n"), start=1):
                if len(results) >= _MAX_RESULTS:
                    break
                if matcher(line):
                    results.append(f"{rel_path}:{lineno}:{line.strip()[:_MAX_LINE_LENGTH]}")

        return self._respond(results)

    def _build_matcher(self, query: str, fuzzy: bool) -> Callable[[str], bool]:
        if not fuzzy:
            return lambda line: query in line

        pattern = re.compile(self._fuzzy_pattern(query), re.IGNORECASE)
        return lambda line: pattern.search(line) is not None

    def _iter_files(self, scope: Path, patterns: list[str]) -> Iterator[Path]:
        if scope.is_file():
            yield scope
            return

        stack = [scope]
        while stack:
            current = stack.pop()
            try:
                entries = sorted(current.iterdir(), key=lambda p: p.name)
            except OSError:
                continue
            for entry in entries:
                is_dir = entry.is_dir()
                if should_skip(entry.name, is_dir, patterns):
                    continue
                if is_dir:
                    stack.append(entry)
                else:
                    yield entry

    def _respond(self, results: list[str]) -> ToolResult:
        if not results:
            return ToolResult.ok("No matches found.", match_count=0)
        return ToolResult.ok("\n".join(results), match_count=len(results))
