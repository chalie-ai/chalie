# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""FindFilesAbility — search file names/paths inside the code_agent workspace.

One of the 12 inner tools of the ``code_agent`` toolkit
(``abilities/code_agent.py``). code-agent-delegate-exclusive; pinned on
CodeAgentConfig only — never reaches any other channel.

Prefers ripgrep when it is on PATH; Chalie's installer does not bundle it, so
the pure-Python walk below is the tool's NORMAL path in the shipped product,
not a defensive fallback.
"""

from __future__ import annotations

import fnmatch as fnmatch_module
import subprocess
from pathlib import Path
from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._result import ToolResult
from abilities._workspace import get_workspace_root, load_gitignore_patterns, locate_rg, resolve_in_root, should_skip
from configs.enums.param_key import Keys

# Characters whose presence marks a pattern as a real glob rather than a plain
# substring; when none appear the pattern is wrapped as *pattern* so a
# literal-minded model still gets useful matches.
_GLOB_METACHARS = frozenset("*?[]{}")

# Maximum number of paths returned; beyond this the model should narrow the
# pattern rather than drown in results.
_MAX_RESULTS = 200

_SEARCH_TIMEOUT_SECONDS = 30


class FindFilesAbility(Ability):
    DISCOVERABLE: ClassVar[bool] = False  # code-agent-delegate-exclusive; pinned on CodeAgentConfig only

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.pattern,)}

    def get_name(self) -> str:
        return "find_files"

    def get_summary(self) -> str:
        return (
            "Find files by name or glob pattern (e.g. '*scheduler*', '*.ts') "
            "in the code_agent workspace, gitignore-aware, returning paths "
            "relative to the workspace root. A pattern with no glob "
            "metacharacters is matched as a substring."
        )

    def get_examples(self) -> list[str]:
        return [
            "find the file called scheduler",
            "list all TypeScript files in the workspace",
            "find files matching *.json",
            "look for a file named config somewhere",
            "find every test file in the project",
            "search for files with 'utils' in the name",
            "find all files under the src folder",
            "locate the main entry point file",
        ]

    def get_search_tooltip(self) -> str:
        return "find files by name in the code workspace"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.pattern: {
                "type": "string",
                "description": (
                    "Filename glob to match case-insensitively, e.g. "
                    "'*scheduler*', '*.sh' or 'src/**/*.ts'. A pattern with no "
                    "glob metacharacters is matched as a substring."
                ),
            },
            Keys.path: {
                "type": "string",
                "description": "Subdirectory relative to the workspace root to limit the search to.",
            },
        },
        "required": [Keys.pattern],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        pattern = cast(str, self.param(params, Keys.pattern, required=True))
        raw_path = cast("str | None", self.param(params, Keys.path))

        if not pattern.strip():
            return ToolResult.err("pattern must be a non-empty glob or substring.", code="empty-pattern")

        if raw_path:
            try:
                scope = resolve_in_root(raw_path)
            except ValueError as exc:
                return ToolResult.err(str(exc), code="path-escapes-root")
            if not scope.is_dir():
                return ToolResult.err(f"{raw_path} is not an existing directory.", code="not-a-directory")
        else:
            scope = get_workspace_root()

        glob = pattern if any(c in _GLOB_METACHARS for c in pattern) else f"*{pattern}*"

        rg_path = locate_rg()
        if rg_path is not None:
            return self._search_with_rg(rg_path, glob, pattern, scope)
        return self._search_pure_python(glob, pattern, scope)

    def _search_with_rg(self, rg_path: str, glob: str, pattern: str, scope: Path) -> ToolResult:
        root = get_workspace_root()

        # A command-line inclusion glob overrides .gitignore in ripgrep, so a
        # broad pattern would surface gitignored files. Stay gitignore-aware by
        # intersecting the glob matches with the plain `rg --files` universe
        # (no inclusion glob, so .gitignore is fully honoured) — rg supplies
        # both the glob semantics and the ignore logic; only the overlap counts.
        matched, err = self._rg_files(rg_path, root, scope, ["--iglob", glob])
        if err is not None:
            return err
        if not matched:
            return ToolResult.ok(f"No files match {pattern!r}.", match_count=0)

        allowed, err = self._rg_files(rg_path, root, scope, [])
        if err is not None:
            return err

        return self._respond(sorted(matched & allowed), pattern)

    def _rg_files(
        self, rg_path: str, root: Path, scope: Path, extra: list[str]
    ) -> tuple[set[str], ToolResult | None]:
        try:
            proc = subprocess.run(  # noqa: S603
                [rg_path, "--files", *extra, str(scope)],
                capture_output=True,
                text=True,
                timeout=_SEARCH_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return set(), ToolResult.err("The search timed out.", code="timeout")

        if proc.returncode not in (0, 1):
            return set(), ToolResult.err(
                proc.stderr.strip() or f"rg exited with code {proc.returncode}.",
                code="search-failed",
            )

        paths: set[str] = set()
        for line in proc.stdout.splitlines():
            if not line:
                continue
            try:
                paths.add(str(Path(line).resolve().relative_to(root.resolve())))
            except ValueError:
                continue
        return paths, None

    def _search_pure_python(self, glob: str, pattern: str, scope: Path) -> ToolResult:
        root = get_workspace_root()
        gitignore_patterns = load_gitignore_patterns(root)

        matches: list[str] = []
        stack = [scope]
        while stack:
            current = stack.pop()
            try:
                entries = sorted(current.iterdir(), key=lambda p: p.name)
            except OSError:
                continue
            for entry in entries:
                is_dir = entry.is_dir()
                if should_skip(entry.name, is_dir, gitignore_patterns):
                    continue
                if is_dir:
                    stack.append(entry)
                    continue
                rel_path = str(entry.relative_to(root))
                if fnmatch_module.fnmatch(entry.name.lower(), glob.lower()) or fnmatch_module.fnmatch(
                    rel_path.lower(), glob.lower()
                ):
                    matches.append(rel_path)

        return self._respond(sorted(matches), pattern)

    def _respond(self, paths: list[str], pattern: str) -> ToolResult:
        total = len(paths)
        if total == 0:
            return ToolResult.ok(f"No files match {pattern!r}.", match_count=0)

        truncated = total > _MAX_RESULTS
        body = "\n".join(paths[:_MAX_RESULTS])
        if truncated:
            body += f"\n\n… {total - _MAX_RESULTS} more match(es) not shown; narrow the pattern to see the rest."

        return ToolResult.ok(body, match_count=total, truncated=truncated)
