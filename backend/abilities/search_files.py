"""SearchFilesAbility — locate files by name (glob) or content (grep).

Cross-platform alternative to ``bash find`` / ``bash grep`` so the LLM
gets consistent behaviour across macOS, Linux, and Windows — including
mounted drives and connected storage.

Returns a sealed :class:`abilities._result.ToolResult` (never a wire envelope):

* ``glob`` success → ``{"files": [<abs path>, …]}`` with ``meta count=<n>``.
* ``grep`` success → ``[{"file": <abs>, "line": <int>, "text": <line>,
  "context": <surrounding lines>}, …]`` — one row per matched line — with
  ``meta count=<rows>``.
* A cap that bit (more matches than ``max_files`` OR the walk budget expired)
  → ``meta truncated=true`` so the cut is never silent.
* Zero hits → SUCCESS with ``count=0``, empty rows, and a broaden suggestion in
  the body — not an error.
* Bad inputs → ``err()`` with a stable kebab ``code`` (``unknown-action`` /
  ``invalid-param`` / ``empty-query`` / ``directory-not-found`` /
  ``not-a-directory`` / ``invalid-regex``).
"""

import os
import re
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import ClassVar

from abilities._ability import Ability
from abilities._params import Keys
from abilities._result import ToolResult

_GLOB_DEFAULT_MAX_FILES = 10
_GREP_DEFAULT_MAX_FILES = 5
_MAX_MAX_FILES = 200
_DEFAULT_CONTEXT_LINES = 5
_MAX_CONTEXT_LINES = 20
_BUDGET_RATIO = 0.8
# Cooperative wall-clock budget (seconds) for a single glob/grep walk. This is
# NOT a framework execution timeout — it bounds the filesystem traversal itself
# so a walk of an enormous tree returns a partial result with exhausted=True
# instead of crawling indefinitely.
_WALK_BUDGET_S = 30

_NARROW_HINT = "Narrow the directory or tighten the pattern for complete results."
_BROADEN_HINT = "Broaden the pattern or widen the directory and try again."


class SearchFilesAbility(Ability):
    # The dispatcher pre-gates these BEFORE the policy gate: an unknown action →
    # code=unknown-action with valid=(glob, grep); a present action missing
    # 'query' → a single code=missing-params error. A present-but-whitespace
    # query passes the pre-gate (the value is truthy), so run() still guards it.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "glob": (Keys.query,),
        "grep": (Keys.query,),
    }

    def get_name(self) -> str:
        return "search_files"

    def get_summary(self) -> str:
        return (
            "Locate files on disk by filename pattern (glob) or by content (grep). "
            "Use this BEFORE reaching for bash when you need to find a file you "
            "don't already know the path of."
        )

    def get_examples(self) -> list[str]:
        return [
            "find all yaml files under config",
            "where is the message_processor file",
            "search the backend for files containing 'PolicyService'",
            "list every markdown doc under docs/",
            "grep for 'TODO' in my notes folder",
            "which file defines _FIND_TOOLS_GUARDRAILS",
            "show me all log files in /tmp",
            "find files matching test_*.py",
        ]

    def get_search_tooltip(self) -> str:
        return "Find files by name or content"

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": ["glob", "grep"],
                "description": (
                    "'glob' = match files by name/pattern (e.g. '*.py', "
                    "'**/test_*.yaml'). 'grep' = search for content matches "
                    "inside files. Pick 'glob' when you know the filename "
                    "shape, 'grep' when you know what's inside the file."
                ),
            },
            Keys.query: {
                "type": "string",
                "description": (
                    "For 'glob': a filename or glob pattern (e.g. '*.md', "
                    "'**/*.log', 'config.*'). For 'grep': the literal "
                    "string or regex to search file contents for."
                ),
            },
            Keys.directory: {
                "type": "string",
                "description": (
                    "Optional directory to search under. Defaults to the "
                    "filesystem root (/). Provide a narrower path when "
                    "possible for faster results. Absolute path or "
                    "~-prefixed home-relative path."
                ),
            },
            Keys.max_files: {
                "type": "integer",
                "description": (
                    "Maximum number of files to return. "
                    "Defaults to 10 for glob, 5 for grep. Maximum 200. "
                    "When more matches exist than this cap, the result carries "
                    "truncated=true."
                ),
            },
            Keys.context_lines: {
                "type": "integer",
                "description": (
                    "Grep only. Number of lines to show above AND below "
                    "each matched line, carried on each row's 'context' field. "
                    "Defaults to 5. Maximum 20."
                ),
            },
        },
        "required": [Keys.action, Keys.query],
    }

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    def run(self, params: dict) -> ToolResult:
        action = params.get(Keys.action, "")
        query = (params.get(Keys.query) or "").strip()
        directory = (params.get(Keys.directory) or "").strip()

        # The dispatcher's ACTION_REQUIRED pre-gate has already rejected an
        # unknown action and a missing 'query'; this guards the residue it lets
        # through — a present-but-whitespace query (truthy to the pre-gate).
        if action not in ("glob", "grep"):
            return ToolResult.err(
                f"Invalid action {action!r}. Must be 'glob' or 'grep'.",
                code="unknown-action",
                valid=("glob", "grep"),
            )
        if not query:
            return ToolResult.err(
                "query is required and must not be empty.",
                code="empty-query",
                hint="pass a non-empty filename pattern (glob) or search string (grep).",
            )

        default_max = _GREP_DEFAULT_MAX_FILES if action == "grep" else _GLOB_DEFAULT_MAX_FILES
        max_files_raw = params.get(Keys.max_files)
        if max_files_raw is not None:
            try:
                max_files = int(max_files_raw)
            except (TypeError, ValueError):
                return ToolResult.err(
                    f"max_files must be an integer, got {max_files_raw!r}.",
                    code="invalid-param",
                    hint=f"pass an integer between 1 and {_MAX_MAX_FILES}.",
                )
            if max_files < 1 or max_files > _MAX_MAX_FILES:
                return ToolResult.err(
                    f"max_files must be between 1 and {_MAX_MAX_FILES}, got {max_files}.",
                    code="invalid-param",
                    hint=f"pass an integer between 1 and {_MAX_MAX_FILES}.",
                )
        else:
            max_files = default_max

        context_lines = _DEFAULT_CONTEXT_LINES
        if action == "grep":
            cl_raw = params.get(Keys.context_lines)
            if cl_raw is not None:
                try:
                    context_lines = int(cl_raw)
                except (TypeError, ValueError):
                    return ToolResult.err(
                        f"context_lines must be an integer, got {cl_raw!r}.",
                        code="invalid-param",
                        hint=f"pass an integer between 0 and {_MAX_CONTEXT_LINES}.",
                    )
                if context_lines < 0 or context_lines > _MAX_CONTEXT_LINES:
                    return ToolResult.err(
                        f"context_lines must be between 0 and {_MAX_CONTEXT_LINES}, "
                        f"got {context_lines}.",
                        code="invalid-param",
                        hint=f"pass an integer between 0 and {_MAX_CONTEXT_LINES}.",
                    )

        root_str = directory or "/"
        try:
            root = Path(os.path.expanduser(root_str)).resolve()
        except (OSError, RuntimeError) as exc:
            return ToolResult.err(
                f"Invalid directory {root_str!r}: {str(exc)[:120]}",
                code="directory-not-found",
                hint="pass an absolute path or a ~-prefixed home-relative path.",
            )

        if not root.exists():
            return ToolResult.err(
                f"Directory not found: {root}",
                code="directory-not-found",
                hint="pass a directory that exists; an absolute path is safest.",
            )
        if not root.is_dir():
            return ToolResult.err(
                f"Not a directory: {root}",
                code="not-a-directory",
                hint="pass a directory path, not a file path.",
            )

        budget = _WALK_BUDGET_S * _BUDGET_RATIO
        deadline = time.monotonic() + budget

        # re.error is the ONLY caught error — a bad grep pattern is a user input
        # fault. Every other failure bubbles to the dispatcher's
        # unhandled-exception wrapper; there is no broad except here.
        if action == "glob":
            files, exhausted = _do_glob(root, query, max_files, deadline)
            return _glob_result(files, exhausted, max_files)

        try:
            rows, exhausted, capped = _do_grep(root, query, max_files, context_lines, deadline)
        except re.error as exc:
            return ToolResult.err(
                f"Invalid regex {query!r}: {str(exc)[:120]}",
                code="invalid-regex",
                hint="escape the special characters or pass a valid Python regex.",
            )
        return _grep_result(rows, exhausted, capped)


def _glob_result(files: list[str], exhausted: bool, max_files: int) -> ToolResult:
    truncated = len(files) > max_files
    shown = files[:max_files] if truncated else files
    capped = truncated or exhausted

    body: dict = {"files": shown}
    if not shown:
        body["note"] = f"No files matched. {_BROADEN_HINT}"
    elif capped:
        body["note"] = _NARROW_HINT

    meta: dict = {"count": len(shown)}
    if capped:
        meta["truncated"] = True
    return ToolResult.ok(body, **meta)


def _grep_result(rows: list[dict], exhausted: bool, capped: bool) -> ToolResult:
    """*capped* is True when the max_files file cap stopped the walk early;
    *exhausted* is True when the walk budget expired. Either signal surfaces as
    ``meta truncated=true``.
    """
    truncated = capped or exhausted
    if not rows:
        return ToolResult.ok(
            {"matches": [], "note": f"No matches found. {_BROADEN_HINT}"},
            count=0,
        )

    meta: dict = {"count": len(rows)}
    if truncated:
        meta["truncated"] = True
        return ToolResult.ok({"matches": rows, "note": _NARROW_HINT}, **meta)
    return ToolResult.ok(rows, **meta)


def _iter_files(root: Path, deadline: float):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if time.monotonic() > deadline:
                return
            yield os.path.join(dirpath, name)


def _do_glob(root: Path, pattern: str, max_files: int, deadline: float) -> tuple[list[str], bool]:
    """The returned list may be LONGER than *max_files*; the caller slices and
    decides whether to flag truncation, so a "more existed" signal is not lost
    in the slice.
    """
    matches: list[tuple[float, str]] = []
    recursive = "**" in pattern or "/" in pattern
    exhausted = False
    for fp in _iter_files(root, deadline):
        if recursive:
            rel = os.path.relpath(fp, root)
            if not (fnmatch(rel, pattern) or fnmatch(fp, pattern)):
                continue
        else:
            if not fnmatch(os.path.basename(fp), pattern):
                continue
        try:
            mtime = os.path.getmtime(fp)
        except OSError:
            continue
        matches.append((mtime, fp))

    if time.monotonic() > deadline:
        exhausted = True

    matches.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in matches], exhausted


def _do_grep(
    root: Path, query: str, max_files: int, context_lines: int, deadline: float
) -> tuple[list[dict], bool, bool]:
    """*capped* is True when the max_files file cap stopped the walk before the
    tree was exhausted; *exhausted* is True when the walk budget expired. Files
    are visited newest-first so the rows favour recent files.
    """
    pattern = re.compile(query)
    files_with_hits: list[tuple[float, str, list[str]]] = []
    capped = False
    exhausted = False
    for fp in _iter_files(root, deadline):
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        if not any(pattern.search(line) for line in lines):
            continue
        try:
            mtime = os.path.getmtime(fp)
        except OSError:
            mtime = 0.0
        files_with_hits.append((mtime, fp, lines))
        if len(files_with_hits) >= max_files:
            capped = True
            break

    if time.monotonic() > deadline:
        exhausted = True

    files_with_hits.sort(key=lambda t: t[0], reverse=True)

    rows: list[dict] = []
    for _mtime, fp, lines in files_with_hits:
        rows.extend(_match_rows(fp, lines, pattern, context_lines))
    return rows, exhausted, capped


def _match_rows(
    fp: str, lines: list[str], pattern: re.Pattern, context_lines: int
) -> list[dict]:
    rows: list[dict] = []
    for i, line in enumerate(lines):
        if not pattern.search(line):
            continue
        row: dict = {"file": fp, "line": i + 1, "text": line.rstrip()}
        if context_lines > 0:
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            row["context"] = "\n".join(lines[ln].rstrip() for ln in range(start, end))
        rows.append(row)
    return rows
