"""SearchFilesAbility — locate files by name (glob) or content (grep).

Cross-platform alternative to ``bash find`` / ``bash grep`` so the LLM
gets consistent behaviour across macOS, Linux, and Windows — including
mounted drives and connected storage.
"""

import logging
import os
import re
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import ClassVar

from abilities._base import Ability

logger = logging.getLogger(__name__)

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


class SearchFilesAbility(Ability):
    NAME = "search_files"
    SUMMARY = (
        "Locate files on disk by filename pattern (glob) or by content (grep). "
        "Use this BEFORE reaching for bash when you need to find a file you "
        "don't already know the path of."
    )
    SEARCH_TOOLTIP = "Find files by name or content"
    POLICY_CATEGORY = "Files"
    POLICY_LABELS: ClassVar[dict[str, str]] = {
        "glob": "Find files by name/pattern",
        "grep": "Search file contents",
    }
    EXAMPLES: ClassVar[list[str]] = [
        "find all yaml files under chalie-nightly-test",
        "where is the message_processor file",
        "search the backend for files containing 'PolicyService'",
        "list every markdown doc under docs/",
        "grep for 'TODO' in my notes folder",
        "which file defines _FIND_TOOLS_GUARDRAILS",
        "show me all log files in /tmp",
        "find files matching test_*.py",
    ]
    INPUT_SCHEMA: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["glob", "grep"],
                "description": (
                    "'glob' = match files by name/pattern (e.g. '*.py', "
                    "'**/test_*.yaml'). 'grep' = search for content matches "
                    "inside files. Pick 'glob' when you know the filename "
                    "shape, 'grep' when you know what's inside the file."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "For 'glob': a filename or glob pattern (e.g. '*.md', "
                    "'**/*.log', 'config.*'). For 'grep': the literal "
                    "string or regex to search file contents for."
                ),
            },
            "directory": {
                "type": "string",
                "description": (
                    "Optional directory to search under. Defaults to the "
                    "filesystem root (/). Provide a narrower path when "
                    "possible for faster results. Absolute path or "
                    "~-prefixed home-relative path."
                ),
            },
            "max_files": {
                "type": "integer",
                "description": (
                    "Maximum number of files to return. "
                    "Defaults to 10 for glob, 5 for grep. Maximum 200."
                ),
            },
            "context_lines": {
                "type": "integer",
                "description": (
                    "Grep only. Number of lines to show above AND below "
                    "each matched line. Defaults to 5. Maximum 20."
                ),
            },
        },
        "required": ["action", "query"],
    }

    def run(self, params: dict) -> dict:
        action = params.get("action", "")
        query = (params.get("query") or "").strip()
        directory = (params.get("directory") or "").strip()

        if action not in ("glob", "grep"):
            return {"text": f"Invalid action '{action}'. Must be 'glob' or 'grep'."}
        if not query:
            return {"text": "query is required and must not be empty."}

        default_max = _GREP_DEFAULT_MAX_FILES if action == "grep" else _GLOB_DEFAULT_MAX_FILES
        max_files_raw = params.get("max_files")
        if max_files_raw is not None:
            try:
                max_files = int(max_files_raw)
            except (TypeError, ValueError):
                return {"text": f"max_files must be an integer, got '{max_files_raw}'."}
            if max_files < 1 or max_files > _MAX_MAX_FILES:
                return {"text": f"max_files must be between 1 and {_MAX_MAX_FILES}, got {max_files}."}
        else:
            max_files = default_max

        context_lines = _DEFAULT_CONTEXT_LINES
        if action == "grep":
            cl_raw = params.get("context_lines")
            if cl_raw is not None:
                try:
                    context_lines = int(cl_raw)
                except (TypeError, ValueError):
                    return {"text": f"context_lines must be an integer, got '{cl_raw}'."}
                if context_lines < 0 or context_lines > _MAX_CONTEXT_LINES:
                    return {"text": f"context_lines must be between 0 and {_MAX_CONTEXT_LINES}, got {context_lines}."}

        root_str = directory or "/"
        try:
            root = Path(os.path.expanduser(root_str)).resolve()
        except (OSError, RuntimeError) as exc:
            return {"text": f"Invalid directory '{root_str}': {str(exc)[:120]}"}

        if not root.exists():
            return {"text": f"Directory not found: {root}"}
        if not root.is_dir():
            return {"text": f"Not a directory: {root}"}

        budget = _WALK_BUDGET_S * _BUDGET_RATIO
        deadline = time.monotonic() + budget

        try:
            if action == "glob":
                paths, exhausted = _do_glob(root, query, max_files, deadline)
                text = "\n".join(paths) if paths else "No files matched."
                if exhausted:
                    text += (
                        f"\n\n(search stopped after {budget:.0f}s — "
                        "try narrowing the directory for complete results)"
                    )
                return {"text": text}
            else:
                text, exhausted = _do_grep(root, query, max_files, context_lines, deadline)
                if not text:
                    text = "No matches found."
                if exhausted:
                    text += (
                        f"\n\n(search stopped after {budget:.0f}s — "
                        "try narrowing the directory for complete results)"
                    )
                return {"text": text}
        except re.error as exc:
            return {"text": f"Invalid regex '{query}': {str(exc)[:120]}"}
        except Exception as exc:
            logger.exception("[SEARCH_FILES] Unexpected error action=%s query=%r: %s", action, query, exc)
            return {"text": f"Search failed: {str(exc)[:120]}"}


def _iter_files(root: Path, deadline: float):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if time.monotonic() > deadline:
                return
            yield os.path.join(dirpath, name)


def _do_glob(root: Path, pattern: str, max_files: int, deadline: float) -> tuple[list[str], bool]:
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
    return [p for _, p in matches[:max_files]], exhausted


def _do_grep(root: Path, query: str, max_files: int, context_lines: int, deadline: float) -> tuple[str, bool]:
    pattern = re.compile(query)
    file_blocks: list[tuple[float, str]] = []
    exhausted = False
    for fp in _iter_files(root, deadline):
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except OSError:
            continue

        block = _format_file_block(fp, lines, pattern, context_lines)
        if not block:
            continue

        try:
            mtime = os.path.getmtime(fp)
        except OSError:
            mtime = 0.0
        file_blocks.append((mtime, block))
        if len(file_blocks) >= max_files:
            break

    if time.monotonic() > deadline:
        exhausted = True

    file_blocks.sort(key=lambda t: t[0], reverse=True)
    return "\n----\n".join(b for _, b in file_blocks), exhausted


def _format_file_block(fp: str, lines: list[str], pattern: re.Pattern, context_lines: int) -> str | None:
    match_indices: list[int] = []
    for i, line in enumerate(lines):
        if pattern.search(line):
            match_indices.append(i)

    if not match_indices:
        return None

    regions: list[tuple[int, int]] = []
    for idx in match_indices:
        start = max(0, idx - context_lines)
        end = min(len(lines), idx + context_lines + 1)
        if regions and start <= regions[-1][1]:
            regions[-1] = (regions[-1][0], end)
        else:
            regions.append((start, end))

    snippet_lines: list[str] = [fp]
    for start, end in regions:
        for ln in range(start, end):
            snippet_lines.append(f"ln {ln + 1}: {lines[ln].rstrip()}")

    return "\n".join(snippet_lines)
