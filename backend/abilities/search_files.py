"""SearchFilesAbility — locate files by name (glob) or content (grep).

Cross-platform alternative to ``bash find`` / ``bash grep`` so the LLM
gets consistent behaviour across macOS, Linux, and Windows — including
mounted drives and connected storage.  No path restrictions: the LLM may
search any directory on the system.
"""

import json
import logging
import os
import re
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import ClassVar

from abilities._base import Ability

logger = logging.getLogger(__name__)

_DEFAULT_MAX_FILES = 10
_MAX_MAX_FILES = 200
_GREP_MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB
_DEFAULT_CONTEXT_LINES = 3
_MAX_CONTEXT_LINES = 50
_WALK_DEADLINE_SECONDS = 14  # must finish before TIMEOUT (15s)
_HINT = "To read the contents of the file use the 'read' tool"


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
                    "user's home directory. Absolute path or ~-prefixed "
                    "home-relative path."
                ),
            },
            "max_files": {
                "type": "integer",
                "description": (
                    "Maximum number of files to return. Defaults to 10."
                ),
                "minimum": 1,
                "maximum": 200,
            },
            "context_lines": {
                "type": "integer",
                "description": (
                    "Grep only. Number of lines to show above AND below "
                    "each matched line. Defaults to 3."
                ),
                "minimum": 0,
                "maximum": 50,
            },
        },
        "required": ["action", "query"],
    }
    TIMEOUT = 15

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        action = params.get("action", "")
        query = (params.get("query") or "").strip()
        directory = (params.get("directory") or "").strip()
        raw_max = params.get("max_files")
        max_files = max(1, min(int(raw_max), _MAX_MAX_FILES)) if raw_max is not None else _DEFAULT_MAX_FILES
        raw_ctx = params.get("context_lines")
        context_lines = max(0, min(int(raw_ctx), _MAX_CONTEXT_LINES)) if raw_ctx is not None else _DEFAULT_CONTEXT_LINES

        if action not in ("glob", "grep"):
            return _error("invalid-action", action=action)
        if not query:
            return _error("query-required", action=action)

        root_str = directory or str(Path.home())
        try:
            root = Path(os.path.expanduser(root_str)).resolve()
        except (OSError, RuntimeError) as exc:
            return _error("invalid-directory", action=action, directory=root_str, detail=str(exc)[:120])

        if not root.exists():
            return _error("directory-not-found", action=action, directory=str(root))
        if not root.is_dir():
            return _error("not-a-directory", action=action, directory=str(root))

        try:
            if action == "glob":
                paths, truncated = _do_glob(root, query, max_files)
                return {"text": json.dumps({
                    "status": "success",
                    "action": action,
                    "query": query,
                    "directory": str(root),
                    "count": len(paths),
                    "truncated": truncated,
                    "paths": paths,
                    "hint": _HINT,
                })}
            else:
                results, truncated = _do_grep(root, query, max_files, context_lines)
                return {"text": json.dumps({
                    "status": "success",
                    "action": action,
                    "query": query,
                    "directory": str(root),
                    "count": len(results),
                    "truncated": truncated,
                    "results": results,
                    "hint": _HINT,
                })}
        except re.error as exc:
            return _error("invalid-regex", action=action, query=query, detail=str(exc)[:120])
        except ValueError as exc:
            return _error("invalid-glob", action=action, query=query, detail=str(exc)[:120])
        except Exception as exc:
            logger.exception("[SEARCH_FILES] Unexpected error action=%s query=%r: %s", action, query, exc)
            return _error("search-failed", action=action, query=query, detail=str(exc)[:120])


def _iter_files(root: Path, deadline: float):
    """Yield absolute file paths under *root*, stopping at *deadline*."""
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        if time.monotonic() > deadline:
            return
        for name in filenames:
            yield os.path.join(dirpath, name)


def _do_glob(root: Path, pattern: str, max_files: int) -> tuple[list[str], bool]:
    deadline = time.monotonic() + _WALK_DEADLINE_SECONDS
    matches: list[tuple[float, str]] = []
    recursive = "**" in pattern or "/" in pattern
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

    matches.sort(key=lambda t: t[0], reverse=True)
    truncated = len(matches) > max_files
    return [p for _, p in matches[:max_files]], truncated


def _do_grep(root: Path, query: str, max_files: int, context_lines: int) -> tuple[list[dict], bool]:
    deadline = time.monotonic() + _WALK_DEADLINE_SECONDS
    pattern = re.compile(query)
    results: list[tuple[float, dict]] = []
    truncated = False
    for fp in _iter_files(root, deadline):
        try:
            size = os.path.getsize(fp)
        except OSError:
            continue
        if size > _GREP_MAX_FILE_BYTES:
            continue
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except OSError:
            continue

        file_matches = _extract_context(fp, lines, pattern, context_lines)
        if not file_matches:
            continue

        try:
            mtime = os.path.getmtime(fp)
        except OSError:
            mtime = 0.0
        results.append((mtime, file_matches))
        if len(results) > max_files:
            truncated = True
            break

    results.sort(key=lambda t: t[0], reverse=True)
    return [r for _, r in results[:max_files]], truncated


def _extract_context(fp: str, lines: list[str], pattern: re.Pattern, context_lines: int) -> dict | None:
    """Build a context-annotated result for a single file, or None if no matches."""
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

    snippets: list[str] = []
    for start, end in regions:
        snippet_lines: list[str] = []
        for ln in range(start, end):
            snippet_lines.append(f"ln {ln + 1}: {lines[ln].rstrip()}")
        snippets.append("\n".join(snippet_lines))

    return {
        "file": fp,
        "matches": snippets,
    }


def _error(code: str, **fields) -> dict:
    payload = {"status": "error", "error": code}
    payload.update({k: v for k, v in fields.items() if v not in (None, "")})
    return {"text": json.dumps(payload)}
