"""SearchFilesAbility — locate files by name (glob) or content (grep).

Safe purpose-built alternative to ``bash find`` / ``bash grep`` so the LLM
does not need shell to discover files. Returns a minimal JSON list of
absolute file paths (no excerpts, no line numbers) plus a ``hint`` pointing
to the ``read`` tool for follow-up.
"""

import json
import logging
import os
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import ClassVar

from abilities._base import Ability

logger = logging.getLogger(__name__)

_RESULT_CAP = 200
_GREP_MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB
_SKIP_DIR_NAMES: frozenset[str] = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".cache",
    ".mypy_cache", ".pytest_cache", ".tox",
})
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
                    "user's home directory. Must be an absolute path."
                ),
            },
        },
        "required": ["action", "query"],
    }
    TIMEOUT = 15

    _BLOCKED_PATH_PREFIXES: ClassVar[tuple] = ("/etc", "/proc", "/dev", "/sys", "/var/run")

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        action = params.get("action", "")
        query = (params.get("query") or "").strip()
        directory = (params.get("directory") or "").strip()

        if action not in ("glob", "grep"):
            return _error("invalid-action", action=action)
        if not query:
            return _error("query-required", action=action)

        root_str = directory or str(Path.home())
        try:
            root = Path(os.path.expanduser(root_str)).resolve()
        except (OSError, RuntimeError) as exc:
            return _error("invalid-directory", action=action, directory=root_str, detail=str(exc)[:120])

        if _is_blocked(root) or _is_blocked(Path(os.path.expanduser(root_str))):
            return _error("system-path-blocked", action=action, directory=str(root))
        if not root.exists():
            return _error("directory-not-found", action=action, directory=str(root))
        if not root.is_dir():
            return _error("not-a-directory", action=action, directory=str(root))

        try:
            if action == "glob":
                paths, truncated = _do_glob(root, query)
            else:
                paths, truncated = _do_grep(root, query)
        except re.error as exc:
            return _error("invalid-regex", action=action, query=query, detail=str(exc)[:120])
        except ValueError as exc:
            return _error("invalid-glob", action=action, query=query, detail=str(exc)[:120])
        except Exception as exc:
            logger.exception("[SEARCH_FILES] Unexpected error action=%s query=%r: %s", action, query, exc)
            return _error("search-failed", action=action, query=query, detail=str(exc)[:120])

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


def _is_blocked(path: Path) -> bool:
    s = str(path)
    return any(s == p or s.startswith(p + os.sep) for p in SearchFilesAbility._BLOCKED_PATH_PREFIXES)


def _iter_files(root: Path):
    """Yield absolute file paths under *root*, skipping VCS/cache dirs and symlinks."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES]
        for name in filenames:
            yield os.path.join(dirpath, name)


def _do_glob(root: Path, pattern: str) -> tuple[list[str], bool]:
    matches: list[tuple[float, str]] = []
    recursive = "**" in pattern or "/" in pattern
    for fp in _iter_files(root):
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
    truncated = len(matches) > _RESULT_CAP
    return [p for _, p in matches[:_RESULT_CAP]], truncated


def _do_grep(root: Path, query: str) -> tuple[list[str], bool]:
    pattern = re.compile(query)
    hits: list[tuple[float, str]] = []
    truncated = False
    for fp in _iter_files(root):
        try:
            size = os.path.getsize(fp)
        except OSError:
            continue
        if size > _GREP_MAX_FILE_BYTES:
            continue
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                if not pattern.search(fh.read()):
                    continue
            mtime = os.path.getmtime(fp)
        except OSError:
            continue
        hits.append((mtime, fp))
        if len(hits) > _RESULT_CAP:
            truncated = True
            break

    hits.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in hits[:_RESULT_CAP]], truncated


def _error(code: str, **fields) -> dict:
    payload = {"status": "error", "error": code}
    payload.update({k: v for k, v in fields.items() if v not in (None, "")})
    return {"text": json.dumps(payload)}
