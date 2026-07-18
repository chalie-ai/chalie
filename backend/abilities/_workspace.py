# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared workspace infrastructure for the code_agent inner toolkit.

The leading underscore keeps this module out of ability auto-discovery
(``AbilityRegistry`` globs ``*.py`` and skips ``_``-prefixed files) — it is
plumbing, not a tool.

Chalie has exactly ONE global workspace root (``FileMapperService``'s
``code_agent_workspace_path``), unlike the coding-agent source this toolkit is
ported from, which resolved paths under a per-invocation ``root``. Every
resolution helper below is therefore single-argument, always anchored to that
one root.

Exports
-------
get_workspace_root      : the lazily-created sandbox root every path resolves under
resolve_in_root         : verify a candidate path stays under the workspace root
resolve_existing_file    : resolve a path under the root and confirm it is a regular file
looks_line_numbered      : True when a search string carries read_file's number prefix
load_gitignore_patterns  : parse the workspace root's .gitignore, if present
should_skip              : True when a directory-walk entry is built-in-ignored or gitignored
locate_rg                : the ripgrep binary path, or None when it is not on PATH
IGNORED_DIRS             : directory names never worth walking when a tool sweeps the tree
"""

from __future__ import annotations

import fnmatch
import re
import shutil
from pathlib import Path

from abilities._result import ToolResult

# Directory names never worth walking when a tool sweeps the workspace tree:
# version-control internals, Python bytecode caches, and vendored/virtual-env
# trees. Shared here so every tool that walks the tree agrees on the same skip
# set instead of keeping private copies that can drift apart. Tools with a
# deliberately different skip policy (list_files' gitignore-driven listing)
# keep their own filter on top of this one.
IGNORED_DIRS = frozenset({".git", "__pycache__", "node_modules", ".venv", "venv"})

# read_file renders content cat -n style ("     1\t<line>"); this matches a
# leading, optionally space-padded line number followed by the tab separator.
_LINE_NUMBER_PREFIX = re.compile(r"^\s*\d+\t")


def get_workspace_root() -> Path:
    """Return the code_agent sandbox root, creating it on first use.

    Lazy so importing this module (and every ability built on it) never
    touches the filesystem — only an actual tool call does.
    """
    from services.file_mapper_service import FileMapperService  # noqa: PLC0415

    root = FileMapperService.get_code_agent_workspace_path()
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_in_root(candidate: str | Path) -> Path:
    """Resolve *candidate* as a path under the code_agent workspace root.

    The candidate must be relative; absolute candidates are rejected. Symlinks
    are followed via :meth:`Path.resolve`. After joining the root and
    *candidate* the resolved real path must not land outside the resolved real
    root — this covers dot-dot traversal as well as symlinks pointing outside
    the workspace.

    Args:
        candidate: A relative path to resolve under the workspace root.

    Returns:
        The resolved absolute ``Path`` when it is safe (equal to or inside the root).

    Raises:
        ValueError: When *candidate* is absolute, or the resolved path escapes the root.
    """
    root_path = get_workspace_root().resolve()
    cand_path = Path(candidate)

    if cand_path.is_absolute():
        raise ValueError(
            f"absolute paths are not allowed; paths must be given relative "
            f"to the workspace root. Got: {candidate!r}"
        )

    resolved = (root_path / cand_path).resolve()

    if resolved == root_path or resolved.is_relative_to(root_path):
        return resolved

    raise ValueError(
        f"path escapes the workspace root: {candidate!r} "
        f"resolves to {resolved}, which is not under {root_path}"
    )


def resolve_existing_file(raw_path: str) -> "tuple[Path | None, ToolResult | None]":
    """Resolve *raw_path* under the workspace root and confirm it is a regular file.

    Returns ``(resolved, None)`` when the path stays under the root and points
    at an existing regular file, or ``(None, error)`` carrying the shared
    error contract otherwise: ``path-escapes-root``, ``not-found`` (with the
    create_file hint), or ``not-a-file``.
    """
    try:
        resolved = resolve_in_root(raw_path)
    except ValueError as exc:
        return None, ToolResult.err(str(exc), code="path-escapes-root")

    if not resolved.exists():
        return None, ToolResult.err(
            f"{raw_path} does not exist.",
            code="not-found",
            hint="Use create_file to create a new file.",
        )

    if not resolved.is_file():
        return None, ToolResult.err(
            f"{raw_path} is not a regular file.",
            code="not-a-file",
        )

    return resolved, None


def looks_line_numbered(text: str) -> bool:
    """Return True when every non-empty line of *text* carries a line-number prefix.

    read_file renders content cat -n style (``     1\\t<line>``); a model
    sometimes pastes those numbered lines straight into a ``search`` string,
    which then matches nothing. This cheap check runs only on the no-match
    path so the replace tools can point at the real cause instead of a
    generic "not found".
    """
    non_empty = [line for line in text.split("\n") if line != ""]
    if not non_empty:
        return False
    return all(_LINE_NUMBER_PREFIX.match(line) for line in non_empty)


def load_gitignore_patterns(root: Path) -> list[str]:
    """Load glob patterns from the ``.gitignore`` at *root*, if present.

    Skips blank lines and lines starting with ``#``, keeps trailing-slash
    directory-only patterns as-is, and returns everything else verbatim so
    :func:`should_skip` can match against a basename later.
    """
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return []

    patterns: list[str] = []
    for line in gitignore.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)
    return patterns


def should_skip(name: str, is_dir: bool, patterns: list[str]) -> bool:
    """Return True when *name* should be skipped during a workspace tree walk.

    Built-in exclusions (:data:`IGNORED_DIRS`) always apply; *patterns*
    (from :func:`load_gitignore_patterns`) apply on top when the workspace
    carries a ``.gitignore``.
    """
    if is_dir and name in IGNORED_DIRS:
        return True

    for pat in patterns:
        if pat.endswith("/"):
            if is_dir and fnmatch.fnmatch(name, pat.rstrip("/")):
                return True
            continue
        if fnmatch.fnmatch(name, pat):
            return True

    return False


def locate_rg() -> str | None:
    """Return the ripgrep binary's path, or None when it is not on PATH.

    Chalie's installer does not bundle ripgrep, so every caller of this
    helper MUST carry a pure-Python fallback for the None case — it is the
    normal path in the shipped product, not a defensive nicety.
    """
    return shutil.which("rg")
