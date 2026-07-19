# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared base for the replace_one and replace_all abilities.

The leading underscore keeps this module out of ability auto-discovery
(``AbilityRegistry`` globs ``*.py`` and skips ``_``-prefixed files) — it is
plumbing, not a tool.

Holds the walk skip-set both replace tools agree on and the path guard that
validates absolute paths against the filesystem. There is deliberately no
workspace-root containment — the code_agent workspace is a default location,
not an enforced boundary, so any absolute path on the filesystem is accepted.
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult


class ReplaceAbilityBase(Ability, ABC):
    """Shared base for the replace_one and replace_all abilities.

    ``ABC`` sits directly in the bases on purpose: that is the house escape
    hatch (`_ability.py.__init_subclass__`) that skips the concrete-metadata
    probe for abstract mix-ins, while the registry walk still recurses into
    the subtree — unlike ``_SYNTHETIC``, which prunes descendants entirely
    (the MCP-proxy mechanism).
    """

    #: Directory names never worth walking when a tool sweeps a tree:
    #: version-control internals, Python bytecode caches, and vendored or
    #: virtual-env trees. Shared here so both replace tools agree on the same
    #: skip set instead of keeping private copies that can drift apart.
    IGNORED_DIRS: ClassVar[frozenset[str]] = frozenset(
        {".git", "__pycache__", "node_modules", ".venv", "venv"}
    )

    @classmethod
    def _validate_file_path(
        cls, raw_path: str
    ) -> "tuple[Path | None, ToolResult | None]":
        """Validate that *raw_path* is an absolute path to an existing regular file.

        Deliberately enforces no workspace-root containment — any absolute
        path on the filesystem is accepted.

        Returns ``(resolved, None)`` when the path is absolute, resolves, and
        points at an existing regular file, or ``(None, error)`` carrying the
        shared error contract otherwise: ``invalid-path`` (non-absolute or
        resolution failure) or ``not-found``.
        """
        try:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                return None, ToolResult.err(
                    f"path must be absolute, got {raw_path!r}.",
                    code="invalid-path",
                )
            resolved = candidate.resolve()
        except (OSError, ValueError):
            return None, ToolResult.err(
                f"invalid path: {raw_path!r}.",
                code="invalid-path",
            )

        if not resolved.exists():
            return None, ToolResult.err(
                f"{raw_path} does not exist.",
                code="not-found",
                hint="use file_write to create the file first.",
            )

        if not resolved.is_file():
            return None, ToolResult.err(
                f"{raw_path} is not a regular file.",
                code="not-a-file",
            )

        return resolved, None
