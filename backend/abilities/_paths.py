# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared absolute-path gate for the file-operation abilities.

One check, one implementation: every file-touching ability that takes a
``path`` parameter resolves it to an absolute :class:`pathlib.Path` before
doing anything else. This module owns that check — the blank-path case needs
no handling here because every caller's :class:`ParamBag` rejects a blank path
with ``code=missing-params`` before :meth:`~abilities._ability.Ability.run` is
reached.

The function returns either a resolved :class:`pathlib.Path` or a
:class:`ToolResult` error describing why the path is invalid. Callers unpack
the result with an ``isinstance`` guard and early-return on error, so the
resolved path is always safe to use after the guard.
"""

from __future__ import annotations

from pathlib import Path

from abilities._result import ToolResult


def absolute_target(path_str: str) -> Path | ToolResult:
    """Resolve *path_str* to an absolute :class:`pathlib.Path`, or refuse.

    A non-absolute path is rejected with ``code=invalid-path`` and a hint that
    names the fix. The blank-path case cannot happen here: the caller's bag
    rejects a missing or blank ``path`` before this function is reached.
    """
    candidate = Path(path_str)
    if not candidate.is_absolute():
        return ToolResult.err(
            f"Path is not absolute: {path_str!r}.",
            code="invalid-path",
            hint="pass an absolute path (one starting from the filesystem root).",
        )
    return candidate.resolve()
