# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SetPermissionsAbility — change Unix permissions (chmod) on a file or folder.

One action, one outcome: change the Unix permissions of an existing file or
folder to an octal mode. The target must exist; a missing path is an error.
The permission code is validated before any system call — an invalid code is
rejected with ``code=invalid-param`` before the kernel is touched.

Returns a sealed :class:`abilities._result.ToolResult` (never a wire envelope):

* Success → ``{"path": <abs>, "mode_octal": <str>}`` with the mode that was
  actually set (which may differ from the requested mode if the kernel
  masked bits).
* Bad inputs → ``err()`` with a stable kebab ``code`` (``not-found`` /
  ``invalid-param`` / ``permission-denied``) — errors never masquerade as
  success.
"""

from __future__ import annotations

import os
from typing import ClassVar

from abilities._ability import Ability
from abilities._paths import absolute_target
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.set_permissions_params_bag import SetPermissionsParamsBag
from contracts.params.param_bag import ParamBag
from configs.enums.ability_category import AbilityCategory


class SetPermissionsAbility(Ability[SetPermissionsParamsBag]):
    """Change the Unix permissions (chmod) of an existing file or folder to an
    octal mode."""

    #: Action-less tool — the ``""`` key drives the dispatcher's
    #: ACTION_REQUIRED pre-gate to reject a missing/blank ``path`` or
    #: ``permission_code`` with ``code=missing-params`` BEFORE run().
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "": (Keys.path, Keys.permission_code),
    }

    NAME: ClassVar[str] = "set_permissions"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.FILE_OPERATIONS

    # The typed input contract: the dispatch seam builds the bag via
    # SetPermissionsParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = SetPermissionsParamsBag

    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("file permissions", "permissions", "chmod")

    def get_summary(self) -> str:
        return (
            "Change the Unix permissions (chmod) of an existing file or folder "
            "to an octal mode."
        )

    def get_examples(self) -> list[str]:
        return [
            "make /workspace/scripts/deploy.sh executable with 0755",
            "set permissions on /workspace/config.ini to 0644",
            "chmod 0755 /workspace/scripts",
            "change permissions on the file to 0600",
            "make the folder executable",
            "set /workspace/data to 0755",
            "chmod 0644 /home/user/docs",
            "change permissions on /workspace/cache to 0700",
        ]

    def get_search_tooltip(self) -> str:
        return "Change file or folder permissions"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.path: {
                "type": "string",
                "description": "Absolute path of the file or folder to change.",
            },
            Keys.permission_code: {
                "type": "string",
                "description": (
                    "Unix octal permission code as a string, e.g. '0644' or "
                    "'0755'."
                ),
            },
        },
        "required": [Keys.path, Keys.permission_code],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: SetPermissionsParamsBag) -> ToolResult:
        target = absolute_target(params.path)
        if isinstance(target, ToolResult):
            return target

        if not target.exists():
            return ToolResult.err(
                f"No such file or directory: {target}",
                code="not-found",
                hint="pass an absolute path to an existing file or directory.",
            )

        mode = self._parse_octal(params.permission_code)
        if mode is None:
            return ToolResult.err(
                f"Invalid permission code: {params.permission_code!r}.",
                code="invalid-param",
                hint="pass a 3- or 4-digit octal string like '0644' or '0755'.",
            )

        try:
            before = target.stat().st_mode & 0o7777
        except PermissionError as exc:
            return ToolResult.err(
                f"Permission denied reading {target}: {exc}",
                code="permission-denied",
                hint=f"you do not own {target} or cannot stat it.",
            )

        try:
            os.chmod(target, mode)
            after = target.stat().st_mode & 0o7777
        except PermissionError as exc:
            return ToolResult.err(
                f"Permission denied changing {target}: {exc}",
                code="permission-denied",
                hint=f"you must own {target} to change its permissions.",
            )
        except OSError as exc:
            return ToolResult.err(
                f"Could not change permissions on {target}: {exc}",
                code="invalid-param",
                hint="check the path exists and is a regular file or directory.",
            )

        return ToolResult.ok(
            {"path": str(target), "mode_octal": self._format_octal(after)},
            previous=self._format_octal(before),
        )

    @staticmethod
    def _parse_octal(text: str) -> int | None:
        """Parse a 3- or 4-digit octal string into an int; reject anything else.

        Accepts ``"755"`` and ``"0755"`` alike. Rejects empty strings, non-octal
        characters, and any value outside ``0o000``--``0o7777``.
        """
        if not text:
            return None
        if len(text) not in (3, 4):
            return None
        if not all(c in "01234567" for c in text):
            return None
        value = int(text, 8)
        if value > 0o7777:
            return None
        return value

    @staticmethod
    def _format_octal(st_mode: int) -> str:
        return format(st_mode & 0o7777, "04o")
