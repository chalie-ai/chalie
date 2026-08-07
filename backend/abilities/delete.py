# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DeleteAbility — delete a file or folder at an absolute path.

One action, one outcome: remove a file, or a folder and everything inside it
(recursively). The deletion is permanent — there is no undo. A missing target
is an error.

Returns a sealed :class:`abilities._result.ToolResult` (never a wire envelope):

* Success → ``{"path": <abs>, "deleted": True}``.
* Bad inputs → ``err()`` with a stable kebab ``code`` (``not-found`` /
  ``permission-denied``) — errors never masquerade as success.
"""

from __future__ import annotations

import shutil
from typing import ClassVar

from abilities._ability import Ability
from abilities._paths import Paths
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.delete_params_bag import DeleteParamsBag
from contracts.params.param_bag import ParamBag
from configs.enums.ability_category import AbilityCategory


class DeleteAbility(Ability[DeleteParamsBag]):
    """Delete a file, or a folder and everything inside it. The deletion is
    permanent and recursive for folders — there is no undo."""

    #: Action-less tool — the ``""`` key drives the dispatcher's ACTION_REQUIRED
    #: pre-gate to reject a missing/blank ``path`` with ``code=missing-params``
    #: BEFORE run().
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.path,)}

    NAME: ClassVar[str] = "delete"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.FILE_OPERATIONS

    # The typed input contract: the dispatch seam builds the bag via
    # DeleteParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = DeleteParamsBag

    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("delete file", "delete folder")

    def get_summary(self) -> str:
        return (
            "Delete a file, or a folder and everything inside it. The deletion "
            "is permanent and recursive for folders — there is no undo."
        )

    def get_examples(self) -> list[str]:
        return [
            "delete the file /workspace/old_report.csv",
            "remove the folder /workspace/temp",
            "delete /workspace/scripts/deploy.sh",
            "remove the entire /workspace/logs directory",
            "delete the file at /tmp/data/raw",
            "remove /workspace/cache",
            "delete the folder /home/user/docs",
            "remove /workspace/backup.tar.gz",
        ]

    def get_search_tooltip(self) -> str:
        return "Delete a file or folder"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.path: {
                "type": "string",
                "description": (
                    "Absolute path of the file or folder to delete. Folders "
                    "are deleted recursively."
                ),
            },
        },
        "required": [Keys.path],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: DeleteParamsBag) -> ToolResult:
        target = Paths.absolute_target(params.path)
        if isinstance(target, ToolResult):
            return target

        if not target.exists():
            return ToolResult.err(
                f"No such file or directory: {target}",
                code="not-found",
                hint="pass an absolute path to an existing file or directory.",
            )

        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        except PermissionError as exc:
            return ToolResult.err(
                f"Permission denied deleting {target}: {exc}",
                code="permission-denied",
                hint=f"you do not own {target} or cannot remove it.",
            )
        except OSError as exc:
            return ToolResult.err(
                f"Could not delete {target}: {exc}",
                code="not-found",
                hint="the path may have been removed by another process.",
            )

        return ToolResult.ok({"path": str(target), "deleted": True})
