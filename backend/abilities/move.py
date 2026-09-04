"""MoveAbility — move or rename a file or folder between two absolute paths.

Returns a sealed :class:`abilities._result.ToolResult` (never a wire envelope):

* Success → ``{"from": <src>, "to": <dst>, "moved": True}``.
* Bad inputs → ``err()`` with a stable kebab ``code`` (``invalid-path`` /
  ``not-found`` / ``destination-exists`` / ``permission-denied``) — errors never
  masquerade as success.
"""

import shutil
from typing import ClassVar

from abilities._ability import Ability
from abilities._paths import absolute_target
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.move_params_bag import MoveParamsBag
from contracts.params.param_bag import ParamBag
from configs.enums.ability_category import AbilityCategory


class MoveAbility(Ability[MoveParamsBag]):
    #: Action-less tool — the ``""`` key drives the dispatcher's ACTION_REQUIRED
    #: pre-gate to reject a missing/blank ``current_path`` with
    #: ``code=missing-params`` BEFORE run().
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.current_path, Keys.new_path)}

    # The typed input contract: the dispatch seam builds the bag via
    # MoveParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = MoveParamsBag

    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("move file", "rename file", "rename")
    NAME: ClassVar[str] = "move"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.FILE_OPERATIONS

    def get_summary(self) -> str:
        return "Move or rename a file or folder from one absolute path to another."

    def get_examples(self) -> list[str]:
        return [
            "rename this file",
            "move the script into the scripts folder",
            "rename that folder to something more descriptive",
            "move the project directory to a new location",
            "rename the config file to a proper name",
            "move the downloaded file to my documents",
        ]

    def get_search_tooltip(self) -> str:
        return "Move or Rename file"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.current_path: {
                "type": "string",
                "description": "Absolute path of the file or folder to move.",
            },
            Keys.new_path: {
                "type": "string",
                "description": "Absolute destination path (new name or new location).",
            },
        },
        "required": [Keys.current_path, Keys.new_path],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: MoveParamsBag) -> ToolResult:
        current_path_str = params.current_path
        new_path_str = params.new_path

        src = absolute_target(current_path_str, label="current_path")
        if isinstance(src, ToolResult):
            return src
        dst = absolute_target(new_path_str, label="new_path")
        if isinstance(dst, ToolResult):
            return dst

        if not src.exists():
            return ToolResult.err(
                f"{src} does not exist.",
                code="not-found",
                hint="check the source path and that the file or folder is present.",
            )

        if dst.exists():
            return ToolResult.err(
                f"{dst} already exists.",
                code="destination-exists",
                hint="delete it first or pick another name.",
            )

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        except PermissionError as exc:
            return ToolResult.permission_denied("moving", src, exc, hint="you do not have the required permissions to move this item.")
        except OSError as exc:
            return ToolResult.path_os_error("move", src, exc, code="invalid-path", hint="check the path shape and that its parent can hold the destination.")

        return ToolResult.ok({"from": str(src), "to": str(dst), "moved": True})
