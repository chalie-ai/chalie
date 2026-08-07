# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MakeDirAbility — create a directory at an absolute path.

One job, and only that job: make a directory (and any missing parents) at the
caller-supplied absolute path. It cannot create a FILE — that belongs to
``write_file`` alone, which is what dissolves the old two-tools-one-outcome
overlap. Mode comes from the process default; a caller wanting a specific mode
calls :mod:`abilities.set_permissions` afterwards.

Returns a sealed :class:`abilities._result.ToolResult` (never a wire envelope):

* Success → ``{"path": <abs>, "created": True}``.
* Bad inputs → ``err()`` with a stable kebab ``code`` (``already-exists`` /
  ``permission-denied`` / ``invalid-path``) — errors never masquerade as
  success.
"""

from __future__ import annotations

from typing import ClassVar

from abilities._ability import Ability
from abilities._paths import Paths
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.make_dir_params_bag import MakeDirParamsBag
from contracts.params.param_bag import ParamBag
from services.file_mapper_service import FileMapperService
from configs.enums.ability_category import AbilityCategory


class MakeDirAbility(Ability[MakeDirParamsBag]):
    """Create a directory at an absolute path. Directories only — a file is
    ``write_file``'s job."""

    #: Action-less tool — the ``""`` key drives the dispatcher's ACTION_REQUIRED
    #: pre-gate to reject a missing/blank ``path`` with ``code=missing-params``
    #: BEFORE run().
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.path,)}

    NAME: ClassVar[str] = "make_dir"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.FILE_OPERATIONS

    # The typed input contract: the dispatch seam builds the bag via
    # MakeDirParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = MakeDirParamsBag

    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("make directory", "create folder", "mkdir")

    def get_summary(self) -> str:
        from abilities.write_file import WriteFileAbility  # noqa: PLC0415

        # At build/introspection time (mp is None) return the bare base text so
        # the search index + SHA map stay machine-independent. On a live request
        # append the placement steer with the resolved path.
        base = (
            "Create a directory at an absolute path. Parent directories are created "
            "automatically, and creating one that already exists is an error. This tool "
            f"creates DIRECTORIES ONLY — use '{WriteFileAbility.NAME}' to create a file."
        )
        if self.mp is None:
            return base
        return (
            base
            + f" When creating new directories for documents, ALWAYS create under `{FileMapperService.get_documents_path()}` unless explicitly specified by the user."
        )

    def get_examples(self) -> list[str]:
        return [
            "create a new directory at /workspace/output",
            "make a new folder called /workspace/logs",
            "create the directory /workspace/scripts/deploy",
            "make a new directory for the project",
            "create a folder to keep the exported reports in",
            "set up a directory structure for this project",
        ]

    def get_search_tooltip(self) -> str:
        return "Create a directory"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.path: {
                "type": "string",
                "description": (
                    "Absolute path of the directory to create. Parent "
                    "directories are created automatically."
                ),
            },
        },
        "required": [Keys.path],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: MakeDirParamsBag) -> ToolResult:
        target = Paths.absolute_target(params.path)
        if isinstance(target, ToolResult):
            return target

        try:
            # ``exist_ok=False`` is the existence check: mkdir raises
            # FileExistsError for an existing directory AND for an existing
            # file, so a separate exists() probe would only add a race window.
            target.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            return ToolResult.err(
                f"Path already exists: {target}",
                code="already-exists",
                hint="choose a different path, or delete the existing one first.",
            )
        except PermissionError as exc:
            return ToolResult.err(
                f"Permission denied creating {target}: {exc}",
                code="permission-denied",
                hint=f"you do not have write access to {target.parent}.",
            )
        except OSError as exc:
            # NotADirectoryError lands here: a path component is a file, so no
            # directory can be created below it.
            return ToolResult.err(
                f"Could not create {target}: {exc}",
                code="invalid-path",
                hint="check the path shape and that its parent can hold a directory.",
            )

        return ToolResult.ok({"path": str(target), "created": True})
