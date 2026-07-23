"""ManageFilesAbility — single merged file-CRUD tool.

Replaces the old create_file / delete_file / create_folder / file_permissions
abilities. One tool, three actions:

* ``create`` — make a new file (empty) or folder. Auto-creates parent dirs.
  Applies a permission code (defaults to 0644/0755 when omitted). Creating a
  path that already exists is an error.
* ``delete`` — remove a file, or a folder recursively. Missing target is an
  error.
* ``update_permission`` — chmod a file or folder to an octal mode. The
  permission_code param is required for this action.

All errors go through :class:`abilities._result.ToolResult.err` with a stable
kebab ``code`` (``already-exists`` / ``not-found`` / ``invalid-path`` /
``invalid-param`` / ``permission-denied``) — never silent.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.manage_files_params_bag import (
    ManageFilesCreateParams,
    ManageFilesDeleteParams,
    ManageFilesParamsBag,
    ManageFilesUpdatePermissionParams,
)
from contracts.params.param_bag import ParamBag


class ManageFilesAbility(Ability[ManageFilesParamsBag]):
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "create": (Keys.path,),
        "delete": (Keys.path,),
        "update_permission": (Keys.path, Keys.permission_code),
    }

    # The typed input contract: the dispatch seam builds the bag via
    # ManageFilesParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = ManageFilesParamsBag

    def get_name(self) -> str:
        return "manage_files"

    def get_summary(self) -> str:
        return (
            "Create, delete, or change permissions on a file or folder. "
            "Agents SHOULD create and modify files within the designated workspace UNLESS otherwise stated by the user. "
            "Use this for creating new files and folders, removing them, or adjusting their Unix permissions."
        )

    def get_examples(self) -> list[str]:
        return [
            "create a new empty file at /workspace/output.txt",
            "make a new folder called /workspace/logs",
            "remove the file /workspace/old_report.csv",
            "delete the entire /workspace/temp directory and everything in it",
            "make /workspace/scripts/deploy.sh executable with 0755",
            "set permissions on /workspace/config.ini to 0644",
            "create a new directory at /workspace/data/raw",
            "delete /workspace/backup.tar.gz",
        ]

    def get_search_tooltip(self) -> str:
        return "Create, delete, or change permissions on files and folders"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": ["create", "delete", "update_permission"],
                "description": (
                    "'create' = make a new file or folder (empty file, or directory). "
                    "'delete' = remove a file or folder (recursively for folders). "
                    "'update_permission' = change Unix permissions (chmod) on a file or folder."
                ),
            },
            Keys.path: {
                "type": "string",
                "description": (
                    "Absolute path to the file or folder. For 'create': if the path "
                    "ends with '/' it creates a folder, otherwise an empty file. "
                    "Parent directories are created automatically."
                ),
            },
            Keys.permission_code: {
                "type": "string",
                "description": (
                    "Unix octal permission code, e.g. '0644' or '0755'. Required for "
                    "'update_permission'. For 'create', defaults to 0644 for files and "
                    "0755 for folders when not provided."
                ),
            },
        },
        "required": [Keys.action, Keys.path],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: ManageFilesParamsBag) -> ToolResult:
        # The router factory only ever yields the three leaves; a hand-built
        # foreign subclass (or the bare router) is rejected loudly BEFORE the
        # shared path validation reads a field it may not carry.
        if not isinstance(
            params,
            (ManageFilesCreateParams, ManageFilesDeleteParams, ManageFilesUpdatePermissionParams),
        ):
            return ToolResult.err(
                f"Unknown manage_files params bag: {type(params).__name__}.",
                code="unknown-action",
                valid=("create", "delete", "update_permission"),
            )

        # Residue guard — defends directly constructed bags; the dispatch path
        # cannot get here blank (the bag's require_str strips and rejects it).
        path_str = params.path
        if not path_str or not path_str.strip():
            return ToolResult.err(
                "path is required and must not be empty.",
                code="missing-params",
                hint="pass an absolute path.",
            )

        if not Path(path_str).is_absolute():
            return ToolResult.err(
                f"Path is not absolute: {path_str!r}.",
                code="invalid-path",
                hint="pass an absolute path (one starting from the filesystem root).",
            )

        target = Path(path_str).resolve()
        target_str = str(target)

        # Distinguish "create folder" from "create file" BEFORE resolve strips
        # the trailing slash.
        is_folder = path_str.rstrip().endswith("/")

        # The isinstance chain hands each handler its exact leaf type; after
        # the create/delete branches the remaining type IS update_permission.
        if isinstance(params, ManageFilesCreateParams):
            return self._do_create(target, target_str, is_folder, params.permission_code)
        if isinstance(params, ManageFilesDeleteParams):
            return self._do_delete(target, target_str)
        return self._do_update_permission(target, target_str, params.permission_code)

    def _do_create(self, target: Path, target_str: str, is_folder: bool, perm_str: str) -> ToolResult:
        if target.exists():
            return ToolResult.err(
                f"Path already exists: {target_str}",
                code="already-exists",
                hint="choose a different path or delete the existing one first.",
            )

        # Parse permission code (empty string → use default).
        mode = self._parse_octal(perm_str) if perm_str else None
        if mode is None and perm_str:
            return ToolResult.err(
                f"Invalid permission code: {perm_str!r}.",
                code="invalid-param",
                hint="pass a 3- or 4-digit octal string like '0644' or '0755'.",
            )
        if mode is None:
            mode = 0o755 if is_folder else 0o644

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if is_folder:
                target.mkdir(mode=mode, exist_ok=False)
            else:
                fd = os.open(target_str, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
                os.close(fd)
            # os.open/mkdir modes are filtered by the process umask; chmod
            # afterwards so the requested code is honoured exactly.
            os.chmod(target, mode)
        except FileExistsError:
            return ToolResult.err(
                f"Path already exists: {target_str}",
                code="already-exists",
                hint="choose a different path or delete the existing one first.",
            )
        except PermissionError as exc:
            return ToolResult.err(
                f"Permission denied creating {target_str}: {exc}",
                code="permission-denied",
                hint=f"you do not have write access to {target.parent}.",
            )
        except OSError as exc:
            return ToolResult.err(
                f"Could not create {target_str}: {exc}",
                code="invalid-path",
                hint="check the path shape and that its parent can hold a file or folder.",
            )

        return ToolResult.ok(
            {"path": target_str, "created": True, "is_folder": is_folder},
            mode=self._format_octal(mode),
        )

    def _do_delete(self, target: Path, target_str: str) -> ToolResult:
        if not target.exists():
            return ToolResult.err(
                f"No such file or directory: {target_str}",
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
                f"Permission denied deleting {target_str}: {exc}",
                code="permission-denied",
                hint=f"you do not own {target_str} or cannot remove it.",
            )
        except OSError as exc:
            return ToolResult.err(
                f"Could not delete {target_str}: {exc}",
                code="not-found",
                hint="the path may have been removed by another process.",
            )

        return ToolResult.ok({"path": target_str, "deleted": True})

    def _do_update_permission(self, target: Path, target_str: str, perm_str: str) -> ToolResult:
        if not target.exists():
            return ToolResult.err(
                f"No such file or directory: {target_str}",
                code="not-found",
                hint="pass an absolute path to an existing file or directory.",
            )

        mode = self._parse_octal(perm_str)
        if mode is None:
            return ToolResult.err(
                f"Invalid permission code: {perm_str!r}.",
                code="invalid-param",
                hint="pass a 3- or 4-digit octal string like '0644' or '0755'.",
            )

        try:
            before = target.stat().st_mode & 0o7777
        except PermissionError as exc:
            return ToolResult.err(
                f"Permission denied reading {target_str}: {exc}",
                code="permission-denied",
                hint=f"you do not own {target_str} or cannot stat it.",
            )

        try:
            os.chmod(target, mode)
            after = target.stat().st_mode & 0o7777
        except PermissionError as exc:
            return ToolResult.err(
                f"Permission denied changing {target_str}: {exc}",
                code="permission-denied",
                hint=f"you must own {target_str} to change its permissions.",
            )
        except OSError as exc:
            return ToolResult.err(
                f"Could not change permissions on {target_str}: {exc}",
                code="invalid-param",
                hint="check the path exists and is a regular file or directory.",
            )

        return ToolResult.ok(
            {"path": target_str, "mode_octal": self._format_octal(after)},
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
