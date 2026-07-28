"""FileWriteAbility — write content to a file at a caller-supplied absolute path.

Act-trail guard: if the target file already exists, a prior ``read`` call on
the same resolved path in the current turn is required before the write is
executed — so an overwrite is never blind. The guard keys on the turn (every
tool call the turn recorded), not on the single input row, so a ``read`` issued
on any assistant step of the turn satisfies it. If the most recent matching
read was truncated the write is refused too — a full overwrite would drop the
tail the model never saw.

Returns a sealed :class:`abilities._result.ToolResult` (never a wire envelope):

* Success → ``{"path": <abs>, "bytes": <int>, "created": <bool>}`` with
  ``created`` True when the file did not exist before the write. An empty
  ``contents`` is VALID user data: a 0-byte file is written and echoed with
  ``bytes: 0`` (touch / .gitkeep / truncate).
* Bad inputs → ``err()`` with a stable kebab ``code`` (``missing-params`` /
  ``invalid-param`` / ``invalid-path`` / ``permission-denied`` /
  ``read-required`` / ``truncated-read``) — errors never masquerade as success.
"""

import json
import logging
from pathlib import Path
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.file_write_params_bag import FileWriteParamsBag
from contracts.params.param_bag import ParamBag
from models.tool_call import ToolCall
from services.file_mapper_service import FileMapperService
from configs.enums.ability_category import AbilityCategory

logger = logging.getLogger(__name__)


class FileWriteAbility(Ability[FileWriteParamsBag]):
    #: Action-less tool — the ``""`` key drives the dispatcher's ACTION_REQUIRED
    #: pre-gate to reject a missing/blank ``path`` with ``code=missing-params``
    #: BEFORE run(). ``contents`` is deliberately NOT listed: the pre-gate check
    #: is truthiness-based, so an empty-string ``contents`` (valid user data)
    #: would be falsely rejected there. The bag's from_params guards a MISSING
    #: contents key by presence.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.path,)}

    #: ``contents`` is the file body, written byte for byte — the leading indent
    #: and the trailing newline are the model's choice, not noise to tidy away.
    #: ``path`` is still scrubbed.
    VERBATIM: ClassVar[tuple[str, ...]] = (Keys.contents,)

    NAME: ClassVar[str] = "file_write"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.FILE_OPERATIONS

    # The typed input contract: the dispatch seam builds the bag via
    # FileWriteParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = FileWriteParamsBag

    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("write file", "create file", "save file")


    def get_summary(self) -> str:
        from abilities.read import ReadAbility  # noqa: PLC0415

        # At build/introspection time (mp is None) return the bare base text so
        # the search index + SHA map stay machine-independent. On a live request
        # append the docs-placement steer with the resolved path.
        base = f"Write content to a file. You MUST call the '{ReadAbility.NAME}' tool on the target path before writing."
        if self.mp is None:
            return base
        return (
            base
            + f" When creating new documents, ALWAYS create under `{FileMapperService.get_documents_path()}` unless explicitly specified by the user."
        )

    def get_examples(self) -> list[str]:
        return [
            "save this text to a file",
            "write this configuration to /etc/myapp/config.yaml",
            "create a new script file",
            "save the output to a temporary file",
            "write this JSON to a file so I can use it later",
            "overwrite the contents of that file",
        ]

    def get_search_tooltip(self) -> str:
        return "Create or Save file"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.path: {
                "type": "string",
                "description": "Absolute path to write to.",
            },
            Keys.contents: {
                "type": "string",
                "description": (
                    "Content to write to the file. Pass an empty string to "
                    "create a 0-byte file or truncate an existing one."
                ),
            },
        },
        "required": [Keys.path, Keys.contents],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: FileWriteParamsBag) -> ToolResult:
        path_str = params.path
        contents = params.contents

        if not Path(path_str).is_absolute():
            return ToolResult.err(
                f"Path is not absolute: {path_str!r}.",
                code="invalid-path",
                hint="pass an absolute path (one starting from the filesystem root).",
            )

        target = Path(path_str).resolve()

        existed = target.exists()
        if existed:
            refusal = self._overwrite_guard(target)
            if refusal is not None:
                return refusal

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(contents)
            bytes_written = target.stat().st_size
        except PermissionError as exc:
            return ToolResult.err(
                f"Permission denied writing to {target}: {exc}",
                code="permission-denied",
                hint=f"you do not have write access to {target} or its parent directory.",
            )
        except OSError as exc:
            return ToolResult.err(
                f"Could not write to {target}: {exc}",
                code="invalid-path",
                hint="check the path shape and that its parent can hold a file.",
            )

        return ToolResult.ok(
            {"path": str(target), "bytes": bytes_written, "created": not existed}
        )

    def _overwrite_guard(self, target: Path) -> ToolResult | None:
        """The act-trail read-before-write check — the refusal to return, or
        ``None`` when the overwrite may proceed.

        Bypasses (returns ``None``) when there is no live processor or the
        turn has no turn_id / channel — the act-trail cannot be inspected
        there, so the write is allowed as before. Otherwise the MOST RECENT
        ``read`` row on this turn matching ``target`` decides: no row →
        ``read-required``; a row whose read was truncated → ``truncated-read``
        (a full overwrite would destroy the portion the model never saw).
        """
        proc = self.mp
        if proc is None:
            return None
        turn_id = getattr(proc, "turn_id", None)
        channel = getattr(getattr(proc, "config", None), "channel", None)
        if turn_id is None or channel is None:
            logger.warning("file_write read-guard: active processor has no turn — guard bypassed")
            return None

        target_str = str(target)
        last_match: ToolCall | None = None
        for row in ToolCall.by_turn(channel, turn_id):
            if row.tool_name != "read":
                continue
            try:
                p = json.loads(row.params)
            except (json.JSONDecodeError, TypeError):
                continue
            source = p.get("source") or p.get("path") or p.get("url", "")
            if not source:
                continue
            # `read` expands ``~`` before reading, but records the raw arg — so the
            # guard must expanduser too, or a tilde-path read would never match.
            try:
                if Path(source).expanduser().resolve() == target:
                    last_match = row
            except (ValueError, OSError):
                if source == target_str:
                    last_match = row

        if last_match is None:
            from abilities.read import ReadAbility  # noqa: PLC0415

            return ToolResult.err(
                f"You must read {target} before overwriting it.",
                code="read-required",
                hint=f"call the '{ReadAbility.NAME}' tool on {target} first, then retry the write",
            )
        # Truncation is flagged in the envelope OPEN TAG (its first line) only —
        # matching the whole result would false-positive on file content that
        # happens to contain the marker.
        if "truncated=true" in (last_match.result or "").split("\n", 1)[0]:
            from abilities.edit_file import EditFileAbility  # noqa: PLC0415

            return ToolResult.err(
                f"Your last read of {target} was truncated — a full overwrite would destroy the unread portion.",
                code="truncated-read",
                hint=f"use {EditFileAbility.NAME} for targeted changes, or re-read with a higher max_chars to get the full file first.",
            )
        return None
