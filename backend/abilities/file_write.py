"""FileWriteAbility — write content to a file at a caller-supplied absolute path.

Act-trail guard: if the target file already exists, a prior ``read`` call on
the same resolved path in the current turn is required before the write is
executed — so an overwrite is never blind. The guard keys on the turn (every
tool call the turn recorded), not on the single input row, so a ``read`` issued
on any assistant step of the turn satisfies it.

Returns a sealed :class:`abilities._result.ToolResult` (never a wire envelope):

* Success → ``{"path": <abs>, "bytes": <int>, "created": <bool>}`` with
  ``created`` True when the file did not exist before the write. An empty
  ``contents`` is VALID user data: a 0-byte file is written and echoed with
  ``bytes: 0`` (touch / .gitkeep / truncate).
* Bad inputs → ``err()`` with a stable kebab ``code`` (``missing-params`` /
  ``invalid-param`` / ``invalid-path`` / ``permission-denied`` /
  ``read-required``) — errors never masquerade as success.
"""

import json
import logging
from pathlib import Path
from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._params import Keys
from abilities._result import ToolResult

logger = logging.getLogger(__name__)


class FileWriteAbility(Ability):
    #: Action-less tool — the ``""`` key drives the dispatcher's ACTION_REQUIRED
    #: pre-gate to reject a missing/blank ``path`` with ``code=missing-params``
    #: BEFORE run(). ``contents`` is deliberately NOT listed: the pre-gate check
    #: is truthiness-based, so an empty-string ``contents`` (valid user data)
    #: would be falsely rejected there. run() guards a MISSING contents key.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.path,)}

    def get_name(self) -> str:
        return "file_write"

    def get_summary(self) -> str:
        return "Write content to a file. You MUST call the 'read' tool on the target path before writing."

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
        return "File writing and creation"

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

    def run(self, params: dict[str, object]) -> ToolResult:
        path_str = cast(str, params.get(Keys.path, ""))

        # 'contents' is NOT pre-gated (an empty string is valid), so a MISSING
        # key is guarded here; a present "" proceeds to write a 0-byte file.
        if Keys.contents not in params:
            return ToolResult.err(
                "contents is required.",
                code="missing-params",
                hint="pass a 'contents' string (an empty string writes a 0-byte file).",
            )
        contents = params[Keys.contents]
        if not isinstance(contents, str):
            return ToolResult.err(
                f"contents must be a string, got {type(contents).__name__}.",
                code="invalid-param",
                hint="pass the file body as a string; serialise structured data yourself first.",
            )

        if not Path(path_str).is_absolute():
            return ToolResult.err(
                f"Path is not absolute: {path_str!r}.",
                code="invalid-path",
                hint="pass an absolute path (one starting from the filesystem root).",
            )

        target = Path(path_str).resolve()

        existed = target.exists()
        if existed and not self._read_called_first(target):
            return ToolResult.err(
                f"You must read {target} before overwriting it.",
                code="read-required",
                hint=f"call the 'read' tool on {target} first, then retry the write",
            )

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

    def _read_called_first(self, target: Path) -> bool:
        proc = self.mp
        if proc is None:
            return True

        turn_id = getattr(proc, "turn_id", None)
        channel = getattr(getattr(proc, "config", None), "channel", None)
        if turn_id is None or channel is None:
            logger.warning("file_write read-guard: active processor has no turn — guard bypassed")
            return True

        from services.act_trail import ActTrail
        target_str = str(target)
        for row in ActTrail().fetch_by_turn(channel, turn_id):
            if row.get("tool_name") != "read":
                continue
            try:
                p = json.loads(cast(str, row["params"]))
            except (json.JSONDecodeError, TypeError):
                continue
            source = p.get("source") or p.get("path") or p.get("url", "")
            if not source:
                continue
            # `read` expands ``~`` before reading, but records the raw arg — so the
            # guard must expanduser too, or a tilde-path read would never match.
            try:
                if Path(source).expanduser().resolve() == target:
                    return True
            except (ValueError, OSError):
                if source == target_str:
                    return True

        return False
