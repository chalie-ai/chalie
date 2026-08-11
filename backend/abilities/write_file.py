"""WriteFileAbility — write content to a file at a caller-supplied absolute path.

Act-trail guard: if the target file already exists, the shared
:func:`abilities._read_guard.read_guard` must clear the overwrite first — the
model's most recent ``read`` of the path has to be newer than the most recent
successful change to it on this turn, so an overwrite is never blind. This tool
opts into the extra ``partial-read`` refusal: a full overwrite would drop the
lines a windowed (``start_line``/``end_line``) read never showed the model.

Returns a sealed :class:`abilities._result.ToolResult` (never a wire envelope):

* Success → ``{"path": <abs>, "bytes": <int>, "created": <bool>}`` with
  ``created`` True when the file did not exist before the write. An empty
  ``contents`` is VALID user data: a 0-byte file is written and echoed with
  ``bytes: 0`` (touch / .gitkeep / truncate).
* Bad inputs → ``err()`` with a stable kebab ``code`` (``missing-params`` /
  ``invalid-param`` / ``invalid-path`` / ``permission-denied`` /
  ``read-required`` / ``partial-read``) — errors never masquerade as success.
"""

from typing import ClassVar

from abilities._ability import Ability
from abilities._paths import absolute_target
from abilities._read_guard import read_guard
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.write_file_params_bag import WriteFileParamsBag
from contracts.params.param_bag import ParamBag
from services.file_mapper_service import FileMapperService
from configs.enums.ability_category import AbilityCategory


class WriteFileAbility(Ability[WriteFileParamsBag]):
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

    NAME: ClassVar[str] = "write_file"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.FILE_OPERATIONS

    # The typed input contract: the dispatch seam builds the bag via
    # WriteFileParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = WriteFileParamsBag

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

    def run(self, params: WriteFileParamsBag) -> ToolResult:
        path_str = params.path
        contents = params.contents

        target = absolute_target(path_str)
        if isinstance(target, ToolResult):
            return target

        existed = target.exists()
        if existed:
            refusal = read_guard(self.mp, target, refuse_partial=True)
            if refusal is not None:
                return refusal

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(contents)
            bytes_written = target.stat().st_size
        except PermissionError as exc:
            return ToolResult.permission_denied("writing to", target, exc, hint=f"you do not have write access to {target} or its parent directory.")
        except OSError as exc:
            return ToolResult.path_os_error("write to", target, exc, code="invalid-path", hint="check the path shape and that its parent can hold a file.")

        return ToolResult.ok(
            {"path": str(target), "bytes": bytes_written, "created": not existed}
        )
