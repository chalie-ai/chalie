# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""EditFileAbility — replace a single occurrence of literal text in one file.

The anchored-edit primitive: discoverable on every channel via ``find_tools``
and additionally pinned on the ``code_agent`` delegate. Replaces whole-file
rewrites for targeted changes — every line outside the replaced span stays
byte-identical.

Single-file, single-occurrence: ``path`` MUST point to an existing file, and
the ``search`` string must match the file content exactly once — zero matches
and ambiguous (2+) matches are both loud errors, never silent no-ops. The
search is literal and whitespace-significant — no regex, no glob, no
directory walk. An empty ``replace`` deletes the matched text.

Act-trail guard: the shared :func:`abilities._read_guard.read_guard` must clear
the edit first — the model's most recent ``read`` of the path has to be newer
than the most recent successful change to it on this turn. A stale anchor
otherwise returns ``not-found``, which the model reads as "retry" rather than
"look again", and it loops. Unlike a whole-file overwrite, a partial read is
fine here: an anchored edit only touches text the model quoted back verbatim.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from abilities._ability import Ability
from abilities._read_guard import read_guard
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.edit_file_params_bag import EditFileParamsBag
from contracts.params.param_bag import ParamBag
from services.file_text_service import FileTextService
from configs.enums.ability_category import AbilityCategory


class EditFileAbility(Ability[EditFileParamsBag]):
    DISCOVERABLE: ClassVar[bool] = True
    NAME: ClassVar[str] = "edit_file"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.FILE_OPERATIONS
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("file edit", "modify file", "replace text")
    ALLOW_EMPTY: ClassVar[tuple[str, ...]] = (Keys.replace_,)

    #: ``search`` and ``replace`` are literal file bytes, not instructions to
    #: interpret: the anchor must match the file character for character, and the
    #: replacement is written as-is. Trimming either would make a whole-line edit
    #: ("DELETE THIS LINE\n" -> "") impossible, break indentation fixes, and turn a
    #: whitespace-only replacement into a deletion. ``path`` is still scrubbed.
    VERBATIM: ClassVar[tuple[str, ...]] = (Keys.search, Keys.replace_)

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.search, Keys.replace_, Keys.path)}

    # The typed input contract: the dispatch seam builds the bag via
    # EditFileParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = EditFileParamsBag


    def get_summary(self) -> str:
        from abilities.read import ReadAbility  # noqa: PLC0415

        return (
            "Replace a single occurrence of literal text in a single file. "
            f"You MUST call the '{ReadAbility.NAME}' tool on the target path "
            "before editing, and again after any edit of your own, so `search` "
            "matches the file as it stands now. "
            "Point ``path`` at an absolute file path, ``search`` for the exact "
            "literal text to find (whitespace-significant), and ``replace`` "
            "for the replacement text. The search string must appear exactly "
            "once in the file — if it appears zero times, you get a not-found "
            "error; if it appears more than once, you must include more "
            "surrounding context to make it unique. Every line outside the "
            "replaced span is left byte-identical."
        )

    def get_examples(self) -> list[str]:
        return [
            "replace the API URL in this config file",
            "update the import path in this single file",
            "swap the function name in this source file",
            "change the version string in this package file",
            "fix the typo in this documentation file",
            "update the environment variable name in this file",
            "replace the deprecated method call in this file",
            "change the configuration key in this settings file",
        ]

    def get_search_tooltip(self) -> str:
        return "Find and replace text in file"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.search: {
                "type": "string",
                "description": "The exact literal text to find. Whitespace-significant; must appear exactly once in the file.",
            },
            Keys.replace_: {
                "type": "string",
                "description": "The replacement text. Whitespace-significant. An empty string deletes the matched text.",
            },
            Keys.path: {
                "type": "string",
                "description": "Absolute path to the single file to edit. Required.",
            },
        },
        "required": [Keys.search, Keys.replace_, Keys.path],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: EditFileParamsBag) -> ToolResult:
        if not Path(params.path).is_absolute():
            return ToolResult.err(
                f"path must be an absolute path, got {params.path!r}.",
                code="invalid-path",
            )

        target = Path(params.path)
        if not target.exists():
            from abilities.read import ReadAbility  # noqa: PLC0415

            return ToolResult.err(
                f"{params.path} does not exist.",
                code="not-found",
                hint=f"check the path with the {ReadAbility.NAME} tool.",
            )

        if target.is_dir():
            return ToolResult.err(
                f"{params.path} is a directory, not a file.",
                code="invalid-path",
                hint="point path at a single file.",
            )

        # A blind edit anchors on text the model never saw — or saw before its own
        # earlier edit moved it — and comes back not-found, which reads to the
        # model as "try again" rather than "look again". Refuse once here instead
        # of letting it retry into the runaway backstop. ``partial-read`` is NOT
        # opted into: an anchored edit only touches text the model quoted back.
        refusal = read_guard(self.mp, target.resolve(), refuse_partial=False)
        if refusal is not None:
            return refusal

        # Read with NO newline translation (the default open() would fold every
        # \r\n to \n on the way in — the line-ending rewrite bug this closes).
        # Strict
        # UTF-8: an undecodable file is refused with a clean error, never a
        # stack trace — and the `read` tool decodes leniently (errors='replace'),
        # so the model CAN see such a file and must not be able to edit it.
        try:
            raw = FileTextService.read_raw(target)
        except UnicodeDecodeError:
            return ToolResult.err(
                f"{params.path} is not valid UTF-8 and cannot be edited.",
                code="decode-error",
                hint="re-save the file as UTF-8 first; an anchored edit must not rewrite bytes it cannot decode.",
            )

        ending = FileTextService.detect_ending(raw)
        if ending == "mixed":
            # A file that mixes \r\n and \n cannot be restored byte-identical
            # for every untouched line under ANY single choice of ending —
            # normalizing it here would silently rewrite the lines carrying the
            # other convention. Refuse loudly and leave the file alone.
            return ToolResult.err(
                f"{params.path} mixes \\r\\n and \\n line endings and cannot be edited without rewriting unrelated lines.",
                code="mixed-line-endings",
                hint="normalize the file to a single line ending, re-read it, then edit it.",
            )

        # Match on the LF-normalized form — the same text the `read` tool's
        # display showed the model — so an anchor quoted from `read` matches.
        # The params get the SAME fold: `read` never shows the model a \r, so
        # any \r\n it sends in search/replace is intent-for-a-newline (typically
        # because it believes the file is CRLF), and folding it here keeps
        # `restore` from doubling it to \r\r\n on write.
        content = FileTextService.normalize(raw)
        search = FileTextService.normalize(params.search)
        replace = FileTextService.normalize(params.replace_)
        count = content.count(search)

        if count == 0:
            return ToolResult.err(
                "The search text was not found in the file.",
                code="not-found",
                hint="re-read the file — `search` must match the current content exactly, including whitespace.",
            )

        if count > 1:
            return ToolResult.err(
                "You can only replace 1 occurrence at a time, include more of the surrounding text to make the `search` unique.",
                code="not-unique",
                hint=f"found {count} occurrences; include more context to make it unique.",
            )

        # Restore the file's original ending (CRLF files get their \r\n back)
        # and write with NO translation, so every line outside the replaced
        # span stays byte-identical.
        new_content = content.replace(search, replace, 1)
        FileTextService.write_raw(target, FileTextService.restore(new_content, ending))

        return ToolResult.ok(
            f"Replaced 1 occurrence in {params.path}.",
            files_changed=1,
            total_replacements=1,
        )
