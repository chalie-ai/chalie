"""ReadAbility — thin wrapper around :class:`services.text_reader.TextReader`.

Reads local files only — URLs are rejected with a redirect to ``web_fetch``,
the single URL-owning tool. Extraction and the system-path guard live in
``services/text_reader.py``; this file maps its raises to stable kebab-case
``code``\\ s and gates oversized content: no window supplied means a hard
20 000-character cap (a loud ``too-large`` error, never a silent clip), and a
``start_line``/``end_line`` window returns exactly those lines with
``partial=True`` meta when the window is smaller than the file.
"""

from __future__ import annotations

from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.ability_category import AbilityCategory
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from contracts.params.read_params_bag import ReadParamsBag
from exceptions import (
    NoTextContent,
    NotAFile,
    SourceIsImage,
    SystemPathBlocked,
)
from services.text_reader import TextReader


class ReadAbility(Ability[ReadParamsBag]):
    PARAMS: ClassVar[type[ParamBag] | None] = ReadParamsBag
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("read file",)
    NAME: ClassVar[str] = "read"
    RETURNS_UNTRUSTED_CONTENT: ClassVar[bool] = True  # file contents — including whatever web_fetch just saved
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.FILE_OPERATIONS

    def get_summary(self) -> str:
        return (
            "Read text from a local file — PDFs, DOCX, PPTX, Markdown, or plain text."
        )

    def get_examples(self) -> list[str]:
        return [
            "read my PDF at /home/user/report.pdf",
            "extract the text from the document at /tmp/readme.md",
            "show me the contents of src/main.py",
            "load /tmp/output.txt into context",
            "display the first few lines of /usr/share/dict/words",
            "read the config at /tmp/config.yaml",
            "give me a summary of /opt/docs/changelog.txt",
            "read the CSV file at /tmp/data.csv",
        ]

    def get_search_tooltip(self) -> str:
        return "Read contents of a local file"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.source: {
                "type": "string",
                "description": (
                    "Filesystem path to read, e.g. '/home/user/doc.pdf'. "
                    "The alias 'path' is also accepted."
                ),
            },
            Keys.start_line: {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "First line to read, counting from 1. "
                    "Leave out to start at the beginning of the file."
                ),
            },
            Keys.end_line: {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Last line to read; this line is included in the result. "
                    "Leave out to read to the end of the file."
                ),
            },
        },
        "required": [Keys.source],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    _MAX_RETURN_CHARS: ClassVar[int] = 20_000

    def run(self, params: ReadParamsBag) -> ToolResult:
        source = params.source

        if TextReader.is_url(source):
            from abilities.web_fetch import WebFetchAbility  # noqa: PLC0415

            return ToolResult.err(
                f"This tool reads local files only — URLs are fetched with the `{WebFetchAbility.NAME}` tool.",
                code="url-not-supported",
                hint=f"use the {WebFetchAbility.NAME} tool to fetch and persist the URL, then read the saved file.",
                source=source,
            )

        try:
            text = TextReader(source).get_value()
        except SystemPathBlocked as e:
            return ToolResult.err(str(e), code="system-path-blocked", source=source)
        except FileNotFoundError as e:
            return ToolResult.err(
                str(e),
                code="file-not-found",
                hint="check the path is correct",
                source=source,
            )
        except NotAFile as e:
            return ToolResult.err(str(e), code="not-a-file", source=source)
        except PermissionError as e:
            return ToolResult.err(str(e), code="no-read-permission", source=source)
        except SourceIsImage as e:
            from abilities.vision import VisionAbility  # noqa: PLC0415

            return ToolResult.err(
                str(e),
                code="not-text",
                hint=f"use the {VisionAbility.NAME} tool to see it",
                source=source,
            )
        except NoTextContent as e:
            return ToolResult.err(str(e), code="no-text-content", source=source)

        return self._filter_text(text, params, source=source)

    # ── text gating -----------------------------------------------------

    def _filter_text(
        self,
        text: str,
        params: ReadParamsBag,
        *,
        source: str,
    ) -> ToolResult:
        lines = text.splitlines(keepends=True)
        total = len(lines)

        if params.start_line is None and params.end_line is None:
            if len(text) <= self._MAX_RETURN_CHARS:
                return ToolResult.ok(text, source=source)
            return ToolResult.err(
                (
                    f"File is too large to load into context, select chunks of the "
                    f"document by supplying `start_line`/`end_line`. The file has "
                    f"{total} lines."
                ),
                code="too-large",
                total_lines=total,
                source=source,
            )

        # Window mode
        start = params.start_line or 1
        end = params.end_line or total
        if start > total:
            return ToolResult.err(
                f"Line number {start} exceeds file length ({total} lines).",
                code="line-out-of-range",
                hint=f"The file has {total} lines.",
                source=source,
            )
        if end > total:
            end = total
        chunk = "".join(lines[start - 1 : end])
        if len(chunk) > self._MAX_RETURN_CHARS:
            return ToolResult.err(
                (
                    f"Selected range is over the {self._MAX_RETURN_CHARS}-character "
                    f"limit. Please select a narrower range."
                ),
                code="too-large",
                total_lines=total,
                source=source,
            )
        window = f"{start}-{end}"
        if start > 1 or end < total:
            return ToolResult.ok(chunk, source=source, lines=window, total_lines=total, partial=True)
        return ToolResult.ok(chunk, source=source, lines=window, total_lines=total)
