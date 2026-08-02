"""ReadAbility — Thin wrapper around :class:`services.text_reader.TextReader`.

The fetch/extract/guard logic now lives in ``services/text_reader.py``; this
file maps its raises to stable tool codes and gates oversized content through
either a hard 20000-character cap (no window supplied) or a 1-indexed line
window (``start_line`` / ``end_line``) when the caller supplies one. Content
over the 20k limit is never silently clipped — the model gets a loud error
telling it to select a smaller range.

Security:
  - SSRF guard: a SINGLE gate in ``services.web_fetch`` blocks requests to
    private/internal IP ranges (resolved, not string-matched) before any socket
    opens. :class:`FetchBlocked` propagates untouched from the service and is
    mapped here to ``code=private-or-internal-url-blocked``.
  - File guard: reads from system paths (/etc, /proc, /dev, /sys, /var/run)
    raise :class:`SystemPathBlocked` in the service and are mapped to
    ``code=system-path-blocked`` here.

Result contract: every return is a :class:`abilities._result.ToolResult` built
only via ``ok()`` / ``err()``; the dispatcher renders the wire envelope. Errors
carry a stable kebab-case ``code`` (never the ``code="error"`` placeholder) so a
weak model can self-correct without re-reading the schema.

Passthrough: a response whose content type is ``text/*`` (but not ``text/html``),
or a URL/file whose path ends with a known plain-text extension (``.diff``,
``.patch``, ``.txt`` …), skips HTML extraction — raw patches and diffs come back
verbatim instead of being stripped to ``no-readable-content``.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import requests

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.ability_category import AbilityCategory
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from contracts.params.read_params_bag import ReadParamsBag
from exceptions import (
    FetchBlocked,
    NoReadableContent,
    NoTextContent,
    NotAFile,
    SourceIsImage,
    SystemPathBlocked,
)
from services.text_reader import TextReader

logger = logging.getLogger(__name__)


class ReadAbility(Ability[ReadParamsBag]):
    PARAMS: ClassVar[type[ParamBag] | None] = ReadParamsBag
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = (
        "fetch url",
        "read file",
        "open url",
        "read page",
        "fetch",
    )
    NAME: ClassVar[str] = "read"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.FILE_OPERATIONS

    def get_summary(self) -> str:
        return (
            "Fetch and extract clean text from any URL or local file — web pages, "
            "PDFs, DOCX, PPTX, and plain text."
        )

    def get_examples(self) -> list[str]:
        return [
            "can you read this page and tell me what it says? https://example.com",
            "summarise the article at this link",
            "fetch the content of https://bbc.com/news/science",
            "read my PDF at /home/user/report.pdf",
            "what does that URL say",
            "open this link and give me a summary",
            "read the documentation page at https://docs.python.org/3/library/json.html",
            "extract the text from this document",
        ]

    def get_search_tooltip(self) -> str:
        return "Read contents of file or URL"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.source: {
                "type": "string",
                "description": (
                    "URL (e.g. 'https://example.com/article') or filesystem path "
                    "(e.g. '/home/user/doc.pdf'). Aliases 'url' and 'path' are also accepted."
                ),
            },
            Keys.start_line: {
                "type": "integer",
                "minimum": 1,
                "description": "1-indexed first line of the window to return (optional).",
            },
            Keys.end_line: {
                "type": "integer",
                "minimum": 1,
                "description": "1-indexed last line of the window, inclusive (optional).",
            },
        },
        "required": [Keys.source],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    #: Input validation lives in :class:`ReadParamsBag`, constructed by the
    #: dispatcher before this runs — a missing ``source`` never reaches here.
    #: (The ``url`` / ``path`` / ``link`` … aliases a model naturally emits are
    #: healed to ``source`` even earlier, at the dispatch seam, via the shared
    #: ``configs.enums.param_key.VARIANTS[Keys.source]`` ladder.)

    _MAX_RETURN_CHARS: ClassVar[int] = 20_000

    def run(self, params: ReadParamsBag) -> ToolResult:
        source = params.source

        try:
            text = TextReader(source).get_value()
        except FetchBlocked:
            # The single SSRF gate in web_fetch refused this host.
            return ToolResult.err(
                "This URL resolves to a private or internal address and was blocked.",
                code="private-or-internal-url-blocked",
                source=source,
            )
        except requests.RequestException as e:
            return ToolResult.err(
                f"Could not fetch the URL: {str(e)[:150]}",
                code="fetch-failed",
                hint="check the URL is reachable and try again",
                source=source,
            )
        except NoReadableContent as e:
            return ToolResult.err(str(e), code="no-readable-content", source=source)
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
