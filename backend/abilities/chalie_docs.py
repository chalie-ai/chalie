"""ChalieDocsAbility — Chalie's self-reference, fetched server-side.

Resolves a fixed query (``basics`` / ``tools`` / ``releases`` / ``code-base``) to
its documentation URL(s), reads each one through :class:`services.text_reader.TextReader`
— the same path the ``read`` tool takes, prose extraction included —
and returns the documentation text directly.

A url that is refused, fails, or holds no readable prose is collected as a failed
url rather than sinking the call: a PARTIAL outage still returns the pages that
came back.

Previously, the ability answered with an INSTRUCTION ("use the read tool and
visit …") — a round-trip a weak model fumbles. It now does the fetch itself and
returns the doc prose, so one call yields the answer.

Result contract: every return is a :class:`abilities._result.ToolResult` built
only via ``ok()`` / ``err()``; the dispatcher renders the wire envelope. Success
bodies are the extracted documentation (prose) with the source url(s) + the
``version`` + truncation in the meta; an unknown query is ``code=doc-not-found``
with a ``valid=`` ladder and a closest-match hint, and a total fetch outage is
``code=fetch-failed`` (never the banned ``code=error`` placeholder).
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from typing import TypedDict

    class _DocsMeta(TypedDict, total=False):
        source: str
        version: str
        truncated: bool
        failed_sources: str

import requests

from abilities._ability import Ability
from abilities._result import ToolResult, truncate
from configs.enums.param_key import Keys
from contracts.params.chalie_docs_params_bag import ChalieDocsParamsBag
from contracts.params.param_bag import ParamBag
from services.app_version import get_version
from services.text_reader import TextReader
from exceptions import NoReadableContent
from configs.enums.ability_category import AbilityCategory

#: Documentation cap — a chalie.ai page (or the joined basics pages) is returned
#: as prose, clipped here so a multi-MB body never floods the context unbounded.
_MAX_CHARS = 20000

_QUERY_URLS: dict[str, list[str]] = {
    "basics": [
        "https://chalie.ai/guide/getting-started/",
        "https://chalie.ai/how-it-works/",
    ],
    "tools": [
        "https://chalie.ai/abilities/",
    ],
    "releases": [
        "https://chalie.ai/releases/",
    ],
    "code-base": [
        "https://github.com/chalie-ai/chalie",
    ],
}


class ChalieDocsAbility(Ability[ChalieDocsParamsBag]):
    #: Action-less tool: the canonical ``query`` is the one required input. The
    #: dispatcher's ACTION_REQUIRED pre-gate rejects a call with no query as
    #: ``code=missing-params`` BEFORE run() (and before the policy gate).
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.query,)}
    NAME: ClassVar[str] = "chalie_docs"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.INFORMATION

    # The typed input contract: the dispatch seam builds the bag via
    # ChalieDocsParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = ChalieDocsParamsBag

    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("chalie documentation", "own documentation", "harness documentation", "self reference", "about chalie")


    def get_summary(self) -> str:
        return "Look up Chalie's own documentation — what it is, its tools, release history, or codebase."

    def get_examples(self) -> list[str]:
        return [
            "what is chalie",
            "how does chalie work",
            "what tools does chalie have",
            "show me the latest chalie release notes",
            "where is the chalie source code",
            "tell me about chalie's capabilities",
        ]

    def get_search_tooltip(self) -> str:
        return "chalie documentation and self-reference"

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.query: {
                "type": "string",
                "enum": ["basics", "tools", "releases", "code-base"],
                "description": (
                    "basics — what Chalie is and how it works. "
                    "tools — available tools and capabilities. "
                    "releases — version history and changelogs. "
                    "code-base — source code repository."
                ),
            },
        },
        "required": [Keys.query],
    }

    def run(self, params: ChalieDocsParamsBag) -> ToolResult:
        query = params.query.strip().lower()
        urls = _QUERY_URLS.get(query)
        if not urls:
            # Unknown query — a STABLE, routable error (never the banned
            # code=error). Surface a closest-match suggestion so a near-miss like
            # 'toolz' is nudged to 'tools' without re-reading the schema.
            close = difflib.get_close_matches(query, _QUERY_URLS, n=1)
            hint = f"choose one of: {', '.join(_QUERY_URLS)}."
            if close:
                hint = f"did you mean '{close[0]}'? {hint}"
            return ToolResult.err(
                f"Unknown documentation query '{query}'.",
                code="doc-not-found",
                hint=hint,
                valid=tuple(_QUERY_URLS),
            )

        # Fetch each url; collect the extracted prose, noting which urls failed so
        # a PARTIAL outage still returns the pages that came back (rather than
        # failing the whole call) with the failed url named in the meta.
        sections: list[str] = []
        failed: list[str] = []
        for url in urls:
            try:
                text = TextReader(url).get_value()
            except (NoReadableContent, requests.RequestException):
                # NoReadableContent covers a page that fetched but held no prose —
                # the same "this url gave us nothing" outcome as an outright
                # failure, so it lands in the same bucket.
                failed.append(url)
                continue
            sections.append(f"=== {url} ===\n\n{text.strip()}")

        if not sections:
            # Every url was unreachable / yielded no readable content — a loud,
            # routable failure, NOT the old 'use the read tool and visit …' prose.
            return ToolResult.err(
                f"Could not fetch the Chalie documentation for '{query}'.",
                code="fetch-failed",
                hint="the documentation site may be unreachable — try again shortly",
                source=" & ".join(urls),
            )

        body = "\n\n".join(sections)
        clipped, was_clipped = truncate(body, _MAX_CHARS)
        meta: "_DocsMeta" = {
            "source": " & ".join(urls),
            "version": get_version(),
        }
        if was_clipped:
            meta["truncated"] = True
        if failed:
            meta["failed_sources"] = " & ".join(failed)
        return ToolResult.ok(clipped, **meta)
