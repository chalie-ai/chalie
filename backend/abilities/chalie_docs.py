"""ChalieDocsAbility — Chalie's self-reference, fetched server-side.

Resolves a fixed query (``basics`` / ``tools`` / ``releases`` / ``code-base``) to
its documentation URL(s), fetches each page through the shared SSRF-guarded
:mod:`services.web_fetch` stack, extracts the readable prose with
:func:`services.text_extractor.extract_html`, and returns the documentation text
directly.

TKT-901: the old ability answered with an INSTRUCTION ("use the read tool and
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
from typing import ClassVar

import requests

from abilities._ability import Ability
from abilities._params import Keys
from abilities._result import ToolResult, truncate
from services.file_mapper_service import FileMapperService
from services.text_extractor import extract_html
from services.web_fetch import BROWSER, FetchBlocked, fetch_page

_VERSION_FILE = FileMapperService.get_version_path()

#: Documentation cap — a chalie.ai page (or the joined basics pages) is returned
#: as prose, clipped here so a multi-MB body never floods the context unbounded.
_MAX_CHARS = 20000

#: Per-request fetch timeout (seconds) — mirrors ``read``'s URL branch.
_FETCH_TIMEOUT = 15

_QUERY_URLS: dict[str, list[str]] = {
    "basics": [
        "https://chalie.ai/guide/getting-started/",
        "https://chalie.ai/how-it-works/",
    ],
    "tools": [
        "https://chalie.ai/guide/getting-started/",
    ],
    "releases": [
        "https://chalie.ai/releases/",
    ],
    "code-base": [
        "https://github.com/chalie-ai/chalie",
    ],
}


def _read_version() -> str:
    try:
        return _VERSION_FILE.read_text().strip()
    except OSError:
        return "unknown"


def _fetch_doc(url: str) -> str:
    """Fetch *url* through the shared SSRF-guarded stack and extract its prose.

    Returns the extracted text (possibly empty when the page has no readable
    content). Raises :class:`FetchBlocked` / :class:`requests.RequestException`
    on a refused or failed fetch — the caller decides whether a partial outage is
    survivable.
    """
    html, _content_type = fetch_page(url, profile=BROWSER, timeout=_FETCH_TIMEOUT)
    return extract_html(html, url=url)


class ChalieDocsAbility(Ability):
    #: Action-less tool: the canonical ``query`` is the one required input. The
    #: dispatcher's ACTION_REQUIRED pre-gate rejects a call with no query as
    #: ``code=missing-params`` BEFORE run() (and before the policy gate).
    ACTION_REQUIRED: ClassVar[dict] = {"": (Keys.query,)}

    def get_name(self) -> str:
        return "chalie_docs"

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

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    _PARAMETERS: ClassVar[dict] = {
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

    def run(self, params: dict) -> ToolResult:
        raw = self.param(params, Keys.query, required=True)
        query = str(raw).strip().lower()
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
                text = _fetch_doc(url)
            except (FetchBlocked, requests.RequestException):
                failed.append(url)
                continue
            if text and text.strip():
                sections.append(f"=== {url} ===\n\n{text.strip()}")
            else:
                failed.append(url)

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
        meta: dict = {
            "source": " & ".join(urls),
            "version": _read_version(),
        }
        if was_clipped:
            meta["truncated"] = True
        if failed:
            meta["failed_sources"] = " & ".join(failed)
        return ToolResult.ok(clipped, **meta)
