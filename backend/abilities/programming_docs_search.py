"""ProgrammingDocsSearchAbility — official documentation lookup.

Resolves a language/framework name to a documentation source, asks that source
for candidate doc URLs, fetches the top candidate, extracts its readable text,
and returns it. 23 sources share ONE generic engine driven by a declarative
:data:`SOURCES` table — adding a 24th source is a single ``Source(...)`` entry,
never a new class.

Truthfulness contract (TKT-892): this tool NEVER fabricates documentation. A
source that yields no candidate, or whose top candidate cannot be fetched /
extracted, returns ``ToolResult.err(code="source-unavailable")`` — a loud error
the model can route around — not invented page text presented as the answer. A
real-but-imprecise result (e.g. a documentation search-results page instead of
the exact symbol page) succeeds with ``meta degraded=true``.

All HTTP goes through :mod:`services.web_fetch` (one SSRF gate, the ``API``
profile's identified bot UA) and all HTML extraction through
:func:`services.text_extractor.extract_html` — no local fetch or parser plumbing.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, cast

import requests

from abilities._ability import Ability
from abilities._params import Keys
from abilities._result import ToolResult, truncate
from services.text_extractor import extract_html
from services.web_fetch import API, FetchBlocked, fetch_text

# ── Tunables ──────────────────────────────────────────────────────────────────

_MAX_CHARS = 12_000          # excerpt cap for the fetched top candidate
_MAX_RESULTS = 5             # candidate rows returned per lookup
_FETCH_TIMEOUT = 8           # per-request seconds
_RE_HTML_TAGS = re.compile(r"<[^>]+>")


# ── Low-level fetch helpers (all through services.web_fetch) ────────────────────

def _fetch(url: str) -> str:
    return fetch_text(url, profile=API, timeout=_FETCH_TIMEOUT)


def _strip_tags(html: str) -> str:
    return _RE_HTML_TAGS.sub("", html).strip()


def _abs_url(href: str, base: str) -> str:
    if href.startswith("http"):
        return href
    return urllib.parse.urljoin(base, href)


# ── Generic candidate resolvers ─────────────────────────────────────────────────
# A resolver is ``(query) -> list[{"url", "title"}]``. Three factories cover every
# source; a source's table entry picks one and parameterises it. There are NO
# per-source subclasses — only declarative config bound into a generic closure.


def _templates(
    *patterns: str,
    title: str = "{q}",
) -> Callable[[str], list[dict[str, object]]]:
    """Templates are tried cheapest-first WITHOUT a separate HEAD probe — the
    fetch IS the validation (a dead URL falls through to the next candidate or
    to ``source-unavailable``)."""

    def resolve(query: str) -> list[dict[str, object]]:
        vars_ = _query_vars(query)
        out: list[dict[str, object]] = []
        seen: set[str] = set()
        for pat in patterns:
            try:
                url = pat.format(**vars_)
            except (KeyError, IndexError):
                continue
            if url in seen:
                continue
            seen.add(url)
            out.append({"url": url, "title": title.format(**vars_)})
        return out

    return resolve


def _json_api(
    endpoint: str,
    *,
    results_key: str,
    url_field: str,
    title_field: str,
    url_prefix: str = "",
) -> Callable[[str], list[dict[str, object]]]:
    """A request failure bubbles to the engine, which surfaces
    ``source-unavailable`` — it never fabricates a fallback URL."""

    def resolve(query: str) -> list[dict[str, object]]:
        url = endpoint.format(q=urllib.parse.quote(query))
        data = json.loads(_fetch(url))
        out: list[dict[str, object]] = []
        for item in data.get(results_key, []):
            item_url = item.get(url_field, "")
            if not item_url:
                continue
            out.append({"url": f"{url_prefix}{item_url}", "title": item.get(title_field, "")})
            if len(out) >= _MAX_RESULTS:
                break
        return out

    return resolve


def _html_search(
    endpoint: str,
    *,
    link_pattern: str,
    base: str,
    skip_self: bool = True,
) -> Callable[[str], list[dict[str, object]]]:
    """A request failure bubbles to the engine."""
    rx = re.compile(link_pattern, re.DOTALL | re.IGNORECASE)

    def resolve(query: str) -> list[dict[str, object]]:
        url = endpoint.format(q=urllib.parse.quote(query))
        html = _fetch(url)
        out: list[dict[str, object]] = []
        seen: set[str] = set()
        for m in rx.finditer(html):
            href = m.group(1)
            title = _strip_tags(m.group(2))
            if not title:
                continue
            if skip_self and "search" in href.lower():
                continue
            full = _abs_url(href, base)
            if full in seen:
                continue
            seen.add(full)
            out.append({"url": full, "title": title})
            if len(out) >= _MAX_RESULTS:
                break
        return out

    return resolve


def _query_vars(query: str) -> dict[str, str]:
    q = query.strip()
    lower = q.lower()
    last = re.split(r"[.\\/:#]+", q)[-1] or q
    base_seg = re.split(r"[.:#]", q)[0].split("\\")[-1] or q
    return {
        "q": lower,
        "raw": q,
        "dash": lower.replace(" ", "-").replace("_", "-"),
        "us": lower.replace(" ", "_").replace("-", "_"),
        "dot": lower.replace(" ", "."),
        "slash": lower.replace("::", "/").replace(" ", "_"),
        "Title": q[:1].upper() + q[1:] if q else q,
        "last": last,
        "base": base_seg,
    }


# ── The declarative source table ────────────────────────────────────────────────

@dataclass
class Source:
    """``base_url`` is both the human "where these docs live" pointer AND the value
    the engine fetches when a resolver yields no candidate of its own (a real
    landing/search page — never a fabricated symbol page). It is a plain mutable
    field so deployment/test can repoint a source without touching code.
    """

    id: str
    name: str
    aliases: tuple[str, ...]
    base_url: str
    resolve: Callable[[str], list[dict[str, object]]] = field(repr=False)


SOURCES: list[Source] = [
    Source(
        id="php", name="PHP", aliases=("php", "php8"),
        base_url="https://www.php.net/manual/en/",
        resolve=_templates(
            "https://www.php.net/manual/en/function.{dash}.php",
            "https://www.php.net/manual/en/class.{us}.php",
            "https://www.php.net/manual/en/{slash}.php",
        ),
    ),
    Source(
        id="python", name="Python", aliases=("python", "py", "python3"),
        base_url="https://docs.python.org/3/",
        resolve=_templates(
            "https://docs.python.org/3/library/{base}.html",
            "https://docs.python.org/3/library/functions.html",
        ),
    ),
    Source(
        id="javascript", name="JavaScript (MDN)",
        aliases=("javascript", "js", "typescript", "ts", "mdn"),
        base_url="https://developer.mozilla.org/en-US/docs/Web/JavaScript",
        resolve=_json_api(
            "https://developer.mozilla.org/api/v1/search?q={q}&locale=en-US&size=5",
            results_key="documents", url_field="mdn_url", title_field="title",
            url_prefix="https://developer.mozilla.org",
        ),
    ),
    Source(
        id="go", name="Go", aliases=("go", "golang", "gin", "echo", "fiber"),
        base_url="https://pkg.go.dev/",
        resolve=_html_search(
            "https://pkg.go.dev/search?q={q}&m=symbol",
            link_pattern=r'<a\s+href="(/[^"?]+)"[^>]*data-test-id="snippet-title"[^>]*>(.*?)</a>',
            base="https://pkg.go.dev",
        ),
    ),
    Source(
        id="rust", name="Rust", aliases=("rust", "rs", "tokio", "serde", "actix"),
        base_url="https://doc.rust-lang.org/std/",
        resolve=_templates(
            "https://doc.rust-lang.org/std/{slash}/index.html",
            "https://doc.rust-lang.org/std/{slash}/struct.{last}.html",
            "https://doc.rust-lang.org/std/{slash}/enum.{last}.html",
            "https://doc.rust-lang.org/std/{slash}/trait.{last}.html",
            "https://doc.rust-lang.org/std/{slash}/fn.{last}.html",
        ),
    ),
    Source(
        id="java", name="Java", aliases=("java", "jdk"),
        base_url="https://docs.oracle.com/en/java/javase/21/docs/api/",
        resolve=_templates(
            "https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/lang/{last}.html",
            "https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/{last}.html",
            "https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/io/{last}.html",
            "https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/net/{last}.html",
        ),
    ),
    Source(
        id="ruby", name="Ruby", aliases=("ruby", "rb"),
        base_url="https://docs.ruby-lang.org/en/master/",
        resolve=_templates(
            "https://docs.ruby-lang.org/en/master/{base}.html",
            title="{base} — Ruby docs",
        ),
    ),
    Source(
        id="csharp", name="C# / .NET",
        aliases=("csharp", "c#", "cs", "dotnet", "aspnet", "entityframework", "ef"),
        base_url="https://learn.microsoft.com/en-us/dotnet/",
        resolve=_json_api(
            "https://learn.microsoft.com/api/search?search={q}&locale=en-us&%24top=5&scope=.NET",
            results_key="results", url_field="url", title_field="title",
        ),
    ),
    Source(
        id="dart", name="Dart", aliases=("dart",),
        base_url="https://api.dart.dev/stable/",
        resolve=_templates(
            "https://api.dart.dev/stable/dart-core/{base}-class.html",
            title="{base} — Dart core",
        ),
    ),
    Source(
        id="c", name="C/C++", aliases=("c", "cpp", "c++", "cppreference"),
        base_url="https://en.cppreference.com/w/",
        resolve=_templates(
            "https://en.cppreference.com/w/cpp/container/{us}",
            "https://en.cppreference.com/w/cpp/algorithm/{us}",
            "https://en.cppreference.com/w/cpp/string/basic_string/{us}",
            "https://en.cppreference.com/w/c/io/{us}",
            "https://en.cppreference.com/w/c/string/byte/{us}",
        ),
    ),
    Source(
        id="bash", name="Bash", aliases=("bash", "sh", "shell", "zsh"),
        base_url="https://www.gnu.org/software/bash/manual/bash.html",
        resolve=_templates(
            "https://www.gnu.org/software/bash/manual/bash.html",
            title="Bash Reference Manual",
        ),
    ),
    Source(
        id="sql", name="SQL (SQLite)",
        aliases=("sql", "sqlite", "mysql", "postgres", "postgresql"),
        base_url="https://www.sqlite.org/docs.html",
        resolve=_html_search(
            "https://www.sqlite.org/search?s=d&q={q}",
            link_pattern=r'<a\s[^>]*href=["\']([^"\']+\.html)["\'][^>]*>(.*?)</a>',
            base="https://www.sqlite.org",
        ),
    ),
    Source(
        id="laravel", name="Laravel", aliases=("laravel",),
        base_url="https://laravel.com/docs/11.x",
        resolve=_templates(
            "https://laravel.com/docs/11.x/{dash}",
            title="Laravel: {q}",
        ),
    ),
    Source(
        id="django", name="Django", aliases=("django",),
        base_url="https://docs.djangoproject.com/en/5.0/",
        resolve=_html_search(
            "https://docs.djangoproject.com/en/5.0/search/?q={q}",
            link_pattern=r'<dt>\s*<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            base="https://docs.djangoproject.com",
        ),
    ),
    Source(
        id="flask", name="Flask", aliases=("flask",),
        base_url="https://flask.palletsprojects.com/en/3.0.x/",
        resolve=_html_search(
            "https://flask.palletsprojects.com/en/3.0.x/search/?q={q}",
            link_pattern=r'<li[^>]*>\s*<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            base="https://flask.palletsprojects.com/en/3.0.x/",
        ),
    ),
    Source(
        id="numpy", name="NumPy", aliases=("numpy", "np"),
        base_url="https://numpy.org/doc/stable/reference/index.html",
        resolve=_templates(
            "https://numpy.org/doc/stable/reference/generated/numpy.{dot}.html",
            title="numpy.{dot}",
        ),
    ),
    Source(
        id="pandas", name="Pandas", aliases=("pandas", "pd"),
        base_url="https://pandas.pydata.org/docs/reference/index.html",
        resolve=_templates(
            "https://pandas.pydata.org/docs/reference/api/pandas.{raw}.html",
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.{raw}.html",
            "https://pandas.pydata.org/docs/reference/api/pandas.Series.{raw}.html",
            title="pandas.{raw}",
        ),
    ),
    Source(
        id="node", name="Node.js", aliases=("node", "nodejs"),
        base_url="https://nodejs.org/api/",
        resolve=_templates(
            "https://nodejs.org/api/{base}.html",
            title="Node.js: {base}",
        ),
    ),
    Source(
        id="react", name="React", aliases=("react", "reactjs"),
        base_url="https://react.dev/reference/react",
        resolve=_templates(
            "https://react.dev/reference/react/{raw}",
            "https://react.dev/reference/react-dom/{raw}",
            title="React: {raw}",
        ),
    ),
    Source(
        id="vue", name="Vue.js", aliases=("vue", "vuejs", "vue3"),
        base_url="https://vuejs.org/api/",
        resolve=_templates(
            "https://vuejs.org/api/{dash}.html",
            "https://vuejs.org/guide/{dash}.html",
            title="Vue: {raw}",
        ),
    ),
    Source(
        id="spring", name="Spring", aliases=("spring", "springboot", "spring-boot"),
        base_url="https://docs.spring.io/spring-framework/reference/",
        resolve=_templates(
            "https://docs.spring.io/spring-framework/reference/{dash}.html",
            "https://docs.spring.io/spring-boot/reference/{dash}.html",
            title="Spring: {q}",
        ),
    ),
    Source(
        id="rails", name="Ruby on Rails", aliases=("rails", "rubyonrails", "ror"),
        base_url="https://api.rubyonrails.org/",
        resolve=_templates(
            "https://api.rubyonrails.org/classes/{base}.html",
            "https://guides.rubyonrails.org/{us}.html",
            title="Rails: {raw}",
        ),
    ),
    Source(
        id="flutter", name="Flutter", aliases=("flutter",),
        base_url="https://api.flutter.dev/flutter/",
        resolve=_templates(
            "https://api.flutter.dev/flutter/widgets/{last}-class.html",
            "https://api.flutter.dev/flutter/material/{last}-class.html",
            "https://api.flutter.dev/flutter/cupertino/{last}-class.html",
            title="Flutter: {last}",
        ),
    ),
]


def _alias_map() -> dict[str, Source]:
    out: dict[str, Source] = {}
    for src in SOURCES:
        out[src.id.lower()] = src
        for alias in src.aliases:
            out[alias.lower()] = src
    return out


_ALIAS_MAP = _alias_map()


def resolve_source(language: str) -> "Source | None":
    return _ALIAS_MAP.get(language.strip().lower())


# ── The single generic lookup engine ────────────────────────────────────────────

def lookup(language: str, query: str) -> ToolResult:
    """Truthful failure modes (NEVER fabrication):
      * unknown language → ``code=unknown-source`` + valid= the table's ids.
      * no candidate AND base_url unfetchable → ``code=source-unavailable``.
      * top candidate fetch/extract fails → ``code=source-unavailable``.

    Degraded-but-real: when no symbol candidate yields content but the source's
    real base/landing page does, succeed with ``meta degraded=true`` — a true
    page, just less precise than the symbol the model asked for.
    """
    source = resolve_source(language)
    if source is None:
        return ToolResult.err(
            f"Unknown language/framework: {language}.",
            code="unknown-source",
            hint="pick a supported language id, or use the search tool",
            valid=tuple(s.id for s in SOURCES),
        )

    try:
        candidates = source.resolve(query)
    except (requests.RequestException, FetchBlocked, ValueError):
        # A search-API/HTML-search resolver failed to reach its endpoint; we still
        # try the source's real base page below before giving up — never invent.
        candidates = []

    # Fetch candidates in order; the first that yields real content wins. Fall
    # back to the source's real base/landing page (a TRUE page) marked degraded.
    degraded = False
    ordered = list(candidates)
    base_row: dict[str, object] = {"url": source.base_url, "title": source.name}
    if base_row["url"] and base_row["url"] not in {c["url"] for c in ordered}:
        ordered.append(base_row)

    for index, candidate in enumerate(ordered):
        try:
            html = _fetch(cast(str, candidate["url"]))
        except (requests.RequestException, FetchBlocked):
            continue
        excerpt = extract_html(html, url=cast(str, candidate["url"]))
        if not excerpt:
            continue
        excerpt, _ = truncate(excerpt, _MAX_CHARS)
        # We landed on the base/landing page rather than a symbol candidate, or
        # the only thing that resolved was the base row → real, but imprecise.
        degraded = candidate is base_row or (not candidates)
        rows = [{
            "source": source.name,
            "title": candidate["title"],
            "url": candidate["url"],
            "excerpt": excerpt,
        }]
        # Surface remaining candidates (url/title only) so the model can read on.
        for other in ordered[index + 1:]:
            if other["url"] == candidate["url"]:
                continue
            rows.append({
                "source": source.name,
                "title": other["title"],
                "url": other["url"],
                "excerpt": "",
            })
            if len(rows) >= _MAX_RESULTS:
                break
        meta: dict[str, object] = {"source": source.id, "count": len(rows)}
        if degraded:
            meta["degraded"] = True
        return ToolResult.ok(rows, **cast("dict[str, dict[str, object] | None]", meta))

    return ToolResult.err(
        f"Could not reach {source.name} documentation for '{query}'.",
        code="source-unavailable",
        hint="try another source or the search tool",
        source=source.id,
    )


# ── Ability class ────────────────────────────────────────────────────────────────

class ProgrammingDocsSearchAbility(Ability):
    def get_name(self) -> str:
        return "programming_docs_search"

    def get_summary(self) -> str:
        return "Search official documentation for 12 programming languages and 11 frameworks by language name and query."

    def get_examples(self) -> list[str]:
        return [
            "how do I use asyncio in Python",
            "what is the signature of array_map in PHP",
            "look up the useState hook in React",
            "how does Eloquent ORM work in Laravel",
            "what is the difference between Arc and Rc in Rust",
            "Django queryset filter documentation",
            "how to use goroutines in Go",
            "look up the pandas DataFrame merge method",
        ]

    def get_search_tooltip(self) -> str:
        return "programming docs search"

    # The enum is DERIVED from the source table, so a schema-obedient model can
    # never name a language the ability then rejects as unknown — the schema and
    # the resolver are the same source of truth (every id first, then aliases).
    _LANGUAGE_ENUM: ClassVar[list[str]] = list(
        dict.fromkeys(
            [s.id for s in SOURCES]
            + [a.lower() for s in SOURCES for a in s.aliases]
        )
    )

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.language: {
                "type": "string",
                "enum": _LANGUAGE_ENUM,
                "description": "Programming language or framework to search documentation for.",
            },
            Keys.query: {
                "type": "string",
                "description": "What to look up — function name, class, method, module, or concept in plain language.",
            },
        },
        "required": [Keys.language, Keys.query],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        language = self.param(params, Keys.language, required=True)
        query = self.param(params, Keys.query, required=True)
        language = str(language).strip()
        query = str(query).strip()
        if not language:
            return ToolResult.err(
                "Required parameter 'language' is missing.",
                code="invalid-param", hint="pass language and query.",
            )
        if not query:
            return ToolResult.err(
                "Required parameter 'query' is missing.",
                code="invalid-param", hint="pass language and query.",
            )
        return lookup(language, query)
