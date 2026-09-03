"""HTML sanitisation primitives for Chalie's structured response format.

The backend sanitises every assistant response through ``nh3`` before
persisting/sending to the frontend.

Security: the LLM is NOT allowed to emit ``<a>`` (URLs are linkified by the
frontend), so no arbitrary anchors can be injected into the rendered DOM.

``<span id="tool_N">`` survives sanitisation because the rich-media protocol
requires these tags at the WS-send boundary; only the ``id`` attribute is
permitted — ``class``, ``style``, ``onclick``, and all other span attributes
are stripped.

``<img src>`` is gate-checked by ``_safe_img_src``: only absolute ``http:``
and ``https:`` schemes survive — relative paths (``/api/logout``),
protocol-relative URLs (``//evil.com/track.gif``), ``data:`` URIs, and
``javascript:`` URIs are all dropped (src attr removed, leaving the img
inert). The nh3 ``url_schemes`` kwarg alone would not have caught the
scheme-less cases; ``attribute_filter`` closes that gap.
"""
from __future__ import annotations

import html as _html
import re
from urllib.parse import urlsplit

import nh3

# Block-level tags that must produce a word break in spoken output. Without
# this, ``<li>A</li><li>B</li>`` collapses to ``AB`` once nh3 strips tags,
# and phonemizer silently drops the resulting gibberish token. We insert a
# space at each opening / closing boundary before tag stripping so adjacent
# items stay separable.
_BLOCK_BOUNDARY_RE = re.compile(r"</?(?:p|li|ul|h1|br|tr|td|th)\b[^>]*>", re.IGNORECASE)

# ── Tag allowlist ───────────────────────────────────────────────────────────

# The single authority for Chalie's HTML contract: the tags the model is told to
# emit, in the order the system prompt lists them. ``PromptService`` renders the
# instruction from this tuple and ``sanitize`` enforces it, so widening or
# narrowing the contract is one edit and the two cannot drift apart.
PROMPT_TAGS = (
    "p", "h1", "b", "i", "u", "code", "ul", "li",
    "table", "thead", "tbody", "tfoot", "tr", "th", "td",
)

# Programmatic-only attributes. ``<img>`` carries src/alt; ``<span>`` carries only
# ``id`` — the rich-media pairing key.
_ATTRIBUTES = {
    "img": {"src", "alt"},
    "span": {"id"},
    "td": {"colspan", "rowspan", "headers"},
    "th": {"colspan", "rowspan", "scope", "headers"},
}

# Image src URLs are restricted to absolute http(s) only. Relative paths,
# protocol-relative URLs, ``data:`` URIs, and ``javascript:`` URIs are all
# dropped by ``_safe_img_src`` (nh3 ``attribute_filter``). The nh3
# ``url_schemes`` kwarg below additionally rejects any *present* scheme that
# isn't http/https (e.g. ``ftp:``) — the two gates together close both sides.
_URL_SCHEMES = {"http", "https"}

# Bound input length to keep sanitiser memory + time predictable on hostile
# input. Real content is bounded by LLM token limits; 5 MB is generous.
_MAX_CONTENT_LEN = 5_000_000


# ── Public API ──────────────────────────────────────────────────────────────


def _safe_img_src(tag: str, attr: str, value: str) -> str | None:
    """Gate for ``<img src>``: only absolute ``http:``/``https:`` schemes survive.

    Closes the scheme-less bypass that nh3's ``url_schemes`` alone leaves open.
    nh3 rejects *present* non-allowlisted schemes (e.g. ``ftp:``), but a URL
    with no scheme at all (relative ``/api/logout``, protocol-relative
    ``//evil.com/track.gif``, ``data:`` URI, ``javascript:`` URI) slips through
    the ``url_schemes`` gate — ``urlsplit().scheme`` is empty for those. This
    helper drops the ``src`` attribute entirely for anything that isn't an
    absolute http/https URL, leaving the ``<img>`` inert. All other tags and
    attributes pass through unchanged (``<span>`` ``id``, etc.).
    """
    if tag != "img" or attr != "src":
        return value
    scheme = urlsplit(value).scheme.lower()
    return value if scheme in {"http", "https"} else None


def sanitize(html: str | None) -> str:
    """Single chokepoint for LLM-emitted markup. Empty/None input returns ""."""
    if not html:
        return ""
    bounded = html if len(html) <= _MAX_CONTENT_LEN else html[:_MAX_CONTENT_LEN]
    return nh3.clean(
        bounded,
        # The model is only ever told about PROMPT_TAGS. Two more survive because
        # Chalie injects them itself, never the LLM: <span id> pairs rich media at
        # the WS-send boundary, and <img src> is gate-checked by _safe_img_src.
        tags=set(PROMPT_TAGS) | {"span", "img"},
        attributes=_ATTRIBUTES,
        url_schemes=_URL_SCHEMES,
        attribute_filter=_safe_img_src,  # drops scheme-less / non-http(s) img src
        link_rel=None,  # we do not allow <a> from the LLM, so no rel injection needed
    )


# ── Heuristic markdown fallback ───────────────────────────────────────────────
#
# The system prompt instructs the LLM to emit Chalie's HTML subset directly, but
# models still occasionally leak the most common markdown emphasis markers. This
# is a best-effort, INLINE-ONLY fallback applied to the final user-facing text
# before it reaches ``sanitize()``. It rewrites only the handful of markers that
# map cleanly onto allowlisted tags and leaves everything else untouched (HTML
# the model already emitted, unrecognised markdown, block constructs, links). It
# is deliberately NOT a markdown parser.
#
#   **bold**   →  <b>bold</b>
#   *italic*   →  <i>italic</i>
#   _under_    →  <u>under</u>      (Chalie maps ``_`` to underline, not italic)
#   `code`     →  <code>code</code>

# Inline code is masked before emphasis runs so markers INSIDE a code span (e.g.
# ``a_b`` or ``x**y``) survive verbatim and are not rewritten.
_CODE_SPAN_RE = re.compile(r"`([^`\n]+?)`")
# Bold before italic: the ``**`` pair must be consumed before the single-``*``
# rule sees it. ``(?=\S)`` / ``(?<=\S)`` forbid leading / trailing whitespace so
# a stray ``** `` or a multiplication ``2 * 3`` is left alone.
_BOLD_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.DOTALL)
_ITALIC_RE = re.compile(r"(?<!\*)\*(?=\S)([^*\n]+?)(?<=\S)\*(?!\*)")
# Underline markers must not sit between word chars, so identifiers like
# ``snake_case`` and ``__dunder__`` are never mangled.
_UNDERLINE_RE = re.compile(r"(?<!\w)_(?=\S)([^_\n]+?)(?<=\S)_(?!\w)")
_CODE_TOKEN_RE = re.compile("\x00C(\\d+)\x00")


def markdown_to_html(text: str | None) -> str:
    """Best-effort inline markdown→HTML pre-pass for ``sanitize()`` when the LLM leaks emphasis markers. NOT a full markdown parser."""
    if not text:
        return ""

    # 1. Mask inline code so emphasis markers inside it are preserved verbatim.
    code_spans: list[str] = []

    def _stash(m: re.Match[str]) -> str:
        code_spans.append(m.group(1))
        return f"\x00C{len(code_spans) - 1}\x00"

    out = _CODE_SPAN_RE.sub(_stash, text)

    # 2. Emphasis — bold before italic so ``**`` is not eaten by the ``*`` rule.
    out = _BOLD_RE.sub(r"<b>\1</b>", out)
    out = _ITALIC_RE.sub(r"<i>\1</i>", out)
    out = _UNDERLINE_RE.sub(r"<u>\1</u>", out)

    # 3. Restore masked code spans as <code> tags.
    return _CODE_TOKEN_RE.sub(
        lambda m: f"<code>{code_spans[int(m.group(1))]}</code>", out
    )


def extract_plaintext(html: str) -> str:
    """Used by the TTS / speak button. Strips all HTML tags and normalises whitespace."""
    if not html:
        return ""
    spaced = _BLOCK_BOUNDARY_RE.sub(" ", html)
    stripped = nh3.clean(spaced, tags=set())
    return " ".join(_html.unescape(stripped).split()).strip()
