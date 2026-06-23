"""HTML sanitisation primitives for Chalie's structured response format.

The backend sanitises every assistant response through ``nh3`` before
persisting/sending to the frontend.

Security: the LLM is NOT allowed to emit ``<a>`` (URLs are linkified by the
frontend), so no arbitrary anchors can be injected into the rendered DOM.

``<span id="tool_N">`` survives sanitisation because the rich-media protocol
requires these tags at the WS-send boundary; only the ``id`` attribute is
permitted — ``class``, ``style``, ``onclick``, and all other span attributes
are stripped.
"""
from __future__ import annotations

import html as _html
import re

import nh3

# Block-level tags that must produce a word break in spoken output. Without
# this, ``<li>A</li><li>B</li>`` collapses to ``AB`` once nh3 strips tags,
# and phonemizer silently drops the resulting gibberish token. We insert a
# space at each opening / closing boundary before tag stripping so adjacent
# items stay separable.
_BLOCK_BOUNDARY_RE = re.compile(r"</?(?:p|li|ul|h1|br|tr|td|th)\b[^>]*>", re.IGNORECASE)

# ── Tag allowlist ───────────────────────────────────────────────────────────

LLM_TAGS = frozenset({
    "b", "i", "u", "h1", "code", "p", "ul", "li", "span",
    "table", "thead", "tbody", "tfoot", "tr", "td", "th",
})
PROGRAMMATIC_TAGS = frozenset({"img", "actions", "action"})
ALLOWED_TAGS = LLM_TAGS | PROGRAMMATIC_TAGS

# Programmatic-only attributes. ``<img>`` carries src/alt; ``<action>`` carries
# the chat-button label/value plus the overlay daemon's data-* hooks.
# ``<span>`` carries only ``id`` — the rich-media pairing key.
_ATTRIBUTES = {
    "img": {"src", "alt"},
    "action": {"label", "value", "execute", "collect", "target", "open-url", "payload", "style"},
    "span": {"id"},
}

# Image src URLs are restricted to http(s). ``data:`` and ``javascript:`` are
# blocked by virtue of not matching this allowlist.
_URL_SCHEMES = {"http", "https"}

# Bound input length to keep sanitiser memory + time predictable on hostile
# input. Real content is bounded by LLM token limits; 5 MB is generous.
_MAX_CONTENT_LEN = 5_000_000


# ── Public API ──────────────────────────────────────────────────────────────


def sanitize(html: str | None) -> str:
    """Single chokepoint for LLM-emitted markup. Empty/None input returns ""."""
    if not html:
        return ""
    bounded = html if len(html) <= _MAX_CONTENT_LEN else html[:_MAX_CONTENT_LEN]
    return nh3.clean(
        bounded,
        tags=set(ALLOWED_TAGS),
        attributes=_ATTRIBUTES,
        url_schemes=_URL_SCHEMES,
        link_rel=None,  # we do not allow <a> from the LLM, so no rel injection needed
    )


def actions_to_xml(actions: list[dict[str, object]]) -> str:
    """Action dicts are built by the harness which bypasses the LLM, so
    output feeds straight to ``sanitize()`` alongside any LLM body."""
    if not actions:
        return ""
    parts = ["<actions>"]
    for a in actions:
        # Defensive: callers (or upstream JSON) may include strings/None in the
        # list. Skip non-dict entries rather than crashing the whole render.
        if not isinstance(a, dict):
            continue
        label = escape_attr(str(a.get("label", "")))
        value = escape_attr(str(a.get("value", "")))
        parts.append(f'<action label="{label}" value="{value}"/>')
    parts.append("</actions>")
    return "".join(parts)



def escape_attr(value: str) -> str:
    if not value:
        return ""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
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
    """Used by the TTS / speak button. ``<actions>`` blocks are dropped via ``clean_content_tags`` because UI affordances don't belong in spoken output."""
    if not html:
        return ""
    spaced = _BLOCK_BOUNDARY_RE.sub(" ", html)
    stripped = nh3.clean(spaced, tags=set(), clean_content_tags={"actions"})
    return " ".join(_html.unescape(stripped).split()).strip()
