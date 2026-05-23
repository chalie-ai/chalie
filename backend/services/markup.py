"""HTML sanitisation primitives for Chalie's structured response format.

The LLM emits a strict subset of HTML. The backend sanitises every assistant
response through ``nh3`` (Rust binding to the OWASP-aligned ``ammonia``
library) before persisting / sending to the frontend. That single chokepoint
replaces every hand-rolled tokenizer, regex tag-matcher, and markdown
converter that lived here previously.

Allowlist (11 tags total):
- LLM-emittable formatting (8): b, i, u, h1, code, p, ul, li
- Rich-media pairing (1): span (id attribute only — used by RichMediaParser)
- Programmatic only (3): img, actions, action

The LLM is NOT allowed to emit ``<a>``. Plain-text URLs in the LLM's
response are linkified by the frontend so the model cannot inject
arbitrary anchors into the rendered DOM.

``<span id="tool_N">`` is allowed because the rich-media protocol requires
these tags to survive the sanitisation pass so the parser can read them at
the WS-send boundary. Only the ``id`` attribute is permitted; ``class``,
``style``, ``onclick``, and all other span attributes are stripped.

``sanitize()`` is the only entry point for LLM output. It accepts mixed
plain text + allowlisted HTML and passes both through unchanged — text
nodes are valid HTML, so no wrapping or escaping heuristics are needed.
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
    """Strip every tag / attribute outside the allowlist. Returns clean HTML.

    Single chokepoint for LLM-emitted markup. Empty / ``None`` input returns
    ``""`` so downstream code never has to nil-check.
    """
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


def actions_to_xml(actions: list[dict]) -> str:
    """Render programmatic action buttons as XML.

    Each action dict requires ``label`` and ``value`` keys. The harness
    builds these — they bypass the LLM entirely — so the output here is
    fed straight to ``sanitize()`` alongside any LLM body.
    """
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
    """Escape value for use inside double-quoted XML attribute."""
    if not value:
        return ""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def extract_plaintext(html: str) -> str:
    """Strip all tags + drop ``<actions>`` subtree, return spoken plain text.

    Used by the TTS / speak button. ``<actions>`` blocks are dropped entirely
    via ``clean_content_tags`` — UI affordances do not belong in spoken
    output. Every other tag (``<p>``, ``<b>``, ``<img>``, …) collapses to
    its inner text content; image ``alt`` is *not* spoken (it's an a11y
    label for the visual surface, not narration). Entities are decoded so
    the speaker says ``cats & dogs``, not ``cats &amp; dogs``. Whitespace
    is collapsed.
    """
    if not html:
        return ""
    spaced = _BLOCK_BOUNDARY_RE.sub(" ", html)
    stripped = nh3.clean(spaced, tags=set(), clean_content_tags={"actions"})
    return " ".join(_html.unescape(stripped).split()).strip()
