"""HTML sanitisation primitives for Chalie's structured response format.

The LLM emits a strict subset of HTML. The backend sanitises every assistant
response through ``nh3`` (Rust binding to the OWASP-aligned ``ammonia``
library) before persisting / sending to the frontend. That single chokepoint
replaces every hand-rolled tokenizer, regex tag-matcher, and markdown
converter that lived here previously.

Allowlist (10 tags total):
- LLM-emittable formatting (8): b, i, u, h1, code, p, ul, li
- Programmatic only (3): img, actions, action

The LLM is NOT allowed to emit ``<a>``. Plain-text URLs in the LLM's
response are linkified by the frontend so the model cannot inject
arbitrary anchors into the rendered DOM.
"""
from __future__ import annotations

import html as _html

import nh3

# ── Tag allowlist ───────────────────────────────────────────────────────────

LLM_TAGS = frozenset({"b", "i", "u", "h1", "code", "p", "ul", "li"})
PROGRAMMATIC_TAGS = frozenset({"img", "actions", "action"})
ALLOWED_TAGS = LLM_TAGS | PROGRAMMATIC_TAGS

# Programmatic-only attributes. ``<img>`` carries src/alt; ``<action>`` carries
# the chat-button label/value plus the overlay daemon's data-* hooks.
_ATTRIBUTES = {
    "img": {"src", "alt"},
    "action": {"label", "value", "execute", "collect", "target", "open-url", "payload", "style"},
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


def is_xml_content(text: str | None) -> bool:
    """True if *text* is non-empty and its first non-whitespace char is ``<``.

    Single chokepoint for the "did the LLM emit XML or plaintext?" decision.
    Used by output_service + websocket fallbacks to decide whether to wrap
    the response in ``<p>...</p>``.
    """
    return bool((text or "").lstrip().startswith("<"))


def wrap_text_xml(text: str) -> str:
    """Wrap a plain string as ``<p>{escaped}</p>``. Empty input returns empty.

    Used when the model emits plain text instead of structured markup, so the
    frontend always receives well-formed HTML.
    """
    stripped = (text or "").strip()
    if not stripped:
        return ""
    return f"<p>{escape_text(stripped)}</p>"


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


def escape_text(text: str) -> str:
    """Escape & < > for HTML text content. Used by callers that build their own
    structured tags around model-supplied text (e.g. ``wrap_text_xml``).
    """
    if not text:
        return ""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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
    stripped = nh3.clean(html, tags=set(), clean_content_tags={"actions"})
    return " ".join(_html.unescape(stripped).split()).strip()
