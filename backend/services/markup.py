"""XML markup primitives for Chalie's structured response format.

Allowlist (11 tags total):
- LLM-emittable (8): b, i, u, h1, code, p, ul, li, a
- Programmatic only (3): img, actions, action

Parser is tolerant: unknown tags render as escaped plaintext, unclosed tags
auto-close at EOF. Nesting is not validated — model produces sensible structure.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

LLM_TAGS = frozenset({"b", "i", "u", "h1", "code", "p", "ul", "li", "a"})
PROGRAMMATIC_TAGS = frozenset({"img", "actions", "action"})
ALLOWED_TAGS = LLM_TAGS | PROGRAMMATIC_TAGS

VOID_TAGS = frozenset({"img", "action"})  # self-closing, no children

_ENTITY_AMP = "&amp;"
_ENTITY_LT = "&lt;"
_ENTITY_GT = "&gt;"
_ENTITY_QUOT = "&quot;"


def escape_text(text: str) -> str:
    """Escape & < > for XML text content. Does not escape attribute quotes."""
    if not text:
        return ""
    return text.replace("&", _ENTITY_AMP).replace("<", _ENTITY_LT).replace(">", _ENTITY_GT)


def escape_attr(value: str) -> str:
    """Escape value for use inside double-quoted XML attribute."""
    if not value:
        return ""
    return (
        value.replace("&", _ENTITY_AMP)
        .replace("<", _ENTITY_LT)
        .replace(">", _ENTITY_GT)
        .replace('"', _ENTITY_QUOT)
    )


def is_xml_content(text: str | None) -> bool:
    """True if *text* is non-empty and its first non-whitespace char is '<'.

    Single chokepoint for the "did the LLM emit XML or plaintext?" decision.
    Used by output_service + websocket fallbacks to decide whether to wrap
    the response in <p>...</p>.
    """
    return bool((text or "").lstrip().startswith("<"))


def wrap_text_xml(text: str) -> str:
    """Wrap a plain string as <p>{escaped}</p>. Empty input returns empty."""
    stripped = (text or "").strip()
    if not stripped:
        return ""
    return f"<p>{escape_text(stripped)}</p>"


def actions_to_xml(actions: list[dict]) -> str:
    """Render programmatic action buttons as XML.

    Each action dict requires 'label' and 'value' keys.
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


@dataclass
class Token:
    kind: str  # "text" | "open" | "close" | "void"
    name: str  # tag name for open/close/void; raw text for "text"
    attrs: dict[str, str] = field(default_factory=dict)


# Bound input to keep parser memory + time predictable on hostile input.
# Real content is bounded by LLM token limits; 5 MB is generous.
_MAX_CONTENT_LEN = 5_000_000


class _TokenAccumulator(HTMLParser):
    """Tolerant HTML parser tuned for Chalie's allowlisted XML markup.

    Behaviour:
      - Tag names are case-folded by HTMLParser.
      - Text data has character references auto-decoded (convert_charrefs=True).
        Attribute values are decoded explicitly via ``html.unescape``.
      - Tags outside ``ALLOWED_TAGS`` round-trip as raw text tokens, preserving
        the original opening / closing markup so renderers can show them as-is.
      - Tags in ``VOID_TAGS`` always emit ``Token('void', ...)`` regardless of
        whether the source used ``<img>`` or ``<img/>``.
      - Unclosed tags are NOT auto-closed here; the renderer is the auto-close
        chokepoint.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tokens: list[Token] = []

    # ── Tag handlers ──────────────────────────────────────────────────────────

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in ALLOWED_TAGS:
            self._emit_raw_tag()
            return
        decoded_attrs = self._decoded_attrs(attrs)
        if tag in VOID_TAGS:
            self.tokens.append(Token("void", tag, decoded_attrs))
            return
        self.tokens.append(Token("open", tag, decoded_attrs))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in ALLOWED_TAGS:
            self._emit_raw_tag()
            return
        self.tokens.append(Token("void", tag, self._decoded_attrs(attrs)))

    def handle_endtag(self, tag: str) -> None:
        if tag not in ALLOWED_TAGS:
            # Preserve the literal close marker so unknown-tag round-trips
            # render correctly — _merge_adjacent_text will coalesce.
            self.tokens.append(Token("text", f"</{tag}>", {}))
            return
        self.tokens.append(Token("close", tag, {}))

    def handle_data(self, data: str) -> None:
        if data:
            self.tokens.append(Token("text", data, {}))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _emit_raw_tag(self) -> None:
        """Round-trip a non-allowlisted tag back to text, using the source slice."""
        raw = self.get_starttag_text() or ""
        self.tokens.append(Token("text", raw, {}))

    @staticmethod
    def _decoded_attrs(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        # convert_charrefs only affects data; attribute values come back raw.
        return {name: html.unescape(value or "") for name, value in attrs}


def _merge_adjacent_text(tokens: list[Token]) -> list[Token]:
    """Merge runs of consecutive text tokens (produced by adjacent unknown tags + raw text)."""
    merged: list[Token] = []
    for tok in tokens:
        if merged and merged[-1].kind == "text" and tok.kind == "text":
            merged[-1] = Token("text", merged[-1].name + tok.name, {})
        else:
            merged.append(tok)
    return merged


def tokenize(content: str) -> list[Token]:
    """Tokenize XML markup. Unknown tags become text tokens (escaped allowlist enforcement).

    Returns flat list. Nesting validity is the renderer's concern. Input is
    truncated at _MAX_CONTENT_LEN to bound parser execution time on hostile
    input.
    """
    if not content:
        return []
    if len(content) > _MAX_CONTENT_LEN:
        content = content[:_MAX_CONTENT_LEN]

    parser = _TokenAccumulator()
    parser.feed(content)
    parser.close()
    return _merge_adjacent_text(parser.tokens)


_DROP_TAGS = frozenset({"actions"})  # contents not voiced


def _consume_token_for_plaintext(tok: Token, skip_depth: int, out: list[str]) -> int:
    """Per-token state machine for `extract_plaintext`. Returns the new skip_depth."""
    if skip_depth > 0:
        if tok.kind == "open" and tok.name in _DROP_TAGS:
            return skip_depth + 1
        if tok.kind == "close" and tok.name in _DROP_TAGS:
            return skip_depth - 1
        return skip_depth
    if tok.kind == "open" and tok.name in _DROP_TAGS:
        return 1
    if tok.kind == "void" and tok.name == "img":
        alt = tok.attrs.get("alt", "").strip()
        if alt:
            out.append(alt)
    elif tok.kind == "text":
        out.append(tok.name)
    return skip_depth


def extract_plaintext(content: str) -> str:
    """Strip tags for TTS / speak button. Drops <actions> entirely. Uses <img alt>."""
    if not content:
        return ""
    out: list[str] = []
    skip_depth = 0
    for tok in tokenize(content):
        skip_depth = _consume_token_for_plaintext(tok, skip_depth, out)
    return re.sub(r"\s+", " ", " ".join(out)).strip()
