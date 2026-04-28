"""XML markup primitives for Chalie's structured response format.

Allowlist (11 tags total):
- LLM-emittable (8): b, i, u, h1, code, p, ul, li, a
- Programmatic only (3): img, actions, action

Parser is tolerant: unknown tags render as escaped plaintext, unclosed tags
auto-close at EOF. Nesting is not validated — model produces sensible structure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

LLM_TAGS = frozenset({"b", "i", "u", "h1", "code", "p", "ul", "li", "a"})
PROGRAMMATIC_TAGS = frozenset({"img", "actions", "action"})
ALLOWED_TAGS = LLM_TAGS | PROGRAMMATIC_TAGS

VOID_TAGS = frozenset({"img", "action"})  # self-closing, no children


def escape_text(text: str) -> str:
    """Escape & < > for XML text content. Does not escape attribute quotes."""
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


_TAG_RE = re.compile(
    r"<\s*(/)?\s*([a-zA-Z][a-zA-Z0-9]*)\s*((?:[a-zA-Z][a-zA-Z0-9-]*\s*=\s*\"[^\"]*\"\s*)*)\s*(/)?\s*>"
)
_ATTR_RE = re.compile(r'([a-zA-Z][a-zA-Z0-9-]*)\s*=\s*"([^"]*)"')
_ENTITY_MAP = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'"}


def _decode_entities(text: str) -> str:
    for ent, char in _ENTITY_MAP.items():
        text = text.replace(ent, char)
    return text


def tokenize(content: str) -> list[Token]:
    """Tokenize XML markup. Unknown tags become text tokens (escaped allowlist enforcement).

    Returns flat list. Nesting validity is the renderer's concern.
    """
    if not content:
        return []
    tokens: list[Token] = []
    pos = 0
    for match in _TAG_RE.finditer(content):
        # text before this tag
        if match.start() > pos:
            text_chunk = content[pos : match.start()]
            tokens.append(Token("text", _decode_entities(text_chunk), {}))
        is_close = bool(match.group(1))
        name = match.group(2).lower()
        attr_blob = match.group(3) or ""
        is_void = bool(match.group(4))

        if name not in ALLOWED_TAGS:
            # Unknown tag — keep raw text (escaped allowlist enforcement)
            tokens.append(Token("text", match.group(0), {}))
        else:
            attrs: dict[str, str] = {}
            for am in _ATTR_RE.finditer(attr_blob):
                attrs[am.group(1).lower()] = _decode_entities(am.group(2))
            if is_close:
                tokens.append(Token("close", name, {}))
            elif is_void or name in VOID_TAGS:
                tokens.append(Token("void", name, attrs))
            else:
                tokens.append(Token("open", name, attrs))
        pos = match.end()
    # trailing text
    if pos < len(content):
        tokens.append(Token("text", _decode_entities(content[pos:]), {}))
    # Merge consecutive text tokens (produced by adjacent unknown tags + raw text)
    merged: list[Token] = []
    for tok in tokens:
        if merged and merged[-1].kind == "text" and tok.kind == "text":
            merged[-1] = Token("text", merged[-1].name + tok.name, {})
        else:
            merged.append(tok)
    return merged


_DROP_TAGS = frozenset({"actions"})  # contents not voiced


def extract_plaintext(content: str) -> str:
    """Strip tags for TTS / speak button. Drops <actions> entirely. Uses <img alt>."""
    if not content:
        return ""
    tokens = tokenize(content)
    out: list[str] = []
    skip_depth = 0
    for tok in tokens:
        if skip_depth > 0:
            if tok.kind == "open" and tok.name in _DROP_TAGS:
                skip_depth += 1
            elif tok.kind == "close" and tok.name in _DROP_TAGS:
                skip_depth -= 1
            continue
        if tok.kind == "open" and tok.name in _DROP_TAGS:
            skip_depth = 1
        elif tok.kind == "void" and tok.name == "img":
            alt = tok.attrs.get("alt", "").strip()
            if alt:
                out.append(alt)
        elif tok.kind == "text":
            out.append(tok.name)
    raw = " ".join(out)
    return re.sub(r"\s+", " ", raw).strip()
