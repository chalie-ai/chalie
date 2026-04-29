"""One-shot markdown → XML migration for legacy transcript rows.

Runs at boot via run_if_needed(). Idempotent: a sentinel column on transcript
records which rows have been converted.

Conversion is delegated to mistune (CommonMark + GFM-style strikethrough +
GFM tables) and walked into Chalie's 11-tag allowlist. Markdown features
without an XML equivalent (tables, horizontal rules, strikethrough, headings
>h1, blockquotes' inner block structure) are flattened or dropped per the
allowlist.
"""
from __future__ import annotations

import logging
import sqlite3

import mistune

from services.markup import escape_attr, escape_text

logger = logging.getLogger(__name__)

# Bound input to keep mistune memory + time predictable on hostile input.
# Real transcript rows are bounded by LLM token limits; 5 MB is generous.
_MAX_CONTENT_LEN = 5_000_000

# Tags an LLM might emit at the start of a legitimate XML response. If we see
# one of these, the row is already XML and passes through unchanged.
_XML_SENTINELS = (
    "<p>", "<h1>", "<ul>", "<li>",
    "<b>", "<i>", "<u>",
    "<code>", "<a ", "<a>",
    "<img", "<actions",
)


def _looks_like_xml(text: str) -> bool:
    """Heuristic: starts with an allowlisted opening tag.

    Without this, rows that begin with inline-only tags (e.g. `<b>...`,
    `<a href=...>`) would be re-processed on every boot — silently corrupting
    valid XML emitted by the runtime.
    """
    return text.lstrip().startswith(_XML_SENTINELS)


# Single-shared parser, reused across all conversions. Plugins registered:
#   - strikethrough (GFM ~~text~~)
#   - table         (GFM pipe tables)
_md_parser = mistune.create_markdown(renderer=None, plugins=["strikethrough", "table"])


def markdown_to_xml(md: str) -> str:
    """Convert markdown string to allowlisted XML.

    Idempotent: input that already looks like XML passes through unchanged.
    Input is truncated at _MAX_CONTENT_LEN to bound parser execution time on
    hostile input.
    """
    if not md:
        return ""
    if len(md) > _MAX_CONTENT_LEN:
        md = md[:_MAX_CONTENT_LEN]
    if _looks_like_xml(md):
        return md.strip()

    ast = _md_parser(md)
    return "".join(_render_block(node) for node in ast)


# ── AST walker ────────────────────────────────────────────────────────────────


def _render_block(node: dict) -> str:
    """Render a top-level (block) AST node to XML. Returns '' to drop the node."""
    kind = node.get("type")
    handler = _BLOCK_HANDLERS.get(kind)
    if handler is None:
        # Unknown block type — render the children as text, defensively. This
        # keeps content surfaced rather than silently dropping it.
        return _render_inline_children(node.get("children") or [])
    return handler(node)


def _render_paragraph(node: dict) -> str:
    children = node.get("children") or []
    # Standalone image — emit a bare <img/> rather than wrapping in <p>.
    if len(children) == 1 and children[0].get("type") == "image":
        return _render_image(children[0])
    inner = _render_inline_children(children)
    return f"<p>{inner}</p>" if inner else ""


def _render_heading(node: dict) -> str:
    # All heading levels collapse to <h1>; the allowlist has no <h2>+.
    inner = _render_inline_children(node.get("children") or [])
    return f"<h1>{inner}</h1>" if inner else ""


def _render_list(node: dict) -> str:
    items = "".join(_render_list_item(child) for child in node.get("children") or [])
    return f"<ul>{items}</ul>" if items else ""


def _render_list_item(node: dict) -> str:
    """Flatten a list_item to a single <li>. Nested lists become peer <li> entries."""
    own_inline_parts: list[str] = []
    nested_items: list[str] = []
    for child in node.get("children") or []:
        ctype = child.get("type")
        if ctype == "block_text":
            own_inline_parts.append(_render_inline_children(child.get("children") or []))
        elif ctype == "paragraph":
            own_inline_parts.append(_render_inline_children(child.get("children") or []))
        elif ctype == "list":
            # Flatten nested list into peer <li>s alongside the parent's own item.
            for nested_child in child.get("children") or []:
                nested_items.append(_render_list_item(nested_child))
    own = "".join(p for p in own_inline_parts if p)
    return (f"<li>{own}</li>" if own else "") + "".join(nested_items)


def _render_block_quote(node: dict) -> str:
    """Blockquote → italic paragraph (best-effort flatten)."""
    parts = []
    for child in node.get("children") or []:
        if child.get("type") == "paragraph":
            parts.append(_render_inline_children(child.get("children") or []))
    inner = " ".join(p for p in parts if p)
    return f"<p><i>{inner}</i></p>" if inner else ""


_NEWLINE = "\n"


def _render_block_code(node: dict) -> str:
    """Fenced or indented code block → <code>{escaped}</code> (no <p> wrap)."""
    raw = node.get("raw") or ""
    # mistune appends a trailing newline; strip it so output matches the
    # original "<code>x = 1</code>" expectation rather than "<code>x = 1\n</code>".
    body = escape_text(raw.rstrip(_NEWLINE))
    return f"<code>{body}</code>"


def _render_table(node: dict) -> str:
    """Tables flatten: every row becomes a <p> with cells space-separated."""
    parts: list[str] = []
    for section in node.get("children") or []:
        section_type = section.get("type")
        if section_type == "table_head":
            parts.append(_render_table_row_from_section(section, is_head=True))
        elif section_type == "table_body":
            for row in section.get("children") or []:
                parts.append(_render_table_row_cells(row))
    return "".join(p for p in parts if p)


def _render_table_row_from_section(section: dict, *, is_head: bool) -> str:
    """``table_head`` is a flat list of ``table_cell`` nodes (no inner row)."""
    del is_head  # head/body shape is identical; kept for caller readability.
    cells = [_render_inline_children(c.get("children") or []) for c in section.get("children") or []]
    cells = [c for c in cells if c]
    return f"<p>{' '.join(cells)}</p>" if cells else ""


def _render_table_row_cells(row: dict) -> str:
    cells = [_render_inline_children(c.get("children") or []) for c in row.get("children") or []]
    cells = [c for c in cells if c]
    return f"<p>{' '.join(cells)}</p>" if cells else ""


def _render_dropped(node: dict) -> str:  # noqa: ARG001 — signature consistency with other handlers
    """Drop the block entirely (thematic_break, blank_line, etc.)."""
    return ""


def _render_block_html(node: dict) -> str:
    """Treat block-level raw HTML as plain text — escape to keep allowlist semantics."""
    raw = node.get("raw") or ""
    inner = escape_text(raw.strip())
    return f"<p>{inner}</p>" if inner else ""


_BLOCK_HANDLERS: dict[str, callable] = {
    "paragraph": _render_paragraph,
    "heading": _render_heading,
    "list": _render_list,
    "block_quote": _render_block_quote,
    "block_code": _render_block_code,
    "table": _render_table,
    "thematic_break": _render_dropped,
    "blank_line": _render_dropped,
    "block_html": _render_block_html,
    "block_error": _render_block_html,
}


# ── Inline rendering ──────────────────────────────────────────────────────────


def _render_inline_children(children: list) -> str:
    return "".join(_render_inline(c) for c in children)


def _render_inline(node: dict) -> str:
    """Render an inline AST node to XML."""
    kind = node.get("type")
    handler = _INLINE_HANDLERS.get(kind)
    if handler is None:
        # Unknown inline node — best-effort text recovery via children or raw.
        if node.get("children"):
            return _render_inline_children(node["children"])
        return escape_text(node.get("raw") or "")
    return handler(node)


def _render_text(node: dict) -> str:
    return escape_text(node.get("raw") or "")


def _render_strong(node: dict) -> str:
    inner = _render_inline_children(node.get("children") or [])
    return f"<b>{inner}</b>" if inner else ""


def _render_emphasis(node: dict) -> str:
    inner = _render_inline_children(node.get("children") or [])
    return f"<i>{inner}</i>" if inner else ""


def _render_codespan(node: dict) -> str:
    return f"<code>{escape_text(node.get('raw') or '')}</code>"


def _render_strikethrough(node: dict) -> str:
    """No <s> in our allowlist — drop the markers, keep the text."""
    return _render_inline_children(node.get("children") or [])


def _render_link(node: dict) -> str:
    url = (node.get("attrs") or {}).get("url") or ""
    label = _render_inline_children(node.get("children") or [])
    if not label:
        label = escape_text(url)
    return f'<a href="{escape_attr(url)}">{label}</a>'


def _render_image(node: dict) -> str:
    attrs = node.get("attrs") or {}
    url = attrs.get("url") or ""
    alt_children = node.get("children") or []
    # Image alt is always plain text — recover via raw to skip nested escape.
    alt_text = "".join(c.get("raw", "") for c in alt_children)
    return f'<img src="{escape_attr(url)}" alt="{escape_attr(alt_text)}"/>'


def _render_softbreak(node: dict) -> str:  # noqa: ARG001
    return " "


def _render_linebreak(node: dict) -> str:  # noqa: ARG001
    return " "


def _render_inline_html(node: dict) -> str:
    """Inline raw HTML → escaped text. Same rationale as block_html."""
    return escape_text(node.get("raw") or "")


_INLINE_HANDLERS: dict[str, callable] = {
    "text": _render_text,
    "strong": _render_strong,
    "emphasis": _render_emphasis,
    "codespan": _render_codespan,
    "strikethrough": _render_strikethrough,
    "link": _render_link,
    "image": _render_image,
    "softbreak": _render_softbreak,
    "linebreak": _render_linebreak,
    "inline_html": _render_inline_html,
}


# ── Boot-time batch migration ────────────────────────────────────────────────

_BATCH_SIZE = 500


def run_if_needed(conn: sqlite3.Connection) -> int:
    """Convert all unmigrated transcript rows from markdown → XML.

    Idempotent. Returns count of rows processed (including NULL-content rows
    that just get the sentinel flipped). Streams in batches of _BATCH_SIZE to
    keep memory bounded on large transcripts (10K+ rows).
    """
    # Match DatabaseService PRAGMAs so the migration plays nicely under WAL
    # contention with concurrent readers (boot is the only writer here, but
    # other long-lived connections may still hold the WAL).
    try:
        conn.execute("PRAGMA busy_timeout=15000")
    except sqlite3.Error:
        pass

    cursor = conn.execute(
        "SELECT id, role, content FROM transcript WHERE xml_migrated = 0"
    )
    total = 0
    while True:
        batch = cursor.fetchmany(_BATCH_SIZE)
        if not batch:
            break
        for row_id, role, content in batch:
            # Only assistant rows render through the XML markup pipeline; user,
            # tool, proactive_thought, scheduled, etc. are rendered as plaintext
            # (FE uses textContent) or fed back to the model verbatim. Wrapping
            # them in <p>...</p> would either leak the literal tag into the UI
            # or mangle structured tool output. Skip content conversion for
            # non-assistant roles — just flip the sentinel so the migration is
            # not retried.
            if content is None or role != "assistant":
                conn.execute(
                    "UPDATE transcript SET xml_migrated = 1 WHERE id = ?",
                    (row_id,),
                )
                continue
            try:
                converted = markdown_to_xml(content)
            except Exception as exc:
                logger.warning(
                    "xml-migration: row %d conversion failed (%s); marking migrated to skip retry",
                    row_id,
                    exc,
                )
                converted = content  # best-effort: keep original
            conn.execute(
                "UPDATE transcript SET content = ?, xml_migrated = 1 WHERE id = ?",
                (converted, row_id),
            )
        conn.commit()
        total += len(batch)
        logger.info("xml-migration: committed batch (%d rows total)", total)
    if total:
        logger.info("xml-migration: completed %d rows", total)
    return total
