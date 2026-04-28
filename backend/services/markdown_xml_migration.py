"""One-shot markdown → XML migration for legacy transcript rows.

Runs at boot via run_if_needed(). Idempotent: a sentinel column on transcript
records which rows have been converted.

Conversion is regex-based and lossy. Markdown features without an XML
equivalent (tables, horizontal rules, strikethrough, blockquotes, headings >h1)
are flattened or dropped.
"""
from __future__ import annotations

import logging
import re
import sqlite3

from services.markup import escape_attr, escape_text

logger = logging.getLogger(__name__)

_FENCED_CODE_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)\n?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")
_BOLD_ASTERISK_RE = re.compile(r"\*\*([^*\n]+?)\*\*")
_BOLD_UNDERSCORE_RE = re.compile(r"__([^_\n]+?)__")
_ITALIC_ASTERISK_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_ITALIC_UNDERSCORE_RE = re.compile(r"(?<!_)_([^_\n]+?)_(?!_)")
_STRIKE_RE = re.compile(r"~~([^~\n]+?)~~")
_LINK_RE = re.compile(r"\[([^\]]+?)\]\(([^)\s]+?)\)")
_IMAGE_RE = re.compile(r"!\[([^\]]*?)\]\(([^)\s]+?)\)")
_HORIZONTAL_RULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")


def _looks_like_xml(text: str) -> bool:
    """Heuristic: starts with an allowlisted opening tag.

    Covers all 11 allowlisted tags (LLM_TAGS + PROGRAMMATIC_TAGS). Without this,
    rows that begin with inline-only tags (e.g. `<b>...`, `<a href=...>`) would
    be re-processed on every boot once Phase C cuts over to XML-emitting
    writers — silently corrupting valid XML.
    """
    stripped = text.lstrip()
    return stripped.startswith((
        "<p>", "<h1>", "<ul>", "<li>",
        "<b>", "<i>", "<u>",
        "<code>", "<a ", "<a>",
        "<img", "<actions",
    ))


def _convert_inline(text: str) -> str:
    """Apply inline markdown conversions to a chunk of escaped text.

    Order matters: code first (so its contents aren't touched), then bold (so
    ** doesn't get parsed as nested *italic*), then italic, then strike, then
    images (before links — image syntax is `![alt](url)` and starts with `!`),
    then links.
    """
    # Inline code: protect contents from further parsing by replacing with a
    # placeholder, restoring after all other conversions.
    placeholders: list[str] = []

    def stash_code(match: re.Match) -> str:
        # NOTE: text has already been XML-escaped by the caller (markdown_to_xml).
        # Re-escaping group(1) here would double-escape `&` `<` `>`. Use the
        # captured group directly.
        placeholders.append(f"<code>{match.group(1)}</code>")
        return f"\x00{len(placeholders) - 1}\x00"

    text = _INLINE_CODE_RE.sub(stash_code, text)

    # Bold (must run before italic so ** isn't eaten as two *)
    text = _BOLD_ASTERISK_RE.sub(r"<b>\1</b>", text)
    text = _BOLD_UNDERSCORE_RE.sub(r"<b>\1</b>", text)
    # Italic
    text = _ITALIC_ASTERISK_RE.sub(r"<i>\1</i>", text)
    text = _ITALIC_UNDERSCORE_RE.sub(r"<i>\1</i>", text)
    # Strikethrough — no equivalent tag, drop the tildes, keep the text
    text = _STRIKE_RE.sub(r"\1", text)
    # Images first (so `!` doesn't get swallowed by link regex)
    # escape_attr both groups — alt may contain quotes; src may contain &.
    text = _IMAGE_RE.sub(
        lambda m: f'<img src="{escape_attr(m.group(2))}" alt="{escape_attr(m.group(1))}"/>',
        text,
    )
    # Links — href must be attr-escaped; label is already text-escaped.
    text = _LINK_RE.sub(
        lambda m: f'<a href="{escape_attr(m.group(2))}">{m.group(1)}</a>',
        text,
    )

    # Restore code placeholders
    for idx, code_xml in enumerate(placeholders):
        text = text.replace(f"\x00{idx}\x00", code_xml)
    return text


def markdown_to_xml(md: str) -> str:
    """Convert markdown string to allowlisted XML.

    Idempotent: input that already looks like XML passes through unchanged.
    """
    if not md:
        return ""
    if _looks_like_xml(md):
        return md.strip()

    # Step 1: extract fenced code blocks first so their contents are untouched.
    fenced_blocks: list[str] = []

    def stash_fenced(match: re.Match) -> str:
        fenced_blocks.append(f"<code>{escape_text(match.group(1))}</code>")
        return f"\x01{len(fenced_blocks) - 1}\x01"

    md = _FENCED_CODE_RE.sub(stash_fenced, md)

    # Step 2: split into block-level units separated by blank lines.
    raw_blocks = re.split(r"\n\s*\n", md.strip())
    out: list[str] = []
    for raw in raw_blocks:
        block = raw.rstrip()
        if not block:
            continue

        # Restore fenced code if this block IS a fenced placeholder
        if re.fullmatch(r"\x01\d+\x01", block):
            idx = int(block.strip("\x01"))
            out.append(fenced_blocks[idx])
            continue

        # Horizontal rule — drop entirely
        if _HORIZONTAL_RULE_RE.match(block):
            continue

        # Heading (any level → h1)
        heading_match = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", block)
        if heading_match:
            inner = _convert_inline(escape_text(heading_match.group(1)))
            out.append(f"<h1>{inner}</h1>")
            continue

        # List (unordered or ordered)
        lines = block.split("\n")
        if all(re.match(r"^\s*(?:[-*]|\d+\.)\s+", ln) for ln in lines if ln.strip()):
            items = []
            for ln in lines:
                if not ln.strip():
                    continue
                item_text = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", ln).rstrip()
                inner = _convert_inline(escape_text(item_text))
                items.append(f"<li>{inner}</li>")
            out.append(f"<ul>{''.join(items)}</ul>")
            continue

        # Blockquote — best-effort italic paragraph
        if all(ln.strip().startswith(">") for ln in lines if ln.strip()):
            inner_lines = [re.sub(r"^\s*>\s?", "", ln) for ln in lines]
            inner = _convert_inline(escape_text(" ".join(inner_lines).strip()))
            out.append(f"<p><i>{inner}</i></p>")
            continue

        # Table (markdown pipe table) — flatten each row to a paragraph
        if any("|" in ln for ln in lines) and any(re.match(r"^\s*\|?\s*[-:]+", ln) for ln in lines):
            for ln in lines:
                if re.match(r"^\s*\|?\s*[-:]+", ln):
                    continue  # skip separator
                cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                cells = [c for c in cells if c]
                if cells:
                    inner = _convert_inline(escape_text(" ".join(cells)))
                    out.append(f"<p>{inner}</p>")
            continue

        # Standalone image — emit bare <img/> without <p> wrapper
        _img_only = _IMAGE_RE.fullmatch(block.strip())
        if _img_only:
            out.append(
                f'<img src="{escape_attr(_img_only.group(2))}" '
                f'alt="{escape_attr(_img_only.group(1))}"/>'
            )
            continue

        # Default: paragraph
        inner = _convert_inline(escape_text(block))
        out.append(f"<p>{inner}</p>")

    # Step 3: restore any remaining fenced code placeholders that ended up
    # adjacent to text (rare — they're usually in their own block).
    result = "".join(out)
    for idx, code_xml in enumerate(fenced_blocks):
        result = result.replace(f"\x01{idx}\x01", code_xml)
    return result


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
        "SELECT id, content FROM transcript WHERE xml_migrated = 0"
    )
    total = 0
    while True:
        batch = cursor.fetchmany(_BATCH_SIZE)
        if not batch:
            break
        for row_id, content in batch:
            if content is None:
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
