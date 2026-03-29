"""
Blocks Render Service

Converts any content into the universal block format: a JSON-serialisable list of
dicts.  This is the core of the block protocol that replaces HTML injection in the
frontend.

Block schema (every block has at least a ``type`` key):

    {"type": "text",      "content": "...", "format": "markdown"|"plain"}
    {"type": "header",    "content": "...", "level": 1-6}
    {"type": "code",      "content": "...", "language": "python"|""}
    {"type": "list",      "style": "unordered"|"ordered", "items": [...]}
    {"type": "table",     "headers": [...], "rows": [[...]]}
    {"type": "keyvalue",  "pairs": [{"key": "K", "value": "V"}]}
    {"type": "image",     "url": "...", "alt": "..."}
    {"type": "link",      "url": "...", "text": "..."}
    {"type": "divider"}
    {"type": "actions",   "buttons": [{"label": "...", "skill": "...", "payload": {...}}]}
    {"type": "carousel",  "slides": [{"blocks": [...]}]}

No external dependencies — stdlib only.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# Block types that can be emitted via fenced-code directives (```metric, ```card, etc.)
# When the LLM writes a fenced block with one of these as the language tag and the
# content is valid JSON, the parser emits the parsed block instead of a code block.
DIRECTIVE_TYPES: frozenset[str] = frozenset({
    "metric", "card", "chart", "progress", "timeline",
    "alert", "badge", "keyvalue", "columns", "section",
})


class BlocksRenderService:
    """
    Stateless service that converts various content types into the universal block
    format.

    All public methods return ``list[dict]`` and are safe to call from any thread.
    Input/output are pure Python values — no I/O side effects.
    """

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def from_markdown(self, text: str) -> list[dict]:
        """Parse a markdown string into a list of typed blocks.

        Handles:
        - ATX headings (``# … ######``)
        - Fenced code blocks (``` ``` ``` with optional language specifier)
        - Unordered lists (``- `` / ``* ``)
        - Ordered lists (``1. ``)
        - Horizontal rules (``---``, ``***``, ``___``)
        - GFM-style pipe tables
        - Blank-line-separated paragraphs → individual text blocks
        - Inline markdown (bold, italic, code, links) is **not** parsed; it stays
          inside text blocks with ``"format": "markdown"`` for the client to render.

        Args:
            text: Raw markdown string, typically an LLM response.

        Returns:
            List of block dicts.  Empty list for blank/whitespace-only input.
        """
        if not text or not text.strip():
            return []

        lines = text.splitlines()
        blocks: list[dict] = []

        # Pending paragraph lines accumulated before we know the block type.
        pending_paragraph: list[str] = []
        # Pending list items before a list block is flushed.
        pending_list_items: list[str] = []
        pending_list_style: str | None = None

        # Pending table state.
        pending_table_headers: list[str] | None = None
        pending_table_rows: list[list[str]] = []
        in_table: bool = False

        i = 0
        n = len(lines)

        def flush_paragraph() -> None:
            """Emit accumulated paragraph lines as a text block."""
            nonlocal pending_paragraph
            joined = "\n".join(pending_paragraph).strip()
            if joined:
                blocks.append({"type": "text", "content": joined, "format": "markdown"})
            pending_paragraph = []

        def flush_list() -> None:
            """Emit accumulated list items as a list block."""
            nonlocal pending_list_items, pending_list_style
            if pending_list_items and pending_list_style:
                blocks.append(
                    {
                        "type": "list",
                        "style": pending_list_style,
                        "items": list(pending_list_items),
                    }
                )
            pending_list_items = []
            pending_list_style = None

        def flush_table() -> None:
            """Emit an accumulated table block."""
            nonlocal pending_table_headers, pending_table_rows, in_table
            if pending_table_headers is not None:
                blocks.append(
                    {
                        "type": "table",
                        "headers": list(pending_table_headers),
                        "rows": [list(r) for r in pending_table_rows],
                    }
                )
            pending_table_headers = None
            pending_table_rows = []
            in_table = False

        while i < n:
            line = lines[i]
            stripped = line.strip()

            # ------------------------------------------------------------------
            # 1. Fenced code block  (``` or ~~~)
            # ------------------------------------------------------------------
            fence_match = re.match(r"^(`{3,}|~{3,})(.*)?$", line)
            if fence_match:
                # Flush any open context first.
                flush_paragraph()
                flush_list()
                flush_table()

                fence_char = fence_match.group(1)[0]  # '`' or '~'
                fence_len = len(fence_match.group(1))
                language = (fence_match.group(2) or "").strip()

                # Collect lines until the matching closing fence.
                code_lines: list[str] = []
                i += 1
                closing_fence = re.compile(
                    r"^" + re.escape(fence_char * fence_len) + r"`*$"
                )
                while i < n:
                    if closing_fence.match(lines[i].rstrip()):
                        i += 1
                        break
                    code_lines.append(lines[i])
                    i += 1

                # Preserve internal whitespace; strip only the leading/trailing blank
                # lines that the author may have placed inside the fence.
                raw_code = "\n".join(code_lines)
                # Strip a single leading newline and a single trailing newline only.
                raw_code = raw_code.strip("\n")

                # Check if the language tag is a block directive (e.g. ```metric)
                if language in DIRECTIVE_TYPES:
                    try:
                        directive_data = json.loads(raw_code)
                        if isinstance(directive_data, dict):
                            directive_data["type"] = language
                            blocks.append(directive_data)
                            continue
                    except (json.JSONDecodeError, ValueError):
                        pass  # Fall through to regular code block

                blocks.append(
                    {"type": "code", "content": raw_code, "language": language}
                )
                continue  # i already advanced

            # ------------------------------------------------------------------
            # 2. ATX heading  (# … ######)
            # ------------------------------------------------------------------
            heading_match = re.match(r"^(#{1,6})\s+(.*)", stripped)
            if heading_match:
                flush_paragraph()
                flush_list()
                flush_table()
                level = len(heading_match.group(1))
                content = heading_match.group(2).strip()
                blocks.append({"type": "header", "content": content, "level": level})
                i += 1
                continue

            # ------------------------------------------------------------------
            # 3. Horizontal rule  (--- / *** / ___ on its own line)
            # ------------------------------------------------------------------
            if re.match(r"^(\*{3,}|-{3,}|_{3,})\s*$", stripped):
                flush_paragraph()
                flush_list()
                flush_table()
                blocks.append({"type": "divider"})
                i += 1
                continue

            # ------------------------------------------------------------------
            # 4. GFM table row
            # ------------------------------------------------------------------
            # A table row must start and end with '|' (after stripping).
            if stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 1:
                # Is this a separator row?  e.g. |---|---| or | :---: | --- |
                cell_values = [c.strip() for c in stripped[1:-1].split("|")]
                is_separator = all(
                    re.match(r"^:?-{1,}:?$", c) for c in cell_values if c
                )

                if in_table:
                    if is_separator:
                        # Second time we see a separator inside an active table —
                        # treat it as decoration and skip it.
                        i += 1
                        continue
                    # Normal data row.
                    pending_table_rows.append(cell_values)
                    i += 1
                    continue
                else:
                    # Are we at the start of a new table?
                    # Peek at the next line to see if it's a separator.
                    if i + 1 < n:
                        next_stripped = lines[i + 1].strip()
                        if next_stripped.startswith("|") and next_stripped.endswith("|"):
                            next_cells = [
                                c.strip() for c in next_stripped[1:-1].split("|")
                            ]
                            next_is_sep = all(
                                re.match(r"^:?-{1,}:?$", c)
                                for c in next_cells
                                if c
                            )
                            if next_is_sep:
                                # Start of a new table.
                                flush_paragraph()
                                flush_list()
                                # (No table to flush yet — we're starting one.)
                                pending_table_headers = cell_values
                                pending_table_rows = []
                                in_table = True
                                i += 2  # skip header row + separator row
                                continue

                # Not a table — fall through to paragraph handling.

            elif in_table:
                # We were in a table but this line doesn't look like a table row.
                flush_table()

            # ------------------------------------------------------------------
            # 5. Unordered list item  (- text  or  * text)
            # ------------------------------------------------------------------
            ul_match = re.match(r"^(\s*)[-*]\s+(.+)", line)
            if ul_match:
                flush_paragraph()
                flush_table()
                item_text = ul_match.group(2).strip()

                if pending_list_style == "ordered":
                    # List type changed — flush the previous list.
                    flush_list()

                pending_list_style = "unordered"
                pending_list_items.append(item_text)
                i += 1
                continue

            # ------------------------------------------------------------------
            # 6. Ordered list item  (1. text)
            # ------------------------------------------------------------------
            ol_match = re.match(r"^(\s*)\d+\.\s+(.+)", line)
            if ol_match:
                flush_paragraph()
                flush_table()
                item_text = ol_match.group(2).strip()

                if pending_list_style == "unordered":
                    flush_list()

                pending_list_style = "ordered"
                pending_list_items.append(item_text)
                i += 1
                continue

            # ------------------------------------------------------------------
            # 7. Blank line — paragraph boundary
            # ------------------------------------------------------------------
            if not stripped:
                # A blank line terminates an active list and an active paragraph.
                flush_list()
                flush_paragraph()
                i += 1
                continue

            # ------------------------------------------------------------------
            # 8. Everything else → paragraph / text block
            # ------------------------------------------------------------------
            flush_list()
            if in_table:
                flush_table()
            # Accumulate; consecutive non-blank lines merge into one text block.
            pending_paragraph.append(line)
            i += 1

        # Flush any remaining state at end-of-input.
        flush_list()
        flush_paragraph()
        flush_table()

        return blocks

    # ------------------------------------------------------------------

    def from_tool_result(self, result: dict) -> list[dict]:
        """Convert a tool output dict into blocks.

        Recognised keys (checked in priority order):

        - ``error`` (str) — yields a system message block at level ``"error"``.
        - ``text`` (str) — passed through ``from_markdown()``.
        - ``results`` (list[dict]) — each item rendered as a key-value block when
          it is a dict, or collected into an unordered list block when it is a
          scalar value.

        Unknown / empty results fall back to an empty list.

        Args:
            result: Tool output dict as returned by the ACT dispatcher.

        Returns:
            List of block dicts.
        """
        if not isinstance(result, dict):
            logger.warning("from_tool_result received non-dict: %r", type(result))
            return []

        blocks: list[dict] = []

        # Error case — surface immediately, skip other keys.
        if "error" in result:
            blocks.extend(self.from_system_message(str(result["error"]), level="error"))
            return blocks

        # Free-form text — parse as markdown.
        if "text" in result and result["text"]:
            blocks.extend(self.from_markdown(str(result["text"])))

        # Structured results list.
        if "results" in result and isinstance(result["results"], list):
            items = result["results"]
            if items:
                # Determine whether items are dicts or scalars.
                if all(isinstance(item, dict) for item in items):
                    # Render each dict as a key-value block.
                    for item in items:
                        pairs = [
                            {"key": str(k), "value": str(v)} for k, v in item.items()
                        ]
                        if pairs:
                            blocks.append({"type": "keyvalue", "pairs": pairs})
                else:
                    # Mixed or scalar list — render as an unordered list.
                    list_items = [str(item) for item in items]
                    blocks.append(
                        {"type": "list", "style": "unordered", "items": list_items}
                    )

        return blocks

    # ------------------------------------------------------------------

    def from_system_message(self, message: str, level: str = "info") -> list[dict]:
        """Wrap a system or error message in a single plain text block.

        Args:
            message: Human-readable message string.
            level: Severity level — ``"info"``, ``"warning"``, or ``"error"``.
                   Stored on the block for client-side styling; does not affect
                   parsing.

        Returns:
            A single-element list containing the text block.
        """
        return [
            {
                "type": "text",
                "content": message.strip() if message else "",
                "format": "plain",
                "level": level,
            }
        ]

    # ------------------------------------------------------------------

    def from_actions(self, actions: list[dict]) -> list[dict]:
        """Wrap a list of action-button dicts in an actions block.

        Each button dict should contain at minimum a ``label`` key.  Optional
        keys include ``skill`` (innate skill name to invoke) and ``payload``
        (arbitrary data forwarded to the skill).

        Args:
            actions: List of button descriptor dicts.

        Returns:
            A single-element list containing the actions block, or an empty list
            if ``actions`` is empty or not a list.
        """
        if not isinstance(actions, list) or not actions:
            return []
        return [{"type": "actions", "buttons": actions}]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Convenience: validate a block list (used in tests / debug)
    # ------------------------------------------------------------------

    KNOWN_TYPES: frozenset[str] = frozenset(
        {
            "text",
            "header",
            "code",
            "list",
            "table",
            "keyvalue",
            "image",
            "link",
            "divider",
            "actions",
            "carousel",
            # Rich render directive types
            "metric",
            "card",
            "chart",
            "progress",
            "timeline",
            # Layout & feedback types
            "columns",
            "section",
            "tabs",
            "container",
            "badge",
            "alert",
            "loading",
            "thought",
            "input",
            "select",
            "toggle",
            "form",
        }
    )

    def validate(self, blocks: list[dict]) -> list[str]:
        """Validate a list of blocks and return a list of human-readable errors.

        Useful during development and testing.  Production callers should not
        depend on this method.

        Args:
            blocks: Block list to validate.

        Returns:
            List of error strings.  Empty list means the blocks are valid.
        """
        errors: list[str] = []
        for idx, block in enumerate(blocks):
            prefix = f"Block[{idx}]"
            if not isinstance(block, dict):
                errors.append(f"{prefix}: not a dict (got {type(block).__name__})")
                continue
            if "type" not in block:
                errors.append(f"{prefix}: missing required 'type' key")
                continue
            btype = block["type"]
            if btype not in self.KNOWN_TYPES:
                errors.append(f"{prefix}: unknown type '{btype}'")
                continue

            # Type-specific field checks.
            if btype in ("text",) and "content" not in block:
                errors.append(f"{prefix}: text block missing 'content'")
            if btype == "header":
                if "content" not in block:
                    errors.append(f"{prefix}: header block missing 'content'")
                level = block.get("level")
                if not isinstance(level, int) or not (1 <= level <= 6):
                    errors.append(f"{prefix}: header 'level' must be int 1-6, got {level!r}")
            if btype == "code" and "content" not in block:
                errors.append(f"{prefix}: code block missing 'content'")
            if btype == "list":
                if block.get("style") not in ("unordered", "ordered"):
                    errors.append(
                        f"{prefix}: list 'style' must be 'unordered' or 'ordered', "
                        f"got {block.get('style')!r}"
                    )
                if not isinstance(block.get("items"), list):
                    errors.append(f"{prefix}: list block missing 'items' list")
            if btype == "table":
                if not isinstance(block.get("headers"), list):
                    errors.append(f"{prefix}: table block missing 'headers' list")
                if not isinstance(block.get("rows"), list):
                    errors.append(f"{prefix}: table block missing 'rows' list")
            if btype == "keyvalue" and not isinstance(block.get("pairs"), list):
                errors.append(f"{prefix}: keyvalue block missing 'pairs' list")
            if btype == "actions" and not isinstance(block.get("buttons"), list):
                errors.append(f"{prefix}: actions block missing 'buttons' list")
            if btype == "carousel" and not isinstance(block.get("slides"), list):
                errors.append(f"{prefix}: carousel block missing 'slides' list")
            # Rich render directive types
            if btype == "metric":
                if "label" not in block:
                    errors.append(f"{prefix}: metric block missing 'label'")
                if "value" not in block:
                    errors.append(f"{prefix}: metric block missing 'value'")
            if btype == "card" and "title" not in block:
                errors.append(f"{prefix}: card block missing 'title'")
            if btype == "chart":
                if not isinstance(block.get("series"), list):
                    errors.append(f"{prefix}: chart block missing 'series' list")
            if btype == "progress":
                if "label" not in block:
                    errors.append(f"{prefix}: progress block missing 'label'")
                if not isinstance(block.get("value"), (int, float)):
                    errors.append(f"{prefix}: progress 'value' must be a number")
            if btype == "timeline" and not isinstance(block.get("events"), list):
                errors.append(f"{prefix}: timeline block missing 'events' list")
            if btype == "columns" and not isinstance(block.get("columns"), list):
                errors.append(f"{prefix}: columns block missing 'columns' list")
            if btype == "section" and not isinstance(block.get("blocks"), list):
                errors.append(f"{prefix}: section block missing 'blocks' list")
            if btype == "badge" and "text" not in block:
                errors.append(f"{prefix}: badge block missing 'text'")
            if btype == "alert" and "message" not in block:
                errors.append(f"{prefix}: alert block missing 'message'")

        return errors
