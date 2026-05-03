"""Rich-media segment parser.

A single pure function ``parse(content, tool_calls)`` converts a sanitised
assistant response (which may contain ``<span id='tool_N'>…</span>`` tags)
plus the turn's ``tool_calls`` rows into an ordered list of typed segments.

Segment shapes:
    {"type": "text", "content": "<safe HTML>"}
    {"type": "rich", "tag": "weather_1", "synthesis": "<safe HTML>", "payload": {…}}

The parser is called at the WS-send boundary and again on the
``/conversation/recent`` refresh path.  Both paths operate against the same
persisted surfaces (``transcript.content`` + ``tool_calls.result``) so they
produce byte-identical output.

Edge cases that result in a ``text`` segment (with a warning log):
- Orphan tag whose tool_call result cannot be found
- LLM forgets the closing ``</span>`` (non-greedy regex simply misses it)
- Tag prefix with no registered frontend module (frontend falls back to text)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Matches <span id='tool_name_N'>…</span> or <span id="tool_name_N">…</span>
# Group 1: tool name prefix (e.g. "weather")
# Group 2: ordinal string (e.g. "1")
# Group 3: inner synthesis text (may span multiple lines)
_TAG_RE = re.compile(
    r"""<span\s+id=['""]([a-z][a-z0-9_]*)_(\d+)['""][ \t]*>(.*?)</span>""",
    re.DOTALL | re.IGNORECASE,
)


def strip_spans(content: str) -> str:
    """Remove every ``<span id='name_N'>…</span>`` wrapper, keeping inner text.

    Used at the subagent → parent boundary as a defensive scrub: subagents are
    architecturally prevented from receiving the rich-media trailer (the
    dispatcher gates ordinal injection on ``channel == 'user'``), but if a
    subagent ever emits a stray span — via memorised prior turns, hallucination,
    or a future tool-trailer leak — we still strip it before the text reaches
    the parent. Idempotent: a string with no spans is returned unchanged.

    Args:
        content: Arbitrary text that may contain rich-media span wrappers.

    Returns:
        The same text with every matching span tag replaced by its inner
        synthesis text. Whitespace is preserved as-is so the caller can decide
        whether to re-strip.
    """
    if not content:
        return content
    return _TAG_RE.sub(lambda m: m.group(3), content)


def parse(content: str, tool_calls: list[dict]) -> list[dict[str, Any]]:
    """Convert sanitised assistant text + tool_calls rows to a segment list.

    Args:
        content: Post-sanitisation LLM response (``transcript.content``).
                 May contain ``<span id='tool_N'>…</span>`` tags.
        tool_calls: List of tool_call row dicts for this turn.  Must include
                    rows where ``ephemeral = 1`` so inline weather results
                    are visible.  Each dict should carry at least ``result``
                    (the full tool return string).

    Returns:
        Ordered list of segment dicts.  Never raises — edge cases produce
        ``text`` segments and emit a warning log entry.
    """
    if not content:
        return []

    segments: list[dict] = []
    cursor = 0

    for match in _TAG_RE.finditer(content):
        # Emit any leading prose before this tag as a text segment
        if match.start() > cursor:
            head = content[cursor:match.start()].strip()
            if head:
                segments.append({"type": "text", "content": head})

        tool_name = match.group(1)
        ordinal = match.group(2)
        tag = f"{tool_name}_{ordinal}"
        synthesis = match.group(3).strip()

        payload = _find_payload(tag, tool_calls)
        if payload is not None:
            segments.append({
                "type": "rich",
                "tag": tag,
                "synthesis": synthesis,
                "payload": payload,
            })
        else:
            logger.warning("rich_media: orphan tag %s — no matching tool_call result found", tag)
            if synthesis:
                segments.append({"type": "text", "content": synthesis})

        cursor = match.end()

    # Emit any trailing prose after the last tag
    tail = content[cursor:].strip()
    if tail:
        segments.append({"type": "text", "content": tail})

    # If nothing matched, return the full content as a single text segment
    if not segments and content.strip():
        segments.append({"type": "text", "content": content.strip()})

    return segments


def _find_payload(tag: str, tool_calls: list[dict]) -> dict | str | None:
    """Scan tool_calls for the row whose result contains the tag's instruction string.

    The instruction trailer the tool embeds contains the literal span tag, so
    we look for the exact substring ``<span id='tag'>`` or ``<span id="tag">``
    in each tool_call's ``result`` field.  First hit wins.

    Returns:
        Parsed payload dict (or raw string if JSON parse fails), or None if
        no matching row is found.
    """
    needle_single = f"<span id='{tag}'>"
    needle_double = f'<span id="{tag}">'

    for row in tool_calls:
        result = row.get("result") or ""
        if not isinstance(result, str):
            continue
        if needle_single in result or needle_double in result:
            return _extract_data(result)

    return None


def _extract_data(result: str) -> dict | str:
    """Extract the structured data portion from a tool result string.

    The tool serialises its payload as JSON, then appends ``\\n\\n`` followed
    by the rich-media instruction trailer.  We split on the first blank line
    and parse the head as JSON.

    Args:
        result: Full tool result string (data + trailer).

    Returns:
        Parsed JSON dict, or the raw head string if JSON parsing fails.
    """
    # Split on first double-newline to separate data from instruction trailer
    head = result.split("\n\n", 1)[0].strip()
    try:
        return json.loads(head)
    except (json.JSONDecodeError, ValueError):
        return head
