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


def resolve_tool_call_transcript_ids(assistant_transcript_id: int, conn) -> list[int]:
    """Return the transcript row IDs whose tool_calls feed the parser for one assistant turn.

    Tool_calls are stored against the user-input transcript row that triggered
    the ACT loop — not against the assistant output row.  This function queries
    the DB to find that user-input row for the given assistant turn: the most
    recent row with ``id < assistant_transcript_id`` that shares the same
    ``channel`` and has ``role = 'user'`` or ``role = 'subagent_return'``.

    Both the WS-send path (``api/websocket.py``) and the
    ``/conversation/recent`` refresh path (``api/conversation.py``) use this
    function so they are guaranteed to resolve the same transcript IDs.  If the
    storage rule ever changes (e.g. tool_calls keyed to the assistant row
    instead), updating this single function is sufficient — neither call site
    needs to change.

    Args:
        assistant_transcript_id: The ``transcript.id`` of the assistant output row.
        conn: An open SQLite connection (or compatible cursor) for the query.

    Returns:
        List containing the single user-input transcript ID for this turn, or an
        empty list if no preceding user row can be found (should not happen in
        normal operation — logged as a warning).
    """
    row = conn.execute(
        "SELECT t2.id FROM transcript t1 "
        "JOIN transcript t2 ON t2.channel = t1.channel "
        "  AND t2.id < t1.id "
        "  AND t2.role IN ('user', 'subagent_return') "
        "WHERE t1.id = ? "
        "ORDER BY t2.id DESC LIMIT 1",
        (assistant_transcript_id,),
    ).fetchone()
    if row is None:
        logger.warning(
            "resolve_tool_call_transcript_ids: no preceding user row for assistant_id=%s",
            assistant_transcript_id,
        )
        return []
    return [row[0]]

# Matches <span id='tool_name_N'>…</span> or <span id="tool_name_N">…</span>
# Group 1: tool name prefix (e.g. "weather")
# Group 2: ordinal string (e.g. "1")
# Group 3: inner synthesis text (may span multiple lines)
_TAG_RE = re.compile(
    r"""<span\s+id=['"]([a-z][a-z0-9_]*)_(\d+)['"][ \t]*>(.*?)</span>""",
    re.DOTALL | re.IGNORECASE,
)


def strip_spans(content: str | None) -> str | None:
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
        _append_leading_prose(content, cursor, match.start(), segments)
        _append_tag_segment(match, tool_calls, segments)
        cursor = match.end()

    tail = content[cursor:].strip()
    if tail:
        segments.append({"type": "text", "content": tail})

    if not segments and content.strip():
        segments.append({"type": "text", "content": content.strip()})

    return segments


def _append_leading_prose(content: str, cursor: int, match_start: int, segments: list) -> None:
    """Emit any prose that precedes a tag match as a text segment."""
    if match_start > cursor:
        head = content[cursor:match_start].strip()
        if head:
            segments.append({"type": "text", "content": head})


def _append_tag_segment(match: "re.Match", tool_calls: list[dict], segments: list) -> None:
    """Resolve a span tag to a rich or text segment and append it."""
    tool_name = match.group(1)
    ordinal = match.group(2)
    tag = f"{tool_name}_{ordinal}"
    synthesis = match.group(3).strip()

    matched = _find_payload(tag, tool_calls)
    if matched is not None:
        payload, row = matched
        payload = _enrich_payload(tool_name, payload, row)
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


def _enrich_payload(tool_name: str, payload, row: dict):
    """Hand the payload to the matching ability so it can resolve runtime state.

    The parser stays tool-agnostic — every ability declares its own
    ``enrich_rich_payload`` classmethod (defaults to identity). This is the
    single hook so cards needing live data (timer's wall-clock anchor, list's
    live items) can graft it on without the parser learning their names.

    A non-dict payload (i.e. JSON parse failed and ``_extract_data`` returned
    the raw string) is passed through unchanged — there is nothing to enrich.
    """
    if not isinstance(payload, dict):
        return payload
    try:
        from abilities._registry import AbilityRegistry
        ability = AbilityRegistry.get(tool_name)
    except Exception:
        return payload
    if ability is None:
        return payload
    try:
        return ability.enrich_rich_payload(payload, row)
    except Exception as exc:
        logger.warning(
            "rich_media: enrich_rich_payload(%s) failed — falling back to raw payload: %s",
            tool_name, exc,
        )
        return payload


def _find_payload(tag: str, tool_calls: list[dict]) -> tuple[dict | str, dict] | None:
    """Scan tool_calls for the row whose rich-media trailer references this tag.

    Matching is anchored to the trailer section (the part of the result string
    that follows the first ``\\n\\n`` separator) so that tool results whose
    *payload* JSON happens to contain a literal span tag — for example, a search
    snippet that scraped a page containing ``<span id='weather_1'>`` in its body
    — are never mistakenly paired.  The span tag must appear in the trailer, not
    anywhere in the full result string.

    All rich-media tool trailers follow the format::

        <json payload>

        This tool supports rich-media rendering. ... <span id='tag'>

    So the span tag is always in the trailer section.  A search snippet or
    document body containing a literal span would be part of the JSON payload
    (before the first ``\\n\\n``), and will never satisfy this check.

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
        # Anchor to the trailer section — everything after the first blank line.
        # A span tag present only in the JSON payload body (before \n\n) must
        # not match; only trailer-section occurrences qualify.
        parts = result.split("\n\n", 1)
        if len(parts) < 2:
            continue
        trailer = parts[1]
        if needle_single in trailer or needle_double in trailer:
            return _extract_data(result), row

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
    except ValueError:
        return head
