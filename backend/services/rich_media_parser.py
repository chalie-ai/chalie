"""Rich-media segment parser.

A single pure function ``parse(content, tool_calls)`` converts a sanitised
assistant response (which may contain ``<span id='tool_N'>…</span>`` tags)
plus the turn's ``tool_calls`` rows into an ordered list of typed segments.

Segment shapes:
    {"type": "text", "content": "<safe HTML>"}
    {"type": "rich", "tag": "weather_1", "synthesis": "<safe HTML>", "payload": {…}}

The parser is called at the WS-send boundary and again on the thread-block
refresh path (``serialize_turn`` — the REST reads + WS refetch) — both against
persisted surfaces (``transcript.content`` + ``tool_calls.result``) so they
produce byte-identical output.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import cast

logger = logging.getLogger(__name__)


# Matches <span id='tool_name_N'>…</span> or <span id="tool_name_N">…</span>.
# Any additional span attributes are tolerated and ignored (``[^>]*``) so that
# historical persisted content — which this parser re-reads, not just fresh
# output — still splits cleanly even if it carries an attribute from a retired
# span variant.
# Group 1: tool name prefix (e.g. "weather")
# Group 2: ordinal string (e.g. "1")
# Group 3: inner synthesis text (may span multiple lines)
_TAG_RE = re.compile(
    r"""<span\s+id=['"]([a-z][a-z0-9_]*)_(\d+)['"][^>]*>(.*?)</span>""",
    re.DOTALL | re.IGNORECASE,
)


def parse(content: str, tool_calls: list[dict[str, object]]) -> list[dict[str, object]]:
    """Convert sanitised assistant text + tool_calls rows to a segment list."""
    if not content:
        return []

    segments: list[dict[str, object]] = []
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


def _append_leading_prose(content: str, cursor: int, match_start: int, segments: list[dict[str, object]]) -> None:
    if match_start > cursor:
        head = content[cursor:match_start].strip()
        if head:
            segments.append({"type": "text", "content": head})


def _append_tag_segment(match: re.Match[str], tool_calls: list[dict[str, object]], segments: list[dict[str, object]]) -> None:
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


def _enrich_payload(tool_name: str, payload: object, row: dict[str, object]) -> object:
    """Hand the payload to the matching ability so it can resolve runtime state.

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
        return ability.enrich_rich_payload(payload, cast("datetime | str | None", row.get("created_at")))
    except Exception as exc:
        logger.warning(
            "rich_media: enrich_rich_payload(%s) failed — falling back to raw payload: %s",
            tool_name, exc,
        )
        return payload


_SKILL_TAG_RE = re.compile(
    r"\A\[(?P<name>\w+)\([^)\n]*\)\]\n(?P<body>.*)\n\[end:(?P=name)\]\s*\Z",
    re.DOTALL,
)


def _unwrap_skill_tag(result: str) -> str:
    """Strip a ``[name(...)]\\n…\\n[end:name]`` outer wrapper if present.

    The list ability wraps its rich-media trailer in the canonical skill-output
    block so every list result has the same on-the-wire shape (plain and
    rich-media alike). The parser must see the inner ``{json}\\n\\n{instruction}``
    payload, so this strips the wrapper before splitting on the blank line.
    Other rich-media abilities don't wrap — for them this is a no-op.
    """
    if not isinstance(result, str):
        return result
    match = _SKILL_TAG_RE.match(result)
    return match.group("body") if match else result


def _find_payload(tag: str, tool_calls: list[dict[str, object]]) -> tuple[object, dict[str, object]] | None:
    """Scan tool_calls for the row whose rich-media trailer references this tag."""
    needle_single = f"<span id='{tag}'>"
    needle_double = f'<span id="{tag}">'

    for row in tool_calls:
        result = row.get("result") or ""
        if not isinstance(result, str):
            continue
        body = _unwrap_skill_tag(result)
        # Anchor to the trailer section — everything after the first blank line.
        # A span tag present only in the JSON payload body (before \n\n) must
        # not match; only trailer-section occurrences qualify.
        parts = body.split("\n\n", 1)
        if len(parts) < 2:
            continue
        trailer = parts[1]
        if needle_single in trailer or needle_double in trailer:
            return _extract_data(result), row

    return None


def _extract_data(result: str) -> object:
    """Extract the structured data portion from a tool result string.

    The tool serialises its payload as JSON, then appends ``\\n\\n`` followed
    by the rich-media instruction trailer.  We split on the first blank line
    and parse the head as JSON.

    Args:
        result: Full tool result string (data + trailer).

    Returns:
        Parsed JSON dict, or the raw head string if JSON parsing fails.
    """
    # Split on first double-newline to separate data from instruction trailer.
    # First strip any [name(...)]\n…\n[end:name] outer wrapper (list ability).
    body = _unwrap_skill_tag(result)
    head = body.split("\n\n", 1)[0].strip()
    try:
        return cast(object, json.loads(head))
    except ValueError:
        return head
