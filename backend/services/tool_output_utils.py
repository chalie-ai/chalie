"""
Tool Output Utilities — Shared formatting, sanitization, and telemetry.

Extracted from ToolRegistryService and _CronToolWorker to eliminate
verbatim duplication across both classes.
"""

import re
import json
from typing import Any


# Strip patterns: remove action-like text from tool output so the LLM
# doesn't misinterpret tool results as action instructions.
STRIP_PATTERNS = [
    re.compile(r'\{[^}]*\}'),
    re.compile(
        r'\b(recall|memorize|associate|introspect)\s*\(',
        re.IGNORECASE,
    ),
    re.compile(r'ACTION\s*:', re.IGNORECASE),
]

MAX_OUTPUT_CHARS = 3000


def format_tool_result(result: Any) -> str:
    """Convert a tool result dict/str into plain text for LLM consumption.

    Handles three structured output patterns:
    - {"results": [...]} — search-style results with title/snippet/url
    - {"content": "..."} — page extraction with optional truncation
    - Generic dict — key/value pairs

    Args:
        result: Raw tool output (str, dict, or other)

    Returns:
        Formatted plain text string
    """
    if isinstance(result, str):
        return result

    if not isinstance(result, dict):
        return str(result)

    lines = []

    # Search-style results
    if "results" in result and isinstance(result["results"], list):
        results = result["results"]
        if not results:
            lines.append(result.get("message", "No results found."))
        else:
            for i, r in enumerate(results, 1):
                if isinstance(r, dict):
                    title = r.get("title", "")
                    snippet = r.get("snippet", "")
                    url = r.get("url", "")
                    lines.append(f"{i}. {title}")
                    if snippet:
                        lines.append(f"   {snippet}")
                    if url:
                        lines.append(f"   {url}")
                else:
                    lines.append(f"{i}. {r}")
        if result.get("count") is not None and result["count"] > 0:
            lines.append(f"\n{result['count']} results returned.")
        return "\n".join(lines)

    # Page extraction
    if "content" in result and isinstance(result["content"], str):
        content = result["content"]
        if result.get("error"):
            return f"Error: {result['error']}"
        if not content:
            return "No content extracted from page."
        parts = [content]
        if result.get("truncated"):
            parts.append(f"(truncated to {result.get('char_count', '?')} chars)")
        return "\n".join(parts)

    # Generic key/value
    for key, value in result.items():
        if key in ("budget_remaining",):
            continue
        if isinstance(value, (list, dict)):
            lines.append(f"{key}: {json.dumps(value, default=str)[:500]}")
        else:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)


def sanitize_tool_output(text: str) -> str:
    """Strip action-like patterns from tool output text.

    Prevents the LLM from misinterpreting tool results as action
    instructions (e.g. `recall(...)` appearing in web scrape results).
    """
    for pattern in STRIP_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def build_tool_telemetry(raw_telemetry: dict) -> dict:
    """Flatten client context telemetry into the tool contract format.

    Extracts location, time, locale, language, and device context from
    the raw ClientContextService output.

    Args:
        raw_telemetry: Raw dict from ClientContextService.get()

    Returns:
        Flattened telemetry dict suitable for tool container env vars
    """
    loc = raw_telemetry.get("location") or {}
    loc_name = raw_telemetry.get("location_name", "")
    city, country = "", ""
    if "," in loc_name:
        city, country = [p.strip() for p in loc_name.split(",", 1)]

    device = raw_telemetry.get("device") or {}

    result = {
        "lat": loc.get("lat"),
        "lon": loc.get("lon"),
        "location_name": raw_telemetry.get("location_name", ""),
        "city": city,
        "country": country,
        "time": raw_telemetry.get("local_time", ""),
        "locale": raw_telemetry.get("locale", ""),
        "language": raw_telemetry.get("language", ""),
    }

    # Device context — so tools can tailor output to user's device
    if device_class := device.get("class"):
        result["device_class"] = device_class
    if platform := device.get("platform"):
        result["platform"] = platform
    if "pwa" in device:
        result["pwa"] = device["pwa"]

    return result
