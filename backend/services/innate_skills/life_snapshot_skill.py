"""
Life Snapshot Skill — On-demand cross-capability day synthesis.

Reuses morning_brief and evening_brief building blocks.
"""

import logging
from datetime import timedelta

from services.time_utils import utc_now

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "life_snapshot",
    "description": (
        "Get a snapshot of the user's day — today's calendar events, "
        "unanswered emails, and who they're meeting. Use when the user "
        "asks \"what's going on?\", \"catch me up\", \"what's my day?\", "
        "or similar."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "include_tomorrow": {
                "type": "boolean",
                "description": "Include tomorrow's events too.",
            },
        },
        "required": [],
    },
}


def handle_life_snapshot(topic: str, params: dict) -> str:
    """Return a cross-capability day snapshot."""
    now = utc_now()
    sections = [_calendar(now), _people(now), _email(now)]
    if params.get("include_tomorrow"):
        sections.append(_tomorrow(now))
    return "[LIFE SNAPSHOT]\n\n" + "\n\n".join(filter(None, sections))


def _calendar(now) -> str:
    try:
        from capabilities.morning_brief import (
            _fetch_today_events, _build_calendar_section,
        )
        return _build_calendar_section(_fetch_today_events(now))
    except Exception:
        return ""


def _people(now) -> str:
    try:
        from capabilities.morning_brief import (
            _fetch_today_events, _build_people_section,
        )
        return _build_people_section(_fetch_today_events(now))
    except Exception:
        return ""


def _email(now) -> str:
    try:
        from capabilities import load_capabilities
        cap = load_capabilities().get("imap")
        if not (cap and cap.is_connected()):
            return ""
        fn = {t["name"]: t["handler"]
              for t in cap.get_tools()}.get("imap_search_email")
        if not fn:
            return ""
        since = (now - timedelta(days=2)).strftime("%Y-%m-%d")
        emails = fn(None, {"date_from": since, "unanswered": True,
                           "triage": "actionable", "limit": 8}
                    ).get("emails", [])
        if not emails:
            return ""
        lines = [f"  - {e.get('from_name', '?')}: \"{e.get('subject', '?')}\""
                 for e in emails]
        return f"Unanswered emails ({len(emails)}):\n" + "\n".join(lines)
    except Exception:
        return ""


def _tomorrow(now) -> str:
    try:
        from capabilities.evening_brief import _build_tomorrow_section
        return _build_tomorrow_section(now)
    except Exception:
        return ""
