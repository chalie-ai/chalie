"""
ReviewToolCallsAbility — Drill back into raw tool call data from a previous turn.

Lets the LLM re-read raw tool call records from earlier in the conversation when
the compact synthesis in Previous Turns doesn't include a specific detail it needs.
Returns all tool calls (including ephemeral records) within ±5 minutes of a given
timestamp.
"""

import logging

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)


class ReviewToolCallsAbility(Ability):
    NAME = "review_tool_calls"
    SEARCH_TOOLTIP = "inspect past tool calls"
    SYSTEM = True
    SUMMARY = "Retrieve raw tool call records within ±5 minutes of a timestamp to inspect details not captured in turn synthesis."
    EXAMPLES = [
        "what tools did you use at 2pm yesterday",
        "show me the raw output of the search call you made earlier",
        "can you review the tool calls from this morning",
        "what did the weather tool return at 10am",
        "I need to see the exact result from that memory recall",
        "show me what tools fired during our last conversation about the report",
        "review the tool calls from around 3:30pm today",
        "what happened in that tool call that returned an error",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "date_time": {
                "type": "string",
                "description": (
                    "ISO timestamp to anchor the search "
                    "(e.g. 2026-04-07T14:30:00+00:00). Tool calls within ±5 minutes "
                    "of this time will be returned."
                ),
            },
        },
        "required": ["date_time"],
    }

    def run(self, params: dict) -> dict:
        date_time = (params.get("date_time") or "").strip()
        if not date_time:
            return {"text": _skill_tag("review_tool_calls", error="date-time-required")}

        try:
            from services.tool_call_service import ToolCallService
            records = ToolCallService().get_by_timerange(date_time, buffer_minutes=5)
        except Exception as e:
            logger.exception(f"[REVIEW TOOL CALLS] Query failed for date_time={date_time!r}: {e}")
            return {"text": _skill_tag("review_tool_calls", error=f"query-failed:{str(e)[:150]}")}

        if not records:
            return {
                "text": _skill_tag(
                    "review_tool_calls",
                    f"No tool calls found within ±5 minutes of {date_time}",
                    anchor=date_time,
                )
            }

        lines = [f"{len(records)} record(s) within ±5 min of {date_time}:\n"]
        for rec in records:
            tool_name = rec.get("tool_name", "unknown")
            params_str = rec.get("params", "{}")
            result = str(rec.get("result") or "")
            status_hint = "error" if result.lower().startswith("error") else "ok"
            from services.time_formatter_service import TimeFormatterService
            created = TimeFormatterService.local(rec.get("created_at"), fmt="%Y-%m-%d %H:%M:%S") \
                or str(rec.get("created_at", ""))[:19]
            lines.append(f"  [{created}] {tool_name} params={params_str} → {result} ({status_hint})")

        body = "\n".join(lines)
        return {"text": _skill_tag("review_tool_calls", body, anchor=date_time, count=len(records))}
