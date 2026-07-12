"""
ReviewToolCallsAbility — Drill back into raw tool call data from a previous turn.

Lets the LLM re-read raw tool call records from earlier in the conversation when
the compact synthesis in Previous Turns doesn't include a specific detail it needs.
Returns all durable tool calls (excluding narration rows) within ±5 minutes of a
given timestamp. Rows are retained for 7 days by the decay-engine janitor.

All retrieval/formatting/error logic lives in ``ReviewWindowAbility``; this class
declares only the windowed ``tool_calls`` SELECT and the structured row shape.
"""

from typing import ClassVar

from configs.enums.param_key import Keys
from abilities._result import ToolResult
from abilities._review_window import ReviewWindowAbility
from models.tool_call import ToolCall

# Tool-call params summaries can be large; clip so one row stays a single readable
# line of structured JSON.
_PARAMS_SUMMARY_CHARS = 120


class ReviewToolCallsAbility(ReviewWindowAbility):
    SYSTEM = True

    def get_name(self) -> str:
        return "review_tool_calls"

    def get_summary(self) -> str:
        return "Retrieve raw tool call records within ±5 minutes of a timestamp to inspect details not captured in turn synthesis."

    def get_examples(self) -> list[str]:
        return [
            "what tools did you use at 2pm yesterday",
            "show me the raw output of the search call you made earlier",
            "can you review the tool calls from this morning",
            "what did the weather tool return at 10am",
            "I need to see the exact result from that memory recall",
            "show me what tools fired during our last conversation about the report",
            "review the tool calls from around 3:30pm today",
            "what happened in that tool call that returned an error",
        ]

    def get_search_tooltip(self) -> str:
        return "inspect past tool calls"

    def get_follow_up(self, tr: ToolResult) -> str:
        """Direct the model to the untruncated transcript for the same window."""
        anchor = tr.meta.get("anchor")
        if not anchor:
            return ""
        return (
            "Params are clipped to ~120 chars and full results are omitted. For the "
            "exact wording a user or assistant used in that window, call "
            f"`review_transcript(date_time={anchor})` to read the untruncated messages."
        )

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.date_time: {
                "type": "string",
                "description": (
                    "ISO timestamp to anchor the search "
                    "(e.g. 2026-04-07T14:30:00+00:00). Tool calls within ±5 minutes "
                    "of this time will be returned."
                ),
            },
        },
        "required": [Keys.date_time],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    # ── ReviewWindowAbility hooks ──────────────────────────────────────────────

    def _fetch(self, lo: str, hi: str, params: dict[str, object]) -> list[dict[str, object]]:
        """Excludes legacy ``tool_name='narration'`` rows.

        Narration as a separate tool_calls row was removed — mid-turn assistant
        text is no longer persisted at all (a turn emits one end message), so no
        new narration rows are written. The literal filter stays only to hide any
        such rows left in an existing DB from before that change; it is not a
        live tool name."""
        return (
            ToolCall.filter("created_at", lo, ">=")
            .filter("created_at", hi, "<=")
            .filter("tool_name", "narration", "!=")
            .order_by("created_at ASC, id ASC")
            .select("tool_name", "params", "result", "created_at")
        )

    def _row(self, rec: dict[str, object], ordinal: int) -> dict[str, object]:
        params_str = str(rec.get("params") or "{}")
        if len(params_str) > _PARAMS_SUMMARY_CHARS:
            params_str = params_str[:_PARAMS_SUMMARY_CHARS]
        return {
            "iter": ordinal,
            "ts": self._ts(rec.get("created_at")),
            "tool": rec.get("tool_name") or "unknown",
            "params_summary": params_str,
            "ok": _result_ok(rec.get("result")),
        }

    def _empty_hint(self, date_time: str, buffer: int) -> str:
        return (
            f"No tool calls found within ±{buffer} minutes of {date_time}. "
            "Try a different timestamp."
        )


def _result_ok(result: object) -> bool:
    """A recorded result is now a dispatcher envelope whose first line is
    ``[<tool>(status=…)]``. The call failed iff that first line carries
    ``status=error``."""
    first_line = str(result or "").splitlines()[0] if result else ""
    return "status=error" not in first_line
