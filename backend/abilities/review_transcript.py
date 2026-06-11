"""
ReviewTranscriptAbility — Read back recent conversation transcript rows.

Lets the LLM re-read actual user/assistant messages from earlier in the
conversation when the compacted summary doesn't include a detail it needs
(e.g. a drafted email, a quoted price, exact wording the user approved).
Returns transcript rows within ±N minutes of a given timestamp.

When include_subagent_transcripts is true, rows from the 'subagent' channel
are included alongside the 'user' channel — useful for reviewing what a past
subagent task found or produced.

All retrieval/window/error logic lives in ``ReviewWindowAbility``; this class
declares only the clamped buffer, the windowed ``transcript`` SELECT, and the
structured row shape.
"""

from typing import ClassVar

from abilities._result import ToolResult
from abilities._review_window import ReviewWindowAbility

_DEFAULT_BUFFER_MINUTES = 5
_MIN_BUFFER_MINUTES = 1
_MAX_BUFFER_MINUTES = 30


class ReviewTranscriptAbility(ReviewWindowAbility):
    SYSTEM = True

    def get_name(self) -> str:
        return "review_transcript"

    def get_summary(self) -> str:
        return (
            "Retrieve recent conversation messages (user and assistant) within "
            "±N minutes of a timestamp to re-read details lost to compaction. "
            "Set include_subagent_transcripts=true to also search subagent task history."
        )

    def get_examples(self) -> list[str]:
        return [
            "what did I say about the email earlier",
            "what was the draft I approved at 3pm",
            "re-read what you said about the meeting",
            "what exactly did I ask you to send",
            "check what we discussed around 2pm today",
            "show me the subagent task results from this afternoon",
        ]

    def get_search_tooltip(self) -> str:
        return "review conversation history"

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "date_time": {
                "type": "string",
                "description": (
                    "ISO timestamp to anchor the search "
                    "(e.g. 2026-04-07T14:30:00+00:00). Transcript rows within "
                    "±buffer_minutes of this time will be returned."
                ),
            },
            "buffer_minutes": {
                "type": "integer",
                "description": (
                    f"Half-window size in minutes (default {_DEFAULT_BUFFER_MINUTES}, "
                    f"max {_MAX_BUFFER_MINUTES}). Increase to widen the search."
                ),
            },
            "include_subagent_transcripts": {
                "type": "boolean",
                "description": (
                    "When true, include rows from the subagent channel in addition "
                    "to the user channel. Use this to review what a past subagent "
                    "task found or produced. Defaults to false."
                ),
            },
        },
        "required": ["date_time"],
    }

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    # ── ReviewWindowAbility hooks ──────────────────────────────────────────────

    def _buffer(self, params: dict) -> int | ToolResult:
        """Parse ``buffer_minutes`` defensively and clamp to [1, 30].

        A non-int / unparseable value is a loud ``invalid-param`` naming the valid
        range (the old ``int(...)`` crashed the whole dispatch to
        ``unhandled-exception``)."""
        raw = params.get("buffer_minutes", _DEFAULT_BUFFER_MINUTES)
        try:
            minutes = int(raw)
        except (TypeError, ValueError):
            return ToolResult.err(
                f"buffer_minutes must be a whole number of minutes, got {raw!r}.",
                code="invalid-param",
                hint=(
                    f"pass an integer between {_MIN_BUFFER_MINUTES} and "
                    f"{_MAX_BUFFER_MINUTES} (default {_DEFAULT_BUFFER_MINUTES})."
                ),
            )
        return max(_MIN_BUFFER_MINUTES, min(_MAX_BUFFER_MINUTES, minutes))

    def _fetch(self, lo: str, hi: str, params: dict) -> list[dict]:
        """Transcript rows in the window from the user channel — plus the subagent
        channel when ``include_subagent_transcripts`` is truthy — oldest first."""
        from services.database_service import get_shared_db_service

        include_subagent = bool(params.get("include_subagent_transcripts", False))
        channels = ("user", "subagent") if include_subagent else ("user",)
        placeholders = ", ".join("?" for _ in channels)

        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, channel, role, content, created_at
                FROM transcript
                WHERE created_at BETWEEN ? AND ?
                  AND channel IN ({placeholders})
                ORDER BY id ASC
                """,
                (lo, hi, *channels),
            )
            columns = ("id", "channel", "role", "content", "created_at")
            rows = [dict(zip(columns, r)) for r in cursor.fetchall()]
            cursor.close()
        return rows

    def _row(self, rec: dict, ordinal: int) -> dict:
        # Content is NOT clipped: this tool exists so the model can re-read EXACT
        # wording lost to compaction (a drafted email, a quoted price, approved
        # phrasing). Clipping would defeat that purpose, and the window is already
        # bounded (±30 min max). Only newlines are flattened so each row stays a
        # single structured line.
        content = (rec.get("content") or "").replace("\n", " ")
        return {
            "iter": ordinal,
            "ts": self._ts(rec.get("created_at")),
            "channel": rec.get("channel") or "unknown",
            "role": rec.get("role") or "unknown",
            "content": content,
        }

    def _empty_hint(self, date_time: str, buffer: int) -> str:
        return (
            f"No transcript rows found within ±{buffer} minutes of {date_time}. "
            "Widen buffer_minutes or try a different timestamp."
        )
