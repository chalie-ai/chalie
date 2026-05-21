"""
ReviewTranscriptAbility — Read back recent conversation transcript rows.

Lets the LLM re-read actual user/assistant messages from earlier in the
conversation when the compacted summary doesn't include a detail it needs
(e.g. a drafted email, a quoted price, exact wording the user approved).
Returns transcript rows within ±N minutes of a given timestamp.

When include_subagent_transcripts is true, rows from the 'subagent' channel
are included alongside the 'user' channel — useful for reviewing what a past
subagent task found or produced.
"""

import logging

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)

_DEFAULT_BUFFER_MINUTES = 5
_MAX_BUFFER_MINUTES = 30
_MAX_CONTENT_LENGTH = 500


class ReviewTranscriptAbility(Ability):
    NAME = "review_transcript"
    SEARCH_TOOLTIP = "review conversation history"
    SYSTEM = True
    SUMMARY = (
        "Retrieve recent conversation messages (user and assistant) within "
        "±N minutes of a timestamp to re-read details lost to compaction. "
        "Set include_subagent_transcripts=true to also search subagent task history."
    )
    EXAMPLES = [
        "what did I say about the email earlier",
        "what was the draft I approved at 3pm",
        "re-read what you said about the meeting",
        "what exactly did I ask you to send",
        "check what we discussed around 2pm today",
        "show me the subagent task results from this afternoon",
    ]
    INPUT_SCHEMA = {
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
    TIMEOUT = 10

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        date_time = (params.get("date_time") or "").strip()
        if not date_time:
            return {"text": _skill_tag("review_transcript", error="date-time-required")}

        buffer = min(
            int(params.get("buffer_minutes", _DEFAULT_BUFFER_MINUTES)),
            _MAX_BUFFER_MINUTES,
        )
        include_subagent = bool(params.get("include_subagent_transcripts", False))

        try:
            rows = _query_transcript_window(date_time, buffer, include_subagent)
        except Exception as exc:
            logger.error(
                "[REVIEW TRANSCRIPT] Query failed for date_time=%r: %s",
                date_time, exc, exc_info=True,
            )
            return {"text": _skill_tag("review_transcript", error=f"query-failed:{str(exc)[:150]}")}

        if not rows:
            return {
                "text": _skill_tag(
                    "review_transcript",
                    f"No transcript rows found within ±{buffer} minutes of {date_time}",
                    anchor=date_time,
                    buffer=buffer,
                ),
            }

        lines = [f"{len(rows)} message(s) within ±{buffer} min of {date_time}:\n"]
        for row in rows:
            lines.append(_format_row(row))

        body = "\n".join(lines)
        return {
            "text": _skill_tag(
                "review_transcript", body,
                anchor=date_time, buffer=buffer, count=len(rows),
            ),
        }


def _query_transcript_window(
    date_time: str,
    buffer_minutes: int,
    include_subagent: bool = False,
) -> list[dict]:
    """Fetch transcript rows within ±buffer_minutes of date_time.

    Args:
        date_time: ISO timestamp string used as the search anchor.
        buffer_minutes: Half-window in minutes around the anchor.
        include_subagent: When True, include rows from the 'subagent' channel
            in addition to the default 'user' channel.
    """
    from services.database_service import get_shared_db_service
    from services.time_utils import parse_utc

    anchor = parse_utc(date_time)
    from datetime import timedelta
    lo = (anchor - timedelta(minutes=buffer_minutes)).isoformat()
    hi = (anchor + timedelta(minutes=buffer_minutes)).isoformat()

    db = get_shared_db_service()
    with db.connection() as conn:
        cursor = conn.cursor()
        if include_subagent:
            cursor.execute(
                """
                SELECT id, channel, role, content, created_at
                FROM transcript
                WHERE created_at BETWEEN ? AND ?
                  AND channel IN ('user', 'subagent')
                ORDER BY id ASC
                """,
                (lo, hi),
            )
        else:
            cursor.execute(
                """
                SELECT id, channel, role, content, created_at
                FROM transcript
                WHERE created_at BETWEEN ? AND ?
                  AND channel = 'user'
                ORDER BY id ASC
                """,
                (lo, hi),
            )
        columns = ("id", "channel", "role", "content", "created_at")
        rows = [dict(zip(columns, r)) for r in cursor.fetchall()]
        cursor.close()

    return rows


def _format_row(row: dict) -> str:
    """Format a single transcript row for LLM consumption."""
    from services.time_formatter_service import TimeFormatterService

    ts = TimeFormatterService.local(row.get("created_at"), fmt="%Y-%m-%d %H:%M:%S") \
        or str(row.get("created_at", ""))[:19]
    role = row.get("role", "unknown")
    content = (row.get("content") or "").strip()
    if len(content) > _MAX_CONTENT_LENGTH:
        content = content[:_MAX_CONTENT_LENGTH] + "..."
    content_oneline = content.replace("\n", " ")
    return f"  [{ts}] {role}: {content_oneline}"
