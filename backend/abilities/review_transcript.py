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

from datetime import datetime
from typing import ClassVar, cast

from configs.enums.param_key import Keys
from abilities._review_window import ReviewWindowAbility
from configs.enums.channels import Channel
from contracts.params.param_bag import ParamBag
from contracts.params.review_transcript_params_bag import (
    DEFAULT_BUFFER_MINUTES,
    MAX_BUFFER_MINUTES,
    ReviewTranscriptParamsBag,
)


class ReviewTranscriptAbility(ReviewWindowAbility[ReviewTranscriptParamsBag]):
    PARAMS: ClassVar[type[ParamBag] | None] = ReviewTranscriptParamsBag
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

    _PARAMETERS: "ClassVar[dict[str, object]]" = {
        "type": "object",
        "properties": {
            Keys.date_time: {
                "type": "string",
                "description": (
                    "ISO timestamp to anchor the search "
                    "(e.g. 2026-04-07T14:30:00+00:00). Transcript rows within "
                    "±buffer_minutes of this time will be returned."
                ),
            },
            Keys.buffer_minutes: {
                "type": "integer",
                "description": (
                    f"Half-window size in minutes (default {DEFAULT_BUFFER_MINUTES}, "
                    f"max {MAX_BUFFER_MINUTES}). Increase to widen the search."
                ),
            },
            Keys.include_subagent_transcripts: {
                "type": "boolean",
                "description": (
                    "When true, include rows from the subagent channel in addition "
                    "to the user channel. Use this to review what a past subagent "
                    "task found or produced. Defaults to false."
                ),
            },
        },
        "required": [Keys.date_time],
    }

    def get_parameters(self) -> "dict[str, object]":
        return self._PARAMETERS

    # ── ReviewWindowAbility hooks ──────────────────────────────────────────────

    def _buffer(self, params: ReviewTranscriptParamsBag) -> int:
        return params.buffer_minutes

    def _bound(self, moment: datetime) -> str:
        # transcript.created_at takes the schema default datetime('now') — naive
        # UTC, space-separated — unlike tool_calls, which stores isoformat(). The
        # base's ISO-T default would match no row at all.
        return moment.strftime("%Y-%m-%d %H:%M:%S")

    def _fetch(self, lo: str, hi: str, params: ReviewTranscriptParamsBag) -> "list[dict[str, object]]":
        from models.transcript import Transcript

        channels = (
            [Channel.USER.value, "subagent"]
            if params.include_subagent_transcripts
            else [Channel.USER.value]
        )
        return (
            Transcript.filter_in("channel", channels)
            .filter("created_at", lo, ">=")
            .filter("created_at", hi, "<=")
            .order_by("created_at ASC, id ASC")
            .select("channel", "role", "content", "created_at")
        )

    def _row(self, rec: "dict[str, object]", ordinal: int) -> "dict[str, object]":
        # Content is NOT clipped: this tool exists so the model can re-read EXACT
        # wording lost to compaction (a drafted email, a quoted price, approved
        # phrasing). Clipping would defeat that purpose, and the window is already
        # bounded (±30 min max). Only newlines are flattened so each row stays a
        # single structured line.
        content = cast(str, rec.get("content") or "").replace("\n", " ")
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
