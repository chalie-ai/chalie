"""ThreadGistConfig — delegate channel that produces a terse topical thread label.

Fired once when a turn first grows into a thread (the first reply past its
settle0). The MP reads ONLY two non-assistant messages from the DB — the thread's
opening message and the first message beyond settle0 (no carried state) — sends a
single delegate call, and the daemon target persists the label via
``ThreadGistService.upsert``. No tools, no act trail, no transcript row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor


_THREAD_GIST_SYSTEM_PROMPT = (
    "Name the subject matter of the two messages below in a very terse 3-5 word "
    "topical label. Be direct and expressive but terse. Name the topic itself — do "
    'NOT use framing like "the user asked" or "the user wants" '
    '(e.g. "Mac Mini Research", "HN Post Feedback"). Return ONLY the label — no '
    "preamble, no quotes, no markdown, no trailing punctuation."
)


class ThreadGistConfig(ProcessorConfig):
    """Delegate channel for per-thread gist generation."""

    uses_delegate_provider: ClassVar[bool] = True
    thinking_mode: ClassVar[str] = "low"

    def __init__(self) -> None:
        super().__init__(
            channel="delegate:thread_gist",
            role="thread_gist",
            policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
            always_available=[],
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp: "MessageProcessor") -> str:
        return ""

    def get_system_prompt(self, mp: "MessageProcessor") -> str:
        return _THREAD_GIST_SYSTEM_PROMPT

    def get_user_prompt(self, mp: "MessageProcessor") -> str:
        """Exactly two non-assistant messages from the DB via trigger context — the
        thread's opening message and the first message beyond settle0."""
        from services.locale_service import CHAT_TIMESTAMP_FMT, format_date  # noqa: PLC0415
        from services.transcript_service import Transcript  # noqa: PLC0415

        trigger_channel = getattr(mp, "_trigger_channel", None)
        trigger_turn_id = getattr(mp, "_trigger_turn_id", None)
        if trigger_channel is None or trigger_turn_id is None:
            return ""

        rows = [
            r for r in Transcript.by_turn(trigger_channel, trigger_turn_id)
            if r.get("role") != "assistant"
        ]
        if not rows:
            return ""

        settle = Transcript.settle0(trigger_channel, trigger_turn_id)
        beyond = next(
            (r for r in rows if settle is not None and cast("int", r.get("id")) > settle),
            None,
        )
        picked = [rows[0]] + ([beyond] if beyond is not None and beyond is not rows[0] else [])

        lines = ["# User Message Prompt", "## User Messages"]
        for r in picked:
            ts = format_date(cast("str", r.get("created_at")), CHAT_TIMESTAMP_FMT, for_ui=True) or ""
            content = cast("str", r.get("content") or "").replace("\n", " ").strip()
            lines.append(f"[{ts}] {content}")
        return "\n".join(lines)
