"""ThreadGistConfig — delegate channel that produces a one-sentence gist per thread.

Fired fire-and-forget on every received user message (workstream D). The MP
reads only that thread's user messages from the DB (no carried state), sends a
single delegate call, and the daemon target persists the resulting gist via
``ThreadGistService.upsert``. No tools, no act trail, no transcript row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor


_THREAD_GIST_SYSTEM_PROMPT = (
    "Create a terse one-line summary of what the user is discussing with the agent. "
    "Return ONLY the summary sentence — no preamble, no quotes, no markdown."
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
        """Derive the thread's user messages from the DB via trigger context."""
        from services.locale_service import CHAT_TIMESTAMP_FMT, format_date  # noqa: PLC0415
        from services.transcript_service import Transcript  # noqa: PLC0415

        trigger_channel = getattr(mp, "_trigger_channel", None)
        trigger_turn_id = getattr(mp, "_trigger_turn_id", None)
        if trigger_channel is None or trigger_turn_id is None:
            return ""

        rows = [
            r for r in Transcript.by_turn(trigger_channel, trigger_turn_id)
            if r.get("role") == "user"
        ]
        if not rows:
            return ""

        lines = ["# User Message Prompt", "## User Messages"]
        for r in rows:
            ts = format_date(cast("str", r.get("created_at")), CHAT_TIMESTAMP_FMT, for_ui=True) or ""
            content = cast("str", r.get("content") or "").replace("\n", " ").strip()
            lines.append(f"[{ts}] {content}")
        return "\n".join(lines)