"""ThreadGistConfig — delegate channel that produces a terse topical thread label.

Fired once when a turn first grows into a thread (the first reply past its
settle0). The MP reads ONLY two non-assistant messages from the DB — the thread's
opening message and the first message beyond settle0 (no carried state) — sends a
single delegate call, and the daemon target persists the label via
``GistService.upsert`` (the ``ThreadGist`` model). No tools, no act trail, no
transcript row.
"""

from __future__ import annotations

from typing import ClassVar

from configs.enums.channels import Channel
from configs.enums.policy_channel import PolicyChannel
from services.processor_config import ProcessorConfig


class ThreadGistConfig(ProcessorConfig):
    """Delegate channel for per-thread gist generation."""

    uses_delegate_provider: ClassVar[bool] = True
    thinking_mode: ClassVar[str] = "none"

    def __init__(self) -> None:
        super().__init__(
            channel=Channel.DELEGATE_THREAD_GIST.value,
            role="thread_gist",
            policy_channel=PolicyChannel.SUBCONSCIOUS,
            always_available=[],
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    @property
    def system_prompt(self) -> str:
        return """Name the subject matter of the two messages below in a very terse 3-5 word topical label. Be direct and expressive but terse. Name the topic itself — do NOT use framing like "the user asked" or "the user wants" (e.g. "Mac Mini Research", "HN Post Feedback"). Your entire reply must be the label itself — no reasoning or thinking-out-loud, no preamble, no quotes, no markdown, no trailing punctuation."""
