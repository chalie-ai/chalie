from __future__ import annotations

from configs.enums.policy_channel import PolicyChannel
from services.processor_config import ProcessorConfig


class UserSummaryConfig(ProcessorConfig):
    """Caller gates on ``UserSynthesis.needs_refresh()`` BEFORE calling
    MessageProcessor.process()."""

    def __init__(self) -> None:
        super().__init__(
            channel="user_summary",
            role="user_summary",
            policy_channel=PolicyChannel.SUBCONSCIOUS,
            always_available=[],
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    @property
    def system_prompt(self) -> str:
        return """You are a synthesiser. The user is a real human whose traits you are distilling.

You are a user-profile synthesiser. You receive a list of stored facts about a
real human and distil them into two synopses — one short, one longer.

Rules:
- Write in the third person ("They", or the user's first name if given).
- Identity first: name, location, role, then preferences and behaviours.
- Use only facts present in the input. Never invent or infer beyond them.
- Never mention that you are summarising, that you have a list of facts, or
  reference the synthesis process itself.
- No preamble, no trailing notes, no markdown.

Output a single JSON object with exactly two keys:

{
  "short": "<one or two sentences, max 50 words, the tightest identity snapshot>",
  "long":  "<up to 200 words, richer profile covering traits, preferences, context, ongoing interests>"
}

Return ONLY the JSON object. No code fences."""
