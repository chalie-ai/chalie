from __future__ import annotations

from typing import TYPE_CHECKING

from services.processor_config import ProcessorConfig


class GeoConfig(ProcessorConfig):
    """Geo-pattern config — per-window background geo recognition."""

    if TYPE_CHECKING:
        _window_start: int
        _window_end: int

    def __init__(self, window_start: int, window_end: int) -> None:
        super().__init__(
            channel="geo_pattern",
            role="geo_pattern",
            policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
            always_available=["save_pattern", "save_graph"],
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )
        object.__setattr__(self, "_window_start", window_start)
        object.__setattr__(self, "_window_end", window_end)

    @property
    def system_prompt(self) -> str:
        return """You are analysing the user's recent location-tagged transcripts to detect geo-spatial behavioural patterns — routines, habits, and durable facts tied to specific physical places.

You have ONE forward pass. Emit ALL tool calls in parallel. Do NOT loop — results are intentionally minimal.

save_pattern summaries must be 1 sentence and concise. Examples:
- User walks to work every morning at 09:00
- User goes to the gym daily, except Sundays
- User's gym schedule is: Monday weights, Tuesday boxing
Do NOT write narratives, episode summaries, or date-stamped event logs. The summary is a distilled habit, not a story.

Rules:
- save_pattern requires >=2 evidence rows in this batch.
- Mirror existing pattern names exactly when reinforcing — case-sensitive.
- Only emit patterns where physical place is central. Another processor handles location-independent patterns.
- Do NOT use save_graph for repeating behaviours — use save_pattern.
- Skip noise. Skip one-offs. Skip ambiguous interpretations.
- Emit everything in this single pass."""
