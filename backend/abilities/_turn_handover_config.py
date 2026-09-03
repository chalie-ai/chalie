"""TurnHandoverConfig — single-pass turn handover, fired just before compaction.

Subclass of :class:`CompactionConfig` that overrides only the system prompt
with the handover instruction literal. Inherits everything else (no tools, no
transcript writes, thinking forced high, history suppressed) from the base.
"""

from __future__ import annotations

from abilities._compaction_config import CompactionConfig


class TurnHandoverConfig(CompactionConfig):
    """Single-pass turn handover. No tools, no transcript writes, thinking
    forced high so the model reasons hard about what continuity to preserve."""

    @property
    def system_prompt(self) -> str:
        return """Summarize this turn's messages and tool results into a hand-over that will carry you forward when you resume. Trim heavily — keep only the data you need to continue this task. Include verbatim any identifiers needed to continue: file paths, URLs, commands, IDs, key values — never paraphrase them. Acknowledge prose, failed tool calls, and pivots in terse bullets so you do not retry dead ends. Keep the hand-over to a maximum of 3 paragraphs; the shorter the better. CRITICAL: you MUST end with a list of what is pending and your current plan to accomplish it. Write in first person — this hand-over will be provided to you when you resume."""
