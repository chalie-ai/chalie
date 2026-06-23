"""Shared base for the single-pass compaction ProcessorConfig.

``ChatHistoryCompactionConfig`` describes a fixed processor shape: no tools, no
transcript writes, subconscious policy channel, thinking forced high, history
suppressed.

``CompactionConfig`` owns that shape once. The concrete config sets a single
``ClassVar`` — :attr:`SYSTEM_PROMPT_CLASS` — and inherits everything else.
"""

from __future__ import annotations

from typing import ClassVar, cast

from services.processor_config import ProcessorConfig


class CompactionConfig(ProcessorConfig):
    """Single-pass compaction processor: no tools, no transcript writes,
    thinking forced high so no fact a later step needs is dropped.

    Subclasses set :attr:`SYSTEM_PROMPT_CLASS` to the system-prompt class whose
    ``get_prompt()`` supplies the compaction instructions. A compaction request
    carries no tools, so the loop completes in a single send with no iteration.
    """

    thinking_mode: ClassVar[str] = "high"

    #: System-prompt class supplying this compactor's instructions. Subclass MUST set.
    SYSTEM_PROMPT_CLASS: ClassVar[type | None] = None

    def __init__(self) -> None:
        super().__init__(
            channel="compaction",
            role="compaction",
            policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
            always_available=[],
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp: object) -> str:
        return ""

    def get_user_prompt(self, mp: object) -> str:
        return cast("str", getattr(mp, "_raw_input", ""))

    def get_system_prompt(self, mp: object) -> str:
        return cast("str", cast("type", self.SYSTEM_PROMPT_CLASS)().get_prompt())
