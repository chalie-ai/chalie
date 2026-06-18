"""Shared base for the two single-pass compaction ProcessorConfigs.

``ToolChainCompactionConfig`` and ``ChatHistoryCompactionConfig`` were
line-for-line identical apart from which system-prompt class they return. Both
describe the same processor shape: one iteration, no tools, no transcript writes,
subconscious policy channel, thinking forced high, history suppressed.

``CompactionConfig`` owns that shared shape once. Each concrete config sets a
single ``ClassVar`` — :attr:`SYSTEM_PROMPT_CLASS` — and inherits everything else,
so the two keep their distinct names and behaviours while the duplication is
gone.
"""

from __future__ import annotations

from typing import ClassVar

from services.processor_config import ProcessorConfig


class CompactionConfig(ProcessorConfig):
    """Single-pass compaction processor: no tools, no transcript writes,
    thinking forced high so no fact a later step needs is dropped.

    Subclasses set :attr:`SYSTEM_PROMPT_CLASS` to the system-prompt class whose
    ``get_prompt()`` supplies the compaction instructions.
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
            max_iterations=1,
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp) -> str:
        return ""

    def get_user_prompt(self, mp) -> str:
        return mp._raw_input

    def get_system_prompt(self, mp) -> str:
        return self.SYSTEM_PROMPT_CLASS().get_prompt()
