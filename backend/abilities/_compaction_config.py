"""Shared base for the single-pass compaction ProcessorConfig.

``ChatHistoryCompactionConfig`` describes a fixed processor shape: no tools, no
transcript writes, subconscious policy channel, thinking forced high, history
suppressed.

``CompactionConfig`` owns that shape once. The concrete config overrides the
inherited :attr:`~services.processor_config.ProcessorConfig.system_prompt`
property with its compaction-instruction literal and inherits everything else.
"""

from __future__ import annotations

from typing import ClassVar

from configs.enums.channels import Channel
from configs.enums.policy_channel import PolicyChannel
from services.processor_config import ProcessorConfig


class CompactionConfig(ProcessorConfig):
    """Single-pass compaction processor: no tools, no transcript writes,
    thinking forced high so no fact a later step needs is dropped.

    Subclasses override :attr:`system_prompt` (inherited from ``ProcessorConfig``)
    with the literal that supplies the compaction instructions. A compaction
    request carries no tools, so the loop completes in a single send with no
    iteration.
    """

    thinking_mode: ClassVar[str] = "high"

    def __init__(self) -> None:
        super().__init__(
            channel=Channel.COMPACTION.value,
            role="compaction",
            policy_channel=PolicyChannel.SUBCONSCIOUS,
            always_available=[],
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            memory_seed=False,
        )
