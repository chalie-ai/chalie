# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""VisionConfig — delegate channel that runs on the brain's Vision Provider.

``uses_vision_provider=True`` is what makes ``ProviderService._resolve`` read the
Vision Provider from the DB instead of the global selected provider;
``policy_channel`` is inherited from the caller. Paired with
``VisionAbility`` in ``abilities/vision.py``: the ability's
``describe_image()`` instantiates this config and calls
``MessageProcessor.process()``.
"""

from __future__ import annotations

from typing import ClassVar

from services.processor_config import ProcessorConfig


class VisionConfig(ProcessorConfig):
    """Single-shot, no-tools. policy_channel inherited from caller."""

    uses_vision_provider: ClassVar[bool] = True

    def __init__(self, policy_channel: "ProcessorConfig.PolicyChannel") -> None:
        super().__init__(
            channel="delegate:vision",
            role="vision",
            policy_channel=policy_channel,
            always_available=[],
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    @property
    def system_prompt(self) -> str:
        return """If the user input contains questions or remarks, answer them directly; otherwise return a detailed description of what you see in the image. Be rich in detail — people, vehicles, animals, resemblances, colours, shapes, and any visible text."""
