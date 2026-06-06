# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""VisionConfig — the delegate channel that runs on the brain's Vision Provider.

The typed ``ProcessorConfig`` for the ``vision`` delegate tool (TKT-838). Paired
with ``VisionAbility`` in ``abilities/vision.py``: the ability's
``describe_image()`` core instantiates this config and calls
``MessageProcessor.process()``. ``uses_vision_provider=True`` is what makes
``Providers._resolve`` read the Vision Provider from the DB instead of the global
selected provider; ``policy_channel`` is inherited from the caller.
"""

from __future__ import annotations

import base64
from typing import ClassVar

from services.processor_config import ProcessorConfig

_VISION_SYSTEM_PROMPT = (
    "If the user input contains questions or remarks, answer them directly; "
    "otherwise return a detailed description of what you see in the image. Be rich "
    "in detail — people, vehicles, animals, resemblances, colours, shapes, and any "
    "visible text."
)


class VisionConfig(ProcessorConfig):
    """Single-shot, no-tools delegate config that runs on the vision provider.
    policy_channel is inherited from the caller."""

    uses_vision_provider: ClassVar[bool] = True

    def __init__(self, policy_channel: "ProcessorConfig.POLICY_CHANNEL") -> None:
        super().__init__(
            channel="delegate:vision",
            role="vision",
            policy_channel=policy_channel,
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=1,
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp) -> str:
        return ""

    def get_system_prompt(self, mp) -> str:
        return _VISION_SYSTEM_PROMPT

    def get_user_prompt(self, mp) -> str:
        return mp._raw_input

    def get_image(self, mp) -> "dict | None":
        meta = mp._metadata or {}
        path = meta.get("image_path")
        if not path:
            return None
        with open(path, "rb") as fh:
            data = base64.b64encode(fh.read()).decode()
        return {"data": data, "mime_type": meta.get("mime_type") or "image/png"}
