# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""WebBrowseConfig — delegate channel for the ``web_browse`` tool.

Writes a real per-turn transcript row on its own ``delegate:web_browse``
channel so the turn uid is assigned and the delegate renders its own
act-trail across ACT iterations (do NOT set the two skip flags True). The
uid keys the per-run browser PageSession and the screenshot ledger; the
delegate runner reads the browser session directly to close it when the run
ends. Paired with ``WebBrowseAbility`` (abilities/web_browse.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from abilities.browser import BrowserAbility
from abilities.memory import MemoryAbility
from abilities.read import ReadAbility
from abilities.vision import VisionAbility
from abilities.web_fetch import WebFetchAbility

from configs.enums.channels import Channel
from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from configs.enums.policy_channel import PolicyChannel

_WEB_BROWSE_TOOLS: tuple[str, ...] = (
    BrowserAbility.NAME,
    WebFetchAbility.NAME,
    ReadAbility.NAME,
    VisionAbility.NAME,
)


class WebBrowseConfig(ProcessorConfig):
    """policy_channel is inherited from the caller; the user-facing permission
    check happens at the outer ``web_browse`` tool."""

    uses_delegate_provider: ClassVar[bool] = True

    def __init__(self, policy_channel: "PolicyChannel") -> None:
        super().__init__(
            channel=Channel.DELEGATE_WEB_BROWSE.value,
            role="web_browse",
            policy_channel=policy_channel,
            always_available=[*_WEB_BROWSE_TOOLS, MemoryAbility.NAME],
            skip_transcript=False,  # uid + own transcript row, or the
            skip_input_row=False,   # act-trail dies and the loop runs blind
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    @property
    def system_prompt(self) -> str:
        return """You are driving a real browser with Playwright to accomplish one goal, given below. You can open pages, read and search them, click and fill by visible text, and take a screenshot — every screenshot comes back already described, so it is how you see the page.

Accomplish the goal in the fewest steps possible. Do not linger, do not retry, do not waste cycles — the moment you can answer, stop and respond.

If you hit a problem or cannot accomplish the goal quickly, return an error instead of grinding at it; you will be invoked again with a clearer goal."""
