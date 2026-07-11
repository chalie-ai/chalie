# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PimConfig — delegate channel for the ``pim`` tool.

Mirrors WebSearchConfig field-for-field: a focused delegate loop that fetches
live data from email, calendar, contacts, and memory. Paired with
:Pclass:`PimAbility` (abilities/pim.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from configs.enums.channels import Channel
from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from configs.enums.policy_channel import PolicyChannel


class PimConfig(ProcessorConfig):
    """``policy_channel`` is supplied by the caller (inherited from whoever
    invoked the tool) rather than hardcoded."""

    uses_delegate_provider: ClassVar[bool] = True

    # Pin thinking to LOW (the floor — "no thinking flag" at the provider) so a
    # user's persisted high/medium override never leaks a thinking turn into the
    # focused PIM loop. resolve_thinking_mode() gives this config pin priority
    # over the override and the gate-computed level.
    thinking_mode: ClassVar[str] = "low"

    def __init__(self, policy_channel: "PolicyChannel") -> None:
        super().__init__(
            channel=Channel.DELEGATE_PIM.value,
            role="pim",
            policy_channel=policy_channel,
            always_available=["email", "calendar", "contacts", "memory"],
            skip_transcript=False,  # write a delegate-channel transcript row so
            skip_input_row=False,   # _setup assigns the uid the act-trail needs
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    @property
    def system_prompt(self) -> str:
        return (
            "You are Chalie's Personal Information Manager. Execute the user's "
            "instruction using your tools: email, calendar, contacts, memory. Calendar "
            "and email are always fetched live (never from a cache). Your final answer "
            "MUST list each tool you called and its result, and be terse. Explicitly "
            "state what succeeded and what failed — never blur a partial failure into a "
            "success. If a calendar, contacts, or email action fails because no mail "
            "provider is connected, tell the user exactly, verbatim: \"You currently do "
            "not have a connected calendar thus I can't perform this action. You need to "
            "connect a mail provider.\""
        )