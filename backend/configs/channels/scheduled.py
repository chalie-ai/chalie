# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ScheduledConfig — work-loop channel for a fired scheduled prompt.

The scheduler hands the result to a UserConfig turn on the user channel; that
turn is what surfaces to the user and gets episodically encoded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abilities._delegate import render_trail
from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor

from configs.channels._common import (
    DEFAULT_ALWAYS_AVAILABLE,
    DEFAULT_DISCOVERABLE,
    DELEGATE_INTERNAL_TOOLS,
    PATTERN_WRITE_TOOLS,
)

_SCHEDULED_SYSTEM_PROMPT = (
    "You are carrying out a single scheduled task on the user's behalf, given "
    "below. It was queued earlier to run at this time; the user is not present "
    "right now.\n\n"
    "Do the work the task asks for — use your tools to gather whatever you need "
    "(search, read, recall the user's memory, check the calendar, and so on). "
    "Work only from the task and what your tools return; never invent facts, "
    "URLs, or results.\n\n"
    "STOP RULE: the moment the task is done — or you know it cannot be — stop "
    "calling tools and write a concise, self-contained result. That result is "
    "handed straight to the user, so phrase it as the finished outcome of the "
    "task, not as a status update about yourself."
)

# The scheduled agent gets the same broad tool surface a user turn has so it can
# actually perform arbitrary scheduled work, minus the background-only blocks
# (pattern-write tools and the delegate-internal raw web tools), matching how
# UserConfig scopes its visibility.
_SCHEDULED_BLOCKED = PATTERN_WRITE_TOOLS | DELEGATE_INTERNAL_TOOLS


class ScheduledConfig(ProcessorConfig):
    """``broadcast_to=None`` — the work loop is silent; the return-path
    UserConfig turn is what reaches the UI."""

    def __init__(self, policy_channel: "ProcessorConfig.PolicyChannel") -> None:
        super().__init__(
            channel="scheduled",
            role="scheduled_worker",
            policy_channel=policy_channel,
            always_available=list(DEFAULT_ALWAYS_AVAILABLE),
            discoverable=list(DEFAULT_DISCOVERABLE),
            blocked=_SCHEDULED_BLOCKED,
            max_iterations=100,
            skip_transcript=False,  # persist the instruction (recoverability)
            skip_input_row=False,   # and give the act-trail a uid to key on
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp: "MessageProcessor") -> str:
        return ""

    def get_user_prompt(self, mp: "MessageProcessor") -> str:
        """Goal-driven user prompt: the scheduled instruction plus the trail."""
        parts = [f"Scheduled task:\n{mp._raw_input}"]
        trail = render_trail(mp)
        if trail:
            parts.append(trail)
        return "\n\n".join(parts)

    def get_system_prompt(self, mp: "MessageProcessor") -> str:
        return _SCHEDULED_SYSTEM_PROMPT
