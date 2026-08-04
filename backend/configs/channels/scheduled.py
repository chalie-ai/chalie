# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ScheduledConfig — the one shared channel for fired scheduled-prompt threads.

Each schedule is one growing thread on the ``schedule`` channel, distinguished by
``turn_id`` (§13.1). A fire — and every user reply — runs the standard
``MessageProcessor`` under this config: full tool surface, channel+turn-scoped
history, episodic encoding, and the five WS turn-state signals. The turn
self-surfaces in its own thread and the scheduler dock; there is no user-channel
relay (§13.9). The schedule row is the per-schedule definition — one config class,
no per-schedule subclass.
"""

from __future__ import annotations

from typing import ClassVar

from abilities.find_tools import FindToolsAbility
from abilities.image_preview import ImagePreviewAbility
from configs.channels._common import DEFAULT_ALWAYS_AVAILABLE
from configs.enums.channels import Channel
from configs.enums.config_type import ConfigTypeEnum
from configs.enums.policy_channel import PolicyChannel
from services.processor_config import ProcessorConfig

# The scheduled agent gets the same broad tool surface a user turn has so it can
# perform arbitrary scheduled work: it carries find_tools, so every DISCOVERABLE
# tool (including the web delegates) is reachable, exactly like UserConfig.


class ScheduledConfig(ProcessorConfig):
    """Config for every schedule thread — fires and replies alike."""

    BROADCASTS_STATE: ClassVar[bool] = True
    # The schedule thread is a rendered surface (its output is converted and
    # sanitized as HTML at persist time), so it carries the same HTML output
    # contract as the user channel.
    RENDERS_HTML: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(
            channel=Channel.SCHEDULE.value,
            role="user",
            # CHAT, not SUBCONSCIOUS: a fired schedule is work the user queued on
            # their own behalf, so it runs under the user's own policy rows —
            # where web_search/web_browse are seeded allow — exactly as the
            # tool-surface note above and the system prompt already promise.
            # SUBCONSCIOUS is the silent-housekeeping family that seeds the web
            # delegates deny; borrowing it here hard-blocked scheduled web work.
            policy_channel=PolicyChannel.CHAT,
            always_available=list(DEFAULT_ALWAYS_AVAILABLE),
            skip_transcript=False,
            skip_input_row=False,
            suppress_history=False,
            memory_seed=True,
            external_turn_id=True,
        )

    def type(self) -> ConfigTypeEnum:
        return ConfigTypeEnum.SCHEDULED

    @property
    def system_prompt(self) -> str:
        return f"""You are carrying out a scheduled task on the user's behalf, given below. It was queued earlier to run at this time.

Do the work the task asks for — use your tools to gather whatever you need (search, read, recall the user's memory, check the calendar, and so on). Work only from the task and what your tools return; never invent facts, URLs, or results.

STOP RULE: the moment the task is done — or you know it cannot be — stop calling tools and write a concise, self-contained result. Phrase it as the finished outcome of the task, not as a status update about yourself.

To show images to the user call `{FindToolsAbility.NAME}` and load the `{ImagePreviewAbility.NAME}` tool"""
