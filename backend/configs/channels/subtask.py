# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Subtask configs — a background turn that runs with its caller's settings.

A subtask is an ordinary turn on the ``delegate:subtask`` channel, one turn per
subtask. It thinks exactly as the turn that started it: the same system prompt,
tools, permissions and provider, because each config here IS its caller's config
with only the routing and background posture changed (``DiscoveryConfig``'s
shape). They must stay subclasses: the prompt builder picks the user persona by
``isinstance(config, UserConfig)``, so a copy or a fresh class would lose it.

Every subtask turn is its own conversation (``turn_scoped_history``): it reads
none of the caller's history and none of another subtask's work, only its own
thread.
"""

from __future__ import annotations

from configs.channels.scheduled import ScheduledConfig
from configs.channels.user import UserConfig
from configs.enums.channels import Channel


def _as_subtask(config: UserConfig | ScheduledConfig, caller_channel: Channel) -> None:
    """Re-route *config* onto the subtask channel, keeping the caller's prompt
    assembly. Writes go to the subtask channel, the history is the subtask's
    own thread, and the turn id is always allocated here (a scheduled caller
    owns its schedule's ids; a subtask never does)."""
    object.__setattr__(config, "channel", Channel.DELEGATE_SUBTASK.value)
    object.__setattr__(config, "read_channel", None)
    object.__setattr__(config, "prompt_channel", caller_channel.value)
    object.__setattr__(config, "external_turn_id", False)
    object.__setattr__(config, "turn_scoped_history", True)


class UserSubtaskConfig(UserConfig):
    """A subtask started from the user's conversation: the user turn's settings
    on the subtask channel.

    Re-declared, not inherited (discovery's rule), because each is a user-facing
    trait the subtask must not carry: it has no live-state surface of its own
    (a watched delegate streams by channel instead), it pairs no rich-media
    cards (its reply goes to its caller, whose ordinals a card would collide
    with), and its spend is background — the user did not start it. It has no
    routing type, like every delegate, so nothing addresses it as a spine."""

    BROADCASTS_STATE = False
    RENDERS_CARDS = False
    USAGE_TYPE = "background"

    def __init__(self) -> None:
        super().__init__()
        _as_subtask(self, Channel.USER)

    def type(self) -> None:
        return None


class ScheduledSubtaskConfig(ScheduledConfig):
    """A subtask started from a scheduled task: the schedule turn's settings on
    the subtask channel. Same re-declarations as :class:`UserSubtaskConfig`
    (``ScheduledConfig`` already bills as background)."""

    BROADCASTS_STATE = False
    RENDERS_CARDS = False
    USAGE_TYPE = "background"

    def __init__(self) -> None:
        super().__init__()
        _as_subtask(self, Channel.SCHEDULE)

    def type(self) -> None:
        return None
