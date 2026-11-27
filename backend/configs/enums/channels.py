# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Channel — the transcript/telemetry namespace string scoping a conversation.

Every ``ProcessorConfig`` carries a ``channel``: the value written to (or
telemetry-tagged on) the ``transcript`` table's ``channel`` column. This enum
catalogues every channel the system currently defines, so callers can reference
them by name instead of repeating raw strings.

The ``external-agent`` channel is dynamic — one per registered external agent,
spelled ``external-agent:{name}``. ``EXTERNAL_AGENT`` holds the prefix; use
:meth:`for_external_agent` to build a concrete value.

A subagent's channel is dynamic too — its own, one per agent, spelled
``delegate:{type}:{agent_id}`` (:meth:`Channel.for_agent`). The ``DELEGATE_*``
values are the bare ``delegate:{type}`` each subagent kind's prompt is chosen
by.
"""

from __future__ import annotations

from enum import Enum


_DELEGATE_PREFIX = "delegate:"

# Every delegate channel — a kind's bare one or a subagent's own — sorts inside
# ``[low, high)``: the prefix, then the prefix with its ``:`` stepped to the
# next character (``delegate;``), so one index range reads them all.
DELEGATE_CHANNEL_SPAN: tuple[str, str] = (
    _DELEGATE_PREFIX,
    _DELEGATE_PREFIX[:-1] + chr(ord(_DELEGATE_PREFIX[-1]) + 1),
)


class Channel(str, Enum):
    # ── User-facing spines ───────────────────────────────────────────────────
    USER          = "user"
    DISCOVERY     = "discovery"
    SCHEDULE      = "schedule"

    # ── Reflection / subconscious ────────────────────────────────────────────
    SKILLS_BUILDING = "skills_building"
    MEMORY_HYGIENE  = "memory_hygiene"
    USER_SYNTHESIS  = "user_synthesis"

    # ── Compaction ───────────────────────────────────────────────────────────
    COMPACTION      = "compaction"

    # ── Delegates (tool-driven sub-loops) ────────────────────────────────────
    DELEGATE_PIM        = "delegate:pim"
    DELEGATE_THREAD_GIST = "delegate:thread_gist"
    DELEGATE_WEB_BROWSE  = "delegate:web_browse"
    DELEGATE_WEB_SEARCH  = "delegate:web_search"
    DELEGATE_VISION      = "delegate:vision"
    DELEGATE_CODE_AGENT  = "delegate:code_agent"
    DELEGATE_SUBTASK     = "delegate:subtask"
    DELEGATE_VOICE_CLEANUP = "delegate:voice_cleanup"

    # ── External agents (dynamic: external-agent:{name}) ─────────────────────
    EXTERNAL_AGENT = "external-agent"

    @classmethod
    def for_external_agent(cls, name: str) -> str:
        """Build the dynamic ``external-agent:{name}`` channel string."""
        return f"{cls.EXTERNAL_AGENT.value}:{name}"

    @classmethod
    def for_agent(cls, name: str, agent_id: int) -> str:
        """The channel subagent ``agent_id``, spawned by the delegate tool
        ``name``, runs its turns on: ``delegate:{name}:{agent_id}``."""
        return f"{_DELEGATE_PREFIX}{name}:{agent_id}"


# The delegate kinds a user may watch: the bare channels of the configs that
# each DelegateAbility tool runs its subagents on — every one writes a
# transcript of its own (vision, thread_gist and voice_cleanup write none, so
# there is nothing to show). Each subagent runs on its own channel under one
# of them (:meth:`Channel.for_agent`); its frames reach the wire addressed by
# that channel, and the threads read accepts no other delegate channel —
# whether the caller waits for the subagent or not.
WATCHABLE_DELEGATE_CHANNELS: frozenset[str] = frozenset({
    Channel.DELEGATE_WEB_SEARCH.value,
    Channel.DELEGATE_WEB_BROWSE.value,
    Channel.DELEGATE_PIM.value,
    Channel.DELEGATE_CODE_AGENT.value,
    Channel.DELEGATE_SUBTASK.value,
})


def delegate_agent_id(channel: str) -> int | None:
    """The subagent whose own channel ``channel`` is — the inverse of
    :meth:`Channel.for_agent` — or ``None`` for any other channel."""
    base, _, tail = channel.rpartition(":")
    if base not in WATCHABLE_DELEGATE_CHANNELS or not (tail.isascii() and tail.isdigit()) or str(int(tail)) != tail:
        return None
    return int(tail)


def delegate_type(channel: str) -> str | None:
    """The delegate tool a watchable delegate channel belongs to: a subagent's
    own ``delegate:{type}:{agent_id}``, or a bare ``delegate:{type}`` — where
    every subagent of that kind ran before each had its own, so those stored
    turns stay readable. ``None`` for any other channel."""
    if delegate_agent_id(channel) is not None:
        channel = channel.rpartition(":")[0]
    if channel not in WATCHABLE_DELEGATE_CHANNELS:
        return None
    return channel.removeprefix(_DELEGATE_PREFIX)
