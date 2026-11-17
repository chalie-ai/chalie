# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test — the shared HTML output-format contract lands in the system
prompt of every channel that renders to a human (``ProcessorConfig.RENDERS_HTML``)
but a subagent's own, and only those channels; a subagent is told to reply in
plain text instead.

Before this change the contract was baked as literal text at the tail of
``UserConfig.system_prompt``. The ``schedule`` channel is exactly as much a
rendered, user-visible surface as ``user`` — its assistant output is converted
and sanitized as HTML at persist time (``MessageProcessor._format``, gated on
``RENDERS_HTML``)
— but it never got the matching prompt guidance, so a fired schedule's response
came back as raw markdown on an HTML surface. The fix hoists the contract into
``PromptService._RESPONSE_FORMAT`` and appends it in
``PromptService.system_prompt()`` whenever ``config.RENDERS_HTML`` is set;
``UserConfig`` and ``ScheduledConfig`` both now set it, and so do the watchable
subagent channels, whose transcript the user reads in the subagent panel.

A subagent's answer, though, reaches the person as a plain-text reply on the
chat, where the HTML the contract asked for showed as raw tags. So every
subagent channel — the four delegates and a subtask started from the chat or a
schedule — is told to reply in plain text in place of the contract.

Driven on the real assembly path with real SQLite (``db``), the real
``MessageProcessor``/``PromptService``/``ProcessorConfig`` classes — zero mocks
— mirroring the sibling ``test_async_system_prompt_guidance.py``, which this
file follows structurally for the new gate."""

import sqlite3
from collections.abc import Callable

import pytest

from configs.channels import (
    DiscoveryConfig,
    ScheduledConfig,
    ScheduledSubtaskConfig,
    UserConfig,
    UserSubtaskConfig,
    VisionConfig,
    WebBrowseConfig,
    WebSearchConfig,
)
from configs.channels.code_agent import CodeAgentConfig
from configs.channels.pim import PimConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig
from services.prompt_service import _RESPONSE_FORMAT

pytestmark = pytest.mark.unit

# Every kind of subagent the person can open and watch, and whose answer can
# come back to the chat as a reply. The caller's policy channel is passed
# through, as the delegating tool does.
_WATCHABLE_SUBAGENTS: dict[str, Callable[[], ProcessorConfig]] = {
    "web_search": lambda: WebSearchConfig(PolicyChannel.CHAT),
    "web_browse": lambda: WebBrowseConfig(PolicyChannel.CHAT),
    "pim": lambda: PimConfig(PolicyChannel.CHAT),
    "code_agent": lambda: CodeAgentConfig(PolicyChannel.CHAT),
    "subtask_from_chat": UserSubtaskConfig,
    "subtask_from_schedule": ScheduledSubtaskConfig,
}

# The exact sentence a subagent is told, so a reworded instruction turns this red.
_PLAIN_TEXT = "Reply with plain text, no markdown, formatting of any kind."


def _assembled_system(config: ProcessorConfig) -> str:
    """Build a real MessageProcessor for *config* and return the system prompt
    its turn would send — the real ``PromptService.system_prompt`` output."""
    mp = MessageProcessor(config, raw_input="Remind me to water the plants")
    mp.active_tools = list(config.always_available or [])
    return mp.prompt_service.system_prompt()


def test_response_format_present_on_schedule_channel(db: sqlite3.Connection) -> None:
    """The bug this change fixes: a fired schedule previously got no
    response-format guidance at all, even though its output is rendered as HTML
    just like the user channel's. ``ScheduledConfig.RENDERS_HTML`` now gates the
    same contract onto the schedule channel's system prompt."""
    system = _assembled_system(ScheduledConfig())

    assert _RESPONSE_FORMAT in system, (
        "HTML response-format contract missing from the schedule-channel system prompt"
    )
    assert "NEVER use markdown syntax" in system
    assert _PLAIN_TEXT not in system


def test_response_format_present_on_user_channel(db: sqlite3.Connection) -> None:
    """UserConfig previously baked the contract as the literal tail of its
    ``system_prompt`` property. The contract moved to a shared constant appended
    by ``PromptService`` itself — the assembled user-channel prompt must still
    carry it."""
    system = _assembled_system(UserConfig())

    assert _RESPONSE_FORMAT in system
    assert "NEVER use markdown syntax" in system
    assert _PLAIN_TEXT not in system


def test_response_format_absent_on_silent_non_html_channel(db: sqlite3.Connection) -> None:
    """Discovery is a genuinely silent background channel (``RENDERS_HTML``
    re-declared False) whose output is never rendered to a human — it must
    never be told to emit HTML."""
    system = _assembled_system(DiscoveryConfig())

    assert _RESPONSE_FORMAT not in system
    assert "## Response format" not in system
    assert _PLAIN_TEXT not in system


@pytest.mark.parametrize(
    "make_config", list(_WATCHABLE_SUBAGENTS.values()), ids=list(_WATCHABLE_SUBAGENTS)
)
def test_watchable_subagent_is_told_to_reply_in_plain_text(
    db: sqlite3.Connection, make_config: Callable[[], ProcessorConfig]
) -> None:
    """A subagent's answer comes back to the chat as a plain-text reply. Told
    to write HTML, it answered with ``<h1>``, ``<ul>`` and ``<b>`` that the
    person then read as raw tags — so every kind of subagent is told, once,
    to reply in plain text, and is never shown the HTML contract."""
    system = _assembled_system(make_config())

    assert system.count(_PLAIN_TEXT) == 1, "a subagent is not told to reply in plain text"
    assert _RESPONSE_FORMAT not in system
    assert "## Response format" not in system
    assert "NEVER use markdown syntax" not in system


def test_response_format_absent_on_vision_channel(db: sqlite3.Connection) -> None:
    """Vision is a subagent too, but its answer is an image description that is
    indexed as plain text and never displayed — it must not be told to emit
    HTML markup that would then pollute the index."""
    system = _assembled_system(VisionConfig(PolicyChannel.CHAT))

    assert _RESPONSE_FORMAT not in system
    assert "## Response format" not in system
    assert _PLAIN_TEXT not in system
