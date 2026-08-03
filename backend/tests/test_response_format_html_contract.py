# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test — the shared HTML output-format contract lands in the system
prompt of every channel that renders to a human (``ProcessorConfig.RENDERS_HTML``),
and only those channels.

Before this change the contract was baked as literal text at the tail of
``UserConfig.system_prompt``. The ``schedule`` channel is exactly as much a
rendered, user-visible surface as ``user`` — its assistant output is converted
and sanitized as HTML at persist time (``MessageProcessor._format``, gated on
``RENDERS_HTML``)
— but it never got the matching prompt guidance, so a fired schedule's response
came back as raw markdown on an HTML surface. The fix hoists the contract into
``PromptService._RESPONSE_FORMAT`` and appends it in
``PromptService.system_prompt()`` whenever ``config.RENDERS_HTML`` is set;
``UserConfig`` and ``ScheduledConfig`` both now set it.

Driven on the real assembly path with real SQLite (``db``), the real
``MessageProcessor``/``PromptService``/``ProcessorConfig`` classes — zero mocks
— mirroring the sibling ``test_async_system_prompt_guidance.py``, which this
file follows structurally for the new gate."""

import sqlite3

import pytest

from configs.channels import DmnConfig, ScheduledConfig, UserConfig
from controllers.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig
from services.prompt_service import _ASYNC_GUIDANCE, _RESPONSE_FORMAT

pytestmark = pytest.mark.unit


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


def test_response_format_still_precedes_async_guidance_on_user_channel(db: sqlite3.Connection) -> None:
    """UserConfig previously baked the contract as the literal tail of its
    ``system_prompt`` property, immediately before ``PromptService`` appended
    the async-tasks guidance. The contract moved to a shared constant appended
    by ``PromptService`` itself — the assembled prompt, and the contract's
    position relative to the async guidance, must be unchanged by that move."""
    system = _assembled_system(UserConfig())

    assert _RESPONSE_FORMAT in system
    assert _ASYNC_GUIDANCE in system
    response_format_end = system.index(_RESPONSE_FORMAT) + len(_RESPONSE_FORMAT)
    assert system[response_format_end : response_format_end + len(_ASYNC_GUIDANCE)] == _ASYNC_GUIDANCE, (
        "HTML contract no longer sits immediately before the async-tasks guidance on the user channel"
    )


def test_response_format_absent_on_silent_non_html_channel(db: sqlite3.Connection) -> None:
    """DMN is a genuinely silent background channel (``RENDERS_HTML`` False) whose
    output is never rendered to a human and which never sets ``RENDERS_HTML`` — it
    must never be told to emit HTML."""
    system = _assembled_system(DmnConfig())

    assert _RESPONSE_FORMAT not in system
    assert "## Response format" not in system
