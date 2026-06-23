# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test — async-usage guidance is injected into the system prompt ONLY on
a ``SUPPORTS_ASYNC`` channel, at the real request-assembly chokepoint.

The guidance that teaches the model WHEN/HOW to use the per-call ``async`` flag
lives in ONE place: appended to the system prompt in
``MessageProcessor._build_send_dto`` whenever ``config.SUPPORTS_ASYNC`` is True —
the SAME gate that exposes the ``async`` tool parameter. Enabling async on a new
ProcessorConfig therefore surfaces this guidance automatically, at no per-tool
description cost.

Driven on the real assembly path with real SQLite (``db``), the real prompt
builders, and the real ability registry — zero mocks. We build a real
MessageProcessor for each channel and call the production method that assembles
the exact ``ProviderApiRequest`` the turn sends, then assert on ``dto.system`` —
i.e. precisely what the model is told. UserConfig (SUPPORTS_ASYNC=True) carries
the guidance verbatim; the non-async DmnConfig channel never does.
"""

import sqlite3

import pytest

from configs.channels import DmnConfig, UserConfig
from services.message_processor import _ASYNC_GUIDANCE, MessageProcessor
from services.processor_config import ProcessorConfig

pytestmark = pytest.mark.unit

# A distinctive marker from the guidance block — its presence proves the whole
# section landed in the assembled request.
_GUIDANCE_MARKER = "## Background tasks"


def _assembled_system(config: ProcessorConfig) -> str:
    """Build a real MessageProcessor for *config* and return the system prompt of
    the request its turn would send — the real ``_build_send_dto`` output."""
    from services.transcript_service import Transcript

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "What is the Maltese election about?", None)
    mp.config = config
    mp.uid = Transcript.write_input_row(config.channel, config.role, "What is the Maltese election about?")
    mp.active_tools = list(config.always_available or [])
    mp.thinking_level = "low"
    mp.thinking_override = None
    return mp._build_send_dto().system


def test_async_guidance_present_on_supports_async_channel(db: sqlite3.Connection) -> None:
    # UserConfig.SUPPORTS_ASYNC is True → the guidance is appended at the chokepoint.
    system = _assembled_system(UserConfig())

    assert _GUIDANCE_MARKER in system, "async guidance missing from the user-channel system prompt"
    # The whole block landed verbatim (single source of truth) and teaches the
    # flag name + the auto-reinvoke contract.
    assert _ASYNC_GUIDANCE in system
    assert "async: true" in system


def test_async_guidance_absent_on_non_async_channel(db: sqlite3.Connection) -> None:
    # DmnConfig leaves SUPPORTS_ASYNC at the base False → no guidance, ever.
    system = _assembled_system(DmnConfig())

    assert _GUIDANCE_MARKER not in system, "async guidance leaked onto a non-async channel"
