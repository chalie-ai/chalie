# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test — every tool-bearing step broadcasts its interim boundary frame.

Regression observed live: the surface collapses the previous step's tool group and
opens the next ONLY when it receives an interim ``message`` frame.
``MessageProcessor._emit_interim`` used to gate that broadcast on ``formatted.strip()``,
so a tool-only step — tool calls with no assistant prose, the common case — emitted
no frame. The prior group never collapsed, every later step's tools piled into it, and
the whole turn rendered as one ever-growing expanded ACT loop.

The boundary frame must fire for a tool-bearing step regardless of whether the
step produced prose. Driven through the real production method + the real
WebSocketBroker; the receiver is the same JSON-over-the-wire consumer the
browser connection is.
"""

import json

import pytest

pytestmark = pytest.mark.unit


def _emit_and_capture(formatted: str) -> list[dict[str, object]]:
    """Run the real _emit_interim on a real user-channel MP and capture frames."""
    from configs.channels import UserConfig
    from services.message_processor import MessageProcessor
    from services.websocket_broker import WebSocketBroker

    mp = object.__new__(MessageProcessor)
    mp.config = UserConfig({})
    mp._metadata = {"exchange_id": "turn-1"}

    received: list[dict[str, object]] = []

    class _Receiver:
        def send(self, raw: str) -> None:
            received.append(json.loads(raw))

    broker = WebSocketBroker()
    receiver = _Receiver()
    broker.connect(receiver)
    try:
        mp._emit_interim(formatted)
    finally:
        broker.disconnect(receiver)
    return [m for m in received if m.get("type") == "message"]


@pytest.mark.parametrize("formatted", ["", "   ", "\n\t "])
def test_tool_only_step_still_emits_interim_boundary(formatted: str) -> None:
    """A prose-less tool-bearing step still broadcasts its interim boundary frame.

    If this regresses to zero, a ``formatted.strip()`` guard was re-added to
    _emit_interim and the multi-step ACT loop is back.
    """
    messages = _emit_and_capture(formatted)
    assert len(messages) == 1, (
        f"empty-prose step ({formatted!r}) emitted {len(messages)} interim frames, "
        "expected exactly 1 — the step boundary is gone and tools will pile into "
        "one ever-growing ACT group"
    )
    assert messages[0]["interim"] is True


def test_interim_carries_step_prose_when_present() -> None:
    """When the step has prose, the same frame carries it (the bubble renders)."""
    from typing import cast
    messages = _emit_and_capture("Let me check the weather.")
    assert len(messages) == 1
    assert messages[0]["interim"] is True
    assert "Let me check the weather." in cast(str, messages[0]["content"])
