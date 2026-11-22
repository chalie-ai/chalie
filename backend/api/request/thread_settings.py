"""Inbound DTO for the thread-settings action."""

from __future__ import annotations

from typing import Literal

from .request import Request


class ThreadSettingsChange(Request):
    """Body for POST /api/threads/settings/<turn_id> — the model and thinking
    level the thread runs on from its next message."""

    provider_id: int
    thinking_level: Literal["auto", "medium", "high"]
