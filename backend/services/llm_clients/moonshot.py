"""
Moonshot AI (Kimi) — api.moonshot.ai.

Publishes the window as ``context_length`` at the root of every ``/v1/models``
entry.
https://platform.moonshot.ai/docs/api/chat
"""

from __future__ import annotations

from typing import ClassVar

from services.llm_clients.openai_compatible import OpenAICompatibleClient


class MoonshotClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'moonshot'
    LABEL: ClassVar[str] = 'Moonshot AI (Kimi)'
    DEFAULT_BASE_URL: ClassVar[str] = 'https://api.moonshot.ai/v1'

    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = ('context_length',)
