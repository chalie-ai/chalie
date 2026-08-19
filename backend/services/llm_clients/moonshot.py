"""
Moonshot AI (Kimi) — api.moonshot.ai.

Publishes the window as ``context_length`` at the root of every ``/v1/models``
entry.
https://platform.moonshot.ai/docs/api/chat

Its reasoning scale is low|high|max, defaulting to ``max`` with no documented
off switch, so it states one instead of inheriting a row built around values
Kimi never published. See MOONSHOT_REASONING_EFFORTS.
"""

from __future__ import annotations

from typing import ClassVar

from configs.enums.thinking_level import ThinkingLevel
from services.llm_clients.openai_compatible import OpenAICompatibleClient
from services.llm_clients.thinking_map import MOONSHOT_REASONING_EFFORTS


class MoonshotClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'moonshot'
    LABEL: ClassVar[str] = 'Moonshot AI (Kimi)'
    DEFAULT_BASE_URL: ClassVar[str] = 'https://api.moonshot.ai/v1'

    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = ('context_length',)

    REASONING_EFFORTS: ClassVar[dict[ThinkingLevel, str]] = MOONSHOT_REASONING_EFFORTS
