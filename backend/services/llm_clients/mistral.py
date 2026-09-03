"""
Mistral — api.mistral.ai.

Publishes the window as ``max_context_length`` at the root of every
``/v1/models`` entry. Note the spelling: not ``context_length``, which is what
most other vendors use.
https://docs.mistral.ai/api/endpoint/models

Mistral documents exactly two reasoning_effort values, ``none`` and ``high``,
so every graduated level collapses onto ``high`` rather than send a ``medium``
the vendor never named. See MISTRAL_REASONING_EFFORTS.
"""

from __future__ import annotations

from typing import ClassVar

from configs.enums.thinking_level import ThinkingLevel
from services.llm_clients.openai_compatible import OpenAICompatibleClient
from services.llm_clients.thinking_map import MISTRAL_REASONING_EFFORTS


class MistralClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'mistral'
    LABEL: ClassVar[str] = 'Mistral'
    DEFAULT_BASE_URL: ClassVar[str] = 'https://api.mistral.ai/v1'

    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = ('max_context_length',)

    REASONING_EFFORTS: ClassVar[dict[ThinkingLevel, str]] = MISTRAL_REASONING_EFFORTS
