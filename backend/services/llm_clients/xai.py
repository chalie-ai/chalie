"""
xAI (Grok) — api.x.ai.

Publishes the window as ``context_length`` on every ``/v1/models`` entry,
documented as "The maximum context length supported by the model, in tokens".
https://docs.x.ai/developers/rest-api-reference/inference/models

Grok reasons on every request — xAI documents that reasoning cannot be disabled
— so the inherited OpenAI row was sending it a ``none`` it does not accept.
See XAI_REASONING_EFFORTS.
"""

from __future__ import annotations

from typing import ClassVar

from configs.enums.thinking_level import ThinkingLevel
from services.llm_clients.openai_compatible import OpenAICompatibleClient
from services.llm_clients.thinking_map import XAI_REASONING_EFFORTS


class XaiClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'xai'
    LABEL: ClassVar[str] = 'xAI (Grok)'
    DEFAULT_BASE_URL: ClassVar[str] = 'https://api.x.ai/v1'

    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = ('context_length',)

    REASONING_EFFORTS: ClassVar[dict[ThinkingLevel, str]] = XAI_REASONING_EFFORTS
