"""
xAI (Grok) — api.x.ai.

Publishes the window as ``context_length`` on every ``/v1/models`` entry,
documented as "The maximum context length supported by the model, in tokens".
https://docs.x.ai/developers/rest-api-reference/inference/models
"""

from __future__ import annotations

from typing import ClassVar

from services.llm_clients.openai_compatible import OpenAICompatibleClient


class XaiClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'xai'
    LABEL: ClassVar[str] = 'xAI (Grok)'
    DEFAULT_BASE_URL: ClassVar[str] = 'https://api.x.ai/v1'

    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = ('context_length',)
