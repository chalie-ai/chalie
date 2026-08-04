"""
Mistral — api.mistral.ai.

Publishes the window as ``max_context_length`` at the root of every
``/v1/models`` entry. Note the spelling: not ``context_length``, which is what
most other vendors use.
https://docs.mistral.ai/api/endpoint/models
"""

from __future__ import annotations

from typing import ClassVar

from services.llm_clients.openai_compatible import OpenAICompatibleClient


class MistralClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'mistral'
    LABEL: ClassVar[str] = 'Mistral'
    DEFAULT_BASE_URL: ClassVar[str] = 'https://api.mistral.ai/v1'

    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = ('max_context_length',)
