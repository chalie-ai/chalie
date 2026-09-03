"""
OpenRouter — openrouter.ai.

The only vendor here that publishes the window twice, and the distinction
matters. ``top_provider.context_length`` is what the backend OpenRouter
actually routes to will serve; ``context_length`` at the entry root is the
headline figure across every backend that serves the model. The nested one is
listed first because it is the limit a request will really be measured against.
https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties
"""

from __future__ import annotations

from typing import ClassVar

from services.llm_clients.openai_compatible import OpenAICompatibleClient


class OpenRouterClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'openrouter'
    LABEL: ClassVar[str] = 'OpenRouter'
    DEFAULT_BASE_URL: ClassVar[str] = 'https://openrouter.ai/api/v1'

    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = (
        'top_provider.context_length',
        'context_length',
    )
