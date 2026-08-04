"""
Novita AI — api.novita.ai.

Publishes the window as ``context_size`` at the root of every
``/openai/v1/models`` entry, documented as required and described as "The
maximum context size of the model".
https://novita.ai/docs/api-reference/model-apis-llm-list-models
"""

from __future__ import annotations

from typing import ClassVar

from services.llm_clients.openai_compatible import OpenAICompatibleClient


class NovitaClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'novita'
    LABEL: ClassVar[str] = 'Novita AI'
    DEFAULT_BASE_URL: ClassVar[str] = 'https://api.novita.ai/openai/v1'

    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = ('context_size',)
