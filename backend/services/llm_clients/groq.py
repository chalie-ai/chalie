"""
Groq — api.groq.com.

Publishes the window as ``context_window`` at the root of every
``/openai/v1/models`` entry.
https://console.groq.com/docs/api-reference#models-list
"""

from __future__ import annotations

from typing import ClassVar

from services.llm_clients.openai_compatible import OpenAICompatibleClient


class GroqClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'groq'
    LABEL: ClassVar[str] = 'Groq'
    DEFAULT_BASE_URL: ClassVar[str] = 'https://api.groq.com/openai/v1'

    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = ('context_window',)
