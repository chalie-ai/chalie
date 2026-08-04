"""
NVIDIA NIM — integrate.api.nvidia.com.

NIM serves the OpenAI protocol from a vLLM-derived stack, so its ``/v1/models``
entries carry the same ``max_model_len`` field, reporting what the deployment
was actually started with.
https://docs.nvidia.com/nim/large-language-models/latest/api-reference.html
"""

from __future__ import annotations

from typing import ClassVar

from services.llm_clients.openai_compatible import OpenAICompatibleClient


class NvidiaClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'nvidia'
    LABEL: ClassVar[str] = 'NVIDIA NIM'
    DEFAULT_BASE_URL: ClassVar[str] = 'https://integrate.api.nvidia.com/v1'

    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = ('max_model_len',)
