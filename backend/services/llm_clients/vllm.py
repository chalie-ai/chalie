"""
vLLM — a self-hosted OpenAI-protocol server.

Its ``/v1/models`` entries are ``ModelCard`` objects carrying ``max_model_len``,
populated straight from ``model_config.max_model_len`` — the length the server
was actually started with, not a figure from the checkpoint. That makes it the
authoritative answer: a vLLM host cannot be wrong about its own window.
https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/engine/protocol.py

No key by default: vLLM only installs its authentication middleware when
``--api-key`` or ``VLLM_API_KEY`` is set, so one is optional here rather than
required.
"""

from __future__ import annotations

from typing import ClassVar

from services.llm_clients.openai_compatible import OpenAICompatibleClient


class VllmClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'vllm'
    LABEL: ClassVar[str] = 'vLLM (self-hosted)'
    DEFAULT_BASE_URL: ClassVar[str] = 'http://localhost:8000/v1'
    REQUIRES_KEY: ClassVar[bool] = False

    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = ('max_model_len',)
