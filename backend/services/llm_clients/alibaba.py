"""
Alibaba Model Studio (Qwen) — DashScope compatible mode.

DashScope's OpenAI-compatible surface documents only the inference endpoints:
there is no ``/models`` listing, and the static model catalog carries no
context-length column either.
https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope

So the probe pings. The default base URL is the international endpoint; the
mainland-China one (``https://dashscope.aliyuncs.com/compatible-mode/v1``)
differs only by host and is set by the user when that is the account's region.
"""

from __future__ import annotations

from typing import ClassVar

from services.llm_clients.openai_compatible import OpenAICompatibleClient


class AlibabaClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'alibaba'
    LABEL: ClassVar[str] = 'Alibaba Model Studio (Qwen)'
    DEFAULT_BASE_URL: ClassVar[str] = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'

    #: Ping only — this vendor serves no listing to read.
    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = ()
