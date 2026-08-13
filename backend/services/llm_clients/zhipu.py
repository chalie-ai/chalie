"""
Zhipu AI (GLM) — open.bigmodel.cn.

Zhipu publishes no model-listing endpoint at all on its OpenAI-compatible
surface: the documented paths are the inference ones, and the window appears
only as prose on each model's page.
https://docs.bigmodel.cn/api-reference

So there is nothing to list and nothing to read, and the probe pings. This is
the honest outcome rather than a bug: with no machine-readable figure
published, the default window is what can be defended, and a hardcoded table
of GLM sizes would be a guess that rots on the vendor's next release.
"""

from __future__ import annotations

from typing import ClassVar

from services.llm_clients.openai_compatible import OpenAICompatibleClient


class ZhipuClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'zhipu'
    LABEL: ClassVar[str] = 'Zhipu AI (GLM)'
    DEFAULT_BASE_URL: ClassVar[str] = 'https://open.bigmodel.cn/api/paas/v4'

    #: Ping only — this vendor serves no listing to read.
    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = ()
