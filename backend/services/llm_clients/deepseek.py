"""
DeepSeek — api.deepseek.com.

``/models`` returns ``id``/``object``/``owned_by`` and nothing else: DeepSeek
documents no context-window field on any endpoint, and publishes the figure
only in a prose pricing table.
https://api-docs.deepseek.com/api/list-models

So there is no listing to read, and the probe goes straight to the ping — which
establishes that the host and model are live and sizes them at the default. A
hardcoded table is deliberately not kept here: it would be a guess wearing a
measurement's clothes, and would rot silently the moment DeepSeek shipped a
model with a different window.

Its reasoning scale is low|high|max with thinking on by default at ``high``, so
it states one: the inherited row's silent LOW would have bought ``high``, the
opposite of what the level names. See DEEPSEEK_REASONING_EFFORTS.
"""

from __future__ import annotations

from typing import ClassVar

from configs.enums.thinking_level import ThinkingLevel
from services.llm_clients.openai_compatible import OpenAICompatibleClient
from services.llm_clients.thinking_map import DEEPSEEK_REASONING_EFFORTS


class DeepSeekClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'deepseek'
    LABEL: ClassVar[str] = 'DeepSeek'
    DEFAULT_BASE_URL: ClassVar[str] = 'https://api.deepseek.com'

    #: Ping only — a listing with no size field is not worth the round trip.
    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = ()

    REASONING_EFFORTS: ClassVar[dict[ThinkingLevel, str]] = DEEPSEEK_REASONING_EFFORTS
