"""
MiniMax — api.minimax.io.

MiniMax documents no context-window field on any endpoint: the figure appears
only in prose on its model pages. The listing is therefore skipped and the
probe pings, which proves the host and model are live and sizes them at the
default rather than at a number nobody published.
https://platform.minimax.io/docs/api-reference/text-chatcompletion-v2
"""

from __future__ import annotations

from typing import ClassVar

from services.llm_clients.openai_compatible import OpenAICompatibleClient


class MiniMaxClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'minimax'
    LABEL: ClassVar[str] = 'MiniMax'
    DEFAULT_BASE_URL: ClassVar[str] = 'https://api.minimax.io/v1'

    #: Ping only — a listing with no size field is not worth the round trip.
    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = ()
