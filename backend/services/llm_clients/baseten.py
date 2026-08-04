"""
Baseten — inference.baseten.co.

The one vendor whose listing is documented to carry the window without
documenting what it is called: Baseten states that ``/v1/models`` returns
"metadata including pricing, context windows, and supported features", but
publishes no response schema or example.
https://docs.baseten.co/inference/model-apis/overview

So this class deliberately does not narrow ``WINDOW_FIELDS``. It keeps the
base's union of documented spellings and tries them against a listing the
vendor says holds the figure; whatever the key turns out to be, an
unrecognised one costs nothing but the fall-through to a ping. When Baseten
publishes the schema, replace the inherited union with the real path — do not
add a Baseten branch to the base.
"""

from __future__ import annotations

from typing import ClassVar

from services.llm_clients.openai_compatible import OpenAICompatibleClient


class BasetenClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'baseten'
    LABEL: ClassVar[str] = 'Baseten'
    DEFAULT_BASE_URL: ClassVar[str] = 'https://inference.baseten.co/v1'
