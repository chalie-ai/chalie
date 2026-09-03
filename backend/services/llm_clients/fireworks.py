"""
Fireworks AI — api.fireworks.ai.

Fireworks does publish a context window, but not anywhere this client can
reach it. The figure lives on the *control-plane* listing,
``GET /v1/accounts/{account_id}/models``, as ``models[].contextLength`` — a
different service from the inference base URL, keyed by an account id that a
provider row does not hold.
https://fireworks.ai/docs/api-reference/list-models

The inference host's own ``/v1/models`` names no size, so the probe pings.
Reaching the control plane would mean asking every user for an account id to
answer a question the inference endpoint should answer itself; that is a
change to the provider contract, not a probe detail, and is not made here.
"""

from __future__ import annotations

from typing import ClassVar

from services.llm_clients.openai_compatible import OpenAICompatibleClient


class FireworksClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'fireworks'
    LABEL: ClassVar[str] = 'Fireworks AI'
    DEFAULT_BASE_URL: ClassVar[str] = 'https://api.fireworks.ai/inference/v1'

    #: Ping only — the inference listing names no size.
    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = ()
