"""Feature test: the generic single-call timeout in Providers.send().

A provider whose send() hangs must never wedge the turn. Providers.send runs the
real client.send(dto) under the single hard wall-clock ceiling
(Providers.SINGLE_CALL_TIMEOUT_S, 300s in production) and abandons the call with
a ProviderResponseError once the ceiling is hit.

Drives the real Providers().send entry point — the same chokepoint production
uses — against a real ProviderClient subclass whose send() genuinely blocks. No
mocks: the client boundary and the ceiling are swapped at the module/class level
(the sanctioned seam, mirrors test_convergence_release_gate). Without the wrapper
this test would hang forever on the never-returning send(); its completion IS the
proof the ceiling bounded the call.
"""

import threading
import time
from typing import ClassVar

import pytest

import services.providers as providers_mod
from services.llm_clients.base import ProviderClient
from services.provider_api import (
    ProviderApiRequest,
    ProviderApiResponse,
    ProviderResponseError,
    ProviderType,
    ThinkingLevel,
)
from services.providers import Providers

pytestmark = pytest.mark.unit


class _HangingClient(ProviderClient):
    """Real ProviderClient whose send() blocks until released — never returns on its own."""

    CONTENT_FIELD_LABEL: ClassVar[str] = "message.content"

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.released = threading.Event()
        self.provider = "hang"
        self.model = "hang-model"

    def get_context_limit(self) -> int:
        return 200_000

    def estimate_request_tokens(self, dto: ProviderApiRequest) -> int:
        return 1

    def send(self, dto: ProviderApiRequest) -> ProviderApiResponse:
        self.entered.set()
        self.released.wait()  # blocks forever unless the test releases it
        return ProviderApiResponse(text="late", model=self.model, provider=self.provider)


def test_send_abandons_a_hung_provider_call_at_the_ceiling() -> None:
    client = _HangingClient()
    original_resolve = Providers._resolve
    original_ceiling = providers_mod.SINGLE_CALL_TIMEOUT_S
    setattr(Providers, "_resolve", lambda self, *_a, **_kw: client)
    providers_mod.SINGLE_CALL_TIMEOUT_S = 1

    dto = ProviderApiRequest(
        system="You are Chalie.",
        messages=[{"role": "user", "content": "hello"}],
        type=ProviderType.CHAT,
        tools=None,
        thinking_mode=ThinkingLevel.LOW,
    )

    try:
        t0 = time.monotonic()
        with pytest.raises(ProviderResponseError) as exc_info:
            Providers().send(dto)
        elapsed = time.monotonic() - t0

        assert client.entered.is_set(), "the real client.send must have actually been entered"
        assert "ceiling" in str(exc_info.value).lower()
        assert exc_info.value.provider == "hang", "the timeout error carries the provider name"
        # The guard fired at ~1s, not the (unbounded) client — proves the generic
        # ceiling, not the client, bounded the call.
        assert elapsed < 4.0, f"send() must return at the 1s ceiling; took {elapsed:.1f}s"
    finally:
        client.released.set()  # let the abandoned daemon thread unwind
        providers_mod.SINGLE_CALL_TIMEOUT_S = original_ceiling
        setattr(Providers, "_resolve", original_resolve)
