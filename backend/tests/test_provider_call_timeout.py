"""Feature test: a wedged provider is bounded at the client HTTP boundary.

There is exactly ONE provider-call timeout — PROVIDER_CALL_TIMEOUT_S — and every
thin client enforces it at its own socket boundary (the only place a wedged call
can actually be interrupted; a Python thread cannot be killed). When the boundary
trips, the client raises ProviderTimeoutError, which the retry helper surfaces
IMMEDIATELY — retrying an unresponsive provider would only multiply the deadline.

This drives the real production path with zero mocks: a real black-hole TCP server
(accepts the connection, then never writes a byte — exactly how a hung model
presents at the socket) and a real OllamaClient built by the production factory.
The only thing adjusted is the deadline constant, so the test finishes in seconds.
Its completion at ~the deadline — not 3× it — IS the proof of both the boundary
timeout and the fail-fast (no-retry) contract.
"""

import socket
import threading
import time
from typing import List

import pytest

import services.providers as providers_mod
from services.llm_clients.factory import build_client
from services.provider_api import (
    ProviderApiRequest,
    ProviderTimeoutError,
    ProviderType,
    ThinkingLevel,
)

pytestmark = pytest.mark.unit


def _black_hole_server() -> "tuple[int, List[socket.socket], socket.socket]":
    """Start a real TCP server that accepts connections and never responds."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    accepted: List[socket.socket] = []

    def _accept_loop() -> None:
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return  # server closed — end the loop
            accepted.append(conn)  # hold the socket open; never write to it

    threading.Thread(target=_accept_loop, name="black-hole", daemon=True).start()
    return srv.getsockname()[1], accepted, srv


def test_wedged_provider_times_out_fast_at_the_client_boundary() -> None:
    port, accepted, srv = _black_hole_server()
    original_timeout = providers_mod.PROVIDER_CALL_TIMEOUT_S
    providers_mod.PROVIDER_CALL_TIMEOUT_S = 2

    # Real production factory → real OllamaClient pointed at the black hole. The
    # provider layer never retries, so the call must trip once at the deadline.
    client = build_client({
        "platform": "ollama",
        "model": "wedged-model",
        "host": f"http://127.0.0.1:{port}",
    })

    dto = ProviderApiRequest(
        system="You are Chalie.",
        messages=[{"role": "user", "content": "hello"}],
        type=ProviderType.CHAT,
        tools=None,
        thinking_mode=ThinkingLevel.LOW,
    )

    try:
        t0 = time.monotonic()
        with pytest.raises(ProviderTimeoutError) as exc_info:
            client.send(dto)
        elapsed = time.monotonic() - t0

        assert accepted, "the client must have really connected to the boundary"
        assert exc_info.value.provider == "ollama", "the timeout error carries the provider"
        # ~1×deadline — the provider layer makes exactly one attempt and trips at
        # the boundary (resend policy lives in the MessageProcessor, not here).
        assert elapsed < 4.0, f"must trip at the {2}s deadline; took {elapsed:.1f}s"
    finally:
        providers_mod.PROVIDER_CALL_TIMEOUT_S = original_timeout
        srv.close()
        for conn in accepted:
            conn.close()
