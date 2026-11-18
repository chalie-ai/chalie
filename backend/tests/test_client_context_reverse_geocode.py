# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test for the geopy-free Nominatim reverse-geocode call.

Spins a real ``http.server`` on 127.0.0.1 with an ephemeral port (no
mocks) and points the module's reverse-URL constant at it via
``monkeypatch``. Proves the wire contract the geopy 2.5 call used —
``User-Agent: Chalie/1.0`` plus ``lat``, ``lon``, ``format=json``,
``accept-language=en``, ``addressdetails=1`` — and the response rules:
an address with only ``town`` + ``country`` resolves to "Town, Country",
a bare ``{"error": "Unable to geocode"}`` body and an HTTP 500 both
degrade to ``None``, and a country-only address resolves to the country
string.
"""

from __future__ import annotations

import http.server
import json
import threading
from collections.abc import Iterator

import pytest

import services.client_context_service as client_context
from services.client_context_service import ClientContextService

pytestmark = pytest.mark.unit


class _StubState:
    """What the stub serves next, plus every request it received.

    ``recorded`` holds ``(request-path, user-agent)`` pairs in arrival
    order, so a test can assert on the last request it made.
    """

    def __init__(self) -> None:
        self.url: str = ""
        self.recorded: list[tuple[str, str | None]] = []
        self.status: int = 200
        self.body: bytes = b"{}"

    def serve(self, status: int, payload: dict[str, object]) -> None:
        """Arm the next response: HTTP ``status`` with a JSON ``payload`` body."""
        self.status = status
        self.body = json.dumps(payload).encode("utf-8")


@pytest.fixture(scope="module")
def nominatim_stub() -> Iterator[_StubState]:
    """A real Nominatim-shaped HTTP server, threaded in the background."""
    state = _StubState()

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — http.server contract
            state.recorded.append((self.path, self.headers.get("User-Agent")))
            self.send_response(state.status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(state.body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: ARG002 — silence
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


def _resolve(
    stub: _StubState,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    payload: dict[str, object],
    lat: float,
    lon: float,
) -> str | None:
    """Point the reverse-URL constant at the stub and reverse-geocode (lat, lon)."""
    stub.serve(status, payload)
    monkeypatch.setattr(client_context, "_NOMINATIM_REVERSE_URL", stub.url)
    service = ClientContextService()
    return service._resolve_location_name(lat, lon)


def test_town_plus_country_resolves_and_wire_contract(
    nominatim_stub: _StubState, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An address with only ``town`` + ``country`` resolves to "Town, Country" —
    and the request carries the exact wire contract the geopy call used."""
    stub = nominatim_stub
    result = _resolve(
        stub, monkeypatch, 200,
        {"name": "Town, Country", "type": "town",
         "address": {"town": "Town", "country": "Country"}},
        41.8781, -87.6298,
    )
    assert result == "Town, Country"

    path, user_agent = stub.recorded[-1]
    prefix, _, query = path.partition("?")
    assert prefix == "/"
    assert dict(pair.split("=", 1) for pair in query.split("&")) == {
        "lat": "41.8781",
        "lon": "-87.6298",
        "format": "json",
        "accept-language": "en",
        "addressdetails": "1",
    }
    assert user_agent == "Chalie/1.0"


def test_unable_to_geocode_is_none(
    nominatim_stub: _StubState, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nominatim's "no result" body degrades to None (the old exactly_one path)."""
    stub = nominatim_stub
    assert _resolve(stub, monkeypatch, 200, {"error": "Unable to geocode"}, 12.0, 34.0) is None


def test_http_500_is_none(
    nominatim_stub: _StubState, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An HTTP error status degrades to None instead of raising to the caller."""
    stub = nominatim_stub
    assert _resolve(stub, monkeypatch, 500, {"error": "boom"}, 12.0, 34.0) is None


def test_country_only_resolves_to_country(
    nominatim_stub: _StubState, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A country-only address resolves to the bare country string."""
    stub = nominatim_stub
    assert _resolve(
        stub, monkeypatch, 200,
        {"name": "Country", "type": "country", "address": {"country": "Country"}},
        12.0, 34.0,
    ) == "Country"
