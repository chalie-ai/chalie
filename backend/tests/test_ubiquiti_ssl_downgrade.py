# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test for the UniFi SSL-verification downgrade regression (#1906).

Spins a real HTTPS server with a self-signed certificate (no mocks) and
proves that when the user opts into ``verify_ssl=True``, an SSLError is
raised to the caller instead of being silently swallowed by retrying the
request with ``verify=False``. This is exactly the narrow MITM exposure the
audit flagged: credentials/CSRF carried over an unverified connection that
the user never asked for.

The same server also proves the ``verify_ssl=False`` path still works, so the
test is a two-sided contract: verify-True must fail loud, verify-False must
connect.
"""

from __future__ import annotations

import http.server
import ssl
import threading
from pathlib import Path
from typing import cast

import pytest
import requests.exceptions

from capabilities.ubiquiti_capability import unifi_rest_handler as rest

pytestmark = pytest.mark.unit

# ── Self-signed cert fixture ────────────────────────────────────────────────

_CERT_DIR = Path(__file__).parent / "_ssl_downgrade_certs"


def _ensure_cert() -> tuple[Path, Path]:
    key = _CERT_DIR / "key.pem"
    cert = _CERT_DIR / "cert.pem"
    if cert.exists() and key.exists():
        return cert, key
    _CERT_DIR.mkdir(parents=True, exist_ok=True)
    import subprocess

    subprocess.run(  # noqa: S603 — trusted args
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key), "-out", str(cert),
            "-days", "1", "-nodes", "-subj", "/CN=localhost",
        ],
        check=True, capture_output=True,
    )
    _key_perms = 0o600
    key.chmod(_key_perms)
    return cert, key


@pytest.fixture(scope="module")
def untrusted_https_server() -> object:
    """A real HTTPS server presenting a self-signed cert, threaded in the background."""
    cert, key = _ensure_cert()

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — http.server contract
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data": []}')

        def log_message(self, *args: object) -> None:  # noqa: ARG002 — silence
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    server.socket = ssl.wrap_socket(  # noqa: S502 — self-signed by design
        server.socket, certfile=str(cert), keyfile=str(key), server_side=True,
    )
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class _ServerHandle:
        url = f"https://127.0.0.1:{port}"

        def stop(self) -> None:
            server.shutdown()
            server.server_close()

    handle = _ServerHandle()
    try:
        yield handle
    finally:
        handle.stop()


# ── The regression contract ────────────────────────────────────────────────


def test_verify_true_raises_on_untrusted_cert(untrusted_https_server: object) -> None:
    """When the user opts into verify_ssl=True, an untrusted cert must raise — not silently downgrade."""
    handle = cast("object", untrusted_https_server)
    url = cast("str", getattr(handle, "url"))
    with pytest.raises(requests.exceptions.SSLError):
        rest.list_devices(url, "default", api_key="k", verify_ssl=True)


def test_verify_false_still_connects(untrusted_https_server: object) -> None:
    """The explicit opt-out path (verify_ssl=False) must still connect to the untrusted server."""
    handle = cast("object", untrusted_https_server)
    url = cast("str", getattr(handle, "url"))
    result = rest.list_devices(url, "default", api_key="k", verify_ssl=False)
    assert result["count"] == 0