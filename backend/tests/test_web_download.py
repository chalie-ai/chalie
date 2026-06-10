"""Tests for WebDownloadAbility.

Two tiers, kept strictly separate:

* ``@pytest.mark.unit`` — pure, deterministic, zero external dependencies.
  Exercises URL validation: blocked schemes, private/internal-IP SSRF
  rejection, and missing-URL handling. These never touch the network.

* ``@pytest.mark.integration`` — real downloads against ``httpbin.org``.
  Guarded by a skip when the host is unreachable so air-gapped CI, proxied
  environments, and httpbin outages do not produce spurious failures.
"""

import os
import socket
import tempfile

import pytest

from abilities.web_download import WebDownloadAbility

_ability = WebDownloadAbility()
_TMP = os.path.join(tempfile.gettempdir(), "chalie_downloads")

_HTTPBIN_HOST = "httpbin.org"


def _httpbin_reachable() -> bool:
    """Best-effort TCP probe so integration tests skip (not fail) offline."""
    try:
        with socket.create_connection((_HTTPBIN_HOST, 443), timeout=3):
            return True
    except OSError:
        return False


@pytest.fixture
def httpbin():
    """Skip the requesting test when httpbin.org is unreachable.

    The probe runs lazily inside the fixture (setup time of an integration
    test) rather than at import — so collecting or running ``-m unit`` makes
    zero network calls.
    """
    if not _httpbin_reachable():
        pytest.skip(f"{_HTTPBIN_HOST} unreachable — skip network-dependent download tests.")


# ── Unit tier: pure URL validation, no network ───────────────────────────────


@pytest.mark.unit
def test_blocked_schemes():
    assert "blocked" in _ability.run({"url": "file:///etc/passwd"}).body.lower()
    assert "blocked" in _ability.run({"url": "data:text/plain;base64,SGVsbG8="}).body.lower()


@pytest.mark.unit
def test_private_ip_blocked():
    result = _ability.run({"url": "http://127.0.0.1/secret"}).body
    assert "blocked" in result.lower() or "private" in result.lower()


@pytest.mark.unit
def test_missing_url_is_rejected():
    assert "required" in _ability.run({}).body.lower()
    assert "required" in _ability.run({"url": ""}).body.lower()


# ── Integration tier: real downloads against httpbin.org ─────────────────────


@pytest.mark.integration
def test_successful_download(httpbin):
    result = _ability.run({"url": "https://httpbin.org/robots.txt"}).body
    assert os.path.isabs(result)
    assert result.startswith(_TMP)
    assert os.path.exists(result)
    assert os.path.getsize(result) > 0
    assert result.endswith("robots.txt")

    second = _ability.run({"url": "https://httpbin.org/robots.txt"}).body
    assert second != result
    assert second.endswith("robots.txt")
    assert os.path.exists(second)

    os.remove(result)
    os.remove(second)


@pytest.mark.integration
def test_timeout_and_edge_values(httpbin):
    result = _ability.run({"url": "https://httpbin.org/bytes/32", "timeout": 5}).body
    assert os.path.exists(result)
    os.remove(result)

    result = _ability.run({"url": "https://httpbin.org/bytes/32", "timeout": 999}).body
    assert os.path.exists(result)
    os.remove(result)

    result = _ability.run({"url": "https://httpbin.org/bytes/32", "timeout": "banana"}).body
    assert os.path.exists(result)
    os.remove(result)


@pytest.mark.integration
def test_http_error_status_is_reported(httpbin):
    result = _ability.run({"url": "https://httpbin.org/status/404"}).body
    assert "404" in result or "not found" in result.lower() or "error" in result.lower()
    assert not os.path.exists(result)
