"""Integration tests for WebDownloadAbility — real downloads against httpbin.org.

The offline contract (missing-url pre-gate, blocked schemes, the single SSRF
guard, module hygiene) is pinned in the feature-test file
``test_ability_web_download_tool_result.py``, which drives the real
``ToolDispatcher.dispatch`` hot path. This file holds ONLY the network tier:
real downloads that cannot run offline, guarded by a skip when httpbin is
unreachable so air-gapped CI, proxied environments, and httpbin outages do not
produce spurious failures.

TKT-900 changed the success contract: ``run()`` now returns a structured
``{"path", "bytes", "content_type"}`` body (was a bare path string) with the
size cap declared in ``meta``; an over-cap download is a ``too-large`` error.
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


# ── Integration tier: real downloads against httpbin.org ─────────────────────


@pytest.mark.integration
def test_successful_download(httpbin):
    result = _ability.run({"url": "https://httpbin.org/robots.txt"})
    assert result.status == "success"
    body = result.body
    assert isinstance(body, dict)
    path = body["path"]
    assert os.path.isabs(path)
    assert path.startswith(_TMP)
    assert os.path.exists(path)
    assert body["bytes"] == os.path.getsize(path) > 0
    assert path.endswith("robots.txt")
    # The cap is declared on success so the model knows the enforced limit.
    assert result.meta["max_bytes"] == WebDownloadAbility._MAX_DOWNLOAD_BYTES
    assert result.meta["source"] == "https://httpbin.org/robots.txt"

    second = _ability.run({"url": "https://httpbin.org/robots.txt"})
    assert second.status == "success"
    second_path = second.body["path"]
    assert second_path != path
    assert second_path.endswith("robots.txt")
    assert os.path.exists(second_path)

    os.remove(path)
    os.remove(second_path)


@pytest.mark.integration
def test_timeout_and_edge_values(httpbin):
    result = _ability.run({"url": "https://httpbin.org/bytes/32", "timeout": 5})
    assert result.status == "success"
    assert os.path.exists(result.body["path"])
    os.remove(result.body["path"])

    # Over-max timeout is clamped (param clamp=(1, 120)) — still downloads.
    result = _ability.run({"url": "https://httpbin.org/bytes/32", "timeout": 999})
    assert result.status == "success"
    assert os.path.exists(result.body["path"])
    os.remove(result.body["path"])

    # Non-numeric timeout falls back via the param clamp default — still downloads.
    result = _ability.run({"url": "https://httpbin.org/bytes/32", "timeout": "banana"})
    assert result.status == "success"
    assert os.path.exists(result.body["path"])
    os.remove(result.body["path"])


@pytest.mark.integration
def test_http_error_status_is_reported(httpbin):
    result = _ability.run({"url": "https://httpbin.org/status/404"})
    assert result.status == "error"
    assert result.code == "download-failed"
    assert "404" in result.body or "not found" in result.body.lower()


@pytest.mark.integration
def test_too_large_download_is_an_error(httpbin):
    """A body exceeding the 100 MB cap aborts with ``too-large`` and leaves no
    partial file — the cap is enforced mid-stream, never silently truncated.

    httpbin's ``/bytes/<n>`` streams ``n`` random bytes; request just past the
    cap so the running byte count trips the abort."""
    over = WebDownloadAbility._MAX_DOWNLOAD_BYTES + 1
    result = _ability.run({"url": f"https://httpbin.org/bytes/{over}"})
    assert result.status == "error"
    assert result.code == "too-large"
    assert result.meta["max_bytes"] == WebDownloadAbility._MAX_DOWNLOAD_BYTES
    # The partial file was removed.
    assert "path" not in (result.body if isinstance(result.body, dict) else {})
