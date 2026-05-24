"""Real-world tests for WebDownloadAbility. No mocks."""

import os
import tempfile

import pytest

from abilities.web_download import WebDownloadAbility

pytestmark = pytest.mark.unit

_ability = WebDownloadAbility()
_TMP = os.path.join(tempfile.gettempdir(), "chalie_downloads")


def test_successful_download():
    result = _ability.execute("text", {"url": "https://httpbin.org/robots.txt"}, None)
    assert os.path.isabs(result)
    assert result.startswith(_TMP)
    assert os.path.exists(result)
    assert os.path.getsize(result) > 0
    assert result.endswith("robots.txt")

    second = _ability.execute("text", {"url": "https://httpbin.org/robots.txt"}, None)
    assert second != result
    assert second.endswith("robots.txt")
    assert os.path.exists(second)

    os.remove(result)
    os.remove(second)


def test_timeout_and_edge_values():
    result = _ability.execute("text", {"url": "https://httpbin.org/bytes/32", "timeout": 5}, None)
    assert os.path.exists(result)
    os.remove(result)

    result = _ability.execute("text", {"url": "https://httpbin.org/bytes/32", "timeout": 999}, None)
    assert os.path.exists(result)
    os.remove(result)

    result = _ability.execute("text", {"url": "https://httpbin.org/bytes/32", "timeout": "banana"}, None)
    assert os.path.exists(result)
    os.remove(result)


def test_blocked_schemes():
    assert "blocked" in _ability.execute("text", {"url": "file:///etc/passwd"}, None).lower()
    assert "blocked" in _ability.execute("text", {"url": "data:text/plain;base64,SGVsbG8="}, None).lower()


def test_private_ip_blocked():
    result = _ability.execute("text", {"url": "http://127.0.0.1/secret"}, None)
    assert "blocked" in result.lower() or "private" in result.lower()


def test_errors():
    assert "required" in _ability.execute("text", {}, None).lower()
    assert "required" in _ability.execute("text", {"url": ""}, None).lower()

    result = _ability.execute("text", {"url": "https://this-domain-does-not-exist-xyz123.example"}, None)
    assert not os.path.exists(result)

    result = _ability.execute("text", {"url": "https://httpbin.org/status/404"}, None)
    assert "404" in result or "not found" in result.lower() or "error" in result.lower()
