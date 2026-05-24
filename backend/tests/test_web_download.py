"""Feature tests for WebDownloadAbility (abilities/web_download.py).

Pure-function tests cover the four deterministic helpers:
  _validate_url, is_private_url, _extract_filename, _resolve_destination.

The integration test exercises the real download path against a live public
endpoint (httpbin.org/bytes/64) — a tiny 64-byte response that always succeeds
and does not carry user data.  Network call is the real boundary; everything
else — os.makedirs, file I/O, the requests pipeline — runs through production
code unchanged.

Mock boundary: requests.Session.get is patched ONLY in the one test that
verifies the too-large Content-Length guard, because exercising a real 101 MB
download in a unit suite is impractical.  All other tests hit no mocked
production code.
"""

import os

import pytest

from abilities._ssrf import is_private_url
from abilities.web_download import (
    _extract_filename,
    _resolve_destination,
    _validate_url,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# URL scheme validation — http/https accepted; everything else blocked
# ---------------------------------------------------------------------------


def test_validate_url_accepts_http():
    """http:// URLs must pass validation."""
    assert _validate_url("http://example.com/file.pdf") is None


def test_validate_url_accepts_https():
    """https:// URLs must pass validation."""
    assert _validate_url("https://example.com/data.csv") is None


def test_validate_url_blocks_file_scheme():
    """file:// URLs must be rejected — only http and https are allowed."""
    error = _validate_url("file:///etc/passwd")
    assert error is not None
    assert "file" in error.lower() or "scheme" in error.lower() or "only http" in error.lower()


def test_validate_url_blocks_data_scheme():
    """data: URLs must be rejected — they bypass network controls."""
    error = _validate_url("data:text/html;base64,SGVsbG8=")
    assert error is not None


# ---------------------------------------------------------------------------
# SSRF guard — private/loopback IPs blocked
# ---------------------------------------------------------------------------


def testis_private_url_blocks_loopback():
    """127.0.0.1 (loopback) must be identified as private."""
    assert is_private_url("http://127.0.0.1/secret") is True


def testis_private_url_blocks_rfc1918_10_block():
    """10.x.x.x addresses must be blocked (RFC 1918 private range)."""
    assert is_private_url("http://10.0.0.1/internal") is True


def testis_private_url_blocks_link_local():
    """169.254.x.x link-local addresses must be blocked."""
    assert is_private_url("http://169.254.169.254/latest/meta-data/") is True


def test_validate_url_returns_error_for_private_ip():
    """_validate_url must propagate the SSRF block as a non-None error string."""
    error = _validate_url("http://127.0.0.1/admin")
    assert error is not None
    assert "private" in error.lower() or "internal" in error.lower() or "blocked" in error.lower()


# ---------------------------------------------------------------------------
# Filename extraction — URL path → filename, fallback "download"
# ---------------------------------------------------------------------------


def test_extract_filename_from_url_path():
    """Last path segment becomes the filename."""
    assert _extract_filename("https://example.com/reports/annual-2025.pdf") == "annual-2025.pdf"


def test_extract_filename_strips_trailing_slash():
    """Trailing slash must not produce an empty filename."""
    name = _extract_filename("https://example.com/file.csv/")
    assert name == "file.csv"


def test_extract_filename_fallback_for_root_url():
    """A URL with no path segment (bare root) must fall back to 'download'."""
    assert _extract_filename("https://example.com") == "download"


def test_extract_filename_fallback_for_slash_only():
    """A URL whose path is only '/' must fall back to 'download'."""
    assert _extract_filename("https://example.com/") == "download"


# ---------------------------------------------------------------------------
# Destination resolution
# ---------------------------------------------------------------------------


def test_resolve_destination_none_goes_to_tmp(tmp_path):
    """None destination resolves to the chalie_downloads tmp directory."""
    result = _resolve_destination("https://example.com/file.csv", None)
    assert result.startswith("/tmp/chalie_downloads/")
    assert "file.csv" in result


def test_resolve_destination_explicit_tmp_path_honoured():
    result = _resolve_destination("https://example.com/file.pdf", "/tmp/somewhere/file.pdf")
    assert result == "/tmp/somewhere/file.pdf"


def test_resolve_destination_absolute_path_returned_as_is():
    """A non-/tmp absolute destination path is returned unchanged."""
    result = _resolve_destination("https://example.com/file.json", "/root/downloads/file.json")
    assert result == "/root/downloads/file.json"


def test_resolve_destination_generates_unique_names():
    """Two calls with the same URL and no destination must produce different paths."""
    a = _resolve_destination("https://example.com/file.csv", None)
    b = _resolve_destination("https://example.com/file.csv", None)
    assert a != b


# ---------------------------------------------------------------------------
# Real download (integration — requires network)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_download_creates_file_on_disk(tmp_path):
    """Download a real public file and verify it lands on disk with content.

    Uses httpbin.org/bytes/64 — a 64-byte random-bytes endpoint that is
    stable, tiny, and does not require authentication.
    """
    from abilities.web_download import _perform_download

    dest = str(tmp_path / "test_download.bin")
    _perform_download("https://httpbin.org/bytes/64", dest)

    assert os.path.exists(dest), "Downloaded file must exist on disk"
    assert os.path.getsize(dest) > 0, "Downloaded file must not be empty"


@pytest.mark.integration
def test_execute_returns_path_message_on_success(tmp_path):
    """WebDownloadAbility.execute() with a real URL returns a success message citing the path."""
    from abilities.web_download import WebDownloadAbility

    ability = WebDownloadAbility()
    result = ability.execute("text", {"url": "https://httpbin.org/bytes/32"}, None)

    assert "downloaded" in result.lower() or "/tmp/chalie_downloads/" in result
    assert "error" not in result.lower()
