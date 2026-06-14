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


# ===========================================================================
# Migrated from test_ability_web_download_tool_result.py (TKT-975)
# Offline contract: pre-gate, blocked schemes, SSRF guard, module hygiene.
# All four scheme/SSRF tests migrated — none were covered in the network tier above.
# ===========================================================================

import threading  # noqa: E402 — appended to existing file

import abilities.web_download  # noqa: E402
from abilities._dispatcher import ToolDispatcher  # noqa: E402
from configs.channels import DmnConfig  # noqa: E402
from services.act_trail import ActTrail  # noqa: E402
from tests._tool_result_harness import seed_transcript  # noqa: E402


def _seed_dl_transcript(db) -> int:
    """Insert the transcript anchor (tool_calls.transcript_id FK) the trail hangs
    its recorded rows off, and return its id."""
    return seed_transcript(db, channel="dmn", content="download this file for me")


class _DownloadMP:
    """Minimal real MP-shaped context — exactly what dispatch reads off a live
    processor: ``config`` (policy channel + emitter gate), ``uid`` (the transcript
    anchor), and ``cancel_event``. No policy row needed — web_download is INTERNAL."""

    def __init__(self, uid: int):
        self.config = DmnConfig()
        self.uid = uid
        self.DISCOVERABLE: list[str] = []
        self.active_tools: list[str] = []
        self.cancel_event = threading.Event()


@pytest.mark.unit
def test_file_scheme_is_blocked_url(db):
    """A ``file://`` URL is refused with ``code=blocked-url`` — the scheme check is
    network-free and fires before any fetch (was: status=success prose)."""
    mp = _DownloadMP(_seed_dl_transcript(db))

    result = ToolDispatcher(mp).dispatch("web_download", {"url": "file:///etc/passwd"})

    assert "status=error" in result
    assert "code=blocked-url" in result


@pytest.mark.unit
def test_data_scheme_is_blocked_url(db):
    """A ``data:`` URL is refused with ``code=blocked-url`` — urlparse-cheap, no
    network."""
    mp = _DownloadMP(_seed_dl_transcript(db))

    result = ToolDispatcher(mp).dispatch(
        "web_download", {"url": "data:text/plain;base64,SGVsbG8="}
    )

    assert "status=error" in result
    assert "code=blocked-url" in result


@pytest.mark.unit
def test_localhost_ssrf_terminal_outcome_is_blocked_url(db):
    """The single SSRF guard in ``web_fetch.stream_to_file`` refuses
    ``http://localhost`` (resolves private) and the ability maps the resulting
    ``FetchBlocked`` to ``code=blocked-url`` — network-free, and the real trail
    records the outcome against the anchor."""
    transcript_id = _seed_dl_transcript(db)
    mp = _DownloadMP(transcript_id)

    result = ToolDispatcher(mp).dispatch("web_download", {"url": "http://localhost/x"})

    assert "status=error" in result
    assert "code=blocked-url" in result

    rows = ActTrail().fetch_by_transcript_id(transcript_id)
    assert [r["tool_name"] for r in rows] == ["web_download"]
    assert "blocked-url" in rows[0]["result"]


@pytest.mark.unit
def test_127_ssrf_terminal_outcome_is_blocked_url(db):
    """``http://127.0.0.1`` is the same single-guard outcome — proves the guard is
    address-resolved, not host-string special-cased to 'localhost'."""
    mp = _DownloadMP(_seed_dl_transcript(db))

    result = ToolDispatcher(mp).dispatch("web_download", {"url": "http://127.0.0.1/secret"})

    assert "status=error" in result
    assert "code=blocked-url" in result


@pytest.mark.unit
def test_module_no_longer_carries_local_ssrf_guard():
    """The bespoke local ``is_private_url`` import and ``_validate_url`` helper are
    gone — the guard lives in ``web_fetch`` (one source), the scheme check is
    inline. Mirrors the read hygiene assertion in test_ssrf_single_source."""
    assert not hasattr(abilities.web_download, "is_private_url"), (
        "web_download must not re-import the SSRF guard; it reaches it through web_fetch"
    )
    assert not hasattr(abilities.web_download, "_validate_url"), (
        "the local _validate_url pre-check is replaced by the inline scheme check + "
        "the single web_fetch guard"
    )
