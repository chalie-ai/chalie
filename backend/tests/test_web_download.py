"""Integration tests for WebDownloadAbility — real downloads against httpbin.org.

The offline contract (missing-url pre-gate, blocked schemes, the single SSRF
guard, module hygiene) is pinned in the feature-test file
``test_ability_web_download_tool_result.py``, which drives the real
``ToolDispatcher.dispatch`` hot path. This file holds ONLY the network tier:
real downloads that cannot run offline, guarded by a skip when httpbin is
unreachable so air-gapped CI, proxied environments, and httpbin outages do not
produce spurious failures.
"""

import os
import socket
import sqlite3
import tempfile
from typing import cast

import pytest

from abilities._result import ToolParamError
from abilities.web_download import WebDownloadAbility

_ability = WebDownloadAbility()
_TMP = os.path.join(tempfile.gettempdir(), "chalie_downloads")

_HTTPBIN_HOST = "httpbin.org"


def _httpbin_reachable() -> bool:
    try:
        with socket.create_connection((_HTTPBIN_HOST, 443), timeout=3):
            return True
    except OSError:
        return False


@pytest.fixture
def httpbin() -> None:
    if not _httpbin_reachable():
        pytest.skip(f"{_HTTPBIN_HOST} unreachable — skip network-dependent download tests.")


# ── Integration tier: real downloads against httpbin.org ─────────────────────


@pytest.mark.integration
def test_successful_download(httpbin: None) -> None:
    result = _ability.run({"url": "https://httpbin.org/robots.txt"})
    assert result.status == "success"
    body = result.body
    assert isinstance(body, dict)
    path = cast(str, body["path"])
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
    second_path = cast(str, cast(dict[str, object], second.body)["path"])
    assert second_path != path
    assert second_path.endswith("robots.txt")
    assert os.path.exists(second_path)

    os.remove(path)
    os.remove(second_path)


@pytest.mark.integration
def test_timeout_and_edge_values(httpbin: None) -> None:
    result = _ability.run({"url": "https://httpbin.org/bytes/32", "timeout": 5})
    assert result.status == "success"
    assert os.path.exists(cast(str, cast(dict[str, object], result.body)["path"]))
    os.remove(cast(str, cast(dict[str, object], result.body)["path"]))

    # Over-max timeout is clamped (param clamp=(1, 120)) — still downloads.
    result = _ability.run({"url": "https://httpbin.org/bytes/32", "timeout": 999})
    assert result.status == "success"
    assert os.path.exists(cast(str, cast(dict[str, object], result.body)["path"]))
    os.remove(cast(str, cast(dict[str, object], result.body)["path"]))

    # Non-numeric timeout is a present-but-invalid clamp value — raises, it does
    # not fall back to the default (only a MISSING value falls back).
    with pytest.raises(ToolParamError, match="'timeout' must be a number."):
        _ability.run({"url": "https://httpbin.org/bytes/32", "timeout": "banana"})


@pytest.mark.integration
def test_http_error_status_is_reported(httpbin: None) -> None:
    result = _ability.run({"url": "https://httpbin.org/status/404"})
    assert result.status == "error"
    assert result.code == "download-failed"
    assert "404" in cast(str, result.body) or "not found" in cast(str, result.body).lower()


# ===========================================================================
# Migrated from test_ability_web_download_tool_result.py ()
# Offline contract: pre-gate, blocked schemes, SSRF guard, module hygiene.
# All four scheme/SSRF tests migrated — none were covered in the network tier above.
# ===========================================================================

import abilities.web_download  # noqa: E402
from abilities._dispatcher import ToolDispatcher  # noqa: E402
from configs.channels import DmnConfig  # noqa: E402
from services.act_trail import ActTrail  # noqa: E402
from tests._tool_result_harness import seed_transcript  # noqa: E402


def _seed_dl_transcript(db: sqlite3.Connection) -> int:
    """Insert the transcript anchor (tool_calls.transcript_id FK) the trail hangs its recorded rows off."""
    return seed_transcript(db, channel="dmn", content="download this file for me")


class _DownloadMP:
    """Minimal real MP-shaped context — exactly what dispatch reads off a live
    processor: ``config`` (policy channel + emitter gate) and ``uid`` (the
    transcript anchor). No policy row needed — web_download is INTERNAL."""

    def __init__(self, uid: int) -> None:
        self.config = DmnConfig()
        self.uid = uid
        self.DISCOVERABLE: list[str] = []
        self.active_tools: list[str] = []


@pytest.mark.unit
def test_file_scheme_is_blocked_url(db: sqlite3.Connection) -> None:
    mp = _DownloadMP(_seed_dl_transcript(db))

    result = ToolDispatcher(mp).dispatch("web_download", {"url": "file:///etc/passwd"})

    assert "status=error" in result
    assert "code=blocked-url" in result


@pytest.mark.unit
def test_data_scheme_is_blocked_url(db: sqlite3.Connection) -> None:
    mp = _DownloadMP(_seed_dl_transcript(db))

    result = ToolDispatcher(mp).dispatch(
        "web_download", {"url": "data:text/plain;base64,SGVsbG8="}
    )

    assert "status=error" in result
    assert "code=blocked-url" in result


@pytest.mark.unit
def test_localhost_ssrf_terminal_outcome_is_blocked_url(db: sqlite3.Connection) -> None:
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
    assert "blocked-url" in cast(str, rows[0]["result"])


@pytest.mark.unit
def test_127_ssrf_terminal_outcome_is_blocked_url(db: sqlite3.Connection) -> None:
    """``http://127.0.0.1`` is the same single-guard outcome — proves the guard is
    address-resolved, not host-string special-cased to 'localhost'."""
    mp = _DownloadMP(_seed_dl_transcript(db))

    result = ToolDispatcher(mp).dispatch("web_download", {"url": "http://127.0.0.1/secret"})

    assert "status=error" in result
    assert "code=blocked-url" in result


@pytest.mark.unit
def test_module_no_longer_carries_local_ssrf_guard() -> None:
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
