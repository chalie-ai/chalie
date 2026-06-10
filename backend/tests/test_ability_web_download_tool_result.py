# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — the ``web_download`` tool's ToolResult contract (TKT-900).

Real hot path, zero mocks: every test drives the live ``ToolDispatcher.dispatch``
chokepoint exactly as ``MessageProcessor._loop`` does — real ``AbilityRegistry``
resolution, the real ``PolicyManager.wrap`` gate (``web_download`` is in
``PolicyManager.INTERNAL`` so it bypasses the row gate — no seed needed), the real
``WebDownloadAbility.run``, the real single SSRF guard in ``services.web_fetch``,
and the real ``ActTrail`` write read back from the db.

What these pin OFFLINE:
  - ``missing-params``: a missing/blank url is rejected by the dispatcher's
    ACTION_REQUIRED pre-gate BEFORE run() (was: status=success "url parameter is
    required" prose). Proves the ``ACTION_REQUIRED = {"": ("url",)}`` wiring.
  - ``blocked-url`` for the ability-side scheme check: ``file://`` and ``data:``
    URLs are refused (was: status=success "scheme blocked" prose).
  - ``blocked-url`` for the single SSRF guard: ``http://localhost`` resolves to a
    private address and ``stream_to_file``'s guard raises ``FetchBlocked``, which
    the ability maps to ``code=blocked-url`` (was: status=success prose). This is
    the same single guard ``read`` reaches — no local pre-check in the ability.
  - Module hygiene: the bespoke local ``is_private_url`` import and ``_validate_url``
    helper are gone — the guard lives in ``web_fetch`` and the scheme check is
    inline.

Network coverage gap (documented, same as prior tickets): the happy-path success
download (``{"path", "bytes", "content_type"}`` body + ``max_bytes`` meta), the
runtime ``download-failed`` (HTTP/connection error over the wire), and the runtime
``too-large`` (a real over-cap body streamed and aborted mid-stream) all need the
network. ``too-large`` in particular cannot be exercised offline through the
ability: ``stream_to_file`` only reaches its size enforcement AFTER the SSRF guard
passes, and every offline-reachable host (localhost / 127.0.0.1) is SSRF-blocked
*before* the body is pulled. We do NOT mock to fake it. ``too-large`` is exercised
in the integration tier of ``test_web_download.py`` against a real over-cap URL.
"""

import threading

import pytest

import abilities.web_download
from abilities._dispatcher import ToolDispatcher
from configs.channels import DmnConfig
from services.act_trail import ActTrail

pytestmark = pytest.mark.unit


def _seed_transcript(db) -> int:
    """Insert the transcript anchor (tool_calls.transcript_id FK) the trail hangs
    its recorded rows off, and return its id."""
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("dmn", "user", "download this file for me"),
    )
    db.commit()
    return cur.lastrowid


class _MP:
    """Minimal real MP-shaped context — exactly what dispatch reads off a live
    processor: ``config`` (policy channel + emitter gate), ``uid`` (the transcript
    anchor), and ``cancel_event``. No policy row needed — web_download is INTERNAL."""

    def __init__(self, uid: int):
        self.config = DmnConfig()
        self.uid = uid
        self.DISCOVERABLE: list[str] = []
        self.active_tools: list[str] = []
        self.cancel_event = threading.Event()


def test_missing_url_is_pregate_missing_params(db):
    """A missing url is rejected by the ACTION_REQUIRED pre-gate as
    ``missing-params`` BEFORE run() — not a status=success "url required" prose
    body (the old error-as-ok bug)."""
    transcript_id = _seed_transcript(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch("web_download", {})

    assert "status=error" in result
    assert "code=missing-params" in result


def test_blank_url_is_pregate_missing_params(db):
    """A blank url is treated as missing by the pre-gate (``not params.get(p)``
    is true for an empty string)."""
    transcript_id = _seed_transcript(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch("web_download", {"url": ""})

    assert "status=error" in result
    assert "code=missing-params" in result


def test_file_scheme_is_blocked_url(db):
    """A ``file://`` URL is refused with ``code=blocked-url`` — the scheme check is
    network-free and fires before any fetch (was: status=success prose)."""
    transcript_id = _seed_transcript(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch("web_download", {"url": "file:///etc/passwd"})

    assert "status=error" in result
    assert "code=blocked-url" in result


def test_data_scheme_is_blocked_url(db):
    """A ``data:`` URL is refused with ``code=blocked-url`` — urlparse-cheap, no
    network."""
    transcript_id = _seed_transcript(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch(
        "web_download", {"url": "data:text/plain;base64,SGVsbG8="}
    )

    assert "status=error" in result
    assert "code=blocked-url" in result


def test_localhost_ssrf_terminal_outcome_is_blocked_url(db):
    """The single SSRF guard in ``web_fetch.stream_to_file`` refuses
    ``http://localhost`` (resolves private) and the ability maps the resulting
    ``FetchBlocked`` to ``code=blocked-url`` — network-free, and the real trail
    records the outcome against the anchor."""
    transcript_id = _seed_transcript(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch("web_download", {"url": "http://localhost/x"})

    assert "status=error" in result
    assert "code=blocked-url" in result

    rows = ActTrail().fetch_by_transcript_id(transcript_id)
    assert [r["tool_name"] for r in rows] == ["web_download"]
    assert "blocked-url" in rows[0]["result"]


def test_127_ssrf_terminal_outcome_is_blocked_url(db):
    """``http://127.0.0.1`` is the same single-guard outcome — proves the guard is
    address-resolved, not host-string special-cased to 'localhost'."""
    transcript_id = _seed_transcript(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch("web_download", {"url": "http://127.0.0.1/secret"})

    assert "status=error" in result
    assert "code=blocked-url" in result


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
