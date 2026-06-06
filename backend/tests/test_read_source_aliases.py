# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for TKT-834 — the ``read`` tool must accept the argument key a
model naturally emits (``url`` for a URL, ``path``/``file`` for a file), not only
the schema's canonical ``source``.

Real hot path, zero mocks: every test drives the live ``ToolDispatcher.dispatch``
chokepoint exactly as ``MessageProcessor._loop`` does — real ``AbilityRegistry``
resolution, the real ``PolicyManager.wrap`` gate (with ``read`` seeded ``allow``
on the channel, the same row prod writes), the real ``ReadAbility.run``, real
file I/O / real SSRF guard, and the real ``ActTrail`` write read back from the db.

Regression guard: before the fix, a model that called ``read({"url": …})`` or
``read({"path": …})`` got an opaque ``source-required`` and looped forever on URL
variants (Chalie's report: 8 identical failures across GitHub URLs AND a local
file). These tests fail loudly on the old single-key code and pass on the fix.
"""

import threading

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import DmnConfig
from services.act_trail import ActTrail

pytestmark = pytest.mark.unit


def _seed_transcript(db) -> int:
    """Insert the transcript anchor (tool_calls.transcript_id FK) the trail hangs
    its recorded rows off, and return its id."""
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("dmn", "user", "read this for me"),
    )
    db.commit()
    return cur.lastrowid


def _allow_read(db) -> None:
    """Seed the real policy row prod would carry so the gate runs ``read`` on the
    allow path (no mock — the same flat ``policy`` table PolicyManager reads)."""
    db.execute(
        "INSERT OR REPLACE INTO policy (channel, permission, setting) VALUES (?, 'read', 'allow')",
        (DmnConfig().policy_channel.value,),
    )
    db.commit()


class _MP:
    """Minimal real MP-shaped context — exactly what dispatch reads off a live
    processor: ``config`` (policy channel + emitter gate), ``uid`` (the transcript
    anchor), and ``cancel_event``."""

    def __init__(self, uid: int):
        self.config = DmnConfig()
        self.uid = uid
        self.DISCOVERABLE: list[str] = []
        self.active_tools: list[str] = []
        self.cancel_event = threading.Event()


def test_read_resolves_url_alias_through_real_dispatch(db):
    """``read({"url": …})`` — the key a model emits for a URL — must resolve to the
    URL path, not bounce on ``source-required``. Uses a private host so the real
    SSRF guard gives a deterministic, network-free terminal outcome."""
    transcript_id = _seed_transcript(db)
    _allow_read(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch("read", {"url": "http://localhost"})

    # The alias was honoured: it reached the URL branch (SSRF block), NOT the
    # empty-source guard.
    assert "source-required" not in result
    assert "private-or-internal-url-blocked" in result

    # And the real trail recorded the read outcome against the anchor.
    rows = ActTrail().fetch_by_transcript_id(transcript_id)
    assert [r["tool_name"] for r in rows] == ["read"]
    assert "private-or-internal-url-blocked" in rows[0]["result"]


def test_read_resolves_path_alias_with_real_file(db, tmp_path):
    """``read({"path": …})`` — the key a model emits for a file — must read the real
    file end-to-end, not bounce on ``source-required`` (Chalie's local-file case)."""
    transcript_id = _seed_transcript(db)
    _allow_read(db)
    mp = _MP(transcript_id)

    target = tmp_path / "note.txt"
    target.write_text("ALIAS_PATH_CONTENT_OK\n", encoding="utf-8")

    result = ToolDispatcher(mp).dispatch("read", {"path": str(target)})

    assert "source-required" not in result
    assert "ALIAS_PATH_CONTENT_OK" in result


def test_read_missing_source_returns_diagnostic_keys(db):
    """When NO usable key is present, the error must name the keys actually
    received so the model self-corrects instead of looping. ``foobar`` is not an
    alias, so the value is genuinely unusable."""
    transcript_id = _seed_transcript(db)
    _allow_read(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch("read", {"foobar": "https://example.com"})

    # Still an error (the value is under an unrecognised key)…
    assert "source-required" in result
    # …but now diagnostic: it echoes the key the model actually sent.
    assert "foobar" in result
