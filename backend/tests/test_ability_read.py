# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0



import sqlite3
import threading
from pathlib import Path

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import DmnConfig
from tests._tool_result_harness import allow_policy, seed_transcript

pytestmark = pytest.mark.unit


def _seed_transcript(db: sqlite3.Connection) -> int:
    """Insert the transcript anchor (tool_calls.transcript_id FK) the trail hangs
    its recorded rows off, and return its id."""
    return seed_transcript(db, channel="dmn", content="read this for me")


def _allow_read(db: sqlite3.Connection) -> None:
    """Seed the real policy row prod would carry so the gate runs ``read`` on the
    allow path (no mock — the same flat ``policy`` table PolicyManager reads)."""
    allow_policy(db, "read", channel=DmnConfig().policy_channel.value)


class _MP:
    """Minimal real MP-shaped context — exactly what dispatch reads off a live
    processor: ``config`` (policy channel + emitter gate), ``uid`` (the transcript
    anchor), and ``cancel_event``."""

    def __init__(self, uid: int) -> None:
        self.config = DmnConfig()
        self.uid = uid
        self.DISCOVERABLE: list[str] = []
        self.active_tools: list[str] = []
        self.cancel_event = threading.Event()


def test_patch_file_passes_through_verbatim(db: sqlite3.Connection, tmp_path: Path) -> None:
    """A raw ``.patch`` file must come back VERBATIM — the TKT-899 bug was that
    ``extract_html`` stripped non-HTML text to ``no-readable-content``. The patch
    body has zero HTML, so any extraction would lose it."""
    transcript_id = _seed_transcript(db)
    _allow_read(db)
    mp = _MP(transcript_id)

    patch_body = (
        "diff --git a/foo.py b/foo.py\n"
        "index 1234567..89abcde 100644\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-old_line = 1\n"
        "+new_line = 2\n"
        " unchanged = 3\n"
    )
    target = tmp_path / "change.patch"
    target.write_text(patch_body, encoding="utf-8")

    result = ToolDispatcher(mp).dispatch("read", {"source": str(target)})

    assert "status=success" in result
    assert "no-readable-content" not in result
    # The diff markers survive verbatim — no extraction touched them.
    assert "diff --git a/foo.py b/foo.py" in result
    assert "-old_line = 1" in result
    assert "+new_line = 2" in result


def test_diff_file_passes_through_verbatim(db: sqlite3.Connection, tmp_path: Path) -> None:
    """Same passthrough guarantee for the ``.diff`` extension."""
    transcript_id = _seed_transcript(db)
    _allow_read(db)
    mp = _MP(transcript_id)

    diff_body = "@@ -10,2 +10,2 @@\n-removed\n+added\n"
    target = tmp_path / "patchset.diff"
    target.write_text(diff_body, encoding="utf-8")

    result = ToolDispatcher(mp).dispatch("read", {"source": str(target)})

    assert "status=success" in result
    assert "no-readable-content" not in result
    assert "-removed" in result
    assert "+added" in result


def test_html_file_still_goes_through_extraction(db: sqlite3.Connection, tmp_path: Path) -> None:
    """Passthrough is NOT a blanket bypass — an ``.html`` file still runs through
    the real ``extract_html``, so script/style junk is stripped and only the
    readable article text survives."""
    transcript_id = _seed_transcript(db)
    _allow_read(db)
    mp = _MP(transcript_id)

    html_body = (
        "<html><head><title>T</title>"
        "<script>var SECRET_SCRIPT_JUNK = 1;</script>"
        "<style>.x{color:RED_STYLE_JUNK}</style></head>"
        "<body><article><h1>Real Heading</h1>"
        "<p>This is the genuine readable article paragraph that extraction keeps. "
        "It is long enough that trafilatura treats it as the main content block.</p>"
        "</article></body></html>"
    )
    target = tmp_path / "page.html"
    target.write_text(html_body, encoding="utf-8")

    result = ToolDispatcher(mp).dispatch("read", {"source": str(target)})

    assert "status=success" in result
    # The readable text is kept…
    assert "genuine readable article paragraph" in result
    # …and the script/style junk is stripped by real extraction (not passthrough).
    assert "SECRET_SCRIPT_JUNK" not in result
    assert "RED_STYLE_JUNK" not in result


def test_truncation_clips_body_and_flags_meta(db: sqlite3.Connection, tmp_path: Path) -> None:
    """An over-long body is clipped via the shared ``truncate`` helper and the
    envelope carries ``truncated=true``; a short body carries no truncated flag."""
    transcript_id = _seed_transcript(db)
    _allow_read(db)
    mp = _MP(transcript_id)

    target = tmp_path / "big.txt"
    target.write_text("A" * 5000, encoding="utf-8")

    result = ToolDispatcher(mp).dispatch("read", {"source": str(target), "max_chars": 100})

    assert "status=success" in result
    assert "truncated=true" in result
    # The body was actually clipped to the limit — the full 5000-char run is gone.
    assert "A" * 5000 not in result


def test_short_body_carries_no_truncated_flag(db: sqlite3.Connection, tmp_path: Path) -> None:
    """A body under max_chars must NOT carry the truncated flag."""
    transcript_id = _seed_transcript(db)
    _allow_read(db)
    mp = _MP(transcript_id)

    target = tmp_path / "small.txt"
    target.write_text("SHORT_BODY_OK", encoding="utf-8")

    result = ToolDispatcher(mp).dispatch("read", {"source": str(target), "max_chars": 1000})

    assert "status=success" in result
    assert "SHORT_BODY_OK" in result
    assert "truncated=true" not in result


def test_file_read_meta_carries_content_type(db: sqlite3.Connection, tmp_path: Path) -> None:
    """Both branches name the type with ``content_type`` (the file branch's old
    ``mime`` key is renamed for consistency with the URL branch)."""
    transcript_id = _seed_transcript(db)
    _allow_read(db)
    mp = _MP(transcript_id)

    target = tmp_path / "note.txt"
    target.write_text("CONTENT_TYPE_META_PROBE", encoding="utf-8")

    result = ToolDispatcher(mp).dispatch("read", {"source": str(target)})

    assert "status=success" in result
    assert "content_type=" in result
    assert "mime=" not in result


def test_max_chars_clamped_below_floor(db: sqlite3.Connection, tmp_path: Path) -> None:
    """``max_chars`` is clamped via the param helper's ``clamp=(100, …)`` floor —
    a value below 100 cannot drop the body to a sub-floor clip. A 250-char body
    with max_chars=1 survives to the floor (100), so it is still truncated but not
    erased to a single char."""
    transcript_id = _seed_transcript(db)
    _allow_read(db)
    mp = _MP(transcript_id)

    target = tmp_path / "clamp.txt"
    target.write_text("B" * 250, encoding="utf-8")

    result = ToolDispatcher(mp).dispatch("read", {"source": str(target), "max_chars": 1})

    assert "status=success" in result
    assert "truncated=true" in result
    # Floor honoured: at least 100 chars came back (not clipped to 1).
    assert "B" * 100 in result


def test_system_path_blocked_renders_kebab_code(db: sqlite3.Connection) -> None:
    """A system path (``/etc/passwd``) is refused with ``system-path-blocked``.

    The read tool's path-traversal guard (abilities/read.py) is read-specific and
    has no sibling coverage, so this stays here rather than collapsing into the
    central contract test.
    """
    transcript_id = _seed_transcript(db)
    _allow_read(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch("read", {"source": "/etc/passwd"})

    assert "status=error" in result
    assert "code=system-path-blocked" in result
