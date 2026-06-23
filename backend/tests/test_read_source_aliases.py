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
from typing import cast

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import DmnConfig
from services.act_trail import ActTrail

pytestmark = pytest.mark.unit


def _seed_transcript(db: sqlite3.Connection) -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("dmn", "user", "read this for me"),
    )
    db.commit()
    return cast(int, cur.lastrowid)


def _allow_read(db: sqlite3.Connection) -> None:
    db.execute(
        "INSERT OR REPLACE INTO policy (channel, permission, setting) VALUES (?, 'read', 'allow')",
        (DmnConfig().policy_channel.value,),
    )
    db.commit()


class _MP:
    def __init__(self, uid: int):
        self.config = DmnConfig()
        self.uid = uid
        self.DISCOVERABLE: list[str] = []
        self.active_tools: list[str] = []
        self.cancel_event = threading.Event()


def test_read_resolves_url_alias_through_real_dispatch(db: sqlite3.Connection) -> None:
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
    assert "private-or-internal-url-blocked" in cast(str, rows[0]["result"])


def test_read_resolves_path_alias_with_real_file(db: sqlite3.Connection, tmp_path: Path) -> None:
    transcript_id = _seed_transcript(db)
    _allow_read(db)
    mp = _MP(transcript_id)

    target = tmp_path / "note.txt"
    target.write_text("ALIAS_PATH_CONTENT_OK\n", encoding="utf-8")

    result = ToolDispatcher(mp).dispatch("read", {"path": str(target)})

    assert "source-required" not in result
    assert "ALIAS_PATH_CONTENT_OK" in result


def test_read_missing_source_returns_diagnostic_keys(db: sqlite3.Connection) -> None:
    transcript_id = _seed_transcript(db)
    _allow_read(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch("read", {"foobar": "https://example.com"})

    # Still an error (the value is under an unrecognised key)…
    assert "source-required" in result
    # …but now diagnostic: it echoes the key the model actually sent.
    assert "foobar" in result
