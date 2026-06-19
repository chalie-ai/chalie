# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared parsing/seeding plumbing for the ``test_ability_*_tool_result`` suite.

Plumbing only - seeds rows, flips real policy rows, and parses the rendered tag
envelope. Re-implements NO production logic, adds NO mocks, and creates NO
alternative code paths.

The sealed tag envelope every ability renders is::

    [<tool>(status=success, <meta>=...)]
    <body>
    [end:<tool>]

On a user-broadcast turn that carries a rich card the body is structured payload
JSON, then a blank line, then the ``<span id='<tool>_N'>...`` card instruction::

    [<tool>(status=success)]
    {"...card payload..."}

    <span id='<tool>_1'>...instruction...</span>
    [end:<tool>]

``body(..., rich=True)`` returns only the JSON head before that blank line.
Non-broadcast bodies can be a verbatim multi-line string that must NOT be
truncated at a blank line, so the two cases are handled separately.

Does not delegate to ``_tag_helpers.extract_body`` because that helper has no
rich-card awareness and is regex-anchored to the opener line; this extractor is
canonical for this suite. ``_tag_helpers`` remains the generic helper for
non-ToolResult callers.
"""

import sqlite3
from typing import cast
import json


class MP:
    def __init__(self, uid: int, config: object) -> None:
        self.config = config
        self.uid = uid


def seed_transcript(db: sqlite3.Connection, channel: str = "chat", content: str = "do a thing") -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        (channel, "user", content),
    )
    db.commit()
    return cast(int, cur.lastrowid)


def allow_policy(db: sqlite3.Connection, permission: str, channel: str = "chat") -> None:
    """Flip the real ``policy`` table so *permission* is ``allow`` on *channel*.

    A permission that ships as ``ask``/``deny`` by seed would block on a headless
    test channel; this mirrors "always allow" in the production policy store so
    the gate passes through to ``run()``. Writes to the real DB, no mock.
    """
    db.execute(
        "INSERT OR REPLACE INTO policy (channel, permission, setting) "
        "VALUES (?, ?, 'allow')",
        (channel, permission),
    )
    db.commit()


def head(rendered: str, tool: str) -> str:
    line = rendered.splitlines()[0]
    assert line.startswith(f"[{tool}("), rendered
    return line


def body(rendered: str, tool: str, rich: bool = False) -> str:
    """With ``rich=True`` return only the JSON head before the blank line (the
    card payload); with ``rich=False`` return the whole body verbatim.
    """
    start = rendered.index("]\n") + 2
    end = rendered.index(f"\n[end:{tool}]")
    text = rendered[start:end]
    if rich:
        return text.split("\n\n", 1)[0]
    return text


def parse_body(rendered: str, tool: str, rich: bool = False) -> object:
    return json.loads(body(rendered, tool, rich=rich))
