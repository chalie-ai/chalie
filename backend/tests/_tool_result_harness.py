# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared ``ToolResult`` plumbing for ability tests: envelope parsing/seeding
for the ``test_ability_*_tool_result`` suite, plus the :func:`built` unwrap
every happy-path ``from_params`` call site uses."""

import json
import sqlite3
from typing import TypeVar, cast

from abilities._result import ToolResult
from contracts.params.param_bag import ParamBag

B = TypeVar("B", bound=ParamBag)


def built(bag: "B | ToolResult") -> B:
    """Unwrap a ``from_params`` result the test expects to be a valid bag —
    the test-side mirror of the dispatch seam's isinstance guard, failing
    loudly with the bag-authored error if the params were in fact bad."""
    assert not isinstance(bag, ToolResult), f"unexpected param error: {bag.body!r}"
    return bag


class MP:
    def __init__(self, uid: int, config: object) -> None:
        self.config = config
        self.uid = uid


def seed_transcript(db: sqlite3.Connection, channel: str = "chat", content: str = "do a thing") -> int:
    """Seed a turn-anchoring input row exactly as production does."""
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content, turn_id) VALUES (?, ?, ?, "
        "(SELECT COALESCE(MAX(turn_id), 0) + 1 FROM transcript WHERE channel = ?))",
        (channel, "user", content, channel),
    )
    db.commit()
    return cast(int, cur.lastrowid)


def allow_policy(db: sqlite3.Connection, permission: str, channel: str = "chat") -> None:
    """Flip the real ``policy`` table so *permission* is ``allow`` on *channel*."""
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
    card payload); otherwise the full body between the head and ``[end:<tool>]``,
    with any follow-up instruction block stripped — the block is a standing
    nudge appended on success, not part of the structured payload tests assert
    on."""
    start = rendered.index("]\n") + 2
    end = rendered.index(f"\n[end:{tool}]")
    text = rendered[start:end]
    if "[follow_up_instruction]" in text:
        text = text.split("\n[follow_up_instruction]", 1)[0]
    if rich:
        return text.split("\n\n", 1)[0]
    return text


def parse_body(rendered: str, tool: str, rich: bool = False) -> object:
    return json.loads(body(rendered, tool, rich=rich))
