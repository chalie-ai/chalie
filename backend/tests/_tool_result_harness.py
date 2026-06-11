# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared parsing/seeding plumbing for the ``test_ability_*_tool_result`` suite.

These are ZERO-MOCK feature tests that drive the real
``ToolDispatcher(mp).dispatch()`` chokepoint against a real test database. This
module is *plumbing only* — it seeds rows, flips real policy rows, and parses the
rendered tag envelope. It re-implements NO production logic, adds NO mocks, and
creates NO alternative code paths.

The sealed tag envelope every ability renders (TKT-882) is::

    [<tool>(status=success, <meta>=…)]
    <body>
    [end:<tool>]

The body is either a verbatim prose string, a compact-JSON dict/list, or — on a
**user-broadcast** turn that carries a rich card — the structured payload JSON,
then a blank line, then the ``<span id='<tool>_N'>…`` card instruction trailer::

    [<tool>(status=success)]
    {"…card payload…"}

    <span id='<tool>_1'>…instruction…</span>
    [end:<tool>]

``body(rendered, tool, rich=True)`` returns the JSON head before that blank line
so a test can round-trip the card payload; ``rich=False`` (the default) returns
the whole body verbatim. The two are intentionally separate rather than always
splitting, because non-broadcast bodies can legitimately be a verbatim multi-line
string that must NOT be truncated at a blank line.

Why this does not delegate to ``_tag_helpers.extract_body``: that helper returns
the entire body between the markers (it has no rich-card awareness) and is
regex-anchored to the opener line. The pilots assert against the simpler
``index(']\\n')`` slice and need the blank-line split for rich cards, so the body
extractor here is the canonical one for this suite. ``_tag_helpers`` stays the
generic tag helper for non-ToolResult callers.

Public API
----------
``MP(uid, config)``
    Minimal real MP-shaped context the dispatcher reads: ``config`` (policy
    channel + emitter gate + broadcast_to) and ``uid`` (the transcript anchor the
    act-trail records against). Replaces the per-file ``_MP`` copies.
``seed_transcript(db, channel="chat", content="do a thing")`` -> int
    Insert one user transcript row and return its ``lastrowid`` (the uid anchor).
``head(rendered, tool)`` -> str
    First line of the envelope; asserts the ``[<tool>(`` opener prefix.
``body(rendered, tool, rich=False)`` -> str
    Text between the opener line and ``[end:<tool>]``. With ``rich=True`` returns
    the JSON head before the first blank line (the rich-card payload).
``parse_body(rendered, tool, rich=False)`` -> object
    ``body(...)`` JSON-parsed.
``allow_policy(db, permission, channel="chat")`` -> None
    ``INSERT OR REPLACE`` the real ``policy`` table so ``permission`` is ``allow``
    on ``channel`` — the same row the real ``PolicyManager`` gate reads. Lets a
    seed-``ask``/``deny`` gate pass through to ``run()`` on a headless channel,
    exactly as a user picking "always allow" does. No mock — the production store.
"""

import json


class MP:
    """Minimal real MP-shaped context — exactly what dispatch reads off the live
    processor: ``config`` (the policy channel + emitter gate + broadcast_to) and
    ``uid`` (the transcript anchor the act-trail records against)."""

    def __init__(self, uid: int, config) -> None:
        self.config = config
        self.uid = uid


def seed_transcript(db, channel: str = "chat", content: str = "do a thing") -> int:
    """Insert one ``user`` transcript row on *channel* and return its rowid.

    The rowid is the uid anchor the act-trail records each dispatch against.
    """
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        (channel, "user", content),
    )
    db.commit()
    return cur.lastrowid


def allow_policy(db, permission: str, channel: str = "chat") -> None:
    """Flip the REAL ``policy`` table so *permission* is ``allow`` on *channel*.

    The same row the real ``PolicyManager`` gate reads. A destructive permission
    that ships as ``ask``/``deny`` by seed would block (or default-deny) on a
    headless test channel; flipping the real row to ``allow`` — exactly what a user
    does when they pick "always allow" — lets the gate pass through to the
    production ``run()``. No mock; this is the production policy store.
    """
    db.execute(
        "INSERT OR REPLACE INTO policy (channel, permission, setting) "
        "VALUES (?, ?, 'allow')",
        (channel, permission),
    )
    db.commit()


def head(rendered: str, tool: str) -> str:
    """Return the first line of the envelope, asserting the ``[<tool>(`` opener."""
    line = rendered.splitlines()[0]
    assert line.startswith(f"[{tool}("), rendered
    return line


def body(rendered: str, tool: str, rich: bool = False) -> str:
    """Return the body between the opener line and ``[end:<tool>]``.

    With ``rich=True`` the body is a rich card (``<json>\\n\\n<instruction>``);
    return the JSON head before the blank line (the card payload). With
    ``rich=False`` (default) return the whole body verbatim.
    """
    start = rendered.index("]\n") + 2
    end = rendered.index(f"\n[end:{tool}]")
    text = rendered[start:end]
    if rich:
        return text.split("\n\n", 1)[0]
    return text


def parse_body(rendered: str, tool: str, rich: bool = False) -> object:
    """JSON-parse ``body(rendered, tool, rich)`` — proves the envelope is
    structured and machine-parseable."""
    return json.loads(body(rendered, tool, rich=rich))
